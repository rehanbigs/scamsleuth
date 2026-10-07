"""Extract URLs, email addresses, phone numbers and short codes from a message.

Scam messages actively hide their links: zero-width characters inside a domain,
full-width dots, already-defanged text (``hxxp://evil[.]com``), deliberately broken
schemes (``http:/evil.com``) and missing spaces. Text is therefore canonicalised before
extraction.

Nothing here performs network I/O. URLs are only ever parsed as text, and the public
suffix list comes from tldextract's bundled snapshot.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Final

import tldextract

# Soft hyphen, zero-width characters and bidirectional controls.
_INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
_IDEOGRAPHIC_FULL_STOP = "\u3002"
_REPLACEMENT_CHAR = "\ufffd"

_REFANG: Final = (
    (re.compile(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", re.IGNORECASE), "."),
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"\[@\]|\[at\]|\(at\)", re.IGNORECASE), "@"),
    (re.compile(r"\bhxxp(s?)", re.IGNORECASE), r"http\1"),
    (re.compile(r"\bfxp\b", re.IGNORECASE), "ftp"),
    # Deliberately broken schemes: "http:/x", "http/x", "http:l/x", "http:// x".
    (
        re.compile(r"\b(https?|ftp)\s*(?::\s*(?:l/|/{1,2})|/{1,2})\s*(?=[\w\[])", re.IGNORECASE),
        r"\1://",
    ),
)

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SCHEME_URL = re.compile(r"\b(?:https?|ftp)://[^\s<>\"']+", re.IGNORECASE)
_WWW_URL = re.compile(r"(?<![\w.-])www\.[^\s<>\"']+", re.IGNORECASE)
_BARE_URL = re.compile(r"(?<![\w.@/-])(?:[\w-]+\.)+[\w-]+(?:/[^\s<>\"']*)?")
_DATE_TIME = re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
# Digit groups of 2+ separated by single spaces or hyphens: "0844 861 85 85", "+44 20 7946 0018".
_PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[ -]?)?\(?\d{2,}\)?(?:[ -]\(?\d{2,}\)?)*(?!\w)")
# Premium-rate short codes, only after a call to action: "Txt WIN to 87066", "send 2 No: 8883".
_SHORT_CODE = re.compile(
    r"\b(?:to|2|txt|text|send|sms|call|reply|dial)\b\W{0,3}(?:no\W{0,3})?(\d{4,6})(?!\w)",
    re.IGNORECASE,
)
_TRAILING = ".,;:!?)]}'\"\u2026"

# Without a scheme or "www.", missing spaces look like domains: "now.Call", "days.so",
# "message.it" all end in real TLDs. Bare domains are therefore only accepted when followed
# by a path, when the suffix has several labels ("co.uk"), or when the TLD is in this list
# of suffixes common in SMS that are not also short English words.
_BARE_TLDS: Final = frozenset(
    {"com", "net", "org", "info", "biz", "io", "app", "top", "xyz", "online", "site", "club"}
    | {"live", "link", "click", "shop", "store", "mobi", "vip", "icu", "cc", "tk", "ml", "ga"}
    | {"uk", "ru", "cn", "tv", "ly", "ca", "au", "nz", "ie", "ng", "pk", "bd", "za", "gl"}
    | {"ws", "su", "eu", "cf", "gq"}
)

_tld = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


@dataclass(frozen=True)
class Entities:
    urls: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    phones: tuple[str, ...] = ()
    short_codes: tuple[str, ...] = ()

    def defanged(self) -> Entities:
        """Copy with URLs and emails made non-clickable, safe for display."""
        return replace(
            self,
            urls=tuple(defang_url(url) for url in self.urls),
            emails=tuple(defang_email(email) for email in self.emails),
        )


def canonicalize(text: str) -> str:
    """Undo common obfuscation: Unicode compatibility forms, invisible characters, defanging."""
    text = unicodedata.normalize("NFKC", text)  # full-width letters and dots become ASCII
    text = _INVISIBLE.sub("", text).replace(_IDEOGRAPHIC_FULL_STOP, ".")
    text = text.replace(_REPLACEMENT_CHAR, " ")  # encoding damage, often a lost space
    for pattern, repl in _REFANG:
        text = pattern.sub(repl, text)
    return text


def _host(url: str) -> str:
    without_scheme = url.split("://", 1)[-1]
    return re.split(r"[/?#:]", without_scheme, maxsplit=1)[0]


def _is_plausible_bare_domain(candidate: str) -> bool:
    host = _host(candidate)
    parts = _tld(host)
    if not parts.suffix or len(parts.domain) < 2:
        return False
    last = host.rsplit(".", 1)[-1]
    if last[:1].isupper() and not last.isupper():  # "now.Call": a missing space, not a TLD
        return False
    return "/" in candidate or "." in parts.suffix or last.lower() in _BARE_TLDS


def _dedupe(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def extract_entities(text: str) -> Entities:
    """Find URLs, emails, phone numbers and SMS short codes in raw message text.

    URLs and emails are returned refanged (usable for lookups, never for fetching);
    call :meth:`Entities.defanged` before showing them to a person.
    """
    text = canonicalize(text)

    emails = [m.group().rstrip(_TRAILING) for m in _EMAIL.finditer(text)]
    remaining = _EMAIL.sub(" ", text)

    urls: list[str] = []
    for pattern in (_SCHEME_URL, _WWW_URL):
        urls += [m.group().rstrip(_TRAILING) for m in pattern.finditer(remaining)]
        remaining = pattern.sub(" ", remaining)
    for match in _BARE_URL.finditer(remaining):
        candidate = match.group().rstrip(_TRAILING)
        if _is_plausible_bare_domain(candidate):
            urls.append(candidate)

    phones = []
    for match in _PHONE.finditer(_DATE_TIME.sub(" ", remaining)):
        digits = re.sub(r"\D", "", match.group())
        if 7 <= len(digits) <= 15:
            phones.append(("+" if match.group().startswith("+") else "") + digits)

    short_codes = [m.group(1) for m in _SHORT_CODE.finditer(remaining)]

    return Entities(
        urls=_dedupe(urls),
        emails=_dedupe(emails),
        phones=_dedupe(phones),
        short_codes=_dedupe(short_codes),
    )


def defang_url(url: str) -> str:
    """``https://evil.com/a.html`` -> ``hxxps://evil[.]com/a.html``."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        scheme, rest = "", url
    host = _host(rest)
    defanged = host.replace(".", "[.]") + rest[len(host) :]
    if scheme:
        scheme = re.sub(r"^http", "hxxp", scheme, flags=re.IGNORECASE)
        scheme = re.sub(r"^ftp$", "fxp", scheme, flags=re.IGNORECASE)
        return f"{scheme}://{defanged}"
    return defanged


def defang_email(email: str) -> str:
    """``bob@evil.com`` -> ``bob[@]evil[.]com``."""
    local, _, domain = email.partition("@")
    return f"{local}[@]{domain.replace('.', '[.]')}"
