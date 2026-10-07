import pytest

from scamsleuth.features.entities import (
    Entities,
    canonicalize,
    defang_email,
    defang_url,
    extract_entities,
)

# --- URLs -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Pay now: https://parcel-fee.top/x?id=1.", ("https://parcel-fee.top/x?id=1",)),
        ("T&C www.dbuk.net LCCLTD", ("www.dbuk.net",)),
        ("visit txt82228.co.uk for more", ("txt82228.co.uk",)),
        ("Qualify Here:1200cash4study1.com/", ("1200cash4study1.com/",)),
        ("short link (bit.ly/3xYz) today", ("bit.ly/3xYz",)),
        ("two links http://a.com and http://a.com", ("http://a.com",)),
    ],
)
def test_extracts_urls(text: str, expected: tuple[str, ...]) -> None:
    assert extract_entities(text).urls == expected


@pytest.mark.parametrize(
    "text",
    [
        "Sorry da thangam.it's my mistake.",  # missing space, ".it" is a TLD
        "I might come for 2 days.so you can be prepared",
        "Bring the tickets now.Call me when you reach",  # Title-case word after the dot
        "Price is Rs.3 per day",
        "I.ll be there by U.K time",
    ],
)
def test_ignores_sentences_with_missing_spaces(text: str) -> None:
    assert extract_entities(text).urls == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("secure link http:/bit.do/Claim-Tax", "http://bit.do/Claim-Tax"),
        ("visit http/aetacms.com/auo.html", "http://aetacms.com/auo.html"),
        ("Log in http:l/natwestauth.xyz now", "http://natwestauth.xyz"),
        ("claim http:// 103.208.86.96 today", "http://103.208.86.96"),
        ("go to hxxps://evil[.]com/login", "https://evil.com/login"),
        ("go to evil(dot)com/login", "evil.com/login"),
    ],
)
def test_repairs_broken_and_defanged_urls(text: str, expected: str) -> None:
    assert extract_entities(text).urls == (expected,)


@pytest.mark.parametrize(
    "text",
    [
        "pay at ev\N{ZERO WIDTH SPACE}il.com/fee",
        "pay at evil\N{FULLWIDTH FULL STOP}com/fee",
        "pay at evil\N{IDEOGRAPHIC FULL STOP}com/fee",
        "pay at \N{FULLWIDTH LATIN SMALL LETTER E}vil.com/fee",
        "pay at \N{RIGHT-TO-LEFT OVERRIDE}evil.com/fee\N{POP DIRECTIONAL FORMATTING}",
    ],
)
def test_unicode_obfuscation_is_undone(text: str) -> None:
    assert extract_entities(text).urls == ("evil.com/fee",)


def test_lookalike_cyrillic_domain_is_kept_not_normalised() -> None:
    # The Cyrillic "a" must survive: it is evidence of a homoglyph attack.
    spoofed = "p\N{CYRILLIC SMALL LETTER A}ypal.com"
    assert extract_entities(f"Verify at {spoofed}/login").urls == (f"{spoofed}/login",)


def test_replacement_character_does_not_glue_words_to_urls() -> None:
    text = "visit http://x.top/a.aspx\N{REPLACEMENT CHARACTER}to avoid suspension"
    assert extract_entities(text).urls == ("http://x.top/a.aspx",)


# --- Emails ---------------------------------------------------------------------------


def test_emails_are_extracted_and_not_counted_as_urls() -> None:
    entities = extract_entities("Contact MR. Billy: cocacolaclaimsoffice@yahoo.com now")
    assert entities.emails == ("cocacolaclaimsoffice@yahoo.com",)
    assert entities.urls == ()


def test_defanged_email_is_refanged() -> None:
    assert extract_entities("mail bob[at]evil[.]com").emails == ("bob@evil.com",)


# --- Phones and short codes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call 09058097189 NOW", ("09058097189",)),
        ("access number 0844 861 85 85. No prepayment", ("08448618585",)),
        ("call +44 20 7946 0018 today", ("+442079460018",)),
        ("ring (0800) 123-4567", ("08001234567",)),
    ],
)
def test_extracts_phone_numbers(text: str, expected: tuple[str, ...]) -> None:
    assert extract_entities(text).phones == expected


@pytest.mark.parametrize(
    "text",
    [
        "won £100,000 prize",  # money
        "select ( 1 2 3 4 5 6 7 8 9 )",  # single digits
        "Euro 2004 2-4-1 offer",
        "on 29-11-2016 15:04:43",  # dates
    ],
)
def test_ignores_numbers_that_are_not_phones(text: str) -> None:
    assert extract_entities(text).phones == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Txt MUSIC to 87066 TnCs", ("87066",)),
        ("text money 2 88600 now", ("88600",)),
        ("send it to No: 8883 CM", ("8883",)),
        ("I have 87066 reasons", ()),
        ("call 0844 861 85 85 now", ()),  # start of a phone number
    ],
)
def test_extracts_short_codes_after_call_to_action(text: str, expected: tuple[str, ...]) -> None:
    assert extract_entities(text).short_codes == expected


# --- Defanging ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://evil.com/a.html", "hxxps://evil[.]com/a.html"),
        ("HTTP://Evil.com", "hxxp://Evil[.]com"),
        ("http://95.141.32.7:81/x.aspx", "hxxp://95[.]141[.]32[.]7:81/x.aspx"),
        ("ftp://files.evil.com", "fxp://files[.]evil[.]com"),
        ("bit.ly/3xYz", "bit[.]ly/3xYz"),
    ],
)
def test_defang_url(url: str, expected: str) -> None:
    assert defang_url(url) == expected


def test_defang_email() -> None:
    assert defang_email("bob@mail.evil.com") == "bob[@]mail[.]evil[.]com"


def test_defanged_round_trip() -> None:
    text = "Pay http://evil.com/fee or mail bob@evil.com, call 09058097189"
    shown = extract_entities(text).defanged()
    assert shown == Entities(
        urls=("hxxp://evil[.]com/fee",),
        emails=("bob[@]evil[.]com",),
        phones=("09058097189",),
    )
    # Defanged output fed back in is recognised again, so analysis is stable.
    again = extract_entities(" ".join(shown.urls + shown.emails))
    assert again.urls == ("http://evil.com/fee",)
    assert again.emails == ("bob@evil.com",)


def test_canonicalize_is_idempotent() -> None:
    text = "hxxp://ev\N{ZERO WIDTH SPACE}il[.]com \N{FULLWIDTH LATIN CAPITAL LETTER A}"
    once = canonicalize(text)
    assert once == "http://evil.com A"
    assert canonicalize(once) == once
