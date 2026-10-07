# Baseline Error Analysis

Model: TF-IDF (word + character n-grams) + logistic regression (C = 10), isotonic calibration,
threshold 0.328 chosen for a 2% false-alarm budget on validation. Every mistake on the 2,444-message
test split was read and grouped by cause (`uv run python -m scamsleuth.eval.errors`).
Links are defanged throughout.

**153 of 2,444 test messages (6.3%) are misclassified.** Most of them come from two hard boundaries,
spam vs smishing and chatty scams vs chatty friends, rather than from obfuscation.

| Error | Count | Share of errors | Main cause |
|---|---:|---:|---|
| Spam called smishing | 85 | 56% | Fuzzy class boundary and noisy labels |
| Missed scam | 35 | 23% | Conversational "wrong number" scams carry no red flags |
| False alarm on ham | 18 | 12% | Chatty openers and genuine transaction notices |
| Spam called ham | 14 | 9% | Short promotions with no link or cue |
| Ham called spam | 1 | 1% | — |

## 1. Spam called smishing (85)

**a. Noisy labels: the message is phishing but is labelled spam (at least 21 of 85).** These have
a link plus an account, refund or prize hook, which is the definition of smishing used here. The
model is arguably right.

- "Your tax refund has been rejected. Click here to claim your refund: xxurl"
- "Account balance notification: unidentified inbound payment of $130.9: xxurl Description: Amazon Rewards Program"
- "Alert!!: Your Mobile No is Selected as winner of £ on Coke Promo. Go to xxurl to claim."
- "Greetings from store! You have been chosen as the winner of the happy 420 contest! Click on this link to collect your prize money"

**b. Aggressive marketing that looks like fraud.** Loan apps, casino bonuses and "earn 1000 Euros a
day" offers use the same urgency and money language as scams.

- "Hi Dear customer you have met the approval conditions and the credit limit is Rs 50000. Click xxurl to receive"
- "Thank you for supporting us. we will give you 108% CAS|NO Recharge bo-nus, xxurl all."

**c. Feature-phone premium-rate services (25 from the 2022 corpus).** Premium-rate offers sit on
both sides of the boundary in the source labels.

- "Santa calling! Would your little ones like a call from Santa Xmas Eve? Call 09077818151 to book you time. Calls1.50ppm"

## 2. Missed scams (35)

**a. Conversational "wrong number" scams (16 of 35; recall 36% on this type).** They open a chat
and move to money later, so the first message has no link, no urgency and no brand.

- "Did you lose your voice that you can never answer?"
- "Hope you still have this house available on facebook because i really want it so bad!"
- "I'm really sorry. I dialed the wrong number. And I thought you were my friend."
- "Hello. Nice to meet you"

**b. Prize and raffle lures predicted as spam (about 10).** The model sees marketing, not theft.

- "ATT Free Msg: Congrats to 2 lucky users! 's winners of our raffle are: and you,! Claim now: xxurl"
- "You have won a Nokia 7250i. To take part send Nokia to 86021 now."

**c. Label noise.** A few reports are not scams at all.

- "Vaccine scam done in and then Uddhav govt blame shortage.. Check next" (a political message)
- "Hello! Thank you for choosing to access..." (truncated by the source)

## 3. False alarms on legitimate messages (18)

**a. Chatty openers (about 9).** These are indistinguishable by text from the "wrong number"
scams in 2a: the model trades one error for the other.

- "Hi! This is Roger from CL. How are you?"
- "Hello madam how are you ?"
- "ARE YOU IN TOWN? THIS IS V. IMPORTANT"

**b. Genuine transaction and service notices (3).** This is the shortcut the synthetic hard-negative
set measures: anything official-looking reads as a scam.

- "We are pleased to inform that your application for Airtel Broadband is processed successfully."
- "NEFT Transaction with reference number 456367 for Rs.9000 has been credited to the beneficiary account on 5th jan at 3.00PM."

**c. Personal messages about money or emergencies (about 6).** For example, sharing bank details
with a friend, or a message from a hospital ward.

## 4. Spam called ham (14)

Very short promotions with no link, number or money cue: "Borrow money. We can help",
"Dream big, invest wisely, and watch your wealth grow. xxurl".

## What this means for the next model

1. **Spam vs smishing needs cleaner labels**, not a bigger model. Re-labelling the IMC `spam`
   category with a written rule (link or reply + credential/money request = smishing) would remove
   the largest error group.
2. **Wrong-number scams cannot be caught from one message.** Detecting them needs context the
   message does not carry: an unknown sender, a conversation history. This is a job for the agent,
   not the classifier.
3. **Genuine official messages need training examples.** The 2022 corpus has none, and the model
   flags 82% of synthetic genuine look-alikes. The next step is to add genuine official-style
   messages to training (kept separate from the evaluation-only look-alike set) and test whether a
   fine-tuned transformer separates them better than TF-IDF.
