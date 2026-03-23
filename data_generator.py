"""
data_generator.py
Generates realistic synthetic customer support tickets for training/demo.
"""

import random
import json
from datetime import datetime, timedelta

# ── Category templates ──────────────────────────────────────────────────────

TICKET_TEMPLATES = {
    "billing": {
        "high": [
            "I was charged twice for my subscription this month — please refund immediately.",
            "Unauthorized charge of $299 appeared on my card. Fraud alert filed.",
            "My account was suspended despite a successful payment. Need resolution now.",
            "Double billing error — urgent, I need this reversed before end of day.",
            "Payment failed but money was deducted from my account. Very urgent.",
        ],
        "medium": [
            "I'd like to update my billing information and payment method.",
            "Can you explain the charges on my latest invoice? Something looks off.",
            "I want to upgrade my plan — what are the pricing options?",
            "Request for an invoice for my records for tax purposes.",
            "How do I switch from monthly to annual billing?",
        ],
        "low": [
            "When does my subscription renew? Just curious.",
            "Is there a student discount available?",
            "Can I get a receipt for last month's payment?",
            "What payment methods do you accept?",
            "Do you offer nonprofit pricing?",
        ],
    },
    "technical": {
        "high": [
            "Production system is completely down — all users affected, losing revenue!",
            "Critical data loss detected. Files missing after your last update. URGENT.",
            "Security breach suspected. Unauthorized access to our account. Call immediately.",
            "App crashes every time we try to process orders. Business halted.",
            "Database connection error affecting all 500 of our team members right now.",
        ],
        "medium": [
            "The mobile app keeps crashing when I upload images larger than 5MB.",
            "Integration with Salesforce stopped working after yesterday's update.",
            "Two-factor authentication emails are not arriving consistently.",
            "Export to PDF feature produces blank pages for reports over 10 pages.",
            "Search functionality returns irrelevant results since the last patch.",
        ],
        "low": [
            "The dark mode toggle doesn't seem to save my preference.",
            "Minor typo on the settings page under 'Notifictions'.",
            "The loading spinner stays visible a second too long.",
            "Dashboard widgets could use a refresh button.",
            "Would be great if the table columns were resizable.",
        ],
    },
    "account": {
        "high": [
            "Account hacked — password changed without my knowledge. Lock it now!",
            "Cannot access account for 3 days, blocking my entire team's work.",
            "All my data disappeared after attempting to merge two accounts.",
            "Admin rights stripped from my account without authorization.",
            "Locked out of account with a deadline in 1 hour. Emergency.",
        ],
        "medium": [
            "I need to transfer account ownership to a new email address.",
            "How do I add additional users to my team account?",
            "I forgot my security questions and can't complete verification.",
            "My profile picture and bio were reset after last week's update.",
            "Need to change the primary email on my account.",
        ],
        "low": [
            "How do I update my display name in my profile?",
            "Can I have two accounts with the same email address?",
            "Where can I find my account creation date?",
            "How do I set my timezone preferences?",
            "Is there a way to customize my dashboard layout?",
        ],
    },
    "general_inquiry": {
        "high": [
            "GDPR data deletion request — legal deadline in 72 hours, requires immediate action.",
            "Compliance audit requires all our data exported today. Very time sensitive.",
            "Legal hold request: need all communications for account archived immediately.",
        ],
        "medium": [
            "Does your platform support HIPAA compliance for healthcare data?",
            "What are your data retention policies for deleted accounts?",
            "Can you provide an SLA agreement for enterprise customers?",
            "How does your platform handle data residency for EU customers?",
            "We need a custom integration — who should we contact?",
        ],
        "low": [
            "Do you have an affiliate or referral program?",
            "Is there a public API documentation page?",
            "What are your support hours and response times?",
            "Do you offer onboarding assistance for new teams?",
            "Are there video tutorials available for getting started?",
        ],
    },
    "shipping": {
        "high": [
            "Order has been sitting in 'processing' for 7 days — event is tomorrow!",
            "Wrong item delivered for a time-sensitive project. Need correct item today.",
            "Package shows delivered but never arrived. Need immediate replacement.",
        ],
        "medium": [
            "My tracking number shows no updates for 4 days. Is my order lost?",
            "Can I change the delivery address on my pending order?",
            "Order arrived with damaged packaging — contents might be affected.",
            "I need expedited shipping on my order placed yesterday.",
        ],
        "low": [
            "Do you ship internationally to South Africa?",
            "What are your standard delivery timeframes?",
            "Can I pick up my order in person?",
            "Is gift wrapping available for orders?",
        ],
    },
}


def generate_ticket(ticket_id: int) -> dict:
    category = random.choice(list(TICKET_TEMPLATES.keys()))
    priority = random.choices(
        ["high", "medium", "low"], weights=[0.20, 0.45, 0.35]
    )[0]

    templates = TICKET_TEMPLATES[category][priority]
    text = random.choice(templates)

    # Add some noise/variation
    prefixes = [
        "", "", "",  # most have no prefix
        "Hi there, ",
        "Hello support team, ",
        "Good day, ",
        "To whom it may concern, ",
    ]
    suffixes = [
        "", "", "",
        " Please help.",
        " Thank you.",
        " Looking forward to your response.",
        " This is really frustrating.",
    ]

    full_text = random.choice(prefixes) + text + random.choice(suffixes)

    created_at = datetime.now() - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    return {
        "ticket_id": f"TKT-{ticket_id:05d}",
        "text": full_text,
        "category": category,
        "priority": priority,
        "created_at": created_at.isoformat(),
        "customer_id": f"CUST-{random.randint(1000, 9999)}",
        "channel": random.choice(["email", "chat", "web_form", "phone_transcript"]),
    }


def generate_dataset(n: int = 1000, seed: int = 42) -> list[dict]:
    random.seed(seed)
    return [generate_ticket(i + 1) for i in range(n)]


if __name__ == "__main__":
    dataset = generate_dataset(1000)
    with open("data/tickets.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} tickets → data/tickets.json")

    # Quick stats
    from collections import Counter
    cats = Counter(t["category"] for t in dataset)
    pris = Counter(t["priority"] for t in dataset)
    print("\nCategory distribution:", dict(cats))
    print("Priority distribution:", dict(pris))
