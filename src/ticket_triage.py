
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_TOKENS_PER_TICKET = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

ZERO_SHOT_SYSTEM = """
CLASSIFY the customer support ticket.

Choose exactly one category:
billing, delivery, technical, or fraud.

Return JSON only:
{"category": "category_name", "reason": "short reason"}
"""

FEW_SHOT_SYSTEM = """
CLASSIFY the customer support ticket.

Choose exactly one category:
billing, delivery, technical, or fraud.

EXAMPLES:
Ticket: I was charged twice for my order.
Answer: {"category": "billing", "reason": "duplicate charge"}

Ticket: My delivery has not arrived.
Answer: {"category": "delivery", "reason": "late delivery"}

Ticket: The application crashes when I log in.
Answer: {"category": "technical", "reason": "application issue"}

Ticket: Someone made an unauthorized payment using my account.
Answer: {"category": "fraud", "reason": "unauthorized transaction"}

Return JSON only:
{"category": "category_name", "reason": "short reason"}
"""


def redact_text(text):
    if pd.isna(text):
        text = ""

    text = str(text)

    patterns = {
        "emails": (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
        "phones": (r"\b[6-9]\d{9}\b", "[PHONE]"),
        "order_ids": (r"ORD-\d{6}", "[ORDER_ID]"),
    }

    counts = {}

    for name, (pattern, replacement) in patterns.items():
        counts[name] = len(re.findall(pattern, text))
        text = re.sub(pattern, replacement, text)

    return text, counts


def parse_response(response_text):
    try:
        data = json.loads(response_text)

        if (
            isinstance(data, dict)
            and data.get("category") in CATEGORIES
        ):
            return data["category"], "json"

    except (json.JSONDecodeError, TypeError):
        pass

    words = re.findall(r"[a-z]+", str(response_text).lower())
    found = set(words).intersection(CATEGORIES)

    if len(found) == 1:
        return found.pop(), "keyword_fallback"

    return None, "invalid"


def classify_ticket(ticket_text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ticket_text},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = mock_llm.chat(messages)
            message = response.choices[0].message

            category, method = parse_response(message.content)

            return {
                "category": category,
                "method": method,
                "tokens_used": response.usage.total_tokens,
                "status": "classified" if category else "invalid",
            }

        except mock_llm.TransientLLMError:
            delay = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"after transient LLM error. "
                f"Waiting {delay:.3f}s"
            )

            time.sleep(delay)

    return {
        "category": None,
        "method": "invalid",
        "tokens_used": 0,
        "status": "invalid",
    }


def load_data():
    tickets = pd.read_csv(
        RAW / "tickets.csv",
        keep_default_na=False,
    )

    required = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }

    missing = required - set(tickets.columns)

    if missing:
        raise ValueError(f"Missing ticket columns: {missing}")

    segments = pd.read_csv(OUTPUT / "customer_segments.csv")

    if "is_premium" not in segments.columns:
        raise ValueError("is_premium column missing")

    premium_map = (
        segments.drop_duplicates("customer_id")
        .set_index("customer_id")["is_premium"]
        .to_dict()
    )

    return tickets, premium_map


def prepare_tickets(tickets):
    prepared = []

    total_redactions = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0,
    }

    for _, row in tickets.iterrows():
        redacted, counts = redact_text(row["text"])

        for key in total_redactions:
            total_redactions[key] += counts[key]

        token_estimate = mock_llm.est_tokens(redacted)

        if not redacted.strip():
            status = "skipped_empty"
        elif token_estimate > MAX_TOKENS_PER_TICKET:
            status = "flagged_long"
        else:
            status = "eligible"

        prepared.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "order_id": row["order_id"],
            "true_category": row["true_category"],
            "redacted_text": redacted,
            "estimated_tokens": token_estimate,
            "preparation_status": status,
        })

    print("\nPRIVACY REDACTION")

    for key, value in total_redactions.items():
        print(f"{key}: {value}")

    print(
        "Total redactions:",
        sum(total_redactions.values()),
    )

    residual_pattern = (
        r"[\w.+-]+@[\w-]+\.[\w.]+"
        r"|\b[6-9]\d{9}\b"
        r"|ORD-\d{6}"
    )

    residual = sum(
        len(re.findall(residual_pattern, item["redacted_text"]))
        for item in prepared
    )

    print("Residual sensitive patterns:", residual)

    print("\nINPUT GUARDRAILS")

    for status in ["skipped_empty", "flagged_long", "eligible"]:
        count = sum(
            item["preparation_status"] == status
            for item in prepared
        )
        print(f"{status}: {count}")

    return prepared


def run_classification(prepared, system_prompt, run_name):
    mock_llm.reset()
    results = []

    for item in prepared:
        if item["preparation_status"] != "eligible":
            continue

        result = classify_ticket(
            item["redacted_text"],
            system_prompt,
        )

        results.append({
            **item,
            **result,
        })

    classified = [
        item for item in results
        if item["category"] is not None
    ]

    print(f"\n{run_name.upper()} RESULTS")
    print("Eligible tickets:", len(results))
    print("Successfully classified:", len(classified))

    print(
        "JSON outputs:",
        sum(item["method"] == "json" for item in results),
    )

    print(
        "Keyword fallbacks:",
        sum(
            item["method"] == "keyword_fallback"
            for item in results
        ),
    )

    print(
        "Invalid outputs:",
        sum(item["status"] == "invalid" for item in results),
    )

    if classified:
        y_true = [item["true_category"] for item in classified]
        y_pred = [item["category"] for item in classified]

        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred, labels=CATEGORIES))

        print(
            "Accuracy:",
            round(accuracy_score(y_true, y_pred), 4),
        )

        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=CATEGORIES,
                zero_division=0,
            )
        )

    total_tokens = sum(item["tokens_used"] for item in results)

    print("Total tokens:", total_tokens)

    if classified:
        print(
            "Average tokens per classified ticket:",
            round(total_tokens / len(classified), 2),
        )

    return results


def run_budget_demo(prepared, token_budget=6000):
    mock_llm.reset()

    running_tokens = 0
    results = []

    for item in prepared:
        if item["preparation_status"] != "eligible":
            continue

        if running_tokens >= token_budget:
            results.append({
                "ticket_id": item["ticket_id"],
                "status": "budget_exhausted",
                "category": None,
                "tokens_used": 0,
            })
            continue

        result = classify_ticket(
            item["redacted_text"],
            ZERO_SHOT_SYSTEM,
        )

        running_tokens += result["tokens_used"]

        results.append({
            "ticket_id": item["ticket_id"],
            "status": result["status"],
            "category": result["category"],
            "tokens_used": result["tokens_used"],
        })

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT / "triage_budget_demo.csv",
        index=False,
    )

    print("\nBUDGET DEMONSTRATION")
    print("Token budget:", token_budget)
    print("Tokens used:", running_tokens)
    print(output["status"].value_counts())


def route_ticket(category, customer_id, premium_map, status):
    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    ]:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if premium_map.get(customer_id, False):
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


def run_final_triage(prepared, premium_map):
    mock_llm.reset()
    results = []

    for item in prepared:
        if item["preparation_status"] != "eligible":
            status = item["preparation_status"]
            category = None
            method = None
            tokens_used = 0

        else:
            result = classify_ticket(
                item["redacted_text"],
                FEW_SHOT_SYSTEM,
            )

            status = (
                "flagged_invalid_output"
                if result["status"] == "invalid"
                else "classified"
            )

            category = result["category"]
            method = result["method"]
            tokens_used = result["tokens_used"]

        route, priority = route_ticket(
            category,
            item["customer_id"],
            premium_map,
            status,
        )

        results.append({
            **item,
            "category": category,
            "method": method,
            "tokens_used": tokens_used,
            "status": status,
            "segment_name": (
                "Premium"
                if premium_map.get(item["customer_id"], False)
                else "Non-Premium"
            ),
            "priority": priority,
            "route": route,
        })

    output = pd.DataFrame(results)

    required_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    output[required_columns].to_csv(
        OUTPUT / "triage_results.csv",
        index=False,
    )

    print("\nFINAL TRIAGE SUMMARY")
    print(output["status"].value_counts(dropna=False))

    print("\nROUTING SUMMARY")
    print(output["route"].value_counts(dropna=False))

    print("\nPRIORITY SUMMARY")
    print(output["priority"].value_counts(dropna=False))

    print("\nSaved outputs/triage_results.csv")


def main():
    tickets, premium_map = load_data()
    prepared = prepare_tickets(tickets)

    zero_shot_results = run_classification(
        prepared,
        ZERO_SHOT_SYSTEM,
        "zero-shot",
    )

    few_shot_results = run_classification(
        prepared,
        FEW_SHOT_SYSTEM,
        "few-shot",
    )

    pd.DataFrame(zero_shot_results).to_csv(
        OUTPUT / "triage_zero_shot.csv",
        index=False,
    )

    pd.DataFrame(few_shot_results).to_csv(
        OUTPUT / "triage_few_shot.csv",
        index=False,
    )

    run_budget_demo(
        prepared,
        token_budget=6000,
    )

    run_final_triage(
        prepared,
        premium_map,
    )

    print("\nAll Part C classification outputs saved.")


if __name__ == "__main__":
    main()
