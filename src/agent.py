
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from policy_rag import PolicyRAG


RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

ORDERS_FILE = RAW_DIR / "orders.csv"
SCENARIOS_FILE = RAW_DIR / "agent_scenarios.csv"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"

MAX_ITERATIONS = 6
DEFAULT_TOKEN_LIMIT = 6000
DEFAULT_TIME_LIMIT = 10

POLICY_RAG = PolicyRAG(overlap=10)


class SimulatedCrash(Exception):
    """Used only to demonstrate checkpoint recovery."""


# ---------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------

def get_order(order_id, customer_id=None):
    orders = pd.read_csv(ORDERS_FILE, dtype={"order_id": str})

    matches = orders[orders["order_id"] == str(order_id)]

    if matches.empty:
        return {"error": "order_not_found", "order_id": order_id}

    order = matches.iloc[0].to_dict()

    if customer_id and str(order["customer_id"]) != str(customer_id):
        return {"error": "customer_order_mismatch"}

    return {
        "order_id": str(order["order_id"]),
        "customer_id": str(order["customer_id"]),
        "order_ts": str(order["order_ts"]),
        "status": str(order["status"]),
        "order_amount": float(order["order_amount"]),
        "delivery_minutes": (
            None if pd.isna(order["delivery_minutes"])
            else float(order["delivery_minutes"])
        )
    }


def search_policy(query):
    results = POLICY_RAG.retrieve(query, k=3)

    return {
        "query": query,
        "results": [
            {
                "source": item["source"],
                "score": round(item["score"], 4),
                "text": item["text"]
            }
            for item in results
        ]
    }


def calculate_refund(
    order_amount,
    delivery_minutes,
    status="completed",
    request_date=None,
    order_date=None
):
    if str(status).lower() != "completed":
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "reason": "Only completed orders are eligible."
        }

    if delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "reason": "Delivery time is unavailable."
        }

    if request_date and order_date:
        request = pd.Timestamp(request_date)
        ordered = pd.Timestamp(order_date)
        age_days = (request - ordered).days

        if age_days < 0 or age_days > 7:
            return {
                "refund_inr": 0.0,
                "eligible": False,
                "reason": "Refund request is outside the seven-day window.",
                "days_since_order": age_days
            }

        window_status = "verified_within_7_days"
    else:
        window_status = "not_verifiable_missing_request_date"

    if float(delivery_minutes) <= 45:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "reason": "Delivery was completed within 45 minutes.",
            "window_status": window_status
        }

    return {
        "refund_inr": round(float(order_amount), 2),
        "eligible": True,
        "reason": "Completed order delivered in more than 45 minutes.",
        "window_status": window_status
    }


def issue_refund(order_id, refund_inr):
    amount = round(float(refund_inr), 2)

    if amount <= 0:
        return {"success": False, "reason": "Refund amount must be positive."}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate issued refunds.
    if LEDGER_FILE.exists():
        with LEDGER_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                entry = json.loads(line)

                if (
                    entry.get("order_id") == order_id
                    and float(entry.get("refund_inr", 0)) == amount
                    and entry.get("status") == "issued"
                ):
                    return {
                        "success": True,
                        "already_issued": True,
                        "order_id": order_id,
                        "refund_inr": amount
                    }

    entry = {
        "tool_name": "issue_refund",
        "order_id": order_id,
        "refund_inr": amount,
        "status": "issued",
        "timestamp": pd.Timestamp.now().isoformat()
    }

    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {
        "success": True,
        "already_issued": False,
        "order_id": order_id,
        "refund_inr": amount
    }


def escalate_to_human(order_id, refund_inr):
    amount = round(float(refund_inr), 2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "tool_name": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": amount,
        "status": "escalated",
        "route": "human_support_lead",
        "timestamp": pd.Timestamp.now().isoformat()
    }

    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": amount,
        "route": "human_support_lead",
        "reason": "Refund exceeds INR 1000 and requires human approval."
    }


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve an order and verify its customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "customer_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant policy passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility and amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a high-value refund for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


# ---------------------------------------------------------
# CHECKPOINT HELPERS
# ---------------------------------------------------------

def checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(state["run_id"])
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )
    temporary.replace(path)


def load_checkpoint(run_id):
    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for {run_id}")

    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# ---------------------------------------------------------
# AGENT HELPERS
# ---------------------------------------------------------

def extract_order_id(message):
    match = re.search(r"ORD-\d{6}", message.upper())
    return match.group(0) if match else None


def make_fallback(status, reason, steps, llm_calls, tokens_used):
    return {
        "status": status,
        "final_answer": reason,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


def final_status(answer):
    text = (answer or "").strip().upper()

    if text.startswith("REFUND:"):
        return "REFUND"
    if text.startswith("ESCALATE:"):
        return "ESCALATE"
    if text.startswith("DENIED:"):
        return "DENIED"
    if text.startswith("NOT_FOUND:"):
        return "NOT_FOUND"
    if text.startswith("NEEDS_INFO:"):
        return "NEEDS_INFO"

    return "COMPLETED"


def execute_tool(name, arguments):
    if name not in TOOL_MAP:
        return {"error": f"Unknown tool: {name}"}

    try:
        return TOOL_MAP[name](**arguments)
    except Exception as error:
        return {"error": str(error)}


# ---------------------------------------------------------
# REACT AGENT
# ---------------------------------------------------------

def run_agent(
    message,
    customer_id=None,
    max_iterations=MAX_ITERATIONS,
    token_limit=DEFAULT_TOKEN_LIMIT,
    time_limit_seconds=DEFAULT_TIME_LIMIT,
    run_id="agent_run",
    crash_after_tools=None,
    initial_state=None
):
    started = time.monotonic()

    if initial_state is None:
        user_message = message

        if customer_id:
            user_message = (
                f"Customer ID: {customer_id}\n"
                f"Request: {message}"
            )

        state = {
            "run_id": run_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "AGENT: You are an Instamart customer support "
                        "agent. Use the available tools to inspect orders, "
                        "retrieve policies, calculate refunds, issue "
                        "eligible refunds, and escalate refunds above "
                        "INR 1000. Never invent order details or refund "
                        "amounts. Only completed orders delivered in "
                        "more than 45 minutes qualify for a late-delivery "
                        "refund. Cancelled orders are not refundable. "
                        "Refunds above INR 1000 require human approval."
                    )
                },
                {"role": "user", "content": user_message}
            ],
            "steps": [],
            "llm_calls": 0,
            "tokens_used": 0,
            "tool_names": [],
            "tool_execution_count": 0,
            "status": "RUNNING",
            "final_answer": None
        }
        save_checkpoint(state)
    else:
        state = initial_state

    while state["llm_calls"] < max_iterations:
        if time_limit_seconds <= 0 or (
            time.monotonic() - started >= time_limit_seconds
        ):
            result = make_fallback(
                "FALLBACK_TIMEOUT",
                "Human review required: agent time limit reached.",
                state["steps"],
                state["llm_calls"],
                state["tokens_used"]
            )
            state.update(result)
            save_checkpoint(state)
            return result

        if state["tokens_used"] >= token_limit:
            result = make_fallback(
                "FALLBACK_TOKEN_LIMIT",
                "Human review required: token budget reached.",
                state["steps"],
                state["llm_calls"],
                state["tokens_used"]
            )
            state.update(result)
            save_checkpoint(state)
            return result

        try:
            response = mock_llm.chat(
                state["messages"],
                tools=TOOLS_SPEC
            )
        except mock_llm.TransientLLMError as error:
            state["steps"].append({
                "type": "llm_retry",
                "error": str(error)
            })
            save_checkpoint(state)
            continue

        state["llm_calls"] += 1

        usage = getattr(response, "usage", None)
        if usage is not None:
            state["tokens_used"] += int(
                getattr(usage, "total_tokens", 0)
            )

        message_object = response.choices[0].message

        # Stop before executing a tool if this response exceeded budget.
        if state["tokens_used"] > token_limit:
            result = make_fallback(
                "FALLBACK_TOKEN_LIMIT",
                "Human review required: token budget exceeded.",
                state["steps"],
                state["llm_calls"],
                state["tokens_used"]
            )
            state.update(result)
            save_checkpoint(state)
            return result

        tool_calls = getattr(message_object, "tool_calls", [])

        if not tool_calls:
            answer = message_object.content or ""
            status = final_status(answer)

            result = {
                "status": status,
                "final_answer": answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"]
            }

            state.update(result)
            save_checkpoint(state)
            return result

        for tool_call in tool_calls:
            tool_name = tool_call.name
            raw_arguments = tool_call.arguments

            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )

            # Add the scenario customer ID to order lookup when available.
            if tool_name == "get_order" and customer_id:
                arguments.setdefault("customer_id", customer_id)

            observation = execute_tool(tool_name, arguments)

            observation_text = json.dumps(
                observation,
                default=str
            )

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": observation_text
            })

            state["tool_names"].append(tool_name)
            state["tool_execution_count"] += 1

            state["steps"].append({
                "type": "tool",
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            # Persist immediately after every tool observation.
            save_checkpoint(state)

            if (
                crash_after_tools is not None
                and state["tool_execution_count"] == crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after tool {crash_after_tools}"
                )

    result = make_fallback(
        "FALLBACK_MAX_ITERATIONS",
        "Human review required: maximum agent iterations reached.",
        state["steps"],
        state["llm_calls"],
        state["tokens_used"]
    )

    state.update(result)
    save_checkpoint(state)
    return result


def resume_agent(run_id, **kwargs):
    state = load_checkpoint(run_id)

    if state["status"] != "RUNNING":
        return {
            "status": state["status"],
            "final_answer": state["final_answer"],
            "steps": state["steps"],
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]
        }

    return run_agent(
        message="",
        run_id=run_id,
        initial_state=state,
        **kwargs
    )


# ---------------------------------------------------------
# SCENARIO RUNNER AND SAFETY TESTS
# ---------------------------------------------------------

def save_run_result(scenario_id, result):
    record = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "steps": result["steps"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"]
    }
    append_jsonl(RUNS_FILE, record)


def run_scenario(row):
    scenario_id = str(row["scenario_id"])
    customer_id = str(row["customer_id"])
    message = str(row["message"])
    run_id = f"scenario_{scenario_id.lower()}"

    if scenario_id == "S1":
        try:
            run_agent(
                message,
                customer_id=customer_id,
                run_id=run_id,
                crash_after_tools=2
            )
        except SimulatedCrash as error:
            print("\nCHECKPOINT TEST:", error)

        result = resume_agent(run_id)

        expected_tools = [
            "get_order",
            "search_policy",
            "calculate_refund",
            "issue_refund"
        ]

        actual_tools = [
            step["tool"]
            for step in result["steps"]
            if step["type"] == "tool"
        ]

        print("RESUMED TOOL SEQUENCE:", actual_tools)
        print("EXPECTED TOOL SEQUENCE:", expected_tools)
        print("CHECKPOINT RESUME PASSED:", actual_tools == expected_tools)

    else:
        result = run_agent(
            message,
            customer_id=customer_id,
            run_id=run_id
        )

    save_run_result(scenario_id, result)

    print(
        f"{scenario_id}: {result['status']} | "
        f"LLM calls: {result['llm_calls']} | "
        f"Tokens: {result['tokens_used']}"
    )
    print("Answer:", result["final_answer"])

    return result


def run_safety_tests():
    print("\nSAFETY FALLBACK TESTS")

    token_result = run_agent(
        "My order ORD-100048 reached me very late. I would like a refund.",
        customer_id="C0007",
        token_limit=200,
        run_id="test_token_limit"
    )
    print("Token limit:", token_result["status"])

    timeout_result = run_agent(
        "My order ORD-100048 reached me very late. I would like a refund.",
        customer_id="C0007",
        time_limit_seconds=0,
        run_id="test_timeout"
    )
    print("Timeout:", timeout_result["status"])

    loop_result = run_agent(
        "[LOOP] Refund order ORD-100048.",
        customer_id="C0007",
        max_iterations=6,
        run_id="test_max_iterations"
    )
    print("Max iterations:", loop_result["status"])

    return token_result, timeout_result, loop_result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Start this demonstration with clean output files.
    for path in (RUNS_FILE, LEDGER_FILE):
        if path.exists():
            path.unlink()

    print("REACT AGENT — SEVEN SCENARIOS")

    scenarios = pd.read_csv(SCENARIOS_FILE).fillna("")

    results = []

    for _, row in scenarios.iterrows():
        results.append(run_scenario(row))

    print("\nSCENARIO SUMMARY")
    for row, result in zip(scenarios.to_dict("records"), results):
        expected = str(row["expected_outcome"])
        actual = result["status"]
        match = actual == expected or (expected == "FALLBACK" and actual.startswith("FALLBACK_"))
        print(
            row["scenario_id"],
            "| expected:", expected,
            "| actual:", actual,
            "| match:", match
        )

    run_safety_tests()

    print("\nOUTPUT FILES")
    print("Agent runs:", RUNS_FILE.relative_to(ROOT))
    print("Refund ledger:", LEDGER_FILE.relative_to(ROOT))
    print("Checkpoints:", STATE_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
