"""Order-management Lambda target for AgentCore Gateway."""
import json
from datetime import datetime, timezone


def _lookup_order(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "status": "delivered",
        "delivered_at": "2026-05-08",
        "items": [{"sku": "SH-HUB-V2", "name": "SmartHome Hub", "qty": 1, "price_usd": 129.99}],
        "customer_id": "C-88421",
        "total_usd": 129.99,
    }


def _issue_refund(order_id: str, amount_usd: float, reason: str) -> dict:
    return {
        "refund_id": f"R-{abs(hash((order_id, amount_usd, reason))) % 10**8:08d}",
        "order_id": order_id,
        "status": "issued",
        "amount_usd": amount_usd,
        "currency": "USD",
        "reason": reason,
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }


def _tool_and_args(event, context) -> tuple[str, dict]:
    custom = getattr(getattr(context, "client_context", None), "custom", None) or {}
    tool = custom.get("bedrockAgentCoreToolName", "")
    if tool:
        return tool.split("___")[-1], event if isinstance(event, dict) else {}

    event = event if isinstance(event, dict) else {}
    return event.get("tool") or event.get("name") or "lookup_order", event.get("arguments") or event.get("input") or event


def _response(status_code: int, body: dict) -> dict:
    return {"statusCode": status_code, "body": json.dumps(body)}


def lambda_handler(event, context):
    tool, args = _tool_and_args(event, context)

    if tool == "lookup_order":
        order_id = args.get("order_id")
        if not order_id:
            return _response(400, {"error": "order_id is required"})
        return _response(200, _lookup_order(order_id))

    if tool == "issue_refund":
        order_id = args.get("order_id")
        amount = args.get("amount_usd")
        reason = args.get("reason") or "support-agent initiated"
        if not order_id or amount is None:
            return _response(400, {"error": "order_id and amount_usd are required"})
        return _response(200, _issue_refund(order_id, float(amount), reason))

    return _response(400, {"error": f"unknown tool: {tool}"})
