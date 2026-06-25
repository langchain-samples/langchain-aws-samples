"""Issue-management Lambda target for AgentCore Gateway."""
import json


def _lookup_customer_tickets(customer_id: str) -> list[dict]:
    return [
        {"id": "T-1801", "product": "SmartCam", "status": "resolved", "category": "wifi"},
        {"id": "T-2044", "product": "SmartPlug", "status": "resolved", "category": "wifi"},
        {"id": "T-2299", "product": "SmartHome Hub", "status": "open", "category": "wifi"},
    ]


def _tool_and_args(event, context) -> tuple[str, dict]:
    custom = getattr(getattr(context, "client_context", None), "custom", None) or {}
    tool = custom.get("bedrockAgentCoreToolName", "")
    if tool:
        return tool.split("___")[-1], event if isinstance(event, dict) else {}

    event = event if isinstance(event, dict) else {}
    return event.get("tool") or event.get("name") or "lookup_customer_tickets", event.get("arguments") or event.get("input") or event


def lambda_handler(event, context):
    tool, args = _tool_and_args(event, context)

    if tool == "lookup_customer_tickets":
        customer_id = args.get("customer_id")
        if not customer_id:
            return {"statusCode": 400, "body": json.dumps({"error": "customer_id is required"})}
        return {"statusCode": 200, "body": json.dumps(_lookup_customer_tickets(customer_id))}

    return {"statusCode": 400, "body": json.dumps({"error": f"unknown tool: {tool}"})}
