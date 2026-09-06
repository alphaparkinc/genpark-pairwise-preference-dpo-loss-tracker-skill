"""
MCP Server for Pairwise Preference DPO Loss Tracker Skill.
"""

import json
import sys
from client import DPOLossTracker

TRACKER = DPOLossTracker()


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "evaluate_preference_pair",
                    "description": "Compute DPO loss and implicit rewards for a chosen vs rejected pair",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "pi_w": {"type": "number"},
                            "ref_w": {"type": "number"},
                            "pi_l": {"type": "number"},
                            "ref_l": {"type": "number"},
                            "beta": {"type": "number", "default": 0.1}
                        },
                        "required": ["pi_w", "ref_w", "pi_l", "ref_l"]
                    }
                },
                {
                    "name": "evaluate_batch",
                    "description": "Evaluate batch of pairwise samples",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "pairs": {"type": "array", "items": {"type": "object"}}
                        },
                        "required": ["pairs"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "evaluate_preference_pair":
            if "beta" in args:
                TRACKER.beta = args["beta"]
            res = TRACKER.evaluate_pair(args["pi_w"], args["ref_w"], args["pi_l"], args["ref_l"])
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        elif tool_name == "evaluate_batch":
            res = TRACKER.evaluate_batch(args["pairs"])
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
