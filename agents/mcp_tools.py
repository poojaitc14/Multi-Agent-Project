"""MCP tool wiring for the 4 agents -- connects to the single, real
mcp-servers/orchestrator_server.py over streamable-http (project-plan.md
Q55), not an in-memory/mocked transport, since CrewAI's MCPServerAdapter
only supports real transports. Each agent authenticates with its own
per-agent bearer token (Q55) and is only ever given the specific tool names
it's meant to have, matching the "Tool List Per Agent" section -- even
though IssueRefundGate is the actual non-bypassable enforcement, restricting
which tools each Agent even sees is defense in depth, not the real gate.

Requires mcp-servers/orchestrator_server.py to already be running (real
process, real port) before any of these are used -- there is no in-process
fallback, by design: every tool call an agent makes must be a real MCP
round-trip, per the rubric requirement.
"""

import os

from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = (
    f"http://{os.environ.get('MCP_SERVER_HOST', '127.0.0.1')}:"
    f"{os.environ.get('MCP_SERVER_PORT', 8001)}/mcp"
)

ORCHESTRATOR_TOOLS = (
    "resolve_customer_ref", "get_order", "issue_refund", "get_conversation_state",
    "put_conversation_state", "store_photo", "get_photo", "store_transcript",
)
IMAGE_PARSING_TOOLS = ("get_photo", "get_product_reference", "redact_photo", "analyze_image")
FRAUD_SCORING_TOOLS = (
    "get_order", "get_account_info", "get_claim_frequency",
    "increment_claim_frequency", "get_tracking_status", "score_fraud_risk",
)
DECISION_TOOLS = ("search_refund_policy", "apply_decision_matrix")


def _server_params(token: str) -> dict:
    return {
        "url": MCP_SERVER_URL,
        "transport": "streamable-http",
        "headers": {"Authorization": f"Bearer {token}"},
    }


def orchestrator_mcp_adapter() -> MCPServerAdapter:
    return MCPServerAdapter(_server_params(os.environ["ORCHESTRATOR_MCP_TOKEN"]), *ORCHESTRATOR_TOOLS)


def image_parsing_mcp_adapter() -> MCPServerAdapter:
    return MCPServerAdapter(_server_params(os.environ["IMAGE_PARSING_MCP_TOKEN"]), *IMAGE_PARSING_TOOLS)


def fraud_scoring_mcp_adapter() -> MCPServerAdapter:
    return MCPServerAdapter(_server_params(os.environ["FRAUD_SCORING_MCP_TOKEN"]), *FRAUD_SCORING_TOOLS)


def decision_mcp_adapter() -> MCPServerAdapter:
    return MCPServerAdapter(_server_params(os.environ["DECISION_MCP_TOKEN"]), *DECISION_TOOLS)
