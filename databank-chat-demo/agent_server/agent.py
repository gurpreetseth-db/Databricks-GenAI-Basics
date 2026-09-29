from agents.mcp import MCPServer, MCPServerManager
from typing import AsyncGenerator, List

import json
import logging
import os
import mlflow
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_session_id,
    get_user_workspace_client,
    process_agent_stream_events,
)
from agent_server.history import normalize_history_items
from agent_server.memory_manager import LakebaseMemoryManager

logger = logging.getLogger(__name__)

# NOTE: this will work for all databricks models OTHER than GPT-OSS, which uses a slightly different API
set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()

# ---------------------------------------------------------------------------
# Configuration — read from env (the deploy notebook writes these into app.yaml
# from Config_Parameters.py). Fallbacks keep the app runnable if a var is unset.
# ---------------------------------------------------------------------------
NAME = os.getenv("APP_NAME", "databank-chat-demo")

# Memory mode drives ALL persistence/recall behaviour.
#   simple    -> no memory (ephemeral)
#   shortterm -> conversation log in Lakebase for the session, no fact recall
#   longterm  -> conversation log + durable per-user fact extraction/recall
MEMORY_MODE = os.getenv("MEMORY_MODE", "simple").strip().lower()
LONG_TERM_ENABLED = MEMORY_MODE == "longterm"

# Unity Catalog location of the tools (mirrors Config_Parameters.py via env)
_CATALOG = os.getenv("UC_CATALOG", "gurpreetsethi_databank_lab")
_SCHEMA = os.getenv("UC_SCHEMA", "gurpreetsethi_financial_data")
_UC_FUNCTIONS = [
    f.strip()
    for f in os.getenv(
        "UC_FUNCTIONS",
        "calculate_customer_risk,flag_suspicious_transactions,get_portfolio_summary",
    ).split(",")
    if f.strip()
]
_VECTOR_INDEX = os.getenv("VECTOR_INDEX", f"{_CATALOG}.{_SCHEMA}.product-docs-index")
_GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f1b6c9b5ad17bca4c13c1991f01333")
_GENIE_NAME = os.getenv("GENIE_NAME", "gurpreetsethi-DataBank-Financial-Advisor")
SYSTEM_PROMPT = 'Use Vector Search Index to 1. Search DataBank product brochures, FAQs, and terms & conditions, product knowledge and document questions2. Use for questions about product features, interest rates, fees, eligibility criteria, FSCS protection, or regulatory terms.- Use Genie Spaces to 1. Query customer records, account balances, transaction history, and support tickets using natural language SQL.2. Use for any questions about specific customer account data, transaction queries, portfolio values, spending patterns, or open support tickets.- Use calculate_customer_risk to Calculate a numeric risk score (0-100) for a DataBank customer. Use when asked about a customer\'s investment risk tolerance, risk appetite, risk profile calculation, or suitability for investment products.- Use get_portfolio_summary to Retrieve a complete portfolio summary for a customer including all accounts, product types, current balances, and total assets. Use for portfolio reviews, portfolio overviews, financial health checks, or account overviews, account balances,- Use fraud_and_anomaly_check to Check for suspicious, fraudulent, or anomalous transactions for a customer. Use when asked about fraud, unusual activity, blocked transactions, or security concerns on an account.Always:1. If a customer ID is mentioned, call get_portfolio_summary and calculate_risk_score first.2. For product recommendations, consider the customer\'s risk score before recommending.3. For investment products, always include: Capital at risk - investments can fall as well as rise.4. Be concise and professional. Financial advisors are busy.5. Output formatting: whenever you present query results, customer records, portfolios, transactions, balances, support tickets, or any tabular or multi-row data (especially anything returned by Genie or a tool), render it as a GitHub-flavored Markdown table with a header row and one row per record. Never show raw JSON, Python dicts, or field-by-field bullet lists for tabular data. Include currency symbols and units in the cells, right-align numeric columns where possible, and keep column names short. Add a one-line summary sentence before the table and the key takeaways after it. If a result set is very wide or has many rows, show the most relevant columns and the top rows and note that the table was truncated.'
# AI Gateway model-service route (UC-qualified). Points at Claude Sonnet 4.5.
MODEL = os.getenv(
    "MODEL_ROUTE",
    f"{_CATALOG}.{_SCHEMA}.gurpreetsethi-databank-llm-route",
)


def _build_mcp_servers() -> list[tuple[str, str]]:
    """(display_name, mcp_path) for every tool, derived from UC catalog/schema."""
    servers: list[tuple[str, str]] = []
    for fn in _UC_FUNCTIONS:
        servers.append(
            (
                f"UC Function: {_CATALOG}.{_SCHEMA}.{fn}",
                f"/api/2.0/mcp/functions/{_CATALOG}/{_SCHEMA}/{fn}",
            )
        )
    index_short = _VECTOR_INDEX.split(".")[-1]
    servers.append(
        (
            f"Vector Search: {_VECTOR_INDEX}",
            f"/api/2.0/mcp/ai-search/{_CATALOG}/{_SCHEMA}/{index_short}",
        )
    )
    servers.append(
        (
            f"Genie Agent: {_GENIE_NAME}",
            f"/api/2.0/mcp/genie/{_GENIE_SPACE_ID}",
        )
    )
    return servers


MCP_SERVERS = _build_mcp_servers()


# ================================================================
# Memory Integration — Lakebase Postgres (mode-gated)
# ================================================================
# Only shortterm/longterm modes use the backend Lakebase memory manager.
# Simple mode runs fully stateless. Initialisation is graceful: if Lakebase is
# unreachable the app still answers questions, it just loses memory persistence.
if MEMORY_MODE in ("shortterm", "longterm"):
    try:
        memory = LakebaseMemoryManager()
        MEMORY_ENABLED = True
        logger.info(f"Lakebase memory manager initialised (mode={MEMORY_MODE})")
    except Exception as e:
        memory = None
        MEMORY_ENABLED = False
        logger.warning(f"Lakebase memory disabled (app will run stateless): {e}")
else:
    memory = None
    MEMORY_ENABLED = False
    logger.info("MEMORY_MODE=simple — running stateless (no Lakebase memory)")

# Map session_id (from request.context.conversation_id) to a
# Lakebase conversation UUID so we can track turns across calls.
_active_sessions: dict[str, str] = {}

# Extract long-term facts every FACT_EXTRACTION_INTERVAL messages
FACT_EXTRACTION_INTERVAL = 6


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def _get_or_create_conversation(session_id: str, user_id: str) -> str:
    """Map a session to a Lakebase conversation (create on first call)."""
    if session_id in _active_sessions:
        return _active_sessions[session_id]
    conv_id = memory.start_conversation(
        user_id=user_id,
        title=f"App Session {session_id[:8]}"
    )
    _active_sessions[session_id] = conv_id
    return conv_id


def _extract_latest_user_message(input_items: list) -> str | None:
    """Extract the most recent user message text from request.input."""
    for item in reversed(input_items):
        item_dict = item.model_dump() if hasattr(item, "model_dump") else item
        if item_dict.get("role") == "user":
            content = item_dict.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "input_text"
                ]
                return " ".join(texts) if texts else None
    return None


def _extract_assistant_text(output_items: list) -> str:
    """Extract assistant response text from Runner output items."""
    texts = []
    for item in output_items:
        d = item if isinstance(item, dict) else (
            item.model_dump() if hasattr(item, "model_dump") else {}
        )
        if d.get("type") == "message" and d.get("role") == "assistant":
            content = d.get("content", [])
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
    return " ".join(texts)


def _build_enriched_instructions(user_id: str) -> str:
    """Enrich system prompt with long-term memories for this user.

    Only long-term mode recalls durable facts; shortterm/simple return the
    base prompt unchanged.
    """
    if not LONG_TERM_ENABLED:
        return SYSTEM_PROMPT
    try:
        memories = memory.get_long_term_memories(user_id=user_id)
        if not memories:
            return SYSTEM_PROMPT
        ctx = memory.format_memories_for_prompt(memories)
        return (
            f"{SYSTEM_PROMPT}\n{ctx}\n"
            "Use these known facts to personalise your responses "
            "naturally, as a real advisor who remembers past conversations."
        )
    except Exception as e:
        logger.warning(f"Failed to load long-term memories: {e}")
        return SYSTEM_PROMPT


def _maybe_extract_facts(conv_id: str, user_id: str):
    """Trigger background fact extraction after every N messages (longterm only)."""
    if not LONG_TERM_ENABLED:
        return
    try:
        history = memory.get_conversation_history(conv_id)
        # Only extract when we have enough new content
        if len(history) > 0 and len(history) % FACT_EXTRACTION_INTERVAL == 0:
            from openai import OpenAI
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            tok = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")
            client = OpenAI(
                api_key=tok,
                base_url=f"{w.config.host}/ai-gateway/mlflow/v1"
            )
            memory.extract_and_store_facts(client, MODEL, conv_id, user_id)
            logger.info(f"Long-term facts extracted for conversation {conv_id}")
    except Exception as e:
        logger.warning(f"Fact extraction skipped: {e}")


# ----------------------------------------------------------------
# MCP server setup (unchanged)
# ----------------------------------------------------------------
def get_mcp_user_workspace_client():
    # Uncomment the line below to enable on-behalf-of-user authentication
    # return get_user_workspace_client()
    return None


def init_mcp_servers():
    user_workspace_client = get_mcp_user_workspace_client()
    return [
        McpServer(
            name=name,
            url=build_mcp_url(url, user_workspace_client),
            workspace_client=user_workspace_client,
        )
        for (name, url) in MCP_SERVERS
    ]


def create_agent(mcp_servers: list[MCPServer], instructions: str = SYSTEM_PROMPT) -> Agent:
    """Create agent with (optionally enriched) instructions."""
    return Agent(
        name=NAME,
        instructions=instructions,
        model=MODEL,
        mcp_servers=mcp_servers,
    )


# ----------------------------------------------------------------
# invoke() — non-streaming endpoint with memory
# ----------------------------------------------------------------
@invoke()
async def invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers=mcp_servers, connect_in_parallel=True) as manager:

        # --- Memory: resolve user + session ---
        session_id = get_session_id(request)
        user_id = memory.user if MEMORY_ENABLED else "unknown"
        conv_id = None

        if MEMORY_ENABLED and session_id:
            try:
                conv_id = _get_or_create_conversation(session_id, user_id)
                user_msg = _extract_latest_user_message(request.input)
                if user_msg:
                    memory.save_message(conv_id, "user", user_msg)
            except Exception as e:
                logger.warning(f"Memory save (user) failed: {e}")

        # --- Build agent with enriched instructions ---
        instructions = _build_enriched_instructions(user_id)
        agent = create_agent(manager.active_servers, instructions)

        # --- Run agent ---
        messages = normalize_history_items(
            [i.model_dump() for i in request.input]
        )
        result = await Runner.run(agent, messages)
        output = [item.to_input_item() for item in result.new_items]

        # --- Memory: save assistant response ---
        if MEMORY_ENABLED and conv_id:
            try:
                assistant_text = _extract_assistant_text(output)
                if assistant_text:
                    memory.save_message(conv_id, "assistant", assistant_text)
                _maybe_extract_facts(conv_id, user_id)
            except Exception as e:
                logger.warning(f"Memory save (assistant) failed: {e}")

        return ResponsesAgentResponse(output=output)


# ----------------------------------------------------------------
# stream() — streaming endpoint with memory
# ----------------------------------------------------------------
@stream()
async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers=mcp_servers, connect_in_parallel=True) as manager:

        # --- Memory: resolve user + session ---
        session_id = get_session_id(request) if hasattr(request, "context") else None
        user_id = memory.user if MEMORY_ENABLED else "unknown"
        conv_id = None

        if MEMORY_ENABLED and session_id:
            try:
                conv_id = _get_or_create_conversation(session_id, user_id)
                user_msg = _extract_latest_user_message(request.input)
                if user_msg:
                    memory.save_message(conv_id, "user", user_msg)
            except Exception as e:
                logger.warning(f"Memory save (user) failed: {e}")

        # --- Build agent with enriched instructions ---
        instructions = _build_enriched_instructions(user_id)
        agent = create_agent(manager.active_servers, instructions)

        # --- Stream agent ---
        messages = normalize_history_items(
            [i.model_dump() for i in request.input]
        )
        result = Runner.run_streamed(agent, input=messages)

        collected_text = []
        async for event in process_agent_stream_events(result.stream_events()):
            # Capture streamed text for memory persistence
            if isinstance(event, dict):
                if (event.get("type") == "response.output_text.delta"
                        and event.get("delta")):
                    collected_text.append(event["delta"])
            yield event

        # --- Memory: save streamed assistant response ---
        if MEMORY_ENABLED and conv_id and collected_text:
            try:
                full_response = "".join(collected_text)
                memory.save_message(conv_id, "assistant", full_response)
                _maybe_extract_facts(conv_id, user_id)
            except Exception as e:
                logger.warning(f"Memory save (stream) failed: {e}")
