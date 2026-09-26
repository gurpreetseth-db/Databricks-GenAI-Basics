# Databricks notebook source
# DBTITLE 1,Step 1 - Reference Parameters
# MAGIC %run ./Config_Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 08: Playgroun & Databricks Apps (Gradio)
# MAGIC **Duration:** ~25 minutes | **Prerequisite:** Module 07 (agent endpoint deployed)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is a Playground?
# MAGIC
# MAGIC **Playground** a no-code chat environment for testing, promotion and compariing large language models (LLMs) and prototyping AI agents. Key Features:
# MAGIC - **Side-by-Side Comparison** add multipole endpoints to compare responses from different models simultaneously.
# MAGIC - **Parameter Tuning** adjust settings like temperature, top K, and maximum tokns to control model behavior and creativity.
# MAGIC - **Tool Integration** connect user-defined functions from the Unity Catalog or Model Context Protocol (MCP) servers to build and test tool-calling agents.
# MAGIC - **Export Options** move your prototypes directly into Databricks Apps or code notebook when you are ready to productionize.
# MAGIC
# MAGIC
# MAGIC ### What You’ll Deploy
# MAGIC ```
# MAGIC Browser
# MAGIC   ↓  (HTTPS)
# MAGIC Playground (AI/ML)
# MAGIC   ↓  
# MAGIC Choose Gemma 3 12 B model
# MAGIC   ↓
# MAGIC Tools
# MAGIC   ├── Genie Space        ← Databank-Financial-Advisor
# MAGIC   ├── AI Search          ← Product-Docs-Index  
# MAGIC   └── UC Functions       ← calculate_customer_risk, flag_suspicious_transactions, get_portfolio_summary
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - AI/ML -> **Playground**
# MAGIC - Select **Gemma 3 12 B** model
# MAGIC - Click **Task** -> **Add** 
# MAGIC   - **Genie Space**
# MAGIC   - **AI Search Index**
# MAGIC   - **UC Functions (calculate_customer_risk, flag_suspicious_transactions, get_portfolio_summary)**
# MAGIC - **System Prompt**
# MAGIC
# MAGIC   ```Use Vector Search Index to 1. Search DataBank product brochures, FAQs, and terms & conditions, product knowledge and document questions2. Use for questions about product features, interest rates, fees, eligibility criteria, FSCS protection, or regulatory terms.- Use Genie Spaces to 1. Query customer records, account balances, transaction history, and support tickets using natural language SQL.2. Use for any questions about specific customer account data, transaction queries,  portfolio values, spending patterns, or open support tickets.- Use calculate_customer_risk to Calculate a numeric risk score (0-100) for a DataBank customer. Use when asked about a customer's investment risk tolerance, risk appetite, risk profile calculation, or suitability for investment products.- Use get_portfolio_summary to Retrieve a complete portfolio summary for a customer including all accounts, product types, current balances, and total assets. Use for portfolio reviews, portfolio overviews, financial health checks, or account overviews, account balances,- Use fraud_and_anomaly_check to  Check for suspicious, fraudulent, or anomalous transactions for a customer. Use when asked about fraud, unusual activity, blocked transactions, or security concerns on an account.Always:1. If a customer ID is mentioned, call get_portfolio_summary and calculate_risk_score first.2. For product recommendations, consider the customer’s risk score before recommending.3. For investment products, always include: Capital at risk - investments can fall as well as rise.4. Be concise and professional. Financial advisors are busy.```
# MAGIC
# MAGIC ![](./img/Playground_1.jpg) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Click **Get Code**
# MAGIC
# MAGIC ![](./img/Playground_2.jpg) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select **Export to Databricks Apps (Recommended)**
# MAGIC
# MAGIC Select ML Model that we created as part of **00_setup_prerequisites** notebook and hit **Export**
# MAGIC ![](./img/Playground_3.jpg) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Databricks APP gets deployed**
# MAGIC
# MAGIC - A Databricks Application will be deployed using https://github.com/databricks/app-templates as a tempalte

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect Application with AI Gateway
# MAGIC
# MAGIC #### Update agent.py file
# MAGIC
# MAGIC - ***add ```use_ai_gateway=True``` while doing ```AsyncDatabricksOpenAI``` function***
# MAGIC - ***Update Model name to your AI Gateway name***
# MAGIC
# MAGIC ![](./img/Playground_4.jpg)
# MAGIC ```
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agent.py - changes
# MAGIC
# MAGIC Replace code base in your agent.py in case application is giving error to render chat session.
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC def get_mcp_user_workspace_client():
# MAGIC     # Uncomment the line below to enable on-behalf-of-user authentication
# MAGIC     # return get_user_workspace_client()
# MAGIC     return None
# MAGIC
# MAGIC
# MAGIC def init_mcp_servers():
# MAGIC     user_workspace_client = get_mcp_user_workspace_client()
# MAGIC     return [
# MAGIC         McpServer(
# MAGIC             name=name,
# MAGIC             url=build_mcp_url(url, user_workspace_client),
# MAGIC             workspace_client=user_workspace_client,
# MAGIC         )
# MAGIC         for (name, url) in MCP_SERVERS
# MAGIC     ]
# MAGIC
# MAGIC def create_agent(mcp_servers: List[MCPServer]) -> Agent:
# MAGIC     return Agent(
# MAGIC         name=NAME,
# MAGIC         instructions=SYSTEM_PROMPT,
# MAGIC         model=MODEL,
# MAGIC         mcp_servers=mcp_servers,
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC def _extract_text(content):
# MAGIC     """Extract plain text from Responses API content (string or list of blocks)."""
# MAGIC     if isinstance(content, str):
# MAGIC         return content
# MAGIC     if isinstance(content, list):
# MAGIC         parts = []
# MAGIC         for block in content:
# MAGIC             if isinstance(block, dict):
# MAGIC                 # output_text (assistant), input_text (user), or generic text
# MAGIC                 text = block.get("text", "")
# MAGIC                 if text:
# MAGIC                     parts.append(text)
# MAGIC             elif isinstance(block, str):
# MAGIC                 parts.append(block)
# MAGIC         return "\n".join(parts)
# MAGIC     return str(content) if content else ""
# MAGIC
# MAGIC
# MAGIC def normalize_input_items(items):
# MAGIC     """Normalize Responses API input items to chat-completions-style dicts
# MAGIC     that the OpenAI Agents SDK can pass to the model.
# MAGIC
# MAGIC     Only user and assistant *message* items are kept; function_call and
# MAGIC     function_call_output items from prior turns are dropped because the
# MAGIC     agent will make its own tool calls. Consecutive same-role messages
# MAGIC     are merged to satisfy models that require strict alternating roles.
# MAGIC     """
# MAGIC     messages = []
# MAGIC     for item in items:
# MAGIC         dumped = item.model_dump()
# MAGIC         item_type = dumped.get("type")
# MAGIC         role = dumped.get("role")
# MAGIC
# MAGIC         # Only keep message-type items with user or assistant role.
# MAGIC         # Drop function_call, function_call_output, and anything else.
# MAGIC         if item_type != "message" or role not in ("user", "assistant"):
# MAGIC             continue
# MAGIC
# MAGIC         text = _extract_text(dumped.get("content", ""))
# MAGIC         if not text:
# MAGIC             continue
# MAGIC
# MAGIC         messages.append({"role": role, "content": text})
# MAGIC
# MAGIC     # Merge consecutive same-role messages (safety net)
# MAGIC     merged = []
# MAGIC     for msg in messages:
# MAGIC         if merged and merged[-1]["role"] == msg["role"]:
# MAGIC             merged[-1]["content"] += "\n" + msg["content"]
# MAGIC         else:
# MAGIC             merged.append(msg)
# MAGIC
# MAGIC     return merged
# MAGIC
# MAGIC
# MAGIC @invoke()
# MAGIC async def invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC     mcp_servers = init_mcp_servers()
# MAGIC     async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
# MAGIC         agent = create_agent(manager.active_servers)
# MAGIC         messages = normalize_input_items(request.input)
# MAGIC         result = await Runner.run(agent, messages)
# MAGIC         return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])
# MAGIC
# MAGIC
# MAGIC @stream()
# MAGIC async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
# MAGIC     mcp_servers = init_mcp_servers()
# MAGIC     async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
# MAGIC         agent = create_agent(manager.active_servers)
# MAGIC         messages = normalize_input_items(request.input)
# MAGIC         result = Runner.run_streamed(agent, input=messages)
# MAGIC
# MAGIC         async for event in process_agent_stream_events(result.stream_events()):
# MAGIC             yield event
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Track requests via Experiemnts
# MAGIC
# MAGIC #### Click -> ```Experiments``` -> ```<username>_databank_ai_lab experiment```
# MAGIC | | |
# MAGIC |-----|------|
# MAGIC |![](./img/Playground_5.jpg) | ![](./img/Playground_6.jpg) |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Module 08 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | Databricks Application Created | Able to query and able to see how Rate Limits, Guardrail works |
# MAGIC | Experiment Tracing | Able to see traces via underlying Experiment |
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Next: Module 09 — Evaluation
# MAGIC Open **`09_evaluation_llm_judge`** to measure agent response quality using LLM-as-a-judge.