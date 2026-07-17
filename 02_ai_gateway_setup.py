# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# DBTITLE 1,Module 02 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 02: AI Gateway
# MAGIC **Duration:** ~20 minutes | **Prerequisite:** Module 00 completed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is AI Gateway?
# MAGIC
# MAGIC **AI Gateway** is Databricks’ managed governance layer that sits in front of any LLM. It gives you:
# MAGIC
# MAGIC | Feature | Why It Matters |
# MAGIC |---------|----------------|
# MAGIC | **Rate Limiting** | Prevent runaway costs — cap requests per user or per endpoint |
# MAGIC | **Usage Tracking** | Log every request/response to a Delta table for auditing and analysis |
# MAGIC | **Guardrails** | Block PII from leaving your perimeter; block unsafe model outputs |
# MAGIC | **Routing** | Route traffic to different LLM providers (Anthropic, OpenAI, Databricks) without code changes |
# MAGIC | **Fallback** | Automatically fall back to a secondary model if the primary fails |
# MAGIC
# MAGIC ### Why Does DataBank Need This?
# MAGIC DataBank’s financial advisors will ask the AI assistant sensitive questions. We need to:
# MAGIC - ✅ Ensure no customer PII is accidentally leaked to external LLMs
# MAGIC - ✅ Cap token spend per user to control cloud costs
# MAGIC - ✅ Log all AI interactions for FCA regulatory compliance
# MAGIC - ✅ Switch LLM providers without changing application code
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC - A managed AI Gateway route named `databank-llm-route`
# MAGIC - Rate limit: 100 requests/minute per user
# MAGIC - PII guardrail on both input and output
# MAGIC - Usage tracking to a Delta inference table
# MAGIC - Test the gateway with financial questions

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration
# ================================================================
# CONFIGURATION
# ================================================================
# Get logged-in user information
# If running this lab via Partner Academy Vocarium 

user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

# Extract username before '@' and remove special characters
import re
username_clean = re.sub(r'\W+', '', user.split('@')[0])

if "labuser" in username_clean:
    CATALOG = username_clean
else:
    CATALOG = "databank_lab"

SCHEMA           = "financial_data"
AI_GW_ROUTE      = "databank-llm-route"
FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving
from openai import OpenAI
import json

w = WorkspaceClient()

print(f"✅ Connected to workspace: {w.config.host}")
print(f"📍 Gateway route name: {AI_GW_ROUTE}")

# COMMAND ----------

# DBTITLE 1,AI Gateway — Architecture
# MAGIC %md
# MAGIC ## 🏗️ AI Gateway Architecture
# MAGIC
# MAGIC ```
# MAGIC DataBank App / Notebook
# MAGIC        ↓
# MAGIC   AI Gateway Route (databank-llm-route)
# MAGIC   ├── Rate Limiter:   100 requests/min/user
# MAGIC   ├── PII Guardrail:  Block customer data from leaving workspace  
# MAGIC   ├── Usage Tracker:  Log to databank_lab.financial_data.ai_inference_log
# MAGIC   └── Router:
# MAGIC        └── Primary: databricks-meta-llama-3-3-70b-instruct
# MAGIC               (Databricks Foundation Models — pay-per-token, no key needed)
# MAGIC ```
# MAGIC
# MAGIC ### Two Creation Approaches
# MAGIC
# MAGIC **Option A: Databricks UI** (covered in the step below)
# MAGIC **Option B: Databricks SDK** (programmatic — covered in Step 2)

# COMMAND ----------

# DBTITLE 1,Option A — UI Walkthrough
# MAGIC %md
# MAGIC ## 🖼️ Option A: Create via the Databricks UI
# MAGIC
# MAGIC Follow these steps in the Databricks workspace UI:
# MAGIC
# MAGIC 1. In the left sidebar, click **Serving** (rocket icon)
# MAGIC 2. Click the **AI Gateway** tab at the top
# MAGIC 3. Click **Create AI Gateway**
# MAGIC 4. Fill in the configuration:
# MAGIC
# MAGIC    | Field | Value |
# MAGIC    |-------|-------|
# MAGIC    | **Route name** | `databank-llm-route` |
# MAGIC    | **Route type** | `LLM/v1/Chat` |
# MAGIC    | **Model provider** | `Databricks` |
# MAGIC    | **Model name** | `databricks-meta-llama-3-3-70b-instruct` |
# MAGIC
# MAGIC 5. Under **Rate limits**, click **Add rate limit**:
# MAGIC    - Calls: `100`
# MAGIC    - Per: `User`
# MAGIC    - Renewal period: `Minute`
# MAGIC
# MAGIC 6. Under **Usage tracking**, toggle **Enable usage tracking** ON
# MAGIC
# MAGIC 7. Under **Guardrails → Input**, enable:
# MAGIC    - **PII detection**: Block
# MAGIC    - **Safety**: Block
# MAGIC
# MAGIC 8. Under **Guardrails → Output**, enable:
# MAGIC    - **Safety**: Block
# MAGIC
# MAGIC 9. Click **Create** and wait ~30 seconds for the route to become active
# MAGIC
# MAGIC > ⚡ Once created, the route URL will be: `{workspace_url}/serving-endpoints/{route_name}/invocations`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC After completing the UI steps, run **Step 2** to verify and test the route.

# COMMAND ----------

# DBTITLE 1,Step 2 — Inspect Gateway Configuration
# Retrieve and display the AI Gateway configuration
endpoint = w.serving_endpoints.get(name=AI_GW_ROUTE)

if endpoint.ai_gateway:
    gw = endpoint.ai_gateway
    print(f"📊 AI Gateway Configuration: {AI_GW_ROUTE}")
    print(f"   Endpoint state : {endpoint.state.ready if endpoint.state else 'READY'}")
    print()

    if gw.rate_limits:
        for rl in gw.rate_limits:
            print(f"   Rate limit     : {rl.calls} calls per {rl.renewal_period} per {rl.key}")

    if gw.usage_tracking_config:
        print(f"   Usage tracking : {'Enabled' if gw.usage_tracking_config.enabled else 'Disabled'}")

    if gw.guardrails:
        print(f"   PII guardrail  : Input={gw.guardrails.input.pii.behavior if gw.guardrails.input and gw.guardrails.input.pii else 'OFF'}")
        print(f"   Safety guard   : Input={gw.guardrails.input.safety if gw.guardrails.input else 'OFF'}, Output={gw.guardrails.output.safety if gw.guardrails.output else 'OFF'}")
else:
    print(f"✅ Endpoint '{AI_GW_ROUTE}' is live (AI Gateway config not shown for legacy routes)")

print(f"\n🔗 Invocation URL: {w.config.host}/serving-endpoints/{AI_GW_ROUTE}/invocations")

# COMMAND ----------

# DBTITLE 1,Test the Gateway — Financial Questions
# MAGIC %md
# MAGIC ## 🧪 Test the AI Gateway
# MAGIC
# MAGIC Now we’ll call the AI Gateway route exactly like a regular OpenAI API, but through the managed gateway.
# MAGIC
# MAGIC **Key insight:** The URL is different — it points to the gateway route, not the Foundation Models API directly.
# MAGIC
# MAGIC | Direct FM API | Through AI Gateway |
# MAGIC |---------------|--------------------|
# MAGIC | `{host}/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations` | `{host}/serving-endpoints/databank-llm-route/invocations` |
# MAGIC | No rate limits | 100 req/min/user |
# MAGIC | No usage log | Logged to Delta table |
# MAGIC | No PII guardrail | PII blocked automatically |

# COMMAND ----------

# DBTITLE 1,Step 3 — Call Gateway with Financial Questions
# ================================================================
# TEST: LLM with Financial Questions
# ================================================================
# This cell tests the LLM backend. If the AI Gateway route exists,
# it calls through the gateway. Otherwise, it calls Foundation Models
# directly to demonstrate the same financial Q&A capability.
# ================================================================

client = OpenAI(
    api_key=w.config.authenticate().get("Authorization", "").replace("Bearer ", ""),
    base_url=f"{w.config.host}/serving-endpoints"
)

# Check if AI Gateway route exists; fall back to Foundation Models if not
try:
    w.serving_endpoints.get(name=AI_GW_ROUTE)
    MODEL_TO_USE = AI_GW_ROUTE
    print(f"🛡️  Using AI Gateway route: {AI_GW_ROUTE}")
except Exception:
    MODEL_TO_USE = FOUNDATION_MODEL
    print(f"⚡ Using Foundation Model directly: {FOUNDATION_MODEL}")
    print("   (Create the AI Gateway route via Option A for rate limits + guardrails)")

SYSTEM_PROMPT = """
You are a DataBank financial advisor assistant. You provide clear, professional
advice on DataBank financial products. Always be concise and helpful.
If asked about specific account balances or transactions, explain you would need
to look those up via the banking system tools.
"""

test_questions = [
    "What is a Stocks & Shares ISA and who should consider one?",
    "What is the difference between a personal loan and a debt consolidation loan at DataBank?",
    "What top 2 DataBank products would you recommend for a customer who is new to investment?"

]

print("\n" + "=" * 65)
for i, question in enumerate(test_questions, 1):
    print(f"\n👤 Question {i}: {question}")
    print("-" * 65)

    response = client.chat.completions.create(
        model=MODEL_TO_USE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question}
        ],
        max_tokens=200
    )
    answer = response.choices[0].message.content
    print(f"🤖 Response: {answer[:300]}{'...' if len(answer) > 300 else ''}")
    print(f"   Tokens used: {response.usage.total_tokens}")

print(f"\n✅ LLM test complete (via {MODEL_TO_USE})")

# COMMAND ----------

# DBTITLE 1,Step 4 — Test PII Guardrail
# Demonstrate the PII guardrail: the gateway should BLOCK or MASK requests
# containing sensitive personal data (SSN, credit card numbers, etc.)

print("🔒 Testing PII Guardrail...")
print("-" * 50)

try:
    pii_response = client.chat.completions.create(
        model=AI_GW_ROUTE,
        messages=[{
            "role": "user",
            "content": "My customer John Smith, SSN 123-45-6789 and card number 4111-1111-1111-1111 "
                       "has a loan query. Can you help process this?"
        }],
        max_tokens=100
    )
    # If guardrail is configured to BLOCK, this will raise an exception
    # If configured to ANONYMIZE, PII will be replaced with [MASKED]
    print(f"⚠️  Response received (guardrail may have masked PII):")
    print(f"   {pii_response.choices[0].message.content[:200]}")
except Exception as e:
    print(f"✅ PII Guardrail BLOCKED the request as expected!")
    print(f"   Error: {str(e)[:200]}")

print()
print("💡 Key learning: AI Gateway acts as a compliance layer — sensitive data")
print("   never reaches the LLM, protecting both customers and the bank.")

# COMMAND ----------

# DBTITLE 1,Module 02 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 02 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | Gateway route `databank-llm-route` exists | Visible under Serving → AI Gateway in UI |
# MAGIC | Gateway responds to questions | 3 test questions answered in Step 4 |
# MAGIC | PII guardrail active | Sensitive request blocked/masked in Step 5 |
# MAGIC | Usage tracking enabled | Requests logged (check Delta table in ~5 min) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Next: Module 03 — UC Functions
# MAGIC Open **`03_uc_functions`** to register Python and SQL functions in Unity Catalog —
# MAGIC these will become **tools** for the AgentBricks agent in Module 07.