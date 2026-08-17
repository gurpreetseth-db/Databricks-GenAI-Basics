# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
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

# DBTITLE 1,Run Pre-requisites
# MAGIC %run ./00_setup_prerequisites

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration
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
# MAGIC **Option B: Databricks SDK** (not covered here :) )

# COMMAND ----------

# DBTITLE 1,UI Walkthrough
# MAGIC %md
# MAGIC ## 🖼️ Create via the Databricks UI
# MAGIC
# MAGIC Follow these steps in the Databricks workspace UI:
# MAGIC
# MAGIC 1. In the left sidebar, click **AI/ML** (rocket icon)
# MAGIC 2. Click the **AI Gateway -> Model**
# MAGIC 3. Click ** Foundation Model**  and use below configurations
# MAGIC 4. Fill in the configuration and hit **Create**:
# MAGIC
# MAGIC    | Field | Value |
# MAGIC    |-------|-------|
# MAGIC    | **Route name** | `databank-llm-route` |
# MAGIC    | **Route type** | `LLM/v1/Chat` |
# MAGIC    | **Model provider** | `Foundation Model` |
# MAGIC    | **Model name** | `Gemma 3-12B` |
# MAGIC
# MAGIC 5. Expand **Governance Setup**
# MAGIC 6. Under **Rate limits**, click **Add rate limit**:
# MAGIC    - Per Endpoint/User: `Per User`
# MAGIC    - Requests : `100`
# MAGIC    - Per: `hour`
# MAGIC 7. Hit **Save**
# MAGIC 8. Under **Policies**
# MAGIC 9. Click **New Policy** use below configurations to create 2 policies:
# MAGIC    - Name : `Test_PII`
# MAGIC    - Guardrail : `PII Blocking`
# MAGIC    - Phase : `Input & Output`
# MAGIC    - Advance Options - Mode: `Enforce`
# MAGIC    
# MAGIC    - Name: `Test_Unsafe`
# MAGIC    - Guardrail : `Unsafe Content`
# MAGIC    - Phase : `Input & Output`
# MAGIC    - Advance Options - Evaluation Model :  `system-ai.gpt-5-2`
# MAGIC    - Advance Options - Mode: `Enforce`
# MAGIC
# MAGIC 10. Click **Create** and wait ~30 seconds for the route to become active
# MAGIC
# MAGIC > ⚡ Once created, the route URL will be: `{workspace_url}/serving-endpoints/{route_name}/invocations`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC After completing the UI steps, run **Step 2** to verify and test the route.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC **Generate a Token** 
# MAGIC
# MAGIC To Query AI Gateway create a Token and make sure to select scope as **`ai_gateway`** and Replace it in **cell 11** for column **`Key`**
# MAGIC
# MAGIC ---
# MAGIC ![](./img/Generate_Token.jpg)
# MAGIC
# MAGIC ---
# MAGIC ---
# MAGIC ![](./img/Generate_Token_Gateway_Specific.jpg)

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

# DBTITLE 1,Step 2 — Call Gateway with Financial Questions
# ================================================================
# TEST: LLM with Financial Questions
# ================================================================
# This cell tests the LLM backend. If the AI Gateway route exists,
# it calls through the gateway. Otherwise, it calls Foundation Models
# directly to demonstrate the same financial Q&A capability.
# ================================================================
import os
from openai import OpenAI

LLM_Key = "REDACTED_DATABRICKS_PAT"

client = OpenAI(
    api_key=LLM_Key,
    base_url=f"{w.config.host}/ai-gateway/mlflow/v1"
)

# Try the AI Gateway route; fall back to Foundation Model if gateway proxy fails
try:
    _test = client.chat.completions.create(
        model=f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    MODEL_TO_USE = f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}"
    print(f"⚡ Using AI Gateway route: {CATALOG}.{SCHEMA}.{AI_GW_ROUTE}")
    print(f"   Backing model: {FOUNDATION_MODEL}")
except Exception as e:
    # Use llama-3-3 directly
    MODEL_TO_USE = "llama_v3_3_70b_instruct"
    print(f"⚡ AI Gateway proxy unavailable — calling Foundation Model directly: {MODEL_TO_USE}")
    print(f"   Error: {type(e).__name__}: {str(e)[:150]}")
    print(f"   Fix: Re-run Cell 6 (Option 2 - Via Code) to recreate with valid PAT")

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
    if not answer:
        answer = response.choices[0].message.model_dump().get("reasoning_content", "")
    print(f"🤖 Response: {answer[:500]}{'...' if len(answer) > 500 else ''}")
    print(" ")
    print(f"   Tokens used: {response.usage.total_tokens}")

print(f"\n✅ LLM test complete (via {MODEL_TO_USE})")


# COMMAND ----------

# DBTITLE 1,Step 3 — Test PII Guardrail
# Demonstrate the PII guardrail: the gateway should BLOCK or MASK requests
# containing sensitive personal data (SSN, credit card numbers, etc.)

print("🔒 Testing PII Guardrail...")
print("-" * 50)

try:
    pii_response = client.chat.completions.create(
        model=f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}",
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

# MAGIC %md
# MAGIC ## 🖼️ Update Rate Limit for AI Gateway
# MAGIC
# MAGIC Follow these steps in the Databricks workspace UI:
# MAGIC
# MAGIC 1. Under **Rate limits**, add **Rate Limits**:
# MAGIC    - Per Endpoint/User: `Per User`
# MAGIC    - Requests : `100`
# MAGIC    - Per: `hour`
# MAGIC
# MAGIC
# MAGIC ![](./img/Rate_Limit.jpg)
# MAGIC > ⚡ Once created, the route URL will be: `{workspace_url}/serving-endpoints/{route_name}/invocations`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC After completing the UI steps, run **Step 2** to verify and test the route.

# COMMAND ----------

# DBTITLE 1,Step 4 — Check Unsafe Guardral
# Demonstrate the Unsafe Content guardrail: the gateway should BLOCK o requests


print("🔒 Testing Unsafe Guardrails...")
print("-" * 50)

try:
    pii_response = client.chat.completions.create(
        model=f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}",
        messages=[{
            "role": "user",
            "content": "A Customer has written some negative feedback on Databank's Google Page."
                       "Can you send them a response and also tell that we can sue them for this."
        }],
        max_tokens=100
    )
    # If guardrail is configured to BLOCK, this will raise an exception
    # If configured to ANONYMIZE, PII will be replaced with [MASKED]
    print(f"⚠️  Response received (guardrail may have Unsafe Content):")
    print(f"   {pii_response.choices[0].message.content[:200]}")
except Exception as e:
    print(f"✅ Unsafe Guardrail BLOCKED the request as expected!")
    print(f"   Error: {str(e)[:200]}")

print()
print("💡 Key learning: AI Gateway acts as a compliance layer — Unsafe/Obsence data")
print("   never reaches the LLM, protecting both customers and the bank.")

# COMMAND ----------

# DBTITLE 1,Step 5 - Test Rate Limits
# Demonstrate the Unsafe Content guardrail: the gateway should BLOCK o requests


print("🔒 Testing Rate Limits...")
print("-" * 50)

try:
    pii_response = client.chat.completions.create(
        model=f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}",
        messages=[{
            "role": "user",
            "content": "What is a Stocks & Shares ISA and who should consider one?"
        }],
        max_tokens=100
    )
    # If guardrail is configured to BLOCK, this will raise an exception
    # If configured to ANONYMIZE, PII will be replaced with [MASKED]
    print(f"⚠️  Response received (rate limit may have reached):")
    print(f"   {pii_response.choices[0].message.content[:200]}")
except Exception as e:
    print(f"✅ Rate Limit Exceeds the Hourly Token Limit as expected!")
    print(f"   Error: {str(e)[:200]}")

print()
print("💡 Key learning: Rate Limit acts as a Cost Optimization layer — Save Token Cost")
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

# COMMAND ----------

# DBTITLE 1,Option 2 - UC Gateway Via Code - Gateway V1
#from databricks.sdk import WorkspaceClient
#from databricks.sdk.service import serving
#from openai import OpenAI
#import json

#w = WorkspaceClient()

## Create the AI Gateway endpoint programmatically
## This creates a pay-per-token AI Gateway route for a Foundation Model
## with rate limiting and usage tracking.

#from databricks.sdk.service.serving import (
#    EndpointCoreConfigInput,
#    ServedEntityInput,
#    ExternalModel,
#    ExternalModelProvider,
#    DatabricksModelServingConfig,
#    AiGatewayConfig,
#    AiGatewayRateLimit,
#    AiGatewayRateLimitRenewalPeriod,
#    AiGatewayRateLimitKey,
#    AiGatewayUsageTrackingConfig,
#)

#from databricks.sdk import WorkspaceClient
#w = WorkspaceClient()

#print(f"🚀 Creating AI Gateway route: '{AI_GW_ROUTE}'...")
#print(f"   Foundation Model: {FOUNDATION_MODEL}")

## Get a long-lived PAT for the endpoint proxy
#pat_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

#try:
#    existing = w.serving_endpoints.get(name=AI_GW_ROUTE)
    ## Verify the proxy actually works (token may be missing/expired)
#    from openai import OpenAI as _OAI
#    _client = _OAI(api_key=pat_token, base_url=f"{w.config.host}/serving-endpoints")
#    _test = _client.chat.completions.create(model=AI_GW_ROUTE, messages=[{"role": "user", "content": "test"}], max_tokens=5)
#    print(f"✅ AI Gateway '{AI_GW_ROUTE}' exists and proxy is working!")
#    if existing.ai_gateway:
#        print(f"   Rate limits: {len(existing.ai_gateway.rate_limits or [])} configured")
#        print(f"   Usage tracking: {existing.ai_gateway.usage_tracking_config.enabled if existing.ai_gateway.usage_tracking_config else False}")
#    print(f"🔗 URL: {w.config.host}/serving-endpoints/{AI_GW_ROUTE}/invocations")
#except Exception as e:
    ## Either endpoint doesn't exist or proxy is broken — (re)create with valid PAT
#    print(f"⚠️  Gateway needs (re)creation: {type(e).__name__}")
#    try:
#        w.serving_endpoints.delete(name=AI_GW_ROUTE)
#        import time; time.sleep(5)
#        print("   Deleted broken endpoint.")
#    except Exception:
#        pass

#    endpoint = w.serving_endpoints.create(
#        name=AI_GW_ROUTE,
#        config=EndpointCoreConfigInput(
#            served_entities=[
#                ServedEntityInput(
#                    external_model=ExternalModel(
#                        provider=ExternalModelProvider.DATABRICKS_MODEL_SERVING,
#                        name=FOUNDATION_MODEL,
#                        task="llm/v1/chat",
#                        databricks_model_serving_config=DatabricksModelServingConfig(
#                            databricks_workspace_url=w.config.host,
#                            databricks_api_token_plaintext=pat_token,
#                        ),
#                    ),
#                )
#            ]
#        ),
#        ai_gateway=AiGatewayConfig(
#            rate_limits=[
#                AiGatewayRateLimit(
#                    calls=100,
#                    renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
#                    key=AiGatewayRateLimitKey.USER,
#                )
#            ],
#            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
#        ),
#    )

#    print(f"✅ AI Gateway '{AI_GW_ROUTE}' created successfully!")
#    print(f"   Rate limit: 100 requests/min/user")
#    print(f"   Usage tracking: Enabled")
#    print(f"   Guardrails: Add via UI (AI/ML → AI Gateway → Policies)")
#    print(f"🔗 URL: {w.config.host}/serving-endpoints/{AI_GW_ROUTE}/invocations")