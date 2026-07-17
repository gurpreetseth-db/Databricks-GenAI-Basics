# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# DBTITLE 1,Module 07 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 07: AgentBricks Supervisor Agent
# MAGIC **Duration:** ~40 minutes | **Prerequisites:** Modules 01, 03, 04, 05 all completed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is AgentBricks?
# MAGIC
# MAGIC **AgentBricks** are Databricks’ pre-built AI tiles that you assemble like Lego blocks:
# MAGIC
# MAGIC | Brick Type | What It Does | DataBank Use |
# MAGIC |------------|--------------|---------------|
# MAGIC | **Knowledge Assistant (KA)** | RAG over documents (PDFs, text) | Search product brochures & T&Cs |
# MAGIC | **Genie Space** | Natural language → SQL on Delta tables | Query transactions & customers |
# MAGIC | **UC Function** | Call any registered Python/SQL function | Risk score, portfolio, fraud check |
# MAGIC | **Supervisor Agent (MAS)** | Routes user queries to the right tool | DataBank AI Advisor (the final agent) |
# MAGIC
# MAGIC ### DataBank AI Advisor Architecture
# MAGIC
# MAGIC ```
# MAGIC                    User: "What products suit CUST-0042 given their risk score?"
# MAGIC                               ↓
# MAGIC               DataBank AI Advisor (Supervisor Agent)
# MAGIC               ├── Tool 1: Knowledge Assistant  ← "£85,000 FSCS protection... Fixed-Rate Bond..."
# MAGIC               ├── Tool 2: Genie Space          ← "Customer has 3 accounts, balance £45,231..."
# MAGIC               ├── Tool 3: UC Function (risk)   ← "Risk score: 62 (Moderate)"
# MAGIC               └── Tool 4: UC Function (fraud)  ← "No suspicious transactions found."
# MAGIC                               ↓
# MAGIC               Agent synthesises all tool outputs into a final answer
# MAGIC ```
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC 1. **Knowledge Assistant** — RAG over the 7 financial PDFs in the volume
# MAGIC 2. **Supervisor Agent** — routing to KA + Genie + UC Functions
# MAGIC 3. **Test 3 financial scenarios**: portfolio review, product recommendation, fraud check
# MAGIC 4. **Deploy as a Model Serving endpoint** (used by Module 08 app)

# COMMAND ----------

# DBTITLE 1,Step 0 — Knowledge Assistant Configuration
# ================================================================
# CONFIGURATION — FILL IN YOUR IDs FROM EARLIER MODULES
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
    
SCHEMA        = "financial_data"
VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/documents"

# From Module 05: your Genie Space ID (from URL: #genie/<SPACE_ID>)
GENIE_SPACE_ID = ""  # <-- FILL THIS IN

# Knowledge Agent configuration
KA_NAME       = "DataBank-Document-Assistant"
DESCRIPTION   = ("DataBank product and compliance document assistant. Answers questions about savings, loans, " 
                   "investments insurance, credit cards, FAQs and term & conditions."
                )
FILE_NAME     = "PDF"
SOURCE_DESCRIPTION   = "Product brochures (Savings, Loans, Investment, Insurance, Credit Cards), FAQ, Terms & Conditions"
INSTRUCTIONS  = (
    "You are DataBank's document assistant. Answer questions based only on the provided "
    "product brochures, FAQs, and terms & conditions. "
    "Be accurate, concise, and professional. "
    "If the answer is not in the documents, clearly say so."
)



from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

if not GENIE_SPACE_ID:
    print("⚠️  GENIE_SPACE_ID is empty.")
    print("   Complete Module 05 first, then copy your Genie Space ID into GENIE_SPACE_ID above.")
    print("   You can continue to Step 1 while you look it up.")
else:
    print(f"✅ Genie Space ID : {GENIE_SPACE_ID}")
print("                                   ")
print("                                   ")
print("                                   ")
print(" ================================= ")
print(" KNOWLEDGE ASSISTANT CONFIGURATION ")
print(" ================================= ")
print(f"🤖 KA Name            : {KA_NAME}")
print(f"🤖 KA Descriptiom     : {DESCRIPTION}")
print(f"🤖 Instructions       : {INSTRUCTIONS}")
print(f"🤖 Source File Name   : {FILE_NAME}")
print(f"🤖 Source Description : {SOURCE_DESCRIPTION}")


# COMMAND ----------

# DBTITLE 1,Step 1 — Create Knowledge Assistant via UI
# MAGIC %md
# MAGIC ## 📚 Step 1: Create Knowledge Assistant (Document RAG)
# MAGIC
# MAGIC A **Knowledge Assistant (KA)** is an AgentBrick that:
# MAGIC 1. Reads all documents from a Unity Catalog Volume
# MAGIC 2. Automatically chunks, embeds, and indexes them
# MAGIC 3. Deploys a RAG endpoint
# MAGIC 4. Can answer questions about the document contents
# MAGIC
# MAGIC We point it at our Volume (`/Volumes/databank_lab/financial_data/documents/`) which contains:
# MAGIC - Product brochures (Savings, Loans, Investment, Insurance, Credit Cards)
# MAGIC - FAQ
# MAGIC - Terms & Conditions
# MAGIC
# MAGIC > 🕑 KA provisioning takes **2-5 minutes**. Run the cell and move on to Step 2 while it provisions.
# MAGIC
# MAGIC ------------------------------------------------------------------------------------------------------------------------------
# MAGIC
# MAGIC ![](./img/KA-0.jpg)
# MAGIC ![](./img/KA-1.jpg)

# COMMAND ----------

# DBTITLE 1,Step 2 - Supervisor Agent Configurations
print(" ================================== ")
print(" SUPERVISOR ASSISTANT CONFIGURATION ")
print(" ================================== ")

SUPERAGENT_NAME      = "DataBank-AI-Advisor"
SUPERAGENT_INSTRUCTIONS = (
 
    "- Use Knowledge Assistant to "
      "1. Search DataBank product brochures, FAQs, and terms & conditions, product knowledge and document questions"
      "2. Use for questions about product features, interest rates, fees, eligibility criteria, FSCS protection, or regulatory terms."
    "- Use Genie Spaces to "
     "1. Query customer records, account balances, transaction history, and support tickets using natural language SQL."
     "2. Use for any questions about specific customer account data, transaction queries,  portfolio values, spending patterns, or open support tickets."

    "- Use calculate_customer_risk to "
    "Calculate a numeric risk score (0-100) for a DataBank customer. Use when asked about a customer's investment risk tolerance, risk appetite, risk profile calculation, or suitability for investment products."

    "- Use get_portfolio_summary to "
     "Retrieve a complete portfolio summary for a customer including all accounts, product types, current balances, and total assets. Use for portfolio reviews, portfolio overviews, financial health checks, or account overviews, account balances,"

    "- Use fraud_and_anomaly_check to "
     " Check for suspicious, fraudulent, or anomalous transactions for a customer. Use when asked about fraud, unusual activity, blocked transactions, or security concerns on an account."

    "Always:"
     "1. If a customer ID is mentioned, call get_portfolio_summary and calculate_risk_score first."
     "2. For product recommendations, consider the customer’s risk score before recommending."
     "3. For investment products, always include: Capital at risk - investments can fall as well as rise."
     "4. Be concise and professional. Financial advisors are busy."

)



FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
print(f"🤖 Supervisor Name    : {SUPERAGENT_NAME}")
print(f"🤖 Supervisor Instruction:{SUPERAGENT_INSTRUCTIONS}" )


# COMMAND ----------

# DBTITLE 1,Step 2 — Create Supervisor Agent
# MAGIC %md
# MAGIC ## 🤖 Step 2: Assemble the Supervisor Agent
# MAGIC
# MAGIC The **Supervisor Agent (MAS)** is the orchestration layer. It:
# MAGIC 1. Receives the user’s question
# MAGIC 2. Decides which tool(s) to call based on the question and tool descriptions
# MAGIC 3. Calls the tools in sequence or parallel
# MAGIC 4. Synthesises a final natural language answer
# MAGIC
# MAGIC **Tool routing logic** (configured via descriptions):
# MAGIC - Questions about documents, products, terms, rates → **Knowledge Assistant**
# MAGIC ![](./img/SA_KA_2.jpg)
# MAGIC
# MAGIC
# MAGIC - Questions about customer data, balances, transactions, queries → **Genie Space**
# MAGIC ![](./img/SA_Genie_Space_1.jpg)
# MAGIC
# MAGIC
# MAGIC - Questions requiring risk score calculation → **UC Function: calculate_customer_risk**
# MAGIC - Questions about portfolio composition → **UC Function: get_portfolio_summary**
# MAGIC - Questions about suspicious/fraudulent activity → **UC Function: flag_suspicious_transactions**
# MAGIC ![](./img/SA_UC_Function_3.jpg)
# MAGIC
# MAGIC ===============================================================================================
# MAGIC
# MAGIC ![](./img/SA_Final_4.jpg)
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Step 2 — Test Supervise Agent - Is it running
from databricks.sdk import WorkspaceClient
import time

def wait_for_endpoint(endpoint_name: str, timeout_seconds: int = 300) -> bool:
    """Poll the serving endpoint until it is ready or timeout is reached."""
    print(f"⏳ Waiting for endpoint '{endpoint_name}' to be ready...")
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            ep = w.serving_endpoints.get(name=endpoint_name)
            state = ep.state.ready if ep.state else None
            print(f"   Status: {state}", end="\r")
            if state is not None and str(state).split(".")[-1] == "READY":
                print(f"\n✅ Endpoint '{endpoint_name}' is READY")
                return endpoint_name
        except Exception as e:
            print(f"   Waiting... ({e})\r")
        time.sleep(15)
    print(f"\n❌ Timeout: endpoint '{endpoint_name}' not ready after {timeout_seconds}s")
    return False


# -------------------------------------------------------
# Set your Supervisor Agent's human-readable name here
# -------------------------------------------------------

def get_superagent_endpoint_name(superagent_name):
    _w = WorkspaceClient()
    normalise = lambda s: s.lower().replace("-", "").replace("_", "").replace(" ", "")
    all_eps   = list(_w.serving_endpoints.list())
    
    # 1. Exact name match
    endpoint_name = next((ep.name for ep in all_eps if ep.name == SUPERAGENT_NAME), None)
    
    # 2. Fuzzy match (normalised agent name appears inside endpoint name
    if not endpoint_name:
        term = normalise(SUPERAGENT_NAME)
        endpoint_name = next((ep.name for ep in all_eps if term in normalise(ep.name)), None)
    
    # 3. AgentBricks fallback — supervisor agent endpoints are always named mas-<uuid>-endpoint
    if not endpoint_name:
        mas_eps = [ep.name for ep in all_eps
            if ep.name.startswith("mas-") and ep.name.endswith("-endpoint")]
    if len(mas_eps) == 1:
        endpoint_name = mas_eps[0]
    elif len(mas_eps) > 1:
        print(f"⚠️  Multiple AgentBricks supervisor endpoints found.")
        print(f"   Update SUPERAGENT_NAME in cell 5 to one of:")
        for n in mas_eps:
            print(f"   {n}")

    if endpoint_name:
        ENDPOINTNAME = endpoint_name          # available for downstream cells
        print(f"Supervisor Agent Name : {SUPERAGENT_NAME}")
        print(f"Endpoint Name         : {endpoint_name}")
    else:
        print(f"⚠️  Could not resolve endpoint for '{SUPERAGENT_NAME}'")

get_superagent_endpoint_name(SUPERAGENT_NAME)

# Use the endpoint name from the MAS creation response, or the configured name
active_endpoint = endpoint_name
if active_endpoint:
    ENDPOINTNAME = wait_for_endpoint(active_endpoint)
else:
    print("⚠️  Endpoint name not set. Check SUPERAGENT_NAME or set ENDPOINTNAME manually.")
    print("   You can proceed and test once you have the endpoint name.")


# COMMAND ----------

# DBTITLE 1,Step 3 — Test Financial Scenarios
# MAGIC %md
# MAGIC ## 🧪 Step 3: Test 3 Financial Scenarios
# MAGIC
# MAGIC Now we test the agent with 3 real-world advisor scenarios:
# MAGIC
# MAGIC | Scenario | What It Tests |
# MAGIC |----------|---------------|
# MAGIC | **Portfolio Review** | get_portfolio_summary + calculate_customer_risk |
# MAGIC | **Product Recommendation** | calculate_customer_risk + document_search |
# MAGIC | **Fraud Detection** | flag_suspicious_transactions + customer_data_query |

# COMMAND ----------

# DBTITLE 1,Step 3 — Test Agent Scenarios
import openai

# Connect to the deployed Supervisor Agent endpoint
import os
os.environ['OPENAI_API_KEY'] = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")

agent_client = openai.OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=f"{w.config.host}/serving-endpoints"
)

# Get a sample customer for realistic testing
try:
    sample_cust = spark.sql(
        f"SELECT customer_id, full_name, risk_profile FROM {CATALOG}.{SCHEMA}.customers "
        f"WHERE is_active = true LIMIT 1"
    ).collect()[0]
    TEST_CUSTOMER_ID = sample_cust.customer_id
    TEST_CUSTOMER_NAME = sample_cust.full_name
except Exception:
    TEST_CUSTOMER_ID   = "CUST-0001"
    TEST_CUSTOMER_NAME = "Test Customer"

print(f"👤 Test customer: {TEST_CUSTOMER_NAME} ({TEST_CUSTOMER_ID})")
print()

# Define the 3 test scenarios
scenarios = [
    {
        "name": "Portfolio Review",
        "query": f"I have a meeting with customer {TEST_CUSTOMER_ID} in 10 minutes. "
                 f"Can you give me a full portfolio overview and their risk score?"
    },
    {
        "name": "Product Recommendation",
        "query": f"Customer {TEST_CUSTOMER_ID} has £15,000 to invest. Based on their risk profile, "
                 f"what DataBank products would you recommend?"
    },
    {
        "name": "Fraud Detection",
        "query": f"Customer {TEST_CUSTOMER_ID} called in worried about unusual activity on their account. "
                 f"Please check for any suspicious transactions in the last 90 days."
    }
]

for scenario in scenarios:
    print(f"\n{'='*65}")
    print(f"  Scenario: {scenario['name']}")
    print(f"{'='*65}")
    print(f"  Question: {scenario['query']}")
    print("-" * 65)

    try:
        resp = requests.post(
            f"{w.config.host}/serving-endpoints/{ENDPOINTNAME}/invocations",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
            json={"input": [{"role": "user", "content": scenario["query"]}]}
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract the final assistant text from the AgentBricks event stream.
        # data["output"] is a list of events (tool calls, outputs, messages).
        # The final answer is the last 'message' event with role 'assistant' + type 'output_text'.
        answer = None
        output_events = data.get("output", [])
        if isinstance(output_events, list):
            for event in reversed(output_events):
                if event.get("type") == "message" and event.get("role") == "assistant":
                    for block in event.get("content", []):
                        if block.get("type") == "output_text":
                            answer = block.get("text", "").strip()
                            break
                if answer:
                    break
        # Fallback to standard OpenAI format if not an event stream
        if not answer:
            answer = (data.get("choices", [{}])[0].get("message", {}).get("content") or str(data))

        print(f"  Agent: {answer}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"  Make sure the endpoint '{ENDPOINTNAME}' is ready (Step 3).")

print(f"\n{'='*65}")

# COMMAND ----------

# DBTITLE 1,Module 07 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 07 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | Knowledge Assistant created | Visible in AgentBricks sidebar |
# MAGIC | Supervisor Agent created | Endpoint `databank-ai-advisor` in Serving |
# MAGIC | Portfolio review works | Agent returns accounts + risk score |
# MAGIC | Product recommendation works | Agent recommends products matching risk profile |
# MAGIC | Fraud check works | Agent returns suspicious transaction report or clean bill |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📝 Record Your Endpoint Name
# MAGIC ```
# MAGIC Agent Endpoint: ________________________
# MAGIC (used in Module 08 and Module 09)
# MAGIC ```
# MAGIC
# MAGIC ### 🚀 Next: Module 08 — Databricks App
# MAGIC Open **`08_databricks_app`** to deploy the agent as a live Gradio chat application.