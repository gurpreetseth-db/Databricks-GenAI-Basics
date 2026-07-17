# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# DBTITLE 1,Module 05 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 05: Genie Space
# MAGIC **Duration:** ~15 minutes | **Prerequisite:** Module 01 (data tables)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is a Genie Space?
# MAGIC
# MAGIC **Genie** is Databricks’ natural-language-to-SQL interface. A Genie Space is a configured workspace that:
# MAGIC - Connects to specific Unity Catalog tables
# MAGIC - Understands your business domain (via instructions and certified queries)
# MAGIC - Lets anyone ask questions in plain English without writing SQL
# MAGIC
# MAGIC ### DataBank Use Case
# MAGIC Financial advisors at DataBank are not SQL experts. They need to ask:
# MAGIC - *"How many customers have an Aggressive risk profile?"*
# MAGIC - *"Which customers have had more than 5 fraudulent transactions in the last 30 days?"*
# MAGIC - *"What is the total balance across all Premium Savings accounts?"*
# MAGIC
# MAGIC A Genie Space backed by our Delta tables gives them a safe, governed interface.
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC A Genie Space named **DataBank Financial Advisor** with:
# MAGIC - 5 connected tables: `customers`, `accounts`, `transactions`, `products`, `support_tickets`
# MAGIC - Business context instructions (written in plain English)
# MAGIC - 5 certified sample queries that demonstrate what the space can do
# MAGIC
# MAGIC ### Two Approaches
# MAGIC This module covers **both** creation methods:
# MAGIC - **Option A**: UI walkthrough (recommended for first time)
# MAGIC - **Option B**: Genie API (scripted, commented out by default)

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
    
SCHEMA       = "financial_data"
GENIE_NAME   = "DataBank Financial Advisor"

TABLES = [
    f"{CATALOG}.{SCHEMA}.customers",
    f"{CATALOG}.{SCHEMA}.accounts",
    f"{CATALOG}.{SCHEMA}.transactions",
    f"{CATALOG}.{SCHEMA}.products",
    f"{CATALOG}.{SCHEMA}.support_tickets",
]

print(f"🧠 Genie Space: {GENIE_NAME}")
print(f"📊 Tables to connect:")
for t in TABLES:
    count = spark.table(t).count()
    print(f"   ✓ {t} ({count:,} rows)")

# COMMAND ----------

# DBTITLE 1,Option A — UI Walkthrough
# MAGIC %md
# MAGIC ## 🖼️ Option A: Create Genie Space via the UI
# MAGIC
# MAGIC Follow these steps exactly:
# MAGIC
# MAGIC ### 1. Navigate to Genie
# MAGIC - In the left sidebar, click **Genie** (sparkle/wand icon)
# MAGIC - Click **Create Genie space** (top right button)
# MAGIC
# MAGIC ### 2. Connect Tables
# MAGIC In the table selector panel, add all 5 tables:
# MAGIC - `databank_lab.financial_data.customers`
# MAGIC - `databank_lab.financial_data.accounts`
# MAGIC - `databank_lab.financial_data.transactions`
# MAGIC - `databank_lab.financial_data.products`
# MAGIC - `databank_lab.financial_data.support_tickets`
# MAGIC
# MAGIC ### 3. Set the Space Name
# MAGIC | Field | Value |
# MAGIC |-------|-------|
# MAGIC | **Name** | `DataBank Financial Advisor` |
# MAGIC | **Description** | `AI-powered SQL assistant for DataBank financial advisors` |
# MAGIC
# MAGIC ### 4. Add Instructions
# MAGIC In the **Instructions** field, paste the text from **Step 1** (run the cell, then copy the output).
# MAGIC
# MAGIC ### 5. Add Sample Questions (Certified Queries)
# MAGIC In the **Sample Questions** panel, add each query from **Step 2** (run the cell to see them).
# MAGIC
# MAGIC ### 6. Click Save
# MAGIC The space will be available immediately for queries.

# COMMAND ----------

# DBTITLE 1,Step 1 — Generate Instructions Text
# Run this cell to generate the Instructions text
# Copy the output and paste it into the Genie Space Instructions field

instructions = """
You are a financial advisor assistant for DataBank. You have access to the following tables:

- customers: 500 DataBank customers with their risk profile (Conservative/Moderate/Aggressive),
  annual income, age, and membership date.
- accounts: Customer account holdings linking each customer to a DataBank product.
  Includes current balance and account status (Active/Dormant/Closed).
- transactions: 10,000 banking transactions with amount, merchant, category, date,
  fraud flag (is_fraud), and status. Covers the last 180 days.
- products: The DataBank product catalogue — 25 products across Savings, Loan,
  Investment, Insurance, and CreditCard categories.
- support_tickets: 300 customer support interactions with subject, priority,
  ticket_status, and resolution.

Key business rules:
- Risk profiles: Conservative prefers Savings and Insurance, Aggressive prefers Investment products.
- Fraud flag: is_fraud = true means the transaction was flagged as suspicious.
- Account status: Active = normal, Dormant = no recent activity, Closed = account terminated.
- Customer IDs follow the pattern CUST-XXXX (e.g. CUST-0001).
- Amounts are in British Pounds (GBP £).
- Always use proper number formatting (£ symbol, 2 decimal places for amounts).

When asked about a specific customer, always look up their risk profile before recommending products.
For fraud queries, filter WHERE is_fraud = true.
"""

print(instructions)
print("\n" + "-"*60)
print("COPY THE TEXT ABOVE INTO THE GENIE SPACE INSTRUCTIONS FIELD")

# COMMAND ----------

# DBTITLE 1,Step 2 — Generate Sample Certified Queries
# These are the 5 sample questions to add to the Genie Space.
# Genie uses these as examples to understand what queries to generate.

queries = [
    {
        "question": "How many customers have each risk profile?",
        "sql": f"""SELECT risk_profile, COUNT(*) AS customer_count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM {CATALOG}.{SCHEMA}.customers
GROUP BY risk_profile
ORDER BY customer_count DESC"""
    },
    {
        "question": "What are the top 10 customers by total transaction volume in the last 90 days?",
        "sql": f"""SELECT c.full_name, c.customer_id, c.risk_profile,
       COUNT(t.txn_id) AS num_transactions,
       ROUND(SUM(t.amount_gbp), 2) AS total_spend_gbp
FROM {CATALOG}.{SCHEMA}.transactions t
JOIN {CATALOG}.{SCHEMA}.customers c USING (customer_id)
WHERE t.txn_date >= DATE_SUB(CURRENT_DATE(), 90)
GROUP BY c.full_name, c.customer_id, c.risk_profile
ORDER BY total_spend_gbp DESC
LIMIT 10"""
    },
    {
        "question": "Show me all customers with suspicious transactions in the last 30 days",
        "sql": f"""SELECT c.full_name, c.customer_id,
       COUNT(t.txn_id) AS fraud_count,
       ROUND(SUM(t.amount_gbp), 2) AS total_suspicious_gbp
FROM {CATALOG}.{SCHEMA}.transactions t
JOIN {CATALOG}.{SCHEMA}.customers c USING (customer_id)
WHERE t.is_fraud = true
  AND t.txn_date >= DATE_SUB(CURRENT_DATE(), 30)
GROUP BY c.full_name, c.customer_id
ORDER BY fraud_count DESC"""
    },
    {
        "question": "What is the total balance by product type across all active accounts?",
        "sql": f"""SELECT p.product_type,
       COUNT(a.account_id) AS account_count,
       ROUND(SUM(a.balance_gbp), 2) AS total_balance_gbp,
       ROUND(AVG(a.balance_gbp), 2) AS avg_balance_gbp
FROM {CATALOG}.{SCHEMA}.accounts a
JOIN {CATALOG}.{SCHEMA}.products p USING (product_id)
WHERE a.status = 'Active'
GROUP BY p.product_type
ORDER BY total_balance_gbp DESC"""
    },
    {
        "question": "Which support tickets are high priority and still open?",
        "sql": f"""SELECT t.ticket_id, c.full_name, c.customer_id,
       t.subject, t.priority, t.ticket_status, t.created_date,
       DATEDIFF(CURRENT_DATE(), t.created_date) AS days_open
FROM {CATALOG}.{SCHEMA}.support_tickets t
JOIN {CATALOG}.{SCHEMA}.customers c USING (customer_id)
WHERE t.priority = 'High'
  AND t.ticket_status IN ('Open', 'In Progress')
ORDER BY t.created_date ASC"""
    }
]

print("📝 Sample Questions and Certified SQL Queries:")
print("=" * 65)
for i, q in enumerate(queries, 1):
    print(f"\n  Question {i}: {q['question']}")
    print(f"  SQL:")
    for line in q['sql'].split('\n'):
        print(f"    {line}")

print("\n" + "-"*65)
print("ADD EACH QUESTION ABOVE AS A SAMPLE QUESTION IN THE GENIE SPACE")
print("For best results, also add the SQL as the 'Certified Query' for each question.")

# COMMAND ----------

# DBTITLE 1,Option B — Scripted Creation (API)
# MAGIC %md
# MAGIC ## 🐍 Option B: Create Genie Space via API (Scripted)
# MAGIC
# MAGIC The cell below creates the Genie Space programmatically.
# MAGIC This is commented out by default — **run Option A first**, then use this for reference or automation.
# MAGIC
# MAGIC **Use case for Option B:**
# MAGIC - Setting up the lab for multiple participants automatically
# MAGIC - CI/CD pipeline that recreates the Genie Space on data refresh
# MAGIC - Migrating Genie Spaces between workspaces

# COMMAND ----------

# DBTITLE 1,Option B — Create Genie Space (API)
# ============================================================
# OPTION B: Programmatic Genie Space creation via REST API
# Uncomment to run. Requires Admin or Space Creator permissions.
# ============================================================

import json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get the first available SQL warehouse
warehouses = w.warehouses.list()
warehouse_id = None
for wh in warehouses:
    if wh.state and wh.state.value in ("RUNNING", "STARTING"):
        warehouse_id = wh.id
        break
    if wh.id:
        warehouse_id = wh.id  # fallback to any warehouse

if not warehouse_id:
    print("WARNING: No SQL warehouse found. Please create one and re-run this cell.")
else:
    table_identifiers = TABLES
    
    # Check if a Genie Space with this title already exists
    existing_spaces = w.api_client.do("GET", "/api/2.0/genie/spaces")
    genie_space_id = None
    for space in existing_spaces.get("spaces", []):
        if space.get("title") == GENIE_NAME:
            genie_space_id = space.get("space_id")
            print(f"Genie Space '{GENIE_NAME}' already exists (ID: {genie_space_id})")
            break

    if not genie_space_id:
        print(f"Creating Geneie Space '{GENIE_NAME}'...")
        serialized = json.dumps({
            "version": 2,
            "data_sources": {"tables": [{"identifier": t} for t in sorted(table_identifiers)]}
        })
        try:
            resp = w.api_client.do("POST", "/api/2.0/genie/spaces", body={
                "title": GENIE_NAME,
                "description": "AI-powered SQL assistant for DataBank financial advisors",
                "warehouse_id": warehouse_id,
                "instructions": instructions,
                "serialized_space": serialized,
            })
            genie_space_id = resp.get("space_id")
            print(f"Genie Space created (ID: {genie_space_id})")
        except Exception as e:
            print(f"Error creating Genie Space: {e}")

print("=================================================================")
print("Add Instructions and Sample Questions next to Complete the Setup")
print("Complete Option A (UI) first, then note your Genie Space ID for Module 07.")

# COMMAND ----------

# DBTITLE 1,Step 3 — Test Genie Queries via API
# ============================================================
# After creating the Genie Space (Option A or B),
# replace GENIE_SPACE_ID with your actual space ID.
# You can find it in the URL when viewing your Genie Space:
#   https://workspace.azuredatabricks.net/#genie/<SPACE_ID>
# ============================================================

GENIE_SPACE_ID = "01f17b4161221becbadc20a41938d27a"  # <-- FILL THIS IN after creating the space

if not GENIE_SPACE_ID:
    print("⚠️ GENIE_SPACE_ID is empty.")
    print("   1. Complete Option A or B to create the Genie Space.")
    print("   2. Copy the Space ID from the URL bar: #genie/<SPACE_ID>")
    print("   3. Paste it into GENIE_SPACE_ID above and re-run.")
else:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.dashboards import GenieAPI
    import time

    w = WorkspaceClient()
    genie = GenieAPI(w.api_client)

    test_query = "How many customers have each risk profile?"
    print(f"🧠 Testing Genie Space: '{test_query}'")
    print("-" * 55)
    print("⏳ Sending message and waiting for Genie response...")

    # start_conversation_and_wait polls until Genie finishes processing
    message = genie.start_conversation_and_wait(
        space_id=GENIE_SPACE_ID,
        content=test_query
    )

    print(f"✅ Status: {message.status}\n")

    # ----------------------------------------------------------------
    # Genie's answer lives in message.attachments, NOT message.content
    # (message.content is just the original question echoed back)
    # Attachment types:
    #   att.text.content  — prose / explanation text
    #   att.query.query   — the generated SQL
    #   att.query.description — plain-English description of the SQL
    # ----------------------------------------------------------------
    sql_query    = None
    text_summary = None

    for att in (message.attachments or []):
        # Text/explanation response
        if att.text and att.text.content:
            text_summary = att.text.content
        # SQL query response
        if att.query:
            if att.query.description:
                text_summary = att.query.description
            if att.query.query:
                sql_query = att.query.query

    if text_summary:
        print(f"Genie says:\n{text_summary}\n")

    if sql_query:
        print(f"Generated SQL:\n{sql_query}\n")
        print("Query results:")
        display(spark.sql(sql_query))
    elif not text_summary:
        # Debug: show raw fields so we know what Genie actually returned
        print("⚠️  No structured response found. Raw message fields:")
        for k, v in vars(message).items():
            if v is not None:
                print(f"  {k}: {str(v)[:200]}")

    print("\n✅ Genie Space is working")

# COMMAND ----------

# DBTITLE 1,Module 05 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 05 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | Genie Space `DataBank Financial Advisor` created | Visible under Genie in sidebar |
# MAGIC | 5 tables connected | customers, accounts, transactions, products, support_tickets |
# MAGIC | Instructions added | Business context pasted from Step 1 |
# MAGIC | Sample questions added | 5 certified queries from Step 2 |
# MAGIC | GENIE_SPACE_ID recorded | You’ll need this in Module 07 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📝 Note for Module 07
# MAGIC When building the AgentBricks Supervisor Agent, you will need your Genie Space ID.
# MAGIC Record it here for easy reference:
# MAGIC
# MAGIC ```
# MAGIC Genie Space ID: ________________________
# MAGIC (from URL: #genie/<SPACE_ID>)
# MAGIC ```
# MAGIC
# MAGIC ### 🚀 Next: Module 06 — ML Experiment
# MAGIC Open **`06_ml_experiment`** to track and compare prompt experiments using MLflow.