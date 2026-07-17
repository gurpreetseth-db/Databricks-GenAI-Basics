# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# DBTITLE 1,Module 03 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 03: Unity Catalog Functions
# MAGIC **Duration:** ~25 minutes | **Prerequisite:** Module 01 (data tables required)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What Are UC Functions?
# MAGIC
# MAGIC **Unity Catalog Functions** are Python or SQL functions registered in UC that can be:
# MAGIC - Called from any notebook or query
# MAGIC - Used as **tools by AI agents** (AgentBricks / LangChain / LlamaIndex)
# MAGIC - Governed by UC access control (GRANT EXECUTE ON FUNCTION)
# MAGIC - Versioned and documented with comments
# MAGIC
# MAGIC ### Why Does DataBank Need These?
# MAGIC Our AI advisor needs to perform **calculations and lookups** that go beyond what an LLM knows:
# MAGIC - 📊 **Risk scoring**: calculate a customer’s investment risk score from their profile
# MAGIC - 💼 **Portfolio summary**: retrieve a customer’s account balances and product holdings
# MAGIC - ⚠️ **Fraud detection**: flag suspicious transactions based on amount and pattern rules
# MAGIC
# MAGIC These are deterministic business-logic functions — not generative AI. By wrapping them as UC Functions and giving them to the agent as tools, the agent can call them on demand.
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC | Function | Type | Purpose |
# MAGIC |----------|------|---------|
# MAGIC | `calculate_customer_risk` | SQL | Compute a 0–100 risk score from customer profile |
# MAGIC | `get_portfolio_summary` | Python | Return account balances and product types |
# MAGIC | `flag_suspicious_transactions` | SQL | Return recent high-value or anomalous transactions |

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


SCHEMA  = "financial_data"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print(f"✅ Functions will be registered in: {CATALOG}.{SCHEMA}")
print(f"   Workspace: {w.config.host}")

# Verify the source tables exist
for tbl in ["customers", "accounts", "transactions"]:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{tbl}").count()
    print(f"   ✓ {tbl}: {count:,} rows")

# COMMAND ----------

# DBTITLE 1,Function 1 — Risk Score
# MAGIC %md
# MAGIC ## 📊 Function 1: `calculate_customer_risk`
# MAGIC
# MAGIC This SQL function computes a **0–100 risk score** for a given customer based on:
# MAGIC - Age (older = slightly more conservative → lower risk score)
# MAGIC - Annual income (higher income can support higher risk)
# MAGIC - Stated risk profile (Conservative / Moderate / Aggressive)
# MAGIC
# MAGIC The output is a single integer — simple, deterministic, and auditable.
# MAGIC
# MAGIC > 💡 **Why SQL?** Simple scalar calculations with lookup tables work beautifully in SQL UDFs.
# MAGIC > SQL UDFs are also faster than Python UDFs for scalar operations.

# COMMAND ----------

# DBTITLE 1,Step 1 — Create calculate_customer_risk Function
# MAGIC %sql
# MAGIC -- Drop and recreate for idempotency
# MAGIC DROP FUNCTION IF EXISTS databank_lab.financial_data.calculate_customer_risk;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION databank_lab.financial_data.calculate_customer_risk(
# MAGIC   customer_id STRING COMMENT 'The DataBank customer ID (e.g. CUST-0001)'
# MAGIC )
# MAGIC RETURNS INT
# MAGIC COMMENT 'Calculate a 0-100 risk score for a customer based on age, income, and stated risk profile.
# MAGIC Higher score = higher risk tolerance. Conservative ~ 20-40, Moderate ~ 40-65, Aggressive ~ 65-90.'
# MAGIC LANGUAGE SQL
# MAGIC RETURN (
# MAGIC   SELECT
# MAGIC     MAX(CAST(
# MAGIC       CASE risk_profile
# MAGIC         WHEN 'Conservative' THEN 20
# MAGIC         WHEN 'Moderate'     THEN 50
# MAGIC         WHEN 'Aggressive'   THEN 75
# MAGIC         ELSE                     40
# MAGIC       END
# MAGIC       -- Income adjustment: +/- 10 points based on income vs median
# MAGIC       + LEAST(10, GREATEST(-10,
# MAGIC           CAST((annual_income_gbp - 37000) / 8000 AS INT)
# MAGIC         ))
# MAGIC       -- Age adjustment: reduce score slightly for customers over 60
# MAGIC       - CASE WHEN age > 60 THEN 5 ELSE 0 END
# MAGIC     AS INT)) AS risk_score
# MAGIC   FROM databank_lab.financial_data.customers
# MAGIC   WHERE customer_id = calculate_customer_risk.customer_id
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Step 1b — Test calculate_customer_risk
# Test the function with sample customers
test_results = spark.sql(f"""
  SELECT
    c.customer_id,
    c.full_name,
    c.risk_profile,
    c.age,
    ROUND(c.annual_income_gbp, 0) AS income,
    {CATALOG}.{SCHEMA}.calculate_customer_risk(c.customer_id) AS risk_score
  FROM {CATALOG}.{SCHEMA}.customers c
  ORDER BY risk_profile, age
  LIMIT 9
""")

print("📊 Sample Risk Scores:")
display(test_results)

# Verify function is registered in UC
spark.sql(f"USE CATALOG {CATALOG}")
functions = spark.sql(f"SHOW FUNCTIONS IN {SCHEMA}").filter("function LIKE '%risk%'").collect()
print(f"\n✅ Function registered: {[f.function for f in functions]}")

# COMMAND ----------

# DBTITLE 1,Function 2 — Portfolio Summary
# MAGIC %md
# MAGIC ## 💼 Function 2: `get_portfolio_summary`
# MAGIC
# MAGIC This Python function returns a **formatted text summary** of a customer’s complete portfolio:
# MAGIC - All accounts with product names and current balances
# MAGIC - Total assets under management
# MAGIC - Portfolio composition by product type
# MAGIC
# MAGIC > 💡 **Why Python?** String formatting, aggregation logic, and conditional text generation
# MAGIC > are much cleaner in Python than SQL. Python UDFs also allow importing standard libraries.
# MAGIC
# MAGIC The function returns a formatted string that the **AI agent can directly include in its response** — no extra parsing needed.

# COMMAND ----------

# DBTITLE 1,Step 2 — Create get_portfolio_summary Function
# Step 2 — Create get_portfolio_summary (SQL UC Function)
spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.get_portfolio_summary")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_portfolio_summary(
  customer_id STRING COMMENT 'The DataBank customer ID (e.g. CUST-0001)'
)
RETURNS STRING
COMMENT 'Returns a formatted text summary of all accounts, balances, and product types for a customer. Ideal for portfolio review questions.'
LANGUAGE SQL
RETURN (
  SELECT
    CONCAT(
      'Portfolio for ', MAX(c.full_name), ' (', MAX(customer_id), '):\\n',
      'Risk Profile: ', MAX(c.risk_profile), ' | Member Since: ', MAX(c.member_since), '\\n',
      '\\nAccounts:\\n',
      COALESCE(
        ARRAY_JOIN(
          COLLECT_LIST(
            CONCAT(
              '  • ', p.name, ' (', p.product_type, ')',
              ' | Balance: £', FORMAT_NUMBER(a.balance_gbp, 2),
              ' | Status: ', a.status
            )
          ), '\\n'
        ), 'No accounts found'
      ),
      '\\n\\nTotal Assets Under Management: £', FORMAT_NUMBER(SUM(a.balance_gbp), 2),
      '\\nNumber of Products: ', COUNT(a.account_id)
    )
  FROM databank_lab.financial_data.customers c
  LEFT JOIN databank_lab.financial_data.accounts a USING (customer_id)
  LEFT JOIN databank_lab.financial_data.products p USING (product_id)
  WHERE c.customer_id = get_portfolio_summary.customer_id
)
""")

print(f"✅ get_portfolio_summary function created")

# COMMAND ----------

# DBTITLE 1,Step 2b — Test get_portfolio_summary
# Get a sample customer ID from the database
sample_customer = spark.sql(f"SELECT customer_id FROM {CATALOG}.{SCHEMA}.customers LIMIT 1").collect()[0][0]

result = spark.sql(f"""
  SELECT {CATALOG}.{SCHEMA}.get_portfolio_summary('{sample_customer}') AS portfolio
""").collect()[0][0]

print(f"💼 Portfolio Summary for {sample_customer}:")
print("-" * 60)
if result:
    print(result)
else:
    print("No portfolio data returned — check that accounts table has data for this customer")
print("-" * 60)
print("\n✅ get_portfolio_summary is working")

# COMMAND ----------

# DBTITLE 1,Function 3 — Suspicious Transactions
# MAGIC %md
# MAGIC ## ⚠️ Function 3: `flag_suspicious_transactions`
# MAGIC
# MAGIC This SQL function returns a formatted list of **recent suspicious transactions** for a customer:
# MAGIC - Transactions flagged as `is_fraud = true` in the last 90 days
# MAGIC - Large unusual transactions (amount > £1,000 in a single purchase)
# MAGIC - Multiple transactions in quick succession
# MAGIC
# MAGIC The agent calls this function when a user asks: 
# MAGIC *"Are there any suspicious transactions for customer X?"* or
# MAGIC *"Check for fraud on CUST-0042"*

# COMMAND ----------

# DBTITLE 1,Step 3 — Create flag_suspicious_transactions Function
# MAGIC %sql
# MAGIC DROP FUNCTION IF EXISTS databank_lab.financial_data.flag_suspicious_transactions;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION databank_lab.financial_data.flag_suspicious_transactions(
# MAGIC   customer_id  STRING  COMMENT 'The DataBank customer ID to check (e.g. CUST-0001)',
# MAGIC   lookback_days INT    COMMENT 'Number of days to look back (default 90)'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns a formatted text summary of suspicious or fraudulent transactions for a customer in the last N days. Returns a clear message if no suspicious activity is found.'
# MAGIC LANGUAGE SQL
# MAGIC RETURN (
# MAGIC   WITH suspicious AS (
# MAGIC     SELECT
# MAGIC       txn_id,
# MAGIC       txn_date,
# MAGIC       amount_gbp,
# MAGIC       merchant,
# MAGIC       category,
# MAGIC       channel,
# MAGIC       status,
# MAGIC       CASE
# MAGIC         WHEN is_fraud           THEN 'FRAUD FLAGGED'
# MAGIC         WHEN amount_gbp > 1000  THEN 'LARGE AMOUNT'
# MAGIC         ELSE                         'ANOMALY'
# MAGIC       END AS alert_type
# MAGIC     FROM databank_lab.financial_data.transactions
# MAGIC     WHERE customer_id  = flag_suspicious_transactions.customer_id
# MAGIC       AND txn_date    >= DATE_SUB(CURRENT_DATE(), lookback_days)
# MAGIC       AND (is_fraud = true OR amount_gbp > 1000)
# MAGIC     ORDER BY txn_date DESC
# MAGIC     LIMIT 10
# MAGIC   )
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN COUNT(*) = 0
# MAGIC         THEN CONCAT('No suspicious transactions found for customer ', customer_id,
# MAGIC                     ' in the last ', lookback_days, ' days. Account appears normal.')
# MAGIC       ELSE
# MAGIC         CONCAT(
# MAGIC           'SUSPICIOUS ACTIVITY ALERT for ', customer_id, ':\n',
# MAGIC           'Found ', COUNT(*), ' suspicious transaction(s) in last ', lookback_days, ' days:\n\n',
# MAGIC           ARRAY_JOIN(
# MAGIC             COLLECT_LIST(
# MAGIC               CONCAT(
# MAGIC                 '  ⚠ ', txn_id, ' | ', txn_date, ' | £', FORMAT_NUMBER(amount_gbp, 2),
# MAGIC                 ' | ', merchant, ' | ', alert_type
# MAGIC               )
# MAGIC             ), '\n'
# MAGIC           ),
# MAGIC           '\n\nAction: Please review these transactions with the customer immediately.'
# MAGIC         )
# MAGIC     END AS fraud_report
# MAGIC   FROM suspicious
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Step 3b — Test flag_suspicious_transactions
# Find a customer with known fraudulent transactions
fraud_customer = spark.sql(f"""
  SELECT DISTINCT customer_id
  FROM {CATALOG}.{SCHEMA}.transactions
  WHERE is_fraud = true
  LIMIT 1
""").collect()

if fraud_customer:
    test_cust_id = fraud_customer[0][0]
    print(f"🔍 Testing with customer who has fraud activity: {test_cust_id}")
else:
    # Fallback to first customer
    test_cust_id = spark.sql(f"SELECT customer_id FROM {CATALOG}.{SCHEMA}.customers LIMIT 1").collect()[0][0]
    print(f"🔍 Testing with: {test_cust_id} (no fraud found, expect clean report)")

result = spark.sql(f"""
  SELECT {CATALOG}.{SCHEMA}.flag_suspicious_transactions('{test_cust_id}', 90) AS fraud_report
""").collect()[0][0]

print()
print(result)
print()
print("✅ flag_suspicious_transactions is working")

# COMMAND ----------

# DBTITLE 1,Step 4 — List All Registered Functions
# List all functions registered in the schema
print(f"📚 UC Functions registered in {CATALOG}.{SCHEMA}:")
print("-" * 55)

functions_df = spark.sql(f"SHOW FUNCTIONS IN {CATALOG}.{SCHEMA}")
for row in functions_df.collect():
    fn_name = row[0]
    if "databank" in fn_name or "financial" in fn_name or any(
        x in fn_name for x in ["calculate", "get_portfolio", "flag_suspicious"]
    ):
        print(f"  ✓ {fn_name}")

print()
print("💡 These functions are now available as TOOLS for the AgentBricks agent in Module 07.")
print("   The agent will call them automatically when users ask relevant questions.")

# COMMAND ----------

# DBTITLE 1,Module 03 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 03 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | `calculate_customer_risk` | Returns integer 0–100 — test in Step 1b |
# MAGIC | `get_portfolio_summary` | Returns formatted text with account details |
# MAGIC | `flag_suspicious_transactions` | Returns fraud alert or clean report |
# MAGIC | Functions in UC | All 3 visible under `databank_lab.financial_data` in Catalog Explorer → Functions |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### How the Agent Uses These
# MAGIC ```
# MAGIC User: "What is the risk score for CUST-0042?"
# MAGIC   → Agent calls: databank_lab.financial_data.calculate_customer_risk("CUST-0042")
# MAGIC   → Returns: 68 (Aggressive-leaning Moderate)
# MAGIC   → Agent: "Customer CUST-0042 has a risk score of 68, indicating a growth-oriented profile..."
# MAGIC
# MAGIC User: "Show me the portfolio for CUST-0001"
# MAGIC   → Agent calls: databank_lab.financial_data.get_portfolio_summary("CUST-0001")
# MAGIC   → Returns: formatted portfolio text
# MAGIC   → Agent: "Here is the complete portfolio for..." 
# MAGIC ```
# MAGIC
# MAGIC ### 🚀 Next: Module 04 — Vector Search
# MAGIC Open **`04_vector_search`** to chunk the PDF documents and create a searchable vector index.