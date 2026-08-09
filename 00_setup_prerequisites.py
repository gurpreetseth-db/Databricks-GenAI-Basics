# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Module 00 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 00: Setup & Prerequisites
# MAGIC **Duration:** ~15 minutes | **Track:** All participants
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is this lab?
# MAGIC You are an engineer at **DataBank** — a fictional retail bank. Over the next 4 hours you will build a production-grade AI assistant that financial advisors can use to:
# MAGIC - 🔍 Query customer transactions and portfolios using natural language
# MAGIC - 📄 Search product documentation and compliance documents instantly
# MAGIC - ⚠️ Detect suspicious transactions and calculate risk scores
# MAGIC - 💬 Get product recommendations for customers
# MAGIC ### What You'll Build in This Module
# MAGIC - ✅ Python packages for the full lab
# MAGIC - ✅ Unity Catalog: `<USERNAME>_databank_lab` catalog
# MAGIC - ✅ Schema: `<USERNAME>_databank_lab.financial_data`
# MAGIC - ✅ Volume: `/Volumes/databank_lab/financial_data/documents`
# MAGIC - ✅ AI/Vector Search Endpoint: `<USERNAME>_vs_endpoint`
# MAGIC - ✅ Verified access to Databricks Foundation Models API

# COMMAND ----------

# DBTITLE 1,Configuration — Set Your Lab Variables
# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC
# MAGIC The cell below defines **all configuration variables** used across every module in this lab.
# MAGIC
# MAGIC > 💡 Update `CATALOG` and `CATALOG_STORAGE` in cell below to start the setup

# COMMAND ----------

# DBTITLE 1,Step 1 — Install Required Packages
# Install packages needed for the full lab
# This only needs to be run once per session.
# Databricks serverless will automatically restart Python after %pip.

%pip install faker==25.9.1 reportlab==4.2.5 databricks-vectorsearch==0.40 openai>=1.0.0 mlflow[databricks]>=2.15.0 -q

# COMMAND ----------

# DBTITLE 1,Step 2a — Update Storage Location For Your Schema
# ================================================================#
#           PROVIDE YOUR CATALOG_STORAGE_LOCATION                 #
#           PROVIDE YOUR CATALOG NAME (IF EXISTS)                 #
# ================================================================#

CATALOG = "databank_lab"
CATALOG_STORAGE = "s3://gsethi-anz-psa-external-storage"



# COMMAND ----------

# DBTITLE 1,Step 2b - Capture Lab Configuration Variables
# Get logged-in user information
user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

# Extract username before '@' and remove special characters
import re
username_clean = re.sub(r'\W+', '', user.split('@')[0])


# Unity Catalog location for all lab assets
if not CATALOG:
    CATALOG = f"{username_clean}_databank_lab"
else:
    CATALOG = CATALOG

SCHEMA = f"{username_clean}_financial_data"
VOLUME         = "documents"
VOLUME_PATH    = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Vector Search
# Endpoint is created automatically in Step 5 below
AI_VECTOR_SEARCH_ENDPOINT = f"{username_clean}_vs_endpoint"
VS_INDEX_NAME  = f"{CATALOG}.{SCHEMA}.product_docs_index"

# AI Gateway (created in Module 02)
AI_GW_ROUTE    = f"{username_clean}_databank-llm-route"

# Agent serving endpoint (created in Module 07)
AGENT_ENDPOINT = f"{username_clean}_databank-ai-advisor"

# Foundation Model used throughout the lab (no API key needed — hosted by Databricks)
FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_MODEL  = "databricks-gte-large-en"



# COMMAND ----------

# DBTITLE 1,Step 3 — Create Catalog
# Create the top-level catalog for all lab assets
# Using SDK to handle Default Storage enabled workspaces
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import CatalogInfo

w = WorkspaceClient()

try:
    cat = w.catalogs.get(CATALOG)
    print(f"✅ Catalog '{CATALOG}' already exists")
except Exception:
    cat = w.catalogs.create(name=CATALOG, comment='DataBank AI Lab — Hands-on lab catalog', options={'storage_location': CATALOG_STORAGE})
    print(f"✅ Catalog '{CATALOG}' created successfully")

spark.sql(f"""
  ALTER CATALOG {CATALOG}
  SET TAGS ('description' = 'DataBank AI Lab — Hands-on lab catalog')
""")

print(f"✅ Catalog '{CATALOG}' is ready")

# Verify it exists
result = spark.sql(f"SHOW CATALOGS LIKE '{CATALOG}'").collect()
print(f"   Catalog found: {result[0][0] if result else 'NOT FOUND — check permissions'}")

# COMMAND ----------

# DBTITLE 1,Step 4 — Create Schema and Volume
# Create the schema (database) inside the catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"""
  COMMENT ON SCHEMA {CATALOG}.{SCHEMA} IS
  'DataBank AI Lab — financial services synthetic dataset and AI assets'
""")

# Create the volume — this is where PDFs and documents will be stored
# Volumes act like a managed filesystem inside Unity Catalog
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# Verify the volume is accessible by listing its contents (empty at this point)
import os
print(f"\n📂 Volume contents: {os.listdir(VOLUME_PATH) or '(empty — ready for data)'}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Create Vector Search Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
w = WorkspaceClient()

print(f"🔍 Creating Vector Search endpoint: {AI_VECTOR_SEARCH_ENDPOINT}")
print()

try:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=AI_VECTOR_SEARCH_ENDPOINT)
    state = ep.endpoint_status.state if ep.endpoint_status else "READY"
    print(f"✅ Endpoint '{AI_VECTOR_SEARCH_ENDPOINT}' already exists — skipping creation")
    print(f"   Status : {state}")
except Exception:
    try:
        print(f"   Endpoint not found — provisioning now (takes 3-5 min)...")
        ep = w.vector_search_endpoints.create_endpoint_and_wait(
            name=AI_VECTOR_SEARCH_ENDPOINT,
            endpoint_type=EndpointType.STANDARD
        )
        print(f"\n✅ Vector Search endpoint '{AI_VECTOR_SEARCH_ENDPOINT}' is ready!")
    except Exception as create_err:
        if "quota" in str(create_err).lower() or "exceeded" in str(create_err).lower():
            print("⚠️  Workspace endpoint quota exceeded — listing available endpoints...")
            available = list(w.vector_search_endpoints.list_endpoints())
            if available:
                AI_VECTOR_SEARCH_ENDPOINT = available[0].name
                ep = available[0]
                state = ep.endpoint_status.state if ep.endpoint_status else "READY"
                print(f"✅ Using existing endpoint '{AI_VECTOR_SEARCH_ENDPOINT}' — Status: {state}")
                print(f"   Available: {[e.name for e in available]}")
                print(f"   💡 Update AI_VECTOR_SEARCH_ENDPOINT in Step 2 to switch endpoints.")
            else:
                print("❌ No available endpoints found. Contact your workspace admin.")
                raise
        else:
            raise

print(f"   Endpoint : {AI_VECTOR_SEARCH_ENDPOINT}")
print(f"\n💡 This endpoint will be used in Module 04 to index the PDF documents.")

# COMMAND ----------

# DBTITLE 1,Step 5 — Test Foundation Models API
from openai import OpenAI
from databricks.sdk import WorkspaceClient

# WorkspaceClient auto-detects credentials from the notebook environment
w = WorkspaceClient()

# Create an OpenAI-compatible client pointing to Databricks serving endpoints
client = OpenAI(
    api_key=w.config.authenticate().get("Authorization", "").replace("Bearer ", ""),
    base_url=f"{w.config.host}/serving-endpoints"
)

# Quick test — confirm the LLM is reachable
response = client.chat.completions.create(
    model=FOUNDATION_MODEL,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "Respond with exactly: DataBank AI Lab is ready!"}
    ],
    max_tokens=30
)

print("✅ Foundation Models API response:", response.choices[0].message.content)
print(f"   Model used  : {response.model}")
print(f"   Tokens used : {response.usage.total_tokens}")
print(f"   Workspace   : {w.config.host}")

# COMMAND ----------

# ================================================================
# RETURN IF ALL SUCCESSFUL
# ================================================================

print(f"📦  Catalog  : {CATALOG} ✅")
print(f"📦  Catalog Storage Location  : {CATALOG_STORAGE} ✅")
print(f"📁  Schema   : {CATALOG}.{SCHEMA} ✅")
print(f"📄  Volume   : {VOLUME_PATH} ✅")
print(f"🤖  LLM      : {FOUNDATION_MODEL} ✅")
print(f"📐  Embed    : {EMBEDDING_MODEL} ✅")
print(f"🤖  Agent    : {AGENT_ENDPOINT} ✅")
print(f"🤖  AI GW    : {AI_GW_ROUTE} ✅")
print(f"🔍  AI/Vector Search Endpoint : {AI_VECTOR_SEARCH_ENDPOINT} ✅")
print(f"🔍  VS Index : {VS_INDEX_NAME} ✅")



# COMMAND ----------

# DBTITLE 1,Module 00 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 00 Complete — Checkpoint
# MAGIC
# MAGIC Before moving to Module 01, confirm all checks pass:
# MAGIC
# MAGIC | Check | How to Verify |
# MAGIC |-------|---------------|
# MAGIC | Packages installed | No import errors in Step 1 |
# MAGIC | Catalog `databank_lab` exists | Visible in **Catalog Explorer** (left sidebar) |
# MAGIC | Schema `financial_data` exists | Expand `databank_lab` in Catalog Explorer |
# MAGIC | Volume `documents` exists | Expand `financial_data` → Volumes |
# MAGIC | Foundation Models API working | Response printed in Step 6 output |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Next: Module 01 — Data Generation
# MAGIC Open **`01_data_generation`** to generate the DataBank synthetic dataset:
# MAGIC - 500 customers, 500 accounts, 10,000 transactions, 25 products, 300 support tickets
# MAGIC - 7 financial PDFs uploaded to the volume