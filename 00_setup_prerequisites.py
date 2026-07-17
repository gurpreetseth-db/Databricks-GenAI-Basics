# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

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
# MAGIC
# MAGIC The assistant will be deployed as a live **Databricks App** powered by **AgentBricks**, backed by **AI Gateway**, **Vector Search**, and **Unity Catalog Functions**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Lab Modules
# MAGIC | # | Module | Duration | Key Output |
# MAGIC |---|--------|----------|-----------|
# MAGIC | 00 | Setup & Prerequisites | 15 min | Catalog, schema, volume |
# MAGIC | 01 | Data Generation | 25 min | 5 Delta tables + 7 PDFs |
# MAGIC | 02 | AI Gateway | 20 min | Managed LLM route |
# MAGIC | 03 | UC Functions | 25 min | 3 registered functions |
# MAGIC | 04 | Vector Search | 30 min | Searchable document index |
# MAGIC | 05 | Genie Space | 15 min | NL-to-SQL on financial data |
# MAGIC | 06 | ML Experiment | 20 min | Tracked prompt experiments |
# MAGIC | 07 | AgentBricks | 40 min | Deployed AI advisor agent |
# MAGIC | 08 | Databricks App | 25 min | Live Gradio chat application |
# MAGIC | 09 | Evaluation | 20 min | LLM-as-a-judge scores |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What You'll Build in This Module
# MAGIC - ✅ Python packages for the full lab
# MAGIC - ✅ Unity Catalog: `databank_lab` catalog
# MAGIC - ✅ Schema: `databank_lab.financial_data`
# MAGIC - ✅ Volume: `/Volumes/databank_lab/financial_data/documents`
# MAGIC - ✅ Verified access to Databricks Foundation Models API

# COMMAND ----------

# DBTITLE 1,Step 1 — Install Required Packages
# Install packages needed for the full lab
# This only needs to be run once per session.
# Databricks serverless will automatically restart Python after %pip.

%pip install faker==25.9.1 reportlab==4.2.5 databricks-vectorsearch==0.40 openai>=1.0.0 mlflow[databricks]>=2.15.0 -q

# COMMAND ----------

# DBTITLE 1,Configuration — Set Your Lab Variables
# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC
# MAGIC The cell below defines **all configuration variables** used across every module in this lab.
# MAGIC
# MAGIC > 💡 All variables are pre-set with sensible defaults. The Vector Search endpoint `databank-vs-endpoint` is created automatically in **Step 5** below — no manual configuration required.

# COMMAND ----------

# DBTITLE 1,Step 2 — Lab Configuration Variables
# ================================================================
# LAB CONFIGURATION — Review and update if needed
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


# Unity Catalog location for all lab assets
CATALOG        = "databank_lab"
SCHEMA         = "financial_data"
VOLUME         = "documents"
VOLUME_PATH    = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Vector Search
# Endpoint is created automatically in Step 5 below
VS_ENDPOINT    = "databank-vs-endpoint"
VS_INDEX_NAME  = f"{CATALOG}.{SCHEMA}.product_docs_index"

# AI Gateway (created in Module 02)
AI_GW_ROUTE    = "databank-llm-route"

# Agent serving endpoint (created in Module 07)
AGENT_ENDPOINT = "databank-ai-advisor"

# Foundation Model used throughout the lab (no API key needed — hosted by Databricks)
FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_MODEL  = "databricks-gte-large-en"

# ================================================================
print(f"📦  Catalog  : {CATALOG}")
print(f"📁  Schema   : {CATALOG}.{SCHEMA}")
print(f"📄  Volume   : {VOLUME_PATH}")
print(f"🔍  VS Index : {VS_INDEX_NAME}")
print(f"🤖  LLM      : {FOUNDATION_MODEL}")
print(f"📐  Embed    : {EMBEDDING_MODEL}")

# COMMAND ----------

# DBTITLE 1,Infrastructure — Create Catalog, Schema & Volume
# MAGIC %md
# MAGIC ## 🏗️ Infrastructure Setup
# MAGIC
# MAGIC ### Unity Catalog Hierarchy
# MAGIC ```
# MAGIC databank_lab                          ← Catalog (top-level namespace)
# MAGIC └── financial_data                    ← Schema (logical grouping)
# MAGIC     ├── customers                     ← Delta table (Module 01)
# MAGIC     ├── accounts                      ← Delta table (Module 01)
# MAGIC     ├── transactions                  ← Delta table (Module 01)
# MAGIC     ├── products                      ← Delta table (Module 01)
# MAGIC     ├── support_tickets               ← Delta table (Module 01)
# MAGIC     ├── product_docs_chunks           ← Delta table (Module 04)
# MAGIC     ├── product_docs_index            ← Vector Search Index (Module 04)
# MAGIC     └── documents/                    ← Volume (PDFs stored here)
# MAGIC         ├── product_brochures/
# MAGIC         └── compliance/
# MAGIC ```
# MAGIC
# MAGIC **Key concept:** Unity Catalog provides a 3-level namespace (`catalog.schema.table`) giving you centralised governance, access control, and lineage across all data assets.

# COMMAND ----------

# DBTITLE 1,Step 3 — Create Catalog
# Create the top-level catalog for all lab assets
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")

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

print(f"✅ Schema  : {CATALOG}.{SCHEMA}")
print(f"✅ Volume  : {VOLUME_PATH}")

# Verify the volume is accessible by listing its contents (empty at this point)
import os
print(f"\n📂 Volume contents: {os.listdir(VOLUME_PATH) or '(empty — ready for data)'}")

# COMMAND ----------

# DBTITLE 1,Vector Search Endpoint — Concept
# MAGIC %md
# MAGIC ## 🔍 Vector Search Endpoint
# MAGIC
# MAGIC A **Vector Search endpoint** is the compute resource that hosts your vector indexes. It handles:
# MAGIC - Embedding new documents as they are added
# MAGIC - Serving similarity queries at low latency
# MAGIC - Managing the HNSW index structure
# MAGIC
# MAGIC We create a dedicated endpoint named `databank-vs-endpoint` as part of setup. This provisioning takes 3–5 minutes in the background — you can continue to Module 01 while it spins up.
# MAGIC
# MAGIC > ⏱ The endpoint only needs to be created **once**. Subsequent runs of Module 00 detect it already exists and skip creation instantly.

# COMMAND ----------

# DBTITLE 1,Step 5 — Create Vector Search Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
w = WorkspaceClient()

SHARED_ENDPOINT = "one-env-shared-endpoint-10"  # fallback for shared workspaces

print(f"🔍 Creating Vector Search endpoint: {VS_ENDPOINT}")
print()

try:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
    state = ep.endpoint_status.state if ep.endpoint_status else "READY"
    print(f"✅ Endpoint '{VS_ENDPOINT}' already exists — skipping creation")
    print(f"   Status : {state}")
except Exception:
    try:
        print(f"   Endpoint not found — provisioning now (takes 3-5 min)...")
        ep = w.vector_search_endpoints.create_endpoint_and_wait(
            name=VS_ENDPOINT,
            endpoint_type=EndpointType.STANDARD
        )
        print(f"\n✅ Vector Search endpoint '{VS_ENDPOINT}' is ready!")
    except Exception as create_err:
        if "quota" in str(create_err).lower() or "exceeded" in str(create_err).lower():
            print("⚠️  Workspace endpoint quota exceeded — listing available endpoints...")
            available = list(w.vector_search_endpoints.list_endpoints())
            if available:
                VS_ENDPOINT = available[0].name
                ep = available[0]
                state = ep.endpoint_status.state if ep.endpoint_status else "READY"
                print(f"✅ Using existing endpoint '{VS_ENDPOINT}' — Status: {state}")
                print(f"   Available: {[e.name for e in available]}")
                print(f"   💡 Update VS_ENDPOINT in Step 2 to switch endpoints.")
            else:
                print("❌ No available endpoints found. Contact your workspace admin.")
                raise
        else:
            raise

print(f"   Endpoint : {VS_ENDPOINT}")
print(f"\n💡 This endpoint will be used in Module 04 to index the PDF documents.")

# COMMAND ----------

# DBTITLE 1,Foundation Models API — Concept
# MAGIC %md
# MAGIC ## 🧠 Databricks Foundation Models API
# MAGIC
# MAGIC Databricks provides **pay-per-token access** to state-of-the-art LLMs — no external API keys, no model management, billed directly to your workspace.
# MAGIC
# MAGIC | Model | Use Case | Context Window |
# MAGIC |-------|----------|----------------|
# MAGIC | `databricks-meta-llama-3-3-70b-instruct` | Chat, reasoning, agents | 128k tokens |
# MAGIC | `databricks-meta-llama-3-1-405b-instruct` | Complex reasoning | 128k tokens |
# MAGIC | `databricks-gte-large-en` | Text embeddings for search | 8k tokens |
# MAGIC
# MAGIC The API is **OpenAI-compatible** — the same `openai` Python client works by just changing the `base_url`.

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