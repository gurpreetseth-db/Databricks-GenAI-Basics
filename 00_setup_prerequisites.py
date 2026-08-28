# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Module 00 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 00: Setup & Prerequisites
# MAGIC **Duration:** ~10 minutes | **Track:** All participants
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What You'll Build in This Module
# MAGIC - ✅ Python packages for the full lab
# MAGIC - ✅ Unity Catalog
# MAGIC - ✅ Schema
# MAGIC - ✅ Volume
# MAGIC - ✅ AI/Vector Search Endpoint
# MAGIC - ✅ Verified access to Databricks Foundation Models API

# COMMAND ----------

# DBTITLE 1,Step 1 — Install Required Packages
# Install packages needed for the full lab
# This only needs to be run once per session.
# Databricks serverless will automatically restart Python after %pip.

%pip install faker==25.9.1 reportlab==4.2.5 databricks-vectorsearch==0.40 openai>=1.0.0 mlflow[databricks]>=2.15.0 -q

# COMMAND ----------

# DBTITLE 1,Step 2 - Reference Parameters
# MAGIC %run ./Config_Parameters

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

print(f"\n📂 Schema Created Sucessfully : {SCHEMA}")

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
    base_url=f"{w.config.host}/ai-gateway/mlflow/v1"
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

# DBTITLE 1,Step 6 - Create ML Flow Experiment
# MLflow tracks your agent's performance. This creates an experiment where traces and evaluation metrics will be logged.

import mlflow

mlflow.set_tracking_uri("databricks")

try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment and experiment.lifecycle_stage == "active":
        experiment_id = experiment.experiment_id
        print(f"✅ MLflow experiment already exists: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ MLflow experiment created: {experiment_name} (ID: {experiment_id})")
except Exception:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✅ MLflow experiment created: {experiment_name} (ID: {experiment_id})")

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1,Setup Completed
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
print(f"🤖  Experiment : {experiment_name} ✅")
print(f"🔍 Genie Agent: {GENIE_NAME} ✅")



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