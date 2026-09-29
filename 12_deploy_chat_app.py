# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Deploy the DataBank Chat App (3 memory modes)
# MAGIC
# MAGIC One app — **`databank-chat-demo`** — deployed in whichever memory mode you pick.
# MAGIC All three modes use the **same** UC AI Gateway route (Claude Sonnet 4.5), UC
# MAGIC functions, Genie Agent, and Vector Search index from `Config_Parameters`.
# MAGIC
# MAGIC | Mode | Chat history | Agent memory |
# MAGIC |------|--------------|--------------|
# MAGIC | `simple`    | none (ephemeral)                | none |
# MAGIC | `shortterm` | this session only (resets on reopen) | conversation log in Lakebase |
# MAGIC | `longterm`  | all past sessions (per user)    | + durable fact recall across sessions |
# MAGIC
# MAGIC The app code lives in the **`databank-chat-demo/`** folder next to this notebook.
# MAGIC This notebook just resolves config + calls `databank-chat-demo/deploy_app.py`.

# COMMAND ----------

# MAGIC %md ### 1. Load the shared lab configuration

# COMMAND ----------

# MAGIC %run ./Config_Parameters

# COMMAND ----------

# MAGIC %md ### 2. Pick the memory mode

# COMMAND ----------

dbutils.widgets.dropdown("mode", "simple", ["simple", "shortterm", "longterm"], "Memory mode")
dbutils.widgets.text("app_name", "databank-chat-demo", "App name")
dbutils.widgets.text("genie_space_id", "01f1b6c9b5ad17bca4c13c1991f01333", "Genie space id")

MODE = dbutils.widgets.get("mode")
APP_NAME = dbutils.widgets.get("app_name")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id")

# COMMAND ----------

# MAGIC %md ### 3. Resolve UC-qualified names from Config_Parameters

# COMMAND ----------

import mlflow

# The AI Gateway route is registered as a UC model-service (3-level name).
MODEL_ROUTE = f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}"
VECTOR_INDEX = VS_INDEX_NAME  # already 3-level in Config_Parameters

# Resolve the MLflow experiment id from the experiment path in Config_Parameters
_exp = mlflow.get_experiment_by_name(experiment_name)
EXPERIMENT_ID = _exp.experiment_id if _exp else ""

print("Mode         :", MODE)
print("App          :", APP_NAME)
print("Catalog.Schema:", f"{CATALOG}.{SCHEMA}")
print("Model route  :", MODEL_ROUTE)
print("Vector index :", VECTOR_INDEX)
print("Genie space  :", GENIE_SPACE_ID, f"({GENIE_NAME})")
print("Experiment id:", EXPERIMENT_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Deploy
# MAGIC Passes the resolved config into `deploy_app.py`. Requires the Databricks CLI
# MAGIC (installed below if missing) with workspace auth available to the notebook.

# COMMAND ----------

import os

# Export resolved config so the %sh cell can read it
os.environ["DBK_MODE"] = MODE
os.environ["DBK_APP_NAME"] = APP_NAME
os.environ["DBK_CATALOG"] = CATALOG
os.environ["DBK_SCHEMA"] = SCHEMA
os.environ["DBK_MODEL_ROUTE"] = MODEL_ROUTE
os.environ["DBK_VECTOR_INDEX"] = VECTOR_INDEX
os.environ["DBK_GENIE_SPACE_ID"] = GENIE_SPACE_ID
os.environ["DBK_GENIE_NAME"] = GENIE_NAME
os.environ["DBK_EXPERIMENT_ID"] = EXPERIMENT_ID
os.environ["DBK_APP_DIR"] = os.path.join(os.getcwd(), "databank-chat-demo")

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
# MAGIC # Ensure the Databricks CLI is available
# MAGIC if ! command -v databricks >/dev/null 2>&1; then
# MAGIC   echo "Installing Databricks CLI..."
# MAGIC   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# MAGIC fi
# MAGIC cd "$DBK_APP_DIR"
# MAGIC python deploy_app.py \
# MAGIC   --mode "$DBK_MODE" \
# MAGIC   --app-name "$DBK_APP_NAME" \
# MAGIC   --catalog "$DBK_CATALOG" \
# MAGIC   --schema "$DBK_SCHEMA" \
# MAGIC   --model-route "$DBK_MODEL_ROUTE" \
# MAGIC   --vector-index "$DBK_VECTOR_INDEX" \
# MAGIC   --genie-space-id "$DBK_GENIE_SPACE_ID" \
# MAGIC   --genie-name "$DBK_GENIE_NAME" \
# MAGIC   --experiment-id "$DBK_EXPERIMENT_ID"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC - **Switch modes:** change the `mode` widget and re-run — it redeploys the same app.
# MAGIC - **Local alternative:** from a clone, run
# MAGIC   `python databank-chat-demo/deploy_app.py --mode <mode> --profile <profile>`.
# MAGIC - **short/long term** additionally grant the app's service principal
# MAGIC   `CONNECT`+`CREATE` on the Lakebase `agentic-memory` database (handled by
# MAGIC   `deploy_app.py`). The frontend + agent then create their own schemas.
