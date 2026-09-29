# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Deploy the DataBank Chat App (3 memory modes)
# MAGIC
# MAGIC One app — **`databank-chat-demo`** — deployed in whichever memory mode you pick.
# MAGIC **You only choose the mode here.** Everything else (app name, model route, Genie
# MAGIC space, functions, Lakebase, experiment) resolves automatically from
# MAGIC `Config_Parameters.py` — the single source of truth. `app.yaml` is generated from
# MAGIC those values at deploy time (never hand-edited).
# MAGIC
# MAGIC | Mode | Chat history | Agent memory |
# MAGIC |------|--------------|--------------|
# MAGIC | `simple`    | none (ephemeral)                     | none |
# MAGIC | `shortterm` | this session only (resets on reopen) | conversation log in Lakebase |
# MAGIC | `longterm`  | all past sessions (per user)         | + durable fact recall across sessions |

# COMMAND ----------

# MAGIC %md ### 1. Load the shared lab configuration (single source of truth)

# COMMAND ----------

# MAGIC %run ./Config_Parameters

# COMMAND ----------

# MAGIC %md ### 2. Pick the memory mode (the only choice you make)

# COMMAND ----------

dbutils.widgets.dropdown("mode", "simple", ["simple", "shortterm", "longterm"], "Memory mode")
MODE = dbutils.widgets.get("mode")

# COMMAND ----------

# MAGIC %md ### 3. Everything else resolves from Config_Parameters.py

# COMMAND ----------

import mlflow

# Resolve the MLflow experiment id from the experiment path in Config_Parameters
_exp = mlflow.get_experiment_by_name(experiment_name)
EXPERIMENT_ID = _exp.experiment_id if _exp else ""

print("Mode          :", MODE)
print("App           :", APP_NAME)
print("Catalog.Schema:", f"{CATALOG}.{SCHEMA}")
print("Model route   :", MODEL_ROUTE)
print("Vector index  :", VS_INDEX_NAME)
print("Genie         :", GENIE_SPACE_ID, f"({GENIE_NAME})")
print("UC functions  :", ", ".join(UC_FUNCTIONS))
print("Lakebase      :", f"{LAKEBASE_PROJECT}/{LAKEBASE_BRANCH}/{LAKEBASE_ENDPOINT}")
print("Experiment id :", EXPERIMENT_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Deploy
# MAGIC Passes the resolved config into `deploy_app.py`. Requires the Databricks CLI
# MAGIC (installed below if missing) with workspace auth available to the notebook.

# COMMAND ----------

import os

# Export the resolved config so the %sh cell can consume it (all values originate
# from Config_Parameters.py — nothing is defined here except the chosen mode).
os.environ["DBK_MODE"] = MODE
os.environ["DBK_APP_NAME"] = APP_NAME
os.environ["DBK_CATALOG"] = CATALOG
os.environ["DBK_SCHEMA"] = SCHEMA
os.environ["DBK_MODEL_ROUTE"] = MODEL_ROUTE
os.environ["DBK_VECTOR_INDEX"] = VS_INDEX_NAME
os.environ["DBK_GENIE_SPACE_ID"] = GENIE_SPACE_ID
os.environ["DBK_GENIE_NAME"] = GENIE_NAME
os.environ["DBK_UC_FUNCTIONS"] = ",".join(UC_FUNCTIONS)
os.environ["DBK_EXPERIMENT_ID"] = EXPERIMENT_ID
os.environ["DBK_LAKEBASE_PROJECT"] = LAKEBASE_PROJECT
os.environ["DBK_LAKEBASE_BRANCH"] = LAKEBASE_BRANCH
os.environ["DBK_LAKEBASE_ENDPOINT"] = LAKEBASE_ENDPOINT
os.environ["DBK_LAKEBASE_DATABASE"] = LAKEBASE_DATABASE
os.environ["DBK_APP_DIR"] = os.path.join(os.getcwd(), "databank-chat-demo")

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
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
# MAGIC   --uc-functions "$DBK_UC_FUNCTIONS" \
# MAGIC   --experiment-id "$DBK_EXPERIMENT_ID" \
# MAGIC   --lakebase-project "$DBK_LAKEBASE_PROJECT" \
# MAGIC   --lakebase-branch "$DBK_LAKEBASE_BRANCH" \
# MAGIC   --lakebase-endpoint "$DBK_LAKEBASE_ENDPOINT" \
# MAGIC   --lakebase-database "$DBK_LAKEBASE_DATABASE"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC - **Switch modes:** change the `mode` widget and re-run — same app, redeployed.
# MAGIC - **Change any config value:** edit `Config_Parameters.py` only. `app.yaml` is
# MAGIC   regenerated from it on every deploy (it is gitignored / not hand-maintained).
# MAGIC - **Local alternative:** `python databank-chat-demo/deploy_app.py --mode <mode>
# MAGIC   --profile <profile> --catalog ... --schema ...` (pass the same values).
