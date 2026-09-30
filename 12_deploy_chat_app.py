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

# MAGIC %md ### 3. Fetch Genie Space ID from Name

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
genie_space_id = None
existing_spaces = w.api_client.do("GET", "/api/2.0/genie/spaces")
for space in existing_spaces.get("spaces", []):
    if space.get("title") == GENIE_NAME:
        genie_space_id = space.get("space_id")
        break

print(f"Genie Space ID for '{GENIE_NAME}': {genie_space_id}")

# COMMAND ----------

# MAGIC %md ### 4. Everything else resolves from Config_Parameters.py

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
# MAGIC ### 5. Deploy via the Databricks Asset Bundle
# MAGIC The chosen `mode` maps to a bundle **target** (`dev_<mode>`). The `%sh` cell
# MAGIC runs `databricks bundle deploy` (uploads the whole codebase + creates the app)
# MAGIC then `deploy_app.py` (env + grants DAB can't express). Requires the Databricks
# MAGIC CLI (installed below if missing) with the notebook's workspace auth.

# COMMAND ----------

import os

# The mode widget is the only choice; everything else is a bundle variable in
# databricks.yml. We pass the runtime-resolved experiment id, Genie space id and
# cleaned username as --var overrides so the bundle stays the single source.
os.environ["DBK_TARGET"] = f"dev_{MODE}"
os.environ["DBK_USERNAME"] = username_clean
os.environ["DBK_GENIE_SPACE_ID"] = GENIE_SPACE_ID
os.environ["DBK_EXPERIMENT_ID"] = EXPERIMENT_ID
os.environ["DBK_BUNDLE_DIR"] = os.getcwd()  # repo root — where databricks.yml lives

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
# MAGIC if ! command -v databricks >/dev/null 2>&1; then
# MAGIC   echo "Installing Databricks CLI..."
# MAGIC   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# MAGIC fi
# MAGIC cd "$DBK_BUNDLE_DIR"
# MAGIC # 1) Deploy the codebase + app + setup job for this mode's target.
# MAGIC databricks bundle deploy -t "$DBK_TARGET" \
# MAGIC   --var="username=$DBK_USERNAME" \
# MAGIC   --var="genie_space_id=$DBK_GENIE_SPACE_ID" \
# MAGIC   --var="experiment_id=$DBK_EXPERIMENT_ID"
# MAGIC # 2) Apply env + grants DAB can't express, then redeploy the app.
# MAGIC python databank-chat-demo/deploy_app.py -t "$DBK_TARGET"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC - **Switch modes:** change the `mode` widget and re-run — same app, retargeted
# MAGIC   (`simple` → `dev_simple`, etc.).
# MAGIC - **Change any config value:** edit `Config_Parameters.py` (notebook overrides)
# MAGIC   or the `variables:` defaults in `databricks.yml`. The bundle is the single
# MAGIC   source of truth; `deploy_app.py` reads resolved values from it.
# MAGIC - **Provision the lab (optional):** `databricks bundle run databank_lab_setup -t <target>`
# MAGIC   runs the setup notebooks (00→07) on serverless compute.
# MAGIC - **Local alternative:** from the repo root, `databricks bundle deploy -t dev_<mode>
# MAGIC   -p <profile>` then `python databank-chat-demo/deploy_app.py -t dev_<mode> -p <profile>`.
