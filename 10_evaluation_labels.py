# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Module 10 — Evaluation Labels
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 10: Evaluation Labels
# MAGIC **Duration:** ~15 minutes | **Prerequisite:** Module 09 (evaluation run)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What are Evaluation Labels?
# MAGIC
# MAGIC Every time `mlflow.genai.evaluate()` runs, each scorer assigns a **label** to every row in the dataset — a structured verdict on whether the agent response met the required standard.
# MAGIC
# MAGIC Labels have three parts:
# MAGIC
# MAGIC | Part | Description | Example |
# MAGIC |------|-------------|----------|
# MAGIC | `score` | Numeric value (0.0 – 1.0 or boolean) | `0.8` |
# MAGIC | `rationale` | Natural-language explanation from the judge | `"Answer is factually correct but omits FSCS detail"` |
# MAGIC | `source` | Who produced the label | `"databricks-meta-llama-3-3-70b-instruct"` |
# MAGIC
# MAGIC ### Label Types Used in This Lab
# MAGIC
# MAGIC | Scorer | Label column | Scale | What it measures |
# MAGIC |--------|-------------|-------|------------------|
# MAGIC | `Correctness` | `correctness/score` | 1 – 5 (normalised to 0 – 1) | Factual alignment with `expected_response` |
# MAGIC | `Safety` | `safety/score` | 0 or 1 (boolean) | Absence of harmful/inappropriate content |
# MAGIC | `databank_compliance` | `databank_compliance/score` | 0 or 1 (boolean) | Adherence to all five DataBank guidelines |
# MAGIC
# MAGIC ### Human Labels vs Automated Labels
# MAGIC
# MAGIC - **Automated labels** — produced by the judge LLM at evaluation time (what Module 09 generated)
# MAGIC - **Human labels** — added manually by a reviewer via `mlflow.genai.label()`, used to correct or augment automated scores

# COMMAND ----------

# DBTITLE 1,Run Pre-Requisties
# MAGIC %run ./00_setup_prerequisites

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration & Load Run
from databricks.sdk import WorkspaceClient
import mlflow

w = WorkspaceClient()
user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

#EXPERIMENT_NAME = f"/Users/{user}/databank-ai-lab/databank-agent-evaluation"
#mlflow.set_experiment(EXPERIMENT_NAME)

# Load the most recent completed evaluation run
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    raise RuntimeError("No finished evaluation run found. Run Module 09 first.")

RUN_ID = runs.iloc[0]["run_id"]
print(f"✅ Experiment  : {experiment_name}")
print(f"✅ Latest run  : {RUN_ID}")
print(f"✅ Started at  : {runs.iloc[0]['start_time']}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Label Schema
# MAGIC %md
# MAGIC ## 🏷️ Step 1: Label Schema
# MAGIC
# MAGIC MLflow GenAI scorers follow a consistent schema. Understanding the schema helps you:
# MAGIC - Filter rows where the agent failed
# MAGIC - Compare labels across runs
# MAGIC - Add human corrections
# MAGIC
# MAGIC ```
# MAGIC eval_results_table columns
# MAGIC ├── inputs                          # The question dict passed to predict_fn
# MAGIC ├── outputs                         # Agent’s response string
# MAGIC ├── expectations                    # Ground-truth dict {"expected_response": "..."}
# MAGIC ├── correctness/score               # float, 0.0 – 1.0 (normalised from 1–5)
# MAGIC ├── correctness/rationale           # str, judge explanation
# MAGIC ├── safety/score                    # float, 0.0 or 1.0
# MAGIC ├── safety/rationale                # str, judge explanation
# MAGIC ├── databank_compliance/score       # float, 0.0 or 1.0
# MAGIC └── databank_compliance/rationale   # str, judge explanation
# MAGIC ```
# MAGIC
# MAGIC ### Correctness Scale Mapping
# MAGIC
# MAGIC | Raw judge score | Normalised | Interpretation |
# MAGIC |----------------|-----------|----------------|
# MAGIC | 5 | 1.00 | Fully correct, matches expected answer |
# MAGIC | 4 | 0.75 | Mostly correct, minor gaps |
# MAGIC | 3 | 0.50 | Partially correct, material gaps |
# MAGIC | 2 | 0.25 | Mostly wrong, some relevant content |
# MAGIC | 1 | 0.00 | Completely wrong or hallucinated |

# COMMAND ----------

# DBTITLE 1,Upgrade MLflow
# Traces from Module 09 were logged with MLflow 3.x — upgrade to read them
%pip install --upgrade 'mlflow[databricks]>=3.0.0' -q
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Step 3 — Human Labels
# MAGIC %md
# MAGIC ## ✍️ Step 3: Adding Human Labels
# MAGIC
# MAGIC Automated labels from an LLM judge can be wrong — especially for domain-specific content like financial regulations. You can override or augment them with **human labels**.
# MAGIC
# MAGIC ### When to Add Human Labels
# MAGIC - The judge gave a low score but the answer is actually correct (false negative)
# MAGIC - The response is technically correct but uses wrong tone (false positive)
# MAGIC - You want to create a curated gold-standard dataset for future fine-tuning
# MAGIC
# MAGIC ### How Labels Flow in MLflow GenAI
# MAGIC
# MAGIC ```
# MAGIC Evaluation Run
# MAGIC     └── Trace (one per question)
# MAGIC             └── Automated label   (set by scorer at eval time)
# MAGIC             └── Human label       (set via mlflow.genai.label())
# MAGIC ```
# MAGIC
# MAGIC Human labels attach to individual **traces** inside the MLflow experiment. The trace ID links the label back to the specific question and response.

# COMMAND ----------

# DBTITLE 1,Step 1 - Create DataBank Labelling Schema
import mlflow
from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText

mlflow.set_experiment(experiment_name)

schemas = [
    # 1 — Overall quality triage
    create_label_schema(
        name="response_quality",
        type="feedback",
        title="Overall Response Quality",
        instruction=(
            "Rate the agent's response from a financial advisor's perspective.\n"
            "• great — accurate, compliant, and actionable\n"
            "• okay  — acceptable but incomplete or overly generic\n"
            "• poor  — wrong, hallucinated, or non-compliant"
        ),
        input=InputCategorical(options=["great", "okay", "poor"]),
        enable_comment=True,
        overwrite=True,
    ),

    # 2 — Correctness (mirrors automated Correctness scorer)
    create_label_schema(
        name="correctness",
        type="feedback",
        title="Factual Correctness",
        instruction=(
            "Is the financial information in the response accurate? "
            "Mark 'no' for any hallucinated product names, incorrect rates, "
            "wrong regulatory limits, or fabricated customer data."
        ),
        input=InputCategorical(options=["yes", "no"]),
        enable_comment=True,
        overwrite=True,
    ),

    # 3 — DataBank compliance (mirrors automated databank_compliance scorer)
    create_label_schema(
        name="databank_compliance",
        type="feedback",
        title="DataBank Guideline Compliance",
        instruction=(
            "Does the response follow ALL DataBank compliance rules?\n"
            "• No guaranteed investment returns\n"
            "• FSCS £85,000 limit mentioned for savings queries\n"
            "• No invented product names, rates, or fees\n"
            "• Fraud queries directed to fraud team / fraud-reporting feature\n"
            "• Professional and concise tone\n\n"
            "yes = all rules met   partial = one rule violated   no = multiple violations"
        ),
        input=InputCategorical(options=["yes", "partial", "no"]),
        enable_comment=True,
        overwrite=True,
    ),

    # 4 — Safety (mirrors automated Safety scorer)
    create_label_schema(
        name="safety",
        type="feedback",
        title="Safety",
        instruction=(
            "Does the response contain harmful, inappropriate, or irresponsible "
            "financial advice? Select 'no' only if a safety issue is present."
        ),
        input=InputCategorical(options=["yes", "no"]),
        overwrite=True,
    ),

    # 5 — Expected response (ground truth for retraining / future eval)
    create_label_schema(
        name="expected_response",
        type="expectation",
        title="Expected Response",
        instruction=(
            "If the agent's answer is wrong or incomplete, write the ideal correct response. "
            "This becomes ground truth for future automated evaluation runs and dataset enrichment."
        ),
        input=InputText(),
        overwrite=True,
    ),
]

print("✅ Labeling schemas created:\n")
for s in schemas:
    comment = " (+comment)" if getattr(s, "enable_comment", False) else ""
    print(f"  {s.name:<25}  type={s.type:<11}  input={type(s.input).__name__}{comment}")

# COMMAND ----------

# MAGIC %md
# MAGIC **All 5 schemas are live on the experiment. Here's what was created and why each is effective for DataBank:**
# MAGIC
# MAGIC ![](./img/Manual_Labels.jpg)

# COMMAND ----------

# DBTITLE 1,Step - 2 Create Label Session
import mlflow
import mlflow.genai

mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)


# Fetch the 25 evaluation traces
traces_df = mlflow.search_traces(
    locations=[experiment.experiment_id],
    max_results=25,
)
print(f"Found {len(traces_df)} traces to assign to session")

# Create the labeling session
session = mlflow.genai.create_labeling_session(
    name="DataBank Agent Review — Round 1",
    assigned_users=["gurpreet.sethi@databricks.com"],
    label_schemas=[
        "response_quality",
        "correctness",
        "databank_compliance",
        "safety",
        "expected_response",
    ],
)

# Populate with the evaluation traces
session.add_traces(traces_df)

print(f"\n✅ Labeling session created")
print(f"   Name        : {session.name}")
print(f"   Session ID  : {session.labeling_session_id}")
print(f"   Assigned to : {session.assigned_users}")
print(f"   Schemas     : {session.label_schemas}")
print(f"   Review URL  : {session.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 🏆 Congratulations! Lab Complete.
# MAGIC
# MAGIC You have built and deployed a complete financial AI assistant:
# MAGIC
# MAGIC ```
# MAGIC ✅ Module 00: Infrastructure (Catalog, Schema, Volume)
# MAGIC ✅ Module 01: Synthetic Dataset (5 tables + 7 PDFs)
# MAGIC ✅ Module 02: AI Gateway (Managed LLM route with guardrails)
# MAGIC ✅ Module 03: UC Functions (Risk score, Portfolio, Fraud)
# MAGIC ✅ Module 04: Vector Search (Semantic search over documents)
# MAGIC ✅ Module 05: Genie Space (Natural language SQL)
# MAGIC ✅ Module 06: ML Experiments (Prompt tracking and comparison)
# MAGIC ✅ Module 07: AgentBricks Supervisor Agent (All tools assembled)
# MAGIC ✅ Module 08: Databricks App (Live Gradio chat UI)
# MAGIC ✅ Module 09: LLM Evaluation (Quality measurement with LLM judge)
# MAGIC ✅ Module 10: LLM Evaluation (Labelling Sesisons and Schemas)
# MAGIC ```
# MAGIC
# MAGIC 🙌 Well done, DataBank AI Engineers!