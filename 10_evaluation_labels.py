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

# DBTITLE 1,Step 0 — Configuration & Load Run
from databricks.sdk import WorkspaceClient
import mlflow
import mlflow.genai
import pandas as pd

w = WorkspaceClient()
user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

EXPERIMENT_NAME = f"/Users/{user}/databank-ai-lab/databank-agent-evaluation"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load the most recent completed evaluation run
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    raise RuntimeError("No finished evaluation run found. Run Module 09 first.")

RUN_ID = runs.iloc[0]["run_id"]
print(f"✅ Experiment  : {EXPERIMENT_NAME}")
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

# DBTITLE 1,Step 2 — Inspect Labels per Row
# mlflow.genai.evaluate() stores results as traces, not as artifact tables.
# Reconstruct results_df from the traces logged for this run.
client_mlflow = mlflow.tracking.MlflowClient()

traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    filter_string=f"attributes.run_id = '{RUN_ID}'",
    max_results=100
)

if traces.empty:
    raise RuntimeError("No traces found for this run. Ensure Module 09 ran successfully.")

rows = []
for t in traces.itertuples():
    row = {
        "inputs":  getattr(t, "inputs",  getattr(t, "request",  None)),
        "outputs": getattr(t, "outputs", getattr(t, "response", None)),
    }
    # assessments is a list of dicts with keys: assessment_name, feedback, rationale
    for a in (getattr(t, "assessments", None) or []):
        if not isinstance(a, dict):
            continue
        name     = a.get("assessment_name", "")
        feedback = a.get("feedback")        # present on scored assessments
        if not name or not feedback:
            continue                         # skip expectation entries (no feedback)
        raw = feedback.get("value")
        # Scores arrive as 'yes'/'no' strings; normalise to 1.0 / 0.0
        if isinstance(raw, str):
            value = 1.0 if raw.lower() == "yes" else 0.0
        elif isinstance(raw, bool):
            value = float(raw)
        elif isinstance(raw, (int, float)):
            value = float(raw)
        else:
            value = None
        row[f"{name}/score"]     = value
        row[f"{name}/rationale"] = a.get("rationale")
    rows.append(row)

results_df = pd.DataFrame(rows)

# Normalise column names to lowercase
results_df.columns = [c.lower() for c in results_df.columns]

print(f"✅ Loaded {len(results_df)} labelled rows")
print(f"   Columns: {list(results_df.columns)}")
print()

# Show label columns only
label_cols = [c for c in results_df.columns if any(
    c.startswith(p) for p in ["correctness", "safety", "databank_compliance"]
)]
print("Label columns found:")
for c in sorted(label_cols):
    print(results_df[c].notna().sum())
    #non_null = results_df[c].notna().sum()
    #print(f"  {c:<45}  ({non_null}/{len(results_df)} non-null)")

# COMMAND ----------

# DBTITLE 1,Step 3 — Label Distribution
print("📊 Label Score Distribution")
print("=" * 55)

for scorer in ["correctness", "databank_compliance", "safety"]:
    col = f"{scorer}/score"
    if col not in results_df.columns:
        print(f"  {scorer:<30} ⚠️  column not found")
        continue

    scores = results_df[col].dropna()
    if scores.empty:
        print(f"  {scorer:<30} ⚠️  all null (check expectations format)")
        continue

    mean_val = scores.mean()
    bar      = "█" * int(mean_val * 20)
    print(f"  {scorer:<30} mean={mean_val:.3f}  {bar}")

    # Distribution buckets
    bins = {"0.0": 0, "0.1-0.5": 0, "0.6-0.9": 0, "1.0": 0}
    for v in scores:
        if   v == 0.0:        bins["0.0"]     += 1
        elif v < 0.6:         bins["0.1-0.5"] += 1
        elif v < 1.0:         bins["0.6-0.9"] += 1
        else:                 bins["1.0"]     += 1
    print(f"    {'  '.join(f'{k}: {n}' for k, n in bins.items())}")

print()

# Rows that failed all three automated labels
fail_mask = (
    (results_df.get('correctness/score',       pd.Series(dtype=float)) < 0.5) |
    (results_df.get('databank_compliance/score', pd.Series(dtype=float)) < 1.0) |
    (results_df.get('safety/score',             pd.Series(dtype=float)) < 1.0)
)
print(f"⚠️  Rows with at least one label failure : {fail_mask.sum()} / {len(results_df)}")

# COMMAND ----------

# DBTITLE 1,Step 4 — Failed Rows with Rationale
# Show the rows that scored below threshold, with the judge's rationale
threshold = 0.5
failed_df  = results_df[
    results_df.get('correctness/score', pd.Series(1.0, index=results_df.index)).fillna(1.0) < threshold
].copy()

if failed_df.empty:
    print("✅ No rows scored below threshold — agent is performing well!")
else:
    print(f"🔍 {len(failed_df)} rows scored below {threshold} on Correctness:\n")
    for i, row in failed_df.iterrows():
        q = row.get('inputs', {}) or {}
        question    = q.get('question', str(q))[:120] if isinstance(q, dict) else str(q)[:120]
        score       = row.get('correctness/score', 'n/a')
        rationale   = str(row.get('correctness/rationale', 'n/a'))[:300]
        expected    = str(row.get('expectations', {}) or {})[:150]
        print(f"  [{i}] Score: {score}")
        print(f"       Q: {question}")
        print(f"       Rationale: {rationale}")
        print(f"       Expected : {expected}")
        print()

# COMMAND ----------

# DBTITLE 1,Step 5 — Human Labels
# MAGIC %md
# MAGIC ## ✍️ Step 5: Adding Human Labels
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

# DBTITLE 1,Step 6 — Apply a Human Label to a Trace
# Retrieve traces logged during the evaluation run
traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    filter_string=f"attributes.run_id = '{RUN_ID}'",
    max_results=5
)

if traces.empty:
    print("⚠️  No traces found for this run. Make sure Module 09 ran with tracing enabled.")
else:
    print(f"Found {len(traces)} traces for run {RUN_ID}")
    print()

    # Show the first trace so you can decide whether to label it
    first_trace = traces.iloc[0]
    print(f"Trace ID  : {first_trace['trace_id']}")
    print(f"Status    : {first_trace['state']}")
    print()

    # Apply a human correctness label to the first trace
    # score: 1 = correct, 0 = incorrect
    # rationale: your review comment
    TRACE_ID   = first_trace['trace_id']
    SCORE      = True          # True = pass, False = fail
    RATIONALE  = "Reviewed manually: response correctly cites FSCS £85,000 limit and is appropriate for the advisor context."

    mlflow.log_feedback(
        trace_id=TRACE_ID,
        name="correctness",
        value=SCORE,
        rationale=RATIONALE
    )
    print(f"✅ Human label applied to trace {TRACE_ID}")
    print(f"   Score     : {SCORE}")
    print(f"   Rationale : {RATIONALE}")

# COMMAND ----------

# DBTITLE 1,Step 7 — Compare Automated vs Human Labels
# Re-load traces and show both automated and human labels side-by-side
traces_with_labels = mlflow.search_traces(
    locations=[experiment.experiment_id],
    filter_string=f"attributes.run_id = '{RUN_ID}'",
    max_results=25
)

rows = []
for t in traces_with_labels.itertuples():
    automated = None
    human     = None
    for a in (getattr(t, 'assessments', None) or []):
        if not isinstance(a, dict):
            continue
        name   = a.get('assessment_name', '')
        source = a.get('source', {})
        raw    = (a.get('feedback') or {}).get('value')
        value  = 1.0 if str(raw).lower() == 'yes' else (0.0 if raw is not None else None)
        if name == 'correctness':
            if source.get('source_type', '').upper() == 'HUMAN':
                human = value
            else:
                automated = value
    rows.append({
        'trace_id':              getattr(t, 'trace_id', '')[:16] + '...',
        'automated_correctness': automated,
        'human_correctness':     human
    })

df_comparison = pd.DataFrame(rows)
display(df_comparison)

# Agreement rate (where both exist)
both = df_comparison.dropna(subset=['automated_correctness', 'human_correctness'])
if not both.empty:
    agreement = (both['automated_correctness'] == both['human_correctness']).mean()
    print(f"\nAutomated ↔ Human agreement rate: {agreement:.1%} ({len(both)} labelled rows)")
else:
    print("\nNo rows with both automated and human labels yet.")

# COMMAND ----------

# DBTITLE 1,Step 8 - Create DataBank Labelling Schema
import mlflow
from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText

mlflow.set_experiment("/Users/gurpreet.sethi@databricks.com/databank-ai-lab/databank-agent-evaluation")

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

# DBTITLE 1,Step - 9 Create Label Session
import mlflow
import mlflow.genai

mlflow.set_experiment("/Users/gurpreet.sethi@databricks.com/databank-ai-lab/databank-agent-evaluation")
experiment = mlflow.get_experiment_by_name(
    "/Users/gurpreet.sethi@databricks.com/databank-ai-lab/databank-agent-evaluation"
)

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