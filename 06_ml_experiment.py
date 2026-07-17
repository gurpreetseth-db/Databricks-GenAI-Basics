# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# DBTITLE 1,Module 06 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 06: ML Experiment Tracking
# MAGIC **Duration:** ~20 minutes | **Prerequisite:** Module 00
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Why Track LLM Experiments?
# MAGIC
# MAGIC Building an AI agent is an **iterative process**. You will test many prompt variations:
# MAGIC - Different system prompt styles (formal vs conversational)
# MAGIC - Different levels of instruction detail
# MAGIC - Different models (Llama 3.3 70B vs 3.1 405B)
# MAGIC - Different temperature settings
# MAGIC
# MAGIC Without tracking, you lose visibility into what worked and what didn’t.
# MAGIC **MLflow Experiments** solve this by logging every run with its inputs, outputs, and metrics.
# MAGIC
# MAGIC ### MLflow Core Concepts
# MAGIC
# MAGIC | Concept | Description |
# MAGIC |---------|-------------|
# MAGIC | **Experiment** | A named container grouping all related runs |
# MAGIC | **Run** | One specific trial (a unique combination of parameters) |
# MAGIC | **Parameter** | Input setting logged per run (model name, temperature, prompt version) |
# MAGIC | **Metric** | Measurable output (response length, latency, quality score) |
# MAGIC | **Artifact** | Any file logged with the run (prompt text, response JSON, config) |
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC - An MLflow experiment: `databank-prompt-experiments`
# MAGIC - 6 runs comparing different system prompt strategies
# MAGIC - Latency and response quality logged per run
# MAGIC - Identify the best prompt configuration for the DataBank agent

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration & Imports
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


SCHEMA           = "financial_data"
FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
AI_GW_ROUTE      = "databank-llm-route"
EXPERIMENT_NAME  = f"/Users/{user}/databank-ai-lab/databank-prompt-experiments"

# ================================================================
# IMPORTS
# ================================================================
import mlflow
import mlflow.data
from mlflow.models import infer_signature
from openai import OpenAI
from databricks.sdk import WorkspaceClient
import time
import json

w = WorkspaceClient()

# Use AI Gateway route (Module 02) or fall back to Foundation Models directly
try:
    w.serving_endpoints.get(name=AI_GW_ROUTE)
    ACTIVE_MODEL = AI_GW_ROUTE
    print(f"✅ Using AI Gateway route: {AI_GW_ROUTE}")
except Exception:
    ACTIVE_MODEL = FOUNDATION_MODEL
    print(f"⚠️  AI Gateway not found, using Foundation Models directly: {FOUNDATION_MODEL}")

_token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")

client = OpenAI(
    api_key=_token,
    base_url=f"{w.config.host}/serving-endpoints"
)

print(f"\n📈 MLflow Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Create MLflow Experiment
# Create or get experiment
# Using a personal folder path following workspace policy
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

print(f"✅ Experiment: {EXPERIMENT_NAME}")
print(f"   Experiment ID : {experiment.experiment_id if experiment else 'creating...'}")
print(f"   Lifecycle     : {experiment.lifecycle_stage if experiment else 'active'}")
print()
print("💡 Open the MLflow Experiments UI to see your runs:")
print(f"   Left sidebar → Experiments → {EXPERIMENT_NAME.split('/')[-1]}")

# COMMAND ----------

# DBTITLE 1,Prompt Variants to Compare
# MAGIC %md
# MAGIC ## 🔍 Prompt Variants
# MAGIC
# MAGIC We will test **3 prompt strategies** with **2 temperature settings** = 6 runs total.
# MAGIC
# MAGIC | Variant | Strategy | Description |
# MAGIC |---------|----------|-------------|
# MAGIC | `minimal` | Short, direct | Minimal instruction, rely on LLM defaults |
# MAGIC | `detailed` | Comprehensive | Full DataBank context, rules, product knowledge |
# MAGIC | `role_based` | Persona-driven | Assigns a specific advisor persona with values |
# MAGIC
# MAGIC For each prompt, we test the same **3 financial questions** and log:
# MAGIC - Response latency (ms)
# MAGIC - Response length (tokens)
# MAGIC - A simple relevance heuristic (does the answer mention key financial terms?)

# COMMAND ----------

# DBTITLE 1,Step 2 — Define Prompts and Test Questions
# The 3 prompt strategies to compare
PROMPT_VARIANTS = {
    "minimal": "You are a helpful financial advisor.",

    "detailed": """
You are a DataBank financial advisor assistant with expertise in:
- Savings products: ISAs, Fixed-Rate Bonds, Premium Savings
- Lending: Personal Loans, Business Loans, Debt Consolidation
- Investments: Managed Portfolios, Stocks & Shares ISA, Ethical Funds
- Insurance: Life, Critical Illness, Income Protection
- Credit Cards: Rewards, Balance Transfer, Travel

Rules:
1. Always consider a customer's risk profile (Conservative/Moderate/Aggressive) before recommending products.
2. Never recommend high-risk investment products to Conservative customers.
3. Always mention FSCS protection for savings products (£85,000 limit).
4. Be concise but thorough. Use bullet points for product comparisons.
5. Remind customers that investments carry capital risk.
""",

    "role_based": """
You are ALEX, DataBank's Senior Financial Advisor with 15 years of experience.
You are warm, professional, and always put the customer's financial wellbeing first.
You speak in plain English (no jargon), give specific advice, and always explain WHY.
When you don't know something, you say so honestly.
Your goal: help DataBank customers make confident financial decisions.
"""
}

# Test questions that represent real advisor queries
TEST_QUESTIONS = [
    "A 45-year-old customer with a moderate risk profile wants to invest £20,000. What do you recommend?",
    "What is the fastest way for a customer to get a £10,000 personal loan from DataBank?",
    "A customer is worried about protecting their family if they become critically ill. What should they consider?"
]

print(f"💡 {len(PROMPT_VARIANTS)} prompt variants × {len(TEST_QUESTIONS)} questions × 2 temperatures = 12 total calls")
for name, prompt in PROMPT_VARIANTS.items():
    print(f"   {name}: {len(prompt)} chars")

# COMMAND ----------

# DBTITLE 1,Step 3 — Run Experiments
from openai import BadRequestError, RateLimitError, InternalServerError

def run_experiment(prompt_name: str, system_prompt: str, question: str, temperature: float) -> dict:
    """Run one LLM call and return metrics."""
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question}
            ],
            temperature=temperature,
            max_tokens=400
        )
        latency_ms = int((time.time() - start) * 1000)
        answer = response.choices[0].message.content

        # Simple heuristic: does the answer contain financial keywords?
        financial_terms = ["rate", "risk", "protect", "invest", "savings", "tax",
                           "annual", "income", "capital", "interest", "premium"]
        relevance_score = sum(1 for term in financial_terms if term.lower() in answer.lower())

        return {
            "answer":            answer,
            "latency_ms":        latency_ms,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
            "relevance_score":   relevance_score
        }
    except BadRequestError:
        # AI Gateway guardrail triggered — log and continue so the run completes
        latency_ms = int((time.time() - start) * 1000)
        print(f"  ⚠️  Guardrail triggered ({prompt_name}, temp={temperature}): {question[:60]}...")
        return {
            "answer":            "[guardrail_triggered]",
            "latency_ms":        latency_ms,
            "completion_tokens": 0,
            "total_tokens":      0,
            "relevance_score":   0
        }
    except InternalServerError:
        print(f"  ⏳ Server error — waiting 15s before retry...")
        time.sleep(15)
        try:
            response = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question}
                ],
                temperature=temperature,
                max_tokens=400
            )
            latency_ms = int((time.time() - start) * 1000)
            answer = response.choices[0].message.content
            financial_terms = ["rate", "risk", "protect", "invest", "savings", "tax",
                               "annual", "income", "capital", "interest", "premium"]
            relevance_score = sum(1 for term in financial_terms if term.lower() in answer.lower())
            return {
                "answer":            answer,
                "latency_ms":        latency_ms,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
                "relevance_score":   relevance_score
            }
        except Exception:
            latency_ms = int((time.time() - start) * 1000)
            print(f"  ⚠️  Retry also failed — skipping question.")
            return {
                "answer":            "[server_error]",
                "latency_ms":        latency_ms,
                "completion_tokens": 0,
                "total_tokens":      0,
                "relevance_score":   0
            }
    except RateLimitError:
        # TPM limit hit — wait 60s and retry once
        print(f"  ⏳ Rate limit hit — waiting 60s before retry...")
        time.sleep(60)
        try:
            response = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question}
                ],
                temperature=temperature,
                max_tokens=400
            )
            latency_ms = int((time.time() - start) * 1000)
            answer = response.choices[0].message.content
            financial_terms = ["rate", "risk", "protect", "invest", "savings", "tax",
                               "annual", "income", "capital", "interest", "premium"]
            relevance_score = sum(1 for term in financial_terms if term.lower() in answer.lower())
            return {
                "answer":            answer,
                "latency_ms":        latency_ms,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
                "relevance_score":   relevance_score
            }
        except Exception:
            latency_ms = int((time.time() - start) * 1000)
            print(f"  ⚠️  Retry also failed — skipping question.")
            return {
                "answer":            "[rate_limit]",
                "latency_ms":        latency_ms,
                "completion_tokens": 0,
                "total_tokens":      0,
                "relevance_score":   0
            }


TEMPERATURES = [0.0, 0.7]  # deterministic vs creative
run_count = 0

mlflow.set_experiment(EXPERIMENT_NAME)

for temp in TEMPERATURES:
    for prompt_name, system_prompt in PROMPT_VARIANTS.items():
        # One MLflow run per (temperature, prompt_variant)
        with mlflow.start_run(run_name=f"{prompt_name}_temp{temp}"):

            # Log parameters
            mlflow.log_param("prompt_variant",  prompt_name)
            mlflow.log_param("temperature",     temp)
            mlflow.log_param("model",           ACTIVE_MODEL)
            mlflow.log_param("system_prompt_len", len(system_prompt))

            # Log the prompt text as an artifact
            mlflow.log_text(system_prompt, artifact_file="system_prompt.txt")

            # Run all test questions and aggregate metrics
            all_latencies = []
            all_relevance = []

            for i, question in enumerate(TEST_QUESTIONS):
                if i > 0:
                    time.sleep(5)  # small delay between calls to stay within TPM limit
                result = run_experiment(prompt_name, system_prompt, question, temp)

                # Log per-question metrics
                mlflow.log_metric(f"latency_ms_q{i+1}",  result["latency_ms"])
                mlflow.log_metric(f"tokens_q{i+1}",       result["completion_tokens"])
                mlflow.log_metric(f"relevance_q{i+1}",    result["relevance_score"])

                # Log Q&A as artifact
                qa = {"question": question, "answer": result["answer"],
                      "latency_ms": result["latency_ms"]}
                mlflow.log_dict(qa, artifact_file=f"q{i+1}_result.json")

                all_latencies.append(result["latency_ms"])
                all_relevance.append(result["relevance_score"])

            # Log aggregate metrics
            mlflow.log_metric("avg_latency_ms",   sum(all_latencies) / len(all_latencies))
            mlflow.log_metric("avg_relevance",    sum(all_relevance) / len(all_relevance))
            mlflow.log_metric("max_relevance",    max(all_relevance))

            run_count += 1
            print(f"  [{run_count}] {prompt_name} | temp={temp} | "
                  f"avg_latency={sum(all_latencies)//len(all_latencies)}ms | "
                  f"avg_relevance={sum(all_relevance)/len(all_relevance):.1f}")

print(f"\n✅ {run_count} experiment runs completed")
print("   Open Experiments in the sidebar to compare runs visually.")

# COMMAND ----------

# DBTITLE 1,Step 4 — Find and Display Best Run
# Find the best run by highest average relevance score (then lowest latency)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.avg_relevance DESC", "metrics.avg_latency_ms ASC"]
)

if len(runs) == 0:
    print("❌ No runs found. Run Step 3 first.")
else:
    print("🏆 Experiment Results (ranked by relevance score):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Run Name':<30} {'Avg Relevance':<15} {'Avg Latency (ms)':<18}")
    print("-" * 70)
    for i, (_, row) in enumerate(runs.iterrows(), 1):
        print(f"{i:<5} {row['tags.mlflow.runName']:<30} "
              f"{row.get('metrics.avg_relevance', 0):<15.2f} "
              f"{row.get('metrics.avg_latency_ms', 0):<18.0f}")

    best = runs.iloc[0]
    print()
    print(f"✅ Best run : {best['tags.mlflow.runName']}")
    print(f"   Prompt variant : {best.get('params.prompt_variant', 'N/A')}")
    print(f"   Temperature    : {best.get('params.temperature', 'N/A')}")
    print(f"   Avg relevance  : {best.get('metrics.avg_relevance', 0):.2f}")
    print(f"   Avg latency    : {best.get('metrics.avg_latency_ms', 0):.0f} ms")
    print()
    print("💡 The 'detailed' prompt with temperature=0.0 typically scores best for factual")
    print("   financial advice. We will use this configuration in Module 07.")

# COMMAND ----------

# DBTITLE 1,Module 06 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 06 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | Experiment `databank-prompt-experiments` created | Visible in Experiments sidebar |
# MAGIC | 6 runs logged | 3 prompt variants × 2 temperatures |
# MAGIC | Best run identified | `detailed` prompt at temperature 0.0 (typically) |
# MAGIC | Artifacts logged | system_prompt.txt + q1/q2/q3 results in each run |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 What to Observe in the MLflow UI
# MAGIC 1. Click **Experiments** → `databank-prompt-experiments`
# MAGIC 2. Select all 6 runs and click **Compare**
# MAGIC 3. Look at the bar chart of `avg_relevance` — which prompt wins?
# MAGIC 4. Check latency — does the longer `detailed` prompt add meaningful overhead?
# MAGIC 5. Open a run and click **Artifacts** to read the full Q&A responses
# MAGIC
# MAGIC ### 🚀 Next: Module 07 — AgentBricks
# MAGIC Open **`07_agentbricks_agent`** to assemble the DataBank AI Advisor agent using all the components built so far.