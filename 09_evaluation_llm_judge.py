# Databricks notebook source


# COMMAND ----------

# DBTITLE 1,Module 09 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 09: LLM Evaluation
# MAGIC **Duration:** ~20 minutes | **Prerequisite:** Module 07 (agent endpoint)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is LLM-as-a-Judge?
# MAGIC
# MAGIC **LLM-as-a-Judge** uses a powerful LLM to evaluate the outputs of your AI system — just like a human reviewer would, but at scale.
# MAGIC
# MAGIC The judge LLM reads:
# MAGIC - The **input question**
# MAGIC - The **expected answer** (ground truth)
# MAGIC - The **agent’s actual response**
# MAGIC
# MAGIC And scores the response on criteria like accuracy, helpfulness, safety, and groundedness.
# MAGIC
# MAGIC ### Why Does This Matter?
# MAGIC Before deploying an AI agent to production, you need confidence it will:
# MAGIC - Answer financial questions **accurately** (no hallucinated interest rates)
# MAGIC - Stay **grounded in facts** (not invent products that don’t exist)
# MAGIC - Remain **safe** (no harmful financial advice)
# MAGIC - Be **helpful** (give actionable, clear answers)
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC 1. A **gold-standard evaluation dataset** (25 Q&A pairs covering all agent tools)
# MAGIC 2. **Custom scorers** using MLflow’s GenAI evaluation framework
# MAGIC 3. **Run evaluation** against the deployed agent endpoint
# MAGIC 4. **Analyse results** to identify gaps in agent quality

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration & Imports
# ================================================================
# CONFIGURATION
# ================================================================
user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

CATALOG          = "databank_lab"
SCHEMA           = "financial_data"
AGENT_ENDPOINT   = "databank-ai-advisor"   # From Module 07
FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # Used as judge
EXPERIMENT_NAME  = f"/Users/{user}/databank-ai-lab/databank-agent-evaluation"


def get_superagent_endpoint_name(superagent_name):
    _w = WorkspaceClient()
    normalise = lambda s: s.lower().replace("-", "").replace("_", "").replace(" ", "")
    all_eps   = list(_w.serving_endpoints.list())
    
    # 1. Exact name match
    endpoint_name = next((ep.name for ep in all_eps if ep.name == superagent_name), None)
    
    # 2. Fuzzy match (normalised agent name appears inside endpoint name
    if not endpoint_name:
        term = normalise(superagent_name)
        endpoint_name = next((ep.name for ep in all_eps if term in normalise(ep.name)), None)
    
    # 3. AgentBricks fallback — supervisor agent endpoints are always named mas-<uuid>-endpoint
    if not endpoint_name:
        mas_eps = [ep.name for ep in all_eps
            if ep.name.startswith("mas-") and ep.name.endswith("-endpoint")]
    if len(mas_eps) == 1:
        endpoint_name = mas_eps[0]
    elif len(mas_eps) > 1:
        print(f"⚠️  Multiple AgentBricks supervisor endpoints found.")
        print(f"   Update superagent_name in cell 5 to one of:")
        for n in mas_eps:
            print(f"   {n}")

    if endpoint_name:
        ENDPOINTNAME = endpoint_name          # available for downstream cells
        return ENDPOINTNAME
    else:
        print(f"⚠️  Could not resolve endpoint for '{AGENT_ENDPOINT}'")

endpoint_name= get_superagent_endpoint_name(AGENT_ENDPOINT)

# ================================================================
# IMPORTS
# ================================================================
import mlflow
import mlflow.genai
from mlflow.genai.scorers import RetrievalGroundedness, Guidelines, Safety
from openai import OpenAI
from databricks.sdk import WorkspaceClient
import pandas as pd
import json, time

w = WorkspaceClient()
client = OpenAI(
    api_key=w.config.authenticate().get("Authorization", "").replace("Bearer ", ""),
    base_url=f"{w.config.host}/serving-endpoints"
)

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✅ Evaluation experiment: {EXPERIMENT_NAME}")
print(f"🤖 Agent endpoint  : {endpoint_name}")
print(f"⚖️  Judge model     : {FOUNDATION_MODEL}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Build Evaluation Dataset
# MAGIC %md
# MAGIC ## 📝 Step 1: Build the Evaluation Dataset
# MAGIC
# MAGIC A good evaluation dataset covers **all the agent’s capabilities**. We create 25 Q&A pairs:
# MAGIC
# MAGIC | Category | Questions | What it Tests |
# MAGIC |----------|-----------|---------------|
# MAGIC | Product knowledge | 8 | Knowledge Assistant (document RAG) |
# MAGIC | Customer data | 6 | Genie Space (NL-to-SQL) |
# MAGIC | Risk scoring | 4 | UC Function: calculate_customer_risk |
# MAGIC | Portfolio review | 4 | UC Function: get_portfolio_summary |
# MAGIC | Fraud detection | 3 | UC Function: flag_suspicious_transactions |
# MAGIC
# MAGIC Each row has:
# MAGIC - `inputs`: The user’s question
# MAGIC - `expected_response`: The gold-standard correct answer (written by a domain expert)
# MAGIC - `category`: Which tool was expected to be used

# COMMAND ----------

# DBTITLE 1,Step 1 — Create Evaluation Dataset
# Evaluation dataset: 25 curated Q&A pairs for the DataBank AI Advisor
# expected_response = the ideal answer we expect from the agent
# These serve as ground truth for the LLM judge

eval_data = [
    # --- Product Knowledge (document RAG) ---
    {"inputs": "What is the current AER on the DataBank Fixed-Rate Bond for a 3-year term?",
     "expected_response": "The DataBank Fixed-Rate Bond offers 5.20% AER for the 3-year term. This is a fixed rate guaranteed for the full term. Minimum deposit is £1,000. No withdrawals are permitted during the fixed term.",
     "category": "product_knowledge"},

    {"inputs": "What is the annual ISA allowance for the DataBank Cash ISA?",
     "expected_response": "The annual ISA allowance for the DataBank Cash ISA is £20,000 per tax year (2024/25). The account offers 3.80% AER variable, instant access, and is FSCS protected up to £85,000.",
     "category": "product_knowledge"},

    {"inputs": "Does DataBank charge an annual fee for the Rewards Credit Card?",
     "expected_response": "Yes, the DataBank Rewards Credit Card has an annual fee of £20, which is waived in the first year. It offers 1% cashback on all purchases and 2% on travel and dining.",
     "category": "product_knowledge"},

    {"inputs": "What is the FSCS protection limit for DataBank savings accounts?",
     "expected_response": "DataBank savings accounts are protected by the Financial Services Compensation Scheme (FSCS) up to £85,000 per person per institution.",
     "category": "product_knowledge"},

    {"inputs": "Can a customer withdraw money from a Fixed-Rate Bond before the term ends?",
     "expected_response": "No. Withdrawals are not permitted during the fixed term of a DataBank Fixed-Rate Bond. At maturity, the funds are transferred to a nominated account or reinvested at the prevailing rate.",
     "category": "product_knowledge"},

    {"inputs": "What are the deferred period options for DataBank Income Protection insurance?",
     "expected_response": "DataBank Income Protection offers deferred period options of 4, 8, 13, or 26 weeks. The monthly benefit pays up to 60% of gross monthly salary and continues to age 65 or until return to work.",
     "category": "product_knowledge"},

    {"inputs": "What is the representative APR on a DataBank Personal Loan for £10,000?",
     "expected_response": "For a £10,000 personal loan, the representative APR is 6.9% (for amounts between £7,500 and £25,000). There is no arrangement fee and no early repayment charge after month 6.",
     "category": "product_knowledge"},

    {"inputs": "Does the DataBank Travel Credit Card charge fees for foreign transactions?",
     "expected_response": "No. The DataBank Travel Credit Card charges no foreign transaction fees worldwide. It also provides free travel insurance when the trip is paid by card and no ATM fees at DataBank ATMs abroad.",
     "category": "product_knowledge"},

    # --- Customer Data (Genie Space) ---
    {"inputs": "How many DataBank customers have a Conservative risk profile?",
     "expected_response": "Approximately 35% of DataBank customers (around 175 out of 500) have a Conservative risk profile. You can query the exact figure using the customer database.",
     "category": "customer_data"},

    {"inputs": "What product types are most common across DataBank accounts?",
     "expected_response": "DataBank offers products across 5 types: Savings, Loan, Investment, Insurance, and CreditCard. Account distribution depends on the current database state. Savings and CreditCard accounts are typically the most common.",
     "category": "customer_data"},

    {"inputs": "What is the most frequent transaction category in the DataBank transaction history?",
     "expected_response": "The most frequent transaction categories in the DataBank dataset are typically Groceries, Dining, and Shopping, as these reflect everyday consumer spending patterns.",
     "category": "customer_data"},

    {"inputs": "How many transactions in the last 30 days were flagged as fraudulent?",
     "expected_response": "The DataBank transaction dataset contains approximately 2-4% fraud-flagged transactions. The exact count for the last 30 days can be retrieved by querying the transactions table with is_fraud=true and filtering by date.",
     "category": "customer_data"},

    {"inputs": "How many high priority support tickets are currently open?",
     "expected_response": "You can find the count of high-priority open support tickets by querying the support_tickets table where priority='High' and ticket_status in ('Open', 'In Progress').",
     "category": "customer_data"},

    {"inputs": "Which transaction channel is used most often by DataBank customers?",
     "expected_response": "Based on the transaction data, card payments are the most common channel for DataBank customers, accounting for the majority of debit transactions.",
     "category": "customer_data"},

    # --- Risk Scoring ---
    {"inputs": "What does a risk score of 75 mean for a DataBank customer?",
     "expected_response": "A risk score of 75 indicates an Aggressive risk profile. This customer has a high risk tolerance and is suited for higher-growth investment products such as the Global Growth Fund or Stocks & Shares ISA. Always confirm suitability before recommending.",
     "category": "risk_scoring"},

    {"inputs": "What investment products should NOT be recommended to a Conservative customer?",
     "expected_response": "Conservative customers (risk score 20-40) should not be recommended high-volatility products such as the Global Growth Fund or 100% equity portfolios. Suitable products include Cash ISA, Fixed-Rate Bond, Premium Savings, and the Cautious Managed Portfolio.",
     "category": "risk_scoring"},

    {"inputs": "How is a customer's risk score calculated at DataBank?",
     "expected_response": "The DataBank risk score is calculated from: stated risk profile (Conservative=20, Moderate=50, Aggressive=75) plus an income adjustment (+/-10 points based on income vs median of £37,000) minus 5 points for customers over age 60.",
     "category": "risk_scoring"},

    {"inputs": "A customer is 62 years old with a Moderate risk profile and income of £45,000. What is their risk score?",
     "expected_response": "Score = 50 (Moderate) + 1 (income above median) - 5 (age over 60) = approximately 46. This places them in the moderate-conservative range, suitable for balanced investment strategies.",
     "category": "risk_scoring"},

    # --- Portfolio Review ---
    {"inputs": "What information is included in a DataBank portfolio summary?",
     "expected_response": "A DataBank portfolio summary includes: customer name and ID, risk profile, membership date, a list of all accounts with product names and types, current balances, account status (Active/Dormant/Closed), total assets under management, and number of products held.",
     "category": "portfolio_review"},

    {"inputs": "A customer has 3 accounts: a Basic Savings (£2,500), a Stocks ISA (£18,000), and a Personal Loan (£8,000 outstanding). What is their net asset value?",
     "expected_response": "Net asset value = £2,500 (savings) + £18,000 (investment) - £8,000 (loan liability) = £12,500 net. Note: the loan balance is a liability, not an asset.",
     "category": "portfolio_review"},

    {"inputs": "How often should a financial advisor review a customer's portfolio?",
     "expected_response": "DataBank recommends annual portfolio reviews for all customers, with more frequent reviews (quarterly) for customers holding Managed Portfolios. Reviews should be triggered whenever the customer's circumstances change significantly (income, risk profile, life events).",
     "category": "portfolio_review"},

    {"inputs": "What does a Dormant account status mean?",
     "expected_response": "A Dormant status means the account has had no recent customer-initiated activity. DataBank may require the customer to re-activate the account. Interest continues to accrue on savings accounts in dormant status.",
     "category": "portfolio_review"},

    # --- Fraud Detection ---
    {"inputs": "What transaction patterns does DataBank flag as suspicious?",
     "expected_response": "DataBank flags transactions as suspicious when: (1) they are explicitly marked as fraudulent (is_fraud=true), (2) the amount exceeds £1,000 in a single transaction, or (3) multiple high-value transactions occur in quick succession. The fraud_and_anomaly_check tool covers all these patterns.",
     "category": "fraud_detection"},

    {"inputs": "A customer says their card was used abroad without their knowledge. What is the correct process?",
     "expected_response": "1. Run a fraud check to identify the suspicious transactions. 2. Advise the customer to call the 24/7 fraud line (0800 123 4567) or use the DataBank app Report Fraud feature. 3. The card will be frozen and investigated within 3 business days. 4. Under Payment Services Regulations, the customer is not liable for genuinely unauthorised transactions. Refund within 5 business days of confirmed fraud.",
     "category": "fraud_detection"},

    {"inputs": "Is a customer liable for fraudulent transactions on their DataBank card?",
     "expected_response": "Under the Payment Services Regulations, DataBank customers are NOT liable for unauthorised transactions unless they acted fraudulently or with gross negligence. DataBank aims to issue refunds within 5 business days of a confirmed fraud report.",
     "category": "fraud_detection"},
]

df_eval = pd.DataFrame(eval_data)
print(f"✅ Evaluation dataset: {len(df_eval)} rows")
print(f"   Categories: {df_eval['category'].value_counts().to_dict()}")
display(df_eval.head())

# COMMAND ----------

# DBTITLE 1,Step 2 — Define Agent Wrapper for Evaluation
# MLflow genai.evaluate() requires a callable that accepts a single question string
# and returns the agent's response string.

import requests as _requests

def agent_predict(question: str) -> str:
    """
    Wrapper function that calls the DataBank AI Advisor agent.
    Used by mlflow.genai.evaluate() to get predictions for each eval row.
    AgentBricks endpoints require 'input' (not 'messages'), so we call
    /invocations directly instead of using the OpenAI client.
    """
    try:
        token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")
        resp = _requests.post(
            f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"input": [{"role": "user", "content": question}]},
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract final answer from AgentBricks event stream
        for event in reversed(data.get("output", [])):
            if event.get("type") == "message" and event.get("role") == "assistant":
                for block in event.get("content", []):
                    if block.get("type") == "output_text":
                        return block.get("text", "").strip()
        # Fallback to standard OpenAI format
        return (data.get("choices", [{}])[0].get("message", {}).get("content")
                or str(data))
    except Exception as e:
        # Return error string — will score poorly, which is intentional
        return f"[ERROR: {str(e)[:200]}]"

# Quick smoke test
print("🧩 Quick smoke test:")
test_q = "What is the FSCS protection limit for DataBank savings accounts?"
test_resp = agent_predict(test_q)
print(f"   Q: {test_q}")
print(f"   A: {test_resp[:200]}...")
print()
print("✅ Agent wrapper is working")

# COMMAND ----------

# DBTITLE 1,Step 3 — Define Evaluation Scorers
# MAGIC %md
# MAGIC ## ⚖️ Step 3: Define Evaluation Scorers
# MAGIC
# MAGIC We use **3 complementary scorers**:
# MAGIC
# MAGIC | Scorer | What it Checks | Judge Model |
# MAGIC |--------|----------------|-------------|
# MAGIC | **Guidelines** | Does the response follow DataBank-specific rules? | Llama 3.3 70B |
# MAGIC | **Safety** | Is the response free from harmful content? | Llama 3.3 70B |
# MAGIC | **Correctness** | Is the response factually accurate vs the expected answer? | Llama 3.3 70B |
# MAGIC
# MAGIC **Custom Guidelines** for DataBank:
# MAGIC 1. Never give specific investment returns guarantees — always say “capital at risk”
# MAGIC 2. Always mention FSCS protection (£85,000 limit) when discussing savings
# MAGIC 3. Do not hallucinate interest rates or fees not in the product documentation
# MAGIC 4. For fraud queries, always advise contacting the 24/7 fraud line

# COMMAND ----------

# DBTITLE 1,Step 3 — Define and Run Evaluation
from mlflow.genai.scorers import Guidelines, Safety, Correctness

# Custom DataBank-specific guidelines scorer
databank_guidelines = Guidelines(
    name="databank_compliance",
    guidelines=[
        "The response must never guarantee specific investment returns or imply capital is safe for investment products.",
        "For any savings product response, the response should mention FSCS protection or the £85,000 limit.",
        "The response should not invent product names, interest rates, or fees that were not mentioned in the question or the expected answer.",
        "For fraud or suspicious transaction queries, the response should advise the customer to contact the fraud team or use the fraud reporting feature.",
        "The response should be professional, concise, and appropriate for a financial advisory context."
    ]
)

# Correctness: measures factual alignment with expected_response
correctness = Correctness()

# Safety: checks for harmful financial advice or inappropriate content
safety = Safety()

print("✅ Scorers configured:")
print("   • databank_compliance (custom guidelines)")
print("   • correctness (factual accuracy vs expected answer)")
print("   • safety (harmful content detection)")
print()
# mlflow.genai.evaluate requires 'inputs' column as dicts, not plain strings.
df_eval_mlflow = df_eval.copy()
df_eval_mlflow['inputs'] = df_eval_mlflow['inputs'].apply(lambda q: {"question": q})
df_eval_mlflow['expectations'] = df_eval_mlflow['expected_response'].apply(lambda a: {"expected_response": a})

def _mlflow_predict(question):
    return agent_predict(question)

print(f"🏃 Running evaluation on {len(df_eval_mlflow)} questions...")
print("   Using Llama 3.3 70B as judge. Takes ~3-5 minutes.")
print()

# ----------------------------------------------------------------
# Run MLflow GenAI Evaluation
# ----------------------------------------------------------------
with mlflow.start_run(run_name="databank-agent-eval-v1"):
    # Log dataset info
    mlflow.log_param("agent_endpoint",    endpoint_name)
    mlflow.log_param("judge_model",       FOUNDATION_MODEL)
    mlflow.log_param("num_eval_questions", len(df_eval_mlflow))

    # Run evaluation
    eval_results = mlflow.genai.evaluate(
        data=df_eval_mlflow,
        predict_fn=_mlflow_predict,
        scorers=[databank_guidelines, correctness, safety]
    )

print("\n✅ Evaluation complete!")
print(f"   Results logged to experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 4 — Analyse Results
# Display summary metrics
if eval_results and hasattr(eval_results, 'metrics'):
    metrics = eval_results.metrics
    print("\n📊 Evaluation Summary Metrics:")
    print("=" * 55)
    for metric_name, value in sorted(metrics.items()):
        bar = "█" * int(value * 20) if 0 <= value <= 1 else ""
        print(f"  {metric_name:<35} {value:.3f}  {bar}")

# Display per-question results
if eval_results and hasattr(eval_results, 'tables'):
    results_df = eval_results.tables.get('eval_results_table')
    if results_df is not None:
        print("\n📝 Per-Question Results (bottom 5 by correctness):")
        display(results_df.sort_values('correctness/score').head(5)[
            ['inputs', 'outputs', 'expected_response', 'correctness/score',
             'databank_compliance/score', 'category']
        ])

# COMMAND ----------

# DBTITLE 1,Step 5 — Score by Category
# Analyse performance by category to identify which tool needs improvement
if eval_results and hasattr(eval_results, 'tables'):
    results_df = eval_results.tables.get('eval_results_table')
    if results_df is not None and 'category' in results_df.columns:
        category_summary = results_df.groupby('category').agg(
            correctness=("correctness/score", "mean"),
            compliance=("databank_compliance/score", "mean"),
            count=("inputs", "count")
        ).round(3)

        print("📊 Performance by Category:")
        print("-" * 60)
        print(f"{'Category':<25} {'Correctness':<15} {'Compliance':<12} {'N'}")
        print("-" * 60)
        for cat, row in category_summary.iterrows():
            correctness_bar = "█" * int(row.correctness * 10)
            print(f"  {cat:<23} {row.correctness:<15.2f} {row.compliance:<12.2f} {int(row['count'])}")

        # Find the weakest category
        weakest = category_summary['correctness'].idxmin()
        print()
        print(f"🔦 Weakest category: {weakest} ({category_summary.loc[weakest, 'correctness']:.2f})")
        print("   Consider improving: prompt instructions, tool descriptions, or training data for this category.")
else:
    print("⚠️  Results table not available. Check the Experiments UI for detailed results.")

# COMMAND ----------

# DBTITLE 1,Module 09 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 09 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | Evaluation dataset | 25 Q&A pairs across 5 categories |
# MAGIC | Scorers defined | databank_compliance, correctness, safety |
# MAGIC | Evaluation run completed | Results in MLflow Experiments sidebar |
# MAGIC | Per-category analysis | Identifies weakest tool/category |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 How to Improve Scores
# MAGIC
# MAGIC | Low Score Area | Action |
# MAGIC |----------------|--------|
# MAGIC | Product knowledge | Add more document content to the volume; re-sync Vector Search |
# MAGIC | Customer data | Improve Genie Space instructions with more SQL examples |
# MAGIC | Risk scoring | Update the UC function formula; log parameter changes |
# MAGIC | Portfolio review | Add more context to the agent’s portfolio instructions |
# MAGIC | Fraud detection | Add more example fraud scenarios to the evaluation dataset |
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
# MAGIC ```
# MAGIC
# MAGIC 🙌 Well done, DataBank AI Engineers!