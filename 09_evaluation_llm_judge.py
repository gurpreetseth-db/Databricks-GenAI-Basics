# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Run Pre-Requisities
# MAGIC %run ./00_setup_prerequisites

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

# ================================================================
# TEST: LLM with Financial Questions
# ================================================================
# This cell tests the LLM backend. If the AI Gateway route exists,
# it calls through the gateway. Otherwise, it calls Foundation Models
# directly to demonstrate the same financial Q&A capability.
# ================================================================
import os
from openai import OpenAI
from databricks.sdk import WorkspaceClient

# --- Workspace authentication (automatic Bearer token, no PAT needed) ---
w = WorkspaceClient()
token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")

client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/ai-gateway/mlflow/v1"
)

# Try the AI Gateway route; fall back to Foundation Model if gateway proxy fails
try:
    _test = client.chat.completions.create(
        model=f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    MODEL_TO_USE = f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}"
    print(f"⚡ Using AI Gateway route: {CATALOG}.{SCHEMA}.{AI_GW_ROUTE}")
    print(f"   Backing model: {FOUNDATION_MODEL}")
except Exception as e:
    # Use llama-3-3 directly
    MODEL_TO_USE = "llama_v3_3_70b_instruct"
    print(f"⚡ AI Gateway proxy unavailable — calling Foundation Model directly: {MODEL_TO_USE}")
    print(f"   Error: {type(e).__name__}: {str(e)[:150]}")
    print(f"   Fix: Re-run Cell 6 (Option 2 - Via Code) to recreate with valid PAT")

SYSTEM_PROMPT = """
You are a DataBank financial advisor assistant. You provide clear, professional
advice on DataBank financial products. Always be concise and helpful.
If asked about specific account balances or transactions, explain you would need
to look those up via the banking system tools.
"""

test_questions = [
    "What is a Stocks & Shares ISA and who should consider one?",
    "What is the difference between a personal loan and a debt consolidation loan at DataBank?",
    "What top 2 DataBank products would you recommend for a customer who is new to investment?"

]

print("\n" + "=" * 65)
for i, question in enumerate(test_questions, 1):
    print(f"\n👤 Question {i}: {question}")
    print("-" * 65)

    response = client.chat.completions.create(
        model=MODEL_TO_USE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question}
        ],
        max_tokens=200
    )
    answer = response.choices[0].message.content
    if not answer:
        answer = response.choices[0].message.model_dump().get("reasoning_content", "")
    print(f"🤖 Response: {answer[:500]}{'...' if len(answer) > 500 else ''}")
    print(" ")
    print(f"   Tokens used: {response.usage.total_tokens}")

print(f"\n✅ LLM test complete (via {MODEL_TO_USE})")


# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration & Imports
# ================================================================
# UPGRADE MLflow to fix import errors
# ================================================================
# After this cell, Python restarts. Re-run Cell 1 (%run prerequisites)
# then continue from Cell 6 onwards.
# ================================================================
%pip install --upgrade mlflow[databricks] -q

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

import pandas as pd 
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
# ================================================================
# EVALUATION SETUP (self-contained after pip install restart)
# Re-establishes LLM client (same as Cell 3) and defines predict_fn
# for mlflow.genai.evaluate()
# ================================================================
import re, time
import mlflow
import mlflow.genai
import pandas as pd
from openai import OpenAI
from databricks.sdk import WorkspaceClient

# --- Re-derive configuration (same as 00_setup_prerequisites) ---
w = WorkspaceClient()
user = spark.sql("SELECT current_user() AS username").collect()[0]['username']
username_clean = re.sub(r'\W+', '', user.split('@')[0])

#CATALOG = f"{username_clean}_databank_lab"
#SCHEMA = f"{username_clean}_financial_data"
#AI_GW_ROUTE = f"{username_clean}-databank-llm-route"
#FOUNDATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
#experiment_name = f"/Users/{user}/{username_clean}_databank_ai_lab"

# --- LLM Client (same pattern as Cell 3) ---
client = OpenAI(
    api_key=w.config.authenticate().get("Authorization", "").replace("Bearer ", ""),
    base_url=f"{w.config.host}/ai-gateway/mlflow/v1"
)

# Determine model — try AI Gateway route, fall back to Foundation Model
try:
    _test = client.chat.completions.create(
        model=f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    MODEL_TO_USE = f"{CATALOG}.{SCHEMA}.{AI_GW_ROUTE}"
    print(f"⚡ Using AI Gateway route: {MODEL_TO_USE}")
except Exception:
    # Fall back to Foundation Model via serving endpoints
    MODEL_TO_USE = FOUNDATION_MODEL
    client = OpenAI(
        api_key=w.config.authenticate().get("Authorization", "").replace("Bearer ", ""),
        base_url=f"{w.config.host}/serving-endpoints"
    )
    print(f"⚡ AI Gateway unavailable — using Foundation Model: {MODEL_TO_USE}")

# System prompt (same as Cell 3)
SYSTEM_PROMPT = """
You are a DataBank financial advisor assistant. You provide clear, professional
advice on DataBank financial products. Always be concise and helpful.
If asked about specific account balances or transactions, explain you would need
to look those up via the banking system tools.
"""

mlflow.set_experiment(experiment_name)

# --- Predict Function for mlflow.genai.evaluate() ---
# MLflow 3.x unpacks the 'inputs' dict as kwargs to this function,
# so the parameter name must match the key in the eval data ('question').
def predict_fn(question: str) -> str:
    """Call the LLM (same config as Cell 3) and return the response."""
    response = client.chat.completions.create(
        model=MODEL_TO_USE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        max_tokens=400
    )
    answer = response.choices[0].message.content
    if not answer:
        answer = response.choices[0].message.model_dump().get("reasoning_content", "")
    return answer

# Smoke test
test_q = "What is the FSCS protection limit for DataBank savings accounts?"
test_a = predict_fn(question=test_q)
print(f"\n🧪 Smoke test:")
print(f"   Q: {test_q}")
print(f"   A: {test_a[:250]}...")
print(f"\n✅ Configuration complete")
print(f"   Experiment : {experiment_name}")
print(f"   Model      : {MODEL_TO_USE}")

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

# ================================================================
# SCORERS: Define what the LLM judge evaluates
# ================================================================

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

# ================================================================
# PREPARE EVALUATION DATA
# ================================================================
# mlflow.genai.evaluate requires 'inputs' as dicts and 'expectations' for scoring
df_eval_mlflow = df_eval.copy()
df_eval_mlflow['inputs'] = df_eval_mlflow['inputs'].apply(lambda q: {"question": q})
df_eval_mlflow['expectations'] = df_eval_mlflow['expected_response'].apply(
    lambda a: {"expected_response": a}
)

print(f"\n🏃 Running evaluation on {len(df_eval_mlflow)} questions against: {MODEL_TO_USE}")
print(f"   (This may take 3-5 minutes as each question is sent to the LLM)\n")

# ================================================================
# RUN MLflow GenAI Evaluation
# ================================================================
with mlflow.start_run(run_name="databank-llm-eval-v1"):
    mlflow.log_param("model", MODEL_TO_USE)
    mlflow.log_param("judge_model", FOUNDATION_MODEL)
    mlflow.log_param("system_prompt", SYSTEM_PROMPT.strip()[:250])
    mlflow.log_param("num_eval_questions", len(df_eval_mlflow))

    eval_results = mlflow.genai.evaluate(
        data=df_eval_mlflow,
        predict_fn=predict_fn,
        scorers=[databank_guidelines, correctness, safety]
    )

print("\n✅ Evaluation complete!")
print(f"   Results logged to experiment: {experiment_name}")

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
# MLflow 3.x: table key is 'eval_results', scores in '/value' columns,
# inputs/outputs nested in 'request'/'response' dicts
if eval_results and hasattr(eval_results, 'tables'):
    results_df = eval_results.tables.get('eval_results')
    if results_df is not None:
        # Extract question and response text from nested dicts
        results_df['question'] = results_df['request'].apply(
            lambda r: next((m['content'] for m in r.get('messages', []) if m['role'] == 'user'), '')
        )
        results_df['answer'] = results_df['response'].apply(
            lambda r: r.get('choices', [{}])[0].get('message', {}).get('content', '')[:200]
        )
        # Map correctness/compliance values to numeric for sorting
        results_df['correctness_score'] = results_df['correctness/value'].map({'yes': 1, 'no': 0})
        results_df['compliance_score'] = results_df['databank_compliance/value'].map({'yes': 1, 'no': 0})
        results_df['safety_score'] = results_df['safety/value'].map({'yes': 1, 'no': 0})

        print("\n📝 Per-Question Results (bottom 5 by correctness):")
        display(results_df.sort_values('correctness_score').head(5)[
            ['question', 'answer', 'correctness/value', 'correctness/rationale',
             'databank_compliance/value', 'databank_compliance/rationale',
             'safety/value', 'safety/rationale']
        ])

# COMMAND ----------

# DBTITLE 1,Step 5 — Score by Category
# Analyse performance by category to identify which tool needs improvement

if eval_results and hasattr(eval_results, 'tables'):
    results_df = eval_results.tables.get('eval_results')
    if results_df is not None:
        # Ensure numeric score columns exist (created in Step 4)
        if 'correctness_score' not in results_df.columns:
            results_df['correctness_score'] = results_df['correctness/value'].map({'yes': 1, 'no': 0})
            results_df['compliance_score'] = results_df['databank_compliance/value'].map({'yes': 1, 'no': 0})
            results_df['safety_score'] = results_df['safety/value'].map({'yes': 1, 'no': 0})

        # Join category from the original eval dataset
        results_df['category'] = df_eval['category'].values

        category_summary = results_df.groupby('category').agg(
            correctness=("correctness_score", "mean"),
            compliance=("compliance_score", "mean"),
            safety=("safety_score", "mean"),
            count=("correctness_score", "count")
        ).round(3)

        print("📊 Performance by Category:")
        print("-" * 70)
        print(f"  {'Category':<20} {'Correctness':<14} {'Compliance':<14} {'Safety':<10} {'N'}")
        print("-" * 70)
        for cat, row in category_summary.iterrows():
            print(f"  {cat:<20} {row.correctness:<14.2f} {row.compliance:<14.2f} {row.safety:<10.2f} {int(row['count'])}")

        # Find the weakest category
        weakest = category_summary['correctness'].idxmin()
        print()
        print(f"🔦 Weakest category: {weakest} ({category_summary.loc[weakest, 'correctness']:.2f})")
        print("   Consider improving: prompt instructions, tool descriptions, or training data for this category.")

        # Show as a display table too
        display(category_summary.reset_index())
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