# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# DBTITLE 1,Module 01 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 01: Data Generation
# MAGIC **Duration:** ~25 minutes | **Prerequisite:** Module 00 completed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What You'll Build
# MAGIC All subsequent modules depend on this dataset. You will create:
# MAGIC
# MAGIC | Table | Rows | Description |
# MAGIC |-------|------|-------------|
# MAGIC | `products` | 25 | DataBank product catalogue (Savings, Loans, Investments, Insurance, Credit Cards) |
# MAGIC | `customers` | 500 | Synthetic UK customer profiles with risk profiles and income data |
# MAGIC | `accounts` | 500 | Customer account relationships to products |
# MAGIC | `transactions` | 10,000 | Banking transactions with fraud labels |
# MAGIC | `support_tickets` | 300 | Customer support interactions with resolutions |
# MAGIC
# MAGIC Plus **7 PDF documents** uploaded to the Unity Catalog Volume:
# MAGIC - 5 product brochures (one per product category)
# MAGIC - 1 FAQ document
# MAGIC - 1 Terms & Conditions document
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Concepts
# MAGIC - **Spark + Faker + Pandas UDFs** — scalable synthetic data generation
# MAGIC - **Referential integrity** — master tables written to Delta first, then FK tables join back
# MAGIC - **Unity Catalog Volumes** — storing unstructured files alongside Delta tables
# MAGIC - **ReportLab** — programmatic PDF generation in Python

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# MAGIC %pip install faker==25.9.1 reportlab==4.2.5 -q

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration & Imports
# ================================================================
# CONFIGURATION (copy from Module 00 — run this cell first)
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

SCHEMA       = "financial_data"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/documents"

# ================================================================
# IMPORTS
# ================================================================
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, BooleanType
from pyspark.sql.window import Window   # IMPORTANT: use this, NOT F.window (that’s for streaming)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("✅ Imports ready")
print(f"📊 Tables will be created in: {CATALOG}.{SCHEMA}")
print(f"📄 PDFs will be saved to: {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Table 1 — Products (master table)
# MAGIC %md
# MAGIC ## 📄 Table 1: Products
# MAGIC
# MAGIC We create the **products** table first because it is a master/reference table.
# MAGIC Other tables (`accounts`) will hold foreign keys back to this table.
# MAGIC
# MAGIC **Why create master tables first?**
# MAGIC In serverless compute you **cannot use `.cache()` or `.persist()`**. The safe pattern is:
# MAGIC 1. Write master table to Delta ✓
# MAGIC 2. Read it back when creating FK-dependent tables ✓

# COMMAND ----------

# DBTITLE 1,Step 1 — Create Products Table
# 25 DataBank products across 5 categories
# Schema: product_id, name, product_type, interest_rate_pct, min_balance_gbp, description
products_data = [
    # ---- Savings ----
    ("PROD-001", "Basic Savings Account",   "Savings",    2.50,  0,     "Entry-level savings account with no minimum balance and competitive AER."),
    ("PROD-002", "Premium Savings Account", "Savings",    4.10,  5000,  "High-interest savings for balances above £5,000 with instant access."),
    ("PROD-003", "Cash ISA",                "Savings",    3.80,  0,     "Tax-free individual savings account — save up to £20,000 per tax year."),
    ("PROD-004", "Junior ISA",              "Savings",    3.50,  0,     "Tax-free savings account designed for children under 18."),
    ("PROD-005", "Fixed-Rate Bond",         "Savings",    5.20,  1000,  "Locked-in fixed rate over 1, 2, or 3 years. Best rate in the range."),
    # ---- Loans ----
    ("PROD-006", "Personal Loan",           "Loan",       6.90,  0,     "Flexible personal loan from £1,000 to £25,000 with no arrangement fee."),
    ("PROD-007", "Home Improvement Loan",   "Loan",       6.50,  0,     "Dedicated secured loan for renovation, extension, or home upgrade."),
    ("PROD-008", "Debt Consolidation Loan", "Loan",       7.50,  0,     "Simplify your finances by combining multiple debts into one payment."),
    ("PROD-009", "Business Loan",           "Loan",       8.20,  0,     "Flexible working capital funding for small and medium businesses."),
    ("PROD-010", "Graduate Loan",           "Loan",       5.90,  0,     "Special low-rate personal loan for graduates within 3 years of qualifying."),
    # ---- Investment ----
    ("PROD-011", "Stocks & Shares ISA",     "Investment", 0.0,   500,   "Tax-efficient investment in a diverse range of global equity funds."),
    ("PROD-012", "Managed Portfolio",       "Investment", 0.0,   10000, "Professionally managed, risk-rated portfolio rebalanced quarterly."),
    ("PROD-013", "Ethical Growth Fund",     "Investment", 0.0,   1000,  "ESG-screened investment in sustainable and responsible companies."),
    ("PROD-014", "Global Growth Fund",      "Investment", 0.0,   2000,  "High-growth exposure to international equity markets. Higher risk."),
    ("PROD-015", "Income Bond Fund",        "Investment", 0.0,   5000,  "Regular monthly income from a diversified investment-grade bond portfolio."),
    # ---- Insurance ----
    ("PROD-016", "Life Insurance",          "Insurance",  0.0,   0,     "Level-term life cover providing financial protection for your family."),
    ("PROD-017", "Critical Illness Cover",  "Insurance",  0.0,   0,     "Lump sum payout on diagnosis of over 50 specified critical illnesses."),
    ("PROD-018", "Home Insurance",          "Insurance",  0.0,   0,     "Combined buildings and contents insurance with new-for-old replacement."),
    ("PROD-019", "Travel Insurance",        "Insurance",  0.0,   0,     "Annual multi-trip worldwide cover including medical emergency and cancellation."),
    ("PROD-020", "Income Protection",       "Insurance",  0.0,   0,     "Monthly benefit payments of up to 60% of salary if unable to work."),
    # ---- Credit Cards ----
    ("PROD-021", "Standard Credit Card",    "CreditCard", 24.9,  0,     "Everyday Visa credit card with no annual fee and £1,000–£15,000 limit."),
    ("PROD-022", "Rewards Credit Card",     "CreditCard", 22.9,  0,     "Earn 1% cashback on all purchases, 2% on travel. £20 annual fee."),
    ("PROD-023", "Balance Transfer Card",   "CreditCard", 0.0,   0,     "0% interest for 24 months on balance transfers. 2.5% transfer fee."),
    ("PROD-024", "Travel Credit Card",      "CreditCard", 19.9,  0,     "No foreign transaction fees. Free travel insurance when you pay by card."),
    ("PROD-025", "Business Credit Card",    "CreditCard", 20.9,  0,     "Manage team expenses with individual limits, reporting, and 45-day credit."),
]

products_columns = ["product_id", "name", "product_type", "interest_rate_pct",
                    "min_balance_gbp", "description"]

products_df = spark.createDataFrame(products_data, schema=products_columns)
products_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.products")

# Add table + column comments for Genie discoverability
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.products IS 'DataBank product catalogue — 25 financial products across Savings, Loan, Investment, Insurance, and CreditCard categories'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.products ALTER COLUMN product_id COMMENT 'Unique product identifier (PROD-XXX format)'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.products ALTER COLUMN interest_rate_pct COMMENT 'Annual interest rate or APR as a percentage. 0.0 for non-interest products.'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.products ALTER COLUMN min_balance_gbp COMMENT 'Minimum balance required to open the account in GBP'")

print(f"✅ products table: {spark.table(f'{CATALOG}.{SCHEMA}.products').count()} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.products"))

# COMMAND ----------

# DBTITLE 1,Table 2 — Customers (master table)
# MAGIC %md
# MAGIC ## 👥 Table 2: Customers
# MAGIC
# MAGIC **Pandas UDFs** let us run Python/Faker code inside Spark in a vectorised, distributed way.
# MAGIC They receive a `pd.Series` as input and return a `pd.Series` as output — Spark handles parallelism automatically.
# MAGIC
# MAGIC ```python
# MAGIC @F.pandas_udf(StringType())
# MAGIC def fake_name(ids: pd.Series) -> pd.Series:
# MAGIC     from faker import Faker          # Import inside UDF — workers need the import
# MAGIC     fake = Faker('en_GB')
# MAGIC     return pd.Series([fake.name() for _ in range(len(ids))])
# MAGIC ```
# MAGIC
# MAGIC > ⚠️ **Important:** Always import libraries *inside* the UDF function body, not at module level. Spark serialises the function and sends it to workers — worker processes need to find the import themselves.

# COMMAND ----------

# DBTITLE 1,Step 2 — Create Customers Table
# --- Pandas UDFs for Faker data (UK financial context) ---

@F.pandas_udf(StringType())
def fake_name(ids: pd.Series) -> pd.Series:
    from faker import Faker
    fake = Faker("en_GB")
    Faker.seed(42)
    return pd.Series([fake.name() for _ in range(len(ids))])

@F.pandas_udf(StringType())
def fake_email(names: pd.Series) -> pd.Series:
    from faker import Faker
    fake = Faker("en_GB")
    Faker.seed(42)
    return pd.Series([fake.email() for _ in range(len(names))])

@F.pandas_udf(StringType())
def fake_city(ids: pd.Series) -> pd.Series:
    from faker import Faker
    fake = Faker("en_GB")
    Faker.seed(42)
    return pd.Series([fake.city() for _ in range(len(ids))])

@F.pandas_udf(StringType())
def fake_postcode(ids: pd.Series) -> pd.Series:
    from faker import Faker
    fake = Faker("en_GB")
    Faker.seed(42)
    return pd.Series([fake.postcode() for _ in range(len(ids))])

# --- Generate 500 customers ---
N_CUSTOMERS = 500

customers_df = (
    spark.range(0, N_CUSTOMERS, numPartitions=8)
    .select(
        # Primary key
        F.concat(F.lit("CUST-"), F.lpad(F.col("id").cast("string"), 4, "0")).alias("customer_id"),

        # Personal details (Faker)
        fake_name(F.col("id")).alias("full_name"),
        fake_email(F.col("id")).alias("email"),
        fake_city(F.col("id")).alias("city"),
        fake_postcode(F.col("id")).alias("postcode"),

        # Demographics
        (F.rand(seed=42) * 48 + 22).cast(IntegerType()).alias("age"),  # 22–70

        # Income: log-normal distribution (realistic UK income)
        # Mean ~£37k, range roughly £18k–£120k+
        F.round(F.exp(F.lit(10.5) + F.randn(seed=42) * 0.6), -2)
         .cast(DoubleType()).alias("annual_income_gbp"),

        # Risk profile: Conservative 35%, Moderate 40%, Aggressive 25%
        F.when(F.rand(seed=43) < 0.35, "Conservative")
         .when(F.rand(seed=43) < 0.75, "Moderate")
         .otherwise("Aggressive").alias("risk_profile"),

        # Customer type
        F.when(F.rand(seed=44) < 0.72, "Individual")
         .otherwise("Business").alias("customer_type"),

        # Membership tenure: joined 1–10 years ago
        F.date_sub(
            F.current_date(),
            (F.rand(seed=45) * 3650 + 365).cast(IntegerType())
        ).alias("member_since"),

        # Active flag
        (F.rand(seed=46) > 0.05).alias("is_active")
    )
)

# IMPORTANT: Write to Delta first (serverless — no .cache() or .persist() allowed)
customers_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.customers")

# Add table + column comments
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.customers IS 'DataBank customer master table — 500 synthetic UK customers for the AI lab'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.customers ALTER COLUMN risk_profile COMMENT 'Investment risk appetite: Conservative, Moderate, or Aggressive'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.customers ALTER COLUMN annual_income_gbp COMMENT 'Annual gross income in British pounds'")

count = spark.table(f"{CATALOG}.{SCHEMA}.customers").count()
print(f"✅ customers table: {count} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.customers").limit(5))

# COMMAND ----------

# DBTITLE 1,Step 3 — Create Accounts Table (FK join pattern)
# Accounts: 500 rows, one per customer, FK to both customers and products
# Pattern: read master tables from Delta, add row index, join on index for FK assignment

# Add sequential row index to customers (for FK join)
customers_idx = (
    spark.table(f"{CATALOG}.{SCHEMA}.customers")
    .select("customer_id")
    .withColumn("cust_idx", (F.row_number().over(Window.orderBy("customer_id")) - 1).cast("long"))
)

# Add sequential row index to products (for FK join)
products_idx = (
    spark.table(f"{CATALOG}.{SCHEMA}.products")
    .select("product_id", "product_type")
    .withColumn("prod_idx", (F.row_number().over(Window.orderBy("product_id")) - 1).cast("long"))
)

N_PRODUCTS = products_idx.count()  # 25

# Generate 500 accounts (one per customer)
accounts_base = (
    spark.range(0, N_CUSTOMERS, numPartitions=4)
    .withColumn("cust_idx", F.col("id").cast("long"))              # 1:1 with customers
    .withColumn("prod_idx", (F.rand(seed=55) * N_PRODUCTS).cast("long"))
    .withColumn("balance_gbp",
        F.round(F.exp(F.lit(8.5) + F.randn(seed=56) * 1.2), 2))   # log-normal balance
    .withColumn("status",
        F.when(F.rand(seed=57) < 0.85, "Active")
         .when(F.rand(seed=57) < 0.95, "Dormant")
         .otherwise("Closed"))
    .withColumn("opened_date",
        F.date_sub(F.current_date(),
                   (F.rand(seed=58) * 1825 + 30).cast(IntegerType())))
    .withColumn("monthly_fee_gbp",
        F.round(F.when(F.rand(seed=59) < 0.6, 0.0)
                 .otherwise(F.rand(seed=59) * 25 + 5), 2))
)

# Join to get actual customer_id and product_id
accounts_df = (
    accounts_base
    .join(customers_idx, "cust_idx", "left")
    .join(products_idx, "prod_idx", "left")
    .select(
        F.concat(F.lit("ACCT-"), F.lpad(F.col("id").cast("string"), 4, "0")).alias("account_id"),
        "customer_id",
        "product_id",
        "product_type",
        "balance_gbp",
        "status",
        "opened_date",
        "monthly_fee_gbp"
    )
)

accounts_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.accounts")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.accounts IS 'Customer account holdings — links customers to DataBank products'")

print(f"✅ accounts table: {spark.table(f'{CATALOG}.{SCHEMA}.accounts').count()} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.accounts").limit(5))

# COMMAND ----------

# DBTITLE 1,Step 4 — Create Transactions Table
# 10,000 banking transactions with realistic categories and a 2% fraud rate

@F.pandas_udf(StringType())
def fake_merchant(ids: pd.Series) -> pd.Series:
    from faker import Faker
    import random
    fake = Faker("en_GB")
    Faker.seed(42)
    merchants = [fake.company() for _ in range(200)]   # pool of 200 merchants
    return pd.Series([random.choice(merchants) for _ in range(len(ids))])

N_TXN = 10_000

# Read customers (master table) with row index for FK join
customers_idx2 = (
    spark.table(f"{CATALOG}.{SCHEMA}.customers")
    .select("customer_id")
    .withColumn("cust_idx", (F.row_number().over(Window.orderBy("customer_id")) - 1).cast("long"))
)
N_CUST = customers_idx2.count()

categories     = ["Groceries", "Transport", "Dining", "Shopping", "Utilities",
                  "Entertainment", "Healthcare", "Travel", "ATM Withdrawal", "Transfer"]
txn_types      = ["Debit", "Debit", "Debit", "Debit", "Debit", "Debit", "Debit", "Debit", "Debit", "Credit"]
channels       = ["Card", "Card", "Card", "Card", "Online", "Online", "ATM", "Branch"]

category_list  = F.array(*[F.lit(c) for c in categories])
channel_list   = F.array(*[F.lit(c) for c in channels])
txn_type_list  = F.array(*[F.lit(t) for t in txn_types])

txn_base = (
    spark.range(0, N_TXN, numPartitions=16)
    .withColumn("cust_idx",   (F.rand(seed=101) * N_CUST).cast("long"))
    # Transaction date: last 180 days
    .withColumn("txn_date",
        F.date_sub(F.current_date(), (F.rand(seed=102) * 180).cast(IntegerType())))
    # Amount: log-normal (most transactions small, occasional large ones)
    .withColumn("amount_gbp",
        F.round(F.exp(F.lit(3.5) + F.randn(seed=103) * 1.1), 2))
    # Randomly assign category, channel, type
    .withColumn("category",
        F.element_at(category_list, (F.rand(seed=104) * F.lit(len(categories))).cast(IntegerType()) + 1))
    .withColumn("channel",
        F.element_at(channel_list, (F.rand(seed=105) * F.lit(len(channels))).cast(IntegerType()) + 1))
    .withColumn("txn_type",
        F.when(F.rand(seed=106) < 0.9, "Debit").otherwise("Credit"))
    .withColumn("merchant", fake_merchant(F.col("id")))
    # Fraud flag: ~2% base rate, higher for large amounts (> £500) at night
    .withColumn("is_fraud",
        ((F.rand(seed=107) < 0.02) |
         ((F.rand(seed=108) < 0.05) & (F.col("amount_gbp") > 500)))
        .cast(BooleanType()))
    .withColumn("status",
        F.when(F.col("is_fraud"), 
            F.when(F.rand(seed=109) < 0.6, "Blocked").otherwise("Flagged"))
         .when(F.rand(seed=110) < 0.95, "Completed")
         .otherwise("Pending"))
)

# Join to get actual customer_id
txn_df = (
    txn_base
    .join(customers_idx2, "cust_idx", "left")
    .select(
        F.concat(F.lit("TXN-"), F.lpad(F.col("id").cast("string"), 5, "0")).alias("txn_id"),
        "customer_id", "txn_date", "amount_gbp", "category",
        "channel", "txn_type", "merchant", "is_fraud", "status"
    )
)

txn_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.transactions")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.transactions IS '10,000 DataBank transactions — includes ~2% fraud-labelled rows for anomaly detection demonstrations'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.transactions ALTER COLUMN is_fraud COMMENT 'True if this transaction was flagged as fraudulent. Use for fraud detection demos.'")

fraud_count = spark.table(f"{CATALOG}.{SCHEMA}.transactions").filter("is_fraud = true").count()
print(f"✅ transactions table: {spark.table(f'{CATALOG}.{SCHEMA}.transactions').count()} rows")
print(f"🚨 Fraudulent transactions: {fraud_count} ({fraud_count/N_TXN*100:.1f}%)")
display(spark.table(f"{CATALOG}.{SCHEMA}.transactions").limit(5))

# COMMAND ----------

# DBTITLE 1,Step 5 — Create Support Tickets Table
# 300 support tickets with subjects, descriptions, and resolutions

@F.pandas_udf(StringType())
def fake_ticket_description(subjects: pd.Series) -> pd.Series:
    from faker import Faker
    fake = Faker("en_GB")
    Faker.seed(42)
    templates = {
        "Card declined": "My card was declined at {merchant} for a purchase of £{amount}. Please investigate.",
        "Suspicious transaction": "I noticed an unrecognised charge of £{amount} from {merchant} on {date}. This was not me.",
        "Loan enquiry": "I would like to apply for a personal loan of £{amount} to fund home improvements. Please advise.",
        "Investment advice": "I am interested in your managed portfolio product. Can you advise on suitability given my profile?",
        "Account closure": "I wish to close my savings account and transfer the balance to an external account.",
        "Interest rate query": "I notice my savings interest rate has changed. Can you explain the new rate?",
        "Online banking issue": "I am unable to log into my online banking. The app shows an error message.",
        "Statement request": "Please send me 12 months of statements for my current account for mortgage purposes.",
        "Fraud report": "There are several transactions I do not recognise on my account. I believe my card has been compromised.",
        "Mortgage enquiry": "I am a first-time buyer looking for mortgage options. Could I book an appointment with an advisor?"
    }
    results = []
    for subj in subjects:
        template = templates.get(subj, "I need help with my account.")
        text = template.format(
            merchant=fake.company(),
            amount=round(abs(fake.pyfloat(min_value=10, max_value=2000)), 2),
            date=fake.date_this_year()
        )
        results.append(text)
    return pd.Series(results)

@F.pandas_udf(StringType())
def fake_resolution(statuses: pd.Series) -> pd.Series:
    from faker import Faker
    fake = Faker("en_GB")
    Faker.seed(42)
    results = []
    for status in statuses:
        if status in ("Resolved", "Closed"):
            results.append(f"Issue investigated and resolved. Customer contacted on {fake.date_this_month()}. "
                           f"No further action required. Reference: REF-{fake.numerify('######')}.")
        else:
            results.append(None)
    return pd.Series(results)

subjects = ["Card declined", "Suspicious transaction", "Loan enquiry", "Investment advice",
            "Account closure", "Interest rate query", "Online banking issue",
            "Statement request", "Fraud report", "Mortgage enquiry"]
subject_array = F.array(*[F.lit(s) for s in subjects])

N_TICKETS = 300

# Read customers for FK
customers_idx3 = (
    spark.table(f"{CATALOG}.{SCHEMA}.customers")
    .select("customer_id")
    .withColumn("cust_idx", (F.row_number().over(Window.orderBy("customer_id")) - 1).cast("long"))
)

tickets_base = (
    spark.range(0, N_TICKETS, numPartitions=4)
    .withColumn("cust_idx",
        (F.rand(seed=200) * N_CUST).cast("long"))
    .withColumn("subject",
        F.element_at(subject_array, (F.rand(seed=201) * F.lit(len(subjects))).cast(IntegerType()) + 1))
    .withColumn("priority",
        F.when(F.rand(seed=202) < 0.20, "High")
         .when(F.rand(seed=202) < 0.70, "Medium")
         .otherwise("Low"))
    .withColumn("ticket_status",
        F.when(F.rand(seed=203) < 0.25, "Open")
         .when(F.rand(seed=203) < 0.55, "In Progress")
         .when(F.rand(seed=203) < 0.95, "Resolved")
         .otherwise("Closed"))
    .withColumn("created_date",
        F.date_sub(F.current_date(), (F.rand(seed=204) * 90).cast(IntegerType())))
)

# Add Faker-generated fields
tickets_base = (
    tickets_base
    .withColumn("description", fake_ticket_description(F.col("subject")))
    .withColumn("resolution",   fake_resolution(F.col("ticket_status")))
)

# Join for customer_id
tickets_df = (
    tickets_base
    .join(customers_idx3, "cust_idx", "left")
    .select(
        F.concat(F.lit("TICK-"), F.lpad(F.col("id").cast("string"), 4, "0")).alias("ticket_id"),
        "customer_id", "subject", "description", "priority",
        "ticket_status", "created_date", "resolution"
    )
)

tickets_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.support_tickets")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.support_tickets IS 'DataBank customer support tickets with subjects, descriptions and resolutions for RAG demonstrations'")

print(f"✅ support_tickets table: {spark.table(f'{CATALOG}.{SCHEMA}.support_tickets').count()} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.support_tickets").filter("ticket_status = 'Resolved'").limit(5))

# COMMAND ----------

# DBTITLE 1,PDF Generation — Overview
# MAGIC %md
# MAGIC ## 📄 PDF Document Generation
# MAGIC
# MAGIC We generate **7 PDF documents** using [ReportLab](https://www.reportlab.com/) and store them in the Unity Catalog Volume.
# MAGIC These documents will be used in **Module 04 (Vector Search)** and **Module 07 (AgentBricks Knowledge Assistant)**.
# MAGIC
# MAGIC ### Document Structure
# MAGIC ```
# MAGIC /Volumes/databank_lab/financial_data/documents/
# MAGIC ├── product_brochures/
# MAGIC │   ├── savings_products.pdf        ← ISA, Fixed Bond, Premium Savings details
# MAGIC │   ├── loan_products.pdf            ← Personal Loan, Business Loan rates & criteria
# MAGIC │   ├── investment_products.pdf      ← Stocks ISA, Managed Portfolio, Ethical Fund
# MAGIC │   ├── insurance_products.pdf       ← Life, Critical Illness, Income Protection
# MAGIC │   └── credit_card_products.pdf     ← Rewards, 0% Transfer, Travel Card details
# MAGIC └── compliance/
# MAGIC     ├── frequently_asked_questions.pdf
# MAGIC     └── terms_and_conditions.pdf
# MAGIC ```
# MAGIC
# MAGIC **Why PDFs matter:** Real-world financial advisors have to refer to product guides, T&Cs, and regulatory documents. Vector Search lets our AI assistant search these documents semantically — even when the exact words aren’t used.

# COMMAND ----------

# DBTITLE 1,Step 6a — PDF Helper Functions
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

def build_pdf(filepath: str, title: str, subtitle: str, sections: list):
    """
    Build a styled PDF brochure.
    
    sections: list of dicts with keys:
        - 'heading': str
        - 'body':    str (plain text or HTML-like)
        - 'table':   list of lists (optional, adds a data table)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    # Custom styles
    brand_blue = colors.HexColor("#1B3A6B")
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        textColor=brand_blue, fontSize=22, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        textColor=colors.HexColor("#4A90D9"), fontSize=13, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "Heading2", parent=styles["Heading2"],
        textColor=brand_blue, fontSize=13, spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=16, spaceAfter=10
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        textColor=colors.grey, fontSize=8, alignment=TA_CENTER
    )

    story = []

    # Header bar
    story.append(Paragraph("🏦 DataBank", ParagraphStyle(
        "Logo", parent=styles["Normal"], fontSize=10,
        textColor=colors.white, backColor=brand_blue,
        spaceAfter=12, leading=20, leftIndent=-20, rightIndent=-20
    )))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=brand_blue, spaceAfter=14))

    # Sections
    for section in sections:
        story.append(Paragraph(section["heading"], heading_style))
        story.append(Paragraph(section["body"], body_style))
        if "table" in section and section["table"]:
            tbl = Table(section["table"], colWidths=[8*cm, 8*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), brand_blue),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTSIZE",   (0,0), (-1,0), 10),
                ("FONTSIZE",   (0,1), (-1,-1), 9),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.4*cm))

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "DataBank plc | Authorised and regulated by the Financial Conduct Authority | "
        "FCA Register No. 987654 | databank.co.uk",
        footer_style
    ))

    doc.build(story)
    print(f"  ✓ Created: {filepath}")

print("✅ PDF helper function ready")

# COMMAND ----------

# DBTITLE 1,Step 6b — Generate Product Brochures
brochures_dir = f"{VOLUME_PATH}/product_brochures"

# --- 1. Savings Products ---
build_pdf(
    filepath=f"{brochures_dir}/savings_products.pdf",
    title="Savings Products Guide",
    subtitle="Grow your money with DataBank’s competitive savings range",
    sections=[
        {"heading": "Cash ISA",
         "body": "Earn tax-free interest on savings up to £20,000 per tax year (2024/25 ISA allowance). "
                 "Instant access with no penalties. Interest calculated daily and paid monthly. "
                 "Available to UK residents aged 18 and over. Rate: 3.80% AER variable.",
         "table": [["Feature", "Detail"],
                   ["AER", "3.80% (variable)"],
                   ["Min opening balance", "£1"],
                   ["Max annual subscription", "£20,000"],
                   ["Access", "Instant, unlimited withdrawals"],
                   ["FSCS protected", "Yes, up to £85,000"]]},
        {"heading": "Fixed-Rate Bond",
         "body": "Lock in our best rate and guarantee your return. Available in 1-year, 2-year, and 3-year terms. "
                 "No withdrawals permitted during the fixed term. At maturity, funds transferred to your "
                 "nominated account or reinvested at the prevailing rate. Min deposit £1,000.",
         "table": [["Term", "Gross Rate / AER"],
                   ["1 Year", "4.85% / 4.85%"],
                   ["2 Year", "5.10% / 5.10%"],
                   ["3 Year", "5.20% / 5.20%"]]},
        {"heading": "Premium Savings Account",
         "body": "For balances above £5,000. Earn an enhanced rate with same-day access. "
                 "Rate automatically steps down to Basic Savings AER if balance falls below £5,000. "
                 "No arrangement fee. Available to existing DataBank current account holders only.",
         "table": [["Feature", "Detail"],
                   ["AER (balance ≥ £5k)", "4.10% (variable)"],
                   ["AER (balance < £5k)", "2.50% (variable)"],
                   ["Notice period", "None — instant access"],
                   ["Interest paid", "Monthly"]]},
        {"heading": "Junior ISA",
         "body": "Start saving for your child’s future tax-free. Opened by a parent or legal guardian for children "
                 "under 18. The child can take control at 16 and access funds at 18. "
                 "Annual subscription limit £9,000 (2024/25). Cannot be transferred to an adult ISA until the child turns 18.",
         "table": [["Feature", "Detail"],
                   ["AER", "3.50% (variable)"],
                   ["Annual subscription limit", "£9,000"],
                   ["Access", "At age 18 only"],
                   ["Who can open", "Parent or legal guardian"]]},
    ]
)

# --- 2. Loan Products ---
build_pdf(
    filepath=f"{brochures_dir}/loan_products.pdf",
    title="Borrowing Products Guide",
    subtitle="Flexible lending solutions from DataBank",
    sections=[
        {"heading": "Personal Loan",
         "body": "Borrow from £1,000 to £25,000 over 1–7 years. Fixed monthly repayments so you always know "
                 "what you owe. Representative APR 6.9%. No arrangement fee, no early repayment charge "
                 "after month 6. Subject to credit assessment. UK residents aged 21–70.",
         "table": [["Loan Amount", "Representative APR"],
                   ["£1,000 – £2,999", "9.9% APR"],
                   ["£3,000 – £7,499", "7.4% APR"],
                   ["£7,500 – £25,000", "6.9% APR"]]},
        {"heading": "Debt Consolidation Loan",
         "body": "Combine multiple credit balances into a single monthly payment. May reduce your total monthly "
                 "outgoing, but extending the term may increase overall interest paid. "
                 "DataBank will pay creditors directly on your behalf upon completion. "
                 "Representative APR 7.5%. Borrow £3,000 to £30,000.",
         "table": []},
        {"heading": "Business Loan",
         "body": "Flexible working capital and asset finance for sole traders, partnerships, and limited companies. "
                 "Borrow £10,000 to £250,000 over 1–10 years. Security may be required above £50,000. "
                 "Representative APR 8.2%. Dedicated relationship manager assigned for amounts above £25,000.",
         "table": []},
        {"heading": "Graduate Loan",
         "body": "Designed for graduates within 3 years of receiving a UK undergraduate or postgraduate degree. "
                 "Borrow up to £15,000 at our lowest personal loan rate. Proof of qualification required. "
                 "Representative APR 5.9%. 6-month payment holiday available in year 1.",
         "table": []},
    ]
)

# --- 3. Investment Products ---
build_pdf(
    filepath=f"{brochures_dir}/investment_products.pdf",
    title="Investment Products Guide",
    subtitle="Invest for your future with DataBank Wealth Management",
    sections=[
        {"heading": "Important: Capital at Risk",
         "body": "The value of investments can fall as well as rise. You may get back less than you invest. "
                 "Past performance is not a reliable indicator of future results. "
                 "DataBank Investment Services is authorised and regulated by the FCA.",
         "table": []},
        {"heading": "Stocks & Shares ISA",
         "body": "Invest tax-free up to £20,000 per tax year. Choose from over 3,000 funds, shares, and ETFs. "
                 "No capital gains tax or income tax on returns. Flexible ISA — you can withdraw and reinvest "
                 "within the same tax year. Min investment £500 lump sum or £25/month.",
         "table": [["Risk Level", "Example Fund", "5yr Return (past)"],
                   ["Low", "DataBank Income Bond Fund", "+18%"],
                   ["Medium", "DataBank Ethical Growth Fund", "+42%"],
                   ["High", "DataBank Global Growth Fund", "+71%"]]},
        {"heading": "Managed Portfolio Service",
         "body": "Let our investment team manage your portfolio. Available in 5 risk-rated strategies from "
                 "Cautious to Adventurous. Quarterly rebalancing. Min investment £10,000. "
                 "Annual management charge 0.65% + underlying fund charges (avg 0.25%).",
         "table": [["Strategy", "Target Allocation", "Volatility"],
                   ["Cautious", "80% Bonds, 20% Equity", "Low"],
                   ["Balanced", "50% Bonds, 50% Equity", "Medium"],
                   ["Growth", "20% Bonds, 80% Equity", "Medium-High"],
                   ["Adventurous", "100% Equity", "High"]]},
    ]
)

# --- 4. Insurance Products ---
build_pdf(
    filepath=f"{brochures_dir}/insurance_products.pdf",
    title="Protection & Insurance Guide",
    subtitle="Protecting what matters most",
    sections=[
        {"heading": "Life Insurance",
         "body": "Level-term life insurance pays a tax-free lump sum if you die within the policy term. "
                 "Terms from 5 to 40 years, sum assured from £50,000 to £2,000,000. "
                 "Premiums fixed for the full term. Free terminal illness benefit included.",
         "table": []},
        {"heading": "Critical Illness Cover",
         "body": "Pays a lump sum on diagnosis of over 50 specified serious conditions including cancer, "
                 "heart attack, stroke, and multiple sclerosis. Can be taken standalone or combined with life cover. "
                 "Children’s cover included at no extra cost for conditions affecting ages 0–18.",
         "table": []},
        {"heading": "Income Protection",
         "body": "Pays up to 60% of your gross monthly income if you are unable to work due to illness or injury. "
                 "Deferred period options: 4, 8, 13, or 26 weeks. Benefit paid to age 65 or until return to work. "
                 "Own occupation definition during the first 2 years; any suited occupation thereafter.",
         "table": [["Deferred Period", "Monthly Cost (illustrative, age 35, non-smoker)"],
                   ["4 weeks",  "£62/month"],
                   ["13 weeks", "£44/month"],
                   ["26 weeks", "£28/month"]]},
    ]
)

# --- 5. Credit Card Products ---
build_pdf(
    filepath=f"{brochures_dir}/credit_card_products.pdf",
    title="Credit Card Products Guide",
    subtitle="Spend smarter with DataBank credit cards",
    sections=[
        {"heading": "Rewards Credit Card",
         "body": "Earn 1% cashback on all purchases, 2% on travel and dining. "
                 "No cashback cap. Annual fee £20, waived in year 1. "
                 "0% on purchases for 3 months from account opening. Representative APR 22.9%.",
         "table": [["Category", "Cashback Rate"],
                   ["All purchases", "1.0%"],
                   ["Travel & Dining", "2.0%"],
                   ["DataBank products", "3.0%"]]},
        {"heading": "0% Balance Transfer Card",
         "body": "Transfer balances from other cards and pay 0% interest for 24 months. "
                 "Balance transfer fee 2.5% (min £5). Representative APR 24.9% after promotional period. "
                 "Min credit limit £500. Must transfer within 60 days of account opening.",
         "table": []},
        {"heading": "Travel Credit Card",
         "body": "No foreign transaction fees worldwide. Free travel insurance when you pay for your trip by card. "
                 "No ATM fees abroad on DataBank ATMs. Representative APR 19.9%. "
                 "Emergency card replacement within 24 hours anywhere in the world.",
         "table": []},
    ]
)

print("✅ All 5 product brochures generated")

# COMMAND ----------

# DBTITLE 1,Step 6c — Generate Compliance Documents
compliance_dir = f"{VOLUME_PATH}/compliance"

# --- 6. Frequently Asked Questions ---
build_pdf(
    filepath=f"{compliance_dir}/frequently_asked_questions.pdf",
    title="Frequently Asked Questions",
    subtitle="DataBank Customer Services — Quick Answers",
    sections=[
        {"heading": "Savings & Accounts",
         "body": "Q: How do I open a Cash ISA? \nA: Apply online, via the DataBank app, or in branch. You will need to provide proof of ID and address. Processing takes 1–3 business days.\n\n"
                 "Q: Can I have more than one ISA? \nA: From April 2024 you can subscribe to multiple ISAs of the same type in the same tax year. You can only subscribe to one Cash ISA per tax year.\n\n"
                 "Q: What happens to my Fixed-Rate Bond at maturity? \nA: We will contact you 30 days before maturity with reinvestment options. If we do not hear from you, the balance rolls to a Standard Savings Account.",
         "table": []},
        {"heading": "Borrowing & Loans",
         "body": "Q: How quickly can I get a personal loan? \nA: Approved loans are usually paid within 2 hours for existing DataBank current account holders. New customers typically receive funds within 1 business day.\n\n"
                 "Q: Is there a penalty for early repayment? \nA: No early repayment charge applies after month 6. Repaying in months 1–5 incurs a charge equal to 30 days\u2019 interest.\n\n"
                 "Q: Will applying for a loan affect my credit score? \nA: An initial soft search does not affect your credit score. A full credit application leaves a hard footprint visible to other lenders.",
         "table": []},
        {"heading": "Fraud & Security",
         "body": "Q: What should I do if I spot a transaction I don’t recognise? \nA: Call our 24/7 fraud line on 0800 123 4567 immediately or use the \u2018Report fraud\u2019 option in the DataBank app. We will freeze your card and investigate within 3 business days.\n\n"
                 "Q: Am I liable for fraudulent transactions? \nA: Under the Payment Services Regulations you are not liable for unauthorised transactions unless you acted fraudulently or with gross negligence. We aim to refund within 5 business days of a confirmed fraud report.\n\n"
                 "Q: How does DataBank protect my data? \nA: All data is encrypted at rest and in transit. We are registered with the ICO (registration Z1234567) and comply fully with UK GDPR.",
         "table": []},
        {"heading": "Investments",
         "body": "Q: What is the difference between a Cash ISA and a Stocks & Shares ISA? \nA: A Cash ISA pays a fixed or variable interest rate with no investment risk. A Stocks & Shares ISA invests in markets — higher potential returns but your capital is at risk.\n\n"
                 "Q: How do I change my managed portfolio risk level? \nA: Log in to DataBank Online, navigate to Investments > My Portfolio > Change Strategy. Changes take effect at the next quarterly rebalancing date.\n\n"
                 "Q: Can I withdraw from my Stocks & Shares ISA at any time? \nA: Yes. Redemption takes 3–5 business days. Withdrawals are free but reduce your available ISA subscription for the current tax year.",
         "table": []},
    ]
)

# --- 7. Terms & Conditions ---
build_pdf(
    filepath=f"{compliance_dir}/terms_and_conditions.pdf",
    title="General Terms & Conditions",
    subtitle="DataBank plc — Personal and Business Banking",
    sections=[
        {"heading": "1. About These Terms",
         "body": "These Terms and Conditions govern your relationship with DataBank plc (\u2018we\u2019, \u2018us\u2019, \u2018the Bank\u2019). "
                 "They form the basis of the contract between you and us for any account or product held with DataBank. "
                 "By opening an account or applying for a product, you agree to be bound by these terms. "
                 "We may update these terms at any time with 30 days\u2019 written notice.",
         "table": []},
        {"heading": "2. Fees and Charges",
         "body": "Our standard fees are set out in the Schedule of Charges available at databank.co.uk/fees. "
                 "We will give you 30 days\u2019 notice of any fee increase. "
                 "We reserve the right to charge for paper statements (£1.50 per statement) for accounts "
                 "where e-statements have been declined. International payment charges apply per the SWIFT tariff.",
         "table": [["Service", "Fee"],
                   ["Monthly account fee (Standard)", "£0"],
                   ["Monthly account fee (Premier)", "£15"],
                   ["CHAPS payment", "£20"],
                   ["Paper statement", "£1.50"],
                   ["Returned payment charge", "£12"]]},
        {"heading": "3. Closing Your Account",
         "body": "You may close your account at any time by contacting us in writing, by phone, or in branch. "
                 "We may close your account with 60 days\u2019 notice. We may close immediately if we suspect fraud, "
                 "money laundering, or a breach of these terms. "
                 "Any credit balance will be returned to you within 5 business days of closure.",
         "table": []},
        {"heading": "4. Governing Law",
         "body": "These terms are governed by and construed in accordance with the law of England and Wales. "
                 "Any dispute shall be subject to the exclusive jurisdiction of the courts of England and Wales. "
                 "If you are not satisfied with our response, you may refer your complaint to the "
                 "Financial Ombudsman Service (FOS) free of charge.",
         "table": []},
    ]
)

print("✅ Compliance documents generated")

# COMMAND ----------

# DBTITLE 1,Step 7 — Verify All Outputs
import os

print("=" * 55)
print("  DataBank AI Lab — Data Generation Summary")
print("=" * 55)

# Delta tables
for tbl in ["products", "customers", "accounts", "transactions", "support_tickets"]:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{tbl}").count()
    print(f"  ✅ {CATALOG}.{SCHEMA}.{tbl:<20} {count:>6,} rows")

print()
print("  Documents in Volume:")
for root, dirs, files in os.walk(VOLUME_PATH):
    for fname in files:
        full = os.path.join(root, fname)
        size_kb = os.path.getsize(full) / 1024
        rel = full.replace(VOLUME_PATH + "/", "")
        print(f"  📄 {rel:<45} {size_kb:>7.1f} KB")

print()
print("  All data is ready — proceed to Module 02 (AI Gateway)")
print("=" * 55)

# COMMAND ----------

# DBTITLE 1,Module 01 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 01 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | `products` table | 25 rows, visible in Catalog Explorer |
# MAGIC | `customers` table | 500 rows |
# MAGIC | `accounts` table | 500 rows, `customer_id` and `product_id` populated |
# MAGIC | `transactions` table | 10,000 rows, ~200 with `is_fraud = true` |
# MAGIC | `support_tickets` table | 300 rows, mix of statuses |
# MAGIC | PDFs in volume | 7 files across `product_brochures/` and `compliance/` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Next: Module 02 — AI Gateway
# MAGIC Open **`02_ai_gateway_setup`** to configure a managed LLM route that all modules will use for inference.