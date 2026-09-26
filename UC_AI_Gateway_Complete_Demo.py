# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Unity AI Gateway — Model Services Complete Demo
# MAGIC %md
# MAGIC ## 🏛️ Unity AI Gateway — Model Services (Complete Customer Demo)
# MAGIC **Duration:** ~45 minutes | **Audience:** Platform Engineers, Data Leaders, Security Teams
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is a Unity AI Gateway *Model Service*?
# MAGIC
# MAGIC A **Model Service** (a.k.a. Model API) is the latest Unity AI Gateway primitive: a **Unity Catalog securable**
# MAGIC living at `catalog.schema.name`, managed through `/api/2.1/unity-catalog/model-services`. You govern it with the
# MAGIC **same privileges as tables and functions** — `EXECUTE` + `USE CATALOG` + `USE SCHEMA` — so AI access control,
# MAGIC lineage, and audit all flow through Unity Catalog instead of a separate model-serving ACL system.
# MAGIC
# MAGIC A single Model Service can **route** to one or more destinations (pay-per-token foundation models, provisioned
# MAGIC throughput, or external providers), split traffic across them, attach **service policies (guardrails)**, enforce
# MAGIC **rate limits**, and log every request/response to a **Unity Catalog Delta table**.
# MAGIC
# MAGIC ### 🔄 New vs. Old — why this notebook was rebuilt
# MAGIC
# MAGIC | | **OLD** (deprecated in this demo) | **NEW** (this notebook) |
# MAGIC |---|---|---|
# MAGIC | Object | Serving endpoint w/ inline `ai_gateway` block | **Model Service** (UC securable) |
# MAGIC | API | `/api/2.0/serving-endpoints` | `/api/2.1/unity-catalog/model-services` |
# MAGIC | Governance | Endpoint-level permissions | UC grants (`EXECUTE`/`USE CATALOG`/`USE SCHEMA`) |
# MAGIC | Guardrails | `ai_gateway.guardrails` (unsupported on some endpoint types) | **`service_policies`** — built-in handlers that *enforce* |
# MAGIC | Invocation | `/serving-endpoints/<name>/invocations` | `/ai-gateway/mlflow/v1/chat/completions` (OpenAI-compatible) |
# MAGIC
# MAGIC ### Capabilities Demonstrated
# MAGIC
# MAGIC | # | Feature | Business Value |
# MAGIC |---|---------|----------------|
# MAGIC | 1 | **Model Service creation (UI + API)** | Centralized, UC-governed AI access |
# MAGIC | 2 | **Rate Limiting** | Cost control — cap requests per user / per service |
# MAGIC | 3 | **Traffic Splitting** | A/B testing, gradual model migration |
# MAGIC | 4 | **Guardrails (PII + Safety)** | Compliance — block sensitive/unsafe content, enforced |
# MAGIC | 5 | **Usage Tracking** | Audit trail via system tables |
# MAGIC | 6 | **Inference Tables** | Full payload logging to Delta |
# MAGIC | 7 | **AI Functions (ai_query)** | SQL-native, governed LLM access |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC   Your App / Notebook / SQL (ai_query)
# MAGIC            ↓  POST /ai-gateway/mlflow/v1/chat/completions   (model = catalog.schema.name)
# MAGIC   ┌──────────────────────────────────────────────────────┐
# MAGIC   │   Unity AI Gateway — Model Service (UC securable)     │
# MAGIC   │  ┌───────────┐ ┌────────────────┐ ┌───────────────┐  │
# MAGIC   │  │Rate Limits│ │Service Policies│ │Inference Table│  │
# MAGIC   │  │(per user/ │ │  (guardrails:  │ │ (Delta payload│  │
# MAGIC   │  │ service)  │ │  PII + safety) │ │  logging)     │  │
# MAGIC   │  └───────────┘ └────────────────┘ └───────────────┘  │
# MAGIC   │  ┌────────────────────────────────────────────────┐  │
# MAGIC   │  │  Routing: destinations + traffic_percentage     │  │
# MAGIC   │  └────────────────────────────────────────────────┘  │
# MAGIC   └───────────┬─────────────────────┬────────────────────┘
# MAGIC               ↓                     ↓
# MAGIC     Pay-per-token FM        Pay-per-token FM  (…provisioned / external)
# MAGIC     (system.ai.*)           (system.ai.*)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Setup — Install Dependencies
# ================================================================
# SETUP: Install dependencies (latest SDK + OpenAI client)
# ================================================================
%pip install databricks-sdk openai --upgrade --quiet
dbutils.library.restartPython()


# COMMAND ----------

# DBTITLE 1,Initialize — Client, Config & API Helper
from databricks.sdk import WorkspaceClient
from openai import OpenAI
import json

w = WorkspaceClient()

# --- CONFIGURATION (edit for your environment) ---
CATALOG = "gurpreetsethi_databank_lab"
SCHEMA  = "gurpreetsethi_financial_data"
MODEL_SERVICE = "gurpreetsethi-databank-llm-route"     # the Model Service name

# Derived identifiers used throughout
FQN    = f"{CATALOG}.{SCHEMA}.{MODEL_SERVICE}"          # 3-level securable name (used as `model=`)
PARENT = f"schemas/{CATALOG}.{SCHEMA}"                  # parent schema ref for create/logging
HOST   = w.config.host

# Foundation-model destination strings (verbatim — they differ per model)
FM = {
    "llama_3_3_70b":   "models/system.ai.llama_v3_3_70b_instruct",
    "claude_sonnet_4": "models/system.ai.databricks-claude-sonnet-4",
    "gemma_3_12b":     "models/system.ai.gemma-3-12b-it",
    "kimi_k3":         "models/system.ai.databricks-kimi-k3",
}

# --- Model Services management API helper ---
# Every management call (create/get/patch/delete) goes through here.
MS_BASE = "/api/2.1/unity-catalog/model-services"

def ms_api(method, path_suffix="", query=None, body=None):
    """Call the Unity AI Gateway Model Services API and return the response dict.
    method: GET | POST | PATCH | DELETE
    path_suffix: "" for the collection, or f"/{FQN}" for a single service.
    """
    return w.api_client.do(method, MS_BASE + path_suffix, query=query, body=body)

print("✅ Configuration set:")
print(f"   Catalog:        {CATALOG}")
print(f"   Schema:         {SCHEMA}")
print(f"   Model Service:  {MODEL_SERVICE}")
print(f"   FQN (model=):   {FQN}")
print(f"   Parent:         {PARENT}")
print(f"✅ Connected to workspace: {HOST}")
print(f"📍 Workspace ID: {w.get_workspace_id()}")


# COMMAND ----------

# DBTITLE 1,Section 1 — Create a Model Service (UI Walkthrough)
# MAGIC %md
# MAGIC ## 🖥️ Section 1: Create a Model Service via the UI
# MAGIC
# MAGIC ### Step-by-Step (Unity AI Gateway UI)
# MAGIC
# MAGIC 1. **Navigate:** Left sidebar → **AI Gateway** (or open **Catalog Explorer**)
# MAGIC 2. Click **➕ Create** → **Model Service**
# MAGIC 3. **Name & location:** give it a name and pick the **catalog + schema** — this makes it a Unity Catalog securable
# MAGIC 4. **Add destination(s):** choose one or more foundation models (pay-per-token, provisioned throughput, or an
# MAGIC    external provider service). With more than one destination, set **traffic %** per destination (must sum to 100).
# MAGIC 5. **Attach Service Policies (Guardrails):** add built-in policies —
# MAGIC    **PII detection** (`system.ai.detect_sensitive_data`) and **Unsafe content** (`system.ai.block_unsafe_content`).
# MAGIC 6. **Rate Limits:** add per-user and/or per-service request caps.
# MAGIC 7. **Inference logging:** pick a catalog/schema — Databricks auto-creates the payload Delta table.
# MAGIC 8. **Grant access:** on the **Permissions** tab, grant **EXECUTE** to the users/groups who may call it
# MAGIC    (they also need `USE CATALOG` + `USE SCHEMA`).
# MAGIC 9. Click **Create**.
# MAGIC
# MAGIC > 💡 **Talking Point:** Because a Model Service is a UC securable, the *same* grant/lineage/audit model that
# MAGIC > governs your tables now governs runtime AI access — no separate AI access-control plane.
# MAGIC >
# MAGIC > Everything in the UI is available programmatically via the API (Section 2 onward).

# COMMAND ----------

# DBTITLE 1,Section 2 — Create a Model Service (API)
# ================================================================
# SECTION 2: Create the Model Service programmatically (idempotent)
# ================================================================
# Infrastructure-as-Code for AI governance. Routing is set at CREATE time
# (config.routing is NOT patchable afterward). Here: gemma-3-12b + kimi-k3 at 50/50.

create_body = {
    "comment": "Databank demo — Unity AI Gateway Model Service (gemma + kimi, 50/50)",
    "config": {
        "routing": {
            "destinations": [
                {
                    "name": "gemma-3-12b",
                    "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                    "traffic_percentage": 50,
                    "pay_per_token_config": {"model": FM["gemma_3_12b"]},
                },
                {
                    "name": "kimi-k3",
                    "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                    "traffic_percentage": 50,
                    "pay_per_token_config": {"model": FM["kimi_k3"]},
                },
            ]
        }
    },
}

# Create-if-not-exists
try:
    existing = ms_api("GET", f"/{FQN}")
    print(f"ℹ️  Model Service '{FQN}' already exists — using it.")
    print(f"   Securable: {existing.get('securable_type')}  |  Owner: {existing.get('effective_owner')}")
except Exception as e:
    if "does not exist" in str(e) or "NOT_FOUND" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"🚀 Creating Model Service '{FQN}' ...")
        created = ms_api(
            "POST",
            query={"parent": PARENT, "model_service_id": MODEL_SERVICE},
            body=create_body,
        )
        print(f"✅ Created: {created.get('name')}")
    else:
        raise

# Other destination_type values you can use:
#   DESTINATION_TYPE_PROVISIONED_THROUGHPUT_FOUNDATION_MODEL
#   DESTINATION_TYPE_EXTERNAL_FOUNDATION_MODEL   (e.g. OpenAI / Anthropic / Bedrock provider services)

# COMMAND ----------

# DBTITLE 1,Section 3 — Rate Limiting
# MAGIC %md
# MAGIC ## ⏱️ Section 3: Rate Limiting
# MAGIC
# MAGIC Rate limits are part of the service **config** and applied with a `PATCH` (`update_mask=config.rate_limits`).
# MAGIC
# MAGIC | Field | Values |
# MAGIC |-------|--------|
# MAGIC | `key` | `RATE_LIMIT_KEY_SERVICE` (whole service) or `RATE_LIMIT_KEY_USER` (a specific principal) |
# MAGIC | `principal` | **required** when `key = RATE_LIMIT_KEY_USER` (e.g. `user@company.com`) |
# MAGIC | `renewal_period` | `RATE_LIMIT_RENEWAL_PERIOD_MINUTE` or `RATE_LIMIT_RENEWAL_PERIOD_HOUR` |
# MAGIC | `requests` | integer request cap per period |
# MAGIC
# MAGIC > 💡 Exceeding a limit returns an HTTP 429 to the caller — no silent failures.

# COMMAND ----------

# DBTITLE 1,Rate Limiting — Apply via API
# ================================================================
# RATE LIMITING: service-wide cap + a per-user cap
# ================================================================
rate_limits = [
    {  # whole-service ceiling
        "key": "RATE_LIMIT_KEY_SERVICE",
        "renewal_period": "RATE_LIMIT_RENEWAL_PERIOD_MINUTE",
        "requests": 1000,
    },
    {  # per-user quota (principal REQUIRED for USER key)
        "key": "RATE_LIMIT_KEY_USER",
        "principal": w.current_user.me().user_name,
        "renewal_period": "RATE_LIMIT_RENEWAL_PERIOD_MINUTE",
        "requests": 100,
    },
]

updated = ms_api(
    "PATCH", f"/{FQN}",
    query={"update_mask": "config.rate_limits"},
    body={"config": {"rate_limits": rate_limits}},
)

print("✅ Rate limits applied:")
for rl in updated.get("config", {}).get("rate_limits", []):
    scope = rl.get("principal", "all callers")
    print(f"   • {rl['key']}: {rl['requests']} requests / {rl['renewal_period']}  ({scope})")

# Note: only MINUTE and HOUR renewal periods are supported.

# COMMAND ----------

# DBTITLE 1,Section 4 — Traffic Splitting & Routing
# MAGIC %md
# MAGIC ## 🚦 Section 4: Traffic Splitting & Routing
# MAGIC
# MAGIC Routing lives in `config.routing.destinations[]`; each destination has a `traffic_percentage` (must sum to 100).
# MAGIC
# MAGIC | Pattern | Example |
# MAGIC |---------|---------|
# MAGIC | **A/B testing** | 50% model A / 50% model B |
# MAGIC | **Gradual migration** | 90% old / 10% new |
# MAGIC | **Load balancing** | split across providers |
# MAGIC
# MAGIC > ⚠️ **Routing is set at CREATE time** — `config.routing` is *not* patchable on this API. To change routing,
# MAGIC > recreate the service (or edit it in the UI). Fallback/failover ordering is likewise configured at create/UI;
# MAGIC > the programmatic fallback schema is not covered in this notebook.

# COMMAND ----------

# DBTITLE 1,Traffic Splitting — Show Current Routing
# ================================================================
# Show the live routing config (set at create time)
# ================================================================
svc = ms_api("GET", f"/{FQN}")
routes = svc.get("config", {}).get("routing", {}).get("destinations", [])

print("🚦 Current routing:")
print("=" * 55)
for r in routes:
    pct = r.get("traffic_percentage", 0)
    bar = "█" * (pct // 5) + "░" * ((100 - pct) // 5)
    model = r.get("pay_per_token_config", {}).get("model", r.get("destination_type"))
    print(f"   {r['name']:<16} {pct:>3}% {bar}  → {model}")

print("\n💡 Requests are distributed across destinations by these percentages — no app changes needed.")

# COMMAND ----------

# DBTITLE 1,Section 5 — Guardrails (Service Policies)
# MAGIC %md
# MAGIC ## 🔒 Section 5: Guardrails via Service Policies  ⭐
# MAGIC
# MAGIC **This is the headline of the modern gateway.** Guardrails attach as **service policies** and are *enforced* at
# MAGIC the gateway — a blocked request never reaches the model and never returns model output.
# MAGIC
# MAGIC ### Built-in policies
# MAGIC
# MAGIC | Policy | `handler` | Key options |
# MAGIC |--------|-----------|-------------|
# MAGIC | **PII detection** | `system.ai.detect_sensitive_data` | `action` = `block`/`mask`; `categories` (comma list of `class.*`); `phases` |
# MAGIC | **Unsafe content** | `system.ai.block_unsafe_content` | `model_service` (judge model); `phases` |
# MAGIC
# MAGIC Each entry: `{name, policy_type: POLICY_TYPE_BUILTIN | POLICY_TYPE_CUSTOM, handler, rank, options}`.
# MAGIC `phases` = `pre_call,post_call` inspects both the prompt **and** the response. Set `dry_run="true"` to log
# MAGIC without blocking. Custom policies (`POLICY_TYPE_CUSTOM`) point `handler` at a Unity Catalog function.
# MAGIC
# MAGIC > A blocked call returns `finish_reason="content_filter"` with a top-level `databricks_service_policy`
# MAGIC > `{action:"deny", name, phase, reason}` — you satisfy GDPR/CCPA/HIPAA without touching app code.

# COMMAND ----------

# DBTITLE 1,Guardrails — Apply Service Policies via API
# ================================================================
# Attach built-in guardrails: PII (block) + unsafe-content
# ================================================================
# NOTE: PATCH on config.service_policies REPLACES the whole policy list.
# These entries deliberately match the names + options of the policies already
# configured on this service via the UI (PII_GR_Test, Unsafe_GR_Test), so running
# this cell is idempotent — it re-asserts the existing guardrails rather than
# creating renamed duplicates. Rename/adjust freely on a service you own.
service_policies = [
    {
        "name": "PII_GR_Test",
        "policy_type": "POLICY_TYPE_BUILTIN",
        "handler": "system.ai.detect_sensitive_data",
        "rank": 1,
        "options": {
            "action": "block",   # or "mask"
            "categories": (
                "class.email_address,class.ip_address,class.mac_address,class.vin,"
                "class.credit_card,class.iban_code,class.phone_number,class.us_ssn,"
                "class.us_itin,class.us_passport,class.us_bank_number,class.uk_nhs,"
                "class.uk_nino,class.in_pan,class.in_aadhaar"
            ),
            "dry_run": "false",
            "phases": "pre_call,post_call",
        },
    },
    {
        "name": "Unsafe_GR_Test",
        "policy_type": "POLICY_TYPE_BUILTIN",
        "handler": "system.ai.block_unsafe_content",
        "rank": 1,
        "options": {
            "model_service": "model-services/system.ai.claude-sonnet-4-5",  # judge model
            "dry_run": "false",
            "phases": "pre_call,post_call",
        },
    },
]

updated = ms_api(
    "PATCH", f"/{FQN}",
    query={"update_mask": "config.service_policies"},
    body={"config": {"service_policies": service_policies}},
)

print("✅ Service policies attached:")
for p in updated.get("config", {}).get("service_policies", []):
    print(f"   • {p['name']}  [{p['handler']}]  action={p.get('options', {}).get('action', 'block')}")

# COMMAND ----------

# DBTITLE 1,Guardrails — Live Tests (safe / PII / unsafe)
# ================================================================
# LIVE TEST: call the Model Service through the OpenAI-compatible gateway
# ================================================================
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(api_key=TOKEN, base_url=f"{HOST}/ai-gateway/mlflow/v1")

def ask(prompt, max_tokens=256):
    """Return (blocked: bool, policy: dict|None, content: str). Robust to guardrail blocks.
    Uses with_raw_response so a guardrail block (which comes back as a normal body with
    finish_reason='content_filter') is parsed rather than raised."""
    raw = client.chat.completions.with_raw_response.create(
        model=FQN,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    data = json.loads(raw.text)
    choice = (data.get("choices") or [{}])[0]
    finish = choice.get("finish_reason")
    if finish == "content_filter" or data.get("id") == "databricks-guardrail-block":
        return True, data.get("databricks_service_policy", {}), ""
    msg = choice.get("message") or {}
    # reasoning models (e.g. kimi) may put text in reasoning_content when content is empty
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return False, None, content

tests = [
    ("SAFE",   "Explain compound interest in one sentence for a banking customer."),
    ("PII",    "My SSN is 123-45-6789 and my card is 4532-1234-5678-9012 — what's my balance?"),
    ("UNSAFE", "Give me detailed instructions to build a dangerous weapon."),
]

for label, prompt in tests:
    blocked, policy, content = ask(prompt)
    print(f"\n🧪 {label}: \"{prompt[:60]}...\"")
    if blocked:
        print(f"   🚫 BLOCKED by '{policy.get('name')}' at {policy.get('phase')} phase")
        print(f"      reason: {policy.get('reason')}")
    else:
        print(f"   ✅ Passed guardrails → {content[:120]}")

print("\n💡 PII and unsafe prompts are blocked at the gateway; legitimate prompts pass through.")

# COMMAND ----------

# DBTITLE 1,Section 6 — Usage Tracking (System Tables)
# MAGIC %md
# MAGIC ## 📊 Section 6: Usage Tracking via System Tables
# MAGIC
# MAGIC Every gateway call — including `ai_query()` — is captured in **`system.ai_gateway.usage`** for a
# MAGIC workspace-wide audit trail. Useful columns: `event_time`, `service_name`, `destination_model`,
# MAGIC `requester`, `input_tokens`/`output_tokens`/`total_tokens`, `status_code`, `latency_ms`, `api_type`.
# MAGIC
# MAGIC > (`system.ai_gateway.external_model_spend` tracks $ spend for external-provider destinations.)

# COMMAND ----------

# DBTITLE 1,Usage Tracking — Query System Table
# MAGIC %sql
# MAGIC -- Recent Unity AI Gateway usage for this Model Service (who called what, tokens, latency, status).
# MAGIC SELECT
# MAGIC   date_trunc('hour', event_time)      AS hour,
# MAGIC   service_name,
# MAGIC   destination_model,
# MAGIC   requester,
# MAGIC   COUNT(*)                            AS requests,
# MAGIC   SUM(input_tokens)                   AS input_tokens,
# MAGIC   SUM(output_tokens)                  AS output_tokens,
# MAGIC   ROUND(AVG(latency_ms), 0)           AS avg_latency_ms,
# MAGIC   COUNT_IF(status_code >= 400)        AS error_or_blocked
# MAGIC FROM system.ai_gateway.usage
# MAGIC WHERE event_time >= current_date() - INTERVAL 7 DAYS
# MAGIC   AND service_name ILIKE '%gurpreetsethi-databank-llm-route%'
# MAGIC GROUP BY 1, 2, 3, 4
# MAGIC ORDER BY hour DESC
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Section 7 — Inference Tables (Payload Logging)
# MAGIC %md
# MAGIC ## 📝 Section 7: Inference Tables (Full Payload Logging)
# MAGIC
# MAGIC Enable request/response logging to a Unity Catalog Delta table with a `PATCH`
# MAGIC (`update_mask=config.inference_table`). Databricks **auto-creates** the table
# MAGIC `<catalog>.<schema>.<prefix>_payload` (you cannot point at an existing table); rows appear within minutes.

# COMMAND ----------

# DBTITLE 1,Inference Table — Enable via API
# ================================================================
# Enable inference logging → Delta payload table
# ================================================================
LOG_PREFIX = MODEL_SERVICE  # table becomes <CATALOG>.<SCHEMA>.<LOG_PREFIX>_payload

updated = ms_api(
    "PATCH", f"/{FQN}",
    query={"update_mask": "config.inference_table"},
    body={"config": {"inference_table": {"parent": PARENT, "table_name_prefix": LOG_PREFIX}}},
)

it = updated.get("config", {}).get("inference_table", {})
PAYLOAD_TABLE = f"{CATALOG}.{SCHEMA}.{LOG_PREFIX}_payload"
print("📝 Inference logging enabled:")
print(f"   Parent: {it.get('parent')}")
print(f"   Table:  {it.get('table', PAYLOAD_TABLE)}")
print(f"\n➡️  Query it below (data appears within ~minutes of requests).")

# COMMAND ----------

# DBTITLE 1,Inference Table — Query Logged Payloads
# MAGIC %sql
# MAGIC -- Replace the table name below if your prefix differs.
# MAGIC -- Table = <catalog>.<schema>.<model_service_name>_payload
# MAGIC SELECT *
# MAGIC FROM gurpreetsethi_databank_lab.gurpreetsethi_financial_data.`gurpreetsethi-databank-llm-route_payload`
# MAGIC LIMIT 10

# COMMAND ----------

# DBTITLE 1,Section 8 — AI Functions (ai_query through the Gateway)
# MAGIC %md
# MAGIC ## 🧠 Section 8: AI Functions — SQL-Native, Governed
# MAGIC
# MAGIC `ai_query()` and the other SQL AI functions target **Foundation Model serving endpoints** by name
# MAGIC (e.g. `databricks-meta-llama-3-3-70b-instruct`) — they do **not** currently accept a Model Service
# MAGIC 3-level name as their endpoint argument. To invoke the *governed Model Service* from code, use the
# MAGIC OpenAI-compatible gateway path shown in Section 5 (`/ai-gateway/mlflow/v1/chat/completions`), where
# MAGIC routing, guardrails, rate limits, and logging all apply.

# COMMAND ----------

# DBTITLE 1,AI Functions — ai_query() via the Model Service
# MAGIC %sql
# MAGIC -- ai_query() targets a Foundation Model serving endpoint by name (not a Model Service).
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'Explain compound interest in one sentence for a banking customer.'
# MAGIC ) AS explanation

# COMMAND ----------

# DBTITLE 1,AI Functions — Classification & Extraction
# MAGIC %sql
# MAGIC -- ai_classify / ai_analyze_sentiment (Databricks-managed models)
# MAGIC SELECT
# MAGIC   feedback_text,
# MAGIC   ai_classify(feedback_text, ARRAY('complaint','compliment','question','suggestion')) AS category,
# MAGIC   ai_analyze_sentiment(feedback_text) AS sentiment
# MAGIC FROM VALUES
# MAGIC   ('Your mobile app crashes every time I try to transfer money'),
# MAGIC   ('Thank you for the excellent service at the branch today'),
# MAGIC   ('How do I set up automatic payments for my mortgage?')
# MAGIC AS t(feedback_text)

# COMMAND ----------

# DBTITLE 1,Section 9 — Inspect the Model Service
# ================================================================
# INSPECT: full current configuration of the Model Service
# ================================================================
svc = ms_api("GET", f"/{FQN}")
cfg = svc.get("config", {})

print(f"🔍 Model Service: {svc.get('name')}")
print(f"   Owner:   {svc.get('effective_owner')}")
print(f"   Created: {svc.get('create_time')}   Updated: {svc.get('update_time')}")
print(f"   Supported API types: {svc.get('supported_api_types')}")

print("\n🚦 Routing:")
for r in cfg.get("routing", {}).get("destinations", []):
    print(f"   • {r['name']}: {r.get('traffic_percentage')}%  → {r.get('pay_per_token_config', {}).get('model')}")

print("\n⏱️ Rate limits:")
for rl in cfg.get("rate_limits", []) or ["(none)"]:
    print(f"   • {rl}" if isinstance(rl, str) else
          f"   • {rl['key']}: {rl['requests']}/{rl['renewal_period']} {rl.get('principal','')}")

print("\n🔒 Service policies (guardrails):")
for p in cfg.get("service_policies", []) or ["(none)"]:
    print(f"   • {p}" if isinstance(p, str) else f"   • {p['name']} [{p['handler']}]")

print("\n📝 Inference table:")
print(f"   {cfg.get('inference_table', {}).get('table', '(not enabled)')}")

# COMMAND ----------

# DBTITLE 1,Section 10 — Cleanup (Optional)
# ================================================================
# CLEANUP (optional): delete the Model Service
# ================================================================
# ⚠️ This permanently removes the Model Service securable. The inference payload
#    table (a normal UC table) is NOT deleted by this call — drop it separately if desired.
#
# ms_api("DELETE", f"/{FQN}")
# print(f"🗑️  Deleted Model Service: {FQN}")

print("💡 To remove the demo service, uncomment the DELETE call above.")
print(f"   Service: {FQN}")

# COMMAND ----------

# DBTITLE 1,Demo Complete — Key Takeaways
# MAGIC %md
# MAGIC ## ✅ Demo Complete — Key Takeaways
# MAGIC
# MAGIC | # | Feature | How |
# MAGIC |---|---------|-----|
# MAGIC | 1 | **Model Service creation** | UI + `POST /api/2.1/unity-catalog/model-services` |
# MAGIC | 2 | **Rate limiting** | `PATCH config.rate_limits` (per-service / per-user, MINUTE/HOUR) |
# MAGIC | 3 | **Traffic splitting** | `config.routing.destinations[].traffic_percentage` (create-time) |
# MAGIC | 4 | **Guardrails** | `config.service_policies` — PII + unsafe, **enforced** (live-tested) |
# MAGIC | 5 | **Usage tracking** | `system.serving` system tables |
# MAGIC | 6 | **Inference tables** | `PATCH config.inference_table` → Delta payload table |
# MAGIC | 7 | **AI functions** | `ai_query('<fm-endpoint>', …)` (Model Service is called via the gateway REST path) |
# MAGIC
# MAGIC ### Why Model Services
# MAGIC
# MAGIC 1. **Unity Catalog-native** — governed by `EXECUTE`/`USE CATALOG`/`USE SCHEMA`, same as tables & functions.
# MAGIC 2. **Guardrails that enforce** — service policies block PII/unsafe content at the gateway, in `pre_call` and `post_call` phases.
# MAGIC 3. **Single audit trail & lineage** — through Unity Catalog, not a separate AI ACL plane.
# MAGIC 4. **OpenAI-compatible** — call `/ai-gateway/mlflow/v1/chat/completions` with the 3-level name as `model`.
# MAGIC 5. **One securable, full governance** — routing + rate limits + guardrails + logging in one object.
# MAGIC
# MAGIC ### Next Steps
# MAGIC 1. Grant **EXECUTE** to consumer groups (Catalog Explorer → the service → **Permissions**).
# MAGIC 2. Start with `dry_run="true"` guardrails to observe before enforcing.
# MAGIC 3. Add rate limits & (where needed) budgets before broad rollout.
# MAGIC 4. Build dashboards on the usage system tables + inference payload table.
# MAGIC
# MAGIC ---
# MAGIC ### Related
# MAGIC - AI Gateway → **Model Services** docs (`/api/2.1/unity-catalog/model-services`)
# MAGIC - Service policies (guardrails), rate limits, inference logging, governance pages