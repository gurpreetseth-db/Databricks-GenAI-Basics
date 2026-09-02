# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Module 08 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 08: Databricks Apps (Gradio)
# MAGIC **Duration:** ~25 minutes | **Prerequisite:** Module 07 (agent endpoint deployed)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is a Databricks App?
# MAGIC
# MAGIC **Databricks Apps** let you deploy Python web applications that:
# MAGIC - Run **serverlessly** inside Databricks (no infrastructure management)
# MAGIC - Have **OAuth-secured access** to all Databricks resources (SQL warehouses, serving endpoints, Unity Catalog)
# MAGIC - Can be shared with users in your workspace via URL
# MAGIC - Scale automatically, with built-in authentication
# MAGIC
# MAGIC ### Why Gradio?
# MAGIC **Gradio** is ideal for AI demos:
# MAGIC - Built-in chat interface with message history
# MAGIC - Zero JavaScript required
# MAGIC - Pre-installed on Databricks Apps runtime (Gradio 4.44.0)
# MAGIC - Perfect for showcasing conversational AI
# MAGIC
# MAGIC ### What You’ll Deploy
# MAGIC ```
# MAGIC Browser
# MAGIC   ↓  (HTTPS)
# MAGIC Databricks App (Gradio chat UI)
# MAGIC   ↓  (OpenAI-compatible API call)
# MAGIC Agent Serving Endpoint (databank-ai-advisor)
# MAGIC   ↓
# MAGIC DataBank AI Advisor (Supervisor Agent)
# MAGIC   ├── Knowledge Assistant ← PDFs
# MAGIC   ├── Genie Space        ← Delta tables
# MAGIC   └── UC Functions       ← Risk, Portfolio, Fraud
# MAGIC ```
# MAGIC
# MAGIC ### Module Structure
# MAGIC 1. Generate `app.py` (Gradio application)
# MAGIC 2. Generate `app.yaml` (Databricks Apps config)
# MAGIC 3. Deploy using the CLI
# MAGIC 4. Test the live application

# COMMAND ----------

# DBTITLE 1,Step 1 - Reference Parameters
# MAGIC %run ./Config_Parameters

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration
# ================================================================
# CONFIGURATION
# ================================================================
from databricks.sdk import WorkspaceClient
import time

user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

#AGENT_ENDPOINT = "gurpreetsethi_DataBank-AI-Advisor"  # From Module 07
APP_NAME       = f"{username_clean}-databank-ai-app"
APP_DIR        = f"/Workspace/Users/{user}/databank-ai-lab/app"


def get_superagent_endpoint_name(superagent_name):
    _w = WorkspaceClient()
    my_email = _w.current_user.me().user_name
    all_eps  = list(_w.serving_endpoints.list())

    # 1. Exact endpoint name match
    endpoint_name = next((ep.name for ep in all_eps if ep.name == superagent_name), None)

    # 2. Fuzzy match — normalised agent name appears inside an endpoint name
    if not endpoint_name:
        normalise = lambda s: s.lower().replace("-", "").replace("_", "").replace(" ", "")
        term = normalise(superagent_name)
        endpoint_name = next((ep.name for ep in all_eps if term in normalise(ep.name)), None)

    # 3. AgentBricks fallback — MAS endpoints owned by the current user
    if not endpoint_name:
        my_mas_eps = [ep.name for ep in all_eps
                      if ep.name.startswith("mas-") and ep.name.endswith("-endpoint")
                      and getattr(ep, "creator", None) == my_email]
        if len(my_mas_eps) == 1:
            endpoint_name = my_mas_eps[0]
            print(f"✅ Matched your MAS endpoint by creator ({my_email})")
        elif len(my_mas_eps) > 1:
            print(f"⚠️  Multiple MAS endpoints found for {my_email}:")
            for n in my_mas_eps:
                print(f"     {n}")
            print("   Using the first one. Set ENDPOINTNAME manually if this is wrong.")
            endpoint_name = my_mas_eps[0]

    if endpoint_name:
        print(f"Supervisor Agent Name : {superagent_name}")
        print(f"Endpoint Name         : {endpoint_name}")
    else:
        print(f"⚠️  Could not resolve endpoint for '{superagent_name}'")
        print(f"   No MAS endpoints found for creator {my_email}.")
        print(f"   Create the Supervisor Agent first (Step 3a), then re-run.")

    return endpoint_name

endpoint_name= get_superagent_endpoint_name(AGENT_ENDPOINT)

HOST = WorkspaceClient().config.host

print(f"🚀 App Name    : {APP_NAME}")
print(f"📁 App Dir     : {APP_DIR}")
print(f"🏠 Workspace  : {HOST}")

print(f"🤖 Supervise Agent Name   : {AGENT_ENDPOINT}")
print(f"🤖 Endpoint   : {endpoint_name}")


# COMMAND ----------

# DBTITLE 1,Step 1 — Write app.py (Gradio Chat Interface)
# Write the Gradio app source to the workspace filesystem
import os
os.makedirs(APP_DIR, exist_ok=True)

app_py_content = '''
import os
import requests
import gradio as gr
from databricks.sdk.core import Config

# gradio_client._json_schema_to_python_type() does not handle boolean JSON Schemas
# (True/False), which are valid per spec (e.g. additionalProperties: true) but cause
# TypeError / APIInfoParseError on every request. Patch the private recursive function
# so it returns "Any" for any non-dict schema instead of crashing.
from gradio_client import utils as _gc_utils
_orig_parse = _gc_utils._json_schema_to_python_type
def _safe_parse(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_parse(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_parse

# Authentication: Databricks Apps auto-injects OAuth credentials via DATABRICKS_CLIENT_ID / SECRET.
cfg = Config()


def _get_headers():
    """Return fresh auth headers on every call — prevents token expiry in long-running sessions."""
    token = cfg.authenticate().get("Authorization", "").replace("Bearer ", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


AGENT_ENDPOINT_NAME = os.getenv("AGENT_ENDPOINT_NAME", "__AGENT_ENDPOINT_PLACEHOLDER__")
AI_GATEWAY_ENDPOINT = os.getenv("AI_GATEWAY_ENDPOINT", "__AIGATEWAY_ENDPOINT_PLACEHOLDER__")  # AI Gateway endpoint name


# ================================================================
# AI GATEWAY MODE
# When AI_GATEWAY_ENDPOINT is set, uses DatabricksOpenAI with
# use_ai_gateway=True to govern LLM usage via Unity AI Gateway.
# Ref: https://docs.databricks.com/aws/en/agents/custom-agents/author-agent
# ================================================================
_ai_gw_client = None
if AI_GATEWAY_ENDPOINT:
    try:
        from databricks_openai import DatabricksOpenAI
        _ai_gw_client = DatabricksOpenAI(use_ai_gateway=True)
        print(f"\u2705 AI Gateway mode enabled: {AI_GATEWAY_ENDPOINT}")
    except ImportError:
        print("\u26a0\ufe0f  databricks-openai not installed — falling back to direct agent mode")


def _chat_via_gateway(message: str, history: list) -> str:
    """Call an LLM through the Databricks AI Gateway using DatabricksOpenAI."""
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = _ai_gw_client.chat.completions.create(
            model=AI_GATEWAY_ENDPOINT,
            messages=messages,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"\u274c AI Gateway error: {e}"


def _chat_via_agent(message: str, history: list) -> str:
    """Call the AgentBricks supervisor endpoint directly."""
    input_messages = []
    for user_msg, assistant_msg in history:
        input_messages.append({"role": "user",      "content": user_msg})
        input_messages.append({"role": "assistant", "content": assistant_msg})
    input_messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            f"{cfg.host}/serving-endpoints/{AGENT_ENDPOINT}/invocations",
            headers=_get_headers(),
            json={"input": input_messages},
            timeout=120
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return f"\u274c Endpoint error ({resp.status_code}): {resp.text[:300] or str(e)}"
    except Exception as e:
        return f"\u274c Request failed: {e}"

    try:
        data = resp.json()
    except Exception:
        return f"\u274c Endpoint returned non-JSON (status {resp.status_code}): {resp.text[:300]}"

    answer = None
    output_events = data.get("output", [])
    if isinstance(output_events, list):
        for event in reversed(output_events):
            if event.get("type") == "message" and event.get("role") == "assistant":
                for block in event.get("content", []):
                    if block.get("type") == "output_text":
                        answer = block.get("text", "").strip()
                        break
            if answer:
                break
    if not answer:
        answer = (data.get("choices", [{}])[0].get("message", {}).get("content") or str(data))
    return answer


def chat(message: str, history: list) -> str:
    """Route to AI Gateway (if available) or direct agent endpoint."""
    if _ai_gw_client and AI_GATEWAY_ENDPOINT:
        return _chat_via_gateway(message, history)
    return _chat_via_agent(message, history)


# Gradio Chat Interface
with gr.Blocks(
    title="DataBank AI Advisor",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css=".gradio-container { max-width: 900px !important; margin: auto; }"
) as demo:

    gr.Markdown("""
    # \U0001f3e6 DataBank AI Advisor
    **Your intelligent financial advisor assistant**

    Ask me about:
    - \U0001f4bc Customer portfolios and account details
    - \U0001f4ca Risk scores and investment suitability
    - \u26a0\ufe0f Suspicious transactions and fraud checks
    - \U0001f4c4 DataBank product features and eligibility
    """)

    chatbot = gr.Chatbot(
        label="DataBank AI Advisor Chat",
        height=500,
        show_label=True,
        avatar_images=(None, "\U0001f3e6")
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. 'Review portfolio for CUST-0001' or 'What ISA products do we offer?'",
            label="Your question",
            scale=4
        )
        send_btn = gr.Button("\U0001f50d Ask", variant="primary", scale=1)

    gr.Examples(
        examples=[
            "Give me a portfolio overview for CUST-0001",
            "Check for suspicious transactions on CUST-0042 in the last 90 days",
            "What investment products are suitable for a conservative investor with 10,000?",
            "A customer wants to consolidate 15,000 of debt. What loan options do we have?",
            "How many customers have an aggressive risk profile?"
        ],
        inputs=msg_box,
        label="Quick start examples"
    )

    gr.Markdown("---")
    gr.Markdown(
        "*DataBank AI Lab | Powered by Databricks AgentBricks, AI Gateway, and Vector Search*",
    )

    def respond(message, history):
        bot_message = chat(message, history)
        history.append((message, bot_message))
        return "", history

    msg_box.submit(respond, [msg_box, chatbot], [msg_box, chatbot])
    send_btn.click(respond, [msg_box, chatbot], [msg_box, chatbot])


if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("DATABRICKS_APP_PORT", 8000)), show_api=False)
'''

# Inject the actual endpoint name into the template
app_py_content = app_py_content.replace("__AGENT_ENDPOINT_PLACEHOLDER__", endpoint_name)

# Inject the actual AI Gatway endpoint name into the template
app_py_content = app_py_content.replace("__AIGATEWAY_ENDPOINT_PLACEHOLDER__", AI_GW_ROUTE)


with open(f"{APP_DIR}/app.py", "w") as f:
    f.write(app_py_content)

print(f"\u2705 app.py written to {APP_DIR}/app.py")

# COMMAND ----------

# DBTITLE 1,Step 2 — Write app.yaml (App Configuration)
# Set AI_GATEWAY_ENDPOINT to route through AI Gateway (with rate limits, guardrails, usage tracking)
# Leave empty to use the direct agent endpoint (default)
AI_GATEWAY_ENDPOINT_NAME = "databricks-gemma-3-12b"  # System FM endpoint with AI Gateway

app_yaml_content = f'''# Databricks Apps configuration
# Docs: https://docs.databricks.com/dev-tools/databricks-apps/

command:
  - python
  - app.py

env:
  - name: AGENT_ENDPOINT_NAME
    value: {endpoint_name}
  - name: AI_GATEWAY_ENDPOINT
    value: {CATALOG}.{SCHEMA}.{AI_GW_ROUTE}

resources:
  - name: agent-serving-endpoint
    serving_endpoint:
      name: {endpoint_name}
      permission: CAN_QUERY
  - name: ai-gateway-endpoint
    serving_endpoint:
      name: {CATALOG}.{SCHEMA}.{AI_GW_ROUTE}
      permission: CAN_QUERY
'''

with open(f"{APP_DIR}/app.yaml", "w") as f:
    f.write(app_yaml_content)

print(f"✅ app.yaml written to {APP_DIR}/app.yaml")
print()
print("App files ready:")
for fname in os.listdir(APP_DIR):
    size = os.path.getsize(f"{APP_DIR}/{fname}")
    print(f"  {fname} ({size} bytes)")

# COMMAND ----------

# DBTITLE 1,Step 3 - Create requirements.txt
requirementtext = f'''databricks-openai
openai'''

with open(f"{APP_DIR}/requirements.txt", "w") as f:
    f.write(requirementtext)

print(f"✅ requirements.txt written to {APP_DIR}/requirements.txt")
print()
print("Requirement files ready:")
for fname in os.listdir(APP_DIR):
    size = os.path.getsize(f"{APP_DIR}/{fname}")
    print(f"  {fname} ({size} bytes)")

# COMMAND ----------

# DBTITLE 1,Step 3 — Deploy the App
# MAGIC %md
# MAGIC ## 🚀 Step 3: Deploy to Databricks Apps
# MAGIC
# MAGIC Run the cell below to deploy the app using the Databricks CLI.
# MAGIC
# MAGIC **What happens during deployment:**
# MAGIC 1. The `app.py` and `app.yaml` files are packaged
# MAGIC 2. Databricks provisions a serverless Python 3.11 runtime
# MAGIC 3. The app is given OAuth credentials for the workspace
# MAGIC 4. A unique HTTPS URL is generated
# MAGIC 5. The app goes live in ~30-60 seconds
# MAGIC
# MAGIC > 🔐 **Authentication**: The app uses **service principal OAuth** (app auth) by default.
# MAGIC > The agent endpoint is granted `CAN_QUERY` permission via `app.yaml`.

# COMMAND ----------

# DBTITLE 1,Step 4 — Deploy via CLI
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment

w = WorkspaceClient()

print(f"🚀 Deploying app : {APP_NAME}")
print(f"   Source        : {APP_DIR}")
print(f"   Waiting for deployment to complete...")
print()

try:
    deployment = w.apps.deploy(
        app_name=APP_NAME,
        app_deployment=AppDeployment(source_code_path=APP_DIR)
    )
    print(f"✅ Deployment started")
    print(f"   Deployment ID : {deployment.deployment_id}")
    print(f"\n⏳ Run the next cell to wait for the app to be ready.")
except Exception as e:
    print(f"❌ Deployment error: {e}")
    print()
    print("💡 Deploy via UI instead:")
    print("   Left sidebar → Apps → databank-ai-advisor-app → Deploy")
    print(f"   Source : {APP_DIR}")
    print(f"   Name   : {APP_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Get App URL and Test
import time

def wait_for_app(app_name: str, timeout: int = 120) -> str:
    """Wait for the Databricks App to be running and return its URL."""
    print(f"⏳ Waiting for app '{app_name}'...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            app = w.apps.get(name=app_name)
            status = app.compute_status.state if app.compute_status else None
            print(f"   Status: {status}   ", end="\r")
            if str(status) in ("AppState.RUNNING", "RUNNING"):
                url = app.url
                print(f"\n✅ App is RUNNING")
                print(f"   URL: {url}")
                return url
            if str(status) in ("AppState.CRASHED", "CRASHED", "ERROR"):
                print(f"\n❌ App crashed: {app.compute_status.message if app.compute_status else 'unknown error'}")
                return ""
        except Exception as e:
            print(f"   Polling... ({e})\r")
        time.sleep(10)
    print(f"\n⚠️  App not yet ready after {timeout}s. Check the Apps page in the sidebar.")
    return ""

app_url = wait_for_app(APP_NAME)

if app_url:
    print()
    print("=" * 65)
    print(f"🏦 DataBank AI Advisor is LIVE!")
    print(f"   Open in browser: {app_url}")
    print()
    print("   Try these example questions:")
    print("   1. Give me a portfolio overview for CUST-0001")
    print("   2. Check for suspicious transactions on CUST-0042")
    print("   3. What investment products suit a conservative investor?")
    print("=" * 65)

# COMMAND ----------

# DBTITLE 1,Module 08 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 08 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | `app.py` generated | Gradio chat UI code at `{APP_DIR}/app.py` |
# MAGIC | `app.yaml` generated | App config with endpoint resource |
# MAGIC | App deployed | `databank-ai-advisor-app` visible in Apps sidebar |
# MAGIC | App URL accessible | Opens in browser with DataBank branding |
# MAGIC | Chat works | Agent responds to portfolio/product/fraud queries |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛡️ Security Note
# MAGIC The app uses **app-level OAuth** (service principal). The agent endpoint credentials are injected automatically. No tokens are hardcoded.
# MAGIC
# MAGIC ### 🔄 Stopping the App
# MAGIC ```
# MAGIC databricks apps stop databank-ai-advisor-app
# MAGIC ```
# MAGIC Apps are stopped automatically after **3 days** of inactivity per workspace policy.
# MAGIC
# MAGIC ### 🚀 Next: Module 09 — Evaluation
# MAGIC Open **`09_evaluation_llm_judge`** to measure agent response quality using LLM-as-a-judge.