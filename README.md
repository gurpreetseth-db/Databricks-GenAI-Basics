# 🏦 DataBank AI Lab — Facilitator & Participant Guide

**Duration:** ~4.5 hours | **Level:** Mixed (beginner → intermediate)
**Products Covered:** Unity Catalog · Foundation Models API · AI Gateway · Vector Search · UC Functions · Genie Space · MLflow · AgentBricks · Databricks Apps · MLflow GenAI Evaluation & Labeling

---

## Overview

Participants play the role of engineers at a fictional retail bank called **DataBank**. Over the course of the lab they build a production-grade AI assistant from scratch — starting with raw synthetic data and ending with a deployed chat application that is evaluated and human-reviewed using MLflow's GenAI tooling.

The finished assistant helps financial advisors:
- 🔍 Query customer transactions and portfolios in natural language (Genie)
- 📄 Search product brochures, FAQs, and T&Cs instantly (Vector Search + Knowledge Assistant)
- ⚠️ Detect suspicious transactions and calculate risk scores (UC Functions)
- 💬 Get product recommendations — all behind an AI Gateway governance layer

Every module is a standalone Databricks notebook. Each notebook runs `%run ./Config_Parameters` first, so all assets are namespaced per user (`<username>_databank_lab`) and a whole class can share a single workspace without collisions.

---

## Getting Started

1. **Clone the repo** (locally, or as a Databricks Git folder):
   ```bash
   git clone https://github.com/gurpreetseth-db/Databricks-GenAI-Basics.git
   cd Databricks-GenAI-Basics
   ```
2. **Create your config from the template.** `Config_Parameters.py` is gitignored
   so your values stay local — copy the template notebook and edit it:
   ```bash
   cp Config_Parameters-Sample.py Config_Parameters.py
   ```
   In the Databricks workspace, open **`Config_Parameters-Sample`** and
   **File → Clone** it to `Config_Parameters`. Most values derive automatically
   from your username; review `CATALOG_STORAGE`,
   the model names, `GENIE_SPACE_ID`, `LAKEBASE_PROJECT`, and `APP_NAME`. (Keep
   these in sync with the `variables:` defaults in `databricks.yml`, or override
   at deploy time with `--var`.)
3. **Run the lab notebooks** `00 → 10` interactively to create the assets
   (catalog/schema, UC functions, Vector Search index, Genie space, experiment).
4. **Deploy the chat app** with the bundle — see
   [Deploy with Databricks Asset Bundles (DAB)](#deploy-with-databricks-asset-bundles-dab).

> The bundle still uploads your local `Config_Parameters.py` to the workspace
> (via `sync.include` in `databricks.yml`) even though it's gitignored, so the
> notebooks' `%run ./Config_Parameters` keeps working.

---

## Prerequisites Checklist (Pre-Lab Setup)

Complete these **before the lab starts**:

### For Each Participant
- [ ] Databricks workspace with **serverless compute** enabled
- [ ] **Unity Catalog** enabled on the workspace
- [ ] **Foundation Models API** access (Settings → Admin Console → Feature flags)
- [ ] Permission to **create a Vector Search endpoint** (or an existing endpoint to reuse — Module 00 auto-falls-back to one if the quota is hit)
- [ ] `CREATE CATALOG` privilege (each participant gets their own `<username>_databank_lab`)

### For the Facilitator
- [ ] A running **SQL warehouse** (e.g. `Shared Serverless`)
- [ ] **AI Gateway** feature enabled (Serving → AI Gateway tab visible) — needed for Module 02 and the bonus notebook
- [ ] For AgentBricks (Module 07): confirm **Budget Policy** Public Preview compatibility with your entitlement layer (see the ⚠️ note at the top of `07_agentbricks_agent`)

---

## Lab Modules At A Glance

| Module | File | Duration | Key Output |
|--------|------|----------|-----------|
| — | `Config_Parameters` | ~5 min | Central config: catalog/schema/volume names, model IDs, endpoint names (run via `%run` by every module) |
| 00 | `00_setup_prerequisites` | ~10 min | Packages · Catalog · Schema · Volume · Vector Search endpoint · MLflow experiment · FM API smoke test |
| 01 | `01_data_generation` | ~25 min | 5 Delta tables + 7 PDFs in a UC Volume |
| 02 | `02_ai_gateway_setup` | ~20 min | AI Gateway route with rate limits, guardrails, routing, fallback & usage logging |
| 03 | `03_uc_functions` | ~25 min | 3 UC Functions (risk score, portfolio summary, fraud flag) |
| 04 | `04_vector_search` | ~30 min | VS endpoint + semantic index over the product PDFs |
| 05 | `05_genie_space` | ~15 min | Natural-language SQL over the 5 financial tables |
| 06 | `06_ml_experiment` | ~20 min | MLflow experiment comparing prompt / model variants |
| 07 | `07_agentbricks_agent` | ~40 min | Deployed AgentBricks **Supervisor Agent** (Knowledge Assistant + Genie + 3 UC Functions) |
| 08 | `08_Playground_Deploy_App.ipynb` | ~25 min | Test in **AI Playground**, then **Export to Databricks Apps** → live chat app wired to the AI Gateway |
| 09 | `09_evaluation_llm_judge` | ~20 min | Evaluation run with LLM-as-a-judge scorers |
| 10 | `10_evaluation_labels` | ~15 min | Label schemas · human labeling session · review workflow |
| ★ | `UC_AI_Gateway_Complete_Demo` | ~45 min | **Bonus:** end-to-end Unity AI Gateway **Model Services** demo (UC-governed model API) |

> **Note:** `Dont Use - 08_databricks_app.py` is the deprecated hand-coded Gradio version of Module 08. It is kept for reference only — use `08_Playground_Deploy_App.ipynb` instead.

---

## Assets Created (from `Config_Parameters`)

All names are prefixed with your sanitized username so multiple people can share one workspace:

| Asset | Name pattern |
|-------|--------------|
| Catalog | `<username>_databank_lab` |
| Schema | `<username>_financial_data` |
| Volume | `/Volumes/<catalog>/<schema>/documents` |
| MLflow experiment | `/Users/<user>/<username>_databank_ai_lab` |
| Vector Search endpoint | `<username>-vs-endpoint` |
| Vector Search index | `<catalog>.<schema>.product-docs-index` |
| AI Gateway route | `<username>-databank-llm-route` |
| Agent endpoint | `<username>-databank-ai-advisor` |
| Genie Space | `<username>-DataBank-Financial-Advisor` |

**Models used** (all Databricks-hosted — no external API keys):
- Foundation model: `system.ai.gemma-3-12b`
- Embedding model: `databricks-gte-large-en`
- Fallback model: `system.ai.databricks-kimi-k3`

---

## Timing Guide

```
09:00  Introduction + workspace setup check       (10 min)
09:10  Module 00: Setup                           (10 min)
09:20  Module 01: Data Generation                 (25 min)
09:45  Module 02: AI Gateway                      (20 min)
10:05  Module 03: UC Functions                    (25 min)
10:30  Break                                      (10 min)
10:40  Module 04: Vector Search                   (30 min)
11:10  Module 05: Genie Space                     (15 min)
11:25  Module 06: ML Experiment                   (20 min)
11:45  Lunch / Break                              (15 min)
12:00  Module 07: AgentBricks                     (40 min)
12:40  Module 08: Playground → Databricks App     (25 min)
13:05  Module 09: LLM-as-a-Judge Evaluation       (20 min)
13:25  Module 10: Labels & Human Review           (15 min)
13:40  Demo + Q&A / Bonus: UC AI Gateway          (20 min)
14:00  END
```

---

## The Dataset (Module 01)

| Table | Rows | Description |
|-------|------|-------------|
| `products` | 25 | Product catalogue (Savings, Loans, Investments, Insurance, Credit Cards) |
| `customers` | 500 | Synthetic UK customer profiles with risk profiles and income |
| `accounts` | 500 | Customer account relationships to products |
| `transactions` | 10,000 | Banking transactions with fraud labels |
| `support_tickets` | 300 | Support interactions with resolutions |

Plus **7 PDFs** in the UC Volume: 5 product brochures (one per category), 1 FAQ, and 1 Terms & Conditions document.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DataBank AI Lab Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│  User Browser                                                     │
│      ↓                                                            │
│  Databricks App (Playground export) [Module 08]                   │
│      ↓ /invocations                                               │
│  AI Gateway Route [Module 02] ← rate limits, guardrails, logging  │
│      ↓                                                            │
│  AgentBricks Supervisor Agent [Module 07]                         │
│      ├── Knowledge Assistant ← Vector Search index [Module 04]    │
│      │        ↑ product PDFs in /Volumes/.../documents            │
│      ├── Genie Space ← customers/accounts/transactions [Module 05]│
│      ├── UC Function: calculate_customer_risk       [Module 03]   │
│      ├── UC Function: get_portfolio_summary         [Module 03]   │
│      └── UC Function: flag_suspicious_transactions  [Module 03]   │
│                                                                   │
│  Evaluation & Quality Loop                                        │
│      ├── MLflow Experiment [Module 06] ← prompt/model variants    │
│      ├── LLM-as-a-Judge evaluation [Module 09]                    │
│      └── MLflow labeling & human review [Module 10]               │
│                                                                   │
│  Foundation: Unity Catalog (<username>_databank_lab)              │
│      └── financial_data schema: 5 tables + documents volume       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bonus: Unity AI Gateway — Model Services

`UC_AI_Gateway_Complete_Demo.py` is a standalone, self-contained demo (audience: platform engineers, data leaders, security teams) covering the **latest** Unity AI Gateway primitive — the **Model Service** (a.k.a. Model API):

- A Unity Catalog **securable** at `catalog.schema.name`, managed via `/api/2.1/unity-catalog/model-services`
- Governed with the **same grants as tables/functions** (`EXECUTE` + `USE CATALOG` + `USE SCHEMA`)
- Multi-destination **routing** with traffic splitting, **service policies** (guardrails that *enforce*), **rate limits**, and per-request **usage logging** to a Delta table

It contrasts this new API against the older serving-endpoint `ai_gateway` block. Use it as a deeper-dive after the core lab or as a governance-focused demo on its own.

---

## Common Issues & Fixes

### Module 00 — Setup
| Issue | Fix |
|-------|-----|
| `Cannot create catalog` | User needs `CREATE CATALOG` privilege — check with workspace admin |
| FM API 404 | Foundation Models API not enabled — Admin Console → Feature Flags |
| VS endpoint quota exceeded | Module auto-falls-back to an existing endpoint and prints its name — update `AI_VECTOR_SEARCH_ENDPOINT` in `Config_Parameters` if you want a specific one |

### Module 01 — Data Generation
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: faker` | Re-run the `%pip install` cell in Module 00; allow the kernel to restart |
| PDF write fails | Confirm the volume path `/Volumes/<catalog>/<schema>/documents` exists (Module 00 creates it) |

### Module 02 — AI Gateway
| Issue | Fix |
|-------|-----|
| AI Gateway tab not visible | Feature may not be enabled — enable it or use the UI option |
| `ExternalModelProvider` not found | `databricks-sdk` too old — `%pip install databricks-sdk --upgrade` |

### Module 04 — Vector Search
| Issue | Fix |
|-------|-----|
| Endpoint creation fails | User needs compute permissions — check with workspace admin |
| Endpoint already exists | Module is idempotent — the existing endpoint is reused |
| Index sync timeout | Wait 2–3 minutes, then re-run the search test cell |

### Module 07 — AgentBricks
| Issue | Fix |
|-------|-----|
| Budget Policy incompatibility | Disable Budget Policy Public Preview if it conflicts with your entitlement layer (see notebook header) |
| API 404 on AgentBricks endpoint | Fall back to the UI (AgentBricks sidebar) |
| Genie Space ID missing | Grab it from the URL after finishing Module 05 |
| `messages` field rejected | AgentBricks endpoints expect `input`, not `messages` — `requests.post(..., json={"input": [...]})` |
| `w.config.token` is `None` | On serverless OAuth use `w.config.authenticate().get("Authorization", "").replace("Bearer ", "")` |

### Module 08 — Playground → Databricks App
| Issue | Fix |
|-------|-----|
| App response empty / JSON error | AgentBricks returns an event stream under `output[].content[].text` — parse `output`, not `.choices[0].message.content` |
| Wire app to AI Gateway | In the exported `agent.py`, set `use_ai_gateway=True` on `AsyncDatabricksOpenAI` and point the model name at your AI Gateway route |
| App crashes on start | Check logs: `databricks apps logs <app-name>` |

### Module 09 — LLM-as-a-Judge
| Issue | Fix |
|-------|-----|
| `evaluate() got unexpected keyword 'inputs_col'` | Not valid for `mlflow.genai.evaluate()` — remove it |
| `'inputs' column must be a dictionary` | Wrap: `df['inputs'] = df['inputs'].apply(lambda q: {"question": q})` and accept `question` in `predict_fn` |
| `Correctness` scorer returns null | Add `df['expectations'] = df['expected_response'].apply(lambda a: {"expected_response": a})` |

### Module 10 — Labels & Human Review
| Issue | Fix |
|-------|-----|
| `mlflow.load_table("eval_results")` fails | `evaluate()` doesn't persist a table — load from traces via `mlflow.search_traces(locations=[...])` |
| `experiment_ids` FutureWarning | Use `locations=[...]` instead of `experiment_ids=[...]` |
| Iterating `search_traces()` yields column names | It returns a DataFrame — use `traces.itertuples()` |
| `mlflow.genai.label()` does not exist | Use `mlflow.log_feedback(trace_id=..., name=..., value=..., rationale=...)` |

---

## Key Variables Participants Should Record

| Variable | Where to get it | Used in |
|----------|----------------|---------|
| `GENIE_SPACE_ID` | Module 05 → URL bar | Module 07 |
| `AGENT_ENDPOINT` | Module 07 → Serving → Endpoints | Modules 08, 09 |
| `APP_URL` | Module 08 export output | Demo |
| `LABELING_SESSION_URL` | Module 10 output | Human review |

---

## Deploy with Databricks Asset Bundles (DAB)

The whole repo is a **Databricks Asset Bundle** (`databricks.yml`). One command
uploads the entire codebase (notebooks, `Config_Parameters.py`, `lib/`, `img/`,
and the `databank-chat-demo` app) to the workspace and provisions the
**`databank-chat-demo`** Databricks App (+ its UC-function / Vector-Search /
Genie resource bindings).

> Run the lab notebooks (00→10) **yourself, interactively** to create the
> underlying assets (catalog/schema, UC functions, Vector Search index, Genie
> space, MLflow experiment), then deploy the app with the bundle.

### The three memory modes are bundle *targets*

| Target | App memory |
|--------|-----------|
| `dev_simple`    | none (ephemeral) |
| `dev_shortterm` | this browser session only (Lakebase) |
| `dev_longterm`  | per-user history + durable fact recall (Lakebase) |

### Deploy (from the repo root)

```bash
# (first: run notebooks 00→10 interactively to create the lab assets)

# 1) Deploy code + app for the mode you want:
databricks bundle deploy -t dev_longterm -p Myenv

# 2) Apply the app env + grants DAB can't express, then redeploy the app:
python databank-chat-demo/deploy_app.py -t dev_longterm -p Myenv
```

Or run the **`12_deploy_chat_app.py`** notebook in the workspace: pick the `mode`
widget and Run All — it performs both steps for you.

### Why the two-step deploy?

`databricks bundle deploy` creates the app + service principal and syncs the
code. `deploy_app.py` then applies what the bundle schema can't express:

- the app **env** (incl. the live-resolved Lakebase host and the app SP as `PGUSER`),
- **EXECUTE** on the AI Gateway **model-service** route (the `uc_securable` app
  resource type only supports `VOLUME/TABLE/FUNCTION/CONNECTION`),
- catalog **USE** + schema **USE/SELECT/EXECUTE** (table-level SELECT for Genie SQL),
- a **federated Lakebase login role** for the app SP (`shortterm`/`longterm` only).

### Config = single source of truth

Names/derivations live in `databricks.yml` `variables:` (mirroring
`Config_Parameters.py`). `deploy_app.py` reads the *resolved* values from
`databricks bundle summary`, so the two never drift. For another user or
workspace, override at deploy time — e.g. `--var username=jdoe`.

> **Note:** the lab notebooks are meant to be run **interactively** (some, like
> `05_genie_space` and `08_Playground_Deploy_App`, involve manual UI steps). The
> bundle deploys code + the app; it does not run the notebooks for you.

---

## Files in This Repo

```
Databricks-GenAI-Basics/
├── README.md                          ← This guide
├── databricks.yml                     ← Databricks Asset Bundle (deploys the whole repo + app)
├── Config_Parameters-Sample.py        ← Template notebook — copy to Config_Parameters.py (gitignored) and edit
├── .gitignore                         ← Ignores Config_Parameters.py, __pycache__, .databricks/, …
├── 00_setup_prerequisites.py          ← Packages, catalog, schema, volume, VS endpoint, MLflow, FM API test
├── 01_data_generation.py              ← Synthetic data (5 tables) + 7 PDFs
├── 02_ai_gateway_setup.py             ← AI Gateway route (rate limits, guardrails, routing, fallback, logging)
├── 03_uc_functions.py                 ← 3 UC Functions (risk, portfolio, fraud)
├── 04_vector_search.py                ← VS endpoint + index over product PDFs
├── 05_genie_space.py                  ← Genie Space over the financial tables
├── 06_ml_experiment.py                ← MLflow experiment tracking (prompt/model variants)
├── 07_agentbricks_agent.py            ← AgentBricks Supervisor Agent assembly & deploy
├── 08_Playground_Deploy_App.ipynb     ← AI Playground testing → Export to Databricks Apps
├── 09_evaluation_llm_judge.py         ← LLM-as-a-judge evaluation
├── 10_evaluation_labels.py            ← Label schemas, human labeling, review workflow
├── 12_deploy_chat_app.py              ← Notebook that drives the bundle deploy (pick a memory mode)
├── UC_AI_Gateway_Complete_Demo.py     ← Bonus: Unity AI Gateway Model Services demo
├── img/                               ← Screenshots referenced by the notebooks
├── lib/                               ← Helper package
│   ├── __init__.py
│   ├── workspace_links.py             ← Builds workspace UI URLs for setup validation
│   └── provisioning.py.ipynb          ← Provisioning helpers
└── databank-chat-demo/                ← 3-mode chat app (deployed by the bundle)
    ├── app.yaml                       ← Base app spec (deploy_app.py writes the full per-mode env)
    ├── deploy_app.py                  ← Post-deploy configurator (env + grants DAB can't express)
    ├── agent_server/                  ← OpenAI Agents SDK backend (UC funcs, Genie, Vector Search)
    └── e2e-chatbot-app-next/          ← Vercel AI SDK frontend + Lakebase chat history
```

---

*DataBank AI Lab | Updated October 2026 — now a Databricks Asset Bundle: `databricks bundle deploy` ships the whole codebase + the 3-mode `databank-chat-demo` app + a lab-setup job, with memory mode selected per bundle target.*
