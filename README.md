# 🏦 DataBank AI Lab — Facilitator Guide

**Duration:** 4.5 hours | **Level:** Mixed (beginner to intermediate)
**Products Covered:** AgentBricks · AI Gateway · Databricks Apps · Vector Search · Unity Catalog Functions · Genie Space · MLflow · MLflow GenAI Labeling

---

## Overview

Participants are engineers at a fictional bank called **DataBank**. Over 4.5 hours they build a production-grade AI assistant from scratch — from raw synthetic data to a deployed Gradio web application, evaluated and reviewed with MLflow's GenAI evaluation and labeling tools. The assistant helps financial advisors query customer portfolios, search product documents, detect fraud, and recommend products.

---

## Prerequisites Checklist (Pre-Lab Setup)

Complete these **before the lab starts**:

### For Each Participant (own workspace required)
- [ ] Databricks workspace with serverless compute enabled
- [ ] Unity Catalog enabled on the workspace
- [ ] Databricks Foundation Models API access (Settings → Admin Console → Feature flags)
- [ ] Personal Access Token created (Settings → Developer → Access tokens)
- [ ] Permission to create a Vector Search endpoint (granted by workspace admin)

### For the Facilitator (field-eng workspace demo)
- [ ] `Shared Serverless` SQL warehouse is running
- [ ] `databank_lab` catalog has been pre-created with RemoveAfter tag (optional — Module 00 does this)
- [ ] AI Gateway feature enabled in workspace (Serving → AI Gateway tab visible)

---

## Lab Modules At A Glance

| Module | File | Duration | Key Output |
|--------|------|----------|-----------|
| 00 | `00_setup_prerequisites` | 15 min | Catalog · Schema · Volume · Packages |
| 01 | `01_data_generation` | 25 min | 5 Delta tables + 7 PDFs in volume |
| 02 | `02_ai_gateway_setup` | 20 min | AI Gateway route with rate limits + guardrails |
| 03 | `03_uc_functions` | 25 min | 3 UC Functions (risk, portfolio, fraud) |
| 04 | `04_vector_search` | 30 min | VS endpoint + semantic search index over product PDFs |
| 05 | `05_genie_space` | 15 min | Natural language SQL on financial tables |
| 06 | `06_ml_experiment` | 20 min | MLflow experiment comparing 6 prompt variants |
| 07 | `07_agentbricks_agent` | 40 min | Deployed Supervisor Agent with 5 tools |
| 08 | `08_databricks_app` | 25 min | Live Gradio chat application |
| 09 | `09_evaluation_llm_judge` | 20 min | 25-question evaluation with LLM-as-a-judge |
| 10 | `10_evaluation_labels` | 20 min | Label schemas · Labeling session · Human review workflow |
| — | Buffer / Q&A | 5 min | — |

---

## Timing Guide

```
09:00  Introduction + workspace setup check       (10 min)
09:10  Module 00: Setup                           (15 min)
09:25  Module 01: Data Generation                 (25 min)
09:50  Module 02: AI Gateway                      (20 min)
10:10  Module 03: UC Functions                    (25 min)
10:35  Break                                      (10 min)
10:45  Module 04: Vector Search                   (30 min)
11:15  Module 05: Genie Space                     (15 min)
11:30  Module 06: ML Experiment                   (20 min)
11:50  Lunch / Break                              (10 min)
12:00  Module 07: AgentBricks                     (40 min)
12:40  Module 08: Databricks App                  (25 min)
13:05  Module 09: Evaluation                      (20 min)
13:25  Module 10: Labels & Human Review            (20 min)
13:45  Demo + Q&A                                 (15 min)
14:00  END
```

---

## Common Issues & Fixes

### Module 00
| Issue | Fix |
|-------|-----|
| `Cannot create catalog` | User needs `CREATE CATALOG` privilege — check workspace admin settings |
| FM API 404 | Foundation Models API not enabled — Admin Console → Feature Flags → Foundation Models |

### Module 01
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: faker` | Re-run the `%pip install` cell; allow kernel restart |
| `PERSIST TABLE is not supported` | Ensure `.cache()` is removed — serverless does not support it |
| PDF write fails | Check volume path: `/Volumes/databank_lab/financial_data/documents` must exist |

### Module 02
| Issue | Fix |
|-------|-----|
| AI Gateway tab not visible | Feature may not be enabled — use UI Option A |
| `ExternalModelProvider` not found | SDK version older than 0.23 — upgrade with `%pip install databricks-sdk --upgrade` |

### Module 04
| Issue | Fix |
|-------|-----|
| VS endpoint creation fails | User needs compute permissions — check workspace admin settings |
| Endpoint already exists | Module is idempotent — existing endpoint is reused automatically |
| Index sync timeout | Wait 2–3 minutes then re-run the search test cell |

### Module 07
| Issue | Fix |
|-------|-----|
| API 404 on AgentBricks endpoint | Use the UI (AgentBricks sidebar) if API endpoint not available |
| KA tile ID missing | Fill in `KA_TILE_ID` manually after creating KA in UI |
| Genie Space ID missing | Get from URL after completing Module 05 |
| `messages` field rejected by agent | AgentBricks endpoints require `input` not `messages` — use `requests.post()` with `json={"input": [...]}` |
| Supervisor endpoint name unknown | AgentBricks auto-names endpoints `mas-<uuid>-endpoint`. Use the lookup cell to resolve the display name to the endpoint name |
| `w.config.token` is `None` | On serverless OAuth use `w.config.authenticate().get("Authorization", "").replace("Bearer ", "")` |

### Module 08
| Issue | Fix |
|-------|-----|
| App crashes on start | Check `databricks apps logs databank-ai-advisor-app` |
| `Port conflict` | App must bind to `DATABRICKS_APP_PORT` — already handled in `app.py` |
| Agent endpoint 404 | Verify `AGENT_ENDPOINT_NAME` in `app.yaml` — it overrides any default set in `app.py` |
| `TypeError: argument of type 'bool' is not iterable` | Gradio ≥4.42 schema parser fails on boolean JSON schemas. Monkey-patch `gradio_client.utils.get_type` — see `app.py` |
| App response is empty / JSON decode error | AgentBricks returns an event stream under `output[].content[].text`. Parse the `output` list — do not use `.choices[0].message.content` |

### Module 09
| Issue | Fix |
|-------|-----|
| `evaluate() got unexpected keyword 'inputs_col'` | Remove `inputs_col` / `targets_col` — not valid for `mlflow.genai.evaluate()` |
| `'inputs' column must be a dictionary` | Wrap strings: `df['inputs'] = df['inputs'].apply(lambda q: {"question": q})` and accept `question` as a kwarg in `predict_fn` |
| `Correctness` scorer returns all null | Add `df['expectations'] = df['expected_response'].apply(lambda a: {"expected_response": a})` |
| `Error 400 — 'messages' field not supported` | Agent wrapper must use `requests.post()` with `json={"input": [...]}` |

### Module 10
| Issue | Fix |
|-------|-----|
| `mlflow.load_table("eval_results")` fails | `mlflow.genai.evaluate()` does not persist a table artifact — load from traces via `mlflow.search_traces(locations=[...])` |
| `experiment_ids` FutureWarning | Replace `experiment_ids=[...]` with `locations=[...]` in `mlflow.search_traces()` |
| `for t in traces` iterates column names | `search_traces()` returns a DataFrame — use `traces.itertuples()` to iterate rows |
| Assessment dicts have no `.name` attribute | Assessments are plain dicts — use `a.get("assessment_name")` and `a.get("feedback", {}).get("value")` |
| `mlflow.genai.label()` does not exist | Use `mlflow.log_feedback(trace_id=..., name=..., value=..., rationale=...)` |

---

## Key Variables Participants Need to Record

| Variable | Where to get it | Used in |
|----------|----------------|----------|
| `GENIE_SPACE_ID` | Module 05 → URL bar | Module 07 |
| `KA_TILE_ID` | Module 07 Step 1 response | Module 07 Step 2 |
| `AGENT_ENDPOINT` | Module 07 → Serving → Endpoints | Module 08, 09 |
| `APP_URL` | Module 08 Step 4 output | Demo |
| `LABELING_SESSION_URL` | Module 10 Step 6 output | Human review |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DataBank AI Lab Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Browser                                                     │
│      ↓                                                            │
│  Databricks App (Gradio) [Module 08]                              │
│      ↓ requests.post /invocations                                 │
│  AI Gateway Route (Module 02) ← rate limits, guardrails, logging  │
│      ↓                                                            │
│  AgentBricks Supervisor Agent (Module 07)                         │
│      ├── Knowledge Assistant ← /Volumes/.../documents [PDFs]     │
│      │       ↑ Vector Search Index [Module 04]                    │
│      ├── Genie Space ← customers/accounts/transactions [Module 05]│
│      ├── UC Function: calculate_customer_risk [Module 03]         │
│      ├── UC Function: get_portfolio_summary [Module 03]           │
│      └── UC Function: flag_suspicious_transactions [Module 03]   │
│                                                                   │
│  Evaluation & Quality Loop                                        │
│      ├── MLflow Experiment (Module 09) ← LLM-as-a-judge scores   │
│      └── MLflow Review App (Module 10) ← human label sessions    │
│                                                                   │
│  Foundation: Unity Catalog (databank_lab.financial_data)          │
│      ├── customers (500 rows)  [Module 01]                        │
│      ├── accounts (500 rows)   [Module 01]                        │
│      ├── transactions (10k)    [Module 01]                        │
│      ├── products (25 rows)    [Module 01]                        │
│      ├── support_tickets (300) [Module 01]                        │
│      └── /documents volume (7 PDFs) [Module 01]                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files in This Lab

```
databank-ai-lab/
├── README.md                     ← This file (facilitator guide)
├── PARTICIPANT_GUIDE.md          ← Participant reference card
├── 00_setup_prerequisites        ← Notebook: catalog, schema, volume, packages
├── 01_data_generation            ← Notebook: synthetic data + PDF generation
├── 02_ai_gateway_setup           ← Notebook: AI Gateway route setup
├── 03_uc_functions               ← Notebook: Unity Catalog function registration
├── 04_vector_search              ← Notebook: VS endpoint creation + vector index
├── 05_genie_space                ← Notebook: Genie Space setup
├── 06_ml_experiment              ← Notebook: MLflow experiment tracking
├── 07_agentbricks_agent          ← Notebook: Agent assembly and deployment
├── 08_databricks_app             ← Notebook: Gradio app deployment
├── 09_evaluation_llm_judge       ← Notebook: LLM-as-a-judge evaluation
├── 10_evaluation_labels          ← Notebook: Label schemas, human labeling, review workflow
└── app/
    ├── app.py                    ← Gradio application (AgentBricks-compatible)
    ├── app.yaml                  ← Databricks Apps config (env vars + resource bindings)
    └── requirements.txt          ← App Python dependencies
```

---

*DataBank AI Lab v1.1 | Updated July 2026 — added Module 10 (Evaluation Labels & Human Review)*
