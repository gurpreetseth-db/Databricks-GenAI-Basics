# DataBank Chat Demo — three memory modes

A single Databricks App (`databank-chat-demo`) that demonstrates a financial-advisor
assistant over the same UC AI Gateway route (Claude Sonnet 4.5), UC functions, Genie
Agent, and Vector Search index — in one of three selectable **memory modes**.

| Mode | Chat history | Agent memory |
|------|--------------|--------------|
| `simple`    | none (ephemeral)                     | none |
| `shortterm` | this session only (resets on reopen) | conversation log in Lakebase |
| `longterm`  | all past sessions (per user)         | + durable fact recall across sessions |

## Deploy (Databricks Asset Bundle)

This app ships as part of the repo-root **bundle** (`../databricks.yml`). The memory
mode is a bundle **target** (`dev_simple` / `dev_shortterm` / `dev_longterm`). From the
repo root:

```bash
# 1) DAB creates the app + SP and syncs the code:
databricks bundle deploy -t dev_longterm -p <profile>

# 2) deploy_app.py applies the env + grants DAB can't express, then redeploys:
python databank-chat-demo/deploy_app.py -t dev_longterm -p <profile>
```

Or run `../12_deploy_chat_app.py` in the workspace and pick the `mode` widget (it does
both steps). Re-deploying a different target retargets the **same** app to that mode.

`deploy_app.py` reads the resolved config from `databricks bundle summary`, so
`databricks.yml` stays the single source of truth (no values are hand-passed).

## Layout

- `agent_server/` — FastAPI + OpenAI Agents SDK backend. Reads config from env
  (`MEMORY_MODE`, `UC_CATALOG`, `MODEL_ROUTE`, …) written by `deploy_app.py`.
- `agent_server/memory_manager.py` — Lakebase-backed conversation log + long-term
  facts (short/long modes); auto-creates its schema.
- `e2e-chatbot-app-next/` — vendored chat UI. `shortterm` mode scopes history to a
  per-browser session via a session cookie (see `server/src/middleware/auth.ts`).
- `app.yaml` — committed **base** spec (start command + static env) that DAB deploys.
  `deploy_app.py` uploads the full per-mode `app.yaml` to the workspace copy; it does
  not modify the committed file.
- `deploy_app.py` — post-deploy configurator: reads resolved config from the bundle,
  grants the app SP model-service EXECUTE + catalog/schema access + a federated Lakebase
  role (short/long), writes the full `app.yaml`, and redeploys the app.

## Modes: implementation notes

- **Model** is the UC AI Gateway route (Claude Sonnet 4.5). Guardrails/rate-limits on
  the route apply to all modes.
- **short/long** connect to the autoscaling Lakebase project `agentic-memory`; the app
  service principal is granted `CONNECT`+`CREATE` at deploy time and the app creates its
  own schemas (`ai_chatbot` for the UI, `conversation_memory` for the agent).
