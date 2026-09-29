# DataBank Chat Demo — three memory modes

A single Databricks App (`databank-chat-demo`) that demonstrates a financial-advisor
assistant over the same UC AI Gateway route (Claude Sonnet 4.5), UC functions, Genie
Agent, and Vector Search index — in one of three selectable **memory modes**.

| Mode | Chat history | Agent memory |
|------|--------------|--------------|
| `simple`    | none (ephemeral)                     | none |
| `shortterm` | this session only (resets on reopen) | conversation log in Lakebase |
| `longterm`  | all past sessions (per user)         | + durable fact recall across sessions |

## Deploy

From the calling notebook `../12_deploy_chat_app.py` (runs `Config_Parameters`, pick a
mode, deploy), or directly:

```bash
python deploy_app.py --mode simple    --profile <profile>
python deploy_app.py --mode shortterm --profile <profile>
python deploy_app.py --mode longterm  --profile <profile>
```

Re-running with a different `--mode` redeploys the same app in that mode.

## Layout

- `agent_server/` — FastAPI + OpenAI Agents SDK backend. Reads config from env
  (`MEMORY_MODE`, `UC_CATALOG`, `MODEL_ROUTE`, …) written by `deploy_app.py`.
- `agent_server/memory_manager.py` — Lakebase-backed conversation log + long-term
  facts (short/long modes); auto-creates its schema.
- `e2e-chatbot-app-next/` — vendored chat UI. `shortterm` mode scopes history to a
  per-browser session via a session cookie (see `server/src/middleware/auth.ts`).
- `deploy_app.py` — creates/updates the app + tool resources, generates `app.yaml`,
  grants Lakebase to the app service principal (short/long), syncs and deploys.

## Modes: implementation notes

- **Model** is the UC AI Gateway route (Claude Sonnet 4.5). Guardrails/rate-limits on
  the route apply to all modes.
- **short/long** connect to the autoscaling Lakebase project `agentic-memory`; the app
  service principal is granted `CONNECT`+`CREATE` at deploy time and the app creates its
  own schemas (`ai_chatbot` for the UI, `conversation_memory` for the agent).
