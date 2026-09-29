#!/usr/bin/env python3
"""Deploy the DataBank chat demo in one of three memory modes.

    python deploy_app.py                      # simple   (default, no memory)
    python deploy_app.py --mode shortterm     # session-scoped history (Lakebase)
    python deploy_app.py --mode longterm      # per-user history + fact recall

One app (default name ``databank-chat-demo``) is redeployed with different
config depending on ``--mode``. All three modes share the same UC AI Gateway
route (Claude Sonnet 4.5), UC functions, Genie Agent and Vector Search index.

The calling notebook ``12_deploy_chat_app.py`` runs ``Config_Parameters.py``
and passes those names in as flags, so this stays in sync with the lab config.
Run standalone and it derives the same names from the logged-in user.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

MODES = ("simple", "shortterm", "longterm")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    cp = subprocess.run(cmd, capture_output=capture, text=True)
    if check and cp.returncode != 0:
        sys.stderr.write((cp.stdout or "") + (cp.stderr or "") + "\n")
        raise SystemExit(f"Command failed ({cp.returncode}): {' '.join(cmd)}")
    return cp


def _db(args: list[str], profile: str | None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["databricks", *args]
    if profile:
        cmd += ["-p", profile]
    return _run(cmd, check=check)


def _db_json(args: list[str], profile: str | None, check: bool = True):
    cp = _db(args + ["-o", "json"], profile, check=check)
    if cp.returncode != 0:
        return None
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------
def resolve_username(profile: str | None) -> str:
    me = _db_json(["current-user", "me"], profile)
    email = (me or {}).get("userName", "")
    return re.sub(r"\W+", "", email.split("@")[0])


def resolve_lakebase_host(endpoint_path: str, profile: str | None) -> str:
    ep = _db_json(["postgres", "get-endpoint", endpoint_path], profile)
    if not ep:
        raise SystemExit(f"Could not resolve Lakebase endpoint: {endpoint_path}")
    return ep["status"]["hosts"]["host"]


# ---------------------------------------------------------------------------
# App resources (UC functions, vector index, Genie space, experiment)
# ---------------------------------------------------------------------------
def build_resources(cfg: dict) -> list[dict]:
    cat, sch = cfg["catalog"], cfg["schema"]
    resources: list[dict] = []
    if cfg.get("experiment_id"):
        resources.append(
            {
                "name": "experiment",
                "experiment": {"experiment_id": cfg["experiment_id"], "permission": "CAN_EDIT"},
            }
        )
    # App resource `name` must be 2-30 chars (it is only a label / valueFrom key;
    # the actual securable is identified by securable_full_name below).
    for fn in cfg["uc_functions"]:
        full = f"{cat}.{sch}.{fn}"
        resources.append(
            {
                "name": f"fn_{fn}"[:30],
                "uc_securable": {
                    "securable_full_name": full,
                    "securable_type": "FUNCTION",
                    "permission": "EXECUTE",
                },
            }
        )
    resources.append(
        {
            "name": "vector_search",
            "uc_securable": {
                "securable_full_name": cfg["vector_index"],
                "securable_type": "TABLE",
                "securable_kind": "TABLE_ONLINE_VECTOR_INDEX_REPLICA",
                "permission": "SELECT",
            },
        }
    )
    resources.append(
        {
            "name": "genie_space",
            "genie_space": {
                "name": "genie_space",
                "space_id": cfg["genie_space_id"],
                "permission": "CAN_RUN",
            },
        }
    )
    return resources


def grant_model_service(cfg: dict, sp_client_id: str, profile: str | None) -> None:
    """Grant the app SP EXECUTE on the AI Gateway model-service route.

    Without this, resolving the route raises 404 NOT_FOUND (UC hides securables
    the caller can't access). Not expressible as an app `uc_securable` resource,
    so we grant it directly. Idempotent.
    """
    print(f"Granting EXECUTE on model-service {cfg['model_route']} to {sp_client_id}...")
    _patch_perms("model_service", cfg["model_route"], sp_client_id, ["EXECUTE"], profile)


def _patch_perms(securable_type: str, full_name: str, sp: str, privileges: list[str],
                 profile: str | None) -> None:
    changes = {"changes": [{"principal": sp, "add": privileges}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(changes, f)
        p = f.name
    try:
        _db(["api", "patch",
             f"/api/2.1/unity-catalog/permissions/{securable_type}/{full_name}",
             "--json", f"@{p}"], profile)
    finally:
        os.unlink(p)


def grant_data_access(cfg: dict, sp_client_id: str, profile: str | None) -> None:
    """Grant the SP catalog/schema/table data access.

    The Genie Agent runs SQL against the underlying tables as the app SP, so it
    needs USE_CATALOG + USE_SCHEMA + SELECT (cascades to all tables) + EXECUTE
    (functions). App resources only grant the specific attached securables, not
    table SELECT. Idempotent.
    """
    print(f"Granting data access (USE/SELECT/EXECUTE) to {sp_client_id}...")
    _patch_perms("catalog", cfg["catalog"], sp_client_id, ["USE_CATALOG"], profile)
    _patch_perms(
        "schema", f"{cfg['catalog']}.{cfg['schema']}", sp_client_id,
        ["USE_SCHEMA", "SELECT", "EXECUTE"], profile,
    )


def ensure_app(cfg: dict, profile: str | None) -> str:
    """Create or update the app with tool resources. Returns SP client id."""
    name = cfg["app_name"]
    existing = _db_json(["apps", "get", name], profile, check=False)
    body = {
        "name": name,
        "description": f"DataBank chat demo ({cfg['mode']} memory mode)",
        "resources": build_resources(cfg),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        body_path = f.name
    try:
        # CLI quirk: `apps create` forbids the positional NAME alongside --json
        # (name must be in the body), but `apps update` REQUIRES the positional.
        if existing:
            print(f"App '{name}' exists — updating resources...")
            _db(["apps", "update", name, "--json", f"@{body_path}"], profile)
        else:
            print(f"Creating app '{name}'...")
            _db(["apps", "create", "--json", f"@{body_path}"], profile)
    finally:
        os.unlink(body_path)
    info = _db_json(["apps", "get", name], profile)
    sp = info.get("service_principal_client_id")
    print(f"App service principal: {sp}")
    return sp


# ---------------------------------------------------------------------------
# Lakebase grant (short/long): ensure the app SP can connect + create objects
# ---------------------------------------------------------------------------
def grant_lakebase(cfg: dict, sp_client_id: str, profile: str | None) -> None:
    endpoint = cfg["lakebase_endpoint_path"]
    host = resolve_lakebase_host(endpoint, profile)
    cred = _db_json(["postgres", "generate-database-credential", endpoint], profile)
    token = cred["token"]
    me = _db_json(["current-user", "me"], profile)["userName"]

    # macOS DNS workaround: resolve to an IP for psql hostaddr.
    ip = ""
    dig = shutil.which("dig")
    if dig:
        out = subprocess.run([dig, "+short", host], capture_output=True, text=True).stdout.strip()
        ip = out.splitlines()[-1] if out else ""
    conninfo = f"host={host} {'hostaddr=' + ip + ' ' if ip else ''}port=5432 dbname={cfg['lakebase_database']} user={me} sslmode=require"

    # Provision a *federated* OAuth login role for the service principal via the
    # databricks_auth extension. A plain `CREATE ROLE` does NOT link the role to
    # the Databricks identity, so OAuth password auth would fail. Idempotent:
    # skip creation if the role already exists.
    do_block = f"""
CREATE EXTENSION IF NOT EXISTS databricks_auth;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{sp_client_id}') THEN
    PERFORM databricks_create_role('{sp_client_id}', 'SERVICE_PRINCIPAL');
  END IF;
END $$;
GRANT CONNECT ON DATABASE {cfg['lakebase_database']} TO "{sp_client_id}";
GRANT CREATE  ON DATABASE {cfg['lakebase_database']} TO "{sp_client_id}";
"""
    psql = shutil.which("psql")
    if not psql:
        print(
            "!! psql not found. Grant the app SP manually (or run "
            "scripts/grant_lakebase_permissions.py):\n" + do_block
        )
        return
    env = dict(os.environ, PGPASSWORD=token, PGCONNECT_TIMEOUT="20")
    print(f"Granting Lakebase CONNECT+CREATE to {sp_client_id} on {host}...")
    cp = subprocess.run([psql, conninfo, "-v", "ON_ERROR_STOP=1", "-c", do_block],
                        capture_output=True, text=True, env=env)
    sys.stdout.write(cp.stdout)
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        raise SystemExit("Lakebase grant failed. See error above.")
    print("Lakebase grant OK.")


# ---------------------------------------------------------------------------
# app.yaml generation (env differs per mode)
# ---------------------------------------------------------------------------
def build_env(cfg: dict) -> list[dict]:
    env: list[dict] = [
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "API_PROXY", "value": "http://localhost:8000/invocations"},
        {"name": "CHAT_APP_PORT", "value": "3000"},
        {"name": "CHAT_PROXY_TIMEOUT_SECONDS", "value": "300"},
        {"name": "MLFLOW_EXPERIMENT_ID", "valueFrom": "experiment"},
        # ---- app config (mirrors Config_Parameters.py) ----
        {"name": "APP_NAME", "value": cfg["app_name"]},
        {"name": "MEMORY_MODE", "value": cfg["mode"]},
        {"name": "UC_CATALOG", "value": cfg["catalog"]},
        {"name": "UC_SCHEMA", "value": cfg["schema"]},
        {"name": "UC_FUNCTIONS", "value": ",".join(cfg["uc_functions"])},
        {"name": "VECTOR_INDEX", "value": cfg["vector_index"]},
        {"name": "GENIE_SPACE_ID", "value": cfg["genie_space_id"]},
        {"name": "GENIE_NAME", "value": cfg["genie_name"]},
        {"name": "MODEL_ROUTE", "value": cfg["model_route"]},
    ]
    if cfg["mode"] in ("shortterm", "longterm"):
        env += [
            {"name": "LAKEBASE_PROJECT_ID", "value": cfg["lakebase_project"]},
            {"name": "LAKEBASE_BRANCH", "value": cfg["lakebase_branch"]},
            {"name": "LAKEBASE_ENDPOINT", "value": cfg["lakebase_endpoint"]},
            {"name": "LAKEBASE_DATABASE", "value": cfg["lakebase_database"]},
            {"name": "LAKEBASE_SCHEMA", "value": cfg["lakebase_schema"]},
            {"name": "LAKEBASE_AUTOSCALING_ENDPOINT", "value": cfg["lakebase_endpoint_path"]},
            # Frontend (e2e-chatbot) Postgres coordinates — it mints its own
            # OAuth token, so no PGPASSWORD is needed. PGUSER = app SP client id.
            {"name": "PGHOST", "value": cfg["lakebase_host"]},
            {"name": "PGDATABASE", "value": cfg["lakebase_database"]},
            {"name": "PGUSER", "value": cfg["sp_client_id"]},
            {"name": "PGPORT", "value": "5432"},
            {"name": "PGSSLMODE", "value": "require"},
        ]
    return env


def write_app_yaml(cfg: dict) -> None:
    import yaml  # pyyaml is commonly available; fall back to manual dump

    doc = {"command": ["uv", "run", "start-app"], "env": build_env(cfg)}
    path = os.path.join(HERE, "app.yaml")
    try:
        text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    except Exception:
        text = _manual_yaml(doc)
    header = (
        "# GENERATED by deploy_app.py — DO NOT EDIT.\n"
        "# All values derive from Config_Parameters.py; regenerated on every deploy.\n"
        f"# mode={cfg['mode']}\n"
    )
    with open(path, "w") as f:
        f.write(header + text)
    print(f"Wrote {path} (mode={cfg['mode']}, {len(doc['env'])} env vars)")


def _manual_yaml(doc: dict) -> str:
    lines = ["command:"]
    for c in doc["command"]:
        lines.append(f"  - {c}")
    lines.append("env:")
    for e in doc["env"]:
        lines.append(f"  - name: {e['name']}")
        if "value" in e:
            lines.append(f"    value: \"{e['value']}\"")
        else:
            lines.append(f"    valueFrom: \"{e['valueFrom']}\"")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sync + deploy
# ---------------------------------------------------------------------------
def sync_and_deploy(cfg: dict, profile: str | None) -> None:
    ws_path = cfg["workspace_path"]
    print(f"Syncing source -> {ws_path}")
    _db(["sync", HERE, ws_path, "--full"], profile)
    print(f"Deploying app '{cfg['app_name']}'...")
    _db(["apps", "deploy", cfg["app_name"], "--source-code-path", ws_path], profile)
    info = _db_json(["apps", "get", cfg["app_name"]], profile)
    print("\n=== DEPLOYED ===")
    print(f"Mode : {cfg['mode']}")
    print(f"URL  : {info.get('url')}")
    print(f"State: {info.get('app_status', {}).get('state')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="simple")
    p.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    p.add_argument("--app-name", default="databank-chat-demo")
    p.add_argument("--catalog")
    p.add_argument("--schema")
    p.add_argument("--model-route")
    p.add_argument("--vector-index")
    # Lab-specific config has NO defaults here: it is owned by Config_Parameters.py
    # and passed in by 12_deploy_chat_app.py (single source of truth).
    p.add_argument("--genie-space-id")
    p.add_argument("--genie-name")
    p.add_argument("--uc-functions")
    p.add_argument("--experiment-id")
    p.add_argument("--lakebase-project")
    p.add_argument("--lakebase-branch", default="production")
    p.add_argument("--lakebase-endpoint", default="primary")
    p.add_argument("--lakebase-database", default="databricks_postgres")
    p.add_argument("--lakebase-schema", default="conversation_memory")
    p.add_argument("--workspace-path")
    p.add_argument("--skip-deploy", action="store_true", help="Configure only; don't sync/deploy")
    args = p.parse_args()

    # Lab config is owned by Config_Parameters.py (passed by the notebook). Fail
    # clearly if a standalone run omits a required value rather than silently
    # baking in a stale default.
    missing = []
    if not args.genie_space_id:
        missing.append("--genie-space-id")
    if not args.uc_functions:
        missing.append("--uc-functions")
    if args.mode in ("shortterm", "longterm") and not args.lakebase_project:
        missing.append("--lakebase-project")
    if missing:
        raise SystemExit(
            "Missing required config: " + ", ".join(missing) + "\n"
            "These values live in Config_Parameters.py — deploy via the "
            "12_deploy_chat_app.py notebook, or pass them explicitly for a "
            "standalone run."
        )

    username = resolve_username(args.profile)
    catalog = args.catalog or f"{username}_databank_lab"
    schema = args.schema or f"{username}_financial_data"
    me = _db_json(["current-user", "me"], args.profile)["userName"]

    cfg = {
        "mode": args.mode,
        "app_name": args.app_name,
        "catalog": catalog,
        "schema": schema,
        "uc_functions": [f.strip() for f in args.uc_functions.split(",") if f.strip()],
        "vector_index": args.vector_index or f"{catalog}.{schema}.product-docs-index",
        "genie_space_id": args.genie_space_id,
        "genie_name": args.genie_name or f"{username}-DataBank-Financial-Advisor",
        "model_route": args.model_route or f"{catalog}.{schema}.{username}-databank-llm-route",
        "experiment_id": args.experiment_id,
        "lakebase_project": args.lakebase_project,
        "lakebase_branch": args.lakebase_branch,
        "lakebase_endpoint": args.lakebase_endpoint,
        "lakebase_database": args.lakebase_database,
        "lakebase_schema": args.lakebase_schema,
        "lakebase_endpoint_path": (
            f"projects/{args.lakebase_project}/branches/{args.lakebase_branch}"
            f"/endpoints/{args.lakebase_endpoint}"
        ),
        "workspace_path": args.workspace_path or f"/Workspace/Users/{me}/databank_chat_demo_src",
    }

    print(f"\n=== Deploying '{cfg['app_name']}' in {cfg['mode'].upper()} mode ===")
    print(f"Catalog/Schema : {catalog}.{schema}")
    print(f"Model route    : {cfg['model_route']}")

    # 1. App + resources (also provisions the SP)
    cfg["sp_client_id"] = ensure_app(cfg, args.profile)

    # 1b. Grant the SP EXECUTE on the AI Gateway model-service (all modes)
    grant_model_service(cfg, cfg["sp_client_id"], args.profile)

    # 1c. Grant the SP data access (catalog/schema/tables) for Genie SQL (all modes)
    grant_data_access(cfg, cfg["sp_client_id"], args.profile)

    # 2. Lakebase (short/long only): host + SP grant
    if cfg["mode"] in ("shortterm", "longterm"):
        cfg["lakebase_host"] = resolve_lakebase_host(cfg["lakebase_endpoint_path"], args.profile)
        grant_lakebase(cfg, cfg["sp_client_id"], args.profile)

    # 3. app.yaml for this mode
    write_app_yaml(cfg)

    # 4. Sync + deploy
    if args.skip_deploy:
        print("--skip-deploy set: app.yaml written, not deploying.")
        return
    sync_and_deploy(cfg, args.profile)


if __name__ == "__main__":
    main()
