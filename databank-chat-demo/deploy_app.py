#!/usr/bin/env python3
"""Post-deploy configurator for the DataBank chat app (DAB companion).

Run this AFTER ``databricks bundle deploy -t <target>`` has created the app and
synced the code:

    databricks bundle deploy      -t dev_longterm -p Myenv
    python databank-chat-demo/deploy_app.py -t dev_longterm -p Myenv

The bundle (``databricks.yml``) is the single source of truth. This script reads
the resolved config from ``databricks bundle summary -o json -t <target>`` and
then applies the bits DAB cannot express on its own:

  * the app env — including the resolved Lakebase host and the app SP as PGUSER
    (written to the workspace copy of app.yaml, then redeployed)
  * EXECUTE on the AI Gateway model-service route  (uc_securable has no such type)
  * catalog USE + schema USE/SELECT/EXECUTE        (table-level SELECT for Genie)
  * a *federated* Lakebase login role for the app SP (shortterm/longterm only)

DAB owns app creation and code sync; this script owns configuration and grants.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

MODES = ("simple", "shortterm", "longterm")
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.dirname(HERE)  # databricks.yml lives at the repo root


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], check: bool = True, cwd: str | None = None) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if check and cp.returncode != 0:
        sys.stderr.write((cp.stdout or "") + (cp.stderr or "") + "\n")
        raise SystemExit(f"Command failed ({cp.returncode}): {' '.join(cmd)}")
    return cp


def _db(args: list[str], profile: str | None, check: bool = True,
        cwd: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["databricks", *args]
    if profile:
        cmd += ["-p", profile]
    return _run(cmd, check=check, cwd=cwd)


def _db_json(args: list[str], profile: str | None, check: bool = True):
    cp = _db(args + ["-o", "json"], profile, check=check)
    if cp.returncode != 0:
        return None
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Read the resolved config from the bundle (single source of truth)
# ---------------------------------------------------------------------------
def load_bundle_config(target: str, profile: str | None) -> dict:
    """Resolve names/paths from `databricks bundle summary` for the target."""
    cp = _db(["bundle", "summary", "-t", target, "-o", "json"], profile, cwd=BUNDLE_ROOT)
    summary = json.loads(cp.stdout)

    def var(name: str, default: str = "") -> str:
        return (summary.get("variables", {}).get(name, {}) or {}).get("value", default) or default

    app = summary.get("resources", {}).get("apps", {}).get("databank_chat_demo", {})
    app_name = app.get("name")
    source_code_path = app.get("source_code_path")
    if not app_name or not source_code_path:
        raise SystemExit(
            "Could not read the app from the bundle summary — run "
            f"`databricks bundle deploy -t {target}` first."
        )

    username = var("username")
    catalog = var("catalog")
    schema = var("schema")
    mode = var("mode", "simple")
    lakebase_project = var("lakebase_project")
    lakebase_branch = var("lakebase_branch", "production")
    lakebase_endpoint = var("lakebase_endpoint", "primary")

    cfg = {
        "target": target,
        "app_name": app_name,
        "source_code_path": source_code_path,
        "mode": mode,
        "catalog": catalog,
        "schema": schema,
        "uc_functions": [f.strip() for f in var("uc_functions").split(",") if f.strip()],
        "vector_index": f"{catalog}.{schema}.product-docs-index",
        "genie_space_id": var("genie_space_id"),
        "genie_name": f"{username}-DataBank-Financial-Advisor",
        "model_route": f"{catalog}.{schema}.{username}-databank-llm-route",
        "experiment_id": var("experiment_id"),
        "lakebase_project": lakebase_project,
        "lakebase_branch": lakebase_branch,
        "lakebase_endpoint": lakebase_endpoint,
        "lakebase_database": var("lakebase_database", "databricks_postgres"),
        "lakebase_schema": var("lakebase_schema", "conversation_memory"),
        "lakebase_endpoint_path": (
            f"projects/{lakebase_project}/branches/{lakebase_branch}"
            f"/endpoints/{lakebase_endpoint}"
        ),
    }
    return cfg


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------
def resolve_lakebase_host(endpoint_path: str, profile: str | None) -> str:
    ep = _db_json(["postgres", "get-endpoint", endpoint_path], profile)
    if not ep:
        raise SystemExit(f"Could not resolve Lakebase endpoint: {endpoint_path}")
    return ep["status"]["hosts"]["host"]


def get_app_sp(app_name: str, profile: str | None) -> str:
    """Read the app's service-principal client id (DAB created the app)."""
    info = _db_json(["apps", "get", app_name], profile)
    if not info or not info.get("service_principal_client_id"):
        raise SystemExit(
            f"App '{app_name}' has no service principal yet — run "
            "`databricks bundle deploy` first so DAB creates it."
        )
    sp = info["service_principal_client_id"]
    print(f"App service principal: {sp}")
    return sp


# ---------------------------------------------------------------------------
# Grants DAB cannot express
# ---------------------------------------------------------------------------
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


def grant_model_service(cfg: dict, sp: str, profile: str | None) -> None:
    """Grant the app SP EXECUTE on the AI Gateway model-service route.

    Without this, resolving the route raises 404 NOT_FOUND (UC hides securables
    the caller can't access). The app `uc_securable` resource type has no
    model_service, so we grant it directly. Idempotent.
    """
    print(f"Granting EXECUTE on model-service {cfg['model_route']} to {sp}...")
    _patch_perms("model_service", cfg["model_route"], sp, ["EXECUTE"], profile)


def grant_data_access(cfg: dict, sp: str, profile: str | None) -> None:
    """Grant the SP catalog/schema/table data access for Genie SQL.

    The Genie Agent runs SQL against the underlying tables as the app SP, so it
    needs USE_CATALOG + USE_SCHEMA + SELECT (cascades to all tables) + EXECUTE.
    App resources only bind the specific attached securables, not table SELECT.
    Idempotent.
    """
    print(f"Granting data access (USE/SELECT/EXECUTE) to {sp}...")
    _patch_perms("catalog", cfg["catalog"], sp, ["USE_CATALOG"], profile)
    _patch_perms(
        "schema", f"{cfg['catalog']}.{cfg['schema']}", sp,
        ["USE_SCHEMA", "SELECT", "EXECUTE"], profile,
    )


def grant_lakebase(cfg: dict, sp: str, profile: str | None) -> None:
    """Provision a federated OAuth login role for the app SP (short/long only)."""
    endpoint = cfg["lakebase_endpoint_path"]
    host = cfg["lakebase_host"]
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

    # A plain `CREATE ROLE` does NOT link the role to the Databricks identity, so
    # OAuth password auth would fail. Use databricks_create_role(...) to federate.
    # Idempotent: skip creation if the role already exists.
    do_block = f"""
CREATE EXTENSION IF NOT EXISTS databricks_auth;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{sp}') THEN
    PERFORM databricks_create_role('{sp}', 'SERVICE_PRINCIPAL');
  END IF;
END $$;
GRANT CONNECT ON DATABASE {cfg['lakebase_database']} TO "{sp}";
GRANT CREATE  ON DATABASE {cfg['lakebase_database']} TO "{sp}";
"""
    psql = shutil.which("psql")
    if not psql:
        print("!! psql not found. Grant the app SP manually:\n" + do_block)
        return
    env = dict(os.environ, PGPASSWORD=token, PGCONNECT_TIMEOUT="20")
    print(f"Granting Lakebase CONNECT+CREATE to {sp} on {host}...")
    cp = subprocess.run([psql, conninfo, "-v", "ON_ERROR_STOP=1", "-c", do_block],
                        capture_output=True, text=True, env=env)
    sys.stdout.write(cp.stdout)
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        raise SystemExit("Lakebase grant failed. See error above.")
    print("Lakebase grant OK.")


# ---------------------------------------------------------------------------
# app.yaml (env differs per mode) — written to a temp file, uploaded to the
# workspace copy DAB synced. The committed base app.yaml is left untouched.
# ---------------------------------------------------------------------------
def build_env(cfg: dict) -> list[dict]:
    env: list[dict] = [
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "API_PROXY", "value": "http://localhost:8000/invocations"},
        {"name": "CHAT_APP_PORT", "value": "3000"},
        {"name": "CHAT_PROXY_TIMEOUT_SECONDS", "value": "300"},
        # ---- app config (resolved from databricks.yml) ----
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
    if cfg.get("experiment_id"):
        env.append({"name": "MLFLOW_EXPERIMENT_ID", "value": cfg["experiment_id"]})
    if cfg["mode"] in ("shortterm", "longterm"):
        env += [
            {"name": "LAKEBASE_PROJECT_ID", "value": cfg["lakebase_project"]},
            {"name": "LAKEBASE_BRANCH", "value": cfg["lakebase_branch"]},
            {"name": "LAKEBASE_ENDPOINT", "value": cfg["lakebase_endpoint"]},
            {"name": "LAKEBASE_DATABASE", "value": cfg["lakebase_database"]},
            {"name": "LAKEBASE_SCHEMA", "value": cfg["lakebase_schema"]},
            {"name": "LAKEBASE_AUTOSCALING_ENDPOINT", "value": cfg["lakebase_endpoint_path"]},
            # Frontend (e2e-chatbot) Postgres coordinates — it mints its own OAuth
            # token, so no PGPASSWORD is needed. PGUSER = app SP client id.
            {"name": "PGHOST", "value": cfg["lakebase_host"]},
            {"name": "PGDATABASE", "value": cfg["lakebase_database"]},
            {"name": "PGUSER", "value": cfg["sp_client_id"]},
            {"name": "PGPORT", "value": "5432"},
            {"name": "PGSSLMODE", "value": "require"},
        ]
    return env


def _yaml_dump(doc: dict) -> str:
    try:
        import yaml
        return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    except Exception:
        lines = ["command:"]
        for c in doc["command"]:
            lines.append(f"  - {c}")
        lines.append("env:")
        for e in doc["env"]:
            lines.append(f"  - name: {e['name']}")
            key = "value" if "value" in e else "valueFrom"
            lines.append(f"    {key}: \"{e[key]}\"")
        return "\n".join(lines) + "\n"


def write_app_yaml_temp(cfg: dict) -> str:
    doc = {"command": ["uv", "run", "start-app"], "env": build_env(cfg)}
    header = (
        "# GENERATED by deploy_app.py from databricks.yml — DO NOT EDIT.\n"
        f"# target={cfg['target']} mode={cfg['mode']}\n"
    )
    fd, path = tempfile.mkstemp(suffix="_app.yaml")
    with os.fdopen(fd, "w") as f:
        f.write(header + _yaml_dump(doc))
    print(f"Rendered app.yaml ({cfg['mode']}, {len(doc['env'])} env vars) -> {path}")
    return path


# ---------------------------------------------------------------------------
# Upload the full app.yaml to the workspace copy + redeploy
# ---------------------------------------------------------------------------
def upload_and_deploy(cfg: dict, app_yaml_local: str, profile: str | None) -> None:
    ws = cfg["source_code_path"]
    print(f"Uploading app.yaml -> {ws}/app.yaml")
    _db(["workspace", "import", f"{ws}/app.yaml", "--file", app_yaml_local,
         "--format", "AUTO", "--overwrite"], profile)
    # `apps deploy` needs the app RUNNING; start it if it was stopped.
    info = _db_json(["apps", "get", cfg["app_name"]], profile)
    state = (info or {}).get("app_status", {}).get("state")
    if state != "RUNNING":
        print(f"App state is {state}; starting before deploy...")
        _db(["apps", "start", cfg["app_name"]], profile)
    print(f"Deploying app '{cfg['app_name']}'...")
    _db(["apps", "deploy", cfg["app_name"], "--source-code-path", ws], profile)
    info = _db_json(["apps", "get", cfg["app_name"]], profile)
    print("\n=== CONFIGURED ===")
    print(f"Target: {cfg['target']}  Mode: {cfg['mode']}")
    print(f"URL   : {info.get('url')}")
    print(f"State : {info.get('app_status', {}).get('state')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--target", required=True,
                   help="Bundle target (dev_simple | dev_shortterm | dev_longterm)")
    p.add_argument("-p", "--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    p.add_argument("--skip-deploy", action="store_true",
                   help="Apply grants + render app.yaml but don't upload/redeploy")
    args = p.parse_args()

    cfg = load_bundle_config(args.target, args.profile)
    if cfg["mode"] not in MODES:
        raise SystemExit(f"Unexpected mode '{cfg['mode']}' (expected one of {MODES}).")

    print(f"\n=== Configuring '{cfg['app_name']}' [{cfg['target']} / {cfg['mode'].upper()}] ===")
    print(f"Catalog/Schema : {cfg['catalog']}.{cfg['schema']}")
    print(f"Model route    : {cfg['model_route']}")

    # 1. App SP (DAB created the app)
    cfg["sp_client_id"] = get_app_sp(cfg["app_name"], args.profile)

    # 2. Grants DAB cannot express (all modes)
    grant_model_service(cfg, cfg["sp_client_id"], args.profile)
    grant_data_access(cfg, cfg["sp_client_id"], args.profile)

    # 3. Lakebase (shortterm/longterm only): host + federated SP role
    if cfg["mode"] in ("shortterm", "longterm"):
        cfg["lakebase_host"] = resolve_lakebase_host(cfg["lakebase_endpoint_path"], args.profile)
        grant_lakebase(cfg, cfg["sp_client_id"], args.profile)

    # 4. Full app.yaml for this mode -> workspace copy, then redeploy
    app_yaml_local = write_app_yaml_temp(cfg)
    try:
        if args.skip_deploy:
            print("--skip-deploy set: grants applied, app.yaml rendered, not redeploying.")
            return
        upload_and_deploy(cfg, app_yaml_local, args.profile)
    finally:
        os.unlink(app_yaml_local)


if __name__ == "__main__":
    main()
