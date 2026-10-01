import logging
from pathlib import Path

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

_log = logging.getLogger(__name__)

# Load env vars from .env before importing the agent for proper auth
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841

# Git-based model version tracking is a best-effort nicety. In the Databricks
# Apps runtime the source is deployed without a .git dir ("fatal: not a git
# repository"), which makes this raise — don't let it crash server startup.
# MLflow tracing still works via MLFLOW_EXPERIMENT_ID.
try:
    setup_mlflow_git_based_version_tracking()
except Exception as e:  # noqa: BLE001
    _log.warning("Skipping MLflow git-based version tracking (%s: %s)", type(e).__name__, e)


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
