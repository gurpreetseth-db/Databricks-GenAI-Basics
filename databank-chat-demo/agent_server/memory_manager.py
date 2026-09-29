
import os
import json
import threading
import time
import psycopg
from databricks.sdk import WorkspaceClient


class LakebaseMemoryManager:
    """
    Manages short-term (conversation history) and long-term (extracted facts)
    memory for the DataBank AI Advisor app using Lakebase Postgres.

    Usage:
        memory = LakebaseMemoryManager()
        conv_id = memory.start_conversation(user_id="user@example.com")
        memory.save_message(conv_id, "user", "What ISAs do you offer?")
        history = memory.get_conversation_history(conv_id)
        memories = memory.get_long_term_memories(user_id="user@example.com")
    """

    SCHEMA = os.getenv("LAKEBASE_SCHEMA", "conversation_memory")

    def __init__(self):
        self.w = WorkspaceClient()
        self.project_id = os.getenv("LAKEBASE_PROJECT_ID", "agentic-memory")
        self.branch = os.getenv("LAKEBASE_BRANCH", "production")
        self.endpoint_name = os.getenv("LAKEBASE_ENDPOINT", "primary")
        self.database = os.getenv("LAKEBASE_DATABASE", "databricks_postgres")
        self.endpoint_full = (
            f"projects/{self.project_id}/branches/{self.branch}"
            f"/endpoints/{self.endpoint_name}"
        )

        # Get endpoint host
        ep = self.w.postgres.get_endpoint(name=self.endpoint_full)
        self.host = ep.status.hosts.host
        self.user = self.w.current_user.me().user_name

        # Token management
        self._token = self._mint_token()
        self._token_lock = threading.Lock()
        self._start_token_refresh()

        # Connection
        self.conn = self._connect()

        # Make the app self-contained: create the memory schema/tables if absent
        # (requires CREATE on the database, granted to the app SP at deploy time).
        self._ensure_schema()

    def _mint_token(self):
        return self.w.postgres.generate_database_credential(
            endpoint=self.endpoint_full
        ).token

    def _connect(self):
        return psycopg.connect(
            host=self.host,
            dbname=self.database,
            user=self.user,
            password=self._token,
            sslmode="require",
            autocommit=True
        )

    def _start_token_refresh(self):
        """Refresh OAuth token every 40 minutes (expires at 60)."""
        def _refresh():
            while True:
                time.sleep(40 * 60)
                with self._token_lock:
                    self._token = self._mint_token()
                    # Reconnect with fresh token
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                    self.conn = self._connect()
        t = threading.Thread(target=_refresh, daemon=True)
        t.start()

    def _ensure_connection(self):
        """Reconnect if the connection was lost."""
        try:
            self.conn.cursor().execute("SELECT 1")
        except Exception:
            with self._token_lock:
                self._token = self._mint_token()
            self.conn = self._connect()

    def _ensure_schema(self):
        """Create the backend memory schema + tables if they do not exist.

        Idempotent. Keeps the app self-contained so short/long-term modes work
        against a fresh Lakebase branch without a separate setup notebook.
        """
        cur = self.conn.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.SCHEMA}")
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.SCHEMA}.conversations (
                conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id       TEXT NOT NULL,
                title         TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                is_active     BOOLEAN NOT NULL DEFAULT TRUE,
                started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                ended_at      TIMESTAMPTZ
            )"""
        )
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.SCHEMA}.messages (
                message_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                conversation_id UUID NOT NULL REFERENCES {self.SCHEMA}.conversations(conversation_id),
                role            TEXT NOT NULL,
                content         TEXT,
                tokens_used     INTEGER,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"""
        )
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.SCHEMA}.long_term_memory (
                user_id     TEXT NOT NULL,
                category    TEXT NOT NULL,
                fact_key    TEXT NOT NULL,
                fact_value  TEXT,
                confidence  DOUBLE PRECISION DEFAULT 1.0,
                source_conversation_id UUID,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, category, fact_key)
            )"""
        )
        cur.execute(
            f"""CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON {self.SCHEMA}.messages(conversation_id, created_at)"""
        )

    # ---- Short-Term Memory ----

    def start_conversation(self, user_id, title=None):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            f"""INSERT INTO {self.SCHEMA}.conversations (user_id, title)
                VALUES (%s, %s) RETURNING conversation_id""",
            (user_id, title)
        )
        return str(cur.fetchone()[0])

    def save_message(self, conversation_id, role, content, tokens_used=None):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            f"""INSERT INTO {self.SCHEMA}.messages 
                (conversation_id, role, content, tokens_used)
                VALUES (%s, %s, %s, %s)""",
            (conversation_id, role, content, tokens_used)
        )
        cur.execute(
            f"""UPDATE {self.SCHEMA}.conversations 
                SET message_count = message_count + 1
                WHERE conversation_id = %s""",
            (conversation_id,)
        )

    def get_conversation_history(self, conversation_id, limit=50):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            f"""SELECT role, content, tokens_used, created_at
                FROM {self.SCHEMA}.messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC LIMIT %s""",
            (conversation_id, limit)
        )
        return [
            {"role": r[0], "content": r[1], "tokens": r[2], "timestamp": r[3]}
            for r in cur.fetchall()
        ]

    def get_recent_conversations(self, user_id, limit=10):
        """Get recent conversations for populating chat history in UI."""
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            f"""SELECT conversation_id, title, started_at, message_count, is_active
                FROM {self.SCHEMA}.conversations
                WHERE user_id = %s
                ORDER BY started_at DESC LIMIT %s""",
            (user_id, limit)
        )
        return [
            {"id": str(r[0]), "title": r[1], "started": r[2], 
             "messages": r[3], "active": r[4]}
            for r in cur.fetchall()
        ]

    def end_conversation(self, conversation_id):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            f"""UPDATE {self.SCHEMA}.conversations
                SET is_active = FALSE, ended_at = NOW()
                WHERE conversation_id = %s""",
            (conversation_id,)
        )

    # ---- Long-Term Memory ----

    def save_long_term_memory(self, user_id, category, fact_key, fact_value,
                              confidence=1.0, source_conversation_id=None):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            f"""INSERT INTO {self.SCHEMA}.long_term_memory 
                (user_id, category, fact_key, fact_value, confidence, source_conversation_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, category, fact_key)
                DO UPDATE SET fact_value = EXCLUDED.fact_value,
                             confidence = EXCLUDED.confidence,
                             source_conversation_id = EXCLUDED.source_conversation_id,
                             updated_at = NOW()""",
            (user_id, category, fact_key, fact_value, confidence, source_conversation_id)
        )

    def get_long_term_memories(self, user_id, category=None):
        self._ensure_connection()
        cur = self.conn.cursor()
        if category:
            cur.execute(
                f"""SELECT category, fact_key, fact_value, confidence
                    FROM {self.SCHEMA}.long_term_memory
                    WHERE user_id = %s AND category = %s
                    ORDER BY updated_at DESC""",
                (user_id, category)
            )
        else:
            cur.execute(
                f"""SELECT category, fact_key, fact_value, confidence
                    FROM {self.SCHEMA}.long_term_memory
                    WHERE user_id = %s
                    ORDER BY category, updated_at DESC""",
                (user_id,)
            )
        return [
            {"category": r[0], "key": r[1], "value": r[2], "confidence": r[3]}
            for r in cur.fetchall()
        ]

    def format_memories_for_prompt(self, memories):
        if not memories:
            return ""
        lines = ["\n--- Known facts about this user (from previous conversations) ---"]
        for m in memories:
            lines.append(f"- [{m['category']}] {m['key']}: {m['value']}")
        lines.append("--- End of known facts ---\n")
        return "\n".join(lines)

    def extract_and_store_facts(self, client, model, conversation_id, user_id):
        """
        Use the LLM to extract durable facts from a completed conversation
        and store them in long-term memory.
        """
        history = self.get_conversation_history(conversation_id)
        conv_text = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in history if m['role'] != 'system']
        )

        extraction_prompt = f"""Analyze this financial advisor conversation and extract durable facts.
Return a JSON array of objects with: "category" (preference|risk_appetite|frequent_topic|personal_context),
"fact_key" (short identifier), "fact_value" (the fact), "confidence" (0.0-1.0).
Return ONLY the JSON array.

Conversation:
{conv_text}"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=500, temperature=0.1
            )
            raw = resp.choices[0].message.content or ""
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            facts = json.loads(clean)
            for f in facts:
                self.save_long_term_memory(
                    user_id, f.get("category", "personal_context"),
                    f.get("fact_key", "unknown"), f.get("fact_value", ""),
                    f.get("confidence", 0.8), conversation_id
                )
            return facts
        except Exception as e:
            print(f"Fact extraction failed: {e}")
            return []
