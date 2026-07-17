# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Module 04 — Welcome
# MAGIC %md
# MAGIC ## 🏦 DataBank AI Lab — Module 04: Vector Search
# MAGIC **Duration:** ~30 minutes | **Prerequisite:** Module 01 (PDFs in volume)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What is Vector Search?
# MAGIC
# MAGIC Traditional search finds exact keyword matches. **Vector Search** finds *semantically similar* content — even when the words are different.
# MAGIC
# MAGIC ```
# MAGIC Query: "What happens if I can't make my loan repayment?"
# MAGIC   Traditional search: Looks for exact words — finds nothing
# MAGIC   Vector Search:      Understands the meaning — finds "early repayment charge",
# MAGIC                       "missed payment policy", "debt management process"
# MAGIC ```
# MAGIC
# MAGIC ### How It Works
# MAGIC 1. **Embed**: Convert text chunks to high-dimensional vectors (numbers representing meaning)
# MAGIC 2. **Index**: Store vectors in a specialised data structure (HNSW or flat index)
# MAGIC 3. **Query**: Convert user question to a vector, find the nearest neighbours
# MAGIC
# MAGIC ### What You’ll Build
# MAGIC ```
# MAGIC /Volumes/databank_lab/financial_data/documents/   ← Source PDFs (Module 01)
# MAGIC          ↓  (extract + chunk text)
# MAGIC databank_lab.financial_data.product_docs_chunks   ← Delta table with text chunks
# MAGIC          ↓  (embed using databricks-gte-large-en)
# MAGIC databank_lab.financial_data.product_docs_index    ← Vector Search Index
# MAGIC          ↓
# MAGIC Semantic search: "conservative investment products"
# MAGIC          ↓
# MAGIC Results: Cash ISA, Fixed-Rate Bond, Managed Portfolio (Cautious)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Step 0 — Configuration
# ================================================================
# CONFIGURATION
# ================================================================

# Get logged-in user information
# If running this lab via Partner Academy Vocarium 

user = spark.sql("SELECT current_user() AS username").collect()[0]['username']

# Extract username before '@' and remove special characters
import re
username_clean = re.sub(r'\W+', '', user.split('@')[0])

if "labuser" in username_clean:
    CATALOG = username_clean
else:
    CATALOG = "databank_lab"

SCHEMA        = "financial_data"
VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/documents"

# Vector Search endpoint — auto-select an available endpoint (same logic as Module 00)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState
w = WorkspaceClient()
_preferred = "databank-vs-endpoint"
try:
    _ep = w.vector_search_endpoints.get_endpoint(endpoint_name=_preferred)
    VS_ENDPOINT = _preferred
except Exception:
    _endpoints = [
        ep.name for ep in w.vector_search_endpoints.list_endpoints()
        if ep.endpoint_status and ep.endpoint_status.state == EndpointStatusState.ONLINE
    ]
    # Use the designated shared endpoint for this workspace
    VS_ENDPOINT = "one-env-shared-endpoint-10"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_docs_index"
SOURCE_TABLE  = f"{CATALOG}.{SCHEMA}.product_docs_chunks"
EMBED_MODEL   = "databricks-gte-large-en"  # 1024-dim, 8k context, hosted by Databricks

print(f"✅ VS Endpoint : {VS_ENDPOINT}")
print(f"📄 VS Index     : {VS_INDEX_NAME}")
print(f"📐 Embed Model  : {EMBED_MODEL}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Parse and Chunk PDFs
# MAGIC %md
# MAGIC ## 📄 Step 1: Parse & Chunk PDF Documents
# MAGIC
# MAGIC **Chunking strategy matters.** Too large → irrelevant content pollutes results. Too small → loss of context.
# MAGIC
# MAGIC For financial product documents we use:
# MAGIC - **Chunk size**: ~500 tokens (~400 words) — large enough to retain product context
# MAGIC - **Overlap**: 50 tokens — ensures sentences at chunk boundaries are captured in both chunks
# MAGIC - **Metadata**: document name, section heading, page number — for filtering and attribution
# MAGIC
# MAGIC We use Python’s `pypdf` library (lightweight, no external dependencies).
# MAGIC
# MAGIC > ⚠️ `pypdf` may not be pre-installed. If you skipped Module 00, run `%pip install pypdf -q` first.

# COMMAND ----------

# DBTITLE 1,Step 1 — Extract and Chunk PDF Text
# MAGIC %pip install pypdf databricks-vectorsearch -q

# COMMAND ----------

# DBTITLE 1,Step 1b — Build Chunks DataFrame
import os, re
from pypdf import PdfReader
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def extract_chunks_from_pdf(filepath: str, chunk_size: int = 1800, overlap: int = 200) -> list:
    """
    Extract text from a PDF and split into overlapping character-based chunks.
    Returns a list of dicts: {doc_name, chunk_id, chunk_text, source_path}
    """
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text += page_text + "\n\n"

    # Clean the text (remove excess whitespace)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text.strip())

    # Split into overlapping chunks
    chunks = []
    doc_name = os.path.basename(filepath).replace(".pdf", "")
    start = 0
    chunk_idx = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        if len(chunk.strip()) > 50:  # skip tiny/empty chunks
            chunks.append({
                "doc_name":   doc_name,
                "chunk_id":   f"{doc_name}_{chunk_idx:03d}",
                "chunk_text": chunk.strip(),
                "source_path": filepath.replace(VOLUME_PATH + "/", "")
            })
            chunk_idx += 1
        start += chunk_size - overlap

    return chunks

# Walk through all PDFs in the volume and extract chunks
all_chunks = []
for root, dirs, files in os.walk(VOLUME_PATH):
    for fname in files:
        if fname.endswith(".pdf"):
            path = os.path.join(root, fname)
            chunks = extract_chunks_from_pdf(path)
            all_chunks.extend(chunks)
            print(f"   ✓ {fname}: {len(chunks)} chunks")

print(f"\n📊 Total chunks extracted: {len(all_chunks)}")

# Create Spark DataFrame
schema = StructType([
    StructField("doc_name",    StringType(),  True),
    StructField("chunk_id",    StringType(),  False),
    StructField("chunk_text",  StringType(),  True),
    StructField("source_path", StringType(),  True),
])

chunks_df = spark.createDataFrame(all_chunks, schema=schema)

# Write to Delta — this becomes the SOURCE TABLE for the vector index
chunks_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SOURCE_TABLE)

# Enable Change Data Feed — required for Delta Sync Vector Search
spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"✅ Source table created: {SOURCE_TABLE}")
print(f"   Rows: {spark.table(SOURCE_TABLE).count()}")
display(spark.table(SOURCE_TABLE).select("doc_name", "chunk_id", F.substring("chunk_text", 1, 150).alias("preview")).limit(8))

# COMMAND ----------

# DBTITLE 1,Step 2 — Create Vector Search Index
# MAGIC %md
# MAGIC ## 🔍 Step 2: Create Vector Search Index
# MAGIC
# MAGIC **Delta Sync Index (managed embeddings)** is the easiest option:
# MAGIC - You provide the source Delta table and column to embed
# MAGIC - Databricks automatically computes embeddings using `databricks-gte-large-en`
# MAGIC - The index stays in sync with the source table (TRIGGERED or CONTINUOUS sync)
# MAGIC
# MAGIC | Pipeline Type | How It Works |
# MAGIC |---------------|-------------|
# MAGIC | `TRIGGERED`   | Re-embeds on demand (call `sync_index()`) — **use for this lab** |
# MAGIC | `CONTINUOUS`  | Streams new rows automatically — use for production |

# COMMAND ----------

# DBTITLE 1,Step 2 — Create VS Index
import importlib, sys
for _k in list(sys.modules.keys()):
    if _k == 'databricks' or _k.startswith('databricks.'):
        del sys.modules[_k]
importlib.invalidate_caches()

from databricks.vector_search.client import VectorSearchClient

# Verify the endpoint created in Module 00 is ready
try:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
    print(f"✅ VS endpoint '{VS_ENDPOINT}' confirmed ready")
except Exception as e:
    print(f"⚠️  Could not verify endpoint '{VS_ENDPOINT}' via SDK (may lack get permission): {e}")
    print("   Continuing — will attempt index operations via VectorSearchClient...")

vsc = VectorSearchClient()

# Delete existing index if it exists (for re-runs)
try:
    vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)
    print(f"🗑️  Deleted existing index: {VS_INDEX_NAME}")
except Exception:
    pass  # Index did not exist

print(f"🔧 Creating Vector Search Index: {VS_INDEX_NAME}")
print(f"   Endpoint  : {VS_ENDPOINT}")
print(f"   Source    : {SOURCE_TABLE}")
print(f"   Embedding : {EMBED_MODEL}")
print("   This will take 2-4 minutes...")

# Create Delta Sync index with managed embeddings
index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=VS_ENDPOINT,
    source_table_name=SOURCE_TABLE,
    index_name=VS_INDEX_NAME,
    pipeline_type="TRIGGERED",
    primary_key="chunk_id",
    embedding_source_column="chunk_text",
    embedding_model_endpoint_name=EMBED_MODEL
)

print(f"\n✅ Vector Search Index ready: {VS_INDEX_NAME}")
print(f"   Status: {index.describe().get('status', {}).get('ready_for_query', 'unknown')}")

# COMMAND ----------

# DBTITLE 1,Step 3 — Test Semantic Search
# Connect to the index for querying
index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)

def search_docs(query: str, num_results: int = 3) -> None:
    """
    Run a semantic similarity search against the product document index.
    Prints the top matching chunks with their similarity scores.
    """
    print(f"\n🔍 Query: \"{query}\"")
    print("-" * 65)

    results = index.similarity_search(
        query_text=query,
        columns=["chunk_id", "doc_name", "chunk_text", "source_path"],
        num_results=num_results
    )

    for i, row in enumerate(results.get("result", {}).get("data_array", []), 1):
        chunk_id, doc_name, chunk_text, source_path, score = row
        print(f"  Result {i} | Score: {score:.3f} | Doc: {doc_name}")
        print(f"  {chunk_text[:250].strip()}...")
        print()

# Financial advisor test queries
search_docs("What is the interest rate on a fixed rate bond and what are the withdrawal rules?")
search_docs("Which products are suitable for a conservative investor with low risk appetite?")
search_docs("How do I report a fraudulent transaction and what is my liability?")
search_docs("What is the difference between a Cash ISA and a Stocks and Shares ISA?")

# COMMAND ----------

# DBTITLE 1,Step 4 — Verify Index
# Verify the index is ready for queries
index_info = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)
status = index_info.describe()

print(f"📊 Index verification: {VS_INDEX_NAME}")
print(f"   Ready for query : {status.get('status', {}).get('ready_for_query', 'unknown')}")
print(f"   Vectors indexed : {status.get('status', {}).get('indexed_row_count', 'unknown')}")
print(f"   Endpoint        : {VS_ENDPOINT}")
print(f"   Source table    : {SOURCE_TABLE}")
print(f"\n✅ Vector Search index is ready to use in Modules 07 and 08.")

# COMMAND ----------

# DBTITLE 1,Module 04 — Checkpoint
# MAGIC %md
# MAGIC ## ✅ Module 04 Complete — Checkpoint
# MAGIC
# MAGIC | Check | Expected |
# MAGIC |-------|----------|
# MAGIC | `product_docs_chunks` Delta table | All PDF chunks stored with chunk_id, doc_name, chunk_text |
# MAGIC | Vector index `product_docs_index` | Status: Ready for query |
# MAGIC | Semantic search working | Relevant product content returned for financial queries |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What’s Happening Under the Hood
# MAGIC 1. `databricks-gte-large-en` converts each text chunk into a 1024-dimensional vector
# MAGIC 2. Vectors are stored in an HNSW index on the Vector Search endpoint
# MAGIC 3. At query time: your query text is embedded, then nearest vectors are retrieved
# MAGIC
# MAGIC ### 🚀 Next: Module 05 — Genie Space
# MAGIC Open **`05_genie_space`** to create a natural-language SQL interface over the financial tables.