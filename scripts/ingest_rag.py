"""
Script — Ingest Mental Health Counseling Conversations into Qdrant (Module 4).
Run once after setting QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in .env
NOTE: If using the shared cluster, the collection 'health-counseling-dataset' is already populated.
Run: python scripts/ingest_rag.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from datasets import load_dataset
from src.modules.rag_pipeline import RAGPipeline
from src.utils.preprocessing import build_qa_chunk, chunk_text


def main():
    print("Loading heliosbrahma/mental_health_counseling_conversations …")
    ds = load_dataset("heliosbrahma/mental_health_counseling_conversations", split="train")
    print(f"  {len(ds)} records loaded.")

    chunks = []
    for row in ds:
        context  = str(row.get("Context", "")).strip()
        response = str(row.get("Response", "")).strip()
        if not context or not response:
            continue
        chunk = build_qa_chunk(context, response)
        # If chunk is very long, split with overlap
        if len(chunk.split()) > 250:
            sub_chunks = chunk_text(chunk, chunk_size=200, overlap=20)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk)

    print(f"  {len(chunks)} chunks prepared.")

    print("Connecting to Qdrant and ingesting …")
    rag = RAGPipeline()
    rag.create_collection()
    count = rag.ingest(chunks, batch_size=128)
    print(f"  ✓ {count} vectors upserted into collection 'health-counseling-dataset'.")
    print(f"  Collection size: {rag.collection_count()} points.")


if __name__ == "__main__":
    main()
