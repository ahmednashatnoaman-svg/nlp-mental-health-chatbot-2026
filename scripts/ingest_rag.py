"""
Script — Ingest Mental Health Counseling Conversations into Qdrant (Module 4).

Run once after setting QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in .env:
    python scripts/ingest_rag.py

v2: Stores context, response, chunk_idx, and source_row_id as separate payload
    fields in addition to the chunk text. This enables the UI to display
    separate patient Q / counselor A panels in the source expander.

NOTE: If using the shared cluster, 'health-counseling-dataset' is already
      populated. Re-running will upsert (overwrite) existing vectors.
"""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from datasets import load_dataset
from qdrant_client.models import PointStruct

from src.modules.rag_pipeline import RAGPipeline, COLLECTION_NAME
from src.utils.preprocessing import build_qa_chunk, chunk_text


def main():
    print("Loading heliosbrahma/mental_health_counseling_conversations …")
    ds = load_dataset("heliosbrahma/mental_health_counseling_conversations", split="train")
    print(f"  {len(ds)} records loaded.")

    rag = RAGPipeline()
    rag.create_collection()

    # Build structured chunks with full payload
    all_points: list[PointStruct] = []
    for row_id, row in enumerate(ds):
        context  = str(row.get("Context",  "")).strip()
        response = str(row.get("Response", "")).strip()
        if not context or not response:
            continue

        chunk = build_qa_chunk(context, response)

        # Split very long Q&A pairs with overlap to preserve context
        if len(chunk.split()) > 250:
            sub_chunks = chunk_text(chunk, chunk_size=200, overlap=20)
        else:
            sub_chunks = [chunk]

        total_chunks = len(sub_chunks)
        for chunk_idx, sub_chunk in enumerate(sub_chunks):
            all_points.append(
                # Payload fields stored separately for richer UI display:
                #   context     — original patient question
                #   response    — counselor answer
                #   chunk       — embedded text (full Q+A or sub-chunk)
                #   chunk_idx   — position within document
                #   total_chunks — total sub-chunks per source record
                #   source_row_id — original dataset row for traceability
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[],   # filled by embedder below
                    payload={
                        "chunk":          sub_chunk,
                        "context":        context[:500],   # capped for storage
                        "response":       response[:500],
                        "chunk_idx":      chunk_idx,
                        "total_chunks":   total_chunks,
                        "source_row_id":  row_id,
                    },
                )
            )

    print(f"  {len(all_points)} chunks prepared.")

    # Embed in batches and upsert
    BATCH_SIZE = 64
    total_upserted = 0
    texts = [p.payload["chunk"] for p in all_points]

    print("Embedding and ingesting into Qdrant …")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts  = texts[i: i + BATCH_SIZE]
        batch_points = all_points[i: i + BATCH_SIZE]

        embeddings = rag._embedder.encode(
            batch_texts, normalize_embeddings=True, show_progress_bar=False
        )
        for point, emb in zip(batch_points, embeddings):
            point.vector = emb.tolist()

        rag._qdrant.upsert(collection_name=COLLECTION_NAME, points=batch_points)
        total_upserted += len(batch_points)
        print(f"  Upserted {total_upserted}/{len(all_points)} …", end="\r")

    print(f"\n  ✓ {total_upserted} vectors upserted into '{COLLECTION_NAME}'.")
    print(f"  Collection size: {rag.collection_count()} points.")


if __name__ == "__main__":
    main()
