import os
import glob
import shutil
from uuid import uuid4
from typing import List
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .chunker import chunk_pdf, Chunk
from .config import (
    PDF_FOLDER,
    DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    MAX_TOKENS,
    OVERLAP_TOKENS,
    BATCH_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def e5_format_passages(texts: List[str]) -> List[str]:
    """Εφαρμογή του prefix 'passage:' για το E5."""
    return [f"passage: {t}" for t in texts]


def batch_encode(
    model, texts: List[str], batch_size: int = BATCH_SIZE
) -> List[List[float]]:
    """Encode texts σε batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def main(reset_db: bool = False):
    """Κύρια συνάρτηση indexing."""

    if reset_db and os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        logger.info("Database Reset: Διαγραφή παλιάς βάσης δεδομένων.")

    logger.info(f"Φόρτωση του μοντέλου {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.[pP][dD][fF]"))
    logger.info(f"Βρέθηκαν {len(pdf_files)} PDF αρχεία.")

    total_chunks = 0
    failed_files = []

    for pdf_path in tqdm(pdf_files, desc="Indexing PDFs"):
        filename = os.path.basename(pdf_path)

        try:
            # Chunking
            chunks: List[Chunk] = chunk_pdf(
                pdf_path,
                source_name=filename,
                max_tokens=MAX_TOKENS,
                overlap_tokens=OVERLAP_TOKENS,
                extra_metadata={"type": "pdf"},
            )

            if not chunks:
                logger.warning(f"Δεν παρήχθησαν chunks για {filename}")
                continue

            docs = [c.text for c in chunks]
            metadatas = [c.metadata for c in chunks]
            ids = [str(uuid4()) for _ in chunks]

            to_embed = e5_format_passages(docs)

            embeddings = batch_encode(model, to_embed, batch_size=BATCH_SIZE)

            collection.add(
                ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings
            )

            total_chunks += len(docs)
            logger.info(f" {filename}: {len(docs)} chunks")

        except Exception as e:
            logger.error(f" Error processing {filename}: {e}")
            failed_files.append(filename)
            continue

    # Results for debugging
    print("Indexing done!")
    print("=" * 60)
    print(f" Files succeeded: {len(pdf_files) - len(failed_files)}/{len(pdf_files)}")
    print(f" Total chunks: {total_chunks}")
    if len(pdf_files) > 0:
        print(f" Average chunk/PDF: {total_chunks/len(pdf_files):.1f}")

    if failed_files:
        print(f"\n Files failed: ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main(reset_db=False)
