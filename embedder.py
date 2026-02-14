import os
import glob
import shutil
from uuid import uuid4
from typing import List
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chunker import chunk_pdf, Chunk

# Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_FOLDER = r"C:\Users\Vasilis\Desktop\java_pdfs"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "java_oop_course"

# Διαγραφή και επαναδημιουργία της βάσης
RESET_DB = False

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

MAX_TOKENS = 250
OVERLAP_TOKENS = 40
BATCH_SIZE = 64

def e5_format_passages(texts: List[str]) -> List[str]:
    """
    Εφαρμογή του απαραίτητου prefix 'passage: ' για το μοντέλο E5.
    """
    return [f"passage: {t}" for t in texts]

def batch_encode(model, texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """
    Encode texts σε batches για καλύτερη διαχείριση μνήμης.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size
        )
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings

def main():
    # Καθαρισμός φακέλου βάσης αν ζητηθεί reset
    if RESET_DB and os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        logger.info("Database Reset: Διαγραφή παλιάς βάσης δεδομένων.")

    logger.info("Φόρτωση του μοντέλου (Embedding Model)...")    
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Σύνδεση με ChromaDB
    client = chromadb.PersistentClient(path=DB_DIR)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Εντοπισμός αρχείων PDF
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.[pP][dD][fF]"))
    logger.info(f"Βρέθηκαν {len(pdf_files)} PDF αρχεία.")

    total_chunks = 0
    failed_files = []

    for pdf_path in tqdm(pdf_files, desc="Indexing PDFs"):
        filename = os.path.basename(pdf_path)

        try:
            # Μετατροπή PDF σε Chunks
            chunks: List[Chunk] = chunk_pdf(
                pdf_path,
                source_name=filename,
                max_tokens=MAX_TOKENS,
                overlap_tokens=OVERLAP_TOKENS,
                extra_metadata={"type": "pdf"}
            )

            if not chunks:
                logger.warning(f"Δεν παρήχθησαν chunks για {filename}, παράκαμψη.")
                continue

            docs = [c.text for c in chunks]
            metadatas = [c.metadata for c in chunks]
            ids = [str(uuid4()) for _ in chunks]
        
            # Προσθήκη του passage: prefix
            to_embed = e5_format_passages(docs)

            # Batch encoding για performance
            embeddings = batch_encode(model, to_embed, batch_size=BATCH_SIZE)

            # Αποθήκευση στη ChromaDB
            collection.add(
                ids=ids, 
                documents=docs, 
                metadatas=metadatas, 
                embeddings=embeddings
            )

            total_chunks += len(docs)
            logger.info(f"✓ {filename}: {len(docs)} chunks")

        except Exception as e:
            logger.error(f"✗ Error processing {filename}: {e}")
            failed_files.append(filename)
            continue

    # Τελικά στατιστικά
    print("\n" + "="*60)
    print(" Η ΔΙΑΔΙΚΑΣΙΑ INDEXING ΟΛΟΚΛΗΡΩΘΗΚΕ!")
    print("="*60)
    print(f"✓ Επιτυχημένα αρχεία: {len(pdf_files) - len(failed_files)}/{len(pdf_files)}")
    print(f"✓ Συνολικά chunks: {total_chunks}")
    if len(pdf_files) > 0:
        print(f"✓ Μέσος όρος chunks/PDF: {total_chunks/len(pdf_files):.1f}")

    if failed_files:
        print(f"\n✗ Αποτυχημένα αρχεία ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")
    
    print("="*60 + "\n")
            
if __name__ == "__main__":
    main()