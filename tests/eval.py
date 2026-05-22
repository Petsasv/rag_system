import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

sys.path.append(str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.config import (
    MODEL_PATH,
    N_CTX,
    SYSTEM_RAG,
    SYSTEM_GENERAL,
    MAX_RESPONSE_TOKENS,
    TEMPERATURE,
    REPEAT_PENALTY,
    TOP_K,
    TOP_N,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# STEP FLAG — αλλάζεις εδώ τι τρέχεις
# ─────────────────────────────────────────────
RUN_LLM_ONLY = False
RUN_RAG_VARIANTS = True
RAG_VARIANT_TO_RUN = "150_no_overlap"  # αλλάζεις αυτό για κάθε βήμα


def load_dataset() -> List[Dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_llm() -> Llama:
    print("⏳ Φόρτωση μοντέλου...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        chat_format="zephyr",
        verbose=False,
    )
    print("✅ Μοντέλο έτοιμο\n")
    return llm


# ─────────────────────────────────────────────
# LLM ONLY
# ─────────────────────────────────────────────
def run_llm_only(llm: Llama, dataset: List[Dict]) -> List[Dict]:
    """
    Τρέχει κάθε ερώτηση χωρίς RAG — μόνο το LLM.
    Το μοντέλο απαντά από τη δική του γνώση.
    """
    print("=" * 60)
    print("LLM ONLY (χωρίς RAG)")
    print("=" * 60)

    results = []

    for item in dataset:
        qid = item["id"]
        question = item["question"]

        print(f"[{qid:02d}/{len(dataset)}] {question}")

        messages = [
            {"role": "user", "content": question},
        ]

        try:
            out = llm.create_chat_completion(
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_RESPONSE_TOKENS,
                repeat_penalty=REPEAT_PENALTY,
                stop=["<|user|>", "<|system|>", "<|assistant|>", "</s>"],
            )
            answer = out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            answer = f"ERROR: {e}"

        print(f"     → {answer[:80]}{'...' if len(answer) > 80 else ''}\n")

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "system": "llm_only",
                "score": None,  # θα συμπληρωθεί από τον καθηγητή
            }
        )

    # Αποθήκευση
    out_path = RESULTS_DIR / "llm_only.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"💾 Αποτελέσματα: {out_path}")
    return results


# ─────────────────────────────────────────────
# RAG VARIANT RUNNER
# ─────────────────────────────────────────────

RAG_VARIANTS = {
    "250_overlap": {"max_tokens": 250, "overlap_tokens": 40},
    "250_no_overlap": {"max_tokens": 250, "overlap_tokens": 0},
    "150_overlap": {"max_tokens": 150, "overlap_tokens": 40},
    "150_no_overlap": {"max_tokens": 150, "overlap_tokens": 0},
}


def build_index_for_variant(variant_name: str, variant_cfg: Dict) -> None:
    """
    Χτίζει νέο ChromaDB index για συγκεκριμένο chunking variant.
    Χρησιμοποιεί ξεχωριστό collection για κάθε variant.
    """
    import chromadb
    import glob
    from uuid import uuid4
    from sentence_transformers import SentenceTransformer
    from src.chunker import chunk_pdf
    from src.config import PDF_FOLDER, DB_DIR, EMBED_MODEL_NAME, BATCH_SIZE

    collection_name = f"eval_{variant_name}"
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Αν υπάρχει ήδη, skip
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        col = client.get_collection(collection_name)
        print(
            f"  ⏭️  Index '{collection_name}' υπάρχει ήδη ({col.count()} chunks), skipping...\n"
        )
        return

    print(
        f"  📦 Building index: {collection_name} "
        f"(tokens={variant_cfg['max_tokens']}, "
        f"overlap={variant_cfg['overlap_tokens']})"
    )

    # Διαγραφή αν υπάρχει ήδη
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(EMBED_MODEL_NAME)
    pdf_files = glob.glob(str(PDF_FOLDER / "*.[pP][dD][fF]"))

    total = 0
    for pdf_path in pdf_files:
        filename = Path(pdf_path).name
        chunks = chunk_pdf(
            pdf_path,
            source_name=filename,
            max_tokens=variant_cfg["max_tokens"],
            overlap_tokens=variant_cfg["overlap_tokens"],
        )
        if not chunks:
            continue

        docs = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [str(uuid4()) for _ in chunks]
        embeddings = model.encode(
            [f"passage: {d}" for d in docs],
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
        ).tolist()

        collection.add(
            ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings
        )
        total += len(docs)

    print(f"  ✅ Index ready: {total} chunks\n")


def run_rag_variant(
    llm: Llama,
    dataset: List[Dict],
    variant_name: str,
    variant_cfg: Dict,
) -> List[Dict]:
    """
    Τρέχει κάθε ερώτηση με RAG για συγκεκριμένο chunking variant.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from src.config import (
        DB_DIR,
        EMBED_MODEL_NAME,
        RERANK_MODEL_NAME,
        USE_RERANKER,
        RELEVANCE_THRESHOLD,
    )

    print("=" * 60)
    print(f"RAG VARIANT: {variant_name}")
    print("=" * 60)

    collection_name = f"eval_{variant_name}"
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(collection_name)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME) if USE_RERANKER else None

    results = []

    for item in dataset:
        qid = item["id"]
        question = item["question"]

        print(f"[{qid:02d}/{len(dataset)}] {question}")

        # ── Retrieval ──────────────────────────────────────
        q_vec = embed_model.encode(
            [f"query: {question}"], normalize_embeddings=True
        ).tolist()

        res = collection.query(
            query_embeddings=q_vec, n_results=TOP_K, include=["documents", "metadatas"]
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]

        # ── Reranking ──────────────────────────────────────
        if reranker and docs:
            pairs = [(question, d) for d in docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
        else:
            ranked = [(1.0, d, m) for d, m in zip(docs, metas)]

        # ── Relevance check ────────────────────────────────
        top_score = float(ranked[0][0]) if ranked else 0.0
        if top_score < RELEVANCE_THRESHOLD:
            answer = "Η ερώτηση δεν σχετίζεται με το υλικό Java OOP."
            print(f"     → [IRRELEVANT] score={top_score:.3f}\n")
        else:
            # ── Build context ──────────────────────────────
            context_parts = []
            used_tokens = 0

            from src.config import MAX_CONTEXT_TOKENS
            from src.retriever import Retriever
            import tiktoken

            _tmp = Retriever.__new__(Retriever)
            _tmp.tokenizer = tiktoken.get_encoding("cl100k_base")

            for score, chunk, meta in ranked[:TOP_N]:
                ct = len(_tmp.tokenizer.encode(chunk))
                if used_tokens + ct <= MAX_CONTEXT_TOKENS:
                    src = meta.get("source", "?")
                    cidx = meta.get("chunk_idx", "?")
                    context_parts.append(
                        f"[Απόσπασμα - Πηγή: {src}, Chunk: {cidx}]\n{chunk}"
                    )
                    used_tokens += ct

            context = "\n\n".join(context_parts)
            prompt = f"ΑΠΟΣΠΑΣΜΑΤΑ:\n{context}\n\nΕΡΩΤΗΣΗ: {question}\n\nΑΠΑΝΤΗΣΗ:"
            messages = [
                {"role": "system", "content": SYSTEM_RAG},
                {"role": "user", "content": prompt},
            ]

            try:
                out = llm.create_chat_completion(
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    repeat_penalty=REPEAT_PENALTY,
                    stop=["<|user|>", "<|system|>", "<|assistant|>", "</s>"],
                )
                answer = out["choices"][0]["message"]["content"].strip()
            except Exception as e:
                answer = f"ERROR: {e}"

        print(f"     score={top_score:.3f}")
        print(f"     → {answer[:80]}{'...' if len(answer) > 80 else ''}\n")

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "system": f"rag_{variant_name}",
                "top_score": top_score,
                "score": None,
            }
        )

    # ── Save ───────────────────────────────────────────────
    out_path = RESULTS_DIR / f"rag_{variant_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"💾 Αποτελέσματα: {out_path}\n")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    dataset = load_dataset()
    print(f"📋 Dataset: {len(dataset)} ερωτήσεις\n")

    llm = load_llm()

    if RUN_LLM_ONLY:
        run_llm_only(llm, dataset)
        print("\n✅ Βήμα 1 ολοκληρώθηκε\n")

    if RUN_RAG_VARIANTS:
        variant_cfg = RAG_VARIANTS[RAG_VARIANT_TO_RUN]

        # Χτίζει νέο index μόνο αν δεν είναι το τρέχον σύστημα
        if RUN_RAG_VARIANTS:
            variant_cfg = RAG_VARIANTS[RAG_VARIANT_TO_RUN]
            build_index_for_variant(RAG_VARIANT_TO_RUN, variant_cfg)
            run_rag_variant(llm, dataset, RAG_VARIANT_TO_RUN, variant_cfg)
            print(f"\n✅ RAG variant '{RAG_VARIANT_TO_RUN}' ολοκληρώθηκε\n")

    print("🏁 Τέλος.")


if __name__ == "__main__":
    main()
