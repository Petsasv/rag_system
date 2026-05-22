import logging
from typing import List, Tuple, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import tiktoken
from .config import USE_RERANKER
from .config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    TOP_K,
    TOP_N,
    RELEVANCE_THRESHOLD,
    MAX_CONTEXT_TOKENS,
)

logger = logging.getLogger(__name__)


class Retriever:
    """Χειρίζεται το retrieval και reranking των chunks."""

    def __init__(self):
        logger.info("Initializing Retriever...")

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder(RERANK_MODEL_NAME) if USE_RERANKER else None

        client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = client.get_collection(name=COLLECTION_NAME)

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info("Retriever loaded")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def retrieve(self, query: str, k: int = TOP_K) -> Tuple[List[str], List[Dict]]:
        """Semantic search με E5 embeddings."""

        q_text = f"query: {query}"

        q_vec = self.embed_model.encode([q_text], normalize_embeddings=True).tolist()

        try:
            res = self.collection.query(
                query_embeddings=q_vec, n_results=k, include=["documents", "metadatas"]
            )

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]

            logger.info(f"Retrieved {len(docs)} candidates")
            return docs, metas

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return [], []

    def rerank(
        self, query: str, docs: List[str], metas: List[Dict]
    ) -> List[Tuple[float, str, Dict]]:
        """Reranking."""
        if not docs:
            return []

        if self.reranker is None:
            logger.info("Reranker disabled - using embedding order")
            return [(1.0, doc, meta) for doc, meta in zip(docs, metas)]

        pairs = [(query, d) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

        print(f"\nQuery: {query[:60]}")
        for s, _, m in ranked[:5]:
            print(f"  {s:.4f} | {m.get('source','?')} chunk {m.get('chunk_idx','?')}")

        logger.info(f"Top 3 scores: {[f'{s:.3f}' for s, _, _ in ranked[:4]]}")

        return ranked

    def select_chunks(
        self,
        ranked: List[Tuple[float, str, Dict]],
        query: str,
        system_prompt: str,
        n: int = TOP_N,
        max_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> Tuple[List[Tuple[float, str, Dict]], bool]:
        """Επιλέγει chunks που χωράνε στο context budget."""

        if not ranked or ranked[0][0] < RELEVANCE_THRESHOLD:
            logger.warning(f"Best score {ranked[0][0]:.3f} below threshold")
            return [], False

        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        available = max_tokens - system_tokens - query_tokens - 100

        selected = []
        used_tokens = 0

        for score, chunk, meta in ranked[:n]:
            chunk_tokens = self.count_tokens(chunk)

            if used_tokens + chunk_tokens <= available:
                selected.append((score, chunk, meta))
                used_tokens += chunk_tokens
            else:
                logger.warning(f"Chunk {meta.get('chunk_idx')} skipped (budget)")
                break

        logger.info(
            f"Selected {len(selected)} chunks ({used_tokens}/{available} tokens)"
        )
        return selected, True

    def build_context(
        self, chunks: List[Tuple[float, str, Dict]]
    ) -> Tuple[str, List[str]]:
        """Κατασκευή context string και sources."""
        context_parts = []
        sources = []

        for idx, (score, doc, meta) in enumerate(chunks, 1):
            src = meta.get("source", "Unknown")
            cidx = meta.get("chunk_idx", "?")

            context_parts.append(
                f"[Απόσπασμα {idx} - Πηγή: {src}, Chunk: {cidx}]\n{doc}"
            )
            sources.append(f"{src} (Chunk {cidx})")

        return "\n\n".join(context_parts), sources
