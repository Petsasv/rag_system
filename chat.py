import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import tiktoken

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = r"C:\Users\Vasilis\models\llama-krikri-8b-instruct-q4_k_m.gguf"
DB_DIR = "./chroma_db"

# Database
COLLECTION_NAME = "java_oop_course"

# Models
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Retrieval parameters
TOP_K = 20              # Initial retrieval
TOP_N = 4               # After reranking
RELEVANCE_THRESHOLD = -1.0  # Reranker threshold

# LLM parameters
N_CTX = 4096            # Total context window
MAX_RESPONSE_TOKENS = 500
MAX_CONTEXT_TOKENS = 2500  # Reserved for context chunks
TEMPERATURE = 0.0
REPEAT_PENALTY = 1.3

# Tokenizer για token counting
TOKENIZER = tiktoken.get_encoding("cl100k_base")

SYSTEM_RAG = """
Είσαι Καθηγητής Αντικειμενοστραφούς Προγραμματισμού σε Ελληνικό Πανεπιστήμιο.

ΟΔΗΓΙΕΣ ΑΠΑΝΤΗΣΗΣ:
1. Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες από τα δοθέντα αποσπάσματα
2. Εξήγησε με παιδαγωγικό τρόπο (απλά, με παραδείγματα όπου χρειάζεται)
3. Αν κάτι δεν είναι σαφές στα αποσπάσματα, πες το ειλικρινά
4. Για προγραμματιστικές έννοιες, δώσε σύντομο παράδειγμα κώδικα αν βοηθάει
5. ΜΗΝ εφευρίσκεις πληροφορίες που δεν υπάρχουν στα αποσπάσματα

ΣΗΜΑΝΤΙΚΟ:
- Αν τα αποσπάσματα ΔΕΝ απαντούν στην ερώτηση, ΜΗΝ προσπαθήσεις να απαντήσεις από μόνος σου
- Αντ' αυτού, πες: "Δεν βρέθηκε σχετική πληροφορία στα αποσπάσματα. Προτείνω να ρωτήσεις πιο συγκεκριμένα ή να συμβουλευτείς το βιβλίο."

ΦΟΡΜΑ ΑΠΑΝΤΗΣΗΣ:
- Σύντομη εξήγηση (2-3 προτάσεις)
- Παράδειγμα (μόνο αν χρειάζεται)
- Σχόλιο για περαιτέρω μελέτη (προαιρετικά)

ΜΗΝ γράφεις πηγές ή παραπομπές μέσα στην απάντηση - εμφανίζονται ξεχωριστά.
"""

SYSTEM_GENERAL = """Είσαι ένας ευγενικός βοηθός για το μάθημα Java OOP.
Απάντα σύντομα στα ελληνικά. Για τεχνικές ερωτήσεις, παραπέμπω στις σημειώσεις."""

def count_tokens(text: str) -> int:
    """
    Μέτρηση tokens με tiktoken.
    """
    return len(TOKENIZER.encode(text))

def is_small_talk(q: str) -> bool:
    """
    Ανίχνευση απλών χαιρετισμών και γενικών ερωτήσεων.
    """
    ql = q.lower().strip()
    
    # Χαιρετισμοί
    greetings = [
        "γεια", "καλημέρα", "καλησπέρα", "καληνύχτα",
        "τι κάνεις", "πως είσαι", "τι γίνεται",
        "hello", "hi", "hey"
    ]
    
    # Γενικές ερωτήσεις
    general_questions = [
        "ποιος είσαι", "τι είσαι", "πώς λέγεσαι",
        "μπορείς να", "what are you", "who are you"
    ]
    
    words = ql.split()
    
    # Αν είναι πολύ σύντομο (≤3 λέξεις) και περιέχει χαιρετισμό
    if len(words) <= 3 and any(g in ql for g in greetings):
        return True
    
    # Αν είναι γενική ερώτηση
    if any(gq in ql for gq in general_questions):
        return True
    
    return False

def retrieve_candidates(collection, embed_model, query: str, k: int) -> Tuple[List[str], List[Dict]]:
    """
    Σημασιολογική αναζήτηση με E5 embeddings.
    
    Returns:
        (documents, metadatas)
    """
    # E5 formatting
    q_text = f"query: {query}"
    
    # Encode & normalize
    q_vec = embed_model.encode([q_text], normalize_embeddings=True).tolist()
    
    # Query ChromaDB
    try:
        res = collection.query(
            query_embeddings=q_vec,
            n_results=k,
            include=["documents", "metadatas"]
        )
        
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        
        logger.info(f"Retrieved {len(docs)} candidates from database")
        return docs, metas
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return [], []

def fit_chunks_to_budget(
    ranked_chunks: List[Tuple[float, str, Dict]],
    query: str,
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> List[Tuple[float, str, Dict]]:
    """
    Επιλέγει chunks που χωράνε στο context budget.
    
    Args:
        ranked_chunks: List of (score, text, metadata)
        query: User query
        max_tokens: Maximum tokens for context
    
    Returns:
        Filtered list of chunks
    """
    system_tokens = count_tokens(SYSTEM_RAG)
    query_tokens = count_tokens(query)
    
    available = max_tokens - system_tokens - query_tokens - 100  # Safety buffer
    
    selected = []
    used_tokens = 0
    
    for score, chunk, meta in ranked_chunks:
        chunk_tokens = count_tokens(chunk)
        
        if used_tokens + chunk_tokens <= available:
            selected.append((score, chunk, meta))
            used_tokens += chunk_tokens
            logger.debug(f"Added chunk {meta.get('chunk_idx')} ({chunk_tokens} tokens)")
        else:
            logger.warning(f"Chunk {meta.get('chunk_idx')} skipped (would exceed budget)")
            break
    
    logger.info(f"Selected {len(selected)} chunks using {used_tokens}/{available} tokens")
    return selected

def build_context(chunks: List[Tuple[float, str, Dict]]) -> Tuple[str, List[str]]:
    """
    Κατασκευάζει το context string και τη λίστα πηγών.
    
    Returns:
        (context_string, sources_list)
    """
    context_parts = []
    sources = []
    
    for idx, (score, doc, meta) in enumerate(chunks, 1):
        src = meta.get("source", "Unknown")
        cidx = meta.get("chunk_idx", "?")
        
        # Context με αρίθμηση για ευκολία
        context_parts.append(f"[Απόσπασμα {idx} - Πηγή: {src}, Chunk: {cidx}]\n{doc}")
        
        # Source για display
        sources.append(f"{src} (Chunk {cidx})")
        
        logger.debug(f"Chunk {idx}: score={score:.3f}, source={src}, chunk_idx={cidx}")
    
    return "\n\n".join(context_parts), sources

def main():
    logger.info("="*60)
    logger.info("AI TUTOR INITIALIZATION")
    logger.info("="*60)
    
    # Load models
    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    logger.info("Loading reranker model...")
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    
    # Connect to database
    logger.info("Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Connected to collection '{COLLECTION_NAME}'")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    # Load LLM
    logger.info(f"Loading LLM from {MODEL_PATH}...")
    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_gpu_layers=-1,  # Use GPU
            chat_format="zephyr",
            verbose=False
        )
        logger.info("LLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        return
    
    logger.info("="*60)
    logger.info("AI TUTOR READY!")
    logger.info("Πληκτρολογήστε 'exit' ή 'quit' για έξοδο")
    logger.info("="*60 + "\n")
    
    # Main loop
    while True:
        try:
            user_input = input("\n🎓 Φοιτητής: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "έξοδος"]:
                logger.info("Shutting down...")
                break
            
            logger.info(f"Query: {user_input}")
            
            # Check for small talk
            if is_small_talk(user_input):
                logger.info("Detected small talk - using general response")
                messages = [
                    {"role": "system", "content": SYSTEM_GENERAL},
                    {"role": "user", "content": user_input}
                ]
                sources = []
            
            else:
                # RETRIEVAL
                docs, metas = retrieve_candidates(collection, embed_model, user_input, TOP_K)
                
                if not docs:
                    print("\n Σφάλμα στην ανάκτηση δεδομένων. Δοκιμάστε ξανά.")
                    continue
                
                # RERANKING
                logger.info("Reranking candidates...")
                pairs = [(user_input, d) for d in docs]
                scores = reranker.predict(pairs)
                
                # Sort by score
                ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
                
                # Log top scores
                logger.info(f"Top 3 reranker scores: {[f'{s:.3f}' for s, _, _ in ranked[:3]]}")
                
                # Check relevance threshold
                best_score = ranked[0][0]
                if best_score < RELEVANCE_THRESHOLD:
                    logger.warning(f"Best score {best_score:.3f} below threshold {RELEVANCE_THRESHOLD}")
                    print("\n Καθηγητής: Λυπάμαι, η ερώτησή σας δεν φαίνεται να σχετίζεται με το υλικό του μαθήματος Java OOP.")
                    print("   Μπορείτε να διατυπώσετε την ερώτησή σας διαφορετικά;")
                    continue
                
                # Select top chunks within budget
                selected = fit_chunks_to_budget(ranked[:TOP_N], user_input)
                
                if not selected:
                    print("\n Δεν υπάρχει διαθέσιμο περιεχόμενο για απάντηση.")
                    continue
                
                # Build context
                context, sources = build_context(selected)
                
                # Build prompt
                prompt = f"""
                Χρησιμοποίησε τα παρακάτω αποσπάσματα για να απαντήσεις.
                ΑΠΟΣΠΑΣΜΑΤΑ:
                {context}

                ΕΡΩΤΗΣΗ: {user_input}

                ΑΠΑΝΤΗΣΗ:
                """
                
                messages = [
                    {"role": "system", "content": SYSTEM_RAG},
                    {"role": "user", "content": prompt}
                ]
                
                # Log token usage
                total_tokens = count_tokens(SYSTEM_RAG) + count_tokens(prompt)
                logger.info(f"Prompt tokens: {total_tokens}/{N_CTX}")
            
            # GENERATION
            try:
                logger.info("Generating response...")
                out = llm.create_chat_completion(
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    repeat_penalty=REPEAT_PENALTY,
                    stop=["<|user|>", "<|system|>", "<|assistant|>", "</s>", "\n\n🎓"]
                )
                
                reply = out["choices"][0]["message"]["content"].strip()
                
                # response
                print(f"\n Καθηγητής: {reply}")
                
                # sources
                if sources:
                    print(f"\n Πηγές: {', '.join(set(sources))}")
                
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                print("\n Σφάλμα κατά την παραγωγή απάντησης. Δοκιμάστε ξανά.")
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            break
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("\n Σφάλμα. Δοκιμάστε ξανά.")

if __name__ == "__main__":
    main()