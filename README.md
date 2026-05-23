# AI Tutor — Java OOP

> Ελληνικός εκπαιδευτικός βοηθός βασισμένος σε **RAG** (Retrieval-Augmented Generation) για το μάθημα **Αντικειμενοστραφής Προγραμματισμός**.

**Πτυχιακή Εργασία** · ΔΙ.ΠΑ.Ε. · Τμήμα Μηχανικών Πληροφορικής και Ηλεκτρονικών Συστημάτων  
 Πετσαλάκης Βασίλης (ΑΜ 185328) · Επιβλέπων: Καθ. Παναγιώτης Αδαμίδης

---

## Χαρακτηριστικά

- **Τοπική εκτέλεση** — Καμία εξάρτηση από εμπορικά APIs, πλήρης προστασία ιδιωτικότητας
- **Ελληνόφωνη υποστήριξη** — Χρήση του εξειδικευμένου μοντέλου Llama-Krikri-8B
- **RAG Pipeline** — Dense retrieval με E5 embeddings και Cross-encoder reranking
- **Πολυγυρικός διάλογος** — Δομημένη μνήμη με αυτόματη σύνοψη παλαιότερων turns
- **Web Interface** — Καθαρή διεπαφή με syntax highlighting και dark mode
- **Διαφάνεια** — Κάθε απάντηση συνοδεύεται από αναφορά στις πηγές

---

## Αρχιτεκτονική

```
                    ┌─────────────────────────────────────────┐
                    │           OFFLINE INDEXING              │
                    └─────────────────────────────────────────┘

     PDFs ──►  Chunking  ──►  Embeddings (E5)  ──►  ChromaDB
                                                        │
                    ┌───────────────────────────────────┘
                    │
                    ▼
                    ┌─────────────────────────────────────────┐
                    │            ONLINE INFERENCE             │
                    └─────────────────────────────────────────┘

   Query ──►  Rewriting  ──►  Retrieval  ──►  Reranking (BGE)  ──►  LLM (Krikri)  ──►  Response
```

### Τεχνολογικό Stack

| Συνιστώσα | Επιλογή |
|-----------|---------|
| **Language Model** | `Llama-Krikri-8B-Instruct` (GGUF Q4_K_M) |
| **Embedding Model** | `intfloat/multilingual-e5-large-instruct` |
| **Reranker** | `BAAI/bge-reranker-v2-m3` |
| **Vector Database** | ChromaDB (HNSW indexing, cosine similarity) |
| **Backend** | FastAPI |
| **Frontend** | HTML / CSS / JavaScript (vanilla) |

---

## Εγκατάσταση

### Προαπαιτούμενα

- **Python 3.11+**
- **NVIDIA GPU** με τουλάχιστον 4GB VRAM (συνιστάται 6GB+)
- **CUDA Toolkit 12.1**
- **~10GB** ελεύθερος χώρος

### Βήμα 1 — Clone

```bash
git clone https://github.com/Petsasv/rag_system.git
cd llama3_2-RAG
```

### Βήμα 2 — Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

### Βήμα 3 — Dependencies

```bash
pip install -r requirements.txt
```

### Βήμα 4 — GPU Support

> **Κρίσιμο βήμα.** Χωρίς αυτό, το σύστημα θα τρέχει αργά σε CPU.

```bash
# PyTorch με CUDA 12.1
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121

# llama-cpp-python με CUDA
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Επαλήθευση:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Πρέπει να εμφανιστεί `CUDA: True`.

### Βήμα 5 — Λήψη Μοντέλου (5GB)

Κατεβάστε το αρχείο **`llama-krikri-8b-instruct-q4_k_m.gguf`** από το Hugging Face:

**[Direct Download Link](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF/resolve/main/llama-krikri-8b-instruct-q4_k_m.gguf)**

ή μέσω CLI:

```bash
pip install huggingface_hub
huggingface-cli download ilsp/Llama-Krikri-8B-Instruct-GGUF llama-krikri-8b-instruct-q4_k_m.gguf --local-dir models/
```

Τοποθετήστε το αρχείο στο φάκελο:

```
models/llama-krikri-8b-instruct-q4_k_m.gguf
```

### Βήμα 6 — Εκπαιδευτικό Υλικό

Τοποθετήστε τα PDF αρχεία του μαθήματος στον φάκελο:

```
data/pdfs/
```

### Βήμα 7 — Δημιουργία Ευρετηρίου

```bash
python -m src.build_index
```

---

## Εκτέλεση

```bash
uvicorn api.main:app --reload
```

Άνοιξε browser στο **http://localhost:8000**
