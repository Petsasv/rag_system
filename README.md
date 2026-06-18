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

## Εγκατάσταση (Windows)

### Προαπαιτούμενα

- **Python 3.12** (⚠️ **όχι** 3.13 — δεν υπάρχουν διαθέσιμα CUDA wheels για llama-cpp-python)
- **NVIDIA GPU** (συνιστάται 6GB+ VRAM για πλήρες GPU offloading)
- **CUDA Toolkit 13.0** (η εγκατάστασή του εξηγείται παρακάτω)
- **~12GB** ελεύθερος χώρος (3GB CUDA + 5GB μοντέλο + 2GB venv + 2GB caches)

> **Πρόσοχη**: Το llama-cpp-python χρειάζεται **pre-built wheel ταιριαστό με την αρχιτεκτονική της GPU σας**. Παρακάτω παρέχονται οδηγίες για κάθε σύγχρονη NVIDIA GPU.

### Βήμα 1 — Εγκατάσταση Python 3.12

Κατεβάστε από: **https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe**

⚠️ Στον installer:
- ✅ **Add python.exe to PATH** (πρώτη οθόνη, κάτω)
- ✅ **Install for all users**

### Βήμα 2 — Clone

```powershell
git clone https://github.com/Petsasv/rag_system.git
cd rag_system
```

### Βήμα 3 — Virtual Environment (με Python 3.12)

Αν έχετε και Python 3.13 παράλληλα, χρησιμοποιήστε το πλήρες path:

```powershell
& "C:\Users\<YOUR_USER>\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
```

Επιβεβαίωση:

```powershell
python --version
```

Πρέπει να δείτε `Python 3.12.x`.

### Βήμα 4 — Dependencies

```powershell
pip install -r requirements.txt
```

> ⚠️ Αν εμφανιστεί `UnicodeDecodeError`: το `requirements.txt` πρέπει να είναι **UTF-8 encoding χωρίς ελληνικούς χαρακτήρες** στα σχόλια. Ανοίξτε με Notepad → Save As → Encoding: UTF-8.

### Βήμα 5 — PyTorch με CUDA Support

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Επαλήθευση:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Πρέπει να εμφανιστεί `CUDA: True`.

### Βήμα 6 — llama-cpp-python με CUDA (κρίσιμο βήμα)

Πρέπει να επιλέξετε το **σωστό pre-built wheel** για την GPU σας. Δείτε τον παρακάτω πίνακα:

| GPU σας | Αρχιτεκτονική | Wheel URL (Python 3.12) |
|---------|----------------|--------------------------|
| RTX 30xx (3060, 3070, 3080, 3090) | Ampere (sm86) | [download](https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/v0.3.16-cuda13.0-sm86-py312/llama_cpp_python-0.3.16+cuda13.0.sm86.ampere-cp312-cp312-win_amd64.whl) |
| RTX 40xx (4060, 4070, 4080, 4090) | Ada Lovelace (sm89) | [download](https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/v0.3.16-cuda13.0-sm89-py312/llama_cpp_python-0.3.16+cuda13.0.sm89.ada-cp312-cp312-win_amd64.whl) |
| RTX 50xx (5070, 5080, 5090) | Blackwell (sm100) | [download](https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/v0.3.16-cuda13.0-sm100-py312/llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp312-cp312-win_amd64.whl) |
| RTX 20xx, GTX 16xx | Turing (sm75) | [download](https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/v0.3.16-cuda13.0-sm75-py312/llama_cpp_python-0.3.16+cuda13.0.sm75.turing-cp312-cp312-win_amd64.whl) |

Παράδειγμα για RTX 3060:

```powershell
pip install https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/v0.3.16-cuda13.0-sm86-py312/llama_cpp_python-0.3.16+cuda13.0.sm86.ampere-cp312-cp312-win_amd64.whl
```

### Βήμα 7 — Εγκατάσταση CUDA Toolkit 13.0

Τα παραπάνω wheels χρειάζονται τα **CUDA 13.0 runtime DLLs**. Κατεβάστε από:

🔗 **https://developer.nvidia.com/cuda-13-0-0-download-archive**

Επιλογές:
- Operating System: Windows
- Architecture: x86_64
- Version: 11 ή 10 (ανάλογα με τα Windows σας)
- Installer Type: `exe (local)`

Επιλέξτε **Express Installation**. Αγνοήστε το warning για Visual Studio (δεν χρειάζεται).

⚠️ **NVIDIA Driver απαίτηση**: 580+ (Έλεγξε με `nvidia-smi`)

### Βήμα 8 — Restart PowerShell & Test

Κλείστε το PowerShell window και ανοίξτε νέο:

```powershell
cd <path-to-project>
.\venv\Scripts\Activate.ps1
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
```

> **Αν δώσει error για missing DLL** (`Could not find module... or one of its dependencies`): αντιγράψτε χειροκίνητα τα CUDA DLLs στον φάκελο llama_cpp:
> 
> ```powershell
> copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cudart64_13.dll" "venv\Lib\site-packages\llama_cpp\lib\"
> copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cublas64_13.dll" "venv\Lib\site-packages\llama_cpp\lib\"
> copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cublasLt64_13.dll" "venv\Lib\site-packages\llama_cpp\lib\"
> ```

### Βήμα 9 — Λήψη Μοντέλου (5GB)

Κατεβάστε το **`llama-krikri-8b-instruct-q4_k_m.gguf`** από το Hugging Face:

🔗 **[Direct Download Link](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF/resolve/main/llama-krikri-8b-instruct-q4_k_m.gguf)**

ή μέσω CLI:

```powershell
pip install huggingface_hub
huggingface-cli download ilsp/Llama-Krikri-8B-Instruct-GGUF llama-krikri-8b-instruct-q4_k_m.gguf --local-dir models/
```

Τοποθετήστε στον φάκελο:

```
models/llama-krikri-8b-instruct-q4_k_m.gguf
```

### Βήμα 10 — Εκπαιδευτικό Υλικό & Indexing

Τοποθετήστε τα PDF στον φάκελο:

```
data/pdfs/
```

Δημιουργήστε το ευρετήριο (πρώτη φορά μόνο):

```powershell
python -m src.build_index
```

---

## Εκτέλεση

```powershell
uvicorn api.main:app --reload
```

Ανοίξτε browser στο **http://localhost:8000**

> Στην **πρώτη εκτέλεση** θα κατέβουν τα embedding/reranker μοντέλα από Hugging Face (~3.3GB). Καθυστέρηση 5-10 λεπτά. Στις επόμενες εκτελέσεις είναι cached.

---

## Troubleshooting

### Πρόβλημα: `UnicodeDecodeError: 'charmap' codec can't decode`
**Αιτία**: Το `requirements.txt` περιέχει ελληνικούς χαρακτήρες σε σύστημα με Greek locale (cp1253).  
**Λύση**: Αντικαταστήστε όλα τα ελληνικά σχόλια με αγγλικά. Αποθηκεύστε ως UTF-8 χωρίς BOM.

### Πρόβλημα: `Failed building wheel for llama-cpp-python` (CMake error)
**Αιτία**: Το pip προσπαθεί να κάνει build από source γιατί δεν βρίσκει pre-built wheel.  
**Λύση**: Χρησιμοποιήστε το **direct wheel URL** από τον πίνακα στο Βήμα 6.

### Πρόβλημα: `Could not find module 'llama.dll' (or one of its dependencies)`
**Αιτία**: Λείπουν τα CUDA 13.0 runtime DLLs.  
**Λύση**: Δείτε το note στο Βήμα 8 (manual DLL copy).

### Πρόβλημα: GPU utilization χαμηλό (<20%)
**Αιτία**: Το μοντέλο τρέχει σε hybrid CPU+GPU mode (δεν χωράει στη VRAM).  
**Λύση**: Δείτε τα startup logs — αν δείχνει `offloaded XX/33 layers to GPU` με XX < 33, χρειάζεστε περισσότερη VRAM ή μειώστε `N_CTX` στο `config.py` σε `2048`.

### Επιβεβαίωση GPU offloading
Στα startup logs ψάξτε για:
```
load_tensors: offloaded 33/33 layers to GPU
```
Αν δείτε `33/33` → όλο το μοντέλο στο GPU. ✅

Παράλληλα, σε άλλο PowerShell:
```powershell
nvidia-smi -l 2
```
Memory Usage πρέπει να είναι ~5500-6000 MiB όταν το server είναι φορτωμένος.

---

## Δομή Project

```
rag_system/
├── api/
│   └── main.py              # FastAPI server, /chat endpoint
├── src/
│   ├── config.py            # Κεντρικές παράμετροι
│   ├── chunker.py           # PDF extraction + token-based chunking
│   ├── build_index.py       # Δημιουργία ChromaDB
│   ├── retriever.py         # Dense retrieval + cross-encoder reranking
│   ├── conversation.py      # Memory management (LangChain)
│   └── utils.py             # Small talk + query rewriting
├── ui/                      # Frontend (HTML/CSS/JS)
├── models/                  # GGUF model files (5GB+, gitignored)
├── chroma_db/               # Vector database (gitignored)
├── data/pdfs/               # Source PDFs (gitignored)
├── requirements.txt
└── README.md
```

---