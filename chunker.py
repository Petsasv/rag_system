import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from pdfminer.high_level import extract_text
import tiktoken

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Εξαγωγή πρωτογενούς κειμένου από το PDF.
    """
    try:
        text = extract_text(pdf_path)
        return text or ""
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """
    Καθαρισμός και κανονικοποίηση κειμένου.
    """

    if not text:
        return ""

    # Ενοποίηση διαφορετικών τύπων αλλαγής γραμμής
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Αφαίρεση common PDF artifacts (page numbers, headers)
    # Παράδειγμα: "Σελίδα 5" ή "Page 5"
    text = re.sub(r"(?i)(σελίδα|page)\s*\d+", "", text)

    # Αφαίρεση μεμονωμένων αριθμών σε ξεχωριστές γραμμές (συνήθως page numbers)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Κανονικοποίηση κενών (whitespace) για μείωση των άχρηστων tokens
    text = re.sub(r"[ \t]+", " ", text)

    # Διόρθωση συλλαβισμού PDF (π.χ. "προ-\nγραμμα" -> "προγραμμα")
    # Πολύ σημαντικό για να μην σπάνε οι τεχνικοί όροι
    text = re.sub(r"-\n(?=\w)", "", text)

    # Περιορισμός πολλαπλών κενών γραμμών που σπαταλούν το context window
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Αφαίρεση περιττών κενών στην αρχή και το τέλος κάθε γραμμής
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()

def chunk_text(
    text: str,
    *,
    source: str,
    max_tokens: int = 250,
    overlap_tokens: int = 40,
    tokenizer_name: str = "cl100k_base",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Χωρίζει το κείμενο σε τμήματα με semantic awareness.
    """
    # Χρήση του cl100k_base (tokenizer)
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    extra_metadata = extra_metadata or {}

    def tok_len(s: str) -> int:
        """
        Βοηθητική συνάρτηση μέτρησης tokens.
        """
        return len(tokenizer.encode(s))
    
    # Χωρισμός σε παραγράφους με βάση τις διπλές αλλαγές γραμμής
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return []

    raw_chunks: List[str] = []
    cur_parts: List[str] = []
    cur_toks = 0

    def flush():
        """Οριστικοποίηση ενός chunk και προσθήκη στη λίστα."""
        nonlocal cur_parts, cur_toks
        if cur_parts:
            raw_chunks.append("\n\n".join(cur_parts).strip())
            cur_parts = []
            cur_toks = 0

    for p in paragraphs:
        p_toks = tok_len(p)
        if p_toks == 0:
            continue

        # Αν μια παράγραφος από μόνη της υπερβαίνει το όριο, τη χωρίζουμε αναγκαστικά
        if p_toks > max_tokens:
            flush()
            toks = tokenizer.encode(p)
            for i in range(0, len(toks), max_tokens):
                part = tokenizer.decode(toks[i : i + max_tokens]).strip()
                if part:
                    raw_chunks.append(part)
            continue

        # Προσθήκη παραγράφου στο τρέχον chunk αν χωράει στο 'budget' των tokens
        if cur_toks + p_toks <= max_tokens:
            cur_parts.append(p)
            cur_toks += p_toks
        else:
            # Το chunk γέμισε, το αποθηκεύουμε και ξεκινάμε νέο
            flush()
            cur_parts = [p]
            cur_toks = p_toks

    flush()

    # Εφαρμογή επικάλυψης (overlap) μεταξύ των chunks.
    if overlap_tokens > 0 and len(raw_chunks) > 1:
        overlapped: List[str] = []
        prev_tail = []
        for i, ch in enumerate(raw_chunks):
            ch_toks = tokenizer.encode(ch)
            if i == 0:
                overlapped.append(ch)
            else:
                # Προσάρτηση της 'ουράς' του προηγούμενου chunk
                prefix = tokenizer.decode(prev_tail).strip()
                overlapped.append((prefix + "\n" + ch).strip() if prefix else ch)

            # Ενημέρωση της 'ουράς' για το επόμενο chunk
            prev_tail = ch_toks[-overlap_tokens:] if len(ch_toks) > overlap_tokens else ch_toks
        raw_chunks = overlapped

    # Δημιουργία τελικών αντικειμένων Chunk με metadata
    out: List[Chunk] = []
    for idx, ch in enumerate(raw_chunks):
        md = {
            "source": source,
            "chunk_idx": idx,
            "token_count": tok_len(ch),
            **extra_metadata
        }
        out.append(Chunk(text=ch, metadata=md))
    return out

def chunk_pdf(
    pdf_path: str,
    *,
    source_name: Optional[str] = None,
    max_tokens: int = 250,
    overlap_tokens: int = 40,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Κύρια συνάρτηση ενός αρχείου PDF.
    """
    raw = extract_text_from_pdf(pdf_path)
    if not raw:
        logger.warning(f"No text extracted from {pdf_path}")
        return []
    
    cleaned = clean_text(raw)
    source = source_name or pdf_path

    chunks = chunk_text(
        cleaned,
        source=source,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        extra_metadata=extra_metadata,
    )

    if chunks:
        avg_tokens = sum(c.metadata["token_count"] for c in chunks) / len(chunks)
        logger.info(f"{source}: {len(chunks)} chunks, avg {avg_tokens:.0f} tokens")

    return chunks