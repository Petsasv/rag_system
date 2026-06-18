import logging

logger = logging.getLogger(__name__)


def is_small_talk(query: str) -> bool:
    ql = query.lower().strip()

    greetings = ["γεια", "καλημέρα", "καλησπέρα", "καληνύχτα", "hello", "hi", "hey"]
    general = ["ποιος είσαι", "τι είσαι", "πώς λέγεσαι"]
    meta = [
        "δεν ζήτησα",
        "δεν ρώτησα",
        "αυτό δεν το είπα",
        "λάθος απάντησες",
        "δεν καταλαβαίνεις",
        "άλλαξε θέμα",
        "συνόψισε",
        "τι έχουμε μάθει",
        "τι είπαμε",
        "τι συζητήσαμε",
        "ανακεφαλαίωσε",
    ]

    words = ql.split()
    if len(words) <= 3 and any(g in ql for g in greetings):
        return True
    if any(g in ql for g in general):
        return True
    if any(m in ql for m in meta):
        return True

    return False


def rewrite_query_with_context(
    llm, current_query: str, conv_manager, session_id: str
) -> str:
    """
    Ξαναγράφει context-dependent queries σε standalone queries.
    """
    if not conv_manager.needs_rewriting(session_id, current_query):
        logger.debug(f"Query standalone: '{current_query}'")
        return current_query

    history_text = conv_manager.format_history_for_context(session_id)
    if not history_text:
        return current_query

    rewrite_prompt = f"""Δίνεται ιστορικό συνομιλίας και νέα ερώτηση.

ΙΣΤΟΡΙΚΟ:
{history_text}

ΝΕΑ ΕΡΩΤΗΣΗ: {current_query}

Ξαναγράψε την ερώτηση ώστε να είναι αυτοτελής (standalone), διατηρώντας το θέμα από το ιστορικό.

ΠΑΡΑΔΕΙΓΜΑ 1:
Ιστορικό: Τι είναι interface;
Νέα: Δώσε μου παράδειγμα
Standalone: Δώσε μου παράδειγμα interface στην Java

ΠΑΡΑΔΕΙΓΜΑ 2:
Ιστορικό: Τι είναι μία εξαίρεση;
Νέα: Πώς την χρησιμοποιώ στον κώδικα;
Standalone: Πώς χρησιμοποιώ μια εξαίρεση στον κώδικα Java;

Standalone:"""

    try:
        messages = [{"role": "user", "content": rewrite_prompt}]
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=60,
            stop=["\n", "Φοιτητής:", "Καθηγητής:"],
        )
        rewritten = (
            out["choices"][0]["message"]["content"]
            .strip()
            .replace('"', "")
            .replace("'", "")
        )
        if rewritten and len(rewritten) > 5 and rewritten != current_query:
            logger.info(f"Query rewritten: '{current_query}' → '{rewritten}'")
            return rewritten

        logger.debug(f"Rewriting skipped, using original: '{current_query}'")
        return current_query

    except Exception as e:
        logger.error(f"Rewriting failed: {e}")
        return current_query
