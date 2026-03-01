import logging
from typing import List, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConversationManager:
    """Διαχειρίζεται το history των συνομιλιών."""

    def __init__(self, max_turns: int = 3):
        self.conversations = {}
        self.max_turns = max_turns

    def create_session(self, session_id: str = None) -> str:
        if session_id is None:
            session_id = str(uuid4())
        self.conversations[session_id] = []
        logger.info(f"Created session {session_id}")
        return session_id

    def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        chunks_used: Optional[List[str]] = None,
    ):
        """Προσθέτει ένα turn στο history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append(
            {"user": user_msg, "assistant": assistant_msg, "chunks": chunks_used or []}
        )

        if len(self.conversations[session_id]) > self.max_turns:
            self.conversations[session_id] = self.conversations[session_id][
                -self.max_turns :
            ]

        logger.debug(
            f"Session {session_id}: {len(self.conversations[session_id])} turns"
        )

    def get_history(self, session_id: str) -> List[Dict]:
        """Επιστρέφει το history για ένα session."""
        return self.conversations.get(session_id, [])

    def format_history_for_context(self, session_id: str) -> str:
        """Μορφοποιεί το history σε string για το LLM context."""
        history = self.get_history(session_id)

        if not history:
            return ""

        formatted = []
        for turn in history:
            formatted.append(f"Φοιτητής: {turn['user']}")
            formatted.append(f"Καθηγητής: {turn['assistant']}")

        return "\n".join(formatted)

    def needs_rewriting(self, session_id: str, query: str) -> bool:
        history = self.get_history(session_id)
        if not history:
            return False

        query_lower = query.lower()

        strong_indicators = [
            "αυτό",
            "εκείνο",
            "αυτή",
            "εκείνη",
            "αυτοί",
            "ακόμη παράδειγμα",
            "άλλο παράδειγμα",
            "επίσης",
            "και αυτό",
            "το ίδιο",
            "παρόμοιο",
        ]

        if any(ind in query_lower for ind in strong_indicators):
            return True

        weak_indicators = ["παράδειγμα", "μπορείς", "δώσε μου", "πες μου", "και"]
        words = query.split()
        if len(words) < 6 and any(ind in query_lower for ind in weak_indicators):
            return True

        return False

    def clear_session(self, session_id: str):
        """Διαγραφή session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Cleared session {session_id}")
