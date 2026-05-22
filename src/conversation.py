import logging
from typing import List, Dict, Optional
from uuid import uuid4
from langchain_core.language_models.llms import LLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory

logger = logging.getLogger(__name__)


class LlamaLangChainWrapper(LLM):
    """
    Thin wrapper so LangChain's summarizer can call your llama_cpp instance.
    Must be instantiated after the llm is loaded in lifespan.
    """

    llm: object

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "llama_cpp"

    def _call(self, prompt: str, stop=None) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            stop=stop or [],
        )
        return out["choices"][0]["message"]["content"].strip()


class ConversationManager:
    """
    Διαχειρίζεται το history με LangChain SummaryBufferMemory.
    Παλιά turns συμπιέζονται σε summary — άπειρες συνομιλίες.
    """

    def __init__(self, max_token_limit: int = 1000):
        """
        max_token_limit: token budget για raw history πριν την summarization.
        Όταν ξεπεραστεί, τα παλιότερα turns γίνονται summary αυτόματα.
        """
        self.sessions: Dict[str, ConversationSummaryBufferMemory] = {}
        self.max_token_limit = max_token_limit
        self._llm_wrapper: Optional[LlamaLangChainWrapper] = None

    def set_llm(self, llm):
        """Call this once after llama_cpp is loaded (in lifespan)."""
        self._llm_wrapper = LlamaLangChainWrapper(llm=llm)

    def _make_memory(self) -> ConversationSummaryBufferMemory:
        return ConversationSummaryBufferMemory(
            llm=self._llm_wrapper,
            max_token_limit=self.max_token_limit,
            memory_key="history",
            return_messages=True,
        )

    def create_session(self, session_id: str = None) -> str:
        if session_id is None:
            session_id = str(uuid4())
        self.sessions[session_id] = self._make_memory()
        logger.info(f"Created session {session_id}")
        return session_id

    def _get_or_create(self, session_id: str) -> ConversationSummaryBufferMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = self._make_memory()
        return self.sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        chunks_used: Optional[List[str]] = None,  # kept for API compatibility
    ):
        mem = self._get_or_create(session_id)
        mem.save_context(
            {"input": user_msg},
            {"output": assistant_msg},
        )
        logger.debug(f"Session {session_id}: turn saved")

    def get_history(self, session_id: str) -> List[Dict]:
        """Returns raw messages as list of dicts (for needs_rewriting compatibility)."""
        mem = self._get_or_create(session_id)
        history = []
        for msg in mem.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({"user": msg.content, "assistant": ""})
            elif isinstance(msg, AIMessage) and history:
                history[-1]["assistant"] = msg.content
        return history

    def format_history_for_context(self, session_id: str) -> str:
        """
        Returns summary (if any) + recent raw turns as a formatted string.
        This is what gets injected into your RAG prompt.
        """
        mem = self._get_or_create(session_id)

        parts = []

        # Prepend the running summary of older turns if it exists
        summary = mem.moving_summary_buffer
        if summary:
            parts.append(f"[Περίληψη προηγούμενης συζήτησης]: {summary}")

        # Append the recent raw turns still in the buffer
        for msg in mem.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"Φοιτητής: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Καθηγητής: {msg.content}")

        return "\n".join(parts)

    def needs_rewriting(self, session_id: str, query: str) -> bool:
        """Unchanged logic — checks if query needs context to be standalone."""
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
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")
