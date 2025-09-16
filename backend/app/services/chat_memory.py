from typing import List, Dict

class ChatMemory:
    def __init__(self):
        self.store: Dict[str, List[Dict[str, str]]] = {}  # {session_id: [{"role":..,"content":..}, ...]}

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return self.store.get(session_id, [])

    def append(self, session_id: str, role: str, content: str, max_turns: int = 12):
        hist = self.store.get(session_id, [])
        hist.append({"role": role, "content": content})
        # mantém só os últimos N itens
        if len(hist) > 2 * max_turns:
            hist = hist[-2 * max_turns :]
        self.store[session_id] = hist

    def reset(self, session_id: str):
        self.store.pop(session_id, None)

chat_memory = ChatMemory()
