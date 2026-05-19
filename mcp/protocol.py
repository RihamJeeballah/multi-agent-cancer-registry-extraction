from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    confidence: Optional[float] = None

class Protocol:
    def __init__(self):
        self.messages = []

    def send(self, message: Message):
        self.messages.append(message)

    def get_history(self):
        return self.messages

# Shared protocol instance (global state — can be reset per report)
protocol_instance = Protocol()
