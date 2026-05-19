from mcp.protocol import Message, protocol_instance

def log_message(sender: str, receiver: str, content: str, confidence: float = None):
    msg = Message(sender, receiver, content, confidence)
    protocol_instance.send(msg)
