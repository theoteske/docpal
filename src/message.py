class Message:
    """Base class for all message types in the chat system.

    Attributes:
        content: The text content of the message.
    """
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the human user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass