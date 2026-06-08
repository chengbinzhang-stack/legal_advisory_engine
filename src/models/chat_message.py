"""Data models for chat functionality."""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    website_context: Optional[str] = None

@dataclass
class ChatSession:
    """Represents a chat session for a specific website."""
    session_id: str
    website_url: str
    website_domain: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
