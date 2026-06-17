"""Browserless session manager for efficient JS rendering with session persistence."""
import httpx
from typing import Optional, Tuple
import time


class BrowserlessSessionManager:
    """
    Manage Browserless sessions for efficient JS rendering.

    Creates a persistent session once, then reuse it for multiple requests
    to the same domain, saving API units and reducing latency.
    """

    BASE_URL = "https://chrome.browserless.io"

    def __init__(self, api_key: str, ttl: int = 180000):
        """
        Initialize session manager.

        Args:
            api_key: Browserless API key
            ttl: Session time-to-live in milliseconds (default 3 min)
        """
        self.api_key = api_key
        self.ttl = ttl
        self.session_id: Optional[str] = None
        self.session_url: Optional[str] = None
        self.last_activity = 0
        self._lock = False  # Simple lock to prevent concurrent session creation

    def create_session(self) -> bool:
        """Create a new Browserless session. Returns True if successful."""
        if self._lock:
            return False
        self._lock = True

        try:
            response = httpx.post(
                f"{self.BASE_URL}/session?token={self.api_key}",
                json={"ttl": self.ttl},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("data", {}).get("sessionId")
                self.session_url = f"{self.BASE_URL}/content?sessionId={self.session_id}"
                self.last_activity = time.time()
                return True
            return False
        except Exception:
            return False
        finally:
            self._lock = False

    def fetch(self, url: str) -> Tuple[str, int]:
        """
        Fetch URL using the persistent session.

        Returns (content, status_code).
        """
        if not self.session_id:
            # Try to create session
            if not self.create_session():
                return "", 503

        try:
            response = httpx.post(
                self.session_url,
                json={"url": url},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            self.last_activity = time.time()

            if response.status_code == 200:
                return response.text, 200
            elif response.status_code == 503 and "session" in response.text.lower():
                # Session expired, recreate
                self.session_id = None
                self.session_url = None
                if self.create_session():
                    return self.fetch(url)  # Retry once
            return "", response.status_code
        except Exception:
            return "", 503

    def close(self):
        """Close the session."""
        if self.session_id:
            try:
                httpx.get(
                    f"{self.BASE_URL}/close?sessionId={self.session_id}&token={self.api_key}",
                    timeout=10
                )
            except Exception:
                pass
            finally:
                self.session_id = None
                self.session_url = None

    def is_active(self) -> bool:
        """Check if session is still active."""
        if not self.session_id:
            return False
        # Check if session is older than TTL
        elapsed = (time.time() - self.last_activity) * 1000
        return elapsed < self.ttl

    def __enter__(self):
        self.create_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
