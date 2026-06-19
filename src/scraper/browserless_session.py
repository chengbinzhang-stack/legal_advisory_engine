"""Browserless session manager for efficient JS rendering with session persistence."""
import httpx
from typing import Optional, Tuple
import time


class BrowserlessSessionManager:
    """
    Manage Browserless sessions for efficient JS rendering.

    Note: Session persistence via WebSocket requires special setup.
    This implementation uses direct API calls which work reliably.
    For production with high volume, consider using Browserless WebSocket API.
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

    def create_session(self) -> bool:
        """Create session. Returns True if API is accessible."""
        try:
            response = httpx.get(f"{self.BASE_URL}/health?token={self.api_key}", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def fetch(self, url: str) -> Tuple[str, int]:
        """
        Fetch URL using Browserless API.

        Returns (content, status_code).

        Uses domcontentloaded + 5s wait instead of networkidle2,
        because networkidle2 hangs on sites with continuous polling (government SPAs, dashboards).
        """
        import time
        print(f"[Browserless] Starting fetch: {url}", flush=True)
        start = time.time()
        try:
            response = httpx.post(
                f"{self.BASE_URL}/content?token={self.api_key}",
                json={
                    "url": url,
                    "gotoOptions": {
                        "waitUntil": "domcontentloaded",
                        "timeout": 30000
                    },
                    # Give JS time to render after DOM is ready
                    "waitForFunction": "document.body.innerHTML.length > 500",
                    "timeout": 30000
                },
                headers={"Content-Type": "application/json"},
                timeout=40
            )
            elapsed = time.time() - start
            print(f"[Browserless] Done fetch: {url} status={response.status_code} elapsed={elapsed:.1f}s", flush=True)
            if response.status_code == 200:
                return response.text, 200
            return "", response.status_code
        except Exception as e:
            elapsed = time.time() - start
            print(f"[Browserless] Error fetch: {url} elapsed={elapsed:.1f}s error={e}", flush=True)
            return "", 503

    def close(self):
        """Close session (no-op for direct API mode)."""
        pass

    def is_active(self) -> bool:
        """Check if session is active."""
        return True

    def __enter__(self):
        self.create_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
