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

    def fetch_with_click(self, url: str, button_text: str, timeout: int = 30) -> Tuple[str, int]:
        """
        Fetch URL, click a button by its text, and return the modal/dialog content.

        Args:
            url: Page URL to load
            button_text: Text of the button to click (partial match)
            timeout: Seconds to wait for modal to appear

        Returns (modal_text_content, status_code).
        """
        import time
        print(f"[Browserless] Starting fetch_with_click: {url}, button='{button_text}'", flush=True)
        start = time.time()
        try:
            # Use puppeteer script to click button and extract modal content
            script = f"""
            (async () => {{
                await page.goto('{url}', {{ waitUntil: 'domcontentloaded', timeout: 30000 }});
                await page.waitForTimeout(2000);

                // Click the button containing the text
                const button = Array.from(document.querySelectorAll('button, a, [role="button"]'))
                    .find(el => el.textContent.trim().includes('{button_text}'));
                if (!button) {{
                    return JSON.stringify({{ error: 'Button not found', text: '' }});
                }}
                await button.click();
                await new Promise(r => setTimeout(r, {timeout * 1000}));

                // Try to find modal/dialog content
                let modalText = '';
                const modalSelectors = [
                    '[role="dialog"]', '[role="modal"]', '.modal', '.dialog',
                    '.overlay', '.popup', '.terms-content', '.policy-content',
                    '#myModal', '.modal-content', 'iframe'
                ];
                for (const sel of modalSelectors) {{
                    const el = document.querySelector(sel);
                    if (el) {{
                        modalText = el.innerText || el.textContent || '';
                        if (modalText.length > 50) break;
                    }}
                }}
                // Fallback: get entire body if no modal found
                if (!modalText || modalText.length < 50) {{
                    modalText = document.body.innerText || document.body.textContent || '';
                }}
                return JSON.stringify({{ error: null, text: modalText.trim() }});
            }})()
            """
            response = httpx.post(
                f"{self.BASE_URL}/content?token={self.api_key}",
                json={
                    "url": "about:blank",
                    "chromeOptions": {"args": ["--headless"]},
                    "gotoOptions": {"waitUntil": "domcontentloaded", "timeout": 10000},
                    "puppeteerScript": script,
                    "timeout": 60
                },
                headers={"Content-Type": "application/json"},
                timeout=70
            )
            elapsed = time.time() - start
            print(f"[Browserless] Done fetch_with_click: {url} elapsed={elapsed:.1f}s", flush=True)
            if response.status_code == 200:
                import json
                try:
                    data = json.loads(response.text)
                    return data.get("text", ""), 200
                except Exception:
                    return response.text[:5000], 200
            return "", response.status_code
        except Exception as e:
            elapsed = time.time() - start
            print(f"[Browserless] Error fetch_with_click: {url} elapsed={elapsed:.1f}s error={e}", flush=True)
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
