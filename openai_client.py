import json
import os
import time
from typing import Any

import httpx

BASE = "https://api.openai.com/v1"

class OpenAIError(Exception):
    pass

class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "assistants=v2",
            },
            timeout=30.0,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{BASE}{path}"
        resp = self._client.request(method, url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", 1))
            time.sleep(retry_after)
            resp = self._client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            raise OpenAIError(f"{resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def list_threads(self, limit: int = 100, since: str | None = None) -> list[dict]:
        threads: list[dict] = []
        params: dict[str, Any] = {"limit": limit}
        if since:
            params["created_at[gt]"] = since

        after = None
        while True:
            if after:
                params["after"] = after
            data = self._request("GET", "/threads", params=params)
            batch = data.get("data", [])
            threads.extend(batch)
            if not batch or len(threads) >= limit:
                break
            after = batch[-1].get("id")
            if not after:
                break
        return threads[:limit]

    def list_messages(self, thread_id: str) -> list[dict]:
        messages: list[dict] = []
        after = None
        while True:
            params: dict[str, Any] = {"limit": 100}
            if after:
                params["after"] = after
            data = self._request("GET", f"/threads/{thread_id}/messages", params=params)
            batch = data.get("data", [])
            messages.extend(batch)
            if not batch:
                break
            after = batch[-1].get("id")
            if not after:
                break
        return messages
