"""Minimal client for any OpenAI-compatible /chat/completions endpoint."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

LLM = Callable[[list[dict]], str]


def make_llm(base_url: str, key_file: str | Path, model: str, temperature: float = 1.0,
             max_tokens: int = 16000, retries: int = 3, reasoning_effort: str | None = None) -> LLM:
    url = base_url.rstrip("/") + "/chat/completions"
    key = Path(key_file).read_text().split()[0]

    def post(payload: dict) -> dict:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def call(messages: list[dict]) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if reasoning_effort:
            # Some models reason until the request times out unless told how hard to think.
            payload["reasoning"] = {"effort": reasoning_effort}
        for attempt in range(retries + 3):       # room for the output-allowance doublings below
            try:
                body = post(payload)
                choice = body["choices"][0]
                usage = body.get("usage") or {}
                print(f"model call: {usage.get('prompt_tokens', '?')} tokens in, {usage.get('completion_tokens', '?')} out"
                      + (f", cost {usage['cost']:.4f}" if isinstance(usage.get("cost"), (int, float)) else ""), flush=True)
                content = choice["message"]["content"] or ""
                limit_key = "max_tokens" if "max_tokens" in payload else "max_completion_tokens"
                if not content.strip() and choice.get("finish_reason") == "length" and payload[limit_key] < 128000:
                    # A reasoning model spent the whole allowance thinking and wrote nothing: give it more.
                    payload = {**payload, limit_key: min(128000, payload[limit_key] * 2)}
                    continue
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                if exc.code == 400 and "max_tokens" in payload:
                    # Reasoning models reject max_tokens / temperature; retry in their dialect.
                    payload = {k: v for k, v in payload.items() if k not in ("max_tokens", "temperature")}
                    payload["max_completion_tokens"] = max_tokens
                    continue
                if exc.code in (401, 403, 404) or attempt >= retries - 1:
                    raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
                if attempt >= retries - 1:
                    raise RuntimeError(f"LLM request failed: {exc!r}") from exc
            time.sleep(10 * (attempt + 1))
        raise RuntimeError("LLM request failed")

    return call
