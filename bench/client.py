"""OpenAI-compatible client that measures TTFT / end-to-end latency per request.

Decode TPS is defined exactly as in the task statement:

    decode_tps = (output_tokens - 1) / (end_to_end_latency - ttft)

All timings are client side, measured with ``time.perf_counter`` around a
streamed ``/v1/chat/completions`` request:

* ``ttft``  - wall time from sending the request to the first chunk that
              contains a generated token (role-only chunks are ignored).
* ``e2e``   - wall time to the last chunk that carries content / finish_reason
              or the usage summary.

The authoritative output-token count is ``usage.completion_tokens`` from the
final chunk (``stream_options.include_usage = true``). A chunk-count fallback
exists for servers that do not return usage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

import requests


class BenchmarkError(RuntimeError):
    pass


def _iter_sse_lines(resp) -> Iterator[str]:
    """Yield response lines as they arrive.

    ``requests.iter_lines`` wraps ``iter_content``, which fills a 512-byte
    buffer before returning; on a token-at-a-time SSE stream that inflates the
    measured TTFT by hundreds of milliseconds. http.client's ``readline``
    returns as soon as a line is complete, so prefer the underlying file
    object and fall back to the public API if it is unavailable.
    """
    fp = getattr(getattr(resp, "raw", None), "_fp", None)
    if fp is not None and hasattr(fp, "readline"):
        try:
            while True:
                raw = fp.readline()
                if not raw:
                    break
                yield raw.decode("utf-8", errors="replace").rstrip("\r\n")
            return
        except (OSError, ValueError):
            return
    for line in resp.iter_lines(decode_unicode=True):
        yield line or ""


@dataclass
class RequestResult:
    prompt_id: str
    arm: str
    ok: bool
    ttft_s: float | None = None
    e2e_s: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    decode_tps: float | None = None
    wall_tps: float | None = None
    chunks: int = 0
    text: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        connect_timeout: float = 30.0,
        read_timeout: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = (connect_timeout, read_timeout)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # -- infrastructure ---------------------------------------------------

    def health(self, timeout: float = 5.0) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def token_count(self, text: str) -> int | None:
        try:
            resp = self.session.post(
                f"{self.base_url}/tokenize",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if isinstance(payload.get("count"), int):
                return payload["count"]
            if isinstance(payload.get("tokens"), list):
                return len(payload["tokens"])
        except (requests.RequestException, ValueError):
            return None
        return None

    def token_ids(self, text: str) -> list[int] | None:
        """Tokenise text with the served tokenizer (used for quality diffs)."""
        try:
            resp = self.session.post(
                f"{self.base_url}/tokenize",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None
            tokens = (resp.json() or {}).get("tokens")
            if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                return tokens
        except (requests.RequestException, ValueError):
            return None
        return None

    def close(self) -> None:
        self.session.close()

    # -- generation -------------------------------------------------------

    def _body(
        self,
        prompt: str,
        max_tokens: int,
        ignore_eos: bool,
        stream: bool,
        temperature: float = 0.0,
        seed: int = 0,
        extra: dict | None = None,
    ) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "seed": seed,
            "ignore_eos": bool(ignore_eos),
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if extra:
            body.update(extra)
        return body

    def chat(
        self,
        prompt: str,
        max_tokens: int = 192,
        *,
        prompt_id: str = "",
        arm: str = "",
    ) -> RequestResult:
        """Non-streaming request used for quality comparisons."""
        body = self._body(prompt, max_tokens, ignore_eos=False, stream=False)
        started = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout
            )
        except requests.RequestException as exc:
            return RequestResult(
                prompt_id=prompt_id, arm=arm, ok=False, error=f"request failed: {exc}"
            )
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            return RequestResult(
                prompt_id=prompt_id,
                arm=arm,
                ok=False,
                error=f"HTTP {resp.status_code}: {resp.text[:800]}",
            )
        try:
            payload = resp.json()
            choice = payload["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            usage = payload.get("usage") or {}
        except (ValueError, KeyError, IndexError) as exc:
            return RequestResult(
                prompt_id=prompt_id, arm=arm, ok=False, error=f"bad response: {exc}"
            )
        completion = usage.get("completion_tokens")
        return RequestResult(
            prompt_id=prompt_id,
            arm=arm,
            ok=True,
            ttft_s=None,
            e2e_s=elapsed,
            output_tokens=completion,
            prompt_tokens=usage.get("prompt_tokens"),
            decode_tps=None,
            wall_tps=(completion / elapsed) if completion and elapsed > 0 else None,
            text=text,
            meta={"finish_reason": choice.get("finish_reason")},
        )

    def stream_chat(
        self,
        prompt: str,
        max_tokens: int = 256,
        *,
        prompt_id: str = "",
        arm: str = "",
        ignore_eos: bool = True,
    ) -> RequestResult:
        body = self._body(prompt, max_tokens, ignore_eos=ignore_eos, stream=True)
        started = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                stream=True,
                timeout=self.timeout,
                # Keep the socket readable byte-for-byte; compressed SSE would
                # push the first-token timestamp into the decompressor buffer.
                headers={"Accept-Encoding": "identity"},
            )
        except requests.RequestException as exc:
            return RequestResult(
                prompt_id=prompt_id, arm=arm, ok=False, error=f"request failed: {exc}"
            )
        if resp.status_code != 200:
            detail = resp.text[:800]
            resp.close()
            return RequestResult(
                prompt_id=prompt_id,
                arm=arm,
                ok=False,
                error=f"HTTP {resp.status_code}: {detail}",
            )

        ttft: float | None = None
        last_event = started
        chunks = 0
        content_parts: list[str] = []
        usage: dict = {}
        finish_reason = None

        try:
            for raw in _iter_sse_lines(resp):
                if raw is None:
                    continue
                line = raw.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    last_event = time.perf_counter()
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                chunks += 1
                now = time.perf_counter()
                last_event = now
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        if ttft is None:
                            ttft = now
                    reasoning = delta.get("reasoning_content")
                    if reasoning and ttft is None:
                        # Thinking-mode streams may emit reasoning first; it is
                        # still generated output, so the first token counts.
                        ttft = now
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                        last_event = now
        except requests.RequestException as exc:
            resp.close()
            return RequestResult(
                prompt_id=prompt_id, arm=arm, ok=False, error=f"stream failed: {exc}"
            )
        finally:
            resp.close()

        e2e = last_event - started
        completion = usage.get("completion_tokens")
        if completion is None:
            completion = len(content_parts)
        prompt_tokens = usage.get("prompt_tokens")

        text = "".join(content_parts)
        if ttft is None:
            # No token ever arrived; keep the failure visible instead of
            # silently dividing by zero.
            return RequestResult(
                prompt_id=prompt_id,
                arm=arm,
                ok=False,
                ttft_s=None,
                e2e_s=e2e,
                output_tokens=completion,
                prompt_tokens=prompt_tokens,
                chunks=chunks,
                text=text,
                error="no generated token received",
                meta={"finish_reason": finish_reason},
            )

        ttft_rel = (ttft - started) if ttft is not None else None
        decode_span = (e2e - ttft_rel) if ttft_rel is not None else 0.0
        decode_tps = None
        if completion and completion > 1 and decode_span > 0:
            decode_tps = (completion - 1) / decode_span

        return RequestResult(
            prompt_id=prompt_id,
            arm=arm,
            ok=True,
            ttft_s=ttft_rel,
            e2e_s=e2e,
            output_tokens=int(completion) if completion is not None else None,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            decode_tps=decode_tps,
            wall_tps=(completion / e2e) if completion and e2e > 0 else None,
            chunks=chunks,
            text=text,
            meta={"finish_reason": finish_reason},
        )
