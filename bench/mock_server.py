"""A tiny OpenAI-compatible mock server used by ``--self-test``.

It exists so the whole pipeline (server handle -> streaming client -> metrics ->
report) can be validated on a machine without a GPU. Latency is simulated from a
per-arm profile; the resulting numbers are *not* real measurements and the
report is marked ``SIMULATED``.
"""

from __future__ import annotations

import json
import threading
import time
import zlib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .server import ServerSnapshot

PROFILES: dict[str, dict] = {
    "baseline": {"ms_per_token": 22.0, "ttft_ms": 280, "accept": 0.0, "k": 0},
    "flashinfer": {"ms_per_token": 21.0, "ttft_ms": 260, "accept": 0.0, "k": 0},
    "flashinfer_fp8kv": {"ms_per_token": 20.6, "ttft_ms": 255, "accept": 0.0, "k": 0},
    "mtp3": {"ms_per_token": 12.2, "ttft_ms": 270, "accept": 0.72, "k": 3},
    "mtp3_fp8kv": {"ms_per_token": 12.0, "ttft_ms": 265, "accept": 0.72, "k": 3},
    "mtp5": {"ms_per_token": 10.6, "ttft_ms": 275, "accept": 0.55, "k": 5},
    "mtp7": {"ms_per_token": 9.9, "ttft_ms": 280, "accept": 0.42, "k": 7},
    "single_seq": {"ms_per_token": 12.0, "ttft_ms": 270, "accept": 0.72, "k": 3},
    "ngram": {"ms_per_token": 20.0, "ttft_ms": 260, "accept": 0.25, "k": 4},
    "eager": {"ms_per_token": 26.5, "ttft_ms": 300, "accept": 0.0, "k": 0},
}

_WORDS = (
    "the model routes tokens through a sparse mixture of experts while the "
    "decoder attends over a constant size recurrent state and verifies every "
    "speculative token against the target distribution"
).split()


def _word(index: int) -> str:
    return _WORDS[index % len(_WORDS)]


def _text_for(prompt: str, n_words: int) -> str:
    seed = zlib.crc32(prompt.encode("utf-8"))
    return " ".join(_word(seed + i) for i in range(n_words))


def _token_ids(text: str) -> list[int]:
    return [zlib.crc32(word.encode("utf-8")) % 90000 + 100 for word in text.split()]


class _State:
    def __init__(self, profile: dict) -> None:
        self.profile = profile
        self.lock = threading.Lock()
        self.drafts = 0.0
        self.draft_tokens = 0.0
        self.accepted = 0.0

    def observe(self, n_tokens: int) -> None:
        k = self.profile["k"]
        accept = self.profile["accept"]
        if not k or not accept:
            return
        # Roughly one draft step per emitted token group.
        steps = max(1.0, n_tokens / (1 + k * accept))
        with self.lock:
            self.drafts += steps
            self.draft_tokens += steps * k
            self.accepted += steps * k * accept


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # silence request logs
            return

        # -- helpers ---------------------------------------------------
        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _chunk(self, payload: dict) -> None:
            data = f"data: {json.dumps(payload)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()

        # -- routes ----------------------------------------------------
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/health"):
                self._json({"status": "ok"})
            elif self.path.startswith("/v1/models"):
                self._json(
                    {
                        "object": "list",
                        "data": [{"id": "mock-wemm", "object": "model"}],
                    }
                )
            elif self.path.startswith("/metrics"):
                with state.lock:
                    drafts = state.drafts
                    drafted = state.draft_tokens
                    accepted = state.accepted
                lines = [
                    "# HELP vllm:spec_decode_num_drafts_total drafts",
                    "# TYPE vllm:spec_decode_num_drafts_total counter",
                    f'vllm:spec_decode_num_drafts_total{{model_name="mock"}} {drafts}',
                    f'vllm:spec_decode_num_draft_tokens_total{{model_name="mock"}} {drafted}',
                    f'vllm:spec_decode_num_accepted_tokens_total{{model_name="mock"}} {accepted}',
                ]
                body = ("\n".join(lines) + "\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json({"error": "bad json"}, status=400)
                return
            if self.path.startswith("/tokenize"):
                text = body.get("prompt") or ""
                ids = _token_ids(text)
                self._json({"count": len(ids), "tokens": ids})
                return
            if not self.path.startswith("/v1/chat/completions"):
                self._json({"error": "not found"}, status=404)
                return
            self._chat(body)

        # -- chat ------------------------------------------------------
        def _chat(self, body: dict) -> None:
            messages = body.get("messages") or []
            prompt = ""
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    prompt += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt += item.get("text") or ""
            prompt_tokens = len(_token_ids(prompt))
            max_tokens = int(body.get("max_tokens") or 64)
            stream = bool(body.get("stream"))
            profile = state.profile

            if stream:
                delay = profile["ms_per_token"] / 1000.0
                ttft = profile["ttft_ms"] / 1000.0 + prompt_tokens * 0.00002
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                time.sleep(ttft)
                self._chunk(
                    {
                        "id": "mock",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                self._chunk(
                    {
                        "id": "mock",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": _word(0) + " "}, "finish_reason": None}],
                    }
                )
                state.observe(max_tokens)
                remaining = max_tokens - 1
                sent = 1
                batch = 16
                while remaining > 0:
                    size = min(batch, remaining)
                    time.sleep(delay * size)
                    text = " ".join(_word(sent + i) for i in range(size)) + " "
                    self._chunk(
                        {
                            "id": "mock",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                    )
                    sent += size
                    remaining -= size
                self._chunk(
                    {
                        "id": "mock",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    }
                )
                self._chunk(
                    {
                        "id": "mock",
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": max_tokens,
                            "total_tokens": prompt_tokens + max_tokens,
                        },
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            # Non-streaming (quality checks): fast, deterministic.
            time.sleep(0.02)
            text = _text_for(prompt, min(max_tokens, 160))
            self._json(
                {
                    "id": "mock",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(text.split()),
                        "total_tokens": prompt_tokens + len(text.split()),
                    },
                }
            )

    return Handler


@dataclass
class MockSnapshot(ServerSnapshot):
    simulated: bool = True

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["simulated"] = True
        return data


class MockHandle:
    """Same duck-typed surface as :class:`bench.server.ServerHandle`."""

    def __init__(self, arm_name: str, log_path: Path, port: int = 0) -> None:
        self.arm_name = arm_name
        self.profile = dict(PROFILES.get(arm_name, PROFILES["baseline"]))
        self.log_path = Path(log_path)
        self.host = "127.0.0.1"
        self._state = _State(self.profile)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._start_metrics: dict = {}
        self.port = 0
        self.base_url = ""
        self.snapshot = self._snapshot()
        self._log_fh = None

    def _snapshot(self) -> MockSnapshot:
        has_spec = bool(self.profile["k"])
        return MockSnapshot(
            attention_backend="FLASHINFER",
            cuda_graphs=self.arm_name != "eager",
            speculative={
                "method": "mtp" if self.arm_name.startswith(("mtp", "single_seq")) else "ngram",
                "num_speculative_tokens": self.profile["k"],
            }
            if has_spec
            else None,
            tensor_parallel=1,
            kv_cache_dtype="fp8" if "fp8" in self.arm_name else "auto",
            max_model_len=8192,
            gpu_blocks=4321,
            version="mock",
        )

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        self._server = ThreadingHTTPServer((self.host, 0), _make_handler(self._state))
        self.port = self._server.server_address[1]
        self.base_url = f"http://{self.host}:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("w", encoding="utf-8")
        self._log_fh.write(
            "INFO mock server: Using FLASHINFER backend. Capturing CUDA graphs (PIECEWISE).\n"
            f"INFO mock server: profile={self.profile}\n"
        )
        self._log_fh.flush()

    def wait_ready(self) -> None:
        for _ in range(100):
            try:
                import requests

                if requests.get(f"{self.base_url}/health", timeout=1).status_code == 200:
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)
        raise RuntimeError("mock server did not become healthy")

    def is_alive(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def stop(self, grace: float = 1.0) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # -- metrics -------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        with self._state.lock:
            return {
                "vllm:spec_decode_num_drafts_total": self._state.drafts,
                "vllm:spec_decode_num_draft_tokens_total": self._state.draft_tokens,
                "vllm:spec_decode_num_accepted_tokens_total": self._state.accepted,
            }

    def set_metrics_baseline(self, snapshot: dict) -> None:
        self._start_metrics = dict(snapshot)

    def spec_acceptance(self) -> dict | None:
        end = self.metrics()
        if not self._start_metrics:
            return None
        drafts = end["vllm:spec_decode_num_drafts_total"] - self._start_metrics.get(
            "vllm:spec_decode_num_drafts_total", 0.0
        )
        drafted = end["vllm:spec_decode_num_draft_tokens_total"] - self._start_metrics.get(
            "vllm:spec_decode_num_draft_tokens_total", 0.0
        )
        accepted = end["vllm:spec_decode_num_accepted_tokens_total"] - self._start_metrics.get(
            "vllm:spec_decode_num_accepted_tokens_total", 0.0
        )
        if drafts <= 0 or drafted <= 0:
            return None
        return {
            "drafts": drafts,
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / drafted,
            "mean_acceptance_length": 1.0 + accepted / drafts,
        }

    def log_tail(self, lines: int = 60) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
