"""
Sliding-window rate limiter for the gemini-embedding-2 API.

Quotas (per the user-provided spec for this account):
  - 100 requests per minute
  - 30,000 tokens per minute
  - 1,000 requests per day

This module enforces all three. Callers pass an *estimate* of token cost
before the API call (so we can pre-throttle) and the *actual* token cost
after the call (so the rolling counters are precise).

Both image and text embedders share this limiter so a single instance
can be passed around if needed; in practice each script creates its own.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple


@dataclass
class GeminiRateLimiter:
    """Sliding-window limiter for gemini-embedding-2.

    Window is a fixed 60-second sliding range for the per-minute caps.
    Daily counter resets on a 24-hour rolling basis from process start.
    """

    rpm: int = 100
    tpm: int = 30_000
    rpd: int = 1_000

    # Internal state — a deque of (timestamp, tokens) for the last 60s.
    _events: Deque[Tuple[float, int]] = field(default_factory=deque)
    _day_start: float = field(default_factory=time.time)
    _day_requests: int = 0

    # Small extra delay after every call so we never *quite* hit the
    # caps in a burst. 0.05s padding adds up to ~3 extra seconds across
    # 60 calls — negligible vs. the API itself.
    _padding: float = 0.05

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """Block until issuing one request of `estimated_tokens` tokens
        would not breach the per-minute caps.

        Also raises RuntimeError if the daily request cap would be
        exceeded by this call — better to halt than to silently fail.
        """
        if self._day_requests >= self.rpd:
            # Reset if a full 24h has passed since process start.
            if time.time() - self._day_start >= 86_400:
                self._day_start = time.time()
                self._day_requests = 0
            else:
                raise RuntimeError(
                    f"Daily request cap reached ({self.rpd}). "
                    f"Wait or restart tomorrow."
                )

        while True:
            now = time.time()
            self._evict_expired(now)

            requests_in_window = len(self._events)
            tokens_in_window = sum(t for _, t in self._events)

            requests_ok = requests_in_window + 1 <= self.rpm
            tokens_ok = tokens_in_window + max(0, estimated_tokens) <= self.tpm

            if requests_ok and tokens_ok:
                return

            # Sleep until the *oldest* event in the window expires, then
            # re-check (because a single eviction may not be enough if
            # the next-oldest event is also large).
            oldest_ts = self._events[0][0]
            wait = max(0.05, (oldest_ts + 60.0) - now + self._padding)
            time.sleep(wait)

    def record(self, actual_tokens: int) -> None:
        """Record the actual cost of a call that just succeeded."""
        now = time.time()
        self._events.append((now, max(0, actual_tokens)))
        self._day_requests += 1
        # Optional small post-call pad so we don't fire exactly back-to-
        # back. Cheap insurance against clock skew on the API side.
        if self._padding:
            time.sleep(self._padding)

    def stats(self) -> dict:
        """Snapshot of current usage for logging."""
        now = time.time()
        self._evict_expired(now)
        return {
            "requests_last_60s": len(self._events),
            "tokens_last_60s": sum(t for _, t in self._events),
            "requests_today": self._day_requests,
            "rpm_cap": self.rpm,
            "tpm_cap": self.tpm,
            "rpd_cap": self.rpd,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_expired(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()


# ----------------------------------------------------------------------
# Token estimators (cheap, no external deps)
# ----------------------------------------------------------------------

def estimate_text_tokens(text: str) -> int:
    """Conservative token estimate for English-ish text.

    gemini-embedding-2 caps input at 2048 tokens; we estimate at
    `chars / 3.5` which slightly over-counts vs. the typical chars/4
    rule. Over-counting is the safe direction for a rate limiter.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.5) + 1)


def estimate_image_tokens(_payload_bytes: int = 0) -> int:
    """Per Google docs, embedding a single small image is roughly 258
    tokens. Use 300 to leave headroom — 50 covers at 300 = 15k tokens,
    still inside the 30k/min budget but spread over multiple seconds.
    """
    return 300
