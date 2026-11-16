"""
Unified LLM client for the adaptive workflow pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    import openai
except ImportError:  # pragma: no cover - environment dependent
    openai = None

from common_types.workflow_models import ReasoningRecord
from config import (
    MAX_TOKENS_PER_WORKFLOW,
    MODEL_NAME,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that some LLM providers add."""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3].strip()
        return s
    return text

logger = logging.getLogger(__name__)


class TokenBudgetExceeded(RuntimeError):
    """Raised when the workflow token budget is exhausted.

    Agents must not swallow this exception: a workflow that exhausts its token
    budget is recorded as a failure by the workflow controller.
    """


class InsufficientCreditsError(BaseException):
    """Raised when the OpenRouter account has insufficient credits (HTTP 402).

    Inherits from BaseException (not Exception) so that agent-level
    ``except Exception`` fallback handlers cannot swallow it.
    """


class TokenBudget:
    def __init__(self, max_total: int):
        self.max_total = max_total
        self.used = 0
        self.exhausted = False

    def reserve(self, tokens: int) -> None:
        if self.used + tokens > self.max_total:
            self.exhausted = True
            raise TokenBudgetExceeded(
                f"Token budget exceeded: requested {tokens}, used {self.used}, limit {self.max_total}"
            )
        self.used += tokens

    def remaining(self) -> int:
        return max(0, self.max_total - self.used)

    def reset(self, max_total: Optional[int] = None) -> None:
        if max_total is not None:
            self.max_total = max_total
        self.used = 0
        self.exhausted = False


class LLMClient:
    """Thin wrapper around OpenRouter/OpenAI responses with reasoning logging."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = None
        if openai is not None:
            self.client = openai.OpenAI(
                base_url=self.config.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                api_key=self.config.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY) or "missing",
                default_headers={"X-Title": "AutoDFT"},
            )
        self.model = self.config.get("MODEL_NAME", MODEL_NAME)
        self.token_budget = TokenBudget(self.config.get("MAX_TOKENS_PER_WORKFLOW", MAX_TOKENS_PER_WORKFLOW))

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        response_format: Optional[str] = "json_object",
        temperature: float = 0.2,
    ) -> Tuple[str, ReasoningRecord]:
        """Return response text plus a structured reasoning record."""
        approx_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        self.token_budget.reserve(approx_tokens)

        if self.client is None:
            raise RuntimeError("openai package is not installed")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": response_format} if response_format else None,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as api_exc:
            status = getattr(api_exc, "status_code", None)
            body = str(api_exc)
            if status == 402 or ("402" in body and "credits" in body.lower()):
                raise InsufficientCreditsError(
                    f"OpenRouter account has insufficient credits. "
                    f"Please top up at https://openrouter.ai/settings/credits\n"
                    f"Original error: {api_exc}"
                ) from api_exc
            raise

        content = _strip_markdown_fences(response.choices[0].message.content or "")
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }
        if usage["total_tokens"]:
            # Reserve completion tokens after the call as well.
            extra = max(0, usage["total_tokens"] - approx_tokens)
            if extra:
                self.token_budget.reserve(extra)

        parsed_output: Dict[str, Any]
        try:
            parsed_output = json.loads(content) if content else {}
        except json.JSONDecodeError:
            parsed_output = {"raw_response": content}

        record = ReasoningRecord(
            stage=stage,
            input_context={"system_prompt": system_prompt, "user_prompt": user_prompt},
            output_decision=parsed_output,
            reasoning_trace=parsed_output.get("reasoning_trace", content),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self.model,
            token_usage=usage,
        )

        return content, record

    def remaining_budget(self) -> int:
        return self.token_budget.remaining()

    def reset_budget(self, max_total: Optional[int] = None) -> None:
        self.token_budget.reset(max_total=max_total)
