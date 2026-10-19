"""Provider-neutral interface between an agent and the LLM that runs it. Messages are plain dicts,
{"role": ..., "content": ...}, content a string or a list of content-block dicts; a provider with another
wire format translates in its own adapter."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

StopReason = Literal["tool_use", "end_turn", "max_tokens"]


@dataclass(frozen=True)
class ToolDef:
    """One tool's schema, in JSON-schema-ish terms every major provider
    understands - each adapter maps `parameters` to its own field name."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ContentBlock:
    """A normalized piece of an assistant turn: either plain text or a tool call."""

    type: Literal["text", "tool_use"]
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Wire representation used when appending this block to message history."""
        if self.type == "text":
            return {"type": "text", "text": self.text}
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass(frozen=True)
class LLMResponse:
    content: list[ContentBlock]
    stop_reason: StopReason


class LLMProvider(Protocol):
    """Anything an agent can run its tool-use loop against."""

    async def generate(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]
    ) -> LLMResponse: ...
