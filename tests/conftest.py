"""Shared pytest fixtures: a deterministic mock of Small_LLM_Model."""
import torch


class MockLLM:
    """Deterministic stand-in for Small_LLM_Model.

    Encodes each distinct string as a single, unique token id (assigned in
    first-seen order), and can be told which token id to favour at each
    decoding step -- either an explicit per-call sequence of raw ids, or a
    "sticky" id (via `prefer()`) that stays favoured until changed. This
    keeps candidate-narrowing tests independent of any real tokenizer.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        preferred_token_per_step: list[int] | None = None,
    ) -> None:
        self._vocab_size = vocab_size
        self._ids_by_text: dict[str, int] = {}
        self._next_id = 0
        self._preferred_sequence = list(preferred_token_per_step or [])
        self._sticky_preferred: int | None = None
        self._call = 0
        self.encode_calls: list[str] = []

    def token_id_for(self, text: str) -> int:
        """Return the stable, first-seen-order token id assigned to `text`."""
        if text not in self._ids_by_text:
            self._ids_by_text[text] = self._next_id
            self._next_id += 1
        return self._ids_by_text[text]

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` as a single-token tensor, recording the call."""
        self.encode_calls.append(text)
        return torch.tensor([[self.token_id_for(text)]], dtype=torch.long)

    def decode(self, ids: object) -> str:
        """Unused by the decoders under test; present to satisfy LLMProtocol."""
        return ""

    def get_path_to_vocab_file(self) -> str:
        """Unused by the decoders under test; present to satisfy LLMProtocol."""
        return ""

    def prefer(self, text: str) -> None:
        """Favour `text`'s token id on every future step until changed again."""
        self._sticky_preferred = self.token_id_for(text)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return near-uniform logits, spiking whichever id is favoured."""
        logits = [0.0] * self._vocab_size
        if self._call < len(self._preferred_sequence):
            logits[self._preferred_sequence[self._call]] = 100.0
        elif self._sticky_preferred is not None:
            logits[self._sticky_preferred] = 100.0
        self._call += 1
        return logits
