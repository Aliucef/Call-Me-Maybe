from typing import Any, Protocol


class LLMProtocol(Protocol):
    """Structural interface of Small_LLM_Model used by the student decoders.

    A `Protocol` is structural typing: any object that happens to have
    these four methods with these signatures satisfies LLMProtocol, with no
    need to inherit from it. This lets src/decoder/*.py and src/main.py
    type-annotate their `llm` parameter without importing llm_sdk directly
    at the type level, and makes it trivial to swap in a fake/mock LLM for
    testing (see tests/test_function_chooser.py, tests/test_number_decoder.py).

    Crucially, only the four PUBLIC methods below are listed — this mirrors
    the subject constraint that student code may only call these four
    methods on Small_LLM_Model and must never touch private attributes such
    as `llm._tokenizer` or `llm._model`.
    """

    def encode(self, text: str) -> Any: ...
    # Tokenizes a string into model input ids.
    # Example: llm.encode("What is the sum of 2 and 3?") -> tensor([[3838, 374, ...]])
    # Callers immediately do .squeeze(0).tolist() to turn that 2-D tensor
    # into a flat Python list[int], since the rest of the pipeline works
    # with plain lists, not tensors (student code may not import torch).

    def decode(self, ids: Any) -> str: ...
    # Inverse of encode(): turns a list of token ids back into text.
    # Example: llm.decode([9707, 1879]) -> "hello world"
    # Used by src/decoder/string_decoder.py to turn the committed token ids
    # of a decoded string parameter back into readable text.

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...
    # The core primitive of constrained decoding: given the token ids seen
    # so far, run one forward pass through the model and return the raw
    # (un-normalized, no softmax) score for every possible next token.
    # len(result) == vocab size (e.g. ~151k for Qwen3). The decoders in
    # src/decoder/ mask this list (set disallowed entries to -inf) before
    # picking the argmax, which is what "constrains" the generation.

    def get_path_to_vocab_file(self) -> str: ...
    # Returns a local filesystem path to the tokenizer's vocab.json
    # (downloading it from the HF Hub if necessary). src/main.py reads this
    # file directly with json.load() to build a token-string <-> token-id
    # lookup table (id_to_token), which is what lets the decoders inspect
    # what text each token id actually represents, without ever touching
    # the private HuggingFace tokenizer object.
