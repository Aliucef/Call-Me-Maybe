# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

# NOTE FOR THE DEFENSE: this whole file is PROVIDED infrastructure, not
# student code. The subject explicitly forbids student code (src/) from
# importing torch/transformers/huggingface_hub directly, or from touching
# any attribute on Small_LLM_Model other than the four public methods
# declared in src/model/llm_protocol.py (encode, decode,
# get_logits_from_input_ids, get_path_to_vocab_file). Everything with a
# leading underscore below (self._model, self._tokenizer, ...) is exactly
# what student code must never reach into.

import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download
import os


logging.set_verbosity_error()  # keep the console clean
# Suppresses HuggingFace's INFO/WARNING chatter (e.g. "some weights were
# not used...") so the pipeline's own prints/verbose output stay readable.


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM for fast, low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically select ``mps`` when available on macOS,
        ``cuda`` when available, otherwise we fall back to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS we default to ``float16`` to keep memory
        usage reasonable; on CPU we keep ``float32`` for maximum compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name  # kept for later use by get_path_to_vocab_file(), etc.

        # Auto-select device with priority: mps > cuda > cpu
        # (mps = Apple Silicon GPU backend; cuda = NVIDIA GPU; cpu = fallback
        # that always works, just slower — what this 42 project runs on.)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            # float16 halves memory/compute on GPU-like backends; CPU keeps
            # float32 since many CPU kernels don't support float16 well.
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        # --- load tokenizer & model -------------------------------------------------
        # Downloads (or reuses a local cache of) the tokenizer files for
        # e.g. "Qwen/Qwen3-0.6B" and builds the HuggingFace tokenizer object.
        # This is the object src/main.py deliberately avoids touching
        # directly — it only reaches vocab.json indirectly through
        # get_path_to_vocab_file().
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Downloads (or reuses a cache of) the model weights and builds the
        # actual neural network (a causal / autoregressive language model).
        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)   # move weights to the chosen device
        self._model.eval()             # disable dropout / training-only behavior

        # switch to inference-only mode
        # Freezes every parameter so no gradient bookkeeping happens during
        # forward passes — this is pure inference, never training, so this
        # saves memory and time on every call to get_logits_from_input_ids.
        for p in self._model.parameters():
            p.requires_grad = False


    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a 2-D ``input_ids`` tensor on the target device."""
        # add_special_tokens=False: don't automatically prepend/append
        # things like BOS/EOS tokens — the pipeline builds its own exact
        # prompt text and wants full control over what gets tokenized.
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        # Wrapped in an outer list -> shape [1, seq_len]: a batch of size 1.
        # This is why every caller in src/ immediately does
        # .squeeze(0).tolist() to drop that batch dimension and get a plain
        # Python list[int].
        return torch.tensor([ids], device=self._device, dtype=torch.long)


    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # skip_special_tokens=True drops things like EOS/PAD from the
        # output text if the model happened to emit them.
        return self._tokenizer.decode(ids, skip_special_tokens=True)


    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """
        Given a list of input token ids, return the raw logits (no softmax) for the next token.
        """
        # Re-wrap the flat list as a [1, seq_len] batch tensor, mirroring
        # what encode() produces, so the model can consume it directly.
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            # torch.no_grad() skips building the autograd graph — pure
            # forward pass, no backpropagation is ever needed here.
            out = self._model(input_ids=input_tensor)
        # Get logits for the last token in the sequence for the batch (batch size 1)
        # out.logits has shape [batch=1, seq_len, vocab_size]; indexing
        # [0, -1] takes batch 0's prediction for the position right after
        # the last input token — i.e. "what comes next" — as a 1-D tensor
        # of length vocab_size.
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]
        # This raw, un-normalized vector (no softmax applied) is exactly
        # what src/decoder/*.py mask and argmax over to implement
        # constrained decoding — the whole project's core technique.


    def get_path_to_vocab_file(self) -> str:
        # vocab_files_names is a HF tokenizer attribute mapping logical
        # file roles ("vocab_file", "merges_file", ...) to their expected
        # on-disk filename for this tokenizer's format; falls back to
        # "vocab.json" if the tokenizer doesn't define one explicitly.
        vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
        # Downloads that specific file from the HF Hub repo (or returns the
        # cached local path if already downloaded) — this is the sanctioned
        # way student code gets at the token<->id mapping, instead of
        # reaching into self._tokenizer's private internals.
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path


    def get_path_to_merges_file(self) -> str:
        # Same idea as get_path_to_vocab_file(), but for the BPE merges
        # file (merges.txt) — not currently used by src/, kept as an
        # available public helper.
        merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path


    def get_path_to_tokenizer_file(self) -> str:
        # Same idea again, for a combined tokenizer.json (used by "fast"
        # tokenizers) — also unused by src/ today, provided for completeness.
        tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path
