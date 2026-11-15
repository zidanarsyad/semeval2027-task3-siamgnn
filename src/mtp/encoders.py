"""Contextual usage encoders.

``XLLexemeEncoder`` re-implements the inference path of XL-LEXEME
(Cassotti et al., 2023, https://huggingface.co/pierluigic/xl-lexeme): an
XLM-R-large *Siamese* (bi-encoder) network fine-tuned on Word-in-Context pairs.
The target word is wrapped in the special tokens ``<t> ... </t>``, the sequence is
truncated to 128 sub-tokens centred on the target, and the sentence embedding is
the attention-masked mean of the last hidden layer (sentence-transformers
``Pooling(mean)``).  The tokenisation mirrors ``WordTransformer.tokenize_sentence``
from the official repository so that embeddings match the released model.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "pierluigic/xl-lexeme"
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "emb"


def _resolve_local(model_name: str) -> str:
    """Return a local snapshot path if the model is already in the HF cache."""
    if Path(model_name).exists():
        return model_name
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


class XLLexemeEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_seq_length: int = 128,
                 device: str | None = None, num_threads: int | None = None):
        if num_threads:
            torch.set_num_threads(num_threads)
        path = _resolve_local(model_name)
        self.name = model_name.replace("/", "__")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_length = max_seq_length
        self.t_open = self.tokenizer.tokenize("<t>")
        self.t_close = self.tokenizer.tokenize("</t>")

    # -- tokenisation (mirrors WordTransformer) ---------------------------------
    def _tokenize(self, text: str, start: int, end: int) -> list[int]:
        tok = self.tokenizer
        left, target, right = text[:start], text[start:end], text[end:]
        tokens: list[str] = []
        if left:
            tokens += tok.tokenize(left)
        p0 = len(tokens)
        tokens += self.t_open + tok.tokenize(target) + self.t_close
        p1 = len(tokens)
        if right:
            tokens += tok.tokenize(right)
        budget = self.max_seq_length - 2
        if len(tokens) > budget:
            tokens = self._center(tokens, (p0, p1), budget)
        tokens = [tok.cls_token] + tokens + [tok.sep_token]
        return tok.convert_tokens_to_ids(tokens)

    @staticmethod
    def _center(tokens: list[str], pos: tuple[int, int], max_len: int) -> list[str]:
        left, mid, right = tokens[:pos[0]], tokens[pos[0]:pos[1]], tokens[pos[1]:]
        half = int((max_len - len(mid)) / 2)
        over_l, over_r = len(left) - half, len(right) - half
        if over_l > 0 and over_r > 0:
            left, right = left[over_l:], right[:len(right) - over_r]
        elif over_l > 0:
            left = left[over_l:]
        else:
            right = right[:len(right) - over_r]
        out = left + mid + right
        return out[:max_len] if len(out) > max_len else out

    # -- encoding ---------------------------------------------------------------
    @torch.no_grad()
    def encode(self, items: list[tuple[str, int, int]], batch_size: int = 16,
               show_progress: bool = True) -> np.ndarray:
        """Encode ``(text, start, end)`` triples into L2-normalised embeddings."""
        ids = [self._tokenize(t, s, e) for t, s, e in items]
        order = np.argsort([-len(x) for x in ids])
        out = np.zeros((len(ids), self.model.config.hidden_size), dtype=np.float32)
        pad = self.tokenizer.pad_token_id
        rng = range(0, len(ids), batch_size)
        if show_progress:
            from tqdm import tqdm
            rng = tqdm(rng, desc=f"encode[{self.name}]", unit="batch")
        for b in rng:
            idx = order[b:b + batch_size]
            batch = [ids[i] for i in idx]
            L = max(len(x) for x in batch)
            input_ids = torch.tensor([x + [pad] * (L - len(x)) for x in batch], device=self.device)
            mask = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in batch], device=self.device)
            hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            emb = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out[idx] = emb.float().cpu().numpy()
        out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
        return out


def _key(items: list[tuple[str, int, int]]) -> str:
    h = hashlib.sha1()
    for t, s, e in items:
        h.update(f"{s}:{e}:{t}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def encode_cached(encoder: XLLexemeEncoder, items: list[tuple[str, int, int]], tag: str,
                  batch_size: int = 16) -> np.ndarray:
    """Encode with an on-disk cache keyed by model name, tag and content hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(items)
    path = CACHE_DIR / f"{encoder.name}__{tag}__{key}.npy"
    if path.exists():
        return np.load(path)
    # checkpoint every `chunk` items so an interrupted run can resume
    chunk = 500
    parts = []
    part_dir = CACHE_DIR / f"parts__{encoder.name}__{tag}__{key}"
    part_dir.mkdir(exist_ok=True)
    for c, s in enumerate(range(0, len(items), chunk)):
        pp = part_dir / f"{c:05d}.npy"
        if pp.exists():
            parts.append(np.load(pp))
            continue
        e = encoder.encode(items[s:s + chunk], batch_size=batch_size, show_progress=False)
        np.save(pp, e)
        parts.append(e)
        print(f"  [{tag}] {min(s + chunk, len(items))}/{len(items)} encoded", flush=True)
    emb = np.concatenate(parts) if parts else np.zeros((0, 1024), dtype=np.float32)
    np.save(path, emb)
    for pp in part_dir.glob("*.npy"):
        pp.unlink()
    part_dir.rmdir()
    return emb


def load_cached(items: list[tuple[str, int, int]], tag: str,
                model_name: str = DEFAULT_MODEL) -> np.ndarray | None:
    """Read cached embeddings without loading the Transformer (None if not cached)."""
    path = CACHE_DIR / f"{model_name.replace('/', '__')}__{tag}__{_key(items)}.npy"
    return np.load(path) if path.exists() else None


def get_embeddings(items: list[tuple[str, int, int]], tag: str, encoder: "XLLexemeEncoder | None" = None,
                   model_name: str = DEFAULT_MODEL) -> np.ndarray:
    E = load_cached(items, tag, model_name)
    if E is None:
        encoder = encoder or XLLexemeEncoder(model_name)
        E = encode_cached(encoder, items, tag)
    return E


class SentenceEncoder:
    """Sentence-level multilingual Transformer (multilingual-E5): mean pooling, 'query: ' prefix."""

    def __init__(self, path: str | Path = None, max_len: int = 256):
        path = path or (Path(__file__).resolve().parents[2] / "models" / "hf" / "multilingual-e5-base")
        self.name = Path(path).name
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path)).eval()
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        order = np.argsort([-len(t) for t in texts])
        for b in range(0, len(texts), batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok(["query: " + texts[i] for i in idx], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt")
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = ((h * m).sum(1) / m.sum(1)).numpy()
            out.append((idx, e))
        E = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, e in out:
            E[idx] = e
        return E / np.linalg.norm(E, axis=1, keepdims=True)


def sentence_embeddings(texts: list[str], tag: str, encoder: "SentenceEncoder | None" = None) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / f"e5__{tag}__{h}.npy"
    if path.exists():
        return np.load(path)
    encoder = encoder or SentenceEncoder()
    E = encoder.encode(texts)
    np.save(path, E)
    return E


def gloss_item(word: str, definition: str, template: str = "{w}: {d}") -> tuple[str, int, int]:
    """Turn a sense definition into a pseudo-usage so the Siamese encoder can embed it."""
    text = template.format(w=word, d=definition)
    s = text.find(word)
    return text, s, s + len(word)
