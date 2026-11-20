"""
Turns review text into numbers (embeddings) with a multilingual model, so "refund nahi mila" and
"refund not received" land close together. Vectors are cached so each daily run only embeds NEW reviews.
"""
import os

import numpy as np
import pandas as pd

try:                                   # use the OS certificate store (fixes SSL errors behind antivirus/proxies)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from radar.config import DATA, EMBED_MODEL

CACHE = DATA / "embeddings.npz"
MAX_CHARS = 300                        # long reviews are cut; the first 300 characters carry the complaint
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        # Use physical cores, not logical ones: oversubscribing threads made this ~10x slower on a 4-core laptop.
        threads = max(1, (os.cpu_count() or 2) // 2)
        _model = TextEmbedding(EMBED_MODEL, threads=threads)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed in length-sorted batches (less padding = faster), then restore the original order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = np.array(list(_get_model().embed([texts[i] for i in order], batch_size=32)), dtype=np.float32)
    out = np.empty_like(vecs)
    out[order] = vecs
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9          # unit length -> dot product = cosine
    return out


def key(df: pd.DataFrame) -> pd.Series:
    return df["app"] + "|" + df["review_id"].astype(str)


def _save(cache: dict):
    np.savez_compressed(CACHE, keys=np.array(list(cache)), vecs=np.array(list(cache.values()), dtype=np.float16))


def get_embeddings(df: pd.DataFrame, chunk: int = 4000) -> np.ndarray:
    """Return an (n_reviews, dim) matrix aligned with df, embedding only what isn't cached yet."""
    keys = key(df).tolist()
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = dict(zip(z["keys"].tolist(), z["vecs"].astype(np.float32)))
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        print(f"Embedding {len(missing)} new reviews (cached: {len(keys) - len(missing)}) ...", flush=True)
        for start in range(0, len(missing), chunk):                    # chunked so progress is visible and saved
            part = missing[start:start + chunk]
            new = embed_texts([str(df["text"].iloc[i])[:MAX_CHARS] for i in part])
            for i, v in zip(part, new):
                cache[keys[i]] = v
            _save(cache)
            print(f"  embedded {min(start + chunk, len(missing))}/{len(missing)}", flush=True)
    return np.stack([cache[k] for k in keys])
