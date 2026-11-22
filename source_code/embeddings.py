"""Step 5: sentence embeddings for video text (title + description + tags).

Deduplicates identical clean_text strings across the whole corpus (a video's
text is often unchanged across the months it keeps trending) before encoding,
then maps embeddings back onto every (region, video_id, month) row.
"""
import hashlib

import numpy as np
import pandas as pd
import torch

from . import config, data_prep


def _text_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _encode_texts(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.SENTENCE_MODEL, device=device)
    try:
        return model.encode(
            texts, batch_size=128, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model = SentenceTransformer(config.SENTENCE_MODEL, device="cpu")
        return model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )


def build_and_cache(force: bool = False, regions=None) -> None:
    """Caches CACHE/text_embeddings.npy: text_hash -> 384-d embedding,
    and CACHE/embedding_index_{region}.parquet: (video_id, month) -> text_hash.

    `regions` defaults to config.REGIONS. Passing extra regions (e.g. for
    cross-dataset validation) *extends* the existing embedding cache with
    only the newly-seen texts rather than rebuilding it, so the original
    regions' embeddings/results are untouched."""
    import pickle

    regions = regions if regions is not None else config.REGIONS
    emb_path = config.CACHE / "text_embeddings.npy"
    hash_path = config.CACHE / "text_embedding_hashes.pkl"

    all_rows = []
    for region in regions:
        df = data_prep.build_and_cache(region)[["region", "video_id", "month", "clean_text"]].copy()
        df["text_hash"] = df["clean_text"].map(_text_hash)
        df[["region", "video_id", "month", "text_hash"]].to_parquet(
            config.CACHE / f"embedding_index_{region}.parquet", index=False
        )
        all_rows.append(df[["text_hash", "clean_text"]])
    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates("text_hash")

    if emb_path.exists() and hash_path.exists() and not force:
        existing_vecs = np.load(emb_path)
        with open(hash_path, "rb") as f:
            existing_hashes = pickle.load(f)
        new_rows = combined[~combined["text_hash"].isin(set(existing_hashes))]
        if new_rows.empty:
            return
        new_vecs = _encode_texts(new_rows["clean_text"].tolist())
        vecs = np.concatenate([existing_vecs, new_vecs.astype(np.float32)], axis=0)
        hashes = existing_hashes + new_rows["text_hash"].tolist()
    else:
        vecs = _encode_texts(combined["clean_text"].tolist()).astype(np.float32)
        hashes = combined["text_hash"].tolist()

    np.save(emb_path, vecs.astype(np.float32))
    with open(hash_path, "wb") as f:
        pickle.dump(hashes, f)


def load_embedding_lookup():
    import pickle
    vecs = np.load(config.CACHE / "text_embeddings.npy")
    with open(config.CACHE / "text_embedding_hashes.pkl", "rb") as f:
        hashes = pickle.load(f)
    hash_to_row = {h: i for i, h in enumerate(hashes)}
    return vecs, hash_to_row


def get_region_embeddings(region: str) -> pd.DataFrame:
    """Returns dataframe [region, video_id, month, emb] with emb as np.ndarray."""
    vecs, hash_to_row = load_embedding_lookup()
    idx = pd.read_parquet(config.CACHE / f"embedding_index_{region}.parquet")
    idx["row"] = idx["text_hash"].map(hash_to_row)
    idx["emb"] = idx["row"].map(lambda r: vecs[r])
    return idx[["region", "video_id", "month", "emb"]]


if __name__ == "__main__":
    build_and_cache(force=True)
    vecs, hashes = load_embedding_lookup()
    print("unique texts embedded:", vecs.shape)
