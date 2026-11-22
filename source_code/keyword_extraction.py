"""Step 3a: per-video keyword sets via per-snapshot TF-IDF.

For each (region, month) snapshot we fit a TF-IDF vectorizer on the videos
active that month and keep each video's top-K terms as its keyword set.
This replaces the original work's gensim TextRank call with a scalable,
deterministic scorer while preserving the same intent: reduce each video's
text to a small representative keyword set for building a keyword-overlap
graph.
"""
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, data_prep


def extract_keywords_for_region(region: str) -> dict:
    """Returns {month: {video_id: frozenset(keywords)}}."""
    df = data_prep.build_and_cache(region)
    result = {}
    for month, group in df.groupby("month"):
        texts = group["clean_text"].tolist()
        vids = group["video_id"].tolist()
        vectorizer = TfidfVectorizer(
            max_df=0.6,
            min_df=config.MIN_DF_KEYWORDS,
            max_features=config.MAX_KEYWORD_VOCAB,
        )
        try:
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:
            # fewer than min_df distinct docs/terms: fall back to raw tokens
            result[month] = {
                vid: frozenset(t.split(" ")[: config.TOP_K_KEYWORDS])
                for vid, t in zip(vids, texts)
            }
            continue
        vocab = vectorizer.get_feature_names_out()
        month_kw = {}
        tfidf = tfidf.tocsr()
        for row_idx, vid in enumerate(vids):
            row = tfidf.getrow(row_idx)
            if row.nnz == 0:
                month_kw[vid] = frozenset()
                continue
            top_idx = row.data.argsort()[::-1][: config.TOP_K_KEYWORDS]
            terms = row.indices[top_idx]
            month_kw[vid] = frozenset(vocab[terms])
        result[month] = month_kw
    return result


def build_and_cache(region: str, force: bool = False) -> dict:
    out_path = config.CACHE / f"keywords_{region}.pkl"
    if out_path.exists() and not force:
        with open(out_path, "rb") as f:
            return pickle.load(f)
    kw = extract_keywords_for_region(region)
    with open(out_path, "wb") as f:
        pickle.dump(kw, f)
    return kw


if __name__ == "__main__":
    for r in config.REGIONS:
        kw = build_and_cache(r, force=True)
        n_months = len(kw)
        avg_kw = sum(len(s) for m in kw.values() for s in m.values()) / sum(len(m) for m in kw.values())
        print(r, n_months, "months, avg keywords/video =", round(avg_kw, 2))
