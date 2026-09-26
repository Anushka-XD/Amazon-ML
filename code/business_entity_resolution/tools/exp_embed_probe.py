"""Size the embedding job on real records before committing 2-4 CPU-hours.

Downloads the multilingual encoder, embeds a sample of real S1/S2/S3 name+address
pairs, and reports throughput plus the property that matters: does the encoder
actually place a Devanagari name near its ASCII counterpart, which is the whole
reason for using it over romanization (measured: only 9.9% of cross-script pairs
share a romanized token, mean fuzz ~67).

Usage:
    python tools/exp_embed_probe.py --n 4000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "code/business_entity_resolution/src")

MODEL = "intfloat/multilingual-e5-small"  # 384-dim, 118M params, MIT


def log(m):
    print(f"[embed] {m}", flush=True)


def build_text(name: str, addr: str, fields: str = "both") -> str:
    # e5 expects a "query: "/"passage: " prefix for asymmetric retrieval; for
    # symmetric entity matching a single shared prefix on both sides is simplest and
    # keeps S1 and S2/S3 in the same space.
    name = (name or "").strip()
    addr = (addr or "").strip()
    if fields == "name":
        return f"passage: {name}"
    if name and addr:
        return f"passage: {name}; {addr}"
    return f"passage: {name or addr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=64)
    ap.add_argument("--fields", default="both", choices=["both", "name"])
    args = ap.parse_args()

    data = Path("DATA/student_resource/dataset/train")
    log(f"loading {args.n} real records")
    frames = []
    for name in ("train_source2.tsv", "train_source3.tsv"):
        import pandas as pd

        df = pd.read_csv(data / name, sep="\t", dtype=str, keep_default_na=False,
                         nrows=args.n)
        frames.append(df)
    cand = __import__("pandas").concat(frames, ignore_index=True)
    cand = cand[cand.business_name.str.strip() != ""]
    log(f"{len(cand):,} candidate records")

    nonascii = cand[~cand.business_name.str.isascii()]
    log(f"{len(nonascii):,} non-ASCII names in sample")

    log(f"loading {MODEL}")
    t0 = time.time()
    torch.set_num_threads(args.threads)
    model = SentenceTransformer(MODEL, device="cpu")
    model.eval()
    # Business name + address is ~40 tokens; the 512 default wastes compute on padding
    # and lets a pathological row dominate a batch.
    model.max_seq_length = args.max_seq
    log(f"model loaded in {time.time() - t0:.1f}s (threads={args.threads}, "
        f"max_seq={args.max_seq})")

    texts = [build_text(n, a) for n, a in
             zip(cand.business_name.head(2000), cand.business_address.head(2000))]
    log(f"embedding {len(texts)} texts on {torch.get_num_threads()} threads...")
    t0 = time.time()
    with torch.inference_mode():
        emb = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    dt = time.time() - t0
    rate = len(texts) / dt
    log(f"{rate:.0f} texts/s  -> 12.5M texts would take "
        f"{12_500_000 / rate / 3600:.1f} h on CPU")
    log(f"vector store: {12.5e6 * emb.shape[1] * 2 / 1e9:.1f} GB fp16")

    # The decisive property: cross-script pairs from the ground truth.
    gt = __import__("pandas").read_csv(
        "DATA/student_resource/dataset/train/train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False, nrows=200000)
    gt = gt[gt.matched_entity_ids != ""]
    s1 = __import__("pandas").read_csv(
        "DATA/student_resource/dataset/train/train_source1.tsv",
        sep="\t", dtype=str, keep_default_na=False)
    s1map = dict(zip(s1.entity_id, zip(s1.business_name, s1.business_address)))
    candmap = {}
    for src in ("train_source2.tsv", "train_source3.tsv"):
        df = __import__("pandas").read_csv(
            Path("DATA/student_resource/dataset/train") / src, sep="\t", dtype=str,
            keep_default_na=False, nrows=1_500_000)
        candmap.update(dict(zip(df.entity_id, zip(df.business_name, df.business_address))))

    pairs = []
    for _, row in gt.head(4000).iterrows():
        c = row.matched_entity_ids.split(",")[0]
        if row.source1_entity_id in s1map and c in candmap:
            pairs.append((s1map[row.source1_entity_id], candmap[c]))
    cross = [(a, b) for a, b in pairs if not b[0].isascii()]
    log(f"truth pairs resolved: {len(pairs)}, cross-script: {len(cross)}")
    if cross:
        sample = cross[:min(600, len(cross))]
        with torch.inference_mode():
            e = model.encode([build_text(*a, fields=args.fields) for a, _ in sample] +
                             [build_text(*b, fields=args.fields) for _, b in sample],
                             batch_size=args.batch, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        n = len(sample)
        va, vb = e[:n], e[n:]
        sims = (va * vb).sum(axis=1)
        log(f"cross-script cosine: mean={sims.mean():.3f} p10={np.percentile(sims, 10):.3f} "
            f"min={sims.min():.3f}")
        log("examples:")
        order = np.argsort(-sims)[:5]
        for i in order:
            log(f"  cos={sims[i]:.3f}  S1={sample[i][0][0][:42]!r}  CAND={sample[i][1][0][:42]!r}")
        # Baseline: how do UNRELATED records score? This is the separation test - if
        # true cross-script pairs and random pairs overlap, retrieval adds nothing.
        m = min(200, len(sample) // 2)
        with torch.inference_mode():
            e2 = model.encode([build_text(*a, fields=args.fields) for a, _ in sample[:m]] +
                              [build_text(*b, fields=args.fields) for _, b in sample[m:2 * m]],
                              batch_size=args.batch, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
        base = (e2[:m] * e2[m:]).sum(axis=1)
        log(f"unrelated-pair cosine: mean={base.mean():.3f} "
            f"p90={np.percentile(base, 90):.3f} max={base.max():.3f}")
        log(f"SEPARATION: true p10={np.percentile(sims, 10):.3f} vs "
            f"unrelated p90={np.percentile(base, 90):.3f}  "
            f"(gap={np.percentile(sims, 10) - np.percentile(base, 90):+.3f})")
        for thr in (0.80, 0.85, 0.90):
            log(f"  cosine >= {thr}: true={float((sims >= thr).mean()) * 100:5.1f}%  "
                f"unrelated={float((base >= thr).mean()) * 100:5.1f}%")


if __name__ == "__main__":
    main()
