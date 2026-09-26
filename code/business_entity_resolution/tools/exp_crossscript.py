"""For cross-script truth pairs, is there any joinable signal at all?

Romanization turns out to be a weak bridge (all schemes ~equal, only ~36% share a
romanized 5-char prefix). This checks whether address fills the gap, which decides
whether a romanized name pass is worth adding or whether fuzzy blocking is required.
"""
import random
import sys
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.normalize import normalize_name


def clean(s):
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


def main():
    gt = pd.read_csv(
        "DATA/student_resource/dataset/train/train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    gt = gt[gt.matched_entity_ids != ""]
    sample = gt.sample(n=4000, random_state=0)
    pairs = []
    for _, row in sample.iterrows():
        pairs.append((row.source1_entity_id, random.choice(row.matched_entity_ids.split(","))))

    need = {p[0] for p in pairs} | {p[1] for p in pairs}
    rec = {}
    for src in (1, 2, 3):
        path = f"DATA/student_resource/dataset/train/train_source{src}.tsv"
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000
        ):
            hit = chunk[chunk.entity_id.isin(need)]
            for eid, nm, ad in zip(hit.entity_id, hit.business_name, hit.business_address):
                rec[eid] = (nm, ad)
    print(f"resolved {len(rec)} records for {len(pairs)} pairs")

    stats = {
        "cross": [0, 0, 0, 0, 0],
        "latin": [0, 0, 0, 0, 0],
    }
    examples = []
    for s1, c in pairs:
        if s1 not in rec or c not in rec:
            continue
        (n1, a1), (n2, a2) = rec[s1], rec[c]
        bucket = "cross" if not n2.isascii() else "latin"
        s = stats[bucket]
        s[0] += 1
        if not a1.strip() or not a2.strip():
            s[1] += 1
            continue
        ca1, ca2 = clean(a1), clean(a2)
        t1, t2 = set(ca1.split()), set(ca2.split())
        if t1 & t2:
            s[2] += 1
        if any(len(t) >= 5 and t[:5] in {u[:5] for u in t2 if len(u) >= 5} for t in t1):
            s[3] += 1
        if fuzz.token_set_ratio(ca1, ca2) >= 80:
            s[4] += 1
        if bucket == "cross" and len(examples) < 12:
            examples.append((n1, n2, a1, a2, fuzz.token_set_ratio(ca1, ca2)))

    labels = ["total", "addr_missing", "shares_addr_token", "shares_addr_prefix5", "addr_token_set>=80"]
    print(f"\n{'':22s}" + "".join(f"{lab:>22s}" for lab in labels))
    for bucket, s in stats.items():
        n = s[0]
        print(f"{bucket:22s}" + "".join(f"{(v / n * 100 if n else 0):21.1f}%" for v in s))

    print("\n--- cross-script examples (is address usable?) ---")
    for n1, n2, a1, a2, sc in examples:
        print(f"S1  {n1!r}\nCAND {n2!r}\n  A1={a1!r}\n  A2={a2!r}  token_set={sc:.0f}\n")


if __name__ == "__main__":
    main()
