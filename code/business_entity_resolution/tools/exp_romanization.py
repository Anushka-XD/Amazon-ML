"""Which romanization scheme best aligns Indic candidate names with ASCII S1 names?

Uses real ground-truth pairs: for pairs whose candidate name is non-Latin, compare
fuzzy similarity between the S1 name and the candidate romanized under each scheme.
A scheme that scores high here is one that can actually drive blocking and features.
"""
import random
import sys
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, "code/business_entity_resolution/src")
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

SCHEME_BY_START = [
    (0x0900, "DEVANAGARI"), (0x0980, "BENGALI"), (0x0A00, "GURMUKHI"),
    (0x0A80, "GUJARATI"), (0x0B00, "ORIYA"), (0x0B80, "TAMIL"),
    (0x0C00, "TELUGU"), (0x0C80, "KANNADA"), (0x0D00, "MALAYALAM"),
]
OUT_SCHEMES = {
    "itrans": sanscript.ITRANS,
    "iso": sanscript.ISO,
    "hk": sanscript.HK,
    "optitrans": sanscript.OPTITRANS,
    "slp1": sanscript.SLP1,
    "wx": sanscript.WX,
}


def scheme_for(text):
    for start, name in SCHEME_BY_START:
        for ch in text:
            cp = ord(ch)
            if start <= cp < start + 0x80:
                return getattr(sanscript, name)
    return None


def romanize(text, out_scheme, mode):
    src = scheme_for(text)
    if src is None:
        return text
    try:
        out = transliterate(text, src, out_scheme)
    except Exception:
        return ""
    out = "".join(ch if ord(ch) < 128 else " " for ch in out)
    out = " ".join(out.split())
    if mode == "casefold":
        out = out.casefold()
    elif mode == "casefold_strip":
        out = unicodedata.normalize("NFKD", out.casefold())
        out = "".join(ch for ch in out if not unicodedata.combining(ch))
        out = " ".join(out.split())
    return out


def main():
    gt = pd.read_csv(
        "DATA/student_resource/dataset/train/train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    gt = gt[gt.matched_entity_ids != ""]
    random.seed(0)
    sample = gt.sample(n=4000, random_state=0)
    pairs = []
    for _, row in sample.iterrows():
        mids = row.matched_entity_ids.split(",")
        pairs.append((row.source1_entity_id, random.choice(mids)))
    print(f"sampled {len(pairs)} truth pairs")

    need_s1 = {p[0] for p in pairs}
    need_c = {p[1] for p in pairs}
    names = {}

    def grab(path, wanted, col=1):
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, usecols=[0, 1], chunksize=500000
        ):
            hit = chunk[chunk.iloc[:, 0].isin(wanted)]
            for eid, nm in zip(hit.iloc[:, 0], hit.iloc[:, 1]):
                names[eid] = nm

    grab("DATA/student_resource/dataset/train/train_source1.tsv", need_s1)
    grab("DATA/student_resource/dataset/train/train_source2.tsv", need_c)
    grab("DATA/student_resource/dataset/train/train_source3.tsv", need_c)
    print(f"resolved names for {len(names)} entities")

    cross = []
    for s1, c in pairs:
        if s1 in names and c in names:
            a, b = names[s1], names[c]
            if not b.isascii():
                cross.append((a, b))
    print(f"cross-script truth pairs: {len(cross)}")
    for a, b in cross[:5]:
        print(f"   S1={a!r}  CAND={b!r}")

    print(f"\n{'scheme/mode':28s} {'mean':>7s} {'p50':>7s} {'>=85':>7s} {'>=75':>7s}")
    results = []
    for out_name, out_scheme in OUT_SCHEMES.items():
        for mode in ("raw", "casefold", "casefold_strip"):
            scores = []
            for a, b in cross:
                ra = romanize(a, out_scheme, mode)
                rb = romanize(b, out_scheme, mode)
                scores.append(fuzz.ratio(ra, rb))
            if not scores:
                continue
            s = pd.Series(scores)
            label = f"{out_name}/{mode}"
            results.append((s.mean(), label, s))
            print(f"{label:28s} {s.mean():7.1f} {s.median():7.1f} "
                  f"{(s >= 85).mean() * 100:6.1f}% {(s >= 75).mean() * 100:6.1f}%")

    print("\n--- token-level: does a shared 5-char prefix appear? ---")
    for out_name, out_scheme in OUT_SCHEMES.items():
        for mode in ("raw", "casefold", "casefold_strip"):
            shared_tok = 0
            shared_pre = 0
            tot = 0
            for a, b in cross:
                ta = {t[:5] for t in romanize(a, out_scheme, mode).split() if len(t) >= 5}
                tb = {t[:5] for t in romanize(b, out_scheme, mode).split() if len(t) >= 5}
                tb_full = set(romanize(b, out_scheme, mode).split())
                tot += 1
                if ta & tb_full:
                    shared_tok += 1
                if ta & tb:
                    shared_pre += 1
            if tot:
                print(f"{out_name + '/' + mode:28s} shared_full_token={shared_tok / tot * 100:5.1f}%  "
                      f"shared_prefix5={shared_pre / tot * 100:5.1f}%")


if __name__ == "__main__":
    main()
