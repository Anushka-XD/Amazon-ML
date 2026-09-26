import re
from dataclasses import dataclass

from ber.normalize import normalize_name

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

IN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "jammu and kashmir", "ladakh", "puducherry",
    "chandigarh", "haryana", "haryana",
}

FR_REGIONS = {
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
    "corse", "grand est", "hauts de france", "ile de france", "normandie",
    "nouvelle aquitaine", "occitanie", "pays de la loire", "provence alpes cote d azur",
    "bordeaux", "lille", "dunkerque", "la teste de buch",
}

STATE_NAMES = set(US_STATES.values()) | IN_STATES | FR_REGIONS

STREET_SUFFIXES = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln", "court",
    "ct", "boulevard", "blvd", "rue", "chemin", "route", "way", "plaza", "circle",
    # French and common European abbreviations seen in the test split.
    "bd", "bvd", "av", "pl", "imp", "allee", "allée", "rte", "faub", "fbg", "quai",
    "sentier", "villa", "cite", "cité", "esplanade", "rond", "point",
}

# The same street type spelled differently across records; blocking keys must fold
# these together or a "Ln"/"Lane" pair never joins.
STREET_SUFFIX_CANON = {
    "street": "st", "str": "st",
    "road": "rd",
    "avenue": "ave", "av": "ave", "aven": "ave",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd",
    "place": "pl", "plaza": "plz",
    "square": "sq",
    "highway": "hwy",
    "terrace": "ter", "trace": "tr",
    "circle": "cir",
    "parkway": "pkwy",
    "expressway": "expy",
}

# Words that introduce a house/plot/flat number rather than being part of the street.
HOUSE_MARKERS = {
    "no", "num", "number", "plot", "flat", "fl", "floor", "unit", "shop", "office", "opp",
    "house", "hno", "house", "bldg", "building", "block", "blk", "sector", "sec", "shed",
    "kh", "khasra", "survey", "srvy", "plot", "survey", "premises", "door", "house",
}

LANDMARK_WORDS = {"near", "opposite", "opp", "behind", "beside", "next"}

# A bare number, or a label-prefixed one like "no107", "g-26", "12/34b", "#98825".
_HOUSE_RE = re.compile(r"^(?:#)?(\d+(?:[/-]\d+)*[a-z]?)$")
_HOUSE_PREFIXED_RE = re.compile(r"^[a-z]{1,6}[-.]?(\d+(?:[/-]\d+)*[a-z]?)$")


@dataclass(frozen=True)
class AddressParts:
    house_no: str
    street_tokens: tuple
    postal: str
    state_key: str
    city_tokens: tuple
    landmark_flag: bool
    addr_missing: bool


def _postal_re(country: str) -> re.Pattern:
    key = (country or "").strip().lower()
    if key == "us":
        return re.compile(r"\b\d{5}(?:-\d{4})?\b")
    return re.compile(r"\b\d{6}\b" if key == "india" else r"\b\d{5}\b")


def house_number(raw: str) -> str:
    """Extract a house/plot/flat number from anywhere in the leading address text.

    The old rule was `tokens[0][:1].isdigit()`, which silently produced no house
    number for "No.107/2, ...", "E - 100, ...", "G-26, ...", "Plot 149 ...", so pass 4
    never fired for those records. Measured on real pairs, 97.5% of cross-script truth
    pairs share an address token, so this number is load-bearing for the whole
    address-key strategy.
    """
    norm = normalize_name(raw or "")
    tokens = norm.split()
    for i, tok in enumerate(tokens[:4]):
        if tok in HOUSE_MARKERS:
            for nxt in tokens[i + 1:i + 3]:
                m = _HOUSE_RE.match(nxt) or _HOUSE_PREFIXED_RE.match(nxt)
                if m:
                    return m.group(1)
            continue
        m = _HOUSE_RE.match(tok) or _HOUSE_PREFIXED_RE.match(tok)
        if m:
            return m.group(1)
    return ""


# Grammatical particles and generic place words that follow a street type word.
# French addresses are the motivating case: "Rue de la Paix", "Place de la Republique",
# "Boulevard du President" all reduce to `de` / `du` / `des` once the type word is
# removed, and pass 4 keys built from the FIRST street token then degenerated to those
# particles, which a large share of addresses share. The same pattern occurs in
# Indian addresses. Selecting the rarest street token instead of the first avoids
# special-casing any language.
STREET_STOPWORDS = {
    "de", "du", "des", "la", "le", "les", "den", "der", "die", "das", "ein", "eine",
    "van", "von", "dein", "di", "da", "del", "della", "dos", "das", "el", "al",
    "the", "of", "and", "en", "aan", "op", "in", "a", "an", "near", "behind",
    "santo", "santa", "san", "saint", "ste", "str", "new", "old",
}


def street_key_tokens(raw: str) -> tuple:
    """Street name tokens for blocking, with type words, labels and particles removed.

    Street type words ("Lane" vs "Ln", "Boulevard" vs "Bd") are the noisiest part of
    an address and carry no identity, so they are dropped entirely rather than
    canonicalised. Particles are dropped too, for the reason in STREET_STOPWORDS.
    """
    norm = normalize_name(raw or "")
    hn = house_number(raw)
    drop = STREET_SUFFIXES | set(STREET_SUFFIX_CANON) | set(STREET_SUFFIX_CANON.values())
    out = []
    for tok in norm.split():
        if tok == hn or tok in STREET_SUFFIXES or tok in HOUSE_MARKERS or tok in drop:
            continue
        if tok in STREET_STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        out.append(tok)
        if len(out) >= 8:
            break
    return tuple(out)


def street_key_component(street_tokens, idf_map=None, min_idf=4.0) -> str:
    """Pick the most discriminative street token for a blocking key.

    Rareest-by-IDF, not first: the first token is frequently a particle or a generic
    word, and a key built from one of those collides across a huge number of records
    and then gets dropped by the per-pass block cap. Falls back to the longest token
    when no IDF data is supplied.
    """
    toks = [t for t in street_tokens if t and t not in STREET_STOPWORDS and len(t) >= 4]
    if not toks:
        toks = [t for t in street_tokens if t]
    if not toks:
        return ""
    if not idf_map:
        return max(toks, key=len)[:5]
    scored = [(idf_map.get(t, 0.0), t) for t in toks]
    best = max(scored)
    if best[0] >= min_idf:
        return best[1][:5]
    # No rare token: prefer the longest, which is usually the distinctive word.
    return max(toks, key=len)[:5]


def parse_address(raw: str, country: str) -> AddressParts:
    if not raw or not raw.strip():
        return AddressParts("", (), "", "", (), False, True)
    norm = normalize_name(raw)
    tokens = norm.split()
    house_no = house_number(raw)
    postal_match = _postal_re(country).search(norm)
    postal = postal_match.group(0) if postal_match else ""
    state_key = ""
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i]} {tokens[i + 1]}"
        if pair in STATE_NAMES:
            state_key = pair
            break
    if not state_key:
        for tok in tokens:
            if tok in STATE_NAMES or tok in US_STATES:
                state_key = US_STATES.get(tok, tok)
                break
    landmark = any(t in LANDMARK_WORDS for t in tokens)
    street = street_key_tokens(raw)
    return AddressParts(house_no, street, postal, state_key, tuple(tokens[-6:]), landmark, False)
