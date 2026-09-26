from ber.address import parse_address, street_key_component, street_key_tokens
from ber.normalize import strip_legal_suffix

# French is ~15% of the test split with no training labels, and French street names put
# a type word followed by grammatical particles. Selecting the FIRST street token made
# pass 4 keys degenerate to "de"/"du"/"des", which a large share of French addresses
# share, so those blocks were dropped by the per-pass cap.

FRENCH = [
    ("12 Rue de la Paix, 75002 Paris", "12", "paix"),
    ("3 Place de la Republique, 69001 Lyon", "3", "republique"),
    ("175 Boulevard du President Franklin Roosevelt, Bordeaux, 33000", "175", "president"),
    ("45 Avenue des Champs Elysees, 75008 Paris", "45", "champs"),
    ("8 Rue Jean Jaures, 69100 Villeurbanne", "8", "jean"),
]


def test_french_street_tokens_drop_particles():
    for raw, _house, expected in FRENCH:
        toks = street_key_tokens(raw)
        assert "de" not in toks and "du" not in toks and "des" not in toks, raw
        assert "la" not in toks and "le" not in toks, raw
        assert expected in toks, f"{raw} -> {toks}"


def test_french_pass4_key_is_not_a_particle():
    for raw, house, _expected in FRENCH:
        p = parse_address(raw, "France")
        assert p.house_no == house, raw
        comp = street_key_component(p.street_tokens, {}, 4.0)
        assert comp not in {"de", "du", "des", "la", "le"}, f"{raw} -> {comp}"
        assert len(comp) >= 4, f"{raw} -> {comp}"


def test_street_key_component_prefers_the_rarest_token():
    # "main" is common; "fenwick" is rare. The rare one is the discriminative key.
    toks = ("main", "street", "fenwick")
    idf = {"main": 1.0, "fenwick": 9.0}
    assert street_key_component(toks, idf, 4.0) == "fenwi"


def test_street_key_component_falls_back_to_longest():
    toks = ("main", "broadway")
    assert street_key_component(toks, {}, 4.0) == "broad"


def test_street_key_component_ignores_particles_entirely():
    toks = ("de", "la", "paix")
    assert street_key_component(toks, {}, 4.0) == "paix"


def test_street_key_component_empty_is_safe():
    assert street_key_component((), {}, 4.0) == ""
    assert street_key_component(("",), {}, 4.0) == ""


def test_french_legal_suffixes_are_stripped():
    for raw, expected in (
        ("Engages Àrt Pharmacie SCI", "sci"),
        ("SARL Martin", "sarl"),
        ("EURL Dupont", "eurl"),
        ("SELARL Bernard", "selarl"),
    ):
        _stripped, suffix = strip_legal_suffix(raw)
        assert suffix == expected, raw


def test_us_addresses_still_key_correctly():
    # Regression guard: the rarest-token change must not damage the US case that
    # previously keyed on the first street token.
    p = parse_address("8060 Fenwick Lane, Spring Hill, TN", "US")
    assert p.house_no == "8060"
    comp = street_key_component(p.street_tokens, {"fenwick": 9.0, "spring": 5.0}, 4.0)
    assert comp == "fenwi"
