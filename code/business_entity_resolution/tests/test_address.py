import re

from ber.address import house_number, parse_address, street_key_tokens


def test_parse_us_address():
    p = parse_address("1795 Westchester Drive, High Point, NC", "US")
    assert p.house_no == "1795"
    assert "westchester" in p.street_tokens
    assert p.state_key == "north carolina"
    assert p.postal == ""
    assert not p.addr_missing


def test_parse_india_pin():
    p = parse_address("G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh 462033", "India")
    assert p.postal == "462033"
    assert p.state_key == "madhya pradesh"


def test_parse_france_postal():
    p = parse_address("175 Boulevard du President Franklin Roosevelt, Bordeaux, 33000", "France")
    assert p.house_no == "175"
    assert p.postal == "33000"
    assert p.state_key == "bordeaux"


def test_landmark_and_missing():
    p = parse_address("Near SBI ATM, MG Road", "India")
    assert p.landmark_flag is True
    assert parse_address("", "US").addr_missing is True


# --- house-number extraction -------------------------------------------------
# Measured on real cross-script truth pairs: 97.5% of cross-script pairs share an
# address token, so address keys are the strongest available bridge. They only work
# if house_no is extracted, and the previous tokens[0]-must-be-a-digit rule silently
# dropped every address that leads with a label or a plot/flat/sector marker.

HOUSE_CASES = [
    ("1795 Westchester Drive, High Point, NC", "1795"),
    ("No.107/2, Budihal Village, Bangalore Rural", "107"),
    ("E - 100, South Delhi, New Delhi", "100"),
    ("G-26, Kamalakunj, Poddar Road, Mumbai", "26"),
    ("4, KTR NAGA, Bye Pass Road", "4"),
    ("Plot 149 Pragathi Nagar, Hyderabad", "149"),
    ("Flat 302, Empire Village, Mumbai", "302"),
    ("#98825, Main Street, Austin TX", "98825"),
    # normalize_name turns "/" into a space, so only the leading component survives.
    ("12/34B, Anna Salai, Chennai", "12"),
    ("B-14, Sector 18, Rohini, Delhi", "14"),
    ("No. 7, MG Road, Indore", "7"),
    ("27, MG Road", "27"),
]


def test_house_number_extraction():
    for raw, expected in HOUSE_CASES:
        assert house_number(raw) == expected, raw


def test_house_number_absent_when_no_marker():
    assert house_number("Mumbai City, Kamalakunj, Poddar Road") == ""


def test_parse_address_uses_new_house_number():
    p = parse_address("No.107/2, Budihal Village, Bangalore Rural, Karnataka", "India")
    assert p.house_no == "107"
    assert p.state_key == "karnataka"


# --- street key robustness ---------------------------------------------------

def test_street_key_ignores_case_and_suffix_abbreviation():
    a = street_key_tokens("8060 Fenwick Lane, Spring Hill, TN")
    b = street_key_tokens("8060 Fenwick Ln, Spring Hill, Tennessee")
    assert a[:2] == b[:2]


def test_street_key_survives_locale_suffix_translation():
    a = street_key_tokens("175 Boulevard du President, Bordeaux")
    b = street_key_tokens("175 Bd du President, Bordeaux")
    assert a == b
    # "du" is a grammatical particle and is deliberately dropped; keeping it made the
    # blocking key collide across every French address that starts with "du".
    assert a[:2] == ("president", "bordeaux")
