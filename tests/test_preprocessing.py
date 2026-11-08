import pytest

from spamdetect.preprocessing import Preprocessor, preprocess


@pytest.mark.parametrize("text,token", [
    ("visit http://x.com/a now", "urltoken"),
    ("go to www.site.org", "urltoken"),
    ("mail me a.b@c.co.uk", "emailtoken"),
    ("call 09061701461", "phonetoken"),
    ("call +44 7911 123456", "phonetoken"),
    ("win £150 now", "currencytoken"),
    ("only $5.50", "currencytoken"),
    ("just 150p per week", "currencytoken"),
    ("I have 3 cats", "numbertoken"),
])
def test_placeholders(text, token):
    assert token in preprocess(text).split()


def test_digits_and_currency_removed():
    out = preprocess("Win £1000 call 09061701461 or http://a.b")
    assert not any(c.isdigit() for c in out) and "£" not in out and "http" not in out


def test_unicode_nfkc_and_lowercase():
    assert preprocess("ＨＥＬＬＯ") == "hello"


def test_idempotent():
    s = "WIN £150!! call 09061701461 http://x.y a@b.com  3 cats"
    assert preprocess(preprocess(s)) == preprocess(s)


def test_empty_string():
    assert preprocess("") == ""


def test_preprocessor_stopwords():
    assert "the" not in Preprocessor(remove_stopwords=True)("the cat sat").split()
