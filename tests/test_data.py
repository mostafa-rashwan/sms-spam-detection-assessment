import pandas as pd
import pytest

from spamdetect import data


def test_parse_handles_tabs_and_quotes(tmp_path):
    p = tmp_path / "SMSSpamCollection"
    p.write_text('ham\tHe said "hello" there\nspam\tWin "now"!\n', encoding="utf-8")
    df = data.parse(p)
    assert list(df.columns) == ["label", "text"]
    assert df["text"].tolist() == ['He said "hello" there', 'Win "now"!']


def test_parse_latin1_fallback(tmp_path):
    p = tmp_path / "f"
    p.write_bytes("ham\tcaf\xe9 time\n".encode("latin-1"))
    assert data.parse(p)["text"].iloc[0] == "café time"


def test_parse_rejects_bad_label(tmp_path):
    p = tmp_path / "f"
    p.write_text("maybe\thello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.parse(p)


def test_parse_rejects_extra_columns(tmp_path):
    p = tmp_path / "f"
    p.write_text("ham\ta\tb\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.parse(p)


def test_clean_dedups_and_maps_labels():
    df = pd.DataFrame({"label": ["ham", "ham", "spam", "ham"],
                       "text": ["hi", " hi ", "win", "   "]})
    out, stats = data.clean(df)
    assert out["text"].tolist() == ["hi", "win"]
    assert out["label"].tolist() == [0, 1]
    assert stats["duplicates_dropped"] == 1 and stats["empty_dropped"] == 1


def test_clean_label_conflict_raises():
    df = pd.DataFrame({"label": ["ham", "spam"], "text": ["same", "same"]})
    with pytest.raises(ValueError, match="conflicting"):
        data.clean(df)


def test_split_stratified_and_reproducible(raw_df):
    df, _ = data.clean(raw_df)
    a_tr, a_te = data.split(df, 0.25, 7)
    b_tr, b_te = data.split(df, 0.25, 7)
    assert a_te.index.equals(b_te.index)
    assert set(a_tr.index).isdisjoint(a_te.index)
    assert abs(a_tr["label"].mean() - a_te["label"].mean()) < 0.02
