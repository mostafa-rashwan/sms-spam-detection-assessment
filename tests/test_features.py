import numpy as np

from spamdetect.features import MetaFeatures, build_features


def test_vocabulary_fit_on_train_only(tiny_cfg):
    train = np.array(["win free prize now", "see you at lunch", "free prize win", "lunch at six"],
                     dtype=object)
    test = np.array(["zzyzxtoken appears only here"], dtype=object)
    feats = build_features(tiny_cfg).fit(train)
    names = feats.get_feature_names_out()
    assert not any("zzyzxtoken" in n for n in names)
    assert any(n.endswith("prize") for n in names)
    assert feats.transform(test).shape[1] == len(names)


def test_meta_features_shape_and_values():
    m = MetaFeatures().fit(["x"])
    out = m.transform(["WIN £150 now!!"])
    assert out.shape == (1, len(m.get_feature_names_out()))
    names = list(m.get_feature_names_out())
    assert out[0, names.index("n_exclaim")] == 2
    assert out[0, names.index("n_currencytoken")] == 1
