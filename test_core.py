"""Unit tests for core.py. Run with:  python -m pytest -q"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "phones.csv")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import core  # noqa: E402

CAT = core.load_catalog(os.path.join(ROOT, "phones.csv"))
_ART = {}


def art():
    """Train the market model once and reuse it across tests."""
    if "m" not in _ART:
        _ART["m"] = core.train_market(CAT)
    return _ART["m"]


# ------------------------------------------------------------------ helpers
def test_inr_uses_indian_grouping():
    assert core.inr(0) == "₹0"
    assert core.inr(950) == "₹950"
    assert core.inr(24300) == "₹24,300"
    assert core.inr(125000) == "₹1,25,000"
    assert core.inr(1234567) == "₹12,34,567"


def test_tier_boundaries():
    assert core.tier(14999) == 0
    assert core.tier(15000) == 1
    assert core.tier(29999) == 1
    assert core.tier(30000) == 2
    assert core.tier(49999) == 2
    assert core.tier(50000) == 3


def test_fmt_handles_missing_values():
    assert core.fmt(np.nan) == "-"
    assert core.fmt(8, " GB") == "8 GB"
    assert core.fmt(1234567, "", comma=True) == "1,234,567"


# ------------------------------------------------------------------ money tools
def test_emi_known_value():
    r = core.emi(100000, 0, 12, 12)
    assert abs(r["emi"] - 8884.88) < 0.01
    assert abs(r["total_payable"] - r["emi"] * 12) < 1e-6


def test_emi_zero_interest_splits_evenly():
    r = core.emi(24000, 0, 12, 0)
    assert r["emi"] == 2000 and r["interest"] == 0


def test_emi_down_payment_reduces_principal():
    full = core.emi(50000, 0, 6, 14)
    part = core.emi(50000, 20000, 6, 14)
    assert part["principal"] == 30000
    assert part["emi"] < full["emi"]
    assert abs(part["total_cost"] - (part["total_payable"] + 20000)) < 1e-6


def test_deal_verdict_levels():
    est, low, high = 20000, 14000, 28000
    assert core.deal_verdict(12000, est, low, high)[0] == "Great deal"
    assert core.deal_verdict(18000, est, low, high)[0] == "Good deal"
    assert core.deal_verdict(20500, est, low, high)[0] == "Fair price"
    assert core.deal_verdict(25000, est, low, high)[0] == "Slightly expensive"
    assert core.deal_verdict(30000, est, low, high)[0] == "Overpriced"
    assert core.deal_verdict(5000, est, low, high)[0] == "Too good to be true"
    assert abs(core.deal_verdict(22000, est, low, high)[2] - 10.0) < 1e-9


def test_offers_far_below_the_listed_price_are_flagged():
    """An iPhone 17 is listed at about ₹82,900, so a ₹30,000 offer must never look like a good deal."""
    assert core.listed_verdict(30000, 82900)[0] == "Too good to be true"
    assert core.listed_verdict(82900, 82900)[0] == "Fair price"
    assert core.listed_verdict(70000, 82900)[0] == "Great deal"
    assert core.listed_verdict(75000, 82900)[0] == "Good deal"
    assert core.listed_verdict(90000, 82900)[0] == "Slightly expensive"
    assert core.listed_verdict(105000, 82900)[0] == "Overpriced"
    assert abs(core.listed_verdict(41450, 82900)[2] + 50.0) < 1e-9  # half the listed price = 50% below


def test_ram_and_storage_snap_to_real_sizes():
    assert core.snap_value("ram_gb", 9, 8) == 10      # "+" from 8 GB goes to the next real size
    assert core.snap_value("ram_gb", 7, 8) == 6       # "-" goes to the previous real size
    assert core.snap_value("ram_gb", 14, 8) in (12, 16)  # typed or dragged -> nearest real size
    assert core.snap_value("ram_gb", 40, 24) == 24    # clamped to the largest size
    assert core.snap_value("storage_gb", 136, 128) == 256
    assert core.snap_value("storage_gb", 120, 128) == 64
    assert core.snap_value("storage_gb", 200, 128) == 256
    assert core.snap_value("battery_mah", 5555, 5500) == 5555  # other specs are left alone


# ------------------------------------------------------------------ catalog
def test_catalog_is_valid():
    assert core.validate_catalog(CAT) == []
    assert len(CAT) >= 60
    assert set(CAT["foldable"].unique()) <= {0, 1}
    assert set(CAT["five_g"].unique()) <= {0, 1}
    assert CAT["price_inr"].between(5000, core.PRICE_MAX).all()


def test_validate_catalog_catches_problems():
    assert core.validate_catalog(CAT.drop(columns=["price_inr"])) == ["Missing column: price_inr"]
    dup = pd.concat([CAT, CAT.iloc[[0]]], ignore_index=True)
    assert "Duplicate phone names" in core.validate_catalog(dup)
    bad = CAT.copy()
    bad.loc[0, "price_inr"] = -5
    assert any("positive" in p for p in core.validate_catalog(bad))


def test_presets_stay_within_slider_ranges():
    for name, preset in core.PRESETS.items():
        for key, (_, lo, hi, _) in core.SPECS.items():
            assert lo <= preset[key] <= hi, f"{name}: {key}={preset[key]} outside [{lo}, {hi}]"
        assert set(core.TOGGLES) <= set(preset)
        assert preset["ram_gb"] in core.SNAP_VALUES["ram_gb"]
        assert preset["storage_gb"] in core.SNAP_VALUES["storage_gb"]


def test_ui_to_dataset_converts_units():
    d = core.ui_to_dataset(core.PRESETS["Mid-range"])
    assert d["ram"] == 8 * 1024
    assert d["battery_power"] == 5500
    assert len(d) == 20
    assert all(isinstance(d[k], int) for k in core.TOGGLES)


def test_clip_to_range_caps_and_reports():
    raw = pd.DataFrame({"ram": [1000, 20000], "battery_power": [800, 900]})
    lo = pd.Series({"ram": 256, "battery_power": 500})
    hi = pd.Series({"ram": 4000, "battery_power": 2000})
    clipped, capped = core.clip_to_range(raw, lo, hi)
    assert list(clipped["ram"]) == [1000, 4000]
    assert capped == ["ram"]


# ------------------------------------------------------------------ ranking & recommendation
def test_weights_change_the_ranking():
    cheapest = core.score_catalog(CAT, dict(perf=0, batt=0, cam=0, disp=0, cheap=5)).sort_values("score").iloc[-1]
    assert cheapest["price_inr"] == CAT["price_inr"].min()
    fastest = core.score_catalog(CAT, dict(perf=5, batt=0, cam=0, disp=0, cheap=0)).sort_values("score").iloc[-1]
    assert fastest["antutu"] >= CAT["antutu"].quantile(0.9)


def test_scores_are_between_0_and_100_and_raw_values_are_kept():
    s = core.score_catalog(CAT)
    assert s["score"].between(0, 100).all()
    assert abs(s["score"].max() - 100) < 1e-9  # 100 = the best phone for the chosen priorities
    missing = CAT["front_mp"].isna()
    assert s.loc[missing, "front_mp"].isna().all()  # imputation is only used for scoring


def test_filters():
    s = core.score_catalog(CAT)
    out = core.apply_filters(s, brands=["Samsung"], min_ram=6)
    assert len(out) > 0 and (out["brand"] == "Samsung").all() and (out["ram_gb"] >= 6).all()
    four_g = core.apply_filters(s, five_g_only=True)
    assert (four_g["five_g"] == 1).all() and len(four_g) < len(s)
    assert core.apply_filters(s, min_battery=100000).empty


def test_phones_near_falls_back_to_closest():
    s = core.score_catalog(CAT)
    inside, note = core.phones_near(s, 30000, 40000, 35000)
    assert note == "" and inside["price_inr"].between(30000, 40000).all()
    far, note = core.phones_near(s, 500000, 600000, 550000)
    assert note != "" and len(far) == 3


def test_similar_phones():
    out = core.similar_phones(CAT, "vivo T5 5G", 3)
    assert len(out) == 3 and "vivo T5 5G" not in set(out["name"])
    assert out["similarity"].is_monotonic_decreasing
    assert out.iloc[0]["name"] == "iQOO Z11 5G"  # identical specs
    try:
        core.similar_phones(CAT, "Nokia 3310")
        assert False, "should raise"
    except KeyError:
        pass


def test_value_table_sorted_and_complete():
    v = core.value_table(CAT)
    assert v["antutu"].notna().all()
    assert v["points_per_1000"].is_monotonic_decreasing


# ------------------------------------------------------------------ market model
def test_market_model_quality_and_predictions():
    a = art()
    assert a["n"] == int((CAT["foldable"] == 0).sum())  # foldables are not used for training
    assert a["metrics"]["r2"] > 0.5
    flagship = core.market_row(core.PRESETS["Flagship"])
    budget = core.market_row(core.PRESETS["Budget"])
    pf, pb = core.predict_price(a, flagship), core.predict_price(a, budget)
    assert pb > 0 and pf > pb * 3  # a flagship must cost far more than a budget phone
    lo, hi = core.predict_interval(a, pf)
    assert lo < pf < hi


def test_inputs_beyond_catalog_are_capped():
    a = art()
    extreme = core.market_row(dict(core.PRESETS["Flagship"], ram_gb=24, antutu=4000000))
    assert core.out_of_range(a, extreme)
    assert core.predict_price(a, extreme) > 0


def test_shapley_contributions_add_up():
    a = art()
    row = core.market_row(core.PRESETS["Mid-range"])
    base, est, contrib = core.explain(a, row)
    assert abs(contrib.sum() - (est - base)) < 1e-6
    assert abs(est - core.predict_price(a, row)) < 1e-6
    assert set(contrib.index) == set(core.MARKET_NAMES.values())


def test_batch_predictions_match_single_predictions():
    a = art()
    rows = pd.DataFrame([core.market_row(core.PRESETS[k]) for k in ("Budget", "Mid-range", "Flagship")])
    batch = core.predict_prices(a, rows)
    for i, k in enumerate(("Budget", "Mid-range", "Flagship")):
        assert abs(batch[i] - core.predict_price(a, core.market_row(core.PRESETS[k]))) < 1e-6


def test_artifact_roundtrip_and_hash_check(tmp_path=None):
    import tempfile
    a = art()
    path = os.path.join(tmp_path or tempfile.mkdtemp(), "m.joblib")
    core.save_artifact(a, path)
    assert core.load_artifact(path, CAT) is not None
    changed = CAT.copy()
    changed.loc[0, "price_inr"] += 1000
    assert core.load_artifact(path, changed) is None  # catalog changed -> do not reuse the saved model
    assert core.load_artifact(os.path.join(os.path.dirname(path), "missing.joblib"), CAT) is None


def test_saved_model_is_in_sync_with_the_catalog():
    """If this fails, run `python train_models.py` and commit market_model.joblib."""
    path = os.path.join(ROOT, "market_model.joblib")
    if not os.path.exists(path):
        return
    try:
        saved = core.joblib.load(path)
    except Exception:
        return  # different scikit-learn version: the app retrains automatically
    assert saved["hash"] == core.catalog_hash(CAT)
