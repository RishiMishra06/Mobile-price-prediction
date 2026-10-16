"""Core logic for the Mobile Price Predictor (no Streamlit imports, so it can be unit tested)."""
import hashlib
import math
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# ---------------------------------------------------------------- constants
LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
TIER_NAMES = {0: "Budget", 1: "Mid-range", 2: "Upper mid-range", 3: "Premium"}
BOOL_COLS = ["blue", "wifi", "dual_sim", "three_g", "four_g", "touch_screen"]
TOGGLES = {"blue": "Bluetooth", "wifi": "WiFi", "dual_sim": "Dual SIM", "three_g": "3G", "four_g": "4G",
           "touch_screen": "Touch screen"}

# UI field: (label, min, max, step). Ranges follow what real phones sold today have.
SPECS = {
    "ram_gb": ("RAM (GB)", 1, 24, 1),
    "storage_gb": ("Internal storage (GB)", 8, 1024, 8),
    "battery_mah": ("Battery (mAh)", 1000, 10000, 100),
    "charging_w": ("Fast charging (W)", 5, 240, 1),
    "antutu": ("Processor performance (AnTuTu score)", 100000, 4000000, 50000),
    "clock_speed": ("Processor clock speed (GHz)", 0.5, 4.5, 0.1),
    "n_cores": ("Processor cores", 1, 8, 1),
    "talk_time": ("Talk time (hours)", 2, 40, 1),
    "px_height": ("Screen resolution - height (px)", 480, 3400, 10),
    "px_width": ("Screen resolution - width (px)", 480, 1600, 10),
    "sc_h": ("Screen height (cm)", 5, 20, 1),
    "sc_w": ("Screen width (cm)", 2, 10, 1),
    "fc": ("Front camera (MP)", 0, 60, 1),
    "pc": ("Main rear camera (MP)", 0, 200, 1),
    "m_dep": ("Thickness (cm)", 0.1, 1.5, 0.1),
    "mobile_wt": ("Weight (g)", 80, 300, 1),
}
_ON = {k: True for k in TOGGLES}
PRESETS = {
    "Budget": dict(ram_gb=4, storage_gb=64, battery_mah=5000, charging_w=18, antutu=400000, clock_speed=2.0,
                   n_cores=8, talk_time=25, px_height=1600, px_width=720, sc_h=15, sc_w=7, fc=5, pc=13,
                   m_dep=0.9, mobile_wt=195, **_ON),
    "Mid-range": dict(ram_gb=8, storage_gb=128, battery_mah=5500, charging_w=45, antutu=800000, clock_speed=2.6,
                      n_cores=8, talk_time=30, px_height=2400, px_width=1080, sc_h=16, sc_w=7, fc=16, pc=50,
                      m_dep=0.8, mobile_wt=190, **_ON),
    "Flagship": dict(ram_gb=12, storage_gb=256, battery_mah=5500, charging_w=80, antutu=3000000, clock_speed=3.4,
                     n_cores=8, talk_time=35, px_height=3200, px_width=1440, sc_h=16, sc_w=7, fc=32, pc=50,
                     m_dep=0.8, mobile_wt=205, **_ON),
}

MARKET_FEATS = ["ram_gb", "storage_gb", "battery_mah", "charging_w", "rear_mp", "front_mp", "antutu"]
MARKET_NAMES = {"ram_gb": "RAM", "storage_gb": "Storage", "battery_mah": "Battery", "charging_w": "Charging",
                "rear_mp": "Rear camera", "front_mp": "Front camera", "antutu": "Processor power"}
# Found with GridSearchCV (see train_models.py)
BEST_PARAMS = dict(n_estimators=250, max_depth=2, learning_rate=0.1, min_samples_leaf=1, subsample=0.8,
                   random_state=42)
PARAM_GRID = {"n_estimators": [100, 150, 250], "max_depth": [2, 3], "learning_rate": [0.03, 0.05, 0.1],
              "min_samples_leaf": [1, 3], "subsample": [1.0, 0.8]}
COVERAGE = 0.8

NUM_COLS = ["price_inr", "ram_gb", "storage_gb", "battery_mah", "charging_w", "rear_mp", "front_mp",
            "display_in", "antutu"]
DIM_NAMES = {"perf": "performance", "batt": "battery & charging", "cam": "camera MP", "disp": "display size",
             "cheap": "low price"}
DEFAULT_WEIGHTS = {"perf": 3, "batt": 3, "cam": 3, "disp": 3, "cheap": 3}
SIM_FEATS = ["ram_gb", "storage_gb", "battery_mah", "charging_w", "antutu", "display_in", "rear_mp", "front_mp"]
CATALOG_REQUIRED = ["brand", "model", "price_inr", "ram_gb", "storage_gb", "battery_mah", "charging_w",
                    "rear_mp", "front_mp", "display_in", "chipset"]


# ---------------------------------------------------------------- formatting helpers
def inr(n):
    """Format a number in Indian style, e.g. 125000 -> ₹1,25,000"""
    s = str(int(round(n)))
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join(groups) + "," + tail


def tier(price):
    return 0 if price < 15000 else 1 if price < 30000 else 2 if price < 50000 else 3


def fmt(v, suffix="", comma=False):
    if v is None or pd.isna(v):
        return "-"
    return (f"{v:,.0f}" if comma else f"{v:g}") + suffix


# ---------------------------------------------------------------- catalog
def load_catalog(path="phones.csv"):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def validate_catalog(cat):
    """Return a list of problems (empty list = OK)."""
    problems = [f"Missing column: {c}" for c in CATALOG_REQUIRED if c not in cat.columns]
    if problems:
        return problems
    if (pd.to_numeric(cat["price_inr"], errors="coerce") <= 0).any():
        problems.append("price_inr must be a positive number for every phone")
    if (cat["brand"].astype(str) + " " + cat["model"].astype(str)).duplicated().any():
        problems.append("Duplicate phone names")
    return problems


def catalog_hash(cat):
    cols = [c for c in MARKET_FEATS + ["price_inr", "brand", "model"] if c in cat.columns]
    return hashlib.sha1(cat[cols].to_csv(index=False).encode()).hexdigest()


# ---------------------------------------------------------------- spec-based score & recommendation
def score_catalog(cat, weights=None):
    """Spec-based score. Missing values are filled with the column median ONLY for scoring;
    the raw values are kept for display."""
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    d = cat.copy()
    if "antutu" not in d.columns:
        d["antutu"] = np.nan
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    f = d[NUM_COLS].apply(lambda s: s.fillna(s.median())).fillna(0)
    f["price_inr"] = np.log(f["price_inr"].clip(lower=1))

    def nz(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0 + 0.5

    d["perf"] = 0.6 * nz(f["antutu"]) + 0.25 * nz(f["ram_gb"]) + 0.15 * nz(f["storage_gb"])
    d["batt"] = 0.7 * nz(f["battery_mah"]) + 0.3 * nz(f["charging_w"])
    d["cam"] = 0.5 * nz(f["rear_mp"]) + 0.5 * nz(f["front_mp"])
    d["disp"] = nz(f["display_in"])
    d["cheap"] = 1 - nz(f["price_inr"])

    tot = sum(w.values()) or 1
    d["score"] = 100 * sum(w[k] * d[k] for k in w) / tot

    known = {"perf": pd.Series(True, index=d.index), "cheap": pd.Series(True, index=d.index),
             "batt": d["battery_mah"].notna(), "cam": d["rear_mp"].notna() & d["front_mp"].notna(),
             "disp": d["display_in"].notna()}
    d["why"] = [
        " & ".join(DIM_NAMES[k] for k in sorted([k for k in w if known[k][i] and w[k] > 0],
                                                key=lambda k: -w[k] * d.at[i, k])[:2]) or "overall balance"
        for i in d.index
    ]
    d["name"] = d["brand"].astype(str) + " " + d["model"].astype(str)
    return d


def apply_filters(scored, brands=None, five_g_only=False, min_ram=0, min_storage=0, min_battery=0):
    m = pd.Series(True, index=scored.index)
    if brands:
        m &= scored["brand"].isin(brands)
    if five_g_only and "five_g" in scored.columns:
        m &= scored["five_g"].fillna(1).astype(int) == 1
    m &= scored["ram_gb"].fillna(0) >= min_ram
    m &= scored["storage_gb"].fillna(0) >= min_storage
    m &= scored["battery_mah"].fillna(0) >= min_battery
    return scored[m]


def phones_near(scored, lo, hi, est, k=3):
    """Top phones by score inside [lo, hi]; falls back to the closest by price."""
    m = scored[(scored["price_inr"] >= lo) & (scored["price_inr"] <= hi)]
    note = ""
    if m.empty:
        m = scored.iloc[(scored["price_inr"] - est).abs().argsort()[:k]]
        note = "No phone in the catalog falls inside this range, showing the closest by price."
    return m.sort_values("score", ascending=False).head(k), note


def similar_phones(cat, name, k=3):
    """Cosine similarity on standardised specs (price is not used)."""
    names = (cat["brand"].astype(str) + " " + cat["model"].astype(str)).reset_index(drop=True)
    if name not in set(names):
        raise KeyError(name)
    X = cat.reset_index(drop=True).reindex(columns=SIM_FEATS).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    sd = X.std(ddof=0).replace(0, 1)
    Z = ((X - X.mean()) / sd).values
    i = int(names[names == name].index[0])
    norms = np.linalg.norm(Z, axis=1)
    norms[norms == 0] = 1
    sims = (Z @ Z[i]) / (norms * norms[i])
    out = cat.reset_index(drop=True).assign(name=names, similarity=np.round((sims + 1) / 2 * 100, 1))
    out = out[out["name"] != name].sort_values("similarity", ascending=False).head(k)
    return out


def value_table(cat):
    """AnTuTu points per ₹1,000 (only phones with a benchmark score)."""
    d = cat.copy()
    d["antutu"] = pd.to_numeric(d.get("antutu"), errors="coerce")
    d = d.dropna(subset=["antutu"])
    d["name"] = d["brand"].astype(str) + " " + d["model"].astype(str)
    d["points_per_1000"] = (d["antutu"] / (d["price_inr"] / 1000)).round(0)
    return d.sort_values("points_per_1000", ascending=False)


# ---------------------------------------------------------------- money tools
def emi(price, down, months, annual_rate_pct):
    principal = max(float(price) - float(down), 0.0)
    r = annual_rate_pct / 1200.0
    if months <= 0:
        raise ValueError("months must be positive")
    if r == 0:
        m = principal / months
    else:
        g = (1 + r) ** months
        m = principal * r * g / (g - 1)
    payable = m * months
    return dict(emi=m, principal=principal, total_payable=payable, interest=payable - principal,
                total_cost=payable + float(down))


def deal_verdict(asking, est, low, high):
    """Compare an asking price with the model's estimate and its prediction interval."""
    diff = (asking / est - 1) * 100
    if asking < low:
        return "Great deal", 0, diff
    if asking <= est * 0.95:
        return "Good deal", 1, diff
    if asking <= est * 1.05:
        return "Fair price", 2, diff
    if asking <= high:
        return "Slightly expensive", 3, diff
    return "Overpriced", 4, diff


# ---------------------------------------------------------------- market price model
def _market_frame(cat):
    d = cat.copy()
    if "antutu" not in d.columns:
        d["antutu"] = np.nan
    for c in MARKET_FEATS + ["price_inr"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["price_inr"]).reset_index(drop=True)
    d["name"] = d["brand"].astype(str) + " " + d["model"].astype(str)
    med = d[MARKET_FEATS].median()
    X = d[MARKET_FEATS].fillna(med)
    return d, X, np.log(d["price_inr"].values), med


def _shapley_log(model, x, bg):
    """Exact Shapley values (interventional, background = catalog) in log-price space."""
    n = len(x)
    masks = np.arange(2 ** n)
    big = np.repeat(bg[None, :, :], len(masks), axis=0)  # (masks, rows, feats)
    for j in range(n):
        has = ((masks >> j) & 1).astype(bool)
        big[has, :, j] = x[j]
    v = model.predict(pd.DataFrame(big.reshape(-1, n), columns=MARKET_FEATS)).reshape(len(masks), -1).mean(axis=1)
    fact = [math.factorial(i) for i in range(n + 1)]
    phi = np.zeros(n)
    for j in range(n):
        for m in masks:
            if (m >> j) & 1:
                continue
            s = bin(m).count("1")
            phi[j] += fact[s] * fact[n - s - 1] / fact[n] * (v[m | (1 << j)] - v[m])
    return phi, v[0], v[-1]


def global_impact(art):
    """Average absolute Shapley impact per feature, as ±% on the price."""
    bg = art["bg"].values
    acc = np.zeros(len(MARKET_FEATS))
    for row in bg:
        phi, _, _ = _shapley_log(art["model"], row, bg)
        acc += np.abs(phi)
    acc /= len(bg)
    s = pd.Series((np.exp(acc) - 1) * 100, index=[MARKET_NAMES[f] for f in MARKET_FEATS])
    return s.sort_values(ascending=False)


def _finalize(cat, params, oof_log, method, extra_metrics=None):
    d, X, y, med = _market_frame(cat)
    price = d["price_inr"].values
    pred = np.exp(oof_log)
    n = len(y)
    res = np.abs(y - oof_log)
    q = float(np.quantile(res, min(1.0, math.ceil((n + 1) * COVERAGE) / n)))
    model = GradientBoostingRegressor(**params).fit(X, y)
    metrics = dict(r2=float(r2_score(y, oof_log)), mae=float(mean_absolute_error(price, pred)),
                   mape=float(100 * np.mean(np.abs(pred - price) / price)), method=method)
    metrics.update(extra_metrics or {})
    art = dict(model=model, feats=MARKET_FEATS, med=med, lo=X.min(), hi=X.max(), n=n, hash=catalog_hash(cat),
               params=params, metrics=metrics, conformal_q=q, bg=X.reset_index(drop=True),
               oof=pd.DataFrame({"name": d["name"], "Actual price": price, "Predicted price": pred}),
               importance=pd.Series(model.feature_importances_, index=[MARKET_NAMES[f] for f in MARKET_FEATS])
               .sort_values(ascending=False),
               sklearn_version=sklearn.__version__, built_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    art["global_impact"] = global_impact(art)
    return art


def train_market(cat, params=None):
    """Fit on the catalog; out-of-sample predictions come from leave-one-out CV."""
    params = dict(params or BEST_PARAMS)
    d, X, y, _ = _market_frame(cat)
    if len(d) < 15:
        return None
    oof = cross_val_predict(GradientBoostingRegressor(**params), X, y, cv=LeaveOneOut())
    return _finalize(cat, params, oof, "leave-one-out CV")


def save_artifact(art, path):
    joblib.dump(art, path)


def load_artifact(path, cat):
    """Load a saved model only if it matches this catalog and scikit-learn version."""
    if not os.path.exists(path):
        return None
    try:
        art = joblib.load(path)
        if art.get("hash") != catalog_hash(cat) or art.get("sklearn_version") != sklearn.__version__:
            return None
        return art
    except Exception:
        return None


def _clip_row(art, row):
    x = pd.DataFrame([row]).reindex(columns=MARKET_FEATS)
    x = x.fillna(art["med"])
    return x.clip(lower=art["lo"], upper=art["hi"], axis=1)


def market_row(S):
    return {"ram_gb": S["ram_gb"], "storage_gb": S["storage_gb"], "battery_mah": S["battery_mah"],
            "charging_w": S["charging_w"], "rear_mp": S["pc"], "front_mp": S["fc"], "antutu": S["antutu"]}


def out_of_range(art, row):
    return [f"{MARKET_NAMES[f]} ({row[f]:g})" for f in MARKET_FEATS
            if row[f] < art["lo"][f] or row[f] > art["hi"][f]]


def predict_price(art, row):
    return float(np.exp(art["model"].predict(_clip_row(art, row))[0]))


def predict_prices(art, frame):
    x = frame.reindex(columns=MARKET_FEATS).apply(pd.to_numeric, errors="coerce").fillna(art["med"])
    x = x.clip(lower=art["lo"], upper=art["hi"], axis=1)
    return np.exp(art["model"].predict(x))


def predict_interval(art, est):
    f = math.exp(art["conformal_q"])
    return est / f, est * f


def explain(art, row):
    """Exact Shapley contributions, rescaled to ₹ so they add up to (estimate - average phone)."""
    x = _clip_row(art, row).values[0]
    phi, v0, v1 = _shapley_log(art["model"], x, art["bg"].values)
    base, est = math.exp(v0), math.exp(v1)
    scale = (est - base) / (v1 - v0) if abs(v1 - v0) > 1e-9 else 0.0
    contrib = pd.Series(phi * scale, index=[MARKET_NAMES[f] for f in MARKET_FEATS])
    return base, est, contrib


# ---------------------------------------------------------------- classification model (report)
def ui_to_dataset(S):
    """Convert the phone-friendly inputs to the units used by the Kaggle training dataset."""
    d = {"battery_power": S["battery_mah"], "ram": S["ram_gb"] * 1024, "int_memory": S["storage_gb"],
         "clock_speed": S["clock_speed"], "n_cores": S["n_cores"], "talk_time": S["talk_time"],
         "px_height": S["px_height"], "px_width": S["px_width"], "sc_h": S["sc_h"], "sc_w": S["sc_w"],
         "fc": S["fc"], "pc": S["pc"], "m_dep": S["m_dep"], "mobile_wt": S["mobile_wt"]}
    for k in TOGGLES:
        d[k] = int(S[k])
    return d


def clip_to_range(raw, lo, hi):
    """Cap values to the training range (a linear model would extrapolate wildly)."""
    cols = list(raw.columns)
    capped = [c for c in cols if (raw[c] < lo[c]).any() or (raw[c] > hi[c]).any()]
    clipped = pd.DataFrame(np.clip(raw.values, lo[cols].values, hi[cols].values), columns=cols)
    return clipped, capped
