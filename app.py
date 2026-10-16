"""SpecWorth - ML-based smartphone price estimator and deal checker for the Indian market.

Author: Rishi Mishra, B.Tech CSE, BPIT Rohini (Winnovation ML training project).
"""
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import core
import pdf_report
from core import BOOL_COLS, LABELS, MARKET_FEATS, MARKET_NAMES, PRESETS, SPECS, TIER_NAMES, TOGGLES, fmt, inr, tier

st.set_page_config(page_title="SpecWorth", page_icon="📱", layout="centered")

GRAD = {0: ("#16a34a", "#15803d"), 1: ("#2563eb", "#1d4ed8"), 2: ("#d97706", "#b45309"), 3: ("#dc2626", "#b91c1c")}
VERDICT_COLORS = ["#15803d", "#16a34a", "#2563eb", "#d97706", "#dc2626"]
W_KEYS = {"perf": "w_perf", "batt": "w_batt", "cam": "w_cam", "disp": "w_disp", "cheap": "w_price"}
FRIENDLY = {"battery_power": "Battery", "ram": "RAM", "int_memory": "Storage", "clock_speed": "Clock speed",
            "px_height": "Screen height (px)", "px_width": "Screen width (px)", "fc": "Front camera",
            "pc": "Rear camera", "talk_time": "Talk time", "mobile_wt": "Weight", "sc_h": "Screen height",
            "sc_w": "Screen width", "m_dep": "Thickness", "n_cores": "Cores"}

# ---------- styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], button, input {font-family: 'Inter', sans-serif;}
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 780px;}

    .hero {background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #4338ca 100%);
           border-radius: 20px; padding: 1.5rem 1.4rem; color: #fff; margin-bottom: 1.1rem;
           box-shadow: 0 10px 30px rgba(67,56,202,.25);}
    .hero .eyebrow {font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; opacity: .65; font-weight: 600;}
    .hero h1 {font-size: 1.7rem; font-weight: 800; margin: .3rem 0 .4rem 0; padding: 0; color: #fff; letter-spacing: -.02em;}
    .hero p {margin: 0; font-size: .92rem; opacity: .82; line-height: 1.45;}

    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 12px; font-weight: 600; font-size: .86rem;}

    .lbl {font-size: .82rem; font-weight: 600; opacity: .85; margin: .7rem 0 .1rem 0;}
    .sec {font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
          opacity: .6; margin: 1.2rem 0 .45rem 0;}

    div.stButton > button, div.stDownloadButton > button {border-radius: 12px; font-weight: 600; padding: .55rem 1rem;}
    div.stButton > button[kind="primary"] {background: #4f46e5; border: 0; color: #fff;}
    div.stButton > button[kind="primary"]:hover {background: #4338ca; color: #fff;}
    div[data-testid="stExpander"] {border-radius: 14px;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}

    .result {border-radius: 20px; padding: 1.5rem 1.2rem; text-align: center; color: #fff; margin: .8rem 0 .4rem 0;
             box-shadow: 0 10px 28px rgba(0,0,0,.18);}
    .result .eyebrow {font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; opacity: .85; font-weight: 600;}
    .result .price {font-size: 2.6rem; font-weight: 800; letter-spacing: -.02em; margin: .15rem 0 .3rem 0;}
    .result .badge {display: inline-block; background: rgba(255,255,255,.2); padding: 4px 14px; border-radius: 999px;
                    font-weight: 600; font-size: .9rem;}
    .result .sub {margin-top: .7rem; font-size: .85rem; opacity: .92;}

    .pb {display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: .85rem;}
    .pb .l {width: 112px; opacity: .85;}
    .pb .t {flex: 1; height: 8px; border-radius: 99px; background: rgba(128,128,128,.2); overflow: hidden;}
    .pb .t > div {height: 100%; border-radius: 99px;}
    .pb .v {width: 50px; text-align: right; font-weight: 600;}

    .pc {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .85rem 1rem; margin: .55rem 0 .2rem 0;
         background: rgba(128,128,128,.05);}
    .pc .top {display: flex; justify-content: space-between; align-items: baseline; gap: 10px;}
    .pc .nm {font-weight: 700; font-size: .98rem;}
    .pc .pr {font-weight: 800; color: #6366f1; white-space: nowrap;}
    .pc .chip {display: inline-block; padding: 2px 9px; border-radius: 999px; background: rgba(99,102,241,.13);
               font-size: .73rem; margin: 7px 5px 0 0; font-weight: 500;}
    .pc .bar {height: 6px; border-radius: 99px; background: rgba(128,128,128,.2); overflow: hidden; margin-top: 10px;}
    .pc .bar > div {height: 100%; background: linear-gradient(90deg, #6366f1, #a855f7);}
    .pc .why {font-size: .75rem; opacity: .68; margin-top: 6px;}

    .gauge {position: relative; height: 12px; border-radius: 99px; background: rgba(128,128,128,.2); margin: 34px 4px 10px 4px;}
    .gauge .rng {position: absolute; top: 0; height: 100%; border-radius: 99px; background: rgba(99,102,241,.5);}
    .gauge .est {position: absolute; top: -5px; width: 3px; height: 22px; background: #6366f1; border-radius: 2px;}
    .gauge .dot {position: absolute; top: -3px; width: 18px; height: 18px; border-radius: 50%; border: 3px solid #fff;
                 transform: translateX(-50%); box-shadow: 0 2px 6px rgba(0,0,0,.35);}
    .gauge .tag {position: absolute; top: -24px; font-size: .7rem; font-weight: 600; transform: translateX(-50%); white-space: nowrap;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- data & models ----------
@st.cache_data
def default_catalog():
    return core.load_catalog("phones.csv")


def get_catalog():
    ov = st.session_state.get("catalog_override")
    return ov if ov is not None else default_catalog()


@st.cache_resource(show_spinner="Loading the price model...")
def get_market(cat_hash, _cat, is_default):
    """Use the saved joblib model when it matches the catalog, otherwise train one."""
    art = core.load_artifact("market_model.joblib", _cat)
    if art is None:
        art = core.train_market(_cat)
        if art is not None and is_default:
            try:
                core.save_artifact(art, "market_model.joblib")
            except Exception:
                pass
    return art


@st.cache_resource(show_spinner="Training the classification models...")
def train_all():
    df = pd.read_csv("train.csv")
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    models = {
        "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    scores, preds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        preds[name] = m.predict(Xte)
        scores[name] = accuracy_score(y_test, preds[name]) * 100
    return dict(df=df, columns=list(X.columns), scaler=scaler, models=models, scores=scores, preds=preds,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def make_pipeline(name):
    est = {"Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
           "Random Forest": RandomForestClassifier(random_state=42),
           "KNN": KNeighborsClassifier()}[name]
    return Pipeline([("scale", StandardScaler()), ("model", est)])


@st.cache_data(show_spinner=False)
def cv_report():
    T_ = train_all()
    rows = []
    for name in T_["models"]:
        s = cross_val_score(make_pipeline(name), T_["X_train"], T_["y_train"],
                            cv=StratifiedKFold(5, shuffle=True, random_state=42)) * 100
        rows.append({"Model": name, "5-fold CV accuracy (%)": round(s.mean(), 2), "Std": round(s.std(), 2)})
    return pd.DataFrame(rows)


def run_tuning():
    T_ = train_all()
    grids = {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
        "Random Forest": {"model__n_estimators": [100, 300], "model__max_depth": [None, 10, 20],
                          "model__min_samples_leaf": [1, 2]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 9, 11, 15, 21], "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]},
    }
    rows = []
    for name, grid in grids.items():
        gs = GridSearchCV(make_pipeline(name), grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="accuracy", n_jobs=-1).fit(T_["X_train"], T_["y_train"])
        tuned = accuracy_score(T_["y_test"], gs.predict(T_["X_test"])) * 100
        rows.append({"Model": name, "Default test acc (%)": round(T_["scores"][name], 2),
                     "Tuned CV acc (%)": round(gs.best_score_ * 100, 2), "Tuned test acc (%)": round(tuned, 2),
                     "Best parameters": ", ".join(f"{k.split('__')[1]}={v}" for k, v in gs.best_params_.items())})
    return pd.DataFrame(rows)


try:
    T = train_all()
except FileNotFoundError:
    st.error("train.csv not found. Keep the dataset in the same folder as app.py.")
    st.stop()

df, columns, scaler = T["df"], T["columns"], T["scaler"]
model = T["models"]["Logistic Regression"]

_cat = get_catalog()
M = (get_market(core.catalog_hash(_cat), _cat, st.session_state.get("catalog_override") is None)
     if _cat is not None and not _cat.empty and not core.validate_catalog(_cat) else None)

# ---------- state ----------
st.session_state.setdefault("history", [])
st.session_state.setdefault("wishlist", [])
for _k in W_KEYS.values():
    st.session_state.setdefault(_k, 3)


def set_val(key, v):
    if key in SPECS:
        v = float(v) if isinstance(SPECS[key][3], float) else int(v)
        for k in (key, f"{key}__s", f"{key}__n"):
            st.session_state[k] = v
    else:
        st.session_state[key] = bool(v)


def load_preset(name):
    for k, v in PRESETS[name].items():
        set_val(k, v)


if "ram_gb" not in st.session_state:
    load_preset("Mid-range")


def _sync(key, src_):
    v = st.session_state[f"{key}__{src_}"]
    st.session_state[key] = v
    st.session_state[f"{key}__{'n' if src_ == 's' else 's'}"] = v


def spec_input(key):
    """A slider and a number box that stay in sync."""
    label, lo, hi, step = SPECS[key]
    st.markdown(f'<div class="lbl">{label}</div>', unsafe_allow_html=True)
    a, b = st.columns([3.2, 1.3], vertical_alignment="center")
    kw = dict(min_value=lo, max_value=hi, step=step, label_visibility="collapsed")
    a.slider(label, key=f"{key}__s", on_change=_sync, args=(key, "s"), **kw)
    extra = {"format": "%.1f"} if isinstance(step, float) else {}
    b.number_input(label, key=f"{key}__n", on_change=_sync, args=(key, "n"), **kw, **extra)


def current_specs():
    return {k: st.session_state[k] for k in list(SPECS) + list(TOGGLES)}


def weights():
    return {k: st.session_state[v] for k, v in W_KEYS.items()}


def classify(S):
    raw = pd.DataFrame([core.ui_to_dataset(S)])[columns]
    lo, hi = df[columns].min(), df[columns].max()
    clipped, capped = core.clip_to_range(raw, lo, hi)
    probs = model.predict_proba(scaler.transform(clipped))[0]
    return int(probs.argmax()), probs, capped, raw, lo, hi


def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def toggle_wish(name):
    w = st.session_state.wishlist
    if name in w:
        w.remove(name)
    else:
        w.append(name)


def phone_card(r, rank=None, ctx="x", caption=None):
    chips = [f"{fmt(r['ram_gb'])} GB RAM", f"{fmt(r['storage_gb'])} GB storage", f"{fmt(r['battery_mah'])} mAh",
             f"{fmt(r['charging_w'])} W", escape(str(r["chipset"]))]
    if "five_g" in r and r["five_g"] == 1:
        chips.insert(0, "5G")
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    pct = min(max(float(r["score"]), 0), 100)
    prefix = f"#{rank} " if rank else ""
    bottom = caption or f"Match score {pct:.0f}/100 · strong in {escape(str(r['why']))}"
    st.markdown(
        f"""
        <div class="pc">
          <div class="top"><span class="nm">{prefix}{escape(str(r['name']))}</span><span class="pr">{inr(r['price_inr'])}</span></div>
          <div>{chips_html}</div>
          <div class="bar"><div style="width:{pct:.0f}%"></div></div>
          <div class="why">{bottom}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    saved = r["name"] in st.session_state.wishlist
    st.button("★ Saved" if saved else "☆ Save", key=f"wl_{ctx}_{r['name']}", on_click=toggle_wish, args=(r["name"],))


# ---------- header ----------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning · Price Intelligence</div>
        <h1>SpecWorth</h1>
        <p>Know what your phone is worth. Estimate its market price in India, check whether an offer is a good deal, and find the best phones for your budget.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_deal, tab_phones, tab_insights, tab_models, tab_batch, tab_about = st.tabs(
    ["🔮 Predict", "💸 Deal check", "📱 Phones", "📊 Insights", "🏆 Models", "📂 Batch", "ℹ️ About"]
)

# ================= PREDICT =================
with tab_pred:
    sec("Quick start")
    p1, p2, p3 = st.columns(3)
    p1.button("💰 Budget", on_click=load_preset, args=("Budget",), use_container_width=True)
    p2.button("📱 Mid-range", on_click=load_preset, args=("Mid-range",), use_container_width=True)
    p3.button("🚀 Flagship", on_click=load_preset, args=("Flagship",), use_container_width=True)

    sec("Specifications")
    st.caption("Use the slider or type the exact value in the box next to it.")
    with st.expander("⚙️ Performance & Memory", expanded=True):
        for k in ["ram_gb", "storage_gb", "battery_mah", "charging_w", "antutu"]:
            spec_input(k)
        st.caption("Not sure about the AnTuTu score? Typical values: entry-level ~300k, mid-range ~700k-1M, "
                   "upper mid-range ~1.2-2M, flagship 2.5M+ (search '<phone name> AnTuTu').")
        for k in ["clock_speed", "n_cores", "talk_time"]:
            spec_input(k)

    with st.expander("🖥️ Display & Camera"):
        for k in ["px_height", "px_width", "sc_h", "sc_w"]:
            spec_input(k)
        diag = float(np.hypot(st.session_state["sc_h"], st.session_state["sc_w"]) / 2.54)
        st.caption(f"≈ {diag:.1f}-inch display (calculated from screen height and width)")
        for k in ["fc", "pc"]:
            spec_input(k)

    with st.expander("📡 Connectivity & Build"):
        for k in ["m_dep", "mobile_wt"]:
            spec_input(k)
        t1, t2 = st.columns(2)
        for i, (k, lab) in enumerate(TOGGLES.items()):
            (t1 if i % 2 == 0 else t2).toggle(lab, key=k)

    st.write("")
    if st.button("Predict Price", type="primary", use_container_width=True):
        S = current_specs()
        res = dict(S=S)
        res["cls"] = classify(S)
        if M is not None:
            row = core.market_row(S)
            est = core.predict_price(M, row)
            low, high = core.predict_interval(M, est)
            base, _, contrib = core.explain(M, row)
            contrib = contrib.sort_values(key=abs, ascending=False)
            sc = core.score_catalog(_cat, weights())
            near, near_note = core.phones_near(sc, est * 0.85, est * 1.15, est)
            diag_in = float(np.hypot(S["sc_h"], S["sc_w"]) / 2.54)
            spec_rows = [("RAM", f"{S['ram_gb']} GB"), ("Storage", f"{S['storage_gb']} GB"),
                         ("Battery", f"{S['battery_mah']} mAh"), ("Fast charging", f"{S['charging_w']} W"),
                         ("AnTuTu score", f"{S['antutu']:,}"), ("Rear camera", f"{S['pc']} MP"),
                         ("Front camera", f"{S['fc']} MP"), ("Display (approx.)", f'{diag_in:.1f}"'),
                         ("Weight", f"{S['mobile_wt']} g")]
            ph = [dict(name=r["name"], price=r["price_inr"], mem=f"{fmt(r['ram_gb'])} GB / {fmt(r['storage_gb'])} GB",
                       battery=fmt(r["battery_mah"], " mAh")) for _, r in near.iterrows()]
            note = (f"Gradient Boosting model trained on {M['n']} phones sold in India. {M['metrics']['method']}: "
                    f"error about {inr(M['metrics']['mae'])} on average (R2 {M['metrics']['r2']:.2f}). "
                    f"The range is an 80% conformal prediction interval.")
            res.update(est=est, low=low, high=high, base=base, contrib=contrib, near=near, near_note=near_note,
                       outside=core.out_of_range(M, row), pdf=pdf_report.build_estimate_report(
                           spec_rows, est, low, high, TIER_NAMES[tier(est)], base, contrib.to_dict(), ph, note))
            tier_name, est_text = TIER_NAMES[tier(est)], inr(round(est, -2))
        else:
            tier_name, est_text = LABELS[res["cls"][0]], "-"
        st.session_state["last"] = res
        st.session_state.history.append(
            {"RAM (GB)": S["ram_gb"], "Storage (GB)": S["storage_gb"], "Battery (mAh)": S["battery_mah"],
             "Est. price": est_text, "Segment": tier_name})

    res = st.session_state.get("last")
    if res:
        if M is not None and "est" in res:
            est, low, high = res["est"], res["low"], res["high"]
            t = tier(est)
            g1, g2 = GRAD[t]
            st.markdown(
                f"""
                <div class="result" style="background: linear-gradient(135deg, {g1}, {g2});">
                    <div class="eyebrow">Estimated market price</div>
                    <div class="price">{inr(round(est, -2))}</div>
                    <span class="badge">{TIER_NAMES[t]} segment</span>
                    <div class="sub">80% likely between {inr(round(low, -2))} and {inr(round(high, -2))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Gradient Boosting trained on {M['n']} phones sold in India · {M['metrics']['method']}: "
                       f"average error {inr(M['metrics']['mae'])}, R² {M['metrics']['r2']:.2f}. Indicative, not a quote.")
            st.download_button("📄 Download PDF report", res["pdf"], "specworth_estimate.pdf", "application/pdf",
                               use_container_width=True)
            if res["outside"]:
                st.info("Beyond the phones in the catalog, so the estimate treats them as the catalog's limit: "
                        + ", ".join(res["outside"]))

            sec("What drives this estimate")
            st.caption(f"A typical phone in the catalog is estimated at {inr(round(res['base'], -2))}. "
                       "Shapley values show how each spec moves your estimate from there.")
            st.bar_chart(res["contrib"])

            sec("Phones in India around this price")
            if res["near_note"]:
                st.caption(res["near_note"])
            for n, (_, r) in enumerate(res["near"].iterrows(), 1):
                phone_card(r, n, ctx="pred")
            st.caption("Ranked with the priorities set in the Phones tab. Prices are approximate (Oct 2026).")
        elif M is None:
            st.warning("Catalog not available - showing only the classification model from my report.")

        with st.expander("📘 Classification model from my report"):
            pred_c, probs_c, capped, raw, lo_c, hi_c = res["cls"]
            st.markdown(f"**Predicted segment: {LABELS[pred_c]}** · {probs_c[pred_c] * 100:.0f}% confidence")
            bars = "".join(
                f'<div class="pb"><span class="l">{LABELS[i]}</span><span class="t"><div style="width:{probs_c[i] * 100:.1f}%;'
                f'background:{GRAD[i][0]}"></div></span><span class="v">{probs_c[i] * 100:.1f}%</span></div>'
                for i in range(4)
            )
            st.markdown(bars, unsafe_allow_html=True)
            if capped:
                lines = []
                for c in capped:
                    v = raw.at[0, c]
                    lim = hi_c[c] if v > hi_c[c] else lo_c[c]
                    unit = 1024 if c == "ram" else 1
                    lines.append(f"{FRIENDLY.get(c, c)}: {v / unit:g} → {lim / unit:g}")
                st.info("This model was trained on an older phone dataset (RAM up to ~4 GB). "
                        "Values outside its range were capped, so modern phones tend to land in the top segment: "
                        + "; ".join(lines))
            else:
                st.caption("All inputs are within the range of the training data.")

    if st.session_state.history:
        sec("Prediction history (this session)")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        h1, h2 = st.columns(2)
        h1.download_button("⬇️ Download history", hist.to_csv(index=False), "prediction_history.csv",
                           "text/csv", use_container_width=True)
        if h2.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ================= DEAL CHECK =================
with tab_deal:
    if M is None:
        st.info("The phone catalog is needed for the deal checker (phones.csv).")
    else:
        st.subheader("Is this a good deal?")
        st.write("Enter the price you are being offered. The model compares it with what a phone with these specs usually costs.")
        mode = st.radio("Which phone?", ["Pick from the catalog", "My specs from the Predict tab"], horizontal=True)
        names = list(_cat["brand"].astype(str) + " " + _cat["model"].astype(str))
        chosen, note = None, ""
        if mode.startswith("Pick"):
            chosen = st.selectbox("Phone", names, index=0)
            hit = M["oof"][M["oof"]["name"] == chosen]
            if len(hit):
                est = float(hit["Predicted price"].iloc[0])
                note = "Estimated by a model that was not trained on this phone, so it is a fair test."
            else:
                est = core.predict_price(M, _cat[(_cat["brand"].astype(str) + " " + _cat["model"].astype(str)) == chosen]
                                         .rename(columns={}).iloc[0].reindex(MARKET_FEATS).to_dict())
            listed = float(_cat[(_cat["brand"].astype(str) + " " + _cat["model"].astype(str)) == chosen]["price_inr"].iloc[0])
            default_ask = int(listed)
        else:
            est = core.predict_price(M, core.market_row(current_specs()))
            default_ask = int(round(est, -2))
            note = "Uses the specs currently set in the Predict tab."
        low, high = core.predict_interval(M, est)
        asking = st.number_input("Price you are being offered (₹)", 1000, 500000, default_ask, step=500,
                                 key=f"ask_{mode}_{chosen}")
        label, level, diff = core.deal_verdict(asking, est, low, high)
        color = VERDICT_COLORS[level]
        sign = "above" if diff >= 0 else "below"
        st.markdown(
            f"""
            <div class="result" style="background:{color};">
                <div class="eyebrow">Verdict</div>
                <div class="price" style="font-size:2.1rem;">{label}</div>
                <div class="sub">{inr(asking)} is {abs(diff):.0f}% {sign} the estimate of {inr(round(est, -2))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lo_v, hi_v = min(low, asking) * 0.85, max(high, asking) * 1.15
        pos = lambda x: (x - lo_v) / (hi_v - lo_v) * 100
        st.markdown(
            f"""
            <div class="gauge">
              <div class="rng" style="left:{pos(low):.1f}%; width:{pos(high) - pos(low):.1f}%;"></div>
              <div class="est" style="left:{pos(est):.1f}%;"></div>
              <div class="dot" style="left:{pos(asking):.1f}%; background:{color};"></div>
              <div class="tag" style="left:{pos(low):.1f}%;">{inr(round(low, -2))}</div>
              <div class="tag" style="left:{pos(high):.1f}%;">{inr(round(high, -2))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Shaded = expected 80% price range, line = model estimate, dot = your offer. {note} "
                   f"Typical error is about {inr(M['metrics']['mae'])}, so treat this as guidance, not a rule.")

        sc = core.score_catalog(_cat, weights())
        pool = sc[sc["price_inr"] <= asking]
        if chosen is not None:
            mine = float(sc.loc[sc["name"] == chosen, "score"].iloc[0])
            pool = pool[(pool["name"] != chosen) & (pool["score"] > mine)]
        pool = pool.sort_values("score", ascending=False).head(3)
        sec("Better value for the same money")
        if pool.empty:
            st.caption("No other phone in the catalog at or below this price scores higher on your priorities.")
        for n, (_, r) in enumerate(pool.iterrows(), 1):
            phone_card(r, n, ctx="deal")

        st.markdown("---")
        st.subheader("EMI calculator")
        e1, e2 = st.columns(2)
        e_price = e1.number_input("Phone price (₹)", 1000, 500000, int(asking), step=500)
        e_down = e2.number_input("Down payment (₹)", 0, int(e_price), 0, step=500)
        e3, e4 = st.columns(2)
        e_rate = e3.number_input("Interest rate (% per year)", 0.0, 40.0, 14.0, step=0.5)
        e_months = e4.selectbox("Tenure (months)", [3, 6, 9, 12, 18, 24], index=3)
        r1 = core.emi(e_price, e_down, e_months, e_rate)
        k1, k2, k3 = st.columns(3)
        k1.metric("Monthly EMI", inr(r1["emi"]))
        k2.metric("Interest paid", inr(r1["interest"]))
        k3.metric("Total cost", inr(r1["total_cost"]))
        table = pd.DataFrame([
            {"Tenure": f"{m} months", "EMI": inr(core.emi(e_price, e_down, m, e_rate)["emi"]),
             "Interest": inr(core.emi(e_price, e_down, m, e_rate)["interest"]),
             "Total cost": inr(core.emi(e_price, e_down, m, e_rate)["total_cost"])} for m in [3, 6, 9, 12, 18, 24]])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("Set the rate to 0 for a no-cost EMI. Processing fees and taxes are not included.")

# ================= PHONES =================
with tab_phones:
    if _cat is None or _cat.empty or core.validate_catalog(_cat):
        st.info("phones.csv not found or invalid. Add a valid catalog next to app.py to enable this tab.")
    else:
        cat = _cat
        names_all = list(cat["brand"].astype(str) + " " + cat["model"].astype(str))
        t_rec, t_cmp, t_sim, t_val, t_wish, t_cat = st.tabs(
            ["⭐ Recommend", "⚖️ Compare", "🔁 Similar", "💎 Value", "❤️ Wishlist", "🗂️ Catalog"])

        with t_rec:
            st.write("Tell me your budget and what matters most - I'll rank phones available in India.")
            b_lo, b_hi = st.slider("Budget (₹)", 5000, 100000, (15000, 40000), step=1000)
            with st.expander("🔎 Filters"):
                brands = st.multiselect("Brand", sorted(cat["brand"].astype(str).unique()))
                f5g = st.toggle("5G only")
                f1, f2 = st.columns(2)
                min_ram = f1.select_slider("Minimum RAM (GB)", [0, 4, 6, 8, 12, 16], value=0)
                min_sto = f2.select_slider("Minimum storage (GB)", [0, 64, 128, 256, 512], value=0)
                min_bat = st.slider("Minimum battery (mAh)", 0, 10000, 0, step=500)
            sec("What matters to you? (0 = ignore, 5 = very important)")
            c1, c2 = st.columns(2)
            c1.slider("Performance", 0, 5, key="w_perf")
            c2.slider("Battery & charging", 0, 5, key="w_batt")
            c1.slider("Camera", 0, 5, key="w_cam")
            c2.slider("Display size", 0, 5, key="w_disp")
            c1.slider("Low price", 0, 5, key="w_price")

            scored = core.score_catalog(cat, weights())
            pool = scored[(scored["price_inr"] >= b_lo) & (scored["price_inr"] <= b_hi)]
            pool = core.apply_filters(pool, brands, f5g, min_ram, min_sto, min_bat).sort_values("score", ascending=False)
            if pool.empty:
                st.warning("No phones match. Widen the budget or relax the filters.")
            else:
                st.success(f"Best pick for you: {pool.iloc[0]['name']} ({inr(pool.iloc[0]['price_inr'])})")
                for n, (_, r) in enumerate(pool.head(3).iterrows(), 1):
                    phone_card(r, n, ctx="rec")
                if len(pool) > 3:
                    with st.expander(f"See the full ranking ({len(pool)} phones)"):
                        show = pool[["name", "price_inr", "score"]].copy()
                        show["price_inr"] = show["price_inr"].map(inr)
                        show["score"] = show["score"].round(1)
                        st.dataframe(show.rename(columns={"name": "Phone", "price_inr": "Price", "score": "Score"}),
                                     use_container_width=True, hide_index=True)
            st.caption("Score is spec-based: AnTuTu benchmark (where available), RAM, storage, battery, charging, camera MP, "
                       "display size and price. Megapixels do not measure camera quality, so read reviews before buying.")

        with t_cmp:
            picks = st.multiselect("Pick 2 or 3 phones to compare", names_all, default=names_all[:2],
                                   max_selections=3, key="cmp_picks")
            if len(picks) < 2:
                st.info("Select at least two phones.")
            else:
                raw_c = cat.assign(name=names_all).set_index("name").loc[picks]
                sc_c = core.score_catalog(cat, weights()).set_index("name").loc[picks]
                rows = {
                    "Price": raw_c["price_inr"].map(lambda v: "-" if pd.isna(v) else inr(v)),
                    "RAM": raw_c["ram_gb"].map(lambda v: fmt(v, " GB")),
                    "Storage": raw_c["storage_gb"].map(lambda v: fmt(v, " GB")),
                    "Chipset": raw_c["chipset"].astype(str),
                    "AnTuTu": raw_c["antutu"].map(lambda v: fmt(v, "", True)) if "antutu" in raw_c else "-",
                    "Battery": raw_c["battery_mah"].map(lambda v: fmt(v, " mAh")),
                    "Charging": raw_c["charging_w"].map(lambda v: fmt(v, " W")),
                    "Rear camera": raw_c["rear_mp"].map(lambda v: fmt(v, " MP")),
                    "Front camera": raw_c["front_mp"].map(lambda v: fmt(v, " MP")),
                    "Display": raw_c["display_in"].map(lambda v: fmt(v, '"')),
                    "Overall score": sc_c["score"].map(lambda v: f"{v:.0f}/100"),
                }
                comp_df = pd.DataFrame(rows).T
                st.dataframe(comp_df, use_container_width=True)

                best = []
                for lab, col, hi_better in [("Cheapest", "price_inr", False), ("Biggest battery", "battery_mah", True),
                                            ("Fastest charging", "charging_w", True), ("Most RAM", "ram_gb", True),
                                            ("Best benchmark", "antutu", True)]:
                    if col not in raw_c:
                        continue
                    s = pd.to_numeric(raw_c[col], errors="coerce").dropna()
                    if not s.empty:
                        best.append(f"<b>{lab}:</b> {escape(str(s.idxmax() if hi_better else s.idxmin()))}")
                st.markdown("  \n".join(b.replace("<b>", "**").replace("</b>", "**") for b in best))
                winner = sc_c["score"].idxmax()
                st.success(f"Highest overall score: {winner}")
                st.download_button(
                    "📄 Download comparison PDF",
                    pdf_report.build_compare_report(
                        [[i] + [str(v) for v in comp_df.loc[i].tolist()] for i in comp_df.index], picks,
                        best, winner),
                    "specworth_comparison.pdf", "application/pdf")

                sec("Score breakdown (0-100)")
                st.bar_chart((sc_c[["perf", "batt", "cam", "disp", "cheap"]] * 100).rename(
                    columns={"perf": "Performance", "batt": "Battery", "cam": "Camera", "disp": "Display",
                             "cheap": "Price"}).T)

        with t_sim:
            st.write("Pick a phone and see the three with the most similar specs (price is not considered).")
            base_name = st.selectbox("Phone", names_all, key="sim_pick")
            sims = core.similar_phones(cat, base_name, 3)
            sc_s = core.score_catalog(cat, weights()).set_index("name")
            for n, (_, r) in enumerate(sims.iterrows(), 1):
                rr = sc_s.loc[r["name"]].copy()
                rr["name"] = r["name"]
                phone_card(rr, n, ctx="sim", caption=f"{r['similarity']:.0f}% similar to {escape(base_name)}")
            st.caption("Cosine similarity on standardised RAM, storage, battery, charging, benchmark, display and cameras.")

        with t_val:
            st.write("Which phones give the most raw performance for the money? (AnTuTu points per ₹1,000)")
            vt = core.value_table(cat)
            chart = alt.Chart(vt).mark_circle(size=90, opacity=0.85).encode(
                x=alt.X("antutu:Q", title="AnTuTu score"), y=alt.Y("price_inr:Q", title="Price (₹)"),
                color=alt.Color("points_per_1000:Q", scale=alt.Scale(scheme="viridis"), title="Points per ₹1,000"),
                tooltip=["name", "price_inr", "antutu", "points_per_1000"]).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.caption("Top-left is better: higher performance for a lower price. Only phones with a benchmark score are shown.")
            top = vt[["name", "price_inr", "antutu", "points_per_1000"]].head(10).copy()
            top["price_inr"] = top["price_inr"].map(inr)
            top["antutu"] = top["antutu"].map(lambda v: f"{v:,.0f}")
            st.dataframe(top.rename(columns={"name": "Phone", "price_inr": "Price", "antutu": "AnTuTu",
                                             "points_per_1000": "Points per ₹1,000"}),
                         use_container_width=True, hide_index=True)

        with t_wish:
            wl = [n for n in st.session_state.wishlist if n in names_all]
            if not wl:
                st.info("Your wishlist is empty. Tap ☆ Save under any phone to add it here.")
            else:
                sc_w = core.score_catalog(cat, weights()).set_index("name")
                for n, name in enumerate(wl, 1):
                    rr = sc_w.loc[name].copy()
                    rr["name"] = name
                    phone_card(rr, n, ctx="wish")
                w1, w2 = st.columns(2)
                w1.download_button("⬇️ Download wishlist", cat[(cat["brand"].astype(str) + " " + cat["model"].astype(str))
                                                               .isin(wl)].to_csv(index=False), "wishlist.csv", "text/csv",
                                   use_container_width=True)
                if w2.button("🗑️ Clear wishlist", use_container_width=True):
                    st.session_state.wishlist = []
                    st.rerun()
                st.caption("Select up to 3 of these in the Compare tab to see them side by side.")

        with t_cat:
            st.write(f"**{len(cat)} phones in the catalog** - prices are approximate (Oct 2026) and vary by variant and seller.")
            st.dataframe(cat, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download catalog", cat.to_csv(index=False), "phones.csv", "text/csv", use_container_width=True)
            up_cat = d2.file_uploader("Replace with your CSV", type="csv", label_visibility="collapsed")
            if up_cat is not None:
                new = pd.read_csv(up_cat)
                problems = core.validate_catalog(new)
                if problems:
                    st.error("; ".join(problems))
                else:
                    st.session_state.catalog_override = new
                    st.rerun()
            if st.session_state.get("catalog_override") is not None:
                if st.button("↩️ Back to the built-in catalog"):
                    st.session_state.catalog_override = None
                    st.rerun()
            with st.expander("➕ Add a phone (this session)"):
                with st.form("add_phone", clear_on_submit=True):
                    f1, f2 = st.columns(2)
                    brand = f1.text_input("Brand")
                    model_name = f2.text_input("Model")
                    price = f1.number_input("Price (₹)", 1000, 500000, 20000, step=500)
                    chip = f2.text_input("Chipset")
                    ram_g = f1.number_input("RAM (GB)", 1, 24, 8)
                    sto_g = f2.number_input("Storage (GB)", 8, 1024, 128)
                    bat = f1.number_input("Battery (mAh)", 1000, 12000, 5000, step=100)
                    chg = f2.number_input("Charging (W)", 5, 240, 33)
                    rmp = f1.number_input("Rear camera (MP)", 1, 300, 50)
                    fmp = f2.number_input("Front camera (MP)", 1, 100, 16)
                    dsp = f1.number_input("Display (inch)", 4.0, 8.5, 6.7, step=0.01)
                    ant = f2.number_input("AnTuTu score (0 = unknown)", 0, 4000000, 0, step=50000)
                    g5 = f1.checkbox("5G", value=True)
                    if st.form_submit_button("Add phone") and brand and model_name:
                        row_new = pd.DataFrame([dict(
                            brand=brand, model=model_name, price_inr=price, ram_gb=ram_g, storage_gb=sto_g,
                            battery_mah=bat, charging_w=chg, rear_mp=rmp, front_mp=fmp, display_in=dsp,
                            chipset=chip or "-", antutu=ant if ant > 0 else np.nan, source="added by me",
                            five_g=int(g5), last_verified="session")])
                        st.session_state.catalog_override = pd.concat([cat, row_new], ignore_index=True)
                        st.rerun()
            st.caption("Sources: 91mobiles, mysmartprice and smartprix listings read in Oct 2026. Sites disagree on some prices - "
                       "verify on the brand's site or a retailer. The 5G flag is derived from the chipset/model name.")

# ================= INSIGHTS =================
with tab_insights:
    if M is not None:
        mt = M["metrics"]
        st.subheader("Market price model")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Phones", M["n"])
        k2.metric("R²", f"{mt['r2']:.2f}")
        k3.metric("MAE", inr(mt["mae"]))
        k4.metric("Avg error", f"{mt['mape']:.0f}%")
        st.caption(f"Scores come from {mt['method']}: every phone is predicted by a model that never saw it.")
        sec("Average impact of each spec on the price (Shapley)")
        st.bar_chart(M["global_impact"])
        st.caption("How many % the price typically moves because of each spec, compared with an average phone.")
        sec("Actual vs predicted price")
        o = M["oof"].rename(columns={"Actual price": "actual", "Predicted price": "predicted"})
        mx = float(max(o["actual"].max(), o["predicted"].max()))
        pts = alt.Chart(o).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("actual:Q", title="Actual price (₹)"), y=alt.Y("predicted:Q", title="Predicted price (₹)"),
            tooltip=["name", "actual", "predicted"])
        diag_line = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
        st.altair_chart(pts + diag_line, use_container_width=True)
        st.caption("Points close to the dashed line are accurate predictions.")
        st.markdown("---")
    st.subheader("Training data from my report")
    m1, m2, m3 = st.columns(3)
    m1.metric("Phones", len(df))
    m2.metric("Features", len(columns))
    m3.metric("Missing values", int(df.isna().sum().sum()))

    sec("Class distribution")
    st.bar_chart(df["price_range"].map(LABELS).value_counts())

    sec("Average specs per price class")
    avg = df.groupby("price_range")[["ram", "battery_power", "int_memory", "px_height", "px_width"]].mean().round(0)
    avg.index = [LABELS[i] for i in avg.index]
    st.dataframe(avg, use_container_width=True)

    sec("Correlation of each feature with price range")
    corr = df.corr()["price_range"].drop("price_range").sort_values(ascending=False)
    st.bar_chart(corr)
    st.caption("RAM is by far the strongest (~0.92); Bluetooth, WiFi, Dual SIM and Touch Screen are almost zero.")

    sec("Top 10 features in the Logistic Regression model")
    imp = pd.Series(abs(model.coef_).mean(axis=0), index=columns).sort_values(ascending=False).head(10)
    st.bar_chart(imp)

    with st.expander("🔎 Explore raw data"):
        st.dataframe(df.head(100), use_container_width=True)

# ================= MODELS =================
with tab_models:
    if M is not None:
        mt = M["metrics"]
        st.subheader("Market price model - model card")
        st.markdown(
            f"""
            - **Algorithm:** Gradient Boosting regression on log(price)
            - **Features:** {", ".join(MARKET_NAMES[f] for f in MARKET_FEATS)}
            - **Hyperparameters (GridSearchCV):** {", ".join(f"{k}={v}" for k, v in M["params"].items() if k != "random_state")}
            - **Evaluation:** {mt['method']} · R² {mt['r2']:.2f} · MAE {inr(mt['mae'])} · {mt['mape']:.0f}% average error
            - **Prediction interval:** 80% conformal interval (± x{np.exp(M['conformal_q']):.2f}) from out-of-sample residuals
            - **Trained on:** {M['n']} phones · built {M.get('built_at', '-')} · saved with joblib
            """
        )
        if "default_loo" in mt:
            st.caption(f"Tuning helped only a little (untuned model: R² {mt['default_loo']['r2']:.2f}, MAE "
                       f"{inr(mt['default_loo']['mae'])}): with 54 phones, the data is the limit, not the algorithm.")
        if "quantile_coverage" in mt:
            st.caption(f"Interval check: conformal intervals covered {mt['conformal_coverage'] * 100:.0f}% of phones "
                       f"(target 80%). Quantile regression covered only {mt['quantile_coverage'] * 100:.0f}%, "
                       "so conformal prediction is used.")
        st.markdown("---")

    st.subheader("Classification models from my report (20% test set)")
    comp = pd.Series(T["scores"], name="Accuracy (%)").round(2).sort_values(ascending=False)
    st.bar_chart(comp)
    st.success(f"Best model: {comp.index[0]} ({comp.iloc[0]:.2f}% accuracy) - used for the segment prediction in this app.")

    sec("Cross-validation (5-fold, training split)")
    st.dataframe(cv_report(), use_container_width=True, hide_index=True)

    sec("Hyperparameter tuning (GridSearchCV)")
    st.caption("Searches C for Logistic Regression, trees/depth for Random Forest and neighbours/weights/distance for KNN.")
    if st.button("▶️ Run tuning (about a minute)"):
        with st.spinner("Running GridSearchCV..."):
            st.session_state["tuning"] = run_tuning()
    if st.session_state.get("tuning") is not None:
        st.dataframe(st.session_state["tuning"], use_container_width=True, hide_index=True)

    pick = st.selectbox("Inspect a model", list(T["models"].keys()))
    lab_names = [LABELS[i] for i in range(4)]
    cm = pd.DataFrame(confusion_matrix(T["y_test"], T["preds"][pick]),
                      index=[f"Actual {n}" for n in lab_names], columns=[f"Pred {n}" for n in lab_names])
    sec("Confusion matrix")
    st.dataframe(cm, use_container_width=True)
    rep = pd.DataFrame(classification_report(T["y_test"], T["preds"][pick], output_dict=True)).T.round(2)
    sec("Classification report")
    st.dataframe(rep, use_container_width=True)

# ================= BATCH =================
with tab_batch:
    kind = st.radio("What do you want to do?", ["Estimate prices (market model)", "Classify segments (report model)"],
                    horizontal=True)
    if kind.startswith("Estimate"):
        st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in MARKET_FEATS) + ".")
        tmpl = pd.DataFrame([{"ram_gb": 8, "storage_gb": 128, "battery_mah": 5500, "charging_w": 45, "rear_mp": 50,
                              "front_mp": 16, "antutu": 800000},
                             {"ram_gb": 12, "storage_gb": 256, "battery_mah": 5500, "charging_w": 80, "rear_mp": 50,
                              "front_mp": 32, "antutu": 3000000}])
        st.download_button("⬇️ Download CSV template", tmpl.to_csv(index=False), "price_template.csv", "text/csv")
        up_m = st.file_uploader("Upload CSV", type="csv", key="batch_market")
        if up_m is not None:
            data_m = pd.read_csv(up_m)
            miss = [c for c in MARKET_FEATS if c not in data_m.columns]
            if M is None:
                st.error("Price model not available.")
            elif miss:
                st.error("Missing columns: " + ", ".join(miss))
            else:
                est_b = core.predict_prices(M, data_m)
                f_ = np.exp(M["conformal_q"])
                out_m = data_m.copy()
                out_m["estimated_price"] = np.round(est_b, -2)
                out_m["range_low"] = np.round(est_b / f_, -2)
                out_m["range_high"] = np.round(est_b * f_, -2)
                out_m["segment"] = [TIER_NAMES[tier(p)] for p in est_b]
                st.success(f"Estimated {len(out_m)} phones")
                st.dataframe(out_m, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download estimates", out_m.to_csv(index=False), "price_estimates.csv",
                                   "text/csv", type="primary")
    else:
        st.write("Uses the classification model from my report. Upload a CSV with the 20 dataset columns "
                 "(RAM in MB, as in the original dataset). Values outside the training range are capped.")
        template = pd.DataFrame([core.ui_to_dataset(p) for p in PRESETS.values()])[columns]
        st.download_button("⬇️ Download CSV template", template.to_csv(index=False), "template.csv", "text/csv")
        up = st.file_uploader("Upload CSV", type="csv", key="batch_class")
        if up is not None:
            data = pd.read_csv(up)
            missing = [c for c in columns if c not in data.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                lo_b, hi_b = df[columns].min(), df[columns].max()
                Xb, _ = core.clip_to_range(data[columns], lo_b, hi_b)
                pb = model.predict_proba(scaler.transform(Xb))
                out = data.copy()
                out["predicted_class"] = pb.argmax(axis=1)
                out["segment"] = out["predicted_class"].map(LABELS)
                out["confidence_%"] = (pb.max(axis=1) * 100).round(1)
                st.success(f"Classified {len(out)} phones")
                st.dataframe(out[["segment", "confidence_%"] + columns], use_container_width=True, hide_index=True)
                st.bar_chart(out["segment"].value_counts())
                st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                                   "predictions.csv", "text/csv", type="primary")

# ================= ABOUT =================
with tab_about:
    st.markdown(
        """
        **SpecWorth**
        Built during my 6-week Machine Learning training at Winnovation, then extended for the Indian market.

        **Two models**
        1. **Market price model (regression):** Gradient Boosting trained on a hand-collected catalog of phones sold in India
           (`phones.csv`). Features: RAM, storage, battery, charging, cameras and the AnTuTu processor benchmark.
           Tuned with GridSearchCV, evaluated with nested cross-validation, saved with joblib. Predictions come with an
           80% conformal interval and Shapley-value explanations.
        2. **Classification model (from my report):** Logistic Regression on the Kaggle Mobile Price Classification dataset
           (2000 phones, 20 features, 4 classes), compared against Random Forest and KNN.

        **Why two?** The Kaggle dataset only has price classes (0-3) and older phones, so it cannot give a price in ₹
        for a modern phone. The market model does that; the classifier is kept to show the full project.

        **Other tools:** deal checker, EMI calculator, filters, similar phones, value-for-money chart, wishlist,
        PDF reports and batch estimates.

        **Limits:** the catalog has about 54 phones with approximate Oct 2026 prices, so estimates are indicative
        (see error figures in the Insights and Models tabs). The phone ranking is a separate, transparent spec-based score.

        **Tech stack:** Python · Pandas · Scikit-learn · Altair · ReportLab · Streamlit

        Made by **Rishi Mishra** - B.Tech CSE, BPIT Rohini
        """
    )
