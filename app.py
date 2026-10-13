from html import escape

import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
GRAD = {0: ("#16a34a", "#15803d"), 1: ("#2563eb", "#1d4ed8"), 2: ("#d97706", "#b45309"), 3: ("#dc2626", "#b91c1c")}
DEFAULT_BANDS = {0: (5000, 10000), 1: (10000, 20000), 2: (20000, 35000), 3: (35000, 90000)}
BOOL_COLS = ["blue", "wifi", "dual_sim", "three_g", "four_g", "touch_screen"]

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
    .result .sub {margin-top: .7rem; font-size: .85rem; opacity: .9;}

    .pb {display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: .85rem;}
    .pb .l {width: 112px; opacity: .85;}
    .pb .t {flex: 1; height: 8px; border-radius: 99px; background: rgba(128,128,128,.2); overflow: hidden;}
    .pb .t > div {height: 100%; border-radius: 99px;}
    .pb .v {width: 50px; text-align: right; font-weight: 600;}

    .pc {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .85rem 1rem; margin: .55rem 0;
         background: rgba(128,128,128,.05);}
    .pc .top {display: flex; justify-content: space-between; align-items: baseline; gap: 10px;}
    .pc .nm {font-weight: 700; font-size: .98rem;}
    .pc .pr {font-weight: 800; color: #6366f1; white-space: nowrap;}
    .pc .chip {display: inline-block; padding: 2px 9px; border-radius: 999px; background: rgba(99,102,241,.13);
               font-size: .73rem; margin: 7px 5px 0 0; font-weight: 500;}
    .pc .bar {height: 6px; border-radius: 99px; background: rgba(128,128,128,.2); overflow: hidden; margin-top: 10px;}
    .pc .bar > div {height: 100%; background: linear-gradient(90deg, #6366f1, #a855f7);}
    .pc .why {font-size: .75rem; opacity: .68; margin-top: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def inr(n):
    """Format a number in Indian style, e.g. 125000 -> ₹1,25,000"""
    s = str(int(round(n)))
    if len(s) <= 3:
        return "₹" + s
    rest = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3])
    return "₹" + rest + "," + s[-3:]


# --- catalog helpers start ---
NUM_COLS = ["price_inr", "ram_gb", "storage_gb", "battery_mah", "charging_w", "rear_mp", "front_mp", "display_in", "antutu"]
DIM_NAMES = {"perf": "performance", "batt": "battery & charging", "cam": "camera MP",
             "disp": "display size", "cheap": "low price"}
W_KEYS = {"perf": "w_perf", "batt": "w_batt", "cam": "w_cam", "disp": "w_disp", "cheap": "w_price"}


@st.cache_data
def load_catalog():
    try:
        return pd.read_csv("phones.csv")
    except FileNotFoundError:
        return pd.DataFrame()


def get_catalog():
    cat = st.session_state.get("catalog_override")
    return cat if cat is not None else load_catalog()


def fmt(v, suffix="", comma=False):
    if pd.isna(v):
        return "-"
    return (f"{v:,.0f}" if comma else f"{v:g}") + suffix


def score_catalog(cat):
    """Spec-based score. Missing values are filled with the column median ONLY for scoring;
    the raw values are kept for display."""
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

    w = {k: st.session_state.get(v, 3) for k, v in W_KEYS.items()}
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
    d["name"] = d["brand"] + " " + d["model"]
    return d
# --- catalog helpers end ---


# ---------- training (same pipeline as the report) ----------
@st.cache_resource
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

    return dict(df=df, columns=list(X.columns), scaler=scaler, models=models,
                scores=scores, preds=preds, y_test=y_test)


try:
    T = train_all()
except FileNotFoundError:
    st.error("train.csv not found. Keep the dataset in the same folder as app.py.")
    st.stop()

df, columns, scaler = T["df"], T["columns"], T["scaler"]
model = T["models"]["Logistic Regression"]

# ---------- presets & state ----------
PRESETS = {
    "Budget": dict(battery_power=900, ram=600, int_memory=8, clock_speed=1.0, n_cores=2, talk_time=8,
                   px_height=400, px_width=700, sc_h=9, sc_w=4, fc=1, pc=3, m_dep=0.6, mobile_wt=170,
                   blue=False, wifi=False, dual_sim=False, three_g=True, four_g=False, touch_screen=False),
    "Mid-range": dict(battery_power=1300, ram=2000, int_memory=32, clock_speed=1.5, n_cores=4, talk_time=12,
                      px_height=800, px_width=1200, sc_h=12, sc_w=6, fc=5, pc=10, m_dep=0.5, mobile_wt=140,
                      blue=True, wifi=True, dual_sim=True, three_g=True, four_g=False, touch_screen=True),
    "Flagship": dict(battery_power=1800, ram=3600, int_memory=64, clock_speed=2.5, n_cores=8, talk_time=18,
                     px_height=1500, px_width=1800, sc_h=16, sc_w=9, fc=12, pc=18, m_dep=0.3, mobile_wt=120,
                     blue=True, wifi=True, dual_sim=True, three_g=True, four_g=True, touch_screen=True),
}

if "battery_power" not in st.session_state:
    st.session_state.update(PRESETS["Mid-range"])
for i, (lo, hi) in DEFAULT_BANDS.items():
    st.session_state.setdefault(f"lo{i}", lo)
    st.session_state.setdefault(f"hi{i}", hi)
st.session_state.setdefault("history", [])
for _k in W_KEYS.values():
    st.session_state.setdefault(_k, 3)


def load_preset(name):
    st.session_state.update(PRESETS[name])


def bands():
    return {i: (st.session_state[f"lo{i}"], st.session_state[f"hi{i}"]) for i in range(4)}


def estimate_price(probs):
    """Probability-weighted mid-point of the price bands."""
    b = bands()
    return sum(probs[i] * (b[i][0] + b[i][1]) / 2 for i in range(4))


def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def phone_card(r, rank=None):
    chips = [f"{fmt(r['ram_gb'])} GB RAM", f"{fmt(r['storage_gb'])} GB storage", f"{fmt(r['battery_mah'])} mAh",
             f"{fmt(r['charging_w'])} W", escape(str(r["chipset"]))]
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    pct = min(max(float(r["score"]), 0), 100)
    prefix = f"#{rank} " if rank else ""
    st.markdown(
        f"""
        <div class="pc">
          <div class="top"><span class="nm">{prefix}{escape(str(r['name']))}</span><span class="pr">{inr(r['price_inr'])}</span></div>
          <div>{chips_html}</div>
          <div class="bar"><div style="width:{pct:.0f}%"></div></div>
          <div class="why">Match score {pct:.0f}/100 · strong in {escape(str(r['why']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def phones_near(lo, hi, est):
    cat = get_catalog()
    if cat is None or cat.empty:
        return None, ""
    d = score_catalog(cat)
    m = d[(d["price_inr"] >= lo) & (d["price_inr"] <= hi)]
    note = ""
    if m.empty:
        m = d.iloc[(d["price_inr"] - est).abs().argsort()[:3]]
        note = "No phone in the catalog falls inside this band, showing the closest by price."
    return m.sort_values("score", ascending=False).head(3), note


# ---------- header ----------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning · Price Intelligence</div>
        <h1>Mobile Price Predictor</h1>
        <p>Enter a phone's specifications to predict its price segment and estimated price, then find the best phones in India for that budget.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_phones, tab_batch, tab_insights, tab_models, tab_about = st.tabs(
    ["🔮 Predict", "📱 Phones", "📂 Batch", "📊 Insights", "🏆 Models", "ℹ️ About"]
)

# ================= PREDICT =================
with tab_pred:
    sec("Quick start")
    p1, p2, p3 = st.columns(3)
    p1.button("💰 Budget", on_click=load_preset, args=("Budget",), use_container_width=True)
    p2.button("📱 Mid-range", on_click=load_preset, args=("Mid-range",), use_container_width=True)
    p3.button("🚀 Flagship", on_click=load_preset, args=("Flagship",), use_container_width=True)

    sec("Specifications")
    with st.expander("⚙️ Performance & Memory", expanded=True):
        c1, c2 = st.columns(2)
        c1.slider("Battery Power (mAh)", 500, 2000, key="battery_power")
        c2.slider("RAM (MB)", 256, 4000, key="ram")
        c1.slider("Internal Memory (GB)", 2, 64, key="int_memory")
        c2.slider("Clock Speed (GHz)", 0.5, 3.0, step=0.1, key="clock_speed")
        c1.slider("Processor Cores", 1, 8, key="n_cores")
        c2.slider("Talk Time (hours)", 2, 20, key="talk_time")

    with st.expander("🖥️ Display & Camera"):
        c1, c2 = st.columns(2)
        c1.slider("Pixel Height", 0, 2000, key="px_height")
        c2.slider("Pixel Width", 500, 2000, key="px_width")
        c1.slider("Screen Height (cm)", 5, 19, key="sc_h")
        c2.slider("Screen Width (cm)", 0, 18, key="sc_w")
        c1.slider("Front Camera (MP)", 0, 19, key="fc")
        c2.slider("Primary Camera (MP)", 0, 20, key="pc")

    with st.expander("📡 Connectivity & Build"):
        c1, c2 = st.columns(2)
        c1.slider("Depth (cm)", 0.1, 1.0, step=0.1, key="m_dep")
        c2.slider("Weight (g)", 80, 200, key="mobile_wt")
        c1.toggle("Bluetooth", key="blue")
        c1.toggle("WiFi", key="wifi")
        c1.toggle("Dual SIM", key="dual_sim")
        c2.toggle("3G", key="three_g")
        c2.toggle("4G", key="four_g")
        c2.toggle("Touch Screen", key="touch_screen")

    with st.expander("💱 Price band settings (₹)"):
        st.caption("The training dataset has price classes 0-3, not real prices. "
                   "These ₹ bands are an assumption for each class - edit them to match the market you want.")
        for i in range(4):
            a, b = st.columns(2)
            a.number_input(f"{LABELS[i]} - min (₹)", 0, 500000, step=1000, key=f"lo{i}")
            b.number_input(f"{LABELS[i]} - max (₹)", 0, 500000, step=1000, key=f"hi{i}")

    st.write("")
    if st.button("Predict Price", type="primary", use_container_width=True):
        values = {c: st.session_state[c] for c in columns}
        for k in BOOL_COLS:
            values[k] = int(values[k])
        row = pd.DataFrame([values])[columns]
        scaled = scaler.transform(row)

        pred = int(model.predict(scaled)[0])
        probs = model.predict_proba(scaled)[0]
        est = estimate_price(probs)
        lo, hi = bands()[pred]
        g1, g2 = GRAD[pred]

        st.markdown(
            f"""
            <div class="result" style="background: linear-gradient(135deg, {g1}, {g2});">
                <div class="eyebrow">Estimated price</div>
                <div class="price">{inr(est)}</div>
                <span class="badge">{LABELS[pred]}</span>
                <div class="sub">Likely range {inr(lo)} - {inr(hi)} · {probs[pred] * 100:.0f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Indicative estimate: probability-weighted mid-point of the ₹ bands set above.")

        sec("Class probabilities")
        bars = "".join(
            f'<div class="pb"><span class="l">{LABELS[i]}</span><span class="t"><div style="width:{probs[i] * 100:.1f}%;'
            f'background:{GRAD[i][0]}"></div></span><span class="v">{probs[i] * 100:.1f}%</span></div>'
            for i in range(4)
        )
        st.markdown(bars, unsafe_allow_html=True)

        near, near_note = phones_near(lo, hi, est)
        if near is not None:
            sec("Phones in India around this price")
            if near_note:
                st.caption(near_note)
            for n, (_, r) in enumerate(near.iterrows(), 1):
                phone_card(r, n)
            st.caption("Ranked with the priorities set in the Phones tab. Prices are approximate (Oct 2026).")

        sec("Why this prediction?")
        contrib = pd.Series(scaled[0] * model.coef_[pred], index=columns).sort_values(key=abs, ascending=False).head(6)
        st.bar_chart(contrib)
        st.caption("Positive = pushes towards the predicted class, negative = pushes away.")

        st.session_state.history.append(
            {"RAM": values["ram"], "Battery": values["battery_power"], "Storage": values["int_memory"],
             "Class": LABELS[pred], "Est. price": inr(est), "Confidence %": round(probs[pred] * 100, 1)}
        )

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

# ================= PHONES =================
with tab_phones:
    cat = get_catalog()
    if cat is None or cat.empty:
        st.info("phones.csv not found. Add it next to app.py to enable recommendations and comparison.")
    else:
        t_rec, t_cmp, t_cat = st.tabs(["⭐ Recommend", "⚖️ Compare", "🗂️ Catalog"])

        with t_rec:
            st.write("Tell me your budget and what matters most - I'll rank phones available in India.")
            b_lo, b_hi = st.slider("Budget (₹)", 5000, 100000, (15000, 40000), step=1000)
            sec("What matters to you? (0 = ignore, 5 = very important)")
            c1, c2 = st.columns(2)
            c1.slider("Performance", 0, 5, key="w_perf")
            c2.slider("Battery & charging", 0, 5, key="w_batt")
            c1.slider("Camera", 0, 5, key="w_cam")
            c2.slider("Display size", 0, 5, key="w_disp")
            c1.slider("Low price", 0, 5, key="w_price")

            scored = score_catalog(cat)
            pool = scored[(scored["price_inr"] >= b_lo) & (scored["price_inr"] <= b_hi)].sort_values("score", ascending=False)
            if pool.empty:
                st.warning("No phones in this budget. Widen the range or add phones in the Catalog tab.")
            else:
                st.success(f"Best pick for you: {pool.iloc[0]['name']} ({inr(pool.iloc[0]['price_inr'])})")
                for n, (_, r) in enumerate(pool.head(3).iterrows(), 1):
                    phone_card(r, n)
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
            names = cat["brand"] + " " + cat["model"]
            picks = st.multiselect("Pick 2 or 3 phones to compare", list(names), default=list(names[:2]), max_selections=3)
            if len(picks) < 2:
                st.info("Select at least two phones.")
            else:
                raw = cat.assign(name=names).set_index("name").loc[picks]
                sc = score_catalog(cat).set_index("name").loc[picks]
                rows = {
                    "Price": raw["price_inr"].map(lambda v: "-" if pd.isna(v) else inr(v)),
                    "RAM": raw["ram_gb"].map(lambda v: fmt(v, " GB")),
                    "Storage": raw["storage_gb"].map(lambda v: fmt(v, " GB")),
                    "Chipset": raw["chipset"].astype(str),
                    "AnTuTu": raw["antutu"].map(lambda v: fmt(v, "", True)) if "antutu" in raw else "-",
                    "Battery": raw["battery_mah"].map(lambda v: fmt(v, " mAh")),
                    "Charging": raw["charging_w"].map(lambda v: fmt(v, " W")),
                    "Rear camera": raw["rear_mp"].map(lambda v: fmt(v, " MP")),
                    "Front camera": raw["front_mp"].map(lambda v: fmt(v, " MP")),
                    "Display": raw["display_in"].map(lambda v: fmt(v, '"')),
                    "Overall score": sc["score"].map(lambda v: f"{v:.0f}/100"),
                }
                st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

                best = []
                for label, col, hi_better in [("Cheapest", "price_inr", False), ("Biggest battery", "battery_mah", True),
                                              ("Fastest charging", "charging_w", True), ("Most RAM", "ram_gb", True),
                                              ("Best benchmark", "antutu", True)]:
                    if col not in raw:
                        continue
                    s = pd.to_numeric(raw[col], errors="coerce").dropna()
                    if not s.empty:
                        best.append(f"**{label}:** {(s.idxmax() if hi_better else s.idxmin())}")
                st.markdown("  \n".join(best))
                st.success(f"Highest overall score: {sc['score'].idxmax()}")

                sec("Score breakdown (0-100)")
                st.bar_chart((sc[["perf", "batt", "cam", "disp", "cheap"]] * 100).rename(
                    columns={"perf": "Performance", "batt": "Battery", "cam": "Camera", "disp": "Display", "cheap": "Price"}).T)

        with t_cat:
            st.write(f"**{len(cat)} phones in the catalog** - prices are approximate (Oct 2026) and vary by variant and seller.")
            st.dataframe(cat, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download catalog", cat.to_csv(index=False), "phones.csv", "text/csv", use_container_width=True)
            up_cat = d2.file_uploader("Replace with your CSV", type="csv", label_visibility="collapsed")
            if up_cat is not None:
                new = pd.read_csv(up_cat)
                need = ["brand", "model", "price_inr", "ram_gb", "storage_gb", "battery_mah", "charging_w",
                        "rear_mp", "front_mp", "display_in", "chipset"]
                miss = [c for c in need if c not in new.columns]
                if miss:
                    st.error("Missing columns: " + ", ".join(miss))
                else:
                    st.session_state.catalog_override = new
                    st.rerun()
            with st.expander("➕ Add a phone (this session)"):
                with st.form("add_phone", clear_on_submit=True):
                    f1, f2 = st.columns(2)
                    brand = f1.text_input("Brand")
                    model_name = f2.text_input("Model")
                    price = f1.number_input("Price (₹)", 0, 500000, 20000, step=500)
                    chip = f2.text_input("Chipset")
                    ram_g = f1.number_input("RAM (GB)", 1, 24, 8)
                    sto_g = f2.number_input("Storage (GB)", 8, 1024, 128)
                    bat = f1.number_input("Battery (mAh)", 1000, 12000, 5000, step=100)
                    chg = f2.number_input("Charging (W)", 5, 240, 33)
                    rmp = f1.number_input("Rear camera (MP)", 1, 300, 50)
                    fmp = f2.number_input("Front camera (MP)", 1, 100, 16)
                    dsp = f1.number_input("Display (inch)", 4.0, 8.5, 6.7, step=0.01)
                    if st.form_submit_button("Add phone") and brand and model_name:
                        row = pd.DataFrame([dict(brand=brand, model=model_name, price_inr=price, ram_gb=ram_g,
                                                 storage_gb=sto_g, battery_mah=bat, charging_w=chg, rear_mp=rmp,
                                                 front_mp=fmp, display_in=dsp, chipset=chip or "-", source="added by me")])
                        st.session_state.catalog_override = pd.concat([cat, row], ignore_index=True)
                        st.rerun()
            st.caption("Sources: 91mobiles, mysmartprice and smartprix listings read in Oct 2026. Sites disagree on some prices - "
                       "verify on the brand's site or a retailer, and edit phones.csv in your repo to keep it current.")

# ================= BATCH =================
with tab_batch:
    st.subheader("Predict many phones at once")
    st.write("Upload a CSV with the 20 feature columns. Download the template to see the format.")
    template = pd.DataFrame(
        [{k: int(v) if k in BOOL_COLS else v for k, v in p.items()} for p in PRESETS.values()]
    )[columns]
    st.download_button("⬇️ Download CSV template", template.to_csv(index=False), "template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in columns if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            Xb = scaler.transform(data[columns])
            pb = model.predict_proba(Xb)
            out = data.copy()
            out["predicted_class"] = pb.argmax(axis=1)
            out["segment"] = out["predicted_class"].map(LABELS)
            out["estimated_price"] = [inr(estimate_price(p)) for p in pb]
            out["confidence_%"] = (pb.max(axis=1) * 100).round(1)
            st.success(f"Predicted {len(out)} phones")
            st.dataframe(out[["segment", "estimated_price", "confidence_%"] + columns],
                         use_container_width=True, hide_index=True)
            st.bar_chart(out["segment"].value_counts())
            st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv", type="primary")

# ================= INSIGHTS =================
with tab_insights:
    st.subheader("Dataset overview")
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
    st.subheader("Model comparison (live, on the 20% test set)")
    comp = pd.Series(T["scores"], name="Accuracy (%)").round(2).sort_values(ascending=False)
    st.bar_chart(comp)
    best = comp.index[0]
    st.success(f"Best model: {best} ({comp.iloc[0]:.2f}% accuracy) - used for all predictions in this app.")

    pick = st.selectbox("Inspect a model", list(T["models"].keys()))
    names = [LABELS[i] for i in range(4)]
    cm = pd.DataFrame(confusion_matrix(T["y_test"], T["preds"][pick]),
                      index=[f"Actual {n}" for n in names], columns=[f"Pred {n}" for n in names])
    sec("Confusion matrix")
    st.dataframe(cm, use_container_width=True)
    rep = pd.DataFrame(classification_report(T["y_test"], T["preds"][pick], output_dict=True)).T.round(2)
    sec("Classification report")
    st.dataframe(rep, use_container_width=True)

# ================= ABOUT =================
with tab_about:
    st.markdown(
        """
        **Mobile Price Predictor**
        Built during my 6-week Machine Learning training at Winnovation.

        **Pipeline**
        1. Mobile Price Classification dataset (2000 phones, 20 features, 4 classes)
        2. EDA: box plot, violin plot, correlation heatmap, pair plot
        3. 80/20 train-test split and StandardScaler
        4. Compared Logistic Regression, Random Forest and KNN
        5. Deployed with Streamlit

        **About the ₹ estimate:** the dataset contains price *classes* (0-3), not actual prices.
        The ₹ figure maps each class to an editable price band and weights it by the model's class probabilities,
        so it is an indicative estimate, not a true regression output.

        **Phone recommendations:** a hand-collected catalog of phones sold in India (`phones.csv`, approximate Oct 2026 prices)
        is ranked with a transparent spec-based score using the priorities you choose. It is separate from the ML model,
        which was trained on a different dataset.

        **Tech stack:** Python · Pandas · Scikit-learn · Streamlit

        Made by **Rishi Mishra** - B.Tech CSE, BPIT Rohini
        """
    )
