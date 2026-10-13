import re
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
COLORS = {0: "#22c55e", 1: "#3b82f6", 2: "#f59e0b", 3: "#ef4444"}
ICONS = {0: "💚", 1: "💙", 2: "🧡", 3: "🔥"}
DEFAULT_BANDS = {0: (5000, 10000), 1: (10000, 20000), 2: (20000, 35000), 3: (35000, 60000)}
BOOL_COLS = ["blue", "wifi", "dual_sim", "three_g", "four_g", "touch_screen"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 780px;}
    .hero {background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
           padding: 1.6rem 1.4rem; border-radius: 18px; color: white; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 1.7rem; color: white;}
    .hero p {margin: .4rem 0 0 0; opacity: .92; font-size: .95rem;}
    .chip {display: inline-block; background: rgba(255,255,255,.2); padding: 3px 11px;
           border-radius: 20px; font-size: .78rem; margin: 10px 6px 0 0;}
    .result {border-radius: 18px; padding: 1.4rem; text-align: center; color: white; margin: .6rem 0 1rem 0;}
    .result .small {font-size: .82rem; opacity: .92; letter-spacing: .08em; text-transform: uppercase;}
    .result .price {font-size: 2.3rem; font-weight: 800; margin: .15rem 0;}
    .result .band {font-size: 1rem; font-weight: 600;}
    div.stButton > button {border-radius: 12px; font-weight: 600; padding: .6rem 1rem;}
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
NUM_COLS = ["price_inr", "ram_gb", "storage_gb", "battery_mah", "charging_w", "rear_mp", "front_mp", "display_in"]
DIM_NAMES = {"perf": "performance (RAM & storage)", "batt": "battery & charging",
             "cam": "camera (MP-based)", "disp": "display size", "cheap": "lower price"}
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


def fmt(v, suffix=""):
    return "-" if pd.isna(v) else f"{v:g}{suffix}"


def score_catalog(cat):
    """Spec-based score. Missing values are filled with the column median."""
    d = cat.copy()
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(d[c].median()).fillna(0)

    def nz(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0 + 0.5

    d["perf"] = 0.6 * nz(d["ram_gb"]) + 0.4 * nz(d["storage_gb"])
    d["batt"] = 0.7 * nz(d["battery_mah"]) + 0.3 * nz(d["charging_w"])
    d["cam"] = 0.5 * nz(d["rear_mp"]) + 0.5 * nz(d["front_mp"])
    d["disp"] = nz(d["display_in"])
    d["cheap"] = 1 - nz(d["price_inr"])
    w = {k: st.session_state.get(v, 3) for k, v in W_KEYS.items()}
    tot = sum(w.values()) or 1
    d["score"] = 100 * sum(w[k] * d[k] for k in w) / tot
    d["why"] = d.apply(
        lambda r: " & ".join(DIM_NAMES[k] for k in sorted(w, key=lambda k: -w[k] * r[k])[:2]), axis=1
    )
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
acc = T["scores"]["Logistic Regression"]

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



def phone_card(r, rank=None):
    with st.container(border=True):
        title = f"**{'#' + str(rank) + ' ' if rank else ''}{r['name']}** · {inr(r['price_inr'])}"
        st.markdown(title)
        st.caption(f"{fmt(r['ram_gb'], 'GB')} RAM / {fmt(r['storage_gb'], 'GB')} · {fmt(r['battery_mah'], ' mAh')} · "
                   f"{fmt(r['charging_w'], 'W')} · {r['chipset']}")
        st.progress(min(max(float(r["score"]) / 100, 0.0), 1.0), text=f"Match score {r['score']:.0f}/100 · strong in {r['why']}")


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
    f"""
    <div class="hero">
        <h1>📱 Mobile Price Predictor</h1>
        <p>Enter a phone's specifications and get its price segment with an estimated price in ₹.</p>
        <span class="chip">Logistic Regression</span>
        <span class="chip">{acc:.1f}% test accuracy</span>
        <span class="chip">{len(df)} phones · 20 features</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_phones, tab_batch, tab_insights, tab_models, tab_about = st.tabs(
    ["🔮 Predict", "📱 Phones", "📂 Batch", "📊 Insights", "🏆 Models", "ℹ️ About"]
)

# ================= PREDICT =================
with tab_pred:
    st.caption("Quick start - load a sample phone, then tweak the specs:")
    p1, p2, p3 = st.columns(3)
    p1.button("💰 Budget", on_click=load_preset, args=("Budget",), use_container_width=True)
    p2.button("📱 Mid-range", on_click=load_preset, args=("Mid-range",), use_container_width=True)
    p3.button("🚀 Flagship", on_click=load_preset, args=("Flagship",), use_container_width=True)

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
        st.caption(
            "The dataset only has price classes 0-3, not real prices. "
            "These ₹ bands are my assumption for each class - edit them to match the market you want."
        )
        for i in range(4):
            a, b = st.columns(2)
            a.number_input(f"{LABELS[i]} - min (₹)", 0, 500000, step=1000, key=f"lo{i}")
            b.number_input(f"{LABELS[i]} - max (₹)", 0, 500000, step=1000, key=f"hi{i}")

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

        st.markdown(
            f"""
            <div class="result" style="background:{COLORS[pred]};">
                <div class="small">Estimated price</div>
                <div class="price">{inr(est)}</div>
                <div class="band">{ICONS[pred]} {LABELS[pred]} · likely {inr(lo)} - {inr(hi)}</div>
                <div class="small" style="margin-top:.5rem;">Class {pred} · {probs[pred]*100:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Indicative estimate: probability-weighted mid-point of the ₹ bands set above.")

        st.write("**Probability for each class**")
        for i in range(4):
            st.progress(float(probs[i]), text=f"{LABELS[i]} - {probs[i]*100:.1f}%")

        near, near_note = phones_near(lo, hi, est)
        if near is not None:
            st.write("**📱 Phones available in India around this price**")
            if near_note:
                st.caption(near_note)
            for n, (_, r) in enumerate(near.iterrows(), 1):
                phone_card(r, n)
            st.caption("Ranked with the priorities set in the Phones tab. Prices are approximate (Oct 2026).")

        st.write("**Why this prediction?** Features pushing the phone towards this class")
        contrib = pd.Series(scaled[0] * model.coef_[pred], index=columns).sort_values(key=abs, ascending=False).head(6)
        st.bar_chart(contrib)
        st.caption("Positive = pushes towards the predicted class, negative = pushes away (scaled coefficient × scaled value).")

        st.session_state.history.append(
            {"RAM": values["ram"], "Battery": values["battery_power"], "Storage": values["int_memory"],
             "Class": LABELS[pred], "Est. price": inr(est), "Confidence %": round(probs[pred] * 100, 1)}
        )

    if st.session_state.history:
        st.markdown("---")
        st.write("**Prediction history (this session)**")
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
            b_lo, b_hi = st.slider("Budget (₹)", 10000, 60000, (15000, 40000), step=1000)
            c1, c2 = st.columns(2)
            c1.slider("Performance", 0, 5, key="w_perf")
            c2.slider("Battery & charging", 0, 5, key="w_batt")
            c1.slider("Camera", 0, 5, key="w_cam")
            c2.slider("Display size", 0, 5, key="w_disp")
            c1.slider("Lower price", 0, 5, key="w_price")

            scored = score_catalog(cat)
            pool = scored[(scored["price_inr"] >= b_lo) & (scored["price_inr"] <= b_hi)].sort_values("score", ascending=False)
            if pool.empty:
                st.warning("No phones in this budget. Widen the range or add phones in the Catalog tab.")
            else:
                st.success(f"Best pick for you: {pool.iloc[0]['name']} ({inr(pool.iloc[0]['price_inr'])})")
                for n, (_, r) in enumerate(pool.head(3).iterrows(), 1):
                    phone_card(r, n)
                if len(pool) > 3:
                    with st.expander("See the full ranking"):
                        show = pool[["name", "price_inr", "score"]].copy()
                        show["price_inr"] = show["price_inr"].map(inr)
                        show["score"] = show["score"].round(1)
                        st.dataframe(show.rename(columns={"name": "Phone", "price_inr": "Price", "score": "Score"}),
                                     use_container_width=True, hide_index=True)
            st.caption("Score is spec-based (RAM, storage, battery, charging, camera MP, display size, price). "
                       "Megapixels do not measure camera quality, so read reviews before buying.")

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
                    "Battery": raw["battery_mah"].map(lambda v: fmt(v, " mAh")),
                    "Charging": raw["charging_w"].map(lambda v: fmt(v, " W")),
                    "Rear camera": raw["rear_mp"].map(lambda v: fmt(v, " MP")),
                    "Front camera": raw["front_mp"].map(lambda v: fmt(v, " MP")),
                    "Display": raw["display_in"].map(lambda v: fmt(v, '"')),
                    "Chipset": raw["chipset"].astype(str),
                    "Overall score": sc["score"].map(lambda v: f"{v:.0f}/100"),
                }
                st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

                best = []
                for label, col, hi_better in [("Cheapest", "price_inr", False), ("Biggest battery", "battery_mah", True),
                                              ("Fastest charging", "charging_w", True), ("Most RAM", "ram_gb", True)]:
                    s = pd.to_numeric(raw[col], errors="coerce").dropna()
                    if not s.empty:
                        best.append(f"**{label}:** {(s.idxmax() if hi_better else s.idxmin())}")
                st.markdown(" · ".join(best))
                st.success(f"Highest overall score: {sc['score'].idxmax()}")

                st.write("**Score breakdown (0-100)**")
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
                need = ["brand", "model", "price_inr"] + NUM_COLS[1:] + ["chipset"]
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

    st.write("**Class distribution**")
    st.bar_chart(df["price_range"].map(LABELS).value_counts())

    st.write("**Average specs per price class**")
    avg = df.groupby("price_range")[["ram", "battery_power", "int_memory", "px_height", "px_width"]].mean().round(0)
    avg.index = [LABELS[i] for i in avg.index]
    st.dataframe(avg, use_container_width=True)

    st.write("**Correlation of each feature with price range**")
    corr = df.corr()["price_range"].drop("price_range").sort_values(ascending=False)
    st.bar_chart(corr)
    st.caption("RAM is by far the strongest (~0.92); Bluetooth, WiFi, Dual SIM and Touch Screen are almost zero.")

    st.write("**Top 10 features in the Logistic Regression model**")
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
    st.success(f"Best model: {best} ({comp.iloc[0]:.2f}%)")

    pick = st.selectbox("Inspect a model", list(T["models"].keys()))
    names = [LABELS[i] for i in range(4)]
    cm = pd.DataFrame(confusion_matrix(T["y_test"], T["preds"][pick]),
                      index=[f"Actual {n}" for n in names], columns=[f"Pred {n}" for n in names])
    st.write("**Confusion matrix**")
    st.dataframe(cm, use_container_width=True)
    rep = pd.DataFrame(classification_report(T["y_test"], T["preds"][pick], output_dict=True)).T.round(2)
    st.write("**Classification report**")
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

        **Phone recommendations:** a hand-collected catalog of phones sold in India (`phones.csv`, approximate Oct 2026 prices) is ranked with a transparent spec-based score using the priorities you choose. It is separate from the ML model, which was trained on a different dataset.

        **Tech stack:** Python · Pandas · Scikit-learn · Streamlit

        Made by **Rishi Mishra** - B.Tech CSE, BPIT Rohini
        """
    )
