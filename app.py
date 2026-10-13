import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
COLORS = {0: "#22c55e", 1: "#3b82f6", 2: "#f59e0b", 3: "#ef4444"}
ICONS = {0: "💚", 1: "💙", 2: "🧡", 3: "🔥"}

# ---------- styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 760px;}
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 1.6rem 1.4rem; border-radius: 18px; color: white; margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 1.7rem; color: white;}
    .hero p {margin: .4rem 0 0 0; opacity: .92; font-size: .95rem;}
    .chip {
        display: inline-block; background: rgba(255,255,255,.2); padding: 3px 11px;
        border-radius: 20px; font-size: .78rem; margin: 10px 6px 0 0;
    }
    .result {
        border-radius: 18px; padding: 1.4rem; text-align: center; color: white; margin: .6rem 0 1rem 0;
    }
    .result .small {font-size: .85rem; opacity: .9; letter-spacing: .08em; text-transform: uppercase;}
    .result .big {font-size: 2rem; font-weight: 700; margin-top: .2rem;}
    div.stButton > button {border-radius: 12px; font-weight: 600; padding: .6rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- model (same pipeline as in the report) ----------
@st.cache_resource
def train_model():
    df = pd.read_csv("train.csv")
    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, list(X.columns), acc, len(df)


try:
    model, scaler, columns, acc, n_rows = train_model()
except FileNotFoundError:
    st.error("train.csv not found. Keep the dataset in the same folder as app.py.")
    st.stop()


# ---------- presets ----------
PRESETS = {
    "Budget": dict(battery_power=900, ram=600, int_memory=8, clock_speed=1.0, n_cores=2,
                   talk_time=8, px_height=400, px_width=700, sc_h=9, sc_w=4, fc=1, pc=3,
                   m_dep=0.6, mobile_wt=170, blue=False, wifi=False, dual_sim=False,
                   three_g=True, four_g=False, touch_screen=False),
    "Mid-range": dict(battery_power=1300, ram=2000, int_memory=32, clock_speed=1.5, n_cores=4,
                      talk_time=12, px_height=800, px_width=1200, sc_h=12, sc_w=6, fc=5, pc=10,
                      m_dep=0.5, mobile_wt=140, blue=True, wifi=True, dual_sim=True,
                      three_g=True, four_g=False, touch_screen=True),
    "Flagship": dict(battery_power=1800, ram=3600, int_memory=64, clock_speed=2.5, n_cores=8,
                     talk_time=18, px_height=1500, px_width=1800, sc_h=16, sc_w=9, fc=12, pc=18,
                     m_dep=0.3, mobile_wt=120, blue=True, wifi=True, dual_sim=True,
                     three_g=True, four_g=True, touch_screen=True),
}

if "battery_power" not in st.session_state:
    st.session_state.update(PRESETS["Mid-range"])


def load_preset(name):
    st.session_state.update(PRESETS[name])


# ---------- header ----------
st.markdown(
    f"""
    <div class="hero">
        <h1>📱 Mobile Price Range Predictor</h1>
        <p>Enter a phone's specifications and the model predicts which price segment it belongs to.</p>
        <span class="chip">Logistic Regression</span>
        <span class="chip">{acc * 100:.1f}% test accuracy</span>
        <span class="chip">{n_rows} phones · 20 features</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

# ---------- predict tab ----------
with tab_predict:
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

    if st.button("Predict Price Range", type="primary", use_container_width=True):
        values = {col: st.session_state[col] for col in columns}
        for k in ["blue", "wifi", "dual_sim", "three_g", "four_g", "touch_screen"]:
            values[k] = int(values[k])

        input_df = pd.DataFrame([values])[columns]  # keep training column order
        scaled = scaler.transform(input_df)
        pred = int(model.predict(scaled)[0])
        probs = model.predict_proba(scaled)[0]

        st.markdown(
            f"""
            <div class="result" style="background: {COLORS[pred]};">
                <div class="small">Predicted price range</div>
                <div class="big">{ICONS[pred]} {LABELS[pred]}</div>
                <div class="small" style="margin-top:.4rem;">Class {pred} · {probs[pred] * 100:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Probability for each class**")
        for i in range(4):
            st.progress(float(probs[i]), text=f"{LABELS[i]} - {probs[i] * 100:.1f}%")

# ---------- insights tab ----------
with tab_insights:
    st.subheader("What drives the price?")
    st.caption("Average absolute coefficient of the trained model (features are scaled, so they are comparable).")
    importance = pd.Series(abs(model.coef_).mean(axis=0), index=columns).sort_values(ascending=False)
    st.bar_chart(importance.head(10))
    st.write(
        "RAM dominates by a huge margin, followed by battery power and screen resolution. "
        "Bluetooth, WiFi, Dual SIM and Touch Screen barely matter."
    )

    st.subheader("Model comparison")
    comp = pd.DataFrame(
        {"Accuracy (%)": [97.5, 89.25, 53.0]},
        index=["Logistic Regression", "Random Forest", "KNN"],
    )
    st.bar_chart(comp)
    st.caption("Results from my training project report. Logistic Regression was selected as the final model.")

# ---------- about tab ----------
with tab_about:
    st.markdown(
        """
        **Mobile Price Range Prediction**
        Built during my 6-week Machine Learning training at Winnovation.

        **Pipeline**
        1. Mobile Price Classification dataset (2000 phones, 20 features, 4 classes)
        2. EDA: box plot, violin plot, correlation heatmap, pair plot
        3. 80/20 train-test split, StandardScaler
        4. Compared Logistic Regression, Random Forest and KNN
        5. Deployed with Streamlit

        **Tech stack:** Python · Pandas · Scikit-learn · Streamlit

        Made by **Rishi Mishra** - B.Tech CSE, BPIT Rohini
        """
    )
