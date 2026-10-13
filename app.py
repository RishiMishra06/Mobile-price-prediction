import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mobile Price Range Predictor", page_icon="📱", layout="wide")

LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}


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
    return model, scaler, list(X.columns), acc


try:
    model, scaler, columns, acc = train_model()
except FileNotFoundError:
    st.error("train.csv nahi mila. Dataset ko app.py ke same folder me rakho.")
    st.stop()


# ---------- sidebar ----------
with st.sidebar:
    st.title("About")
    st.write(
        "Ye app mobile ke specifications dekh kar batata hai ki phone "
        "kis price range (0-3) me aayega."
    )
    st.write("**Model:** Logistic Regression")
    st.write(f"**Test accuracy:** {acc * 100:.2f}%")
    st.write("**Dataset:** Mobile Price Classification (2000 phones)")
    st.markdown("---")
    st.caption("Winnovation ML Training Project - Rishi Mishra, BPIT")


# ---------- main page ----------
st.title("📱 Mobile Price Range Prediction")
st.write("Phone ke specs bharo aur 'Predict' dabao.")

tab1, tab2, tab3 = st.tabs(["Performance & Memory", "Display & Camera", "Connectivity & Others"])

with tab1:
    c1, c2 = st.columns(2)
    battery_power = c1.slider("Battery Power (mAh)", 500, 2000, 1200)
    ram = c2.slider("RAM (MB)", 256, 4000, 2000)
    int_memory = c1.slider("Internal Memory (GB)", 2, 64, 32)
    clock_speed = c2.slider("Processor Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
    n_cores = c1.slider("Number of Processor Cores", 1, 8, 4)
    talk_time = c2.slider("Talk Time (hours)", 2, 20, 10)

with tab2:
    c1, c2 = st.columns(2)
    px_height = c1.number_input("Pixel Resolution Height", 0, 2000, 900)
    px_width = c2.number_input("Pixel Resolution Width", 500, 2000, 1200)
    sc_h = c1.slider("Screen Height (cm)", 5, 19, 12)
    sc_w = c2.slider("Screen Width (cm)", 0, 18, 6)
    fc = c1.slider("Front Camera (MP)", 0, 19, 5)
    pc = c2.slider("Primary Camera (MP)", 0, 20, 10)

with tab3:
    c1, c2 = st.columns(2)
    m_dep = c1.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)
    mobile_wt = c2.slider("Mobile Weight (g)", 80, 200, 140)
    blue = c1.checkbox("Bluetooth")
    wifi = c1.checkbox("WiFi")
    dual_sim = c1.checkbox("Dual SIM")
    three_g = c2.checkbox("3G", value=True)
    four_g = c2.checkbox("4G")
    touch_screen = c2.checkbox("Touch Screen", value=True)

st.markdown("---")

if st.button("Predict Price Range", type="primary"):
    values = {
        "battery_power": battery_power, "blue": int(blue), "clock_speed": clock_speed,
        "dual_sim": int(dual_sim), "fc": fc, "four_g": int(four_g),
        "int_memory": int_memory, "m_dep": m_dep, "mobile_wt": mobile_wt,
        "n_cores": n_cores, "pc": pc, "px_height": px_height, "px_width": px_width,
        "ram": ram, "sc_h": sc_h, "sc_w": sc_w, "talk_time": talk_time,
        "three_g": int(three_g), "touch_screen": int(touch_screen), "wifi": int(wifi),
    }
    # column order must match training data
    input_df = pd.DataFrame([values])[columns]
    scaled = scaler.transform(input_df)

    pred = int(model.predict(scaled)[0])
    probs = model.predict_proba(scaled)[0]

    st.subheader("Result")
    st.success(f"Predicted Price Range: **{pred} - {LABELS[pred]}**")

    st.write("Confidence for each class:")
    prob_df = pd.DataFrame({"Probability (%)": probs * 100}, index=[LABELS[i] for i in range(4)])
    st.bar_chart(prob_df)
