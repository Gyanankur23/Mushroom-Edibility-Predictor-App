import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time

st.set_page_config(page_title="Mushroom ML Predictor", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button:hover {
        background-color: #004080;
        color: white;
    }
    .metric-card {
        background: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
        box-shadow: 2px 2px 5px #d9d9d9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Mushroom Edibility Prediction Dashboard")

data = {
    "cap_shape": ["bell", "conical", "convex", "flat", "sunken"],
    "cap_color": ["brown", "yellow", "white", "red", "gray"],
    "odor": ["almond", "anise", "none", "foul", "fishy"],
    "edible": [1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df.drop("edible", axis=1)
y = df["edible"]
X_encoded = pd.get_dummies(X)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_encoded, y)

st.sidebar.header("Select Mushroom Features")
cap_shape = st.sidebar.selectbox("Cap Shape", df["cap_shape"].unique())
cap_color = st.sidebar.selectbox("Cap Color", df["cap_color"].unique())
odor = st.sidebar.selectbox("Odor", df["odor"].unique())

input_df = pd.DataFrame([[cap_shape, cap_color, odor]], columns=["cap_shape", "cap_color", "odor"])
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

prediction = model.predict(input_encoded)[0]
prob = model.predict_proba(input_encoded)[0][prediction]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>Selected Shape</h4><p>{cap_shape}</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>Selected Color</h4><p>{cap_color}</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>Selected Odor</h4><p>{odor}</p></div>", unsafe_allow_html=True)

if prediction == 1:
    st.success(f"Predicted: Edible (Confidence {prob:.2f})")
else:
    st.error(f"Predicted: Poisonous (Confidence {prob:.2f})")

placeholder = st.empty()
for i in range(30):
    noise = np.random.randn(len(X_encoded)) * 0.05
    fig, ax = plt.subplots()
    ax.plot(range(len(X_encoded)), model.predict_proba(X_encoded)[:,1] + noise, color="blue", label="Edibility Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    placeholder.pyplot(fig)
    time.sleep(0.1)
