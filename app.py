import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mushroom ML Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --bg-gradient-start: #e8f4ff;
        --bg-gradient-end: #f7fbff;
        --card-bg: rgba(255,255,255,0.85);
        --accent: #0b6fa4;
        --muted: #6b7280;
    }
    .stApp {
        background: linear-gradient(180deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .header {
        padding: 18px 24px;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(11,111,164,0.12), rgba(11,111,164,0.06));
        margin-bottom: 18px;
    }
    .metric {
        background: var(--card-bg);
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(11,111,164,0.06);
        text-align: center;
    }
    .metric h3 { margin: 0; color: var(--accent); }
    .metric p { margin: 4px 0 0 0; color: var(--muted); font-size: 13px; }
    .control-box {
        background: rgba(255,255,255,0.9);
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }
    .footer-note { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .plotly-graph-div .modebar { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"><h1 style="margin:0">Mushroom Edibility — Production Ready Demo</h1><div class="footer-note">A compact, deployable ML dashboard demonstrating data, model, and interactive visualizations</div></div>', unsafe_allow_html=True)

# Synthetic dataset generation (realistic combinatorial expansion)
shapes = ["bell", "conical", "convex", "flat", "sunken"]
colors = ["brown", "yellow", "white", "red", "gray", "buff", "cinnamon"]
odors = ["almond", "anise", "none", "foul", "fishy", "spicy", "musty"]
habitats = ["woods", "grasses", "paths", "urban", "meadows"]
gill_sizes = ["narrow", "broad"]
spore_colors = ["brown", "black", "white", "purple", "green"]

np.random.seed(42)
rows = 1200
data = {
    "cap_shape": np.random.choice(shapes, rows, p=[0.15,0.15,0.25,0.3,0.15]),
    "cap_color": np.random.choice(colors, rows),
    "odor": np.random.choice(odors, rows, p=[0.12,0.08,0.5,0.12,0.06,0.06,0.06]),
    "habitat": np.random.choice(habitats, rows),
    "gill_size": np.random.choice(gill_sizes, rows, p=[0.6,0.4]),
    "spore_color": np.random.choice(spore_colors, rows, p=[0.4,0.2,0.2,0.1,0.1])
}
df = pd.DataFrame(data)

# Create a synthetic target with plausible rules + noise
def synth_label(row):
    score = 0
    if row["odor"] in ["foul", "fishy", "musty"]: score -= 2
    if row["odor"] in ["almond", "anise"]: score += 2
    if row["cap_color"] in ["red", "gray"]: score -= 0.5
    if row["habitat"] == "woods": score += 0.3
    if row["spore_color"] in ["black", "purple"]: score += 0.2
    if row["gill_size"] == "broad": score += 0.1
    score += np.random.normal(0, 0.6)
    return 1 if score > 0.2 else 0

df["edible"] = df.apply(synth_label, axis=1)

# Feature engineering and encoding
X = pd.get_dummies(df.drop(columns=["edible"]), drop_first=False)
y = df["edible"]

# Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
model.fit(X_train, y_train)

# Cross-validated score for headline metric
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
cv_mean = cv_scores.mean()

# Predictions and metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
report = classification_report(y_test, y_pred, output_dict=True)

# Sidebar controls
st.sidebar.markdown('<div class="control-box"><h4 style="margin:0">Interactive Controls</h4></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## Input features")
    sel_shape = st.selectbox("Cap shape", sorted(df["cap_shape"].unique()), index=0)
    sel_color = st.selectbox("Cap color", sorted(df["cap_color"].unique()), index=0)
    sel_odor = st.selectbox("Odor", sorted(df["odor"].unique()), index=2)
    sel_habitat = st.selectbox("Habitat", sorted(df["habitat"].unique()), index=0)
    sel_gill = st.selectbox("Gill size", sorted(df["gill_size"].unique()), index=0)
    sel_spore = st.selectbox("Spore color", sorted(df["spore_color"].unique()), index=0)
    st.markdown("---")
    st.markdown("## Visualization options")
    show_roc = st.checkbox("Show ROC curve", value=True)
    show_confusion = st.checkbox("Show Confusion Matrix", value=True)
    show_feature_imp = st.checkbox("Show Feature Importance", value=True)
    sample_size = st.slider("Sample preview size", 5, 50, 12)

# Prepare single input and prediction
input_row = pd.DataFrame([{
    "cap_shape": sel_shape,
    "cap_color": sel_color,
    "odor": sel_odor,
    "habitat": sel_habitat,
    "gill_size": sel_gill,
    "spore_color": sel_spore
}])
input_encoded = pd.get_dummies(input_row)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
pred = model.predict(input_encoded)[0]
pred_proba = model.predict_proba(input_encoded)[0][1]

# Top metrics row
kpi1, kpi2, kpi3, kpi4 = st.columns([1.2,1.2,1.2,1.2])
with kpi1:
    st.markdown(f'<div class="metric"><h3>{cv_mean:.2f}</h3><p>Cross-validated accuracy</p></div>', unsafe_allow_html=True)
with kpi2:
    edible_pct = df["edible"].mean()
    st.markdown(f'<div class="metric"><h3>{edible_pct:.2%}</h3><p>Training edible ratio</p></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric"><h3>{roc_auc:.2f}</h3><p>ROC AUC (test)</p></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric"><h3>{len(df):,}</h3><p>Samples synthesized</p></div>', unsafe_allow_html=True)

# Main layout: left column for prediction and input, right column for charts
left, right = st.columns([1,2])

with left:
    st.subheader("Predict a Mushroom")
    st.markdown("Provide feature values and get a model prediction with probability.")
    st.table(input_row.T.rename(columns={0:"Value"}))
    if pred == 1:
        st.success(f"Model prediction: Edible (probability {pred_proba:.2f})")
    else:
        st.error(f"Model prediction: Poisonous (probability {1-pred_proba:.2f})")
    st.markdown("---")
    st.subheader("Model diagnostics")
    st.markdown("Classification report (test set)")
    cr_df = pd.DataFrame(report).T.round(2)
    st.dataframe(cr_df.style.format("{:.2f}"), height=220)

with right:
    st.subheader("Dashboard Visualizations")
    # Distribution pie
    fig_pie = px.pie(df, names="edible", title="Training Distribution: Edible vs Poisonous",
                     color="edible", color_discrete_map={1:"#0b6fa4", 0:"#d9534f"})
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Feature importance (top 12)
    if show_feature_imp:
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
        fig_imp = px.bar(importances[::-1], orientation="h", title="Top 12 Feature Importances", labels={'value':'importance','index':'feature'})
        st.plotly_chart(fig_imp, use_container_width=True)

    # Scatter: probability distribution vs selected odor
    st.markdown("Probability distribution by odor (test set)")
    test_df = X_test.copy()
    test_df["prob"] = model.predict_proba(X_test)[:,1]
    # map back odor column from dummies
    odor_cols = [c for c in X.columns if c.startswith("odor_")]
    if odor_cols:
        def extract_cat(row, prefix="odor_"):
            for c in odor_cols:
                if row.get(c,0) == 1:
                    return c.replace(prefix,"")
            return "unknown"
        test_df_reset = test_df.reset_index(drop=True)
        test_df_reset["odor"] = test_df_reset.apply(lambda r: extract_cat(r), axis=1)
        fig_box = px.box(test_df_reset, x="odor", y="prob", title="Predicted Edibility Probability by Odor", color="odor")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# Lower section: confusion matrix and ROC
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    if show_confusion:
        st.subheader("Confusion Matrix (test)")
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=["Pred Poisonous","Pred Edible"],
            y=["Actual Poisonous","Actual Edible"],
            colorscale="Blues",
            zmin=0, zmax=1,
            hovertemplate="Value: %{z:.2f}<extra></extra>"
        ))
        fig_cm.update_layout(margin=dict(t=30,b=10,l=10,r=10), height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

with col_b:
    if show_roc:
        st.subheader("ROC Curve (test)")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}", line=dict(color="#0b6fa4", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="#999999", dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380, margin=dict(t=30,b=10,l=10,r=10))
        st.plotly_chart(fig_roc, use_container_width=True)

# Sample preview and interactive table
st.markdown("---")
st.subheader("Sample Preview & Exploration")
sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)
st.dataframe(sample_df, height=220)

# Explainability: show top rules via partial dependence-like simple view
st.markdown("---")
st.subheader("Quick Explainability View")
top_feats = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(6)
explain_rows = []
for feat in top_feats.index:
    base = feat.split("_",1)
    if len(base) == 2:
        col, val = base
        explain_rows.append({"feature": col, "value": val, "importance": top_feats[feat]})
explain_df = pd.DataFrame(explain_rows)
if not explain_df.empty:
    fig_ex = px.bar(explain_df, x="importance", y="feature", color="value", orientation="h", title="Top feature-value contributions (approx.)")
    st.plotly_chart(fig_ex, use_container_width=True)
else:
    st.write("No categorical feature breakdown available.")

st.markdown("---")
st.markdown("This dashboard demonstrates a deployable pipeline: synthetic data generation, feature engineering, model training, evaluation, and interactive visualizations — all in a single Streamlit app.", unsafe_allow_html=True)
