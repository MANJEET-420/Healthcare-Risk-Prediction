from pathlib import Path
import json
import re
import html

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="MediScope AI | Healthcare Analytics", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#08111f; --panel:#101d30; --line:#263951; --text:#eef5ff; --muted:#9aadc4; --teal:#43e0c2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 15% 0%,#142943 0%,#08111f 45%,#070d18 100%); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] { background:#0a1423; border-right:1px solid #203148; }
section[data-testid="stSidebar"] * { color:#eaf2ff; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
h1 { font-weight:800 !important; }
p,label,.stMarkdown { color:#d5e1f0; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#14253a,#0e1a2b); border:1px solid #2a405a; padding:18px 20px; border-radius:18px; box-shadow:0 8px 25px #00000018; }
div[data-testid="stMetricLabel"] { color:#9aadc4 !important; font-size:.84rem; }
div[data-testid="stMetricValue"] { color:#f4f8ff !important; font-family:'Manrope',sans-serif; }
.stButton>button,.stDownloadButton>button { border-radius:12px; border:1px solid #3cdbc0; background:linear-gradient(100deg,#28bfa7,#48ddc1); color:#06221f; font-weight:800; min-height:2.8rem; }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:#a2ffed; color:#041916; filter:brightness(1.06); }
div[data-baseweb="input"] input,div[data-baseweb="select"]>div,textarea { background:#0b1727 !important; color:#f1f6ff !important; border-color:#30455e !important; border-radius:10px !important; }
div[data-testid="stTabs"] button { color:#b8c8dc; font-weight:700; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#43e0c2; border-bottom-color:#43e0c2; }
hr { border-color:#263951; }
.hero { background:linear-gradient(120deg,#132941 0%,#102338 52%,#123c42 100%); border:1px solid #2c5363; border-radius:24px; padding:28px 30px; margin:8px 0 24px; position:relative; overflow:hidden; }
.hero:after { content:'✚'; position:absolute; right:42px; top:-30px; font-size:180px; color:#43e0c2; opacity:.055; }
.eyebrow { color:#43e0c2; font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.18em; margin-bottom:10px; }
.hero h1 { margin:0; color:#f5f9ff; font-size:2.25rem; }
.hero p { color:#b9cadc; margin:.6rem 0 0; max-width:760px; }
.section-head { display:flex; align-items:center; gap:10px; margin:22px 0 12px; }
.section-head span { color:#43e0c2; font-size:1.2rem; }
.section-head h3 { margin:0; color:#f1f6ff; font-size:1.15rem; }
.panel { background:linear-gradient(145deg,#122238,#0e1a2b); border:1px solid #273d56; border-radius:18px; padding:20px; height:100%; }
.panel-label { color:#92a9c3; text-transform:uppercase; font-size:.72rem; font-weight:800; letter-spacing:.12em; }
.panel-value { color:#f2f7ff; font-size:1.35rem; font-weight:800; margin-top:8px; font-family:'Manrope',sans-serif; }
.panel-note,.small-muted { color:#91a6bf; font-size:.82rem; }
.pill { display:inline-block; border:1px solid #2d766d; background:#10352f; color:#69f0d7; padding:5px 10px; border-radius:100px; font-size:.74rem; font-weight:800; }
.notice { border:1px solid #7a6335; background:#2b2518; color:#f4dba4; padding:13px 16px; border-radius:12px; font-size:.88rem; }
.result-good { border:1px solid #277b69; background:linear-gradient(120deg,#10352f,#102a2c); border-radius:18px; padding:22px; }
.result-caution { border:1px solid #9b7640; background:linear-gradient(120deg,#382b19,#2b241b); border-radius:18px; padding:22px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

model_path = ARTIFACTS / "risk_model.joblib"
features_path = ARTIFACTS / "feature_names.json"
metrics_path = ARTIFACTS / "metrics.json"

@st.cache_resource
def load_artifacts():
    if not (model_path.exists() and features_path.exists()):
        return None, None, {}
    model = joblib.load(model_path)
    features = json.loads(features_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return model, features, metrics

model, feature_names, metrics = load_artifacts()

with st.sidebar:
    st.markdown("## 🩺 MediScope **AI**")
    st.caption("HEALTHCARE ANALYTICS LAB")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Overview", "Risk Prediction", "Medical Text Lab", "Model Evaluation"])
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    st.markdown(f'<span class="pill">{"MODEL READY" if model is not None else "MODEL NOT TRAINED"}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Educational prototype • v1.0")
    st.caption("Not a medical device or diagnostic tool.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI • MACHINE LEARNING • NLP</div>
  <h1>Healthcare Intelligence Hub</h1>
  <p>Explore model-based risk patterns, inspect medical text, and review evaluation metrics in one clean workspace.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ <b>Educational use only.</b> Outputs are not a diagnosis or treatment recommendation. Do not use them to make medical decisions or delay professional care.</div>', unsafe_allow_html=True)
st.write("")

def section(title, icon="✦"):
    st.markdown(f'<div class="section-head"><span>{icon}</span><h3>{title}</h3></div>', unsafe_allow_html=True)

def metric_value(key, default="—"):
    results = metrics.get("results", {})
    selected = metrics.get("best_model")
    return results.get(selected, {}).get(key, default) if selected else default

def make_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix":"%", "font":{"size":35,"color":"#f2f7ff"}},
        title={"text":"Model-estimated positive-class probability","font":{"size":13,"color":"#9aadc4"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#91a6bf","tickfont":{"color":"#91a6bf"}},
               "bar":{"color":"#43e0c2","thickness":.25},"bgcolor":"#101d30","borderwidth":0,
               "steps":[{"range":[0,35],"color":"#153a36"},{"range":[35,70],"color":"#3b3422"},{"range":[70,100],"color":"#42262d"}]}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=270, margin=dict(l=15,r=15,t=45,b=10))
    return fig

if page == "Overview":
    section("Workspace overview", "◈")
    c1,c2,c3,c4 = st.columns(4)
    acc, auc = metric_value("accuracy"), metric_value("roc_auc")
    c1.metric("Input features", len(feature_names) if feature_names else 0)
    c2.metric("Selected model", metrics.get("best_model", "Not trained"))
    c3.metric("Test accuracy", f"{acc:.1%}" if isinstance(acc,(int,float)) else "—")
    c4.metric("Test ROC-AUC", f"{auc:.3f}" if isinstance(auc,(int,float)) else "—")
    left,right = st.columns([1.2,1])
    with left:
        section("Explore the platform","✧")
        st.markdown("""
        <div class="panel"><div class="panel-label">YOUR WORKSPACE</div>
        <h3 style="margin:12px 0 6px;color:#f2f7ff;">From features to insights</h3>
        <p class="small-muted">Use the sidebar to move through each project module.</p><hr>
        <p>📊 <b>Risk Prediction</b> — enter encoded dataset features and inspect model output.</p>
        <p>📝 <b>Medical Text Lab</b> — identify predefined terms and create an extractive summary.</p>
        <p>📈 <b>Model Evaluation</b> — compare test-set metrics and download results.</p></div>
        """, unsafe_allow_html=True)
    with right:
        section("Model performance snapshot","▥")
        results = metrics.get("results", {})
        if results:
            chart_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
            metric_cols = [x for x in ["accuracy","precision","recall","f1","roc_auc"] if x in chart_df.columns]
            if metric_cols:
                melt = chart_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
                fig = px.bar(melt, x="Metric", y="Score", color="Model", barmode="group")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#d5e1f0",legend_title_text="",margin=dict(l=5,r=5,t=15,b=5),height=300)
                fig.update_yaxes(range=[0,1],gridcolor="#263951"); fig.update_xaxes(showgrid=False)
                st.plotly_chart(fig,use_container_width=True)
        else:
            st.markdown('<div class="panel"><div class="panel-label">WAITING FOR TRAINING</div><div class="panel-value">No evaluation yet</div><div class="panel-note">Run python train_model.py in the terminal to load metrics here.</div></div>',unsafe_allow_html=True)
    section("Project information","⌁")
    a,b,c = st.columns(3)
    for col,number,title,desc in [
        (a,"MODULE 01","Risk ML","Tabular feature classification using the trained pipeline."),
        (b,"MODULE 02","Text NLP","Dictionary-based term extraction and extractive summary."),
        (c,"MODULE 03","Evaluation","Held-out metrics, model comparison, and exportable report.")]:
        with col:
            st.markdown(f'<div class="panel"><div class="panel-label">{number}</div><div class="panel-value">{title}</div><div class="panel-note">{desc}</div></div>',unsafe_allow_html=True)

elif page == "Risk Prediction":
    section("Risk prediction studio","◈")
    st.write("Model input fields are generated from the numeric columns used during training.")
    if model is None:
        st.warning("Model artifacts nahi mile. Dataset add karke `python train_model.py` run karein.")
        st.code("data/diabetes_binary_health_indicators_BRFSS2015.csv")
    else:
        st.markdown('<div class="small-muted">Inputs use the dataset’s encoded feature format; this is not a validated clinical questionnaire.</div>',unsafe_allow_html=True)
        with st.form("risk_form"):
            values = {}; cols = st.columns(3)
            for i,feature in enumerate(feature_names):
                with cols[i%3]:
                    values[feature] = st.number_input(feature.replace("_"," ").title(),value=0.0,step=1.0,help="Training dataset ke encoded feature format mein value enter karein.")
            submitted = st.form_submit_button("✦ Run model estimate",type="primary",use_container_width=True)
        if submitted:
            input_df = pd.DataFrame([values],columns=feature_names)
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])
            left,right = st.columns([1,1.2])
            with left:
                section("Prediction summary","◉")
                cls = "result-caution" if prediction == 1 else "result-good"
                label = "Positive class (1)" if prediction == 1 else "Negative class (0)"
                st.markdown(f'<div class="{cls}"><div class="panel-label">MODEL CLASS</div><div class="panel-value">{label}</div><p class="small-muted">Model classification only—not a clinical conclusion.</p></div>',unsafe_allow_html=True)
                st.metric("Estimated positive-class probability",f"{probability:.1%}")
            with right:
                st.plotly_chart(make_gauge(probability),use_container_width=True)
            st.markdown('<div class="notice">Probability is a model output, not a calibrated personal medical risk. Do not start/stop medication or delay care based on this result.</div>',unsafe_allow_html=True)
            with st.expander("Review submitted feature values"):
                st.dataframe(input_df,use_container_width=True)

elif page == "Medical Text Lab":
    section("Medical text intelligence","⌁")
    st.write("A lightweight, rule-based text explorer. It does not understand clinical context like a trained clinical NLP model.")
    note = st.text_area("Sample or de-identified text",placeholder="Example: Patient reports cough and fever for two days. No chest pain reported.",height=170)
    if st.button("✦ Analyze text",type="primary"):
        if not note.strip():
            st.warning("Pehle kuch text enter karein.")
        else:
            patterns = {
                "Symptoms":["fever","cough","headache","fatigue","nausea","pain","shortness of breath","dizziness","vomiting","rash"],
                "Conditions":["diabetes","hypertension","asthma","infection","anemia","migraine","pneumonia"],
                "Medicines":["insulin","metformin","paracetamol","ibuprofen","amoxicillin","aspirin"],
                "Measurements":["blood pressure","bmi","glucose","oxygen saturation","temperature","heart rate"]}
            lower = note.lower()
            found = {cat:[term for term in terms if re.search(r"\b"+re.escape(term)+r"\b",lower)] for cat,terms in patterns.items()}
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+",note.strip()) if s.strip()]
            summary = " ".join(sentences[:3]); total = sum(len(v) for v in found.values())
            c1,c2,c3 = st.columns(3)
            c1.metric("Matched terms",total); c2.metric("Categories with matches",sum(bool(v) for v in found.values())); c3.metric("Summary sentences",min(len(sentences),3))
            left,right = st.columns(2)
            with left:
                section("Extracted terms","✧")
                for category,terms in found.items():
                    with st.container(border=True):
                        st.markdown(f"**{category}**")
                        st.markdown(" ".join(f"`{term}`" for term in terms) if terms else "_No dictionary match_")
            with right:
                section("Extractive summary","▤")
                st.markdown(f'<div class="panel"><div class="panel-label">TEXT SUMMARY</div><p style="color:#eaf2ff;line-height:1.7;">{html.escape(summary)}</p></div>',unsafe_allow_html=True)
                report = f"SUMMARY\n{summary}\n\nEXTRACTED TERMS\n{json.dumps(found,indent=2)}\n"
                st.download_button("⬇ Download analysis as TXT",data=report,file_name="medical_text_analysis.txt",mime="text/plain")
            st.markdown('<div class="notice">Rule-based matching can miss terms and cannot reliably interpret negation or clinical context. Do not treat this output as medical advice.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><div class="panel-label">QUICK TEST</div><div class="panel-note">Enter sample text above and click Analyze text to see term cards and a downloadable report.</div></div>',unsafe_allow_html=True)

elif page == "Model Evaluation":
    section("Model evaluation center","▥")
    if not metrics:
        st.info("Evaluation dekhne ke liye pehle `python train_model.py` run karein.")
    else:
        st.markdown(f'<div class="panel"><div class="panel-label">SELECTED MODEL</div><div class="panel-value">{html.escape(str(metrics.get("best_model","Not available")))}</div><div class="panel-note">Metrics are from the held-out test split in the training script.</div></div>',unsafe_allow_html=True)
        results = metrics.get("results",{})
        if results:
            table = pd.DataFrame(results).T
            visible = [c for c in ["accuracy","precision","recall","f1","roc_auc"] if c in table.columns]
            if visible:
                st.write(""); st.dataframe(table[visible].style.format("{:.3f}"),use_container_width=True)
                chart_df = table[visible].reset_index().rename(columns={"index":"Model"})
                melt = chart_df.melt(id_vars="Model",var_name="Metric",value_name="Score")
                fig = px.bar(melt,x="Model",y="Score",color="Metric",barmode="group",text_auto=".3f")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#d5e1f0",legend_title_text="",margin=dict(l=5,r=5,t=20,b=5),height=380)
                fig.update_yaxes(range=[0,1],gridcolor="#263951")
                st.plotly_chart(fig,use_container_width=True)
            st.download_button("⬇ Download evaluation JSON",data=json.dumps(metrics,indent=2),file_name="model_evaluation.json",mime="application/json")
        st.markdown('<div class="notice">High scores on one test split do not guarantee performance on new populations. Consider class imbalance, dataset bias, leakage, and external validation.</div>',unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="small-muted" style="text-align:center;">MediScope AI • Student project prototype • Educational use only</p>',unsafe_allow_html=True)
