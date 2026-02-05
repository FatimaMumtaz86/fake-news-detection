import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import altair as alt
import pandas as pd

# ────────────────── NLTK ──────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ────────────────── TEXT CLEANING ──────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = nltk.word_tokenize(text)
    return " ".join(
        lemmatizer.lemmatize(w) for w in words if w not in stop_words
    )

# ────────────────── LOAD TRAINED OBJECTS ──────────────────
model = joblib.load("fake_news_pac_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
selector = joblib.load("chi2_selector.pkl")

# ────────────────── PREDICTION FUNCTION ──────────────────
def predict_with_confidence(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    vec = selector.transform(vec)
    pred = model.predict(vec)[0]
    decision = model.decision_function(vec)[0]
    confidence = 1 / (1 + np.exp(-abs(decision)))
    label = "FAKE NEWS" if pred == 1 else "REAL NEWS"
    return label, round(confidence * 100, 2)

# ────────────────── UI Symbols ──────────────────
HEADER = "⊹₊˚‧⋆.ೃ࿔*:･°⊹₊˚‧︵‿₊୨୧₊‿︵‧˚₊⊹⋆.ೃ࿔*:･°⊹₊˚‧⊹₊˚‧⋆.ೃ࿔:･°"
DECOR  = "⋆.ೃ࿔*:･°°❀⋆₊୨୧₊.ೃ࿔*:･°❀⋆.₊୨୧₊ೃ࿔*:･⋆.ೃ࿔*:･°"

# ────────────────── STREAMLIT CONFIG ──────────────────
st.set_page_config(page_title="Fake News Detector", layout="centered")

# 🌸 Custom CSS
st.markdown(
    """
    <style>
    .stApp {background-color: #F8F5FF;}
    h1,h2,h3,h4,h5,h6,label,p,div {color: #702963 !important;}
    textarea {background-color: #E6E6FA !important; color: #702963 !important; border-radius: 14px; border:2px solid #CBC3E3 !important;}
    .stButton>button {background-color:#CBC3E3; color:#702963; border-radius:14px; height:45px; font-size:16px; font-weight:bold; border:none; transition:0.3s ease;}
    .stButton>button:hover {background-color:#AA98A9; color:white; transform:scale(1.03);}
    .stAlert {background-color:#E6E6FA !important; color:#702963 !important; border-radius:10px;}
    .fade-in {animation: fadeIn 1s ease-in;}
    @keyframes fadeIn {from {opacity:0;} to {opacity:1;}}
    </style>
    """, unsafe_allow_html=True
)

# ✨ Header
st.markdown(f"<center>{HEADER}</center>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown(f"<center>{HEADER}</center>", unsafe_allow_html=True)

# ✍️ Input box
text = st.text_area("Enter news text:", height=200)

# # 🔮 Predict button
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some news text first ")
    else:
        # ── Spinner / Loading ──
        with st.spinner("Analyzing news.ೃ࿔*:･°... "):
            time.sleep(1)  # Simulate delay for animation effect
            label, conf = predict_with_confidence(text)

        # ── Fade-in result ──
        st.markdown(
            f"""
            <div class='fade-in' style="
                background-color:#F8F5FF;
                padding:20px;
                border-radius:16px;
                text-align:center;
                border:2px dashed #CBC3E3;
                margin-top:20px;
                margin-bottom:10px;
            ">
                <p style="font-size:18px;">{DECOR}</p>
                <p style="font-size:22px;"><b>Prediction:</b> {label}</p>
                <p style="font-size:20px;"><b>Confidence:</b> {conf}%</p>
                <p style="font-size:18px;">{DECOR}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── CLEAN & FIXED CONFIDENCE CHART ──
        df_conf = pd.DataFrame({
            "Confidence": [conf]
        })

        chart = alt.Chart(df_conf).mark_bar(
            size=60,
            cornerRadiusTopLeft=14,
            cornerRadiusTopRight=14,
        ).encode(
            x=alt.X(
                "Confidence:Q",
                scale=alt.Scale(domain=[0, 100]),
                title=None
            ),
            color=alt.value("#795780")
        ).properties(
            height=120,
            background="#D6BCDB"
        ).configure_axis(
            grid=False,
            domain=True,
            ticks=False,
            labels=True
        ).configure_view(
            strokeWidth=0
        )

        st.altair_chart(chart, use_container_width=True)
