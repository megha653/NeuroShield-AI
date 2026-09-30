import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from multilingual import prepare_multilingual_message
from features import extract_features
from monitoring import log_prediction, get_monitoring_stats


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroShield",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).parent

# Classical ML
ML_MODEL_PATH = BASE / "models" / "text_model.joblib"
TFIDF_PATH = BASE / "models" / "tfidf.joblib"
BEST_MODEL_PATH = BASE / "models" / "best_model.txt"

# Deep Learning
DL_MODEL_PATH = BASE / "models" / "deep_text_model.keras"
TOKENIZER_PATH = BASE / "models" / "deep_tokenizer.json"
CONFIG_PATH = BASE / "models" / "deep_config.json"

# Unsupervised Learning
ANOMALY_MODEL_PATH = BASE / "models" / "anomaly_model.joblib"
ANOMALY_SCALER_PATH = BASE / "models" / "anomaly_scaler.joblib"

# Artifacts
ML_RESULTS_PATH = BASE / "artifacts" / "model_comparison.csv"
DL_RESULTS_PATH = BASE / "artifacts" / "deep_model_results.csv"
STATS_PATH = BASE / "artifacts" / "dataset_statistics.csv"

CLASS_CHART_PATH = BASE / "artifacts" / "class_distribution.png"
LENGTH_CHART_PATH = BASE / "artifacts" / "message_length_distribution.png"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #fafafa;
    color: #171717;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.8rem;
    padding-bottom: 5rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}


/* NAVIGATION */

.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0 22px 0;
    border-bottom: 1px solid #e8e8e8;
}

.brand {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #171717;
}

.brand-mark {
    display: inline-block;
    width: 9px;
    height: 9px;
    margin-right: 9px;
    background: #ff3f6c;
    border-radius: 2px;
}

.nav-right {
    font-size: 12px;
    color: #777777;
    letter-spacing: .5px;
}


/* HERO */

.hero {
    padding: 72px 0 55px 0;
    max-width: 850px;
}

.eyebrow {
    color: #ff3f6c;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 58px;
    line-height: 1.04;
    letter-spacing: -2.8px;
    color: #151515;
    font-weight: 800;
}

.hero-copy {
    font-size: 19px;
    line-height: 1.65;
    color: #686868;
    max-width: 730px;
    margin-top: 22px;
}

.hero-note {
    font-size: 13px;
    color: #999999;
    margin-top: 14px;
}


/* SECTIONS */

.section {
    margin-top: 55px;
}

.section-kicker {
    font-size: 11px;
    font-weight: 800;
    color: #ff3f6c;
    letter-spacing: 1.5px;
    margin-bottom: 7px;
}

.section-title {
    font-size: 30px;
    font-weight: 750;
    letter-spacing: -1px;
    color: #171717;
    margin-bottom: 6px;
}

.section-copy {
    font-size: 14px;
    color: #737373;
    margin-bottom: 22px;
}


/* LANGUAGE */

.language-box {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-left: 4px solid #ff3f6c;
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 18px;
}

.language-label {
    font-size: 10px;
    color: #999999;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.language-value {
    font-size: 16px;
    color: #171717;
    font-weight: 700;
    margin-top: 4px;
}

.language-note {
    color: #777777;
    font-size: 12px;
    margin-top: 5px;
}


/* CARDS */

.card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 14px;
    padding: 22px;
    min-height: 115px;
}

.card-label {
    color: #8b8b8b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.card-value {
    color: #181818;
    font-size: 22px;
    font-weight: 750;
    margin-top: 9px;
}

.card-note {
    color: #8a8a8a;
    font-size: 12px;
    margin-top: 6px;
}


/* VERDICT */

.verdict {
    background: #171717;
    color: white;
    border-radius: 16px;
    padding: 35px;
    margin-top: 24px;
}

.verdict-safe {
    background: #173e2b;
    color: white;
    border-radius: 16px;
    padding: 35px;
    margin-top: 24px;
}

.verdict-review {
    background: #5c4513;
    color: white;
    border-radius: 16px;
    padding: 35px;
    margin-top: 24px;
}

.verdict-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    opacity: .7;
}

.verdict-title {
    font-size: 34px;
    font-weight: 780;
    margin-top: 8px;
    letter-spacing: -1px;
}

.verdict-copy {
    margin-top: 8px;
    font-size: 14px;
    line-height: 1.6;
    opacity: .78;
    max-width: 760px;
}


/* ANOMALY */

.anomaly-box {
    background: #fff9e8;
    border: 1px solid #f2dfaa;
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 18px;
}

.anomaly-title {
    color: #765a0b;
    font-size: 13px;
    font-weight: 750;
}

.anomaly-copy {
    color: #806c37;
    font-size: 12px;
    margin-top: 4px;
}


/* SIGNALS */

.signal {
    display: inline-block;
    padding: 8px 12px;
    margin: 4px 5px 4px 0;
    border-radius: 8px;
    background: #fff1f4;
    color: #c72f54;
    font-size: 12px;
    font-weight: 650;
    border: 1px solid #ffd8e2;
}


/* MODEL LAB */

.lab-note {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 18px;
    color: #696969;
    font-size: 13px;
    line-height: 1.6;
}


/* FLOW */

.flow-item {
    text-align: center;
    padding: 18px 8px;
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    color: #262626;
    font-weight: 700;
    font-size: 13px;
}

.flow-sub {
    display: block;
    margin-top: 5px;
    color: #999999;
    font-size: 10px;
    font-weight: 500;
}

.arrow {
    text-align: center;
    color: #aaaaaa;
    font-size: 20px;
    padding-top: 14px;
}


/* STREAMLIT */

div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid #dedede !important;
    border-radius: 12px !important;
    color: #181818 !important;
    font-size: 15px !important;
    padding: 16px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #ff3f6c !important;
    box-shadow: 0 0 0 1px #ff3f6c !important;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
    border: 1px solid #e1e1e1;
}

div.stButton > button[kind="primary"] {
    background: #171717;
    color: white;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    background: #ff3f6c;
    color: white;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    padding: 18px;
    border-radius: 12px;
}


/* FOOTER */

.footer-wrap {
    border-top: 1px solid #e8e8e8;
    margin-top: 70px;
    padding-top: 22px;
    color: #999999;
    font-size: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# VERIFY FILES
# ============================================================

required_files = [
    ML_MODEL_PATH,
    TFIDF_PATH,
    DL_MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH,
    ANOMALY_MODEL_PATH,
    ANOMALY_SCALER_PATH,
]

missing = [
    path.name
    for path in required_files
    if not path.exists()
]

if missing:
    st.error(
        "Missing required model files: "
        + ", ".join(missing)
    )
    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    ml_model = joblib.load(
        ML_MODEL_PATH
    )

    vectorizer = joblib.load(
        TFIDF_PATH
    )

    dl_model = tf.keras.models.load_model(
        DL_MODEL_PATH
    )

    tokenizer = tokenizer_from_json(
        TOKENIZER_PATH.read_text(
            encoding="utf-8"
        )
    )

    dl_config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    anomaly_model = joblib.load(
        ANOMALY_MODEL_PATH
    )

    anomaly_scaler = joblib.load(
        ANOMALY_SCALER_PATH
    )

    return (
        ml_model,
        vectorizer,
        dl_model,
        tokenizer,
        dl_config,
        anomaly_model,
        anomaly_scaler,
    )


(
    ml_model,
    vectorizer,
    dl_model,
    tokenizer,
    dl_config,
    anomaly_model,
    anomaly_scaler,
) = load_models()


best_ml = (
    BEST_MODEL_PATH
    .read_text(encoding="utf-8")
    .strip()
    if BEST_MODEL_PATH.exists()
    else "Linear SVM"
)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def sigmoid(value):

    value = max(
        min(value, 50),
        -50
    )

    return 1 / (
        1 + math.exp(-value)
    )


def ml_predict(message):

    features = vectorizer.transform(
        [message]
    )

    prediction = int(
        ml_model.predict(
            features
        )[0]
    )

    calibrated = hasattr(
        ml_model,
        "predict_proba"
    )

    if calibrated:

        score = float(
            ml_model.predict_proba(
                features
            )[0][1]
        )

    elif hasattr(
        ml_model,
        "decision_function"
    ):

        raw_score = float(
            ml_model.decision_function(
                features
            )[0]
        )

        score = sigmoid(
            raw_score
        )

    else:

        score = float(
            prediction
        )

    return (
        prediction,
        score,
        features,
        calibrated,
    )


def dl_predict(message):

    sequence = (
        tokenizer.texts_to_sequences(
            [message]
        )
    )

    padded = pad_sequences(
        sequence,
        maxlen=dl_config["max_length"],
        padding="post",
        truncating="post",
    )

    score = float(
        dl_model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    prediction = int(
        score >= 0.5
    )

    return (
        prediction,
        score,
    )


def anomaly_predict(message):

    engineered = np.array(
        [
            extract_features(
                message
            )
        ]
    )

    scaled = anomaly_scaler.transform(
        engineered
    )

    raw_prediction = int(
        anomaly_model.predict(
            scaled
        )[0]
    )

    anomaly_score = float(
        anomaly_model.decision_function(
            scaled
        )[0]
    )

    # Isolation Forest:
    # 1  = normal/inlier
    # -1 = anomaly/outlier

    is_anomaly = int(
        raw_prediction == -1
    )

    return (
        is_anomaly,
        anomaly_score,
    )


def get_evidence(features):

    if not hasattr(
        ml_model,
        "coef_"
    ):
        return []

    terms = (
        vectorizer
        .get_feature_names_out()
    )

    row = features.toarray()[0]

    coefficients = (
        ml_model.coef_[0]
    )

    contributions = (
        row * coefficients
    )

    order = (
        contributions
        .argsort()[::-1]
    )

    evidence = []

    for index in order:

        if contributions[index] <= 0:
            continue

        evidence.append(
            (
                terms[index],
                float(
                    contributions[index]
                ),
            )
        )

        if len(evidence) == 7:
            break

    return evidence


# ============================================================
# SESSION STATE
# ============================================================

if "sample_message" not in st.session_state:

    st.session_state.sample_message = ""


# ============================================================
# NAVIGATION
# ============================================================

st.markdown(
    '<div class="nav">'
    '<div class="brand">'
    '<span class="brand-mark"></span>'
    'NEUROSHIELD'
    '</div>'
    '<div class="nav-right">'
    'ML / DEEP LEARNING / ANOMALY INTELLIGENCE'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">'
    'AI MESSAGE INTELLIGENCE'
    '</div>'
    '<div class="hero-title">'
    'Know what you\'re<br>'
    'about to trust.'
    '</div>'
    '<div class="hero-copy">'
    'NeuroShield combines classical machine learning, '
    'deep learning and unsupervised anomaly detection '
    'to analyse suspicious messages across languages.'
    '</div>'
    '<div class="hero-note">'
    'Because “urgent account verification” deserves '
    'a little more scrutiny than blind optimism.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# QUICK TEST
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'QUICK TEST'
    '</div>'
    '<div class="section-title">'
    'Try a message.'
    '</div>'
    '<div class="section-copy">'
    'Use an example or paste your own.'
    '</div>',
    unsafe_allow_html=True,
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    if st.button(
        "Bank alert",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "Urgent! Your bank account has been suspended. "
            "Verify your identity immediately using this link."
        )


with c2:

    if st.button(
        "Prize message",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "Congratulations! You have won a cash prize. "
            "Click now to claim your reward."
        )


with c3:

    if st.button(
        "Hindi",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "आपका बैंक खाता बंद कर दिया गया है। "
            "तुरंत अपनी पहचान सत्यापित करें।"
        )


with c4:

    if st.button(
        "Normal message",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "Hey, I'll reach home around 7. "
            "Do you want me to pick up dinner?"
        )


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-kicker">'
    'ANALYSE'
    '</div>'
    '<div class="section-title">'
    'Check a message.'
    '</div>'
    '<div class="section-copy">'
    'Paste an SMS or short message. '
    'Language detection happens automatically.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


message = st.text_area(
    "Message",
    value=st.session_state.sample_message,
    height=150,
    placeholder=(
        "Paste the message you want "
        "NeuroShield to analyse..."
    ),
    label_visibility="collapsed",
)


analyse = st.button(
    "Analyse message",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyse:

    if not message.strip():

        st.warning(
            "There is nothing to analyse. "
            "Even AI needs something to work with."
        )

    else:

        # ----------------------------------------------------
        # MULTILINGUAL PREPROCESSING
        # ----------------------------------------------------

        language_info = (
            prepare_multilingual_message(
                message
            )
        )

        analysis_text = (
            language_info[
                "translated_text"
            ]
        )

        detected_language = (
            language_info[
                "language_name"
            ]
        )


        # ----------------------------------------------------
        # LANGUAGE INFORMATION
        # ----------------------------------------------------

        if language_info[
            "was_translated"
        ]:

            language_note = (
                "Translated into English before "
                "model inference."
            )

        else:

            language_note = (
                "No translation required."
            )


        st.markdown(
            '<div class="language-box">'
            '<div class="language-label">'
            'LANGUAGE DETECTED'
            '</div>'
            f'<div class="language-value">'
            f'{detected_language}'
            '</div>'
            f'<div class="language-note">'
            f'{language_note}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


        if language_info[
            "was_translated"
        ]:

            with st.expander(
                "View translated text"
            ):

                st.write(
                    analysis_text
                )


        # ----------------------------------------------------
        # THREE ANALYSIS ENGINES
        # ----------------------------------------------------

        (
            ml_prediction,
            ml_score,
            ml_features,
            ml_calibrated,
        ) = ml_predict(
            analysis_text
        )


        (
            dl_prediction,
            dl_score,
        ) = dl_predict(
            analysis_text
        )


        (
            anomaly_prediction,
            anomaly_score,
        ) = anomaly_predict(
            analysis_text
        )


        # ----------------------------------------------------
        # MONITORING LOG
        # ----------------------------------------------------

        log_prediction(
            detected_language,
            ml_prediction,
            dl_prediction,
            anomaly_prediction,
        )


        # ----------------------------------------------------
        # CONSENSUS
        # ----------------------------------------------------

        both_suspicious = (
            ml_prediction == 1
            and dl_prediction == 1
        )

        both_normal = (
            ml_prediction == 0
            and dl_prediction == 0
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section">'
            '<div class="section-kicker">'
            'RESULT'
            '</div>'
            '<div class="section-title">'
            'NeuroShield\'s take.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


        if both_suspicious:

            st.markdown(
                '<div class="verdict">'
                '<div class="verdict-label">'
                'HIGH SUSPICION'
                '</div>'
                '<div class="verdict-title">'
                'We\'d think twice before clicking.'
                '</div>'
                '<div class="verdict-copy">'
                'Both supervised models found patterns '
                'associated with suspicious messaging. '
                'Verify the sender independently before '
                'taking action.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        elif both_normal:

            st.markdown(
                '<div class="verdict-safe">'
                '<div class="verdict-label">'
                'LOWER SUSPICION'
                '</div>'
                '<div class="verdict-title">'
                'Nothing particularly dramatic here.'
                '</div>'
                '<div class="verdict-copy">'
                'Both supervised models classified this '
                'as more similar to normal messaging. '
                'That is not a guarantee of safety.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        else:

            st.markdown(
                '<div class="verdict-review">'
                '<div class="verdict-label">'
                'MODEL DISAGREEMENT'
                '</div>'
                '<div class="verdict-title">'
                'The models disagree.'
                '</div>'
                '<div class="verdict-copy">'
                'The classical and deep-learning models '
                'reached different conclusions. '
                'Manual review wins this round.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # ANOMALY SIGNAL
        # ----------------------------------------------------

        if anomaly_prediction:

            st.markdown(
                '<div class="anomaly-box">'
                '<div class="anomaly-title">'
                'Unusual behavioural pattern detected'
                '</div>'
                '<div class="anomaly-copy">'
                'The unsupervised Isolation Forest found '
                'this message structurally unusual compared '
                'with normal messages. This is an anomaly '
                'signal, not proof that the message is spam.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # ENGINE RESULTS
        # ----------------------------------------------------

        st.markdown(
            '<div style="margin-top:30px;" '
            'class="section-kicker">'
            'ANALYSIS ENGINES'
            '</div>',
            unsafe_allow_html=True,
        )


        e1, e2, e3 = st.columns(3)


        ml_result = (
            "Suspicious"
            if ml_prediction == 1
            else "Likely normal"
        )


        dl_result = (
            "Suspicious"
            if dl_prediction == 1
            else "Likely normal"
        )


        anomaly_result = (
            "Unusual pattern"
            if anomaly_prediction
            else "Typical pattern"
        )


        with e1:

            st.markdown(
                '<div class="card">'
                '<div class="card-label">'
                'SUPERVISED ML'
                '</div>'
                f'<div class="card-value">'
                f'{ml_result}'
                '</div>'
                f'<div class="card-note">'
                f'{best_ml} / TF-IDF'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        with e2:

            st.markdown(
                '<div class="card">'
                '<div class="card-label">'
                'DEEP LEARNING'
                '</div>'
                f'<div class="card-value">'
                f'{dl_result}'
                '</div>'
                '<div class="card-note">'
                'BiLSTM / learned embeddings'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        with e3:

            st.markdown(
                '<div class="card">'
                '<div class="card-label">'
                'UNSUPERVISED ML'
                '</div>'
                f'<div class="card-value">'
                f'{anomaly_result}'
                '</div>'
                '<div class="card-note">'
                'Isolation Forest / engineered features'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # EXPLAINABLE SIGNALS
        # ----------------------------------------------------

        evidence = get_evidence(
            ml_features
        )


        st.markdown(
            '<div class="section">'
            '<div class="section-kicker">'
            'EXPLAINABILITY'
            '</div>'
            '<div class="section-title">'
            'What caught the model\'s attention?'
            '</div>'
            '<div class="section-copy">'
            'These terms contributed toward the '
            'classical model\'s suspicious classification.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


        if evidence:

            chips = ""

            for (
                term,
                contribution,
            ) in evidence:

                chips += (
                    f'<span class="signal">'
                    f'{term}'
                    f'</span>'
                )

            st.markdown(
                chips,
                unsafe_allow_html=True,
            )

        else:

            st.write(
                "No strong suspicious text "
                "signals were identified."
            )


        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "View technical model details"
        ):

            t1, t2, t3 = st.columns(3)


            with t1:

                st.markdown(
                    "**Classical ML**"
                )

                st.write(
                    f"Model: {best_ml}"
                )

                st.write(
                    "Representation: TF-IDF"
                )

                if ml_calibrated:

                    st.write(
                        f"Probability: "
                        f"{ml_score * 100:.1f}%"
                    )

                else:

                    st.write(
                        f"Decision-derived score: "
                        f"{ml_score * 100:.1f}%"
                    )

                    st.caption(
                        "Linear SVM does not provide "
                        "calibrated probabilities by default."
                    )


            with t2:

                st.markdown(
                    "**Deep Learning**"
                )

                st.write(
                    "Model: Bidirectional LSTM"
                )

                st.write(
                    "Representation: Learned embedding"
                )

                st.write(
                    f"Probability: "
                    f"{dl_score * 100:.1f}%"
                )


            with t3:

                st.markdown(
                    "**Unsupervised ML**"
                )

                st.write(
                    "Model: Isolation Forest"
                )

                st.write(
                    "Input: 8 engineered behavioural features"
                )

                st.write(
                    f"Anomaly decision score: "
                    f"{anomaly_score:.3f}"
                )

                st.caption(
                    "Lower Isolation Forest decision scores "
                    "indicate more unusual observations."
                )


# ============================================================
# MODEL RESEARCH
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-kicker">'
    'MODEL RESEARCH'
    '</div>'
    '<div class="section-title">'
    'We didn\'t train one model and call it AI.'
    '</div>'
    '<div class="section-copy">'
    'Classical algorithms were compared using '
    'multiple evaluation metrics and then tested '
    'alongside a deep-learning model.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


if ML_RESULTS_PATH.exists():

    ml_results = pd.read_csv(
        ML_RESULTS_PATH
    )

    best_row = ml_results.iloc[
        ml_results[
            "F1 Score"
        ].argmax()
    ]

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Best classical model",
        best_row["Model"]
    )

    r2.metric(
        "Classical accuracy",
        f'{best_row["Accuracy"] * 100:.2f}%'
    )

    r3.metric(
        "Classical F1",
        f'{best_row["F1 Score"] * 100:.2f}%'
    )


if DL_RESULTS_PATH.exists():

    dl_results = pd.read_csv(
        DL_RESULTS_PATH
    )

    dl_row = dl_results.iloc[0]

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Deep model",
        "BiLSTM"
    )

    d2.metric(
        "Deep accuracy",
        f'{dl_row["Accuracy"] * 100:.2f}%'
    )

    d3.metric(
        "Deep F1",
        f'{dl_row["F1 Score"] * 100:.2f}%'
    )


with st.expander(
    "View complete experiment results"
):

    if ML_RESULTS_PATH.exists():

        st.markdown(
            "**Classical machine learning**"
        )

        table = pd.read_csv(
            ML_RESULTS_PATH
        )

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ]:

            if column in table.columns:

                table[column] = (
                    table[column] * 100
                ).round(2)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


    if DL_RESULTS_PATH.exists():

        st.markdown(
            "**Deep learning**"
        )

        table = pd.read_csv(
            DL_RESULTS_PATH
        )

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ]:

            if column in table.columns:

                table[column] = (
                    table[column] * 100
                ).round(2)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL LAB
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-kicker">'
    'MODEL LAB'
    '</div>'
    '<div class="section-title">'
    'Data, experiments and model health.'
    '</div>'
    '<div class="section-copy">'
    'A compact view of dataset statistics and '
    'inference behaviour after deployment.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


if STATS_PATH.exists():

    stats = pd.read_csv(
        STATS_PATH
    ).iloc[0]

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Dataset messages",
        int(
            stats[
                "total_messages"
            ]
        )
    )

    s2.metric(
        "Average message length",
        f'{stats["average_message_length"]:.0f} chars'
    )

    s3.metric(
        "Std. deviation",
        f'{stats["std_message_length"]:.0f} chars'
    )


monitor_stats = (
    get_monitoring_stats()
)


m1, m2, m3 = st.columns(3)

m1.metric(
    "Predictions analysed",
    monitor_stats[
        "total_predictions"
    ]
)

m2.metric(
    "ML / DL agreement",
    f'{monitor_stats["agreement_rate"]}%'
)

m3.metric(
    "Suspicious predictions",
    f'{monitor_stats["suspicious_rate"]}%'
)


st.markdown(
    '<div class="lab-note">'
    'Monitoring stores prediction metadata such as '
    'model outputs and agreement rates. '
    '<b>The original message text is not stored.</b>'
    '</div>',
    unsafe_allow_html=True,
)


with st.expander(
    "View dataset analysis"
):

    if CLASS_CHART_PATH.exists():

        st.image(
            str(
                CLASS_CHART_PATH
            ),
            caption=(
                "Class distribution"
            ),
        )


    if LENGTH_CHART_PATH.exists():

        st.image(
            str(
                LENGTH_CHART_PATH
            ),
            caption=(
                "Message length distribution"
            ),
        )


# ============================================================
# PIPELINE
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-kicker">'
    'UNDER THE HOOD'
    '</div>'
    '<div class="section-title">'
    'One message. Three analytical views.'
    '</div>'
    '<div class="section-copy">'
    'Supervised learning predicts known classes, '
    'deep learning learns sequential text patterns, '
    'and unsupervised learning looks for unusual behaviour.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


p1, a1, p2, a2, p3 = st.columns(
    [2, 0.4, 3, 0.4, 2]
)


with p1:

    st.markdown(
        '<div class="flow-item">'
        'MESSAGE'
        '<span class="flow-sub">'
        'Language detection + normalization'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


with a1:

    st.markdown(
        '<div class="arrow">'
        '→'
        '</div>',
        unsafe_allow_html=True,
    )


with p2:

    st.markdown(
        '<div class="flow-item">'
        'THREE ENGINES'
        '<span class="flow-sub">'
        'Linear SVM / BiLSTM / Isolation Forest'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


with a2:

    st.markdown(
        '<div class="arrow">'
        '→'
        '</div>',
        unsafe_allow_html=True,
    )


with p3:

    st.markdown(
        '<div class="flow-item">'
        'RESULT'
        '<span class="flow-sub">'
        'Classification + anomaly + evidence'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TECHNICAL SCOPE
# ============================================================

with st.expander(
    "Technical scope and limitations"
):

    st.write(
        "**Supervised ML:** Logistic Regression, "
        "Linear SVM and Random Forest."
    )

    st.write(
        "**Deep Learning:** TensorFlow/Keras BiLSTM."
    )

    st.write(
        "**Unsupervised ML:** Isolation Forest trained "
        "on engineered behavioural features from normal messages."
    )

    st.write(
        "**Feature engineering:** message length, word count, "
        "digit statistics, uppercase ratio, special-character "
        "count, URL count and urgency indicators."
    )

    st.write(
        "**EDA:** Pandas, Matplotlib and Seaborn."
    )

    st.write(
        "**Multilingual layer:** language detection and "
        "translation-assisted English normalization."
    )

    st.write(
        "**Monitoring:** inference metadata, model agreement "
        "and prediction distribution. Raw message text is not logged."
    )

    st.write(
        "**Important limitation:** the primary classifiers were "
        "trained on English SMS spam/ham data. Translation can alter "
        "context, slang and code-mixed language."
    )

    st.write(
        "NeuroShield is an educational AI/ML prototype and "
        "should not be treated as a definitive fraud-detection system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-wrap">'
    '<b>NEUROSHIELD / 2026</b>'
    '<br><br>'
    'Supervised ML. Deep learning. Unsupervised detection. '
    'A healthy distrust of urgent links.'
    '</div>',
    unsafe_allow_html=True,
)