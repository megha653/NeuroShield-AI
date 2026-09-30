import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from multilingual import prepare_multilingual_message


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

ML_MODEL_PATH = BASE / "models" / "text_model.joblib"
TFIDF_PATH = BASE / "models" / "tfidf.joblib"
BEST_MODEL_PATH = BASE / "models" / "best_model.txt"

DL_MODEL_PATH = BASE / "models" / "deep_text_model.keras"
TOKENIZER_PATH = BASE / "models" / "deep_tokenizer.json"
CONFIG_PATH = BASE / "models" / "deep_config.json"

ML_RESULTS_PATH = BASE / "artifacts" / "model_comparison.csv"
DL_RESULTS_PATH = BASE / "artifacts" / "deep_model_results.csv"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #fafafa;
    --surface: #ffffff;
    --text: #171717;
    --muted: #737373;
    --border: #e8e8e8;
    --accent: #ff3f6c;
    --accent-soft: #fff1f4;
}

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


/* NAV */

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
    padding: 75px 0 55px 0;
    max-width: 850px;
}

.eyebrow {
    color: #ff3f6c;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
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
    max-width: 720px;
    margin-top: 22px;
}

.hero-note {
    font-size: 13px;
    color: #999999;
    margin-top: 14px;
}


/* SECTION */

.section {
    margin-top: 55px;
}

.section-kicker {
    font-size: 11px;
    font-weight: 800;
    color: #ff3f6c;
    letter-spacing: 1.5px;
    text-transform: uppercase;
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
    margin-bottom: 12px;
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
    text-transform: uppercase;
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


/* SIGNAL */

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


/* METRIC CARDS */

.stat-number {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #171717;
    margin-top: 8px;
}

.stat-label {
    font-size: 12px;
    color: #858585;
    margin-top: 4px;
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

[data-testid="stDataFrame"] {
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    overflow: hidden;
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
# LOAD MODELS
# ============================================================

required_files = [
    ML_MODEL_PATH,
    TFIDF_PATH,
    DL_MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH,
]

missing = [
    path.name
    for path in required_files
    if not path.exists()
]

if missing:
    st.error(
        "Missing model files: "
        + ", ".join(missing)
    )
    st.stop()


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

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    return (
        ml_model,
        vectorizer,
        dl_model,
        tokenizer,
        config,
    )


(
    ml_model,
    vectorizer,
    dl_model,
    tokenizer,
    dl_config,
) = load_models()


best_ml = (
    BEST_MODEL_PATH
    .read_text(encoding="utf-8")
    .strip()
    if BEST_MODEL_PATH.exists()
    else "Linear SVM"
)


# ============================================================
# HELPERS
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
# NAV
# ============================================================

st.markdown(
    '<div class="nav">'
    '<div class="brand">'
    '<span class="brand-mark"></span>'
    'NEUROSHIELD'
    '</div>'
    '<div class="nav-right">'
    'MULTILINGUAL ML + DEEP LEARNING'
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
    'MULTILINGUAL AI MESSAGE INTELLIGENCE'
    '</div>'
    '<div class="hero-title">'
    'Know what you\'re<br>'
    'about to trust.'
    '</div>'
    '<div class="hero-copy">'
    'NeuroShield analyses suspicious messages across languages '
    'using language detection, translation-assisted inference, '
    'classical machine learning and deep learning.'
    '</div>'
    '<div class="hero-note">'
    'Scammers went multilingual. '
    'It seemed rude not to keep up.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLES
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'QUICK TEST'
    '</div>'
    '<div class="section-title">'
    'Don\'t have a suspicious message handy?'
    '</div>'
    '<div class="section-copy">'
    'Lucky you. Borrow one of ours.'
    '</div>',
    unsafe_allow_html=True,
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    if st.button(
        "English",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "Urgent! Your bank account has been suspended. "
            "Verify your identity immediately using this link."
        )


with c2:

    if st.button(
        "Hindi",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "आपका बैंक खाता बंद कर दिया गया है। "
            "तुरंत अपनी पहचान सत्यापित करें।"
        )


with c3:

    if st.button(
        "Bengali",
        use_container_width=True,
    ):

        st.session_state.sample_message = (
            "আপনার ব্যাংক অ্যাকাউন্ট বন্ধ করা হয়েছে। "
            "এখনই আপনার পরিচয় যাচাই করুন।"
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
    'Paste an SMS or short message in your language. '
    'NeuroShield will detect the language automatically.'
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
        # MULTILINGUAL LAYER
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
        # LANGUAGE DISPLAY
        # ----------------------------------------------------

        if language_info[
            "was_translated"
        ]:

            st.markdown(
                '<div class="language-box">'
                '<div class="language-label">'
                'LANGUAGE DETECTED'
                '</div>'
                f'<div class="language-value">'
                f'{detected_language}'
                '</div>'
                '<div class="language-note">'
                'Translated into English before '
                'ML and deep-learning analysis.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.expander(
                "View translated text"
            ):

                st.write(
                    analysis_text
                )

        else:

            st.markdown(
                '<div class="language-box">'
                '<div class="language-label">'
                'LANGUAGE DETECTED'
                '</div>'
                f'<div class="language-value">'
                f'{detected_language}'
                '</div>'
                '<div class="language-note">'
                'No translation was required.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # MODEL PREDICTIONS
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


        both_suspicious = (
            ml_prediction == 1
            and dl_prediction == 1
        )

        both_normal = (
            ml_prediction == 0
            and dl_prediction == 0
        )


        # ----------------------------------------------------
        # VERDICT
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
                'Both models found patterns associated '
                'with suspicious or spam messaging. '
                'Verify the sender through another channel '
                'before taking action.'
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
                'Both models classified this as more similar '
                'to normal messaging. That is not a guarantee '
                'of safety — common sense is still supported.'
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
                'The machines are arguing.'
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
        # MODEL OPINIONS
        # ----------------------------------------------------

        st.markdown(
            '<div style="margin-top:30px;" '
            'class="section-kicker">'
            'MODEL OPINIONS'
            '</div>',
            unsafe_allow_html=True,
        )


        left, right = (
            st.columns(2)
        )


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


        with left:

            st.markdown(
                '<div class="card">'
                '<div class="card-label">'
                'Classical machine learning'
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


        with right:

            st.markdown(
                '<div class="card">'
                '<div class="card-label">'
                'Deep learning'
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


        # ----------------------------------------------------
        # SIGNALS
        # ----------------------------------------------------

        evidence = get_evidence(
            ml_features
        )


        st.markdown(
            '<div class="section">'
            '<div class="section-kicker">'
            'SIGNALS'
            '</div>'
            '<div class="section-title">'
            'What caught the model\'s attention?'
            '</div>'
            '<div class="section-copy">'
            'These translated or original English phrases '
            'contributed toward the classical model\'s '
            'suspicious classification.'
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
        # MODEL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "View model details"
        ):

            d1, d2 = (
                st.columns(2)
            )


            with d1:

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
                        f"Suspicious probability: "
                        f"{ml_score * 100:.1f}%"
                    )

                else:

                    st.write(
                        f"Decision-derived display score: "
                        f"{ml_score * 100:.1f}%"
                    )

                    st.caption(
                        "Linear SVM does not provide "
                        "calibrated probabilities by default."
                    )


            with d2:

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
                    f"Suspicious probability: "
                    f"{dl_score * 100:.1f}%"
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
    'We didn\'t just train one model and call it AI.'
    '</div>'
    '<div class="section-copy">'
    'Multiple classical models were evaluated, '
    'then compared with a deep-learning approach.'
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


    m1, m2, m3 = (
        st.columns(3)
    )


    with m1:

        st.markdown(
            '<div class="card">'
            '<div class="card-label">'
            'Best classical model'
            '</div>'
            f'<div class="stat-number">'
            f'{best_row["Model"]}'
            '</div>'
            '<div class="stat-label">'
            'Selected using F1 score'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


    with m2:

        st.markdown(
            '<div class="card">'
            '<div class="card-label">'
            'Classical ML accuracy'
            '</div>'
            f'<div class="stat-number">'
            f'{best_row["Accuracy"] * 100:.2f}%'
            '</div>'
            '<div class="stat-label">'
            'Test-set performance'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


    with m3:

        st.markdown(
            '<div class="card">'
            '<div class="card-label">'
            'Classical ML F1'
            '</div>'
            f'<div class="stat-number">'
            f'{best_row["F1 Score"] * 100:.2f}%'
            '</div>'
            '<div class="stat-label">'
            'Precision / recall balance'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


if DL_RESULTS_PATH.exists():

    dl_results = pd.read_csv(
        DL_RESULTS_PATH
    )

    dl_row = (
        dl_results.iloc[0]
    )


    n1, n2, n3 = (
        st.columns(3)
    )


    with n1:

        st.markdown(
            '<div class="card">'
            '<div class="card-label">'
            'Neural architecture'
            '</div>'
            '<div class="stat-number">'
            'BiLSTM'
            '</div>'
            '<div class="stat-label">'
            'Deep-learning NLP model'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


    with n2:

        st.markdown(
            '<div class="card">'
            '<div class="card-label">'
            'Deep-learning accuracy'
            '</div>'
            f'<div class="stat-number">'
            f'{dl_row["Accuracy"] * 100:.2f}%'
            '</div>'
            '<div class="stat-label">'
            'Test-set performance'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


    with n3:

        st.markdown(
            '<div class="card">'
            '<div class="card-label">'
            'Deep-learning F1'
            '</div>'
            f'<div class="stat-number">'
            f'{dl_row["F1 Score"] * 100:.2f}%'
            '</div>'
            '<div class="stat-label">'
            'Higher accuracy isn\'t everything.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# FULL RESULTS
# ============================================================

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
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-kicker">'
    'UNDER THE HOOD'
    '</div>'
    '<div class="section-title">'
    'Any language in. Two models out.'
    '</div>'
    '<div class="section-copy">'
    'Non-English input is detected and translated '
    'before reaching the existing English-trained '
    'ML and deep-learning models.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


s1, a1, s2, a2, s3, a3, s4 = (
    st.columns(
        [
            1.7,
            0.35,
            1.7,
            0.35,
            2.2,
            0.35,
            1.7,
        ]
    )
)


with s1:

    st.markdown(
        '<div class="flow-item">'
        'MESSAGE'
        '<span class="flow-sub">'
        'Any supported language'
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


with s2:

    st.markdown(
        '<div class="flow-item">'
        'LANGUAGE'
        '<span class="flow-sub">'
        'Detect + translate'
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


with s3:

    st.markdown(
        '<div class="flow-item">'
        'DUAL ANALYSIS'
        '<span class="flow-sub">'
        'TF-IDF + SVM / BiLSTM'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


with a3:

    st.markdown(
        '<div class="arrow">'
        '→'
        '</div>',
        unsafe_allow_html=True,
    )


with s4:

    st.markdown(
        '<div class="flow-item">'
        'VERDICT'
        '<span class="flow-sub">'
        'Classification + evidence'
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
        "**Core stack:** Python, Pandas, "
        "Scikit-learn, TensorFlow/Keras, "
        "TF-IDF, Linear SVM, BiLSTM and Streamlit."
    )

    st.write(
        "**Multilingual layer:** LangDetect + "
        "translation-assisted English normalization."
    )

    st.write(
        "**Training dataset:** labelled English "
        "SMS spam/ham data."
    )

    st.write(
        "**Important:** The SVM and BiLSTM models "
        "were trained on English data. Non-English "
        "messages are translated into English before "
        "classification."
    )

    st.write(
        "**Limitation:** Translation can alter slang, "
        "code-mixed text, cultural context and "
        "scam-specific wording. Performance has not "
        "been independently validated for every language."
    )

    st.write(
        "NeuroShield is an educational AI/ML prototype "
        "and should not be treated as a definitive "
        "fraud-detection system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-wrap">'
    '<b>NEUROSHIELD / 2026</b>'
    '<br><br>'
    'Built with machine learning, deep learning, '
    'multilingual NLP, and a healthy distrust '
    'of urgent links.'
    '</div>',
    unsafe_allow_html=True,
)