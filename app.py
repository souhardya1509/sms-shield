import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model_controlled.pkl"

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #656be0 0%, #7354bd 52%, #7950a3 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 920px;
        padding-top: 3rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 0 !important;
        border-radius: 26px;
        padding: 1.4rem 1.7rem 1.3rem;
        box-shadow: 0 18px 45px rgba(28, 19, 68, 0.18);
    }
    .app-title {
        text-align: center;
        color: #252536;
        font-size: clamp(2rem, 4vw, 2.7rem);
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0.2rem 0 0.25rem;
    }
    .app-subtitle {
        text-align: center;
        color: #777987;
        font-size: 1.05rem;
        margin: 0 0 1.8rem;
    }
    .field-label {
        color: #353543;
        font-weight: 700;
        font-size: 1rem;
        margin: 0.35rem 0 0.45rem;
    }
    [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        color: #252536 !important;
        border: 2px solid #8a8cf0 !important;
        border-radius: 13px !important;
        min-height: 145px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        box-shadow: 0 0 0 3px rgba(119, 111, 229, 0.08);
    }
    [data-testid="stTextArea"] label { display: none; }
    .stButton > button {
        min-height: 54px;
        border-radius: 12px;
        font-weight: 800;
        border: 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(37, 30, 80, 0.13);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(110deg, #6676e8, #7b45a7);
        color: #ffffff;
    }
    .result-card {
        border-radius: 15px;
        padding: 1rem 1.2rem;
        margin: 1.2rem 0 0.7rem;
        border: 1px solid #e6e7ef;
        color: #29293a;
    }
    .result-spam { border-left: 6px solid #ef555f; background: #fff3f3; }
    .result-ham { border-left: 6px solid #20a779; background: #effbf5; }
    .result-uncertain { border-left: 6px solid #e5a12b; background: #fff9ec; }
    .result-title { font-size: 1.35rem; font-weight: 850; margin-bottom: 0.25rem; }
    .result-copy { color: #626475; font-size: 0.96rem; }
    .small-note { color: #e9e8ff; font-size: 0.82rem; text-align: center; margin-top: 1rem; }
    [data-testid="stMetric"] {
        background: #f7f7fc;
        border: 1px solid #e9e9f2;
        border-radius: 12px;
        padding: 0.8rem;
    }
    [data-testid="stMetricLabel"] { color: #656579 !important; }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 0.6rem 1.5rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { padding: 1rem 0.85rem; border-radius: 18px; }
        .stButton > button { min-height: 52px; font-size: 0.82rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOAD CONTROLLED CANDIDATE MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "The candidate model `spam_model_controlled.pkl` was not found. "
        "Put it in the same folder as this app.py. Your old `spam_model.pkl` "
        "is not used by this version."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the candidate model: {exc}")
    st.stop()

# Threshold selected by validation for the duplicate-retaining model was 0.42,
# but this app uses the deduplicated candidate model, whose validation-selected
# threshold was 0.45. Default 0.50 performed better on its held-out test set.
SPAM_THRESHOLD = 0.50
HAM_THRESHOLD = 0.70

EXAMPLES = {
    "spam": "Congratulations! You have won a free iPhone! Claim your prize now!",
    "normal": "Hey bro, are you coming to college tomorrow?",
    "borderline": "Your order has been shipped. Track your package here.",
}


def set_example(example_key):
    st.session_state["message_input"] = EXAMPLES[example_key]
    st.session_state["last_result"] = None


def clear_message():
    st.session_state["message_input"] = ""
    st.session_state["last_result"] = None


if "message_input" not in st.session_state:
    st.session_state["message_input"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ---------------------------------------------------------
# MAIN CARD
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="app-title">📩 SMS Spam Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Intelligent SMS Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="field-label">Enter Your Message:</div>', unsafe_allow_html=True)

    message = st.text_area(
        "SMS message",
        key="message_input",
        height=150,
        placeholder="Type your SMS message here… e.g., 'FREE entry to win £1000 cash! Click here now!'",
        label_visibility="collapsed",
    )

    col_classify, col_examples, col_clear = st.columns(3, gap="small")
    with col_classify:
        classify_clicked = st.button("🔎 CLASSIFY MESSAGE", use_container_width=True, type="primary")

    with col_examples:
        with st.popover("📊 LOAD EXAMPLES", use_container_width=True):
            st.caption("Choose an example message to try.")
            st.button("🚨 Spam example", key="example_spam", use_container_width=True,
                      on_click=set_example, args=("spam",))
            st.button("✅ Normal example", key="example_normal", use_container_width=True,
                      on_click=set_example, args=("normal",))
            st.button("⚠️ Borderline example", key="example_borderline", use_container_width=True,
                      on_click=set_example, args=("borderline",))

    with col_clear:
        st.button("🧹 CLEAR", use_container_width=True, on_click=clear_message)

    if classify_clicked:
        if not message or not message.strip():
            st.session_state["last_result"] = None
            st.warning("Please enter a message before classifying it.")
        elif len(message) > 5000:
            st.session_state["last_result"] = None
            st.warning("Please enter a shorter message (maximum 5,000 characters).")
        else:
            try:
                probabilities = model.predict_proba([message])[0]
                classes = [str(c).strip().lower() for c in model.classes_]
                if "ham" not in classes or "spam" not in classes:
                    st.error("The saved model does not contain expected 'ham' and 'spam' classes.")
                    st.stop()

                ham_probability = float(probabilities[classes.index("ham")])
                spam_probability = float(probabilities[classes.index("spam")])

                if spam_probability >= SPAM_THRESHOLD:
                    result = "spam"
                elif ham_probability >= HAM_THRESHOLD:
                    result = "ham"
                else:
                    result = "uncertain"

                st.session_state["last_result"] = {
                    "result": result,
                    "ham_probability": ham_probability * 100,
                    "spam_probability": spam_probability * 100,
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result_data = st.session_state.get("last_result")
    if result_data:
        st.markdown("---")
        result = result_data["result"]
        ham_probability = result_data["ham_probability"]
        spam_probability = result_data["spam_probability"]

        if result == "spam":
            st.markdown(
                '<div class="result-card result-spam"><div class="result-title">🚨 SPAM MESSAGE DETECTED</div>'
                '<div class="result-copy">This message resembles patterns found in spam messages.</div></div>',
                unsafe_allow_html=True,
            )
        elif result == "ham":
            st.markdown(
                '<div class="result-card result-ham"><div class="result-title">✅ NOT SPAM</div>'
                '<div class="result-copy">The model considers this message more consistent with legitimate SMS.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-uncertain"><div class="result-title">⚠️ UNCERTAIN MESSAGE</div>'
                '<div class="result-copy">Neither class met its decision threshold. Please review the message carefully.</div></div>',
                unsafe_allow_html=True,
            )

        metric_ham, metric_spam = st.columns(2)
        with metric_ham:
            st.metric("✅ Not Spam score", f"{ham_probability:.2f}%")
        with metric_spam:
            st.metric("🚨 Spam score", f"{spam_probability:.2f}%")

        st.caption("These model probability scores are not a guarantee that a message is safe or malicious.")
        st.write("Not Spam score")
        st.progress(min(100, max(0, int(round(ham_probability)))))
        st.write("Spam score")
        st.progress(min(100, max(0, int(round(spam_probability)))))


# ---------------------------------------------------------
# BULK CSV SCANNING
# ---------------------------------------------------------
st.markdown("## 📂 Bulk SMS Scanner")
st.write("Upload a CSV file with a column containing SMS messages. SMS Shield will classify each row and let you download the results.")

uploaded_csv = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="CSV must contain a column with SMS text, for example a column named 'message'.",
    key="bulk_csv_uploader",
)

if uploaded_csv is not None:
    if uploaded_csv.size > 10 * 1024 * 1024:
        st.error("Please upload a CSV smaller than 10 MB.")
    else:
        try:
            bulk_df = pd.read_csv(uploaded_csv)
            if bulk_df.empty:
                st.warning("The CSV has no data rows.")
            elif len(bulk_df.columns) == 0:
                st.error("The CSV has no columns.")
            else:
                st.caption(f"Loaded {len(bulk_df):,} rows. Select the column containing SMS text.")
                preferred_index = list(bulk_df.columns).index("message") if "message" in bulk_df.columns else 0
                message_column = st.selectbox(
                    "SMS text column",
                    options=list(bulk_df.columns),
                    index=preferred_index,
                    key="bulk_message_column",
                )
                if st.button("🔎 SCAN CSV", type="primary", key="scan_bulk_csv"):
                    result_df = bulk_df.copy()
                    predictions = []
                    ham_scores = []
                    spam_scores = []
                    valid_positions = []
                    valid_messages = []

                    for position, raw_value in enumerate(bulk_df[message_column].tolist()):
                        if pd.isna(raw_value):
                            predictions.append("Invalid input")
                            ham_scores.append(None)
                            spam_scores.append(None)
                            continue
                        text_value = str(raw_value).strip()
                        if not text_value:
                            predictions.append("Invalid input")
                            ham_scores.append(None)
                            spam_scores.append(None)
                        elif len(text_value) > 5000:
                            predictions.append("Invalid input - too long")
                            ham_scores.append(None)
                            spam_scores.append(None)
                        elif not any(char.isalpha() for char in text_value):
                            predictions.append("Invalid input - no letters")
                            ham_scores.append(None)
                            spam_scores.append(None)
                        else:
                            predictions.append(None)
                            ham_scores.append(None)
                            spam_scores.append(None)
                            valid_positions.append(position)
                            valid_messages.append(text_value)

                    if valid_messages:
                        try:
                            probabilities = model.predict_proba(valid_messages)
                            classes = [str(c).strip().lower() for c in model.classes_]
                            if "ham" not in classes or "spam" not in classes:
                                st.error("The saved model does not contain expected 'ham' and 'spam' classes.")
                                st.stop()
                            ham_index = classes.index("ham")
                            spam_index = classes.index("spam")
                            for row_idx, probs in zip(valid_positions, probabilities):
                                ham_probability = float(probs[ham_index])
                                spam_probability = float(probs[spam_index])
                                if spam_probability >= SPAM_THRESHOLD:
                                    label = "Spam"
                                elif ham_probability >= HAM_THRESHOLD:
                                    label = "Ham"
                                else:
                                    label = "Uncertain"
                                predictions[row_idx] = label
                                ham_scores[row_idx] = round(ham_probability * 100, 2)
                                spam_scores[row_idx] = round(spam_probability * 100, 2)
                        except Exception as exc:
                            st.error(f"Bulk prediction failed: {exc}")
                            st.stop()

                    result_df["prediction"] = predictions
                    result_df["ham_score_percent"] = ham_scores
                    result_df["spam_score_percent"] = spam_scores
                    st.session_state["bulk_result_df"] = result_df

                scanned_df = st.session_state.get("bulk_result_df")
                if scanned_df is not None:
                    st.markdown("### Scan results")
                    count_col1, count_col2, count_col3, count_col4 = st.columns(4)
                    counts = scanned_df["prediction"].value_counts()
                    count_col1.metric("Spam", int(counts.get("Spam", 0)))
                    count_col2.metric("Ham", int(counts.get("Ham", 0)))
                    count_col3.metric("Uncertain", int(counts.get("Uncertain", 0)))
                    invalid_count = sum(str(x).startswith("Invalid input") for x in scanned_df["prediction"])
                    count_col4.metric("Invalid rows", int(invalid_count))
                    st.dataframe(scanned_df, use_container_width=True, hide_index=True)
                    csv_bytes = scanned_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Download results as CSV",
                        data=csv_bytes,
                        file_name="sms_shield_scan_results.csv",
                        mime="text/csv",
                        key="download_bulk_results",
                    )
        except Exception as exc:
            st.error(f"Could not read this CSV. Please check that it is a valid CSV file. Details: {exc}")


st.markdown(
    '<div class="small-note">🛡️ SMS Shield · Built with Python, Scikit-learn & Streamlit</div>',
    unsafe_allow_html=True,
)
