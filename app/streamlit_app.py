"""Thin Streamlit client over ``spamdetect.predict`` (presentation only)."""

from pathlib import Path

import streamlit as st

from spamdetect.predict import SpamClassifier

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
EXAMPLES = {
    "Obvious spam": "WINNER!! You have won a £1000 prize. Call 09061701461 now to claim.",
    "Normal chat": "Ok see you at the station at 6, bring the umbrella.",
    "Legitimate transactional": "Your Barclays code is 482913. Do not share it. If you did not "
                                "request this, call 0345 734 5345.",
}


@st.cache_resource
def get_classifier() -> SpamClassifier:
    return SpamClassifier.load(MODEL_DIR)


st.set_page_config(page_title="SMS Spam Detector", page_icon="📱")
st.title("📱 SMS Spam Detector")

try:
    clf = get_classifier()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

meta = clf.metadata
with st.sidebar:
    st.header("Model card")
    st.write(f"**Model:** {meta['model_name']}")
    st.write(f"**Test recall:** {meta['test_metrics']['recall']:.3f}")
    st.write(f"**Test ham FPR:** {meta['test_metrics']['ham_fpr']:.4f}")
    st.write(f"**Threshold:** {meta['threshold']:.3f}")
    st.write(f"**Dataset:** {meta['dataset']}")
    st.write(f"**Trained:** {meta['created_at']}")
    st.caption("Small, dated (2011, UK/Singapore) corpus - treat verdicts as indicative.")

if "text" not in st.session_state:
    st.session_state["text"] = ""
cols = st.columns(len(EXAMPLES))
for col, (label, example) in zip(cols, EXAMPLES.items(), strict=True):
    if col.button(label, use_container_width=True):
        st.session_state["text"] = example

text = st.text_area("Message", key="text", height=140, max_chars=clf.max_length)

if st.button("Classify", type="primary"):
    try:
        result = clf.predict(text)
    except ValueError as exc:
        st.warning(f"Please enter a valid message: {exc}")
    else:
        if result["label"] == "spam":
            st.error("🚫 Spam")
        else:
            st.success("✅ Not spam (ham)")
        st.progress(min(max(result["spam_probability"], 0.0), 1.0),
                    text=f"Spam probability {result['spam_probability']:.3f} "
                         f"(threshold {result['threshold']:.3f})")
        top = clf.explain(text, top_k=5)
        if top:
            st.subheader("Top contributing n-grams")
            st.table({"n-gram": [n for n, _ in top],
                      "contribution (+ = spam)": [round(v, 3) for _, v in top]})
