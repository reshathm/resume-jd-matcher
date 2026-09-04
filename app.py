"""
Resume vs JD Matcher — Web UI
------------------------------
A Streamlit front-end for match_resume.py's core logic. Reuses the exact
same TF-IDF + cosine similarity functions as the CLI tool — this file only
adds a browser-based interface around them, no scoring logic is duplicated.

Run with:
    streamlit run app.py
"""

import streamlit as st
from pypdf import PdfReader
from match_resume import (
    compute_match_score,
    missing_keywords,
    check_ats_compatibility,
)

st.set_page_config(page_title="Resume vs JD Matcher", page_icon="📄", layout="centered")

st.title("📄 Resume vs JD Matcher")
st.caption("TF-IDF + cosine similarity match scoring, with ATS compatibility checks.")


def extract_text(uploaded_file) -> str:
    """
    Extract plain text from an uploaded file. Supports PDF (via pypdf,
    page by page) and plain .txt files. Returns an empty string if
    nothing could be extracted, so the caller can show a clear warning
    rather than silently scoring on blank text.
    """
    if uploaded_file is None:
        return ""

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    else:
        # .txt or any other plain-text upload
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()


def input_block(label: str, key_prefix: str) -> str:
    """
    Renders an upload widget (PDF/TXT) and a paste-text fallback for one
    input (resume or JD), and returns whichever content is available,
    preferring the uploaded file when both are provided.
    """
    uploaded = st.file_uploader(f"Upload {label} (PDF or TXT)", type=["pdf", "txt"], key=f"{key_prefix}_file")
    text = ""
    if uploaded is not None:
        text = extract_text(uploaded)
        if text:
            st.success(f"Extracted {len(text.split())} words from {uploaded.name}")
        else:
            st.error("Could not extract any text from this file — it may be a scanned/image-based PDF. Try pasting the text instead.")

    pasted = st.text_area(f"...or paste {label} text", height=200, placeholder=f"Paste {label.lower()} text here...")
    return text if text else pasted


col1, col2 = st.columns(2)
with col1:
    resume_text = input_block("resume", "resume")
with col2:
    jd_text = input_block("job description", "jd")

if st.button("Run match", type="primary", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both a resume and a job description.")
    else:
        score, vectorizer, tfidf_matrix = compute_match_score(resume_text, jd_text)
        missing = missing_keywords(vectorizer, tfidf_matrix, top_n=15)

        st.subheader("Match Score")
        st.metric(label="Similarity", value=f"{score}%")
        st.progress(min(int(score), 100) / 100)

        if score >= 70:
            st.success("Strong match.")
        elif score >= 40:
            st.info("Moderate match — consider tailoring your resume.")
        else:
            st.warning("Weak match — resume and JD share little vocabulary.")

        st.subheader("ATS Compatibility")
        for name, passed, detail in check_ats_compatibility(resume_text):
            icon = "✅" if passed else "⚠️"
            st.write(f"{icon} **{name}**: {detail}")

        st.subheader("Missing Keywords")
        st.caption("High-importance JD terms not found in the resume:")
        if missing:
            st.write(", ".join(f"`{w}`" for w in missing))
        else:
            st.write("None — resume covers the JD's key terms well.")

st.divider()
st.caption("Core logic (TF-IDF vectorization, cosine similarity, ATS checks) lives in match_resume.py and is shared with the CLI version of this tool.")
