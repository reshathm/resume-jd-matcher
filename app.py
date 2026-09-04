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
from match_resume import (
    compute_match_score,
    missing_keywords,
    check_ats_compatibility,
)

st.set_page_config(page_title="Resume vs JD Matcher", page_icon="📄", layout="centered")

st.title("📄 Resume vs JD Matcher")
st.caption("TF-IDF + cosine similarity match scoring, with ATS compatibility checks.")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Resume text", height=300, placeholder="Paste your resume text here...")
with col2:
    jd_text = st.text_area("Job description", height=300, placeholder="Paste the job description here...")

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
