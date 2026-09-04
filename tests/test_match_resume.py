"""
Unit tests for match_resume.py

Run with:
    python3 -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from match_resume import (
    compute_match_score,
    missing_keywords,
    check_ats_compatibility,
)


# ---------- compute_match_score ----------

def test_identical_documents_score_near_100():
    """Two identical documents should score at or very near 100% similarity."""
    text = "Python developer with machine learning and data science experience"
    score, _, _ = compute_match_score(text, text)
    assert score >= 99.0


def test_completely_unrelated_documents_score_low():
    """Documents sharing no meaningful vocabulary should score near 0."""
    resume = "Baking sourdough bread requires patience and a good starter culture"
    jd = "Quantum physics research position studying particle accelerator data"
    score, _, _ = compute_match_score(resume, jd)
    assert score < 15.0


def test_partial_overlap_scores_between_extremes():
    """Documents sharing some but not all vocabulary should score in the middle."""
    resume = "Python developer with strong programming and testing skills"
    jd = "Python developer needed with cloud and machine learning experience"
    score, _, _ = compute_match_score(resume, jd)
    assert 0 < score < 100


def test_score_is_symmetric():
    """Cosine similarity should give the same score regardless of argument order."""
    a = "Python developer with security tooling experience"
    b = "Looking for a Python developer with security background"
    score_ab, _, _ = compute_match_score(a, b)
    score_ba, _, _ = compute_match_score(b, a)
    assert abs(score_ab - score_ba) < 0.01


# ---------- missing_keywords ----------

def test_missing_keywords_finds_jd_only_terms():
    """Words that are important in the JD but absent from the resume should be flagged."""
    resume = "Python developer skilled in git and testing"
    jd = "Python developer needed with tensorflow and pytorch experience"
    _, vectorizer, tfidf_matrix = compute_match_score(resume, jd)
    missing = missing_keywords(vectorizer, tfidf_matrix, top_n=10)
    assert "tensorflow" in missing
    assert "pytorch" in missing


def test_missing_keywords_excludes_shared_terms():
    """A word present in both documents should never appear in the missing list,
    since 'missing' specifically means present in JD but absent from resume."""
    resume = "Python developer with strong python skills"
    jd = "Python developer role requiring python expertise"
    _, vectorizer, tfidf_matrix = compute_match_score(resume, jd)
    missing = missing_keywords(vectorizer, tfidf_matrix, top_n=10)
    assert "python" not in missing


def test_missing_keywords_respects_top_n_limit():
    """The function should never return more than top_n keywords."""
    resume = "Short resume text"
    jd = "A completely different job description with many unique distinctive terms scattered throughout requiring numerous specialized skills across various technical domains"
    _, vectorizer, tfidf_matrix = compute_match_score(resume, jd)
    missing = missing_keywords(vectorizer, tfidf_matrix, top_n=3)
    assert len(missing) <= 3


# ---------- check_ats_compatibility ----------

def test_ats_check_detects_valid_email():
    text = "Contact me at reshathm108@gmail.com for more details"
    results = check_ats_compatibility(text)
    email_result = next(r for r in results if r[0] == "Email detectable")
    assert email_result[1] is True


def test_ats_check_flags_missing_email():
    text = "This resume has no contact information at all"
    results = check_ats_compatibility(text)
    email_result = next(r for r in results if r[0] == "Email detectable")
    assert email_result[1] is False


def test_ats_check_detects_phone_number():
    text = "Reach me at +91 8056033833 any time"
    results = check_ats_compatibility(text)
    phone_result = next(r for r in results if r[0] == "Phone number detectable")
    assert phone_result[1] is True


def test_ats_check_detects_standard_sections():
    text = "PROFESSIONAL SUMMARY ... EDUCATION ... SKILLS ... PROJECTS ..."
    results = check_ats_compatibility(text)
    section_result = next(r for r in results if r[0] == "Standard section headers")
    assert section_result[1] is True
