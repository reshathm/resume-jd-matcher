# Resume vs JD Matcher

A command-line tool that scores how well a resume matches a job description, and flags real ATS-parsing risks in the original document. Built to understand core NLP text-similarity techniques from first principles — no pretrained models, no black box.

---

## Why TF-IDF + cosine similarity?

Comparing two pieces of text for "similarity" requires turning words into numbers first — a computer can't compare meaning directly. TF-IDF (Term Frequency-Inverse Document Frequency) scores each word by how important it is to a specific document: frequent in that document, but not frequent everywhere. A word like "TensorFlow" appearing in a JD but rare across general text carries real signal; a word like "the" carries none and is filtered out entirely (`stop_words="english"`).

Once both documents are vectors of these scores, **cosine similarity** measures the angle between them — same direction means similar word emphasis, regardless of document length. That's why a short resume and a long JD can still score meaningfully: the metric compares proportion, not word count.

This was chosen deliberately over a pretrained embedding model (e.g. sentence-transformers) as a foundational exercise — every number in the output is traceable back to word frequency, nothing is a black box.

---

## Why also check ATS compatibility?

A high vocabulary-match score is meaningless if an actual Applicant Tracking System can't parse the resume in the first place. Real ATS software fails in ways that have nothing to do with content:

- **Structural parsing failures** — text inside tables, multi-column layouts, or images is often skipped or scrambled by ATS parsers, even though it displays perfectly in Word.
- **Missing extractable contact info** — if a name/email/phone isn't in plain, parseable text, some systems fail to register the applicant at all.
- **Non-standard section headers** — ATS looks for expected headings ("Education," "Experience," "Skills") to structure the parsed resume; creative naming can confuse it.

The TF-IDF matcher alone can't catch any of this — it only ever sees whatever text it's given, never the original file's structure. So `check_ats_compatibility()` and `check_docx_structure()` were added as a separate layer, testing the actual `.docx` file rather than just its extracted text.

---

## Features

- **TF-IDF + cosine similarity match score** (0-100%) between a resume and a job description
- **Missing-keyword detection** — surfaces high-importance JD terms absent from the resume
- **ATS compatibility checks** — contact info extractability, standard section header presence
- **Structural risk detection** (`--docx` flag) — flags tables and images in the original Word file that commonly break ATS parsers
- **Unit test suite** covering scoring behavior, keyword logic, and ATS checks

---

## How it works

```
match_resume.py
├── read_file()                  → loads resume/JD text files
├── compute_match_score()        → TF-IDF vectorization + cosine similarity
├── missing_keywords()           → JD terms with zero weight in the resume vector
├── check_ats_compatibility()    → regex checks for email, phone, section headers
├── check_docx_structure()       → inspects the .docx file for tables/images
└── main()                       → CLI wiring via argparse

tests/
└── test_match_resume.py         → unit tests for all core functions

samples/
├── sample_resume.txt            → example input
└── sample_jd.txt                → example input
```

`compute_match_score()` fits a `TfidfVectorizer` on both documents together, so the vocabulary space is shared, then computes cosine similarity between the resulting two vectors. `missing_keywords()` reuses that same vectorizer output rather than recomputing anything — it just looks for words with a nonzero JD score and a zero resume score, sorted by JD importance.

---

## Usage

Basic match score:
```bash
python3 match_resume.py samples/sample_resume.txt samples/sample_jd.txt
```

Include ATS structural checks against the original Word file:
```bash
python3 match_resume.py samples/sample_resume.txt samples/sample_jd.txt --docx path/to/resume.docx
```

Control how many missing keywords are shown (default 15):
```bash
python3 match_resume.py samples/sample_resume.txt samples/sample_jd.txt --top 20
```

### Example output

```
==================================================
ATS COMPATIBILITY CHECK
==================================================
[PASS] Email detectable: Found
[PASS] Phone number detectable: Found
[PASS] Standard section headers: Covers expected sections
[WARN] Structure (tables): 1 table(s) found — text inside tables is
frequently skipped or reordered by ATS parsers, even though it
displays fine visually.

==================================================
Match Score: 19.11%
==================================================
Weak match — resume and JD share little vocabulary.

Top missing keywords (present in JD, absent in resume):
  - learning
  - machine
  - model
  - evaluation
  - intern
  ...
```

This is a real run against an actual resume and a realistic AI Developer Intern JD — not a cherry-picked number. The low score correctly reflects a genuine vocabulary gap between a security-tooling background and ML-specific terminology, which is exactly the kind of signal this tool is meant to surface, not hide.

---

## Design decisions

- **Fitting the vectorizer on both documents together, not separately.** `TfidfVectorizer.fit_transform([resume_text, jd_text])` builds one shared vocabulary space. Fitting separately would produce vectors in different spaces that can't be meaningfully compared with cosine similarity.
- **Stop-word removal.** Without it, common filler words dominate the similarity score and drown out the meaningful technical vocabulary the score is actually meant to capture.
- **Cosine similarity over Euclidean distance.** Euclidean distance is sensitive to document length (a long JD vs. a short resume would score poorly purely on size). Cosine similarity only measures direction/proportion, which is the right invariant here.
- **Structural checks as a separate function from content checks.** `check_docx_structure()` needs the original `.docx` file and an optional dependency (`python-docx`); `check_ats_compatibility()` only needs plain text. Keeping them separate means the tool still works without `python-docx` installed — it just skips the structural layer with a note, rather than crashing.
- **`missing_keywords()` reuses the fitted vectorizer** rather than re-tokenizing from scratch, so keyword importance stays consistent with the score that was just computed.

---

## Testing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

Tests cover: score behavior at the extremes (identical documents score near 100%, unrelated documents score low), score symmetry, missing-keyword correctness (JD-only terms are flagged, shared terms are not), and every ATS check (email/phone detection, section header presence).

---

## What I'd add next

- **Stemming/lemmatization** before vectorizing, so "manage" and "managing" aren't treated as unrelated words
- **N-gram support** (e.g. matching "machine learning" as a phrase, not just two separate word hits)
- **A configurable ATS section-header list**, since expected headings vary somewhat by industry
- **Batch mode** — score one resume against multiple JDs at once, to help prioritize which roles to tailor for

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.8+. `python-docx` is only required for the `--docx` structural check; the core matcher and ATS text checks run on the standard library plus scikit-learn alone.

---

## Disclaimer

This is a learning project built to understand TF-IDF and cosine similarity hands-on, not a production ATS or a substitute for a real one. Real ATS systems (Workday, Greenhouse, Taleo, etc.) use their own proprietary parsing and scoring logic that this tool does not replicate.
