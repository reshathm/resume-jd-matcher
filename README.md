# Resume vs JD Matcher

A command-line tool that scores how well a resume matches a job description, using TF-IDF vectorization and cosine similarity. Built to understand core NLP text-similarity techniques from first principles — no pretrained models, no black box.

## How it works

1. Both the resume and job description are converted into TF-IDF vectors — each word is scored by how important it is to that document, weighted down if it's common across documents (via scikit-learn's `TfidfVectorizer`).
2. Cosine similarity is computed between the two vectors, producing a 0–100% match score based on the angle between them.
3. The tool also surfaces JD keywords that carry high importance but never appear in the resume, so you can see exactly what to add.

## Usage

```bash
python match_resume.py resume.txt jd.txt
```

Optional: control how many missing keywords are shown (default 15):

```bash
python match_resume.py resume.txt jd.txt --top 20
```

### Input format

Plain `.txt` files. Paste your resume text into `resume.txt` and the target job description into `jd.txt`.

## Example output

```
==================================================
Match Score: 42.15%
==================================================
Moderate match — consider tailoring your resume.

Top missing keywords (present in JD, absent in resume):
  - tensorflow
  - embeddings
  - agile
  ...
```

## Requirements

```bash
pip install scikit-learn
```

## Why this approach

TF-IDF + cosine similarity is a classical, fully explainable NLP technique — every number in the output traces back to word frequency, with no hidden model weights. It was chosen deliberately as a foundational exercise in turning text into numbers and measuring similarity mathematically, rather than reaching for a pretrained embedding model.
