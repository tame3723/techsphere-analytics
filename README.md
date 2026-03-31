Alright — now we move from theory → **actual development roadmap**.
This is the exact step-by-step plan you and your friend should follow so the project builds smoothly without chaos.

---

# 🔷 Phase 0 — Environment & Collaboration Setup

### Step 1: Create Repository

Create a repo on GitHub named:

```
techsphere-analytics
```

Add your friend as collaborator.

---

### Step 2: Install Required Tools

Both of you install:

* Python (3.10+)
* Visual Studio Code
* Git
* MySQL

---

# 🔷 Phase 1 — Project Structure (Do this first)

Create this folder structure:

```
techsphere/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── refined/
│
├── src/
│   ├── cleaning/
│   ├── nlp/
│   ├── analytics/
│   └── scoring/
│
├── notebooks/
├── dashboard/
├── docs/
└── requirements.txt
```

This mirrors your pipeline.

---

# 🔷 Phase 2 — Dataset Preparation

### Step 3: Collect Raw Datasets

Put files in:

```
data/raw/
```

Example:

```
linkedin_2023.csv
linkedin_2024.csv
adzuna_2025.csv
```

---

# 🔷 Phase 3 — Data Cleaning Module

### Step 4: Write Cleaning Script

File:

```
src/cleaning/clean_jobs.py
```

This script will:

* Remove duplicates
* Drop null rows
* Normalize job titles
* Save output to:

```
data/cleaned/
```

---

# 🔷 Phase 4 — NLP Skill Extraction

### Step 5: Install NLP Libraries

In terminal:

```
pip install spacy pandas
python -m spacy download en_core_web_sm
```

---

### Step 6: Write NLP Module

File:

```
src/nlp/skill_extractor.py
```

This module:

* Loads cleaned dataset
* Extracts skills using spaCy + PhraseMatcher
* Adds new column:

```
skills_extracted
```

Output:

```
data/refined/jobs_with_skills.csv
```

---

# 🔷 Phase 5 — Domain Classification

### Step 7: Create Domain Mapping Rules

File:

```
src/nlp/domain_classifier.py
```

Example logic:

```
if "tensorflow" in skills:
    domain = "AI/ML"
elif "react" in skills:
    domain = "Web Dev"
```

This adds:

```
domain
```

column to refined dataset.

---

# 🔷 Phase 6 — Feature Engineering

### Step 8: Extract Analytical Features

File:

```
src/analytics/feature_builder.py
```

This script computes:

* experience_level
* remote_flag
* salary_band

Now dataset becomes fully structured.

---

# 🔷 Phase 7 — Metrics Computation

### Step 9: Calculate Core Indicators

File:

```
src/analytics/metrics.py
```

This computes:

* job volume per domain
* growth rate (2023 → 2024)
* skill expansion index
* median salary

Output:

```
domain_metrics.csv
```

---

# 🔷 Phase 8 — Opportunity Score Model

### Step 10: Implement TSOS

File:

```
src/scoring/opportunity_score.py
```

Formula:

```
TSOS = w1*growth + w2*volume + w3*skills + w4*salary
```

Output:

```
domain_rankings.csv
```

---

# 🔷 Phase 9 — Database Integration (Optional but good)

### Step 11: Create MySQL Schema

Tables:

```
jobs_cleaned
jobs_refined
domain_metrics
domain_scores
```

Write script:

```
src/db/upload_to_mysql.py
```

---

# 🔷 Phase 10 — Visualization / Dashboard

### Step 12: Build Dashboard

Use:
Streamlit

File:

```
dashboard/app.py
```

This will display:

* domain ranking table
* skill heatmaps
* salary comparison

---

# 🔷 Phase 11 — Testing & Validation

### Step 13: Validate Outputs

Check:

* Are duplicates removed?
* Are skills extracted correctly?
* Do domains make sense?
* Are rankings stable?

---

# 🔷 Phase 12 — Documentation

### Step 14: Write Documentation

In:

```
docs/
```

Include:

* pipeline diagram
* dataset schema
* scoring formula

This helps in viva and report writing.

---

# 🔷 Recommended Work Division

You:

* NLP + Analytics + Scoring

Friend:

* Data Cleaning + Database + Dashboard

This prevents merge conflicts.

---

# 🔷 Development Flow Summary

```
Raw Data
↓
Cleaning
↓
NLP Extraction
↓
Domain Classification
↓
Feature Engineering
↓
Metrics
↓
Opportunity Score
↓
Dashboard
```

---

# 🔷 If You Want Next

We can now start actual coding with:

* Step 1: requirements.txt
* Step 2: cleaning script template
* Step 3: skill dictionary design

Say:
**“start coding phase 1”** and we’ll build the project line-by-line.
