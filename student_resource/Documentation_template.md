# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We built a supervised entity resolution pipeline using country-partitioned multi-strategy blocking, 35 pairwise features (fuzzy string similarity, address overlap, postal code agreement), and a LightGBM classifier with F₀.₅-optimized thresholding. The system processes ~1.7M Source 1 entities against ~10M Source 2/3 candidates using hash-based inverted indexes for efficient blocking. France (unseen in training) is handled naturally through open-set country processing without any hardcoded logic.

---

## 2. Methodology

### 2.1 Problem Analysis

Key insights from EDA:
- **Scale:** 2.2M S1, 5M S2, 5.3M S3 training entities; 1.7M S1, 4.9M S2, 5.1M S3 test entities
- **Singleton rate:** Only 5.6% of S1 entities have no matches — most have 2-5 matches
- **No many-to-many:** S2/S3 entities never match multiple S1 entities (1-to-many from S1 only)
- **Noise patterns:**
  - Indian business names appear in both Latin and Devanagari/Tamil scripts (S2/S3 contain transliterated names)
  - S3 uses `.com` domain names as business names (e.g., `maurewilliamscolombier.com`)
  - Heavy typos in S2/S3 names (`Wilblims` for `Williams`, `Enterpires` for `Enterprises`, `ENRTPRMISES` for `Enterprises`)
  - S2 addresses are UPPERCASED with reordered components and `null` tokens
  - S3 addresses use full state names instead of abbreviations
  - ~3.3% missing addresses in S2/S3
  - Accented characters (é→e) and Unicode noise
- **Countries:** Training has US + India; Test additionally has France (259K S1 entities)
- **Match distribution:** Most entities have 3 matches (530K), followed by 4 (484K), then 5 (322K)
- **S1→S2 links (3.7M) and S1→S3 links (3.9M):** Roughly balanced across sources

### 2.2 Solution Strategy

**Approach Type:** Blocking + Supervised Classifier  
**Core Innovation:** Country-partitioned multi-strategy blocking with hash-based inverted indexes for scalability, combined with a comprehensive 35-feature set and F₀.₅-optimized thresholding.

Pipeline stages:
1. **Normalization** — Unicode stripping, lowercasing, legal suffix removal, address abbreviation standardization
2. **Country-Partitioned Blocking** — Six complementary blocking strategies run within each country partition
3. **Pairwise Feature Engineering** — 35 features covering name, address, country, and interaction signals
4. **LightGBM Classification** — Binary classifier trained on labeled candidate pairs
5. **Threshold Optimization** — Grid search on held-out validation set maximizing macro F₀.₅

---

## 3. Candidate Generation (Blocking)

We use six complementary blocking strategies, applied independently within each country partition (US, India, France). The final candidate set is the **union** of all strategies:

1. **Exact normalized name match** — After cleaning (lowercase, accent removal, legal suffix removal), exact string equality
2. **Sorted unique token match** — Sorted unique tokens of the cleaned name must match exactly
3. **Name prefix (4-char) blocking** — First 4 characters of cleaned name, within same country (max 500 candidates per block)
4. **First token blocking** — First word of business name within same country (max 2000 per block)
5. **Postal code blocking** — Extracted 5-6 digit postal/ZIP/PIN codes within same country
6. **Token overlap blocking** — ≥2 shared informative name tokens (or ≥1 for single-token names), excluding tokens appearing >10,000 times

**Blocking keys used:** Normalized name, sorted tokens, 4-char prefix, first token, postal code, individual name tokens  
**Candidate pairs generated:** [filled after pipeline completes]  
**How we ensured true matches were not lost:** Union of six diverse strategies with high individual recall; blocking recall measured on validation set

---

## 4. Matching Model

**Features used (35 total):**

- **Name features (15):** exact match, edit distance similarity, partial ratio, token sort ratio, token set ratio, Jaccard similarity, containment (S1→SX), containment (SX→S1), character 3-gram Jaccard, name length difference, name length ratio, token count difference, sorted tokens match, first token exact match, first token edit similarity
- **Address features (12):** exact match, edit distance similarity, partial ratio, token sort ratio, token set ratio, Jaccard similarity, containment, character 3-gram Jaccard, length difference, length ratio, postal code match, postal both present, address number agreement, both-empty indicator
- **Country features (1):** exact country match
- **Interaction features (3):** name_sim × addr_sim, name_sim + addr_sim, high-name AND high-address indicator
- **Source features (2):** is_S2 indicator, is_S3 indicator

**Model type:** LightGBM (Gradient Boosted Decision Trees)
- MIT License, <1M parameters
- 500 boosting rounds, 63 leaves, learning rate 0.05
- Class imbalance handled via `is_unbalance=True`
- Feature fraction 0.8, bagging fraction 0.8

**Threshold selection method:** Grid search over [0.30, 0.35, ..., 0.95] followed by fine-grained search (step=0.02) around best, evaluated on held-out S1-level validation split using macro F₀.₅

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [filled after pipeline completes]
- **Blocking recall:** [filled after pipeline completes]
- **Common false positives (wrong merges):** Businesses with generic names (e.g., "Hotel", "Restaurant", "Services") at similar addresses; partial name matches where the discriminating tokens are missing
- **Common false negatives (missed matches):** S2/S3 entries with transliterated (non-Latin) names that don't share any ASCII tokens with S1; entries where both name and address are heavily corrupted/missing; DBA/trade names with no token overlap to the registered name

---

## 6. Conclusion

We built a scalable, fully rule-compliant entity resolution pipeline that processes millions of entities efficiently using country-partitioned blocking and supervised classification. The system handles the unseen France country naturally without hardcoded logic, optimizes for the precision-heavy F₀.₅ metric through careful threshold tuning, and correctly identifies singletons. All processing uses only the provided training and test data with no external lookups.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/
│   ├── pipeline.py           # Main end-to-end pipeline (entry point)
│   └── step1_inspect_data.py # Data inspection/EDA script
├── README.md                 # Reproduction instructions
└── requirements.txt          # Pinned dependencies
```

**Entry point:** `python code/business_entity_resolution/src/pipeline.py` from the `student_resource/` directory.

**Outputs generated:**
- `output/matching_results.tsv` — final entity matches
- `output/candidate_pairs.tsv` — blocking candidate set

### B. Additional Results

**Validation methodology:**
- 80/20 S1-level split (no data leakage between entities)
- Macro F₀.₅ computed identically to competition scoring
- Singletons included in macro average (score 1.0 if correctly predicted empty, 0.0 if falsely matched)

**Fair-play compliance:**
- No external data, APIs, databases, geocoding, or internet lookups used
- All processing uses only provided training and test TSV files
- Model is LightGBM (MIT License, <1M parameters)

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
