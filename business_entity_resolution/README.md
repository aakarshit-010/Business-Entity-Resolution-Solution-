# Business Entity Resolution Pipeline

## Overview
This pipeline resolves business entities across three independent data sources using supervised machine learning. Source 1 is the deduplicated reference; the system finds all matching records from Source 2 and Source 3 for each Source 1 entity.

## Architecture
1. **Normalization** — Unicode stripping, lowercasing, legal suffix removal, address abbreviation standardization
2. **Country-Partitioned Blocking** — Six complementary hash-based blocking strategies run within each country partition to generate high-recall candidate pairs
3. **Pairwise Feature Engineering** — 35 features including fuzzy string similarities (edit distance, partial ratio, token sort/set ratio), Jaccard/containment similarities, character n-gram overlap, postal code agreement, address number agreement, and interaction features
4. **LightGBM Classifier** — Gradient boosted decision tree trained on labeled candidate pairs with class imbalance handling
5. **Threshold Optimization** — Decision threshold tuned on a held-out S1-level validation split to maximize macro F₀.₅

## Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

## Reproducing Results

### From the `student_resource/` directory:

```bash
# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run the full pipeline (train + predict)
python code/business_entity_resolution/src/pipeline.py
```

This will:
1. Load and preprocess all training and test data
2. Train a LightGBM model with threshold optimization
3. Generate test predictions
4. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`

### Validate the submission:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Blocking Strategies (Union of all)
1. **Exact normalized name** — cleaned name equality
2. **Sorted token match** — sorted unique tokens equality
3. **Name prefix (4-char)** — first 4 characters of cleaned name
4. **First token match** — first word of business name
5. **Postal code match** — extracted ZIP/PIN within same country
6. **Token overlap** — ≥2 shared informative name tokens (≥1 if single-token name)

## Features (35 total)
- **Name:** exact match, edit similarity, partial ratio, token sort ratio, token set ratio, Jaccard, containment (both directions), char 3-gram Jaccard, length diff/ratio, token count diff, sorted tokens match, first token match/sim
- **Address:** exact match, edit similarity, partial ratio, token sort ratio, token set ratio, Jaccard, containment, char 3-gram Jaccard, length diff/ratio, postal match, postal both present, number agreement, both empty flag
- **Country:** exact match
- **Interactions:** name×addr product, name+addr sum, high-name-high-addr flag
- **Source:** is_S2, is_S3 indicators

## Model
- **LightGBM** (MIT License, <1M parameters)
- 500 boosting rounds, 63 leaves, learning rate 0.05
- Class imbalance handled via `is_unbalance=True`

## Key Design Decisions
- **Country partitioning** for blocking — prevents cross-country false matches and reduces comparison space
- **S1-level validation split** — prevents data leakage between train/val
- **Precision-first threshold** — F₀.₅ heavily penalizes false merges
- **No external data** — fully compliant with competition rules
- **France handled naturally** — open-set country processing, no hardcoded country logic
