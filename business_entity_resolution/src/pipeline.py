"""
Business Entity Resolution Pipeline - Optimized for Scale
==========================================================
Handles 1.7M S1 x 10M S2+S3 with hash-based blocking and vectorized features.

Key design decisions:
- Country-partitioned processing (US/India/France independently)
- Hash-based inverted index blocking (no Cartesian product)
- Vectorized feature computation using rapidfuzz
- LightGBM classifier
- Threshold optimized for macro F0.5
"""

import pandas as pd
import numpy as np
import os
import re
import gc
import time
import unicodedata
import warnings
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(BASE, '..', '..'))
TRAIN_DIR = os.path.join(WORKSPACE_ROOT, 'student_resource', 'student_resource', 'dataset', 'train')
TEST_DIR = os.path.join(WORKSPACE_ROOT, 'student_resource', 'student_resource', 'dataset', 'test')
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIX_RE = re.compile(
    r'\b(?:incorporated|inc\.?|corporation|corp\.?|company|co\.?|llc|llp|'
    r'ltd\.?|limited|pvt\.?|private|public|enterprises?|group|holdings?|'
    r'sarl|sas|sa|eurl|sci|gmbh|ag|'
    r'd/?b/?a)\b', re.IGNORECASE
)

ADDR_ABBREVS = {
    'street': 'st', 'road': 'rd', 'avenue': 'ave', 'boulevard': 'blvd',
    'drive': 'dr', 'lane': 'ln', 'court': 'ct', 'place': 'pl',
    'circle': 'cir', 'highway': 'hwy', 'parkway': 'pkwy', 'terrace': 'ter',
    'trail': 'trl', 'square': 'sq', 'apartment': 'apt', 'suite': 'ste',
    'floor': 'fl', 'building': 'bldg', 'room': 'rm',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
}
# Add dotted versions
for k in list(ADDR_ABBREVS.keys()):
    v = ADDR_ABBREVS[k]
    ADDR_ABBREVS[f'{v}.'] = v


def strip_accents(s):
    """Remove accents but keep base chars."""
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(name):
    """Basic name normalization."""
    if not isinstance(name, str) or not name.strip():
        return ''
    s = strip_accents(name).lower().strip()
    s = s.replace('&', ' and ').replace('@', ' at ')
    s = re.sub(r"'s\b", 's', s)
    s = re.sub(r'[^a-z0-9\s-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_name_no_suffix(name):
    """Name without legal suffixes."""
    s = normalize_name(name)
    if not s:
        return ''
    s = LEGAL_SUFFIX_RE.sub('', s)
    s = re.sub(r'[\s,.-]+$', '', s)
    s = re.sub(r'^[\s,.-]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_addr(addr):
    """Address normalization."""
    if not isinstance(addr, str) or not addr.strip():
        return ''
    s = strip_accents(addr).lower().strip()
    s = re.sub(r'[^a-z0-9\s,.-/]', ' ', s)
    # Normalize common abbreviations
    tokens = s.split()
    out = []
    for t in tokens:
        t_clean = t.rstrip('.,')
        out.append(ADDR_ABBREVS.get(t_clean, t_clean))
    s = ' '.join(out)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_postal(addr):
    """Extract postal/ZIP/PIN code."""
    if not isinstance(addr, str):
        return ''
    # 6-digit PIN (India) or 5-digit ZIP (US/France)
    codes = re.findall(r'\b(\d{5,6})\b', addr)
    return codes[-1] if codes else ''


def extract_numbers(addr):
    """Extract numeric tokens from address."""
    if not isinstance(addr, str):
        return ''
    nums = re.findall(r'\d+', addr)
    return ' '.join(nums)


def sorted_tokens(s):
    """Sorted unique tokens of length > 1."""
    if not s:
        return ''
    tokens = sorted(set(t for t in s.split() if len(t) > 1))
    return ' '.join(tokens)


# ============================================================
# VECTORIZED PREPROCESSING
# ============================================================

def preprocess_df(df):
    """Preprocess a source dataframe - vectorized operations."""
    df = df.copy()
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('').str.strip()

    log(f"  Normalizing names ({len(df)} rows)...")
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['name_clean'] = df['business_name'].apply(normalize_name_no_suffix)
    df['name_sorted'] = df['name_clean'].apply(sorted_tokens)

    log(f"  Normalizing addresses...")
    df['addr_norm'] = df['business_address'].apply(normalize_addr)
    df['postal'] = df['business_address'].apply(extract_postal)
    df['addr_nums'] = df['business_address'].apply(extract_numbers)

    # Blocking keys
    df['name_prefix4'] = df['name_clean'].str[:4]
    df['name_prefix3'] = df['name_clean'].str[:3]
    df['first_token'] = df['name_clean'].apply(lambda x: x.split()[0] if x.split() else '')

    return df


# ============================================================
# BLOCKING (Hash-based for speed)
# ============================================================

def build_block_index(sx_df, block_key_func, max_block_size=500):
    """Build inverted index: block_key -> list of entity_ids."""
    idx = defaultdict(list)
    for eid, row in zip(sx_df['entity_id'].values, sx_df.itertuples(index=False)):
        keys = block_key_func(row)
        for k in keys:
            if k:
                idx[k].append(eid)

    # Remove overly large blocks
    idx = {k: v for k, v in idx.items() if len(v) <= max_block_size}
    return idx


import pickle

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def save_checkpoint(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def generate_candidates_country(s1_country_df, sx_country_df, country, max_per_s1=100, prefix="train"):
    """Generate candidates within a single country partition (strategy by strategy)."""
    log(f"  Blocking for {country} ({prefix}): {len(s1_country_df)} S1 x {len(sx_country_df)} SX")
    
    chk_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    os.makedirs(chk_dir, exist_ok=True)
    
    s1_ids = s1_country_df['entity_id'].values
    s1_names = s1_country_df['name_clean'].values
    s1_sorted = s1_country_df['name_sorted'].values
    s1_prefix4 = s1_country_df['name_prefix4'].values
    s1_ft = s1_country_df['first_token'].values
    s1_postal = s1_country_df['postal'].values
    
    all_cands = {sid: set() for sid in s1_ids}
    
    strategies = [
        ('exact_name', 'name_clean'),
        ('sorted_tokens', 'name_sorted'),
        ('name_prefix4', 'name_prefix4'),
        ('first_token', 'first_token'),
        ('postal', 'postal'),
        ('token_overlap', 'name_clean')
    ]
    
    for strat_idx, (strat_name, col_name) in enumerate(strategies):
        strat_num = strat_idx + 1
        chk_file = os.path.join(chk_dir, f"{prefix}_{country}_strat{strat_num}_{strat_name}.pkl")
        legacy_chk_file = os.path.join(chk_dir, f"{country}_strat{strat_num}_{strat_name}.pkl")
        
        strat_cands = None
        if os.path.exists(chk_file):
            strat_cands = load_checkpoint(chk_file)
        elif prefix == "train" and os.path.exists(legacy_chk_file):
            strat_cands = load_checkpoint(legacy_chk_file)
            
        if strat_cands is not None:
            log(f"    Loaded checkpoint for {country} Strategy {strat_num} ({strat_name})")
            for sid, cset in strat_cands.items():
                if sid in all_cands:
                    all_cands[sid].update(cset)
            del strat_cands
            gc.collect()
            continue
            
        log(f"    Running {country} Strategy {strat_num} ({strat_name})...")
        strat_cands = defaultdict(set)
        
        if strat_num == 6:  # token overlap (NumPy vectorized)
            t_idx_start = time.time()
            sx_eids = list(sx_country_df['entity_id'].values)
            name_arr = sx_country_df[col_name].values
            
            raw_idx = defaultdict(list)
            for i, name in enumerate(name_arr):
                if name and isinstance(name, str):
                    toks = set(t for t in name.split() if len(t) > 2)
                    for t in toks:
                        raw_idx[t].append(i)
            
            # Cap posting list size at 5000 and store as sorted int32 numpy arrays
            idx = {k: np.array(v, dtype=np.int32) for k, v in raw_idx.items() if len(v) <= 5000}
            del raw_idx
            gc.collect()
            log(f"      Strat 6 index built: {len(idx)} tokens (cap 5000) in {time.time() - t_idx_start:.1f}s")
            
            t_scan_start = time.time()
            n_s1 = len(s1_ids)
            for i in range(n_s1):
                if i % 100000 == 0 and i > 0:
                    log(f"      Strat 6 progress: {i}/{n_s1} ({time.time() - t_scan_start:.1f}s)")
                
                name = s1_names[i]
                if not name or not isinstance(name, str):
                    continue
                
                toks = [t for t in set(name.split()) if len(t) > 2]
                if not toks:
                    continue
                
                min_overlap = min(2, len(toks))
                postings = [idx[t] for t in toks if t in idx]
                
                if len(postings) < min_overlap:
                    continue
                
                cands = set()
                if min_overlap == 1:
                    cand_inds = postings[0]
                    if len(cand_inds) > max_per_s1:
                        cand_inds = cand_inds[:max_per_s1]
                    cands = {sx_eids[j] for j in cand_inds}
                elif len(postings) == 2:
                    cand_inds = np.intersect1d(postings[0], postings[1], assume_unique=True)
                    if len(cand_inds) > 0:
                        if len(cand_inds) > max_per_s1:
                            cand_inds = cand_inds[:max_per_s1]
                        cands = {sx_eids[j] for j in cand_inds}
                else:
                    concat = np.concatenate(postings)
                    concat.sort()
                    dup_mask = (concat[:-1] == concat[1:])
                    if np.any(dup_mask):
                        cand_inds = np.unique(concat[:-1][dup_mask])
                        if len(cand_inds) > max_per_s1:
                            cand_inds = cand_inds[:max_per_s1]
                        cands = {sx_eids[j] for j in cand_inds}
                
                if cands:
                    s1_id = s1_ids[i]
                    strat_cands[s1_id] = cands
                    all_cands[s1_id].update(cands)
            
            del idx, sx_eids
            gc.collect()
            log(f"      Strat 6 scan complete in {time.time() - t_scan_start:.1f}s")
            
        else:
            # Build index for strategies 1-5
            idx = defaultdict(list)
            for eid, val in zip(sx_country_df['entity_id'].values, sx_country_df[col_name].values):
                if val:
                    if strat_num == 3 and len(val) < 4:
                        continue
                    if strat_num == 4 and len(val) <= 2:
                        continue
                    idx[val].append(eid)
            if strat_num == 3:
                idx = {k: v for k, v in idx.items() if len(v) <= 500}
            elif strat_num == 4:
                idx = {k: v for k, v in idx.items() if len(v) <= 2000}
            
            # Scan S1 for strategies 1-5
            t_scan_strat = time.time()
            n_s1 = len(s1_ids)
            for i in range(n_s1):
                if i % 250000 == 0 and i > 0:
                    log(f"      Strat {strat_num} progress: {i}/{n_s1} ({time.time() - t_scan_strat:.1f}s)")

                s1_id = s1_ids[i]
                cands = set()
                
                if strat_num == 1:
                    v = s1_names[i]
                elif strat_num == 2:
                    v = s1_sorted[i]
                elif strat_num == 3:
                    v = s1_prefix4[i]
                elif strat_num == 4:
                    v = s1_ft[i]
                elif strat_num == 5:
                    v = s1_postal[i]
                else:
                    v = None

                if v and v in idx:
                    matched = idx[v]
                    if len(matched) > max_per_s1:
                        cands = set(matched[:max_per_s1])
                    else:
                        cands = set(matched)
                
                if cands:
                    strat_cands[s1_id] = cands
                    all_cands[s1_id].update(cands)
            
            del idx
            gc.collect()
            log(f"      Strat {strat_num} scan complete in {time.time() - t_scan_strat:.1f}s")
            
        # Save checkpoint and free memory
        save_checkpoint(dict(strat_cands), chk_file)
        del strat_cands
        gc.collect()
        log(f"    Saved checkpoint for {country} Strategy {strat_num} ({strat_name})")

    # Cap candidates per S1
    capped = {}
    for sid, cset in all_cands.items():
        if len(cset) > max_per_s1:
            capped[sid] = set(list(cset)[:max_per_s1])
        elif cset:
            capped[sid] = cset
            
    return capped


def generate_all_candidates(s1_df, sx_df, max_per_s1=100, prefix="train"):
    """Generate candidates partitioned by country."""
    all_candidates = {}
    countries = sorted(s1_df['country'].unique())

    for country in countries:
        s1_c = s1_df[s1_df['country'] == country].reset_index(drop=True)
        sx_c = sx_df[sx_df['country'] == country].reset_index(drop=True)
        if len(s1_c) == 0 or len(sx_c) == 0:
            continue
        cands = generate_candidates_country(s1_c, sx_c, country, max_per_s1=max_per_s1, prefix=prefix)
        all_candidates.update(cands)
        del s1_c, sx_c
        gc.collect()

    return all_candidates


# ============================================================
# FEATURE COMPUTATION (Batch with rapidfuzz)
# ============================================================

def compute_features_for_pairs(pairs_s1_ids, pairs_sx_ids, s1_dict, sx_dict):
    """Compute features for arrays of pair IDs. Returns feature matrix."""
    from rapidfuzz import fuzz

    n = len(pairs_s1_ids)
    # Feature list
    feat_names = [
        'name_exact', 'name_edit_sim', 'name_partial', 'name_token_sort',
        'name_token_set', 'name_jaccard', 'name_contain_s1', 'name_contain_sx',
        'name_char3gram', 'name_len_diff', 'name_len_ratio', 'name_tok_diff',
        'name_sorted_match', 'first_tok_match', 'first_tok_sim',
        'addr_exact', 'addr_edit_sim', 'addr_partial', 'addr_token_sort',
        'addr_token_set', 'addr_jaccard', 'addr_contain_s1',
        'addr_char3gram', 'addr_len_diff', 'addr_len_ratio',
        'postal_match', 'postal_both', 'addr_num_agree',
        'addr_both_empty',
        'country_match',
        'name_x_addr', 'name_plus_addr', 'high_name_high_addr',
        'is_s2', 'is_s3',
    ]
    X = np.zeros((n, len(feat_names)), dtype=np.float32)

    for i in range(n):
        if i % 200000 == 0 and i > 0:
            log(f"    Features: {i}/{n}")

        s1_id = pairs_s1_ids[i]
        sx_id = pairs_sx_ids[i]
        s1 = s1_dict.get(s1_id)
        sx = sx_dict.get(sx_id)
        if s1 is None or sx is None:
            continue

        n1 = s1['name_clean']
        n2 = sx['name_clean']
        a1 = s1['addr_norm']
        a2 = sx['addr_norm']

        # Name features
        X[i, 0] = 1.0 if n1 and n1 == n2 else 0.0
        X[i, 1] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        X[i, 2] = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        X[i, 3] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        X[i, 4] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0

        # Token Jaccard
        t1 = set(n1.split()) if n1 else set()
        t2 = set(n2.split()) if n2 else set()
        inter = len(t1 & t2)
        union = len(t1 | t2)
        X[i, 5] = inter / union if union > 0 else 0.0
        X[i, 6] = inter / len(t1) if t1 else 0.0  # containment s1
        X[i, 7] = inter / len(t2) if t2 else 0.0  # containment sx

        # Char 3-gram Jaccard
        if n1 and n2 and len(n1) >= 3 and len(n2) >= 3:
            ng1 = {n1[j:j+3] for j in range(len(n1)-2)}
            ng2 = {n2[j:j+3] for j in range(len(n2)-2)}
            ng_inter = len(ng1 & ng2)
            ng_union = len(ng1 | ng2)
            X[i, 8] = ng_inter / ng_union if ng_union > 0 else 0.0

        # Length features
        X[i, 9] = abs(len(n1) - len(n2))
        X[i, 10] = min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 1.0
        X[i, 11] = abs(len(t1) - len(t2))

        # Sorted tokens match
        X[i, 12] = 1.0 if s1['name_sorted'] and s1['name_sorted'] == sx['name_sorted'] else 0.0

        # First token
        ft1 = s1['first_token']
        ft2 = sx['first_token']
        X[i, 13] = 1.0 if ft1 and ft1 == ft2 else 0.0
        X[i, 14] = fuzz.ratio(ft1, ft2) / 100.0 if ft1 and ft2 else 0.0

        # Address features
        X[i, 15] = 1.0 if a1 and a1 == a2 else 0.0
        X[i, 16] = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        X[i, 17] = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        X[i, 18] = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        X[i, 19] = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

        at1 = set(a1.split()) if a1 else set()
        at2 = set(a2.split()) if a2 else set()
        a_inter = len(at1 & at2)
        a_union = len(at1 | at2)
        X[i, 20] = a_inter / a_union if a_union > 0 else 0.0
        X[i, 21] = a_inter / len(at1) if at1 else 0.0

        if a1 and a2 and len(a1) >= 3 and len(a2) >= 3:
            ang1 = {a1[j:j+3] for j in range(len(a1)-2)}
            ang2 = {a2[j:j+3] for j in range(len(a2)-2)}
            ang_i = len(ang1 & ang2)
            ang_u = len(ang1 | ang2)
            X[i, 22] = ang_i / ang_u if ang_u > 0 else 0.0

        X[i, 23] = abs(len(a1) - len(a2))
        X[i, 24] = min(len(a1), len(a2)) / max(len(a1), len(a2)) if max(len(a1), len(a2)) > 0 else 1.0

        # Postal
        pc1 = s1['postal']
        pc2 = sx['postal']
        X[i, 25] = 1.0 if pc1 and pc1 == pc2 else 0.0
        X[i, 26] = 1.0 if pc1 and pc2 else 0.0

        # Address number agreement
        an1 = set(s1['addr_nums'].split()) if s1['addr_nums'] else set()
        an2 = set(sx['addr_nums'].split()) if sx['addr_nums'] else set()
        if an1 and an2:
            X[i, 27] = len(an1 & an2) / len(an1 | an2)
        elif not an1 and not an2:
            X[i, 27] = 1.0

        X[i, 28] = 1.0 if not a1 and not a2 else 0.0

        # Country
        X[i, 29] = 1.0 if s1['country'] == sx['country'] else 0.0

        # Interactions
        X[i, 30] = X[i, 1] * X[i, 16]  # name_edit * addr_edit
        X[i, 31] = X[i, 1] + X[i, 16]  # name_edit + addr_edit
        X[i, 32] = 1.0 if X[i, 1] > 0.8 and X[i, 16] > 0.6 else 0.0

        # Source
        X[i, 33] = 1.0 if sx_id.startswith('S2-') else 0.0
        X[i, 34] = 1.0 if sx_id.startswith('S3-') else 0.0

    return X, feat_names


def df_to_dict(df):
    """Convert preprocessed df to dict of dicts for fast lookup."""
    result = {}
    cols = ['name_clean', 'name_sorted', 'addr_norm', 'postal', 'addr_nums',
            'first_token', 'country', 'name_norm']
    for i in range(len(df)):
        eid = df['entity_id'].iloc[i]
        result[eid] = {c: df[c].iloc[i] for c in cols}
    return result


# ============================================================
# EVALUATION
# ============================================================

def f05_entity(pred_set, true_set):
    """F0.5 for single entity."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set:
        return 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)


def macro_f05(preds, truth):
    """Macro F0.5 over all entities."""
    scores = []
    for s1_id in truth:
        p = preds.get(s1_id, set())
        t = truth[s1_id]
        scores.append(f05_entity(p, t))
    return np.mean(scores)


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 60)
    log("BUSINESS ENTITY RESOLUTION PIPELINE")
    log("=" * 60)

    # ---- Load train ----
    log("Loading training data...")
    s1_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str)
    s2_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str)
    s3_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str)
    gt_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    log(f"S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}")

    # Parse ground truth
    gt = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        ids_str = row['matched_entity_ids']
        if isinstance(ids_str, str) and ids_str.strip():
            gt[s1_id] = set(m.strip() for m in ids_str.split(',') if m.strip())
        else:
            gt[s1_id] = set()
    del gt_df

    total_links = sum(len(v) for v in gt.values())
    singletons = sum(1 for v in gt.values() if not v)
    log(f"GT: {len(gt)} entities, {total_links} links, {singletons} singletons ({100*singletons/len(gt):.1f}%)")

    # ---- Preprocess ----
    log("Preprocessing S1...")
    s1_train = preprocess_df(s1_train)
    log("Preprocessing S2...")
    s2_train = preprocess_df(s2_train)
    log("Preprocessing S3...")
    s3_train = preprocess_df(s3_train)

    # Combine S2+S3
    sx_train = pd.concat([s2_train, s3_train], ignore_index=True)
    del s2_train, s3_train
    gc.collect()
    log(f"Combined SX: {len(sx_train)}")

    # ---- Train/Val split (S1-level) ----
    log("Creating train/val split...")
    all_s1 = list(gt.keys())
    np.random.shuffle(all_s1)
    split = int(0.8 * len(all_s1))
    train_ids = set(all_s1[:split])
    val_ids = set(all_s1[split:])
    log(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # ---- Blocking ----
    log("Generating candidates...")
    all_cands = generate_all_candidates(s1_train, sx_train, prefix="train")

    # Measure blocking recall
    tp_block = 0
    total_true = 0
    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        cands = all_cands.get(s1_id, set())
        tp_block += len(true_matches & cands)
        total_true += len(true_matches)

    block_recall = tp_block / total_true if total_true > 0 else 0
    total_cand_pairs = sum(len(v) for v in all_cands.values())
    log(f"Blocking recall: {block_recall:.4f} ({tp_block}/{total_true})")
    log(f"Total candidate pairs: {total_cand_pairs}")

    # ---- Build lookup dicts ----
    log("Building lookup dictionaries...")
    s1_dict = df_to_dict(s1_train)
    sx_dict = df_to_dict(sx_train)
    del s1_train, sx_train
    gc.collect()

    # ---- Generate training pairs ----
    log("Generating training pairs...")
    train_s1_list = []
    train_sx_list = []
    train_labels = []

    for s1_id in train_ids:
        true_matches = gt[s1_id]
        cands = all_cands.get(s1_id, set())

        # Positives (from candidates)
        for mid in true_matches & cands:
            train_s1_list.append(s1_id)
            train_sx_list.append(mid)
            train_labels.append(1)

        # Positives not in candidates (augmentation)
        for mid in true_matches - cands:
            if mid in sx_dict:
                train_s1_list.append(s1_id)
                train_sx_list.append(mid)
                train_labels.append(1)

        # Negatives
        neg = cands - true_matches
        max_neg = max(3, 3 * len(true_matches))
        neg_list = list(neg)
        if len(neg_list) > max_neg:
            neg_list = list(np.random.choice(neg_list, max_neg, replace=False))
        for nid in neg_list:
            train_s1_list.append(s1_id)
            train_sx_list.append(nid)
            train_labels.append(0)

    log(f"Training pairs: {len(train_labels)} (pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)})")

    # ---- Compute training features ----
    log("Computing training features...")
    X_train, feat_names = compute_features_for_pairs(
        train_s1_list, train_sx_list, s1_dict, sx_dict
    )
    y_train = np.array(train_labels, dtype=np.float32)
    del train_s1_list, train_sx_list, train_labels
    gc.collect()
    log(f"Training features: {X_train.shape}")

    # ---- Generate validation pairs ----
    log("Generating validation pairs...")
    val_s1_list = []
    val_sx_list = []
    val_labels = []

    for s1_id in val_ids:
        true_matches = gt[s1_id]
        cands = all_cands.get(s1_id, set())

        for cid in cands:
            val_s1_list.append(s1_id)
            val_sx_list.append(cid)
            val_labels.append(1 if cid in true_matches else 0)

        # Add true matches not in candidates
        for mid in true_matches - cands:
            if mid in sx_dict:
                val_s1_list.append(s1_id)
                val_sx_list.append(mid)
                val_labels.append(1)

    log(f"Validation pairs: {len(val_labels)}")

    log("Computing validation features...")
    X_val, _ = compute_features_for_pairs(val_s1_list, val_sx_list, s1_dict, sx_dict)
    y_val = np.array(val_labels, dtype=np.float32)

    # ---- Train model ----
    log("Training LightGBM...")
    import lightgbm as lgb

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
        'is_unbalance': True,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dtrain], callbacks=[lgb.log_evaluation(100)])

    # Feature importance
    imp = sorted(zip(feat_names, model.feature_importance(importance_type='gain')), key=lambda x: -x[1])
    log("Top features:")
    for fn, iv in imp[:10]:
        log(f"  {fn}: {iv:.0f}")

    del X_train, y_train, dtrain
    gc.collect()

    # ---- Threshold optimization ----
    log("Optimizing threshold...")
    val_probs = model.predict(X_val)

    # Group by S1
    val_preds = defaultdict(list)
    for s1_id, sx_id, prob in zip(val_s1_list, val_sx_list, val_probs):
        val_preds[s1_id].append((sx_id, prob))

    del X_val, y_val, val_s1_list, val_sx_list, val_labels, val_probs
    gc.collect()

    best_thr = 0.5
    best_f05 = 0.0

    for thr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = {}
        for s1_id in val_ids:
            items = val_preds.get(s1_id, [])
            preds[s1_id] = set(sid for sid, p in items if p >= thr)

        val_gt = {s1: gt[s1] for s1 in val_ids}
        f = macro_f05(preds, val_gt)
        n_match = sum(len(v) for v in preds.values())
        n_sing = sum(1 for v in preds.values() if not v)
        log(f"  thr={thr:.2f} F0.5={f:.4f} matches={n_match} singletons={n_sing}")
        if f > best_f05:
            best_f05 = f
            best_thr = thr

    # Fine search
    for thr in np.arange(max(0.1, best_thr - 0.1), min(0.99, best_thr + 0.11), 0.02):
        preds = {}
        for s1_id in val_ids:
            items = val_preds.get(s1_id, [])
            preds[s1_id] = set(sid for sid, p in items if p >= thr)
        val_gt = {s1: gt[s1] for s1 in val_ids}
        f = macro_f05(preds, val_gt)
        if f > best_f05:
            best_f05 = f
            best_thr = thr

    log(f"Best threshold: {best_thr:.3f}, Val F0.5: {best_f05:.4f}")
    del val_preds
    gc.collect()

    # ---- Retrain on full data ----
    log("Retraining on full training data...")

    full_s1_list = []
    full_sx_list = []
    full_labels = []

    for s1_id in gt:
        true_matches = gt[s1_id]
        cands = all_cands.get(s1_id, set())

        for mid in true_matches & cands:
            full_s1_list.append(s1_id)
            full_sx_list.append(mid)
            full_labels.append(1)
        for mid in true_matches - cands:
            if mid in sx_dict:
                full_s1_list.append(s1_id)
                full_sx_list.append(mid)
                full_labels.append(1)
        neg = list(cands - true_matches)
        max_neg = max(3, 3 * len(true_matches))
        if len(neg) > max_neg:
            neg = list(np.random.choice(neg, max_neg, replace=False))
        for nid in neg:
            full_s1_list.append(s1_id)
            full_sx_list.append(nid)
            full_labels.append(0)

    log(f"Full training pairs: {len(full_labels)}")

    log("Computing full training features...")
    X_full, _ = compute_features_for_pairs(full_s1_list, full_sx_list, s1_dict, sx_dict)
    y_full = np.array(full_labels, dtype=np.float32)
    del full_s1_list, full_sx_list, full_labels
    gc.collect()

    dtrain_full = lgb.Dataset(X_full, label=y_full, feature_name=feat_names)
    final_model = lgb.train(params, dtrain_full, num_boost_round=500,
                            valid_sets=[dtrain_full], callbacks=[lgb.log_evaluation(100)])

    del X_full, y_full, dtrain_full, s1_dict, sx_dict, all_cands
    gc.collect()

    # ---- Test inference ----
    log("Loading test data...")
    s1_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t', dtype=str)
    s2_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t', dtype=str)
    s3_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t', dtype=str)
    log(f"Test S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

    log("Preprocessing test data...")
    s1_test = preprocess_df(s1_test)
    s2_test = preprocess_df(s2_test)
    s3_test = preprocess_df(s3_test)

    sx_test = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test
    gc.collect()

    log("Generating test candidates...")
    test_cands = generate_all_candidates(s1_test, sx_test, prefix="test")
    total_test_pairs = sum(len(v) for v in test_cands.values())
    log(f"Test candidate pairs: {total_test_pairs}")

    log("Building test lookup dicts...")
    s1_test_dict = df_to_dict(s1_test)
    sx_test_dict = df_to_dict(sx_test)
    all_s1_test = set(s1_test['entity_id'].tolist())
    del s1_test, sx_test
    gc.collect()

    log("Scoring test pairs...")
    test_s1_list = []
    test_sx_list = []
    for s1_id, cset in test_cands.items():
        for cid in cset:
            test_s1_list.append(s1_id)
            test_sx_list.append(cid)

    log(f"Total test pairs to score: {len(test_s1_list)}")

    X_test, _ = compute_features_for_pairs(test_s1_list, test_sx_list, s1_test_dict, sx_test_dict)
    test_probs = final_model.predict(X_test)

    # Group by S1
    test_preds = defaultdict(list)
    for s1_id, sx_id, prob in zip(test_s1_list, test_sx_list, test_probs):
        test_preds[s1_id].append((sx_id, prob))

    del X_test, test_probs, test_s1_list, test_sx_list
    gc.collect()

    # ---- Write outputs ----
    log("Writing output files...")

    matching_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in sorted(all_s1_test):
            items = test_preds.get(s1_id, [])
            matched = sorted(sid for sid, p in items if p >= best_thr)
            f.write(f"{s1_id}\t{','.join(matched)}\n")

    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in sorted(all_s1_test):
            cands = sorted(test_cands.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    n_matches = sum(1 for s1 in all_s1_test for sid, p in test_preds.get(s1, []) if p >= best_thr)
    n_singletons = sum(1 for s1 in all_s1_test if not any(p >= best_thr for _, p in test_preds.get(s1, [])))

    log("=" * 60)
    log("FINAL RESULTS")
    log(f"  Test S1 count: {len(all_s1_test)}")
    log(f"  Candidate pairs: {total_test_pairs}")
    log(f"  Predicted matches: {n_matches}")
    log(f"  Predicted singletons: {n_singletons}")
    log(f"  Threshold: {best_thr:.3f}")
    log(f"  Val macro F0.5: {best_f05:.4f}")
    log(f"  Output: {matching_path}")
    log(f"  Output: {candidate_path}")
    log("=" * 60)


if __name__ == '__main__':
    main()
