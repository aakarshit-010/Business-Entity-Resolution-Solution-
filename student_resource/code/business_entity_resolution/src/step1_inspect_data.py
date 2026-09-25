"""Quick data inspection - reads just headers, shapes, and samples efficiently."""
import pandas as pd
import os, sys

BASE = r'c:\Users\aakar\OneDrive\Documents\Unstop amazon\student_resource'
TRAIN = os.path.join(BASE, 'dataset', 'train')
TEST = os.path.join(BASE, 'dataset', 'test')
OUT = os.path.join(BASE, 'code', 'business_entity_resolution', 'src', 'inspection_results.txt')

f = open(OUT, 'w', encoding='utf-8')
def p(msg=''):
    f.write(str(msg) + '\n')

# Count lines efficiently
def count_lines(path):
    count = 0
    with open(path, 'r', encoding='utf-8') as fh:
        for _ in fh:
            count += 1
    return count - 1  # subtract header

p("=== ROW COUNTS ===")
for name, path in [
    ('train_source1', os.path.join(TRAIN, 'train_source1.tsv')),
    ('train_source2', os.path.join(TRAIN, 'train_source2.tsv')),
    ('train_source3', os.path.join(TRAIN, 'train_source3.tsv')),
    ('train_ground_truth', os.path.join(TRAIN, 'train_ground_truth.tsv')),
    ('test_source1', os.path.join(TEST, 'test_source1.tsv')),
    ('test_source2', os.path.join(TEST, 'test_source2.tsv')),
    ('test_source3', os.path.join(TEST, 'test_source3.tsv')),
]:
    n = count_lines(path)
    p(f"  {name}: {n:,}")

# Ground truth analysis
p("\n=== GROUND TRUTH ANALYSIS ===")
gt = pd.read_csv(os.path.join(TRAIN, 'train_ground_truth.tsv'), sep='\t', dtype=str)
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
gt['n_matches'] = gt['matched_entity_ids'].apply(lambda x: len([m for m in x.split(',') if m.strip()]) if x.strip() else 0)
singletons = (gt['n_matches'] == 0).sum()
p(f"  Total S1 in GT: {len(gt):,}")
p(f"  Singletons: {singletons:,} ({100*singletons/len(gt):.1f}%)")
p(f"  With matches: {len(gt)-singletons:,}")
p(f"\n  Match count distribution:")
vc = gt['n_matches'].value_counts().sort_index()
for k, v in vc.items():
    p(f"    {k}: {v:,}")

# S2 vs S3 breakdown
s2_count = 0
s3_count = 0
for ids_str in gt['matched_entity_ids']:
    if not isinstance(ids_str, str) or not ids_str.strip():
        continue
    for m in ids_str.split(','):
        m = m.strip()
        if m.startswith('S2-'):
            s2_count += 1
        elif m.startswith('S3-'):
            s3_count += 1
p(f"\n  S1->S2 links: {s2_count:,}")
p(f"  S1->S3 links: {s3_count:,}")

# Check multi-S1 matches
from collections import defaultdict
match_to_s1 = defaultdict(list)
for _, row in gt.iterrows():
    ids_str = row['matched_entity_ids']
    if not isinstance(ids_str, str) or not ids_str.strip():
        continue
    for m in ids_str.split(','):
        m = m.strip()
        if m:
            match_to_s1[m].append(row['source1_entity_id'])
multi = sum(1 for v in match_to_s1.values() if len(v) > 1)
p(f"\n  S2/S3 matching multiple S1: {multi}")

# Sample data
p("\n=== SAMPLE DATA ===")
for name, path in [
    ('S1 train', os.path.join(TRAIN, 'train_source1.tsv')),
    ('S2 train', os.path.join(TRAIN, 'train_source2.tsv')),
    ('S3 train', os.path.join(TRAIN, 'train_source3.tsv')),
    ('S1 test', os.path.join(TEST, 'test_source1.tsv')),
]:
    df = pd.read_csv(path, sep='\t', dtype=str, nrows=5)
    p(f"\n  {name}:")
    p(f"    Columns: {list(df.columns)}")
    for _, row in df.iterrows():
        p(f"    {row.to_dict()}")

# Country distributions (scan efficiently)
p("\n=== COUNTRY DISTRIBUTIONS ===")
for name, path in [
    ('S1 train', os.path.join(TRAIN, 'train_source1.tsv')),
    ('S2 train', os.path.join(TRAIN, 'train_source2.tsv')),
    ('S3 train', os.path.join(TRAIN, 'train_source3.tsv')),
    ('S1 test', os.path.join(TEST, 'test_source1.tsv')),
    ('S2 test', os.path.join(TEST, 'test_source2.tsv')),
    ('S3 test', os.path.join(TEST, 'test_source3.tsv')),
]:
    countries = pd.read_csv(path, sep='\t', dtype=str, usecols=['country'])
    vc = countries['country'].value_counts()
    p(f"\n  {name}:")
    for c, n in vc.items():
        p(f"    {c}: {n:,}")

# Missing values
p("\n=== MISSING VALUES ===")
for name, path in [
    ('S1 train', os.path.join(TRAIN, 'train_source1.tsv')),
    ('S2 train', os.path.join(TRAIN, 'train_source2.tsv')),
    ('S3 train', os.path.join(TRAIN, 'train_source3.tsv')),
]:
    df = pd.read_csv(path, sep='\t', dtype=str)
    p(f"\n  {name}:")
    for col in df.columns:
        miss = df[col].isna().sum()
        p(f"    {col}: {miss:,} ({100*miss/len(df):.2f}%)")
    del df

# Sample matched pairs
p("\n=== SAMPLE MATCHED PAIRS ===")
s1 = pd.read_csv(os.path.join(TRAIN, 'train_source1.tsv'), sep='\t', dtype=str)
s2 = pd.read_csv(os.path.join(TRAIN, 'train_source2.tsv'), sep='\t', dtype=str)
s3 = pd.read_csv(os.path.join(TRAIN, 'train_source3.tsv'), sep='\t', dtype=str)
s1d = s1.set_index('entity_id')
s2d = s2.set_index('entity_id')
s3d = s3.set_index('entity_id')

matched_gt = gt[gt['n_matches'] > 0].head(30)
count = 0
for _, row in matched_gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1d.index:
        continue
    s1_rec = s1d.loc[s1_id]
    for mid in row['matched_entity_ids'].split(','):
        mid = mid.strip()
        if not mid:
            continue
        if mid.startswith('S2-') and mid in s2d.index:
            mrec = s2d.loc[mid]
        elif mid.startswith('S3-') and mid in s3d.index:
            mrec = s3d.loc[mid]
        else:
            continue
        count += 1
        p(f"\n  Pair {count}:")
        p(f"    S1 [{s1_id}]: {s1_rec['business_name']} | {s1_rec['business_address']} | {s1_rec['country']}")
        p(f"    Mx [{mid}]:   {mrec['business_name']} | {mrec['business_address']} | {mrec['country']}")
        if count >= 20:
            break
    if count >= 20:
        break

f.close()
print(f"Results written to: {OUT}")
# Also print the file
with open(OUT, 'r', encoding='utf-8') as fh:
    for line in fh:
        try:
            print(line.rstrip())
        except:
            print(line.encode('ascii', errors='replace').decode('ascii').rstrip())
