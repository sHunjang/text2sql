import json
import random
from collections import defaultdict

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

rows = load_jsonl("data/processed/all_clean.jsonl")
print(f"전체 정제 데이터: {len(rows)}개")

# 정답 SQL이 같은 문항끼리 그룹으로 묶음 (같은 그룹은 항상 같은 split으로)
groups = defaultdict(list)
for row in rows:
    sql = row["messages"][2]["content"].strip()
    groups[sql].append(row)

group_keys = list(groups.keys())
random.seed(42)
random.shuffle(group_keys)

train_rows, valid_rows, test_rows = [], [], []
total = len(rows)
train_target = int(total * 0.8)
valid_target = int(total * 0.1)

for key in group_keys:
    group = groups[key]
    if len(train_rows) < train_target:
        train_rows.extend(group)
    elif len(valid_rows) < valid_target:
        valid_rows.extend(group)
    else:
        test_rows.extend(group)

def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

save("data/processed/train.jsonl", train_rows)
save("data/processed/valid.jsonl", valid_rows)
save("data/processed/test.jsonl", test_rows)

print(f"Train: {len(train_rows)}개")
print(f"Valid: {len(valid_rows)}개")
print(f"Test:  {len(test_rows)}개")
