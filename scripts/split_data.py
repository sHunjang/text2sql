"""
split_data.py

이 스크립트가 하는 일:
    검증을 통과한 전체 데이터(288개)를 아래 세 그룹으로 나눕니다.
        - train.jsonl (약 80%): 모델이 실제로 학습(파라미터 업데이트)에 쓰는 데이터
        - valid.jsonl (약 10%): 학습 도중 "처음 보는 문제를 잘 푸는지" 확인하는 데이터
        - test.jsonl  (약 10%): 학습이 다 끝난 뒤, 딱 한 번 최종 실력을 재는 데이터

왜 이렇게 나누는가 (MT-4 개념):
    같은 데이터로 학습도 하고 평가도 하면, 모델이 "문제를 푼 게" 아니라
    "답을 외운 것"인지 구분할 수 없습니다. 그래서 학습에 쓰지 않은
    데이터로 성능을 재야, 실제 실력을 제대로 확인할 수 있습니다.

한 가지 추가로 신경 쓴 부분:
    표현만 다르고 정답 SQL이 완전히 똑같은 "쌍둥이 문항"들이 있었습니다.
    (예: "실행 상태별 건수를 요약해서 보여주세요" 와
         "실행 상태별로 몇 건씩 있는지 집계해줘"는 정답 SQL이 동일함)
    이런 쌍둥이 문항이 하나는 train에, 하나는 valid/test에 흩어져 들어가면,
    모델이 진짜 실력이 좋아서가 아니라 "거의 외운 문장이라" 잘 맞히는
    착시가 생길 수 있습니다. 그래서 "정답 SQL이 같은 문항끼리는
    항상 같은 그룹으로 묶어서" 나누도록 처리했습니다.
"""

import json
import random
from collections import defaultdict
# defaultdict: 일반 딕셔너리와 비슷하지만, 처음 등장하는 키에 접근할 때
# 에러 없이 자동으로 기본값(여기서는 빈 리스트)을 만들어주는 편리한 도구


def load_jsonl(path):
    """JSONL 파일을 읽어서 파이썬 딕셔너리들의 리스트로 반환."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────
# 1. 정제된 전체 데이터 불러오기
# ─────────────────────────────────────────────
rows = load_jsonl("data/processed/all_clean.jsonl")
print(f"전체 정제 데이터: {len(rows)}개")


# ─────────────────────────────────────────────
# 2. 정답 SQL이 같은 문항끼리 그룹으로 묶기
# ─────────────────────────────────────────────
# groups는 { "정답 SQL 문장": [그 SQL을 정답으로 가진 데이터들] } 형태가 됩니다.
# 이렇게 그룹으로 묶어두면, 나중에 이 그룹을 통째로 train/valid/test 중
# 한 곳에만 배정할 수 있어서 "쌍둥이 문항이 여기저기 흩어지는" 문제를 막습니다.
groups = defaultdict(list)
for row in rows:
    sql = row["messages"][2]["content"].strip()  # messages[2] = assistant(정답 SQL)
    groups[sql].append(row)

# 그룹들의 "키(정답 SQL 문장) 목록"을 뽑아서 무작위로 섞음
group_keys = list(groups.keys())
random.seed(42)          # 시드를 고정해서, 몇 번을 실행해도 항상 같은 결과가 나오게 함
random.shuffle(group_keys)


# ─────────────────────────────────────────────
# 3. 목표 개수를 정하고, 그룹 단위로 순서대로 채워 넣기
# ─────────────────────────────────────────────
train_rows, valid_rows, test_rows = [], [], []

total = len(rows)
train_target = int(total * 0.8)   # 전체의 80%를 train 목표치로 설정
valid_target = int(total * 0.1)   # 전체의 10%를 valid 목표치로 설정
# 나머지는 자연스럽게 test로 감

for key in group_keys:
    group = groups[key]  # 이 그룹에 속한 데이터들(같은 정답 SQL을 가진 문항들)

    # 아직 train이 목표치를 못 채웠으면 train에 통째로 추가
    if len(train_rows) < train_target:
        train_rows.extend(group)
    # train은 다 찼는데 valid가 아직이면 valid에 추가
    elif len(valid_rows) < valid_target:
        valid_rows.extend(group)
    # 둘 다 다 찼으면 나머지는 test로
    else:
        test_rows.extend(group)


def save(path, data):
    """리스트에 담긴 데이터를 JSONL 파일로 저장하는 함수."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 4. 결과를 파일로 저장
# ─────────────────────────────────────────────
# 파일 이름을 train.jsonl / valid.jsonl / test.jsonl로 정확히 맞추는 이유:
# MLX-LM의 mlx_lm.lora 명령이 --data로 지정한 폴더 안에서
# 정확히 이 세 이름의 파일을 자동으로 찾기 때문입니다.
save("data/processed/train.jsonl", train_rows)
save("data/processed/valid.jsonl", valid_rows)
save("data/processed/test.jsonl", test_rows)

print(f"Train: {len(train_rows)}개")
print(f"Valid: {len(valid_rows)}개")
print(f"Test:  {len(test_rows)}개")
