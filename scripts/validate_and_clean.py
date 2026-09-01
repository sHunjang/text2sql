"""
validate_and_clean.py

이 스크립트가 하는 일:
    1. 시드 데이터(15개) + 합성 데이터(300개)를 합칩니다.
    2. 각 SQL 쿼리가 실제로 "문법적으로 말이 되는지", "스키마에 진짜 있는
       테이블/컬럼만 쓰고 있는지"를 SQLite 데이터베이스에 직접 돌려서 검증합니다.
    3. 중복된 질문을 제거합니다.
    4. 통과한 데이터를 저장하고, 사람이 눈으로 검수할 수 있게 무작위로
       20개를 뽑아 화면에 출력합니다.

왜 필요한가 (MT-4 데이터 품질 원칙):
    Claude가 생성한 SQL이라고 해서 전부 정답은 아닙니다. 존재하지 않는
    컬럼을 지어내거나("환각"), 문법이 살짝 틀린 경우가 섞여 있을 수
    있습니다. 이런 불량 데이터를 걸러내지 않고 그대로 학습에 쓰면,
    모델이 오히려 틀린 패턴을 배우게 됩니다.
"""

import json
import sqlite3   # 파이썬에 기본 내장된 가벼운 데이터베이스. 별도 설치 없이 바로 쓸 수 있음
import random    # 무작위로 샘플을 뽑기 위해 사용


# ─────────────────────────────────────────────
# 1. 검증에 사용할 스키마 정의
# ─────────────────────────────────────────────
# 이 스키마로 "진짜 빈 데이터베이스"를 하나 만들어서, 그 위에 생성된
# SQL들을 실제로 실행해봅니다. 데이터가 하나도 없어도 "문법이 맞는지",
# "테이블/컬럼 이름이 실제로 존재하는지"는 확인할 수 있습니다.
SCHEMA_SQL = """
CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);
"""


def load_jsonl(path):
    """
    JSONL 파일(한 줄에 JSON 하나씩 있는 파일)을 읽어서,
    파이썬 딕셔너리들의 리스트로 반환하는 함수.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 줄 앞뒤 공백/줄바꿈 제거
            if line:              # 빈 줄은 건너뜀
                rows.append(json.loads(line))
    return rows


def get_qa(row):
    """
    chat 포맷 한 줄(row)에서 "질문"과 "정답 SQL"만 뽑아내는 함수.
    messages 리스트는 [system, user, assistant] 순서로 되어 있으므로
    인덱스 1이 사용자 질문, 인덱스 2가 정답입니다.
    """
    msgs = row["messages"]
    question = msgs[1]["content"]
    sql = msgs[2]["content"]
    return question, sql


def sql_is_valid(conn, sql: str) -> bool:
    """
    주어진 SQL 문장이 실제로 실행 가능한지 확인하는 함수.

    conn.execute(f"EXPLAIN {sql}")를 쓰는 이유:
    EXPLAIN은 SQL을 "진짜로 실행"하지는 않고, "이 SQL을 어떻게 실행할
    계획인지"만 계산합니다. 그래서 테이블에 실제 데이터가 하나도 없어도
    빠르게 "문법이 맞는지 + 테이블/컬럼 이름이 실제로 존재하는지"를
    검증할 수 있습니다.

    문제가 있으면 sqlite3.Error 예외가 발생하는데, 이걸 잡아서(except)
    "유효하지 않다(False)"로 처리합니다.
    """
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True
    except sqlite3.Error:
        return False


# ─────────────────────────────────────────────
# 2. 시드 + 합성 데이터를 하나로 합치기
# ─────────────────────────────────────────────
seed = load_jsonl("data/raw/seed_examples.jsonl")
synthetic = load_jsonl("data/raw/synthetic_raw.jsonl")
combined = seed + synthetic  # 두 리스트를 이어붙임
print(f"원본 총합: {len(combined)}개 (시드 {len(seed)} + 합성 {len(synthetic)})")


# ─────────────────────────────────────────────
# 3. 검증용 SQLite 데이터베이스 준비
# ─────────────────────────────────────────────
# ":memory:"는 파일이 아니라 컴퓨터 메모리(RAM) 안에만 임시로 만들어지는
# 데이터베이스를 뜻합니다. 검증이 끝나면 자동으로 사라지므로,
# 디스크에 불필요한 파일을 남기지 않아 편리합니다.
conn = sqlite3.connect(":memory:")
conn.executescript(SCHEMA_SQL)  # 위에서 정의한 스키마로 빈 테이블들을 생성


# ─────────────────────────────────────────────
# 4. 문법 검증 + 중복 질문 제거
# ─────────────────────────────────────────────
seen_questions = set()   # 지금까지 등장한 질문들을 저장 (중복 확인용)
valid_rows = []           # 검증을 통과한 데이터
invalid_examples = []     # 검증에 실패한 데이터 (나중에 확인용으로 따로 모아둠)

for row in combined:
    question, sql = get_qa(row)
    q_key = question.strip()

    if q_key in seen_questions:
        continue  # 이미 나온 질문이면 건너뜀 (중복 제거)

    if not sql_is_valid(conn, sql):
        invalid_examples.append((question, sql))
        continue  # 문법/스키마 오류가 있으면 건너뜀

    seen_questions.add(q_key)
    valid_rows.append(row)

print(f"검증 통과: {len(valid_rows)}개")
print(f"불량(문법/스키마 오류): {len(invalid_examples)}개")
print(f"중복 제거됨: {len(combined) - len(valid_rows) - len(invalid_examples)}개")

# 불량 데이터가 있다면, 어떤 게 왜 걸러졌는지 최대 5개만 미리보기로 출력
if invalid_examples:
    print("\n--- 불량 예시 (최대 5개) ---")
    for q, sql in invalid_examples[:5]:
        print(f"Q: {q}\nSQL: {sql}\n")


# ─────────────────────────────────────────────
# 5. 검증을 통과한 데이터를 파일로 저장
# ─────────────────────────────────────────────
with open("data/processed/all_clean.jsonl", "w", encoding="utf-8") as f:
    for row in valid_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 6. 사람이 눈으로 검수할 수 있도록 무작위 샘플 20개 출력
# ─────────────────────────────────────────────
# 기계(SQLite)는 "문법이 맞는지"만 확인할 수 있고, "질문의 의도와
# SQL의 내용이 실제로 일치하는지"는 판단하지 못합니다.
# (예: "실패한 실행을 보여줘"라고 물었는데 status = 'success'로 잘못
#  나온 경우 -- 문법은 완벽하게 맞지만 내용은 틀린 경우)
# 이런 건 사람이 직접 눈으로 봐야 하기 때문에, 무작위 샘플을 뽑아
# 화면에 출력합니다.
print("\n=== 랜덤 샘플 20개 (내용 검수용) ===")
random.seed(42)  # 항상 같은 샘플이 뽑히도록 무작위 시드를 고정 (재현 가능하게)
sample = random.sample(valid_rows, min(20, len(valid_rows)))
for row in sample:
    q, sql = get_qa(row)
    print(f"Q: {q}")
    print(f"A: {sql}")
    print("-" * 60)
