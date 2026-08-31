import json
import sqlite3
import random

SCHEMA_SQL = """
CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);
"""

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_qa(row):
    msgs = row["messages"]
    question = msgs[1]["content"]
    sql = msgs[2]["content"]
    return question, sql

def sql_is_valid(conn, sql: str) -> bool:
    try:
        # EXPLAIN만 돌려서 실제 데이터 없이도 "문법 + 스키마 정합성"만 빠르게 검증
        conn.execute(f"EXPLAIN {sql}")
        return True
    except sqlite3.Error:
        return False

# 1. 원본 로드 (시드 + 합성)
seed = load_jsonl("data/raw/seed_examples.jsonl")
synthetic = load_jsonl("data/raw/synthetic_raw.jsonl")
combined = seed + synthetic
print(f"원본 총합: {len(combined)}개 (시드 {len(seed)} + 합성 {len(synthetic)})")

# 2. 검증용 인메모리 SQLite DB 준비
conn = sqlite3.connect(":memory:")
conn.executescript(SCHEMA_SQL)

# 3. 문법/스키마 검증 + 중복 질문 제거
seen_questions = set()
valid_rows = []
invalid_examples = []

for row in combined:
    question, sql = get_qa(row)
    q_key = question.strip()

    if q_key in seen_questions:
        continue  # 중복 질문 스킵
    if not sql_is_valid(conn, sql):
        invalid_examples.append((question, sql))
        continue

    seen_questions.add(q_key)
    valid_rows.append(row)

print(f"검증 통과: {len(valid_rows)}개")
print(f"불량(문법/스키마 오류): {len(invalid_examples)}개")
print(f"중복 제거됨: {len(combined) - len(valid_rows) - len(invalid_examples)}개")

if invalid_examples:
    print("\n--- 불량 예시 (최대 5개) ---")
    for q, sql in invalid_examples[:5]:
        print(f"Q: {q}\nSQL: {sql}\n")

# 4. 정제된 전체 데이터 저장
with open("data/processed/all_clean.jsonl", "w", encoding="utf-8") as f:
    for row in valid_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# 5. 사람이 눈으로 검수할 랜덤 샘플 20개 출력
print("\n=== 랜덤 샘플 20개 (내용 검수용) ===")
random.seed(42)
sample = random.sample(valid_rows, min(20, len(valid_rows)))
for row in sample:
    q, sql = get_qa(row)
    print(f"Q: {q}")
    print(f"A: {sql}")
    print("-" * 60)
