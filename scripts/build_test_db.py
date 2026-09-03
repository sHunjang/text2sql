"""
build_test_db.py

이 스크립트가 하는 일:
    evaluate_models.py에서 썼던 것과 똑같은 방식(같은 random.seed)으로
    샘플 데이터를 만들어서, 이번엔 메모리가 아니라 실제 파일(test_data.db)로
    저장합니다. 이렇게 저장해두면 이후 터미널이나 GUI 도구로 자유롭게
    쿼리를 실행해보며 결과를 눈으로 확인할 수 있습니다.
"""

import sqlite3
import random

SCHEMA_SQL = """
CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);
"""

random.seed(42)  # evaluate_models.py와 완전히 동일한 시드 -> 완전히 동일한 데이터가 생성됨

# ":memory:" 대신 실제 파일 경로를 지정하면, DB가 파일로 저장됩니다.
conn = sqlite3.connect("test_data.db")
conn.executescript(SCHEMA_SQL)

agent_types = ["planner", "executor", "critic", "llm"]
statuses = ["success", "failed", "timeout", "running"]
tool_names = ["web_search", "sql_query", "file_read", "api_call", "code_exec"]

for agent_id in range(1, 21):
    conn.execute(
        "INSERT INTO agents VALUES (?, ?, ?, ?)",
        (agent_id, f"agent_{agent_id}", random.choice(agent_types),
         f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 09:00:00"),
    )

for execution_id in range(1, 101):
    agent_id = random.randint(1, 20)
    latency = random.randint(200, 9000)
    conn.execute(
        "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (execution_id, agent_id, random.choice(statuses),
         f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 10:00:00",
         f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 10:05:00",
         latency, None),
    )

for tool_call_id in range(1, 301):
    execution_id = random.randint(1, 100)
    conn.execute(
        "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        (tool_call_id, execution_id, random.choice(tool_names),
         random.randint(50, 2000), random.randint(50, 2000),
         round(random.uniform(0, 0.5), 4),
         f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 10:02:00"),
    )

conn.commit()
conn.close()
print("test_data.db 파일 생성 완료")