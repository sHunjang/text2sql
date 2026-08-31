import json

SCHEMA = """당신은 SQL 전문가입니다. 아래 스키마를 기준으로 자연어 질문을 SQL 쿼리로 변환하세요. SQL 쿼리만 출력하고 다른 설명은 하지 마세요.

CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);"""

pairs = [
    # --- 쉬움: 단일 테이블, 단순 필터/정렬 ---
    ("전체 에이전트 목록을 보여줘", "SELECT * FROM agents;"),
    ("planner 타입인 에이전트만 보여줘", "SELECT * FROM agents WHERE agent_type = 'planner';"),
    ("실패한 실행들을 모두 보여줘", "SELECT * FROM executions WHERE status = 'failed';"),
    ("가장 최근에 생성된 에이전트 5개를 보여줘", "SELECT * FROM agents ORDER BY created_at DESC LIMIT 5;"),
    ("레이턴시가 5000ms를 넘는 실행을 보여줘", "SELECT * FROM executions WHERE latency_ms > 5000;"),

    # --- 중간: 집계(GROUP BY/집계함수), 단일 테이블 ---
    ("상태별 실행 개수를 보여줘", "SELECT status, COUNT(*) AS cnt FROM executions GROUP BY status;"),
    ("평균 레이턴시가 가장 긴 상위 3개 상태를 보여줘", "SELECT status, AVG(latency_ms) AS avg_latency FROM executions GROUP BY status ORDER BY avg_latency DESC LIMIT 3;"),
    ("툴 이름별 총 비용을 보여줘", "SELECT tool_name, SUM(cost_usd) AS total_cost FROM tool_calls GROUP BY tool_name;"),
    ("에이전트 타입별 개수를 보여줘", "SELECT agent_type, COUNT(*) AS cnt FROM agents GROUP BY agent_type;"),
    ("가장 비용이 많이 든 툴 콜 상위 5개를 보여줘", "SELECT * FROM tool_calls ORDER BY cost_usd DESC LIMIT 5;"),

    # --- 어려움: JOIN, HAVING, 서브쿼리 필요 ---
    ("실패한 실행이 많은 에이전트 순으로 보여줘",
     "SELECT a.agent_name, COUNT(*) AS failed_count FROM executions e JOIN agents a ON e.agent_id = a.agent_id WHERE e.status = 'failed' GROUP BY a.agent_name ORDER BY failed_count DESC;"),
    ("각 에이전트별 평균 실행 레이턴시를 보여줘",
     "SELECT a.agent_name, AVG(e.latency_ms) AS avg_latency FROM executions e JOIN agents a ON e.agent_id = a.agent_id GROUP BY a.agent_name;"),
    ("실행당 툴 콜 비용 합계가 1달러를 넘는 실행을 보여줘",
     "SELECT e.execution_id, SUM(t.cost_usd) AS total_cost FROM executions e JOIN tool_calls t ON e.execution_id = t.execution_id GROUP BY e.execution_id HAVING SUM(t.cost_usd) > 1.0;"),
    ("한 번도 실행되지 않은 에이전트를 보여줘",
     "SELECT a.* FROM agents a LEFT JOIN executions e ON a.agent_id = e.agent_id WHERE e.execution_id IS NULL;"),
    ("critic 타입 에이전트 중 타임아웃이 가장 많은 에이전트를 보여줘",
     "SELECT a.agent_name, COUNT(*) AS timeout_count FROM executions e JOIN agents a ON e.agent_id = a.agent_id WHERE a.agent_type = 'critic' AND e.status = 'timeout' GROUP BY a.agent_name ORDER BY timeout_count DESC LIMIT 1;"),
]

with open("data/raw/seed_examples.jsonl", "w", encoding="utf-8") as f:
    for question, sql in pairs:
        row = {
            "messages": [
                {"role": "system", "content": SCHEMA},
                {"role": "user", "content": question},
                {"role": "assistant", "content": sql},
            ]
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"{len(pairs)}개 시드 예시 저장 완료 -> data/raw/seed_examples.jsonl")
