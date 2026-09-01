"""
make_seed_data.py

이 스크립트가 하는 일:
    "자연어 질문 -> SQL 쿼리" 예시를 사람이 직접 15개 작성해서
    train.jsonl 형식(MLX-LM이 요구하는 chat 포맷)으로 저장합니다.

왜 필요한가:
    Claude API로 데이터를 대량 생성하기 전에, 먼저 "품질 좋은 예시 몇 개"를
    사람이 직접 만들어두면, 그걸 스타일 가이드로 삼아 AI가 비슷한 수준의
    데이터를 더 안정적으로 만들어낼 수 있습니다. (MT-4: 데이터 품질 원칙)
"""

import json  # 파이썬 데이터(리스트, 딕셔너리)를 JSON 텍스트로 바꿔주는 표준 라이브러리


# ─────────────────────────────────────────────
# 1. 시스템 프롬프트(SCHEMA) 정의
# ─────────────────────────────────────────────
# "시스템 프롬프트"란, 모델에게 "너는 이런 역할이고, 이런 규칙을 지켜야 해"라고
# 미리 알려주는 지시문입니다. 대화의 맨 앞에 항상 붙어서, 모델이 매번
# "나는 SQL 전문가이고, 이 스키마를 기준으로 답해야 한다"는 걸 잊지 않게 해줍니다.
SCHEMA = """당신은 SQL 전문가입니다. 아래 스키마를 기준으로 자연어 질문을 SQL 쿼리로 변환하세요. SQL 쿼리만 출력하고 다른 설명은 하지 마세요.

CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);"""


# ─────────────────────────────────────────────
# 2. (질문, 정답 SQL) 쌍을 파이썬 리스트로 작성
# ─────────────────────────────────────────────
# 튜플(tuple) 형태로 "질문"과 "정답 SQL"을 짝지어서 리스트에 담습니다.
# 난이도를 일부러 3단계로 나눠서 골고루 만들었습니다:
#   - 쉬움: 테이블 하나만 보고 단순히 필터링하거나 정렬하는 수준
#   - 중간: GROUP BY로 집계(개수 세기, 평균 내기 등)하는 수준
#   - 어려움: 여러 테이블을 JOIN하거나, 조건이 여러 겹으로 겹치는 수준
# 난이도를 섞어두는 이유: 쉬운 문제만 학습시키면, 모델이 JOIN처럼
# 복잡한 질문을 만났을 때 제대로 대응하지 못하게 됩니다.
pairs = [
    # --- 쉬움: 단일 테이블, 단순 필터/정렬 ---
    ("전체 에이전트 목록을 보여줘", "SELECT * FROM agents;"),
    ("planner 타입인 에이전트만 보여줘", "SELECT * FROM agents WHERE agent_type = 'planner';"),
    ("실패한 실행들을 모두 보여줘", "SELECT * FROM executions WHERE status = 'failed';"),
    ("가장 최근에 생성된 에이전트 5개를 보여줘", "SELECT * FROM agents ORDER BY created_at DESC LIMIT 5;"),
    ("레이턴시가 5000ms를 넘는 실행을 보여줘", "SELECT * FROM executions WHERE latency_ms > 5000;"),

    # --- 중간: 집계(GROUP BY/집계함수), 단일 테이블 ---
    ("상태별 실행 개수를 보여줘", "SELECT status, COUNT(*) AS cnt FROM executions GROUP BY status;"),
    ("평균 레이턴시가 가장 긴 상위 3개 상태를 보여줘",
     "SELECT status, AVG(latency_ms) AS avg_latency FROM executions GROUP BY status ORDER BY avg_latency DESC LIMIT 3;"),
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


# ─────────────────────────────────────────────
# 3. JSONL 파일로 저장
# ─────────────────────────────────────────────
# "JSONL"은 "JSON Lines"의 줄임말로, 한 줄에 JSON 객체 하나씩 쓰는 형식입니다.
# (일반 JSON처럼 파일 전체가 하나의 큰 배열이 아니라, 한 줄 = 데이터 하나)
# MLX-LM이 이 형식을 학습 데이터로 읽어들이기 때문에 이 형식을 맞춰줍니다.
#
# 각 줄은 아래와 같은 구조를 가집니다 (이걸 "chat 포맷"이라고 부릅니다):
#   {"messages": [
#       {"role": "system", "content": "너는 이런 역할이야"},   <- 규칙 설명
#       {"role": "user", "content": "사용자 질문"},             <- 자연어 질문
#       {"role": "assistant", "content": "모델이 낼 정답"}      <- 정답 SQL
#   ]}
# 이렇게 "대화 형태"로 데이터를 주는 이유는, 우리가 쓸 베이스 모델
# (Qwen2.5-Coder-Instruct)이 원래 이런 대화 형식으로 학습된 모델이라,
# 같은 형식으로 데이터를 줘야 자연스럽게 학습이 되기 때문입니다.
with open("data/raw/seed_examples.jsonl", "w", encoding="utf-8") as f:
    for question, sql in pairs:
        row = {
            "messages": [
                {"role": "system", "content": SCHEMA},
                {"role": "user", "content": question},
                {"role": "assistant", "content": sql},
            ]
        }
        # json.dumps: 파이썬 딕셔너리를 JSON 문자열로 변환
        # ensure_ascii=False: 한글이 깨진 유니코드 코드(\uXXXX)로 안 바뀌고
        #                      사람이 읽을 수 있는 한글 그대로 저장되게 함
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"{len(pairs)}개 시드 예시 저장 완료 -> data/raw/seed_examples.jsonl")
