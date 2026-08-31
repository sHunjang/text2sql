import json
import time
import anthropic

client = anthropic.Anthropic()

SCHEMA = """CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);"""

SYSTEM_PROMPT = f"""당신은 SQL 전문가입니다. 아래 스키마를 기준으로 자연어 질문을 SQL 쿼리로 변환하세요. SQL 쿼리만 출력하고 다른 설명은 하지 마세요.

{SCHEMA}"""

seed_rows = []
with open("data/raw/seed_examples.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        q = row["messages"][1]["content"]
        sql = row["messages"][2]["content"]
        seed_rows.append((q, sql))

seed_text = "\n".join(f"- 질문: {q}\n  SQL: {sql}" for q, sql in seed_rows)

def build_prompt(n: int) -> str:
    # .format()을 쓰지 않고, 호출 시점에 n을 바로 f-string으로 박아넣습니다.
    return f"""아래는 에이전트 실행 로그를 조회하는 Text-to-SQL 데이터셋의 예시입니다.

스키마:
{SCHEMA}

기존 예시:
{seed_text}

위 예시와 같은 스타일, 같은 스키마를 기준으로 새로운 (자연어 질문, SQL 쿼리) 쌍을 {n}개 만들어주세요.

규칙:
- 난이도를 골고루 섞으세요: 단순 필터, GROUP BY 집계, JOIN, HAVING, 서브쿼리, LEFT JOIN(없는 것 찾기) 등
- 같은 질문을 다른 말투로 반복하지 말고, 다양한 컬럼/조건 조합을 다루세요
- 존댓말/반말, "~보여줘"/"~알려줘"/"~조회해줘" 등 표현도 다양하게 섞으세요
- SQL은 반드시 문법적으로 올바르고 스키마에 실제 존재하는 컬럼만 사용하세요
- 아래 JSON 배열 형식으로만 응답하세요. 다른 설명은 절대 붙이지 마세요.

[
  {{"question": "...", "sql": "..."}},
  ...
]"""

def extract_text(resp) -> str:
    # content 블록 중 type이 "text"인 것만 골라서 이어붙임 (ThinkingBlock 등은 건너뜀)
    parts = [block.text for block in resp.content if block.type == "text"]
    return "".join(parts).strip()

def generate_batch(n=20):
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": build_prompt(n)}],
    )
    text = extract_text(resp)
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

all_rows = []
TARGET = 300
BATCH = 20

while len(all_rows) < TARGET:
    try:
        batch = generate_batch(BATCH)
    except Exception as e:
        print("생성 실패, 재시도:", repr(e))
        time.sleep(3)
        continue
    all_rows.extend(batch)
    print(f"누적 {len(all_rows)}개 생성됨")
    time.sleep(1)

with open("data/raw/synthetic_raw.jsonl", "w", encoding="utf-8") as f:
    for item in all_rows:
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["sql"]},
            ]
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"총 {len(all_rows)}개 합성 데이터 저장 완료 -> data/raw/synthetic_raw.jsonl")
