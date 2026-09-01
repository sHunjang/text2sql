"""
generate_synthetic.py

이 스크립트가 하는 일:
    1. 앞서 사람이 직접 만든 시드 데이터(15개)를 읽어옵니다.
    2. 그 시드 데이터를 "스타일 가이드"로 Claude API에 보여주면서,
       "이런 느낌으로 더 많이 만들어줘"라고 요청합니다.
    3. 20개씩 여러 번 요청해서 총 300개를 모은 뒤 JSONL로 저장합니다.

왜 이렇게 하는가 (MT-4 데이터 품질 원칙):
    처음부터 "무작정 300개 만들어줘"라고 하면 품질이 들쭉날쭉해지기 쉽습니다.
    반면 "이미 검증된 좋은 예시 15개"를 먼저 보여주면, AI가 그 패턴과
    수준을 참고해서 더 일관성 있는 데이터를 만들어냅니다.
"""

import json
import time            # 배치 사이에 잠깐 쉬어가기 위해 사용 (API 요청 과부하 방지)
import anthropic        # Claude API를 파이썬에서 쓰게 해주는 공식 라이브러리


# Anthropic 클라이언트 생성.
# API 키를 코드에 직접 쓰지 않고, 미리 터미널에 설정해둔 환경변수(ANTHROPIC_API_KEY)를
# 자동으로 읽어옵니다. (보안상 키를 코드/저장소에 남기지 않기 위한 표준 방법)
client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# 1. 스키마와 시스템 프롬프트 정의
# ─────────────────────────────────────────────
# make_seed_data.py와 동일한 스키마를 그대로 씁니다.
# (스키마가 파일마다 달라지면 데이터 전체의 일관성이 깨지기 때문에
#  같은 문자열을 그대로 복사해서 씁니다)
SCHEMA = """CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);"""

# 최종적으로 학습 데이터의 "system" 자리에 들어갈 지시문
SYSTEM_PROMPT = f"""당신은 SQL 전문가입니다. 아래 스키마를 기준으로 자연어 질문을 SQL 쿼리로 변환하세요. SQL 쿼리만 출력하고 다른 설명은 하지 마세요.

{SCHEMA}"""


# ─────────────────────────────────────────────
# 2. 시드 데이터를 읽어와서, Claude에게 보여줄 "예시 텍스트"로 변환
# ─────────────────────────────────────────────
seed_rows = []
with open("data/raw/seed_examples.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)            # 한 줄(JSON 문자열)을 파이썬 딕셔너리로 변환
        q = row["messages"][1]["content"]  # messages 리스트의 두 번째(인덱스 1)가 user 질문
        sql = row["messages"][2]["content"]  # 세 번째(인덱스 2)가 assistant 정답
        seed_rows.append((q, sql))

# Claude에게 보여줄 형태로 "- 질문: ... / SQL: ..." 목록 텍스트를 만듦
seed_text = "\n".join(f"- 질문: {q}\n  SQL: {sql}" for q, sql in seed_rows)


# ─────────────────────────────────────────────
# 3. Claude에게 보낼 프롬프트를 만드는 함수
# ─────────────────────────────────────────────
def build_prompt(n: int) -> str:
    """
    n: 이번 요청에서 몇 개의 새로운 (질문, SQL) 쌍을 만들지 지정하는 숫자.

    주의: 이 함수는 매번 새로 프롬프트 문자열을 "조립"해서 반환합니다.
    (이전 버전에서는 .format()을 나중에 한 번 더 호출하다가,
     프롬프트 안에 있는 JSON 예시의 중괄호 { } 를 .format()이
     "채워야 할 빈칸"으로 착각해서 에러가 났었습니다.
     그래서 지금은 .format()을 쓰지 않고, f-string 하나로 한 번에 완성합니다.)
    """
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


# ─────────────────────────────────────────────
# 4. Claude 응답에서 안전하게 텍스트만 뽑아내는 함수
# ─────────────────────────────────────────────
def extract_text(resp) -> str:
    """
    Claude API 응답(resp.content)은 여러 종류의 블록으로 구성될 수 있습니다.
    예를 들어 모델이 "생각하는 과정"을 담은 블록(ThinkingBlock)이
    실제 답변 텍스트 블록보다 먼저 올 수도 있습니다.

    그래서 무조건 content[0]이 텍스트라고 가정하면 안 되고,
    "type이 정확히 text인 블록"만 골라서 이어붙여야 안전합니다.
    """
    parts = [block.text for block in resp.content if block.type == "text"]
    return "".join(parts).strip()


# ─────────────────────────────────────────────
# 5. 실제로 Claude API를 호출해서 데이터 한 묶음(batch)을 생성하는 함수
# ─────────────────────────────────────────────
def generate_batch(n=20):
    resp = client.messages.create(
        model="claude-sonnet-4-6",   # 사용할 Claude 모델 (API용 모델 이름)
        max_tokens=4000,              # 응답으로 받을 최대 글자(토큰) 수 제한
        messages=[{"role": "user", "content": build_prompt(n)}],
    )
    text = extract_text(resp)

    # 혹시 모델이 응답을 ```json ... ``` 코드블록으로 감싸서 줄 수도 있어서,
    # 그런 표시가 있으면 제거하고 순수 JSON 텍스트만 남김
    text = text.replace("```json", "").replace("```", "").strip()

    # 문자열을 실제 파이썬 리스트(딕셔너리들의 리스트)로 변환
    return json.loads(text)


# ─────────────────────────────────────────────
# 6. 목표 개수(300개)에 도달할 때까지 반복 생성
# ─────────────────────────────────────────────
all_rows = []
TARGET = 300   # 최종적으로 모으고 싶은 데이터 개수
BATCH = 20     # 한 번의 API 요청으로 몇 개씩 만들지

while len(all_rows) < TARGET:
    try:
        batch = generate_batch(BATCH)
    except Exception as e:
        # API 호출이 실패하거나(네트워크 문제 등),
        # 응답이 JSON으로 잘 안 읽히는 경우 여기로 옵니다.
        # repr(e)를 쓰면 에러의 종류와 원인이 더 자세하게 출력되어 디버깅에 유리합니다.
        print("생성 실패, 재시도:", repr(e))
        time.sleep(3)  # 3초 쉬었다가 다시 시도 (일시적인 문제일 수 있으므로)
        continue

    all_rows.extend(batch)  # 새로 받은 데이터를 전체 리스트에 추가
    print(f"누적 {len(all_rows)}개 생성됨")
    time.sleep(1)  # 요청 사이에 1초씩 쉬어서, API 서버에 너무 빠르게 연속 요청하지 않도록 배려


# ─────────────────────────────────────────────
# 7. 최종 결과를 학습용 JSONL 형식으로 저장
# ─────────────────────────────────────────────
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
