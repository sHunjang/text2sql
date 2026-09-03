"""
evaluate_models.py

이 스크립트가 하는 일:
    1. 베이스 모델 / iter 200 체크포인트 / iter 500(최종) 체크포인트,
       이렇게 세 가지 모델을 순서대로 로드합니다.
    2. 각 모델에게 test.jsonl(30개)의 질문을 주고 SQL을 생성시킵니다.
    3. 생성된 SQL과 정답 SQL을 "샘플 데이터가 채워진 가짜 DB"에 실제로
       돌려보고, 반환되는 결과(행)가 똑같은지 비교해서 정확도를 계산합니다.
       (이걸 "실행 정확도, Execution Accuracy"라고 부릅니다)
    4. 세 모델의 정확도를 표로 비교해서 출력합니다.
"""

import json
import sqlite3
import random
import shutil
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# ─────────────────────────────────────────────
# 1. 기본 설정값
# ─────────────────────────────────────────────
BASE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
ADAPTER_DIR = Path("adapters")          # 학습 때 체크포인트가 저장된 폴더
TEST_PATH = Path("data/processed/test.jsonl")

SCHEMA_SQL = """
CREATE TABLE agents (agent_id INTEGER PRIMARY KEY, agent_name TEXT, agent_type TEXT, created_at DATETIME);
CREATE TABLE executions (execution_id INTEGER PRIMARY KEY, agent_id INTEGER, status TEXT, started_at DATETIME, ended_at DATETIME, latency_ms INTEGER, error_message TEXT);
CREATE TABLE tool_calls (tool_call_id INTEGER PRIMARY KEY, execution_id INTEGER, tool_name TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, called_at DATETIME);
"""


# ─────────────────────────────────────────────
# 2. 체크포인트를 mlx_lm.load()가 읽을 수 있는 폴더 형태로 준비
# ─────────────────────────────────────────────
def prepare_checkpoint_folder(iter_num: int) -> str:
    """
    예: iter_num=200 이면
        adapters/0000200_adapters.safetensors 파일을 복사해서
        adapters/_eval_iter200/adapters.safetensors 로 이름을 맞춰 저장하고,
        adapter_config.json도 같은 폴더에 복사해둡니다.

    이렇게 폴더를 따로 만들어두는 이유는, mlx_lm.load()가
    "adapter_path로 지정된 폴더 안에 adapters.safetensors라는
    정해진 이름의 파일이 있어야 한다"는 규칙을 갖고 있기 때문입니다.
    체크포인트 파일들은 이름이 0000200_adapters.safetensors처럼
    앞에 iteration 번호가 붙어 있어서, 그대로는 인식이 안 됩니다.
    """
    checkpoint_file = ADAPTER_DIR / f"{iter_num:07d}_adapters.safetensors"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_file}")

    eval_folder = ADAPTER_DIR / f"_eval_iter{iter_num}"
    eval_folder.mkdir(exist_ok=True)

    # adapter_config.json 복사 (LoRA 설정 정보 - 어떤 레이어에 어댑터를 붙였는지 등)
    shutil.copy(ADAPTER_DIR / "adapter_config.json", eval_folder / "adapter_config.json")
    # 체크포인트 파일을 "adapters.safetensors"라는 이름으로 복사
    shutil.copy(checkpoint_file, eval_folder / "adapters.safetensors")

    return str(eval_folder)


# ─────────────────────────────────────────────
# 3. 비교할 세 모델 목록 정의
# ─────────────────────────────────────────────
# adapter_path가 None이면 어댑터 없이 순수 베이스 모델만 로드합니다.
CANDIDATES = [
    {"name": "베이스 모델", "adapter_path": None},
    {"name": "iter 200", "adapter_path": prepare_checkpoint_folder(200)},
    {"name": "iter 500 (최종)", "adapter_path": prepare_checkpoint_folder(500)},
]


# ─────────────────────────────────────────────
# 4. test.jsonl 불러오기
# ─────────────────────────────────────────────
def load_test_data():
    """
    test.jsonl의 각 줄에서 시스템 프롬프트, 질문, 정답 SQL을 꺼내
    리스트로 반환합니다.
    """
    items = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            items.append({
                "system": msgs[0]["content"],
                "question": msgs[1]["content"],
                "gold_sql": msgs[2]["content"],  # "gold"는 "정답"이라는 뜻으로 평가 코드에서 흔히 쓰는 이름
            })
    return items


# ─────────────────────────────────────────────
# 5. 검증용 DB에 재현 가능한 샘플 데이터 채워 넣기
# ─────────────────────────────────────────────
def build_seeded_database() -> sqlite3.Connection:
    """
    비어있는 스키마만으로는 "정답 SQL과 생성된 SQL이 실제로 다른 결과를
    내는지" 구별할 수 없습니다 (둘 다 그냥 빈 결과가 나오니까요).
    그래서 random.seed로 고정된 무작위 샘플 데이터를 채워 넣어서,
    두 SQL의 결과가 진짜로 같은지 다른지 의미 있게 비교할 수 있게 합니다.

    random.seed(...)로 시드를 고정하는 이유: 이 스크립트를 몇 번을
    다시 실행해도 항상 "같은 가짜 데이터"가 만들어지게 해서,
    평가 결과가 실행할 때마다 달라지지 않고 재현 가능하게 하기 위함입니다.
    """
    random.seed(42)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)

    agent_types = ["planner", "executor", "critic", "llm"]
    statuses = ["success", "failed", "timeout", "running"]
    tool_names = ["web_search", "sql_query", "file_read", "api_call", "code_exec"]

    # agents 20개 생성
    for agent_id in range(1, 21):
        conn.execute(
            "INSERT INTO agents VALUES (?, ?, ?, ?)",
            (
                agent_id,
                f"agent_{agent_id}",
                random.choice(agent_types),
                f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 09:00:00",
            ),
        )

    # executions 100개 생성 (각 execution은 무작위 agent에 속함)
    for execution_id in range(1, 101):
        agent_id = random.randint(1, 20)
        latency = random.randint(200, 9000)
        conn.execute(
            "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                execution_id,
                agent_id,
                random.choice(statuses),
                f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 10:00:00",
                f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 10:05:00",
                latency,
                None,
            ),
        )

    # tool_calls 300개 생성 (각 tool_call은 무작위 execution에 속함)
    for tool_call_id in range(1, 301):
        execution_id = random.randint(1, 100)
        conn.execute(
            "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                tool_call_id,
                execution_id,
                random.choice(tool_names),
                random.randint(50, 2000),
                random.randint(50, 2000),
                round(random.uniform(0, 0.5), 4),
                f"2026-0{random.randint(1,6)}-{random.randint(10,28):02d} 10:02:00",
            ),
        )

    conn.commit()
    return conn


def run_sql_safely(conn: sqlite3.Connection, sql: str):
    """
    SQL을 실행해서 결과를 반환합니다.
    문법 오류 등으로 실행 자체가 실패하면 None을 반환합니다
    (모델이 아예 실행 불가능한 SQL을 만들어낸 경우를 "오답"으로 처리하기 위함).
    """
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        # 행 순서가 다르더라도 "내용이 같으면 같은 결과"로 보기 위해 정렬해서 비교
        return sorted(rows)
    except sqlite3.Error:
        return None


# ─────────────────────────────────────────────
# 6. 모델 출력에서 SQL 문장만 깔끔하게 뽑아내는 함수
# ─────────────────────────────────────────────
def extract_sql(raw_text: str) -> str:
    """
    모델 출력에서 순수 SQL 문장만 남기고 나머지를 제거합니다.
    """
    text = raw_text.strip()

    text = text.replace("```sql", "").replace("```", "")
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|im_start|>", "")

    text = text.strip()

    # 모델이 정답 SQL을 다 내고도 이상한 문자를 계속 반복하는 경우가 있어서
    # (예: "SELECT ...;" 뒤에 "!"가 수십 번 반복),
    # 첫 번째 세미콜론(;)까지만 잘라내서 진짜 SQL 문장만 남깁니다.
    if ";" in text:
        text = text.split(";")[0] + ";"

    return text.strip()


# ─────────────────────────────────────────────
# 7. 한 모델을 평가하는 함수
# ─────────────────────────────────────────────
def evaluate_model(name: str, adapter_path, test_items, db_conn) -> dict:
    print(f"\n=== [{name}] 모델 로딩 중... ===")

    # adapter_path가 None이면 베이스 모델만, 아니면 어댑터를 얹어서 로드
    if adapter_path is None:
        model, tokenizer = load(BASE_MODEL)     # # type: ignore[misc]
    else:
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)      # # type: ignore[misc]

    # temp=0.0으로 그리리 디코딩(가장 확률 높은 토큰만 선택)을 하는 sampler 생성
    # 예전 버전에서는 generate()에 temp=0.0을 직접 넘기면 됐지만,
    # 최신 mlx-lm에서는 "샘플링 방식 자체를 별도 객체로 만들어서" 넘기는 구조로 바뀜
    sampler = make_sampler(temp=0.0)

    correct = 0
    results = []  # 나중에 어떤 문항이 틀렸는지 살펴보기 위해 상세 기록도 남김

    for i, item in enumerate(test_items, start=1):
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # temperature를 지정하지 않으면 기본적으로 무작위성이 섞여서,
        # 같은 모델이라도 실행할 때마다 답이 조금씩 달라질 수 있습니다.
        # 평가는 "재현 가능해야" 하므로, temp=0.0으로 "항상 가장 확률 높은
        # 답만 고르도록" 고정합니다 (이런 방식을 "그리디 디코딩"이라 부릅니다).
        raw_output = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=256, verbose=False, sampler=sampler,
        )
        predicted_sql = extract_sql(raw_output)

        # print(f"     [원본 출력] {raw_output!r}")

        gold_result = run_sql_safely(db_conn, item["gold_sql"])
        predicted_result = run_sql_safely(db_conn, predicted_sql)

        is_correct = (predicted_result is not None) and (predicted_result == gold_result)
        if is_correct:
            correct += 1

        results.append({
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "predicted_sql": predicted_sql,
            "is_correct": is_correct,
        })

        status_icon = "O" if is_correct else "X"
        print(f"  [{i}/{len(test_items)}] {status_icon}  {item['question'][:40]}")

    accuracy = correct / len(test_items) * 100
    print(f"[{name}] 실행 정확도: {correct}/{len(test_items)} ({accuracy:.1f}%)")

    return {"name": name, "accuracy": accuracy, "correct": correct, "total": len(test_items), "details": results}


# ─────────────────────────────────────────────
# 8. 전체 평가 실행
# ─────────────────────────────────────────────
test_items = load_test_data()
db_conn = build_seeded_database()

all_results = []
for candidate in CANDIDATES:
    result = evaluate_model(candidate["name"], candidate["adapter_path"], test_items, db_conn)
    all_results.append(result)

# ─────────────────────────────────────────────
# 9. 최종 비교 표 출력
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("최종 비교 결과")
print("=" * 50)
for r in all_results:
    print(f"{r['name']:<15} : {r['correct']}/{r['total']}  ({r['accuracy']:.1f}%)")

# 상세 결과(각 문항별 정오답)를 파일로 저장해서, 나중에 뭐가 틀렸는지 분석할 수 있게 함
with open("notes/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("\n상세 결과 저장 완료 -> notes/eval_results.json")