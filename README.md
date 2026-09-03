# text2sql

자연어 질문을 SQL 쿼리로 변환하는 소형 모델 파인튜닝 프로젝트 (MLX-LM, QLoRA)

에이전트 실행 로그를 조회하는 Text-to-SQL 태스크를 대상으로, Qwen2.5-Coder-7B-Instruct(4bit)를
QLoRA로 파인튜닝하고 베이스 모델과 성능을 비교했습니다.

## 결과 요약

| 모델 | 실행 정확도 (Test 30개) |
|---|---|
| 베이스 모델 (파인튜닝 전) | 76.7% (23/30) |
| **파인튜닝 모델 (iter 200, 최종 채택)** | **86.7% (26/30)** |
| 파인튜닝 모델 (iter 500, 최종 iter) | 83.3% (25/30) |

- Validation Loss 기준 iter 150~200 부근이 최적점이었고, 그 이후는 과적합 진행 (iter 500 val loss 0.147 > iter 200 val loss 0.127)
- 실제 test 성능에서도 iter 200 > iter 500으로 확인되어, Loss Curve로 예측한 "마지막 체크포인트가 항상 최선은 아니다"라는 가설이 검증됨

## 환경
- Apple M3 Pro, 18GB, MLX-LM (QLoRA)
- 베이스 모델: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`

## 데이터
- 시드 15개(직접 작성) + Claude API 합성 300개 → 검증/중복제거 후 288개
- Train 230 / Valid 28 / Test 30 (정답 SQL이 동일한 문항은 같은 split으로 묶어 데이터 누수 방지)

## 재현 방법
1. `scripts/make_seed_data.py` — 시드 데이터 생성
2. `scripts/generate_synthetic.py` — Claude API로 합성 데이터 생성 (ANTHROPIC_API_KEY 필요)
3. `scripts/validate_and_clean.py` — SQLite 기반 문법 검증 및 정제
4. `scripts/split_data.py` — Train/Valid/Test 분리
5. `scripts/run_training.py` — LoRA 파인튜닝 실행
6. `scripts/build_test_db.py` — 평가용 샘플 DB 생성
7. `scripts/evaluate_models.py` — 베이스 vs 파인튜닝 모델 실행 정확도 비교

## 주요 배운 점
- Loss Curve만으로 과적합 시점을 예측하고, 실제 test 성능으로 그 예측을 검증하는 흐름
- 실행 정확도(Execution Accuracy) 채점의 장단점: 논리는 맞아도 SELECT 컬럼 구성이 다르면 오답 처리되는 한계
- 모델이 "암묵적으로 LEFT JOIN이 필요한" 질문(예: "~별로"라는 표현)에서 INNER JOIN을 기본값으로 쓰는 경향 발견
