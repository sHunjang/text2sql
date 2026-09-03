"""
run_training.py

이 스크립트가 하는 일:
1. mlx_lm.lora 학습 명령을 파이썬 코드 안에서 실행합니다.
   (터미널에 직접 긴 명령어를 치는 대신, 이 파일을 실행하면 됩니다)
2. 학습 중 터미널에 찍히는 로그(Train loss, Val loss 등)를
   화면에 실시간으로 보여주면서, 동시에 파일로도 저장합니다.
   -> 나중에 Loss Curve를 그리거나 다시 분석할 때 이 로그 파일을 씁니다.
"""

import subprocess   # 파이썬 코드 안에서 터미널 명령어를 실행할 수 있게 해주는 표준 라이브러리
import sys  # 실시간으로 화면에 출력하기 위해 사용
from datetime import datetime
from pathlib import Path


# -- 1. 학습에 사용할 설정값들을 한눈에 보기 좋게 변수로 정리 --
# MT-5의 하이퍼파라미터들을 여기에서 관리
MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"  # 베이스 모델
DATA_DIR = "./data/processed"   # train.jsonl / valid.jsonl / test.jsonl이 들어있는 폴더
ADAPTER_DIR = "./adapters"      # 학습 결과(LoRA 어댑터 가중치)가 저장될 폴더

BATCH_SIZE = 4          # 한 번에 몇 개의 예시를 보고 파라미터를 업데이트할지
NUM_LAYERS = 16         # 모델의 몇 개 레이어에 LoRA 어댑터를 붙일지
ITERS = 500             # 총 업데이트 횟수 (약 8.7 에폭에 해당)
LEARNING_RATE = 1e-5    # 파라미터를 한 번에 얼마나 크게 움직일지
STEPS_PER_REPORT = 10   # 몇 iteration마다 train loss를 출력할지
STEPS_PER_EVAL = 50     # 몇 iteration마다 validation loss를 확인할지
VAL_BATCHES = 7         # validation loss를 잴 때 몇 개의 배치를 쓸지 (28개 ÷ 4 = 7)
SAVE_EVERY = 100        # 몇 iteration마다 중간 체크포인트를 저장할지


# -- 2. 로그를 저장할 폴더/파일 이름 준비 --
# notes 폴더 밑에 실행 시각이 찍힌 로그 파일을 생성하여,
# "이 학습을 언제 돌렸고 결과가 어땠는지" 나중에도 추적할 수 있게 함
Path("notes").mkdir(exist_ok=True)  # notes 폴더가 없으면 새로 생성 -- 이미 있으면 그냥 넘어감
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = Path("notes") / f"train_log_{timestamp}.txt"


# -- 3. 실제로 실행할 터미널 명령어를 리스트 형태로 구성 --
# subprocess == 명령어를 "문자열 하나"가 아니라 "단어별로 쪼갠 리스트"로 받는걸 권장
# (공백 or 특수문자 때문에 명령어가 깨지는 것을 방지하기 위함)
command = [
    "mlx_lm.lora",
    "--model", MODEL_NAME,
    "--train",
    "--data", DATA_DIR,
    "--adapter-path", ADAPTER_DIR,
    "--batch-size", str(BATCH_SIZE),
    "--num-layers", str(NUM_LAYERS),    # subprocess == 모든 값을 문자열로 받아야 해서 str()로 변환
    "--iters", str(ITERS),
    "--learning-rate", str(LEARNING_RATE),
    "--steps-per-report", str(STEPS_PER_REPORT),
    "--steps-per-eval", str(STEPS_PER_EVAL),
    "--val-batches", str(VAL_BATCHES),
    "--save-every", str(SAVE_EVERY),
]

print("** 학습을 시작 **")
print("실행 명령어: ", " ".join(command))
print(f"로그는 다음 파일에도 함께 저장됨: {log_path}")
print()


# -- 4. 명령어 실행 + 로그를 화면과 파일에 동시에 기록 --
# Popen을 쓰는 이유: 학습이 끝날 때까지 기다렸다가 결과를 한 번에 받는 대신,
# 한 줄씩 출력되는 즉시 화면에 보여주고 싶기 때문
# (학습 == 몇 분~몇 십 분 걸리는데, 그동안 아무 반응이 없으면 잘 되고 있는지 알 수 없음)
with open(log_path, "w", encoding="utf-8") as log_file:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,     # 명령어의 출력을 파이썬이 받아올 수 있게 설정
        stderr=subprocess.STDOUT,   # 에러 메세지도 같은 곳에서 합쳐서 받음 -- 놓치는 로그 없게
        text=True,                  # 출력을 바이트가 아니라 문자열로 받음
        bufsize=1,                  # 한 줄씩 바로바로 받아오도록 설정
    )

    # assert: "이 조건이 참이 아니면 프로그램을 여기서 멈추고 에러를 내라"라는 안전장치
    # stdout=PIPE로 설정했기 때문에 process.stdout == 항상 값이 있어야 정상인데,
    # 혹시라도 None이면 여기서 바로 알 수 있음
    assert process.stdout is not None, "표준 출력을 받아오지 못함"

    # 학습이 끝날 때까지, 한 줄씩 나오는 로그를 화면과 파일에 동시에 사용
    for line in process.stdout:
        sys.stdout.write(line)      # 화면(터미널)에 실시간 출력
        log_file.write(line)        # 동시에 파일에도 기록

    process.wait()


# -- 5. 결과 확인 --
if process.returncode == 0:
    print(f"\n학습이 정상적으로 끝남. 로그 파일: {log_path}")
    print(f"어댑터 가중치 == {ADAPTER_DIR} 폴더에 저장")
else:
    print(f"\n학습 중 에러가 발생 (종료 코드: {process.returncode})")
    print(f"자세한 내용은 로그 파일 확인: {log_path}")