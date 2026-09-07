"""
plot_scores.py
----------------
experiment_log.csv를 읽어서 시도(attempt)별 train / valid(CV) / lb(제출 점수) 를
막대그래프 3개(초록/파랑/빨강)로 그려주는 스크립트.

강사님 팁 반영:
- 그래프는 무조건 그려서 기록으로 남길 것
- 막대가 두 개가 아니라 "세 개"여야 함
    - 초록 = train_score (학습 데이터 정확도)
    - 파랑 = valid_score (로컬 K-Fold 교차검증 평균 정확도) ← 실제로는 이게 세번째 막대
    - 빨강 = lb_score (Kaggle 제출 후 실제 리더보드 점수)
- train-valid 갭이 크게 벌어지면(과적합 의심) 자동으로 경고 출력

사용법:
    1) experiment_log.csv 에 시도 한 줄 추가 (train_score / valid_score / lb_score 채우기)
    2) python plot_scores.py 실행
    3) score_history.png 생성됨 + 콘솔에 과적합 의심 경고 출력됨
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정 (없으면 그냥 기본 폰트로 진행 - 라벨이 네모로 깨질 수 있음)
for _font_path in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]:
    if os.path.exists(_font_path):
        font_manager.fontManager.addfont(_font_path)
        matplotlib.rc("font", family=font_manager.FontProperties(fname=_font_path).get_name())
        break
matplotlib.rcParams["axes.unicode_minus"] = False

CSV_PATH = "experiment_log.csv"
OUT_PATH = "score_history.png"
OVERFIT_GAP_THRESHOLD = 0.10  # train - valid 갭이 이 값보다 크면 경고

# 이번 대회의 평가지표 이름 (강사님 원본 그래프처럼 범용 "스코어"로 표시).
# 타이타닉은 실제로 accuracy로 채점되니 이렇게 써도 되고, 나중에 RMSE/F1/AUC를 쓰는
# 다른 대회에 이 스크립트를 재사용할 땐 이 한 줄만 바꾸면 됨.
SCORE_METRIC_NAME = "스코어 (Accuracy)"
# 지표가 0~1 사이로 고정된 게 아니면(RMSE 등) None으로 바꿔서 자동 스케일 쓸 것
SCORE_RANGE = (0, 1.0)


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float_or_none(value):
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[에러] {CSV_PATH} 파일을 찾을 수 없음. 같은 폴더에 두고 실행할 것.")
        return

    rows = load_rows(CSV_PATH)
    # 점수가 하나라도 채워진 시도만 그래프에 포함
    plotted = [r for r in rows if any(
        to_float_or_none(r.get(k)) is not None
        for k in ("train_score", "valid_score", "lb_score")
    )]

    if not plotted:
        print("[안내] 아직 채점된 시도가 없음 (train_score/valid_score/lb_score 전부 빈칸). "
              "제출/검증 결과를 채운 뒤 다시 실행할 것.")
        return

    attempt_labels = [f"#{r['attempt_no']}\n{r['date']}" for r in plotted]
    train_scores = [to_float_or_none(r.get("train_score")) or 0 for r in plotted]
    valid_scores = [to_float_or_none(r.get("valid_score")) or 0 for r in plotted]
    lb_scores = [to_float_or_none(r.get("lb_score")) or 0 for r in plotted]

    x = range(len(plotted))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(plotted) * 1.4), 5))

    ax.bar([i - width for i in x], train_scores, width, label="train_score (초록)", color="#2ca02c")
    ax.bar(list(x), valid_scores, width, label="valid_score / CV (파랑)", color="#1f77b4")
    ax.bar([i + width for i in x], lb_scores, width, label="lb_score / 제출 (빨강)", color="#d62728")

    ax.set_xticks(list(x))
    ax.set_xticklabels(attempt_labels)
    ax.set_ylabel(SCORE_METRIC_NAME)
    ax.set_title(f"시도별 {SCORE_METRIC_NAME} 기록 (train vs valid(CV) vs 실제 제출)")
    if SCORE_RANGE is not None:
        ax.set_ylim(*SCORE_RANGE)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"[완료] {OUT_PATH} 저장함")

    # 최고 lb_score 시도 표시
    best_idx = max(range(len(plotted)), key=lambda i: lb_scores[i])
    print(f"[요약] 지금까지 lb_score 최고 시도: #{plotted[best_idx]['attempt_no']} "
          f"({plotted[best_idx]['date']}, lb_score={lb_scores[best_idx]:.4f})")

    # 과적합(오버피팅) 의심 경고
    print("\n[과적합 점검] train_score - valid_score 갭이 "
          f"{OVERFIT_GAP_THRESHOLD} 이상인 시도:")
    warned = False
    for r, t, v in zip(plotted, train_scores, valid_scores):
        if t and v and (t - v) >= OVERFIT_GAP_THRESHOLD:
            warned = True
            print(f"  ⚠️  #{r['attempt_no']} ({r['date']}): "
                  f"train={t:.4f}, valid={v:.4f}, gap={t - v:.4f} "
                  f"→ 모델이 train 데이터에 과적합했을 가능성 있음. "
                  f"feature 줄이기 / 규제 강화 / max_depth 낮추기 등을 고려할 것")
    if not warned:
        print("  없음 (양호)")


if __name__ == "__main__":
    main()
