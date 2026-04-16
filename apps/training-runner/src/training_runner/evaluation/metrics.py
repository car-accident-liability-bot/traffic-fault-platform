"""
교통사고 과실비율 Video QA 평가 지표

question_type별 1차 지표:
  fault_ratio           -> Exact Match (핵심 지표) + MAE (2차)
  fault_compare         -> Accuracy
  accident_place        -> ROUGE-L + Exact Match (2차)
  accident_place_feature-> ROUGE-L + BERTScore (2차)
  vehicle_a_progress    -> ROUGE-L + BERTScore (2차)
  vehicle_b_progress    -> ROUGE-L + BERTScore (2차)
"""
from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# question_type을 1차 지표 종류로 분류
_ROUGE_L_TYPES = {
    "accident_place",
    "accident_place_feature",
    "vehicle_a_progress",
    "vehicle_b_progress",
}
_EM_TYPES = {"fault_ratio"}
_ACC_TYPES = {"fault_compare"}

# BERTScore 2차 지표 적용 대상
_BERT_SCORE_TYPES = {
    "accident_place_feature",
    "vehicle_a_progress",
    "vehicle_b_progress",
}


# ---------------------------------------------------------------------------
# ROUGE-L (LCS 기반, 외부 라이브러리 없음)
# ---------------------------------------------------------------------------

def _lcs_length(x: list[str], y: list[str]) -> int:
    """Longest Common Subsequence 길이를 DP로 계산합니다."""
    m, n = len(x), len(y)
    # 메모리 절약: 현재 행과 이전 행만 유지
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 점수를 반환합니다 (0.0 ~ 1.0).

    Args:
        prediction: 모델 생성 답변
        reference:  정답 텍스트

    Returns:
        ROUGE-L F1 (precision × recall 조화평균)
    """
    pred_tokens = prediction.strip().split()
    ref_tokens = reference.strip().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> bool:
    """공백 정규화 후 문자열 완전 일치 여부를 반환합니다."""
    return _normalize(prediction) == _normalize(reference)


def _normalize(text: str) -> str:
    """평가용 정규화: 소문자 변환, 연속 공백 단일화, 앞뒤 공백 제거."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# fault_ratio 파싱 및 MAE
# ---------------------------------------------------------------------------

def parse_fault_ratio(text: str) -> tuple[float, float] | None:
    """텍스트에서 A·B 과실 비율(%)을 파싱합니다.

    지원 형식 예시:
      "A 60% B 40%", "A:60, B:40", "60:40",
      "A가 60%, B가 40%", "A 60 B 40"

    Returns:
        (a_ratio, b_ratio) 또는 파싱 실패 시 None
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < 2:
        return None
    a, b = float(numbers[0]), float(numbers[1])
    # 합이 100에서 크게 벗어나면 파싱 실패로 처리 (허용 오차 5)
    if abs(a + b - 100.0) > 5.0:
        return None
    return (a, b)


def fault_ratio_mae(prediction: str, reference: str) -> float:
    """fault_ratio의 평균 절대 오차(MAE)를 반환합니다.

    파싱 실패 시 최대 오차 100.0을 반환합니다.
    """
    pred = parse_fault_ratio(prediction)
    ref = parse_fault_ratio(reference)
    if pred is None or ref is None:
        return 100.0
    return (abs(pred[0] - ref[0]) + abs(pred[1] - ref[1])) / 2.0


# ---------------------------------------------------------------------------
# BERTScore (배치 처리 — 외부에서 일괄 계산 후 결과에 삽입)
# ---------------------------------------------------------------------------

def compute_bert_scores(
    predictions: list[str],
    references: list[str],
    lang: str = "ko",
    model_type: str | None = None,
    device: str | None = None,
    batch_size: int = 64,
    verbose: bool = False,
) -> list[float]:
    """BERTScore F1을 배치로 계산해 리스트로 반환합니다.

    Args:
        predictions:  모델 생성 답변 리스트
        references:   정답 텍스트 리스트
        lang:         언어 코드 ('ko' → snunlp/KR-ELECTRA-discriminator 사용)
        model_type:   명시적 모델 지정 시 lang 대신 사용 (예: 'klue/roberta-base')
        device:       계산 장치 ('cuda', 'cpu' 등; None이면 자동 선택)
        batch_size:   BERTScore 배치 크기
        verbose:      진행 상황 출력 여부

    Returns:
        F1 점수 리스트 (0.0 ~ 1.0)

    Raises:
        ImportError: bert-score 패키지 미설치 시
    """
    try:
        from bert_score import score as _bert_score_fn
    except ImportError as exc:
        raise ImportError(
            "bert-score 패키지가 필요합니다: pip install bert-score"
        ) from exc

    if not predictions:
        return []

    kwargs: dict = dict(batch_size=batch_size, verbose=verbose)
    if model_type:
        kwargs["model_type"] = model_type
    else:
        kwargs["lang"] = lang
    if device:
        kwargs["device"] = device

    _, _, f1 = _bert_score_fn(predictions, references, **kwargs)
    return f1.tolist()


# ---------------------------------------------------------------------------
# 샘플 단위 지표 계산
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    prediction: str,
    reference: str,
    question_type: str,
) -> dict[str, float | bool]:
    """단일 샘플의 모든 해당 지표를 계산합니다.

    Note:
        bert_score 필드는 None으로 초기화됩니다.
        evaluator.run_evaluation() 내부에서 배치 계산 후 채워집니다.

    Returns:
        {
            "rouge_l": float,          # ROUGE-L 타입이면 계산, 아니면 None
            "exact_match": bool,       # EM 타입 또는 accident_place
            "mae": float,              # fault_ratio 타입이면 계산, 아니면 None
            "bert_score": float|None,  # BERTScore 대상 타입이면 사후 채워짐
            "primary_score": float,    # question_type별 1차 지표 (비교에 사용)
        }
    """
    result: dict[str, float | bool | None] = {
        "rouge_l": None,
        "exact_match": None,
        "mae": None,
        "bert_score": None,
        "primary_score": 0.0,
    }

    if question_type in _ROUGE_L_TYPES:
        rl = rouge_l(prediction, reference)
        result["rouge_l"] = rl
        result["primary_score"] = rl
        # accident_place는 EM도 함께 계산 (2차 지표)
        if question_type == "accident_place":
            result["exact_match"] = exact_match(prediction, reference)
        # bert_score 대상 타입은 None 유지 — evaluator에서 배치 계산 후 삽입

    elif question_type in _EM_TYPES:
        em = exact_match(prediction, reference)
        mae = fault_ratio_mae(prediction, reference)
        result["exact_match"] = em
        result["mae"] = mae
        result["primary_score"] = float(em)

    elif question_type in _ACC_TYPES:
        em = exact_match(prediction, reference)
        result["exact_match"] = em
        result["primary_score"] = float(em)

    else:
        # 알 수 없는 타입 — ROUGE-L 기본 적용
        rl = rouge_l(prediction, reference)
        result["rouge_l"] = rl
        result["primary_score"] = rl

    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def aggregate_metrics(sample_results: list[dict]) -> dict:
    """샘플 단위 결과 리스트에서 통합 평가 지표를 집계합니다.

    Args:
        sample_results: compute_sample_metrics 결과 + video_id / category /
                        question_type / prediction / answer 가 포함된 dict 리스트

    Returns:
        {
            "overall": { "rouge_l_mean", "fault_ratio_em", "fault_compare_acc", ... },
            "by_question_type": { question_type: { ... } },
            "by_category": { category: { "rouge_l_mean", "n_samples", ... } },
            "worst_categories": [ (category, score), ... ],  # 하위 5개
        }
    """
    if not sample_results:
        return {}

    # question_type 별 버킷
    by_qtype: dict[str, list[dict]] = defaultdict(list)
    # category (case_code) 별 버킷
    by_cat: dict[str, list[dict]] = defaultdict(list)

    for r in sample_results:
        by_qtype[r["question_type"]].append(r)
        by_cat[r["category"]].append(r)

    # question_type별 집계
    qtype_summary: dict[str, dict] = {}
    for qtype, rows in by_qtype.items():
        qtype_summary[qtype] = _summarize_bucket(rows, qtype)

    # category별 집계 (primary_score 평균 사용)
    cat_summary: dict[str, dict] = {}
    for cat, rows in by_cat.items():
        primary_scores = [r["primary_score"] for r in rows]
        cat_summary[cat] = {
            "n_samples": len(rows),
            "primary_score_mean": mean(primary_scores),
        }

    # 전체 지표
    all_rouge_l = [r["rouge_l"] for r in sample_results if r.get("rouge_l") is not None]
    all_bert_score = [r["bert_score"] for r in sample_results if r.get("bert_score") is not None]
    fault_ratio_rows = [r for r in sample_results if r["question_type"] == "fault_ratio"]
    fault_compare_rows = [r for r in sample_results if r["question_type"] == "fault_compare"]

    overall = {
        "n_samples": len(sample_results),
        "rouge_l_mean": mean(all_rouge_l) if all_rouge_l else None,
        "bert_score_mean": mean(all_bert_score) if all_bert_score else None,
        "fault_ratio_em": (
            mean(float(r["exact_match"]) for r in fault_ratio_rows)
            if fault_ratio_rows else None
        ),
        "fault_ratio_mae_mean": (
            mean(r["mae"] for r in fault_ratio_rows if r.get("mae") is not None)
            if fault_ratio_rows else None
        ),
        "fault_compare_acc": (
            mean(float(r["exact_match"]) for r in fault_compare_rows)
            if fault_compare_rows else None
        ),
    }

    # 하위 5개 카테고리
    sorted_cats = sorted(
        cat_summary.items(), key=lambda x: x[1]["primary_score_mean"]
    )
    worst_categories = [(c, v["primary_score_mean"]) for c, v in sorted_cats[:5]]

    return {
        "overall": overall,
        "by_question_type": qtype_summary,
        "by_category": cat_summary,
        "worst_categories": worst_categories,
    }


def _summarize_bucket(rows: list[dict], question_type: str) -> dict:
    """단일 question_type 버킷의 지표를 집계합니다."""
    n = len(rows)
    summary: dict[str, object] = {"n_samples": n}

    rouge_vals = [r["rouge_l"] for r in rows if r.get("rouge_l") is not None]
    em_vals = [r["exact_match"] for r in rows if r.get("exact_match") is not None]
    mae_vals = [r["mae"] for r in rows if r.get("mae") is not None]
    bert_score_vals = [r["bert_score"] for r in rows if r.get("bert_score") is not None]

    if rouge_vals:
        summary["rouge_l_mean"] = mean(rouge_vals)
    if em_vals:
        summary["exact_match"] = mean(float(v) for v in em_vals)
    if mae_vals:
        summary["mae_mean"] = mean(mae_vals)
    if bert_score_vals:
        summary["bert_score_mean"] = mean(bert_score_vals)

    return summary


# ---------------------------------------------------------------------------
# 결과 출력 헬퍼
# ---------------------------------------------------------------------------

def format_summary(aggregated: dict, experiment_name: str = "") -> str:
    """집계 결과를 사람이 읽기 쉬운 텍스트로 포맷합니다."""
    lines: list[str] = []
    header = f"=== 평가 결과 {f'[{experiment_name}]' if experiment_name else ''} ==="
    lines.append(header)

    overall = aggregated.get("overall", {})
    lines.append(f"  샘플 수          : {overall.get('n_samples', 'N/A')}")
    lines.append(
        f"  fault_ratio EM   : {_fmt(overall.get('fault_ratio_em'))}  ← 핵심 지표"
    )
    lines.append(
        f"  fault_ratio MAE  : {_fmt(overall.get('fault_ratio_mae_mean'))}"
    )
    lines.append(
        f"  fault_compare Acc: {_fmt(overall.get('fault_compare_acc'))}"
    )
    lines.append(
        f"  ROUGE-L 평균     : {_fmt(overall.get('rouge_l_mean'))}"
    )
    lines.append(
        f"  BERTScore 평균   : {_fmt(overall.get('bert_score_mean'))}"
    )

    lines.append("\n[question_type별]")
    for qtype, stats in aggregated.get("by_question_type", {}).items():
        parts = []
        if "rouge_l_mean" in stats:
            parts.append(f"ROUGE-L={_fmt(stats['rouge_l_mean'])}")
        if "exact_match" in stats:
            parts.append(f"EM={_fmt(stats['exact_match'])}")
        if "mae_mean" in stats:
            parts.append(f"MAE={_fmt(stats['mae_mean'])}")
        if "bert_score_mean" in stats:
            parts.append(f"BERTScore={_fmt(stats['bert_score_mean'])}")
        lines.append(f"  {qtype:<30} {' | '.join(parts)}")

    worst = aggregated.get("worst_categories", [])
    if worst:
        lines.append("\n[하위 5개 카테고리 (primary_score 기준)]")
        for cat, score in worst:
            lines.append(f"  {cat:<30} {score:.4f}")

    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A  "
    return f"{value:.4f}"
