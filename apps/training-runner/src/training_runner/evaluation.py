"""평가 데이터 변환 레이어"""
from typing import List, Dict
from transformers.trainer_utils import PredictionOutput
import torch


def convert_trainer_predictions(
    predictions: PredictionOutput,
    test_dataset,
    processor
) -> List[Dict[str, str]]:
    """
    Trainer.predict() 출력을 QAEvaluator 입력 형식으로 변환
    
    Args:
        predictions: trainer.predict()의 반환값 (PredictionOutput)
            - predictions.predictions: (N, seq_len, vocab_size) 또는 (N, seq_len)
            - predictions.label_ids: (N, seq_len)
        test_dataset: TextQADataset (video_id, question_type, answer 정보 포함)
        processor: Qwen2VLProcessor (디코딩용)
    
    Returns:
        [{
            "video_id": str,
            "question_type": str,
            "prediction": str,
            "ground_truth": str
        }, ...]
    """
    results = []
    
    # predictions.predictions 차원 확인
    # Seq2Seq 모델: (batch_size, seq_len) - 이미 argmax된 토큰 ID
    # Causal LM: (batch_size, seq_len, vocab_size) - logits
    pred_outputs = predictions.predictions
    
    # logits인 경우 argmax로 토큰 ID 추출
    if len(pred_outputs.shape) == 3:
        # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len)
        pred_ids = torch.argmax(torch.tensor(pred_outputs), dim=-1)
    else:
        # 이미 토큰 ID
        pred_ids = pred_outputs
    
    # 각 샘플 처리
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        
        try:
            # 토큰 ID를 텍스트로 디코딩
            pred_tokens = pred_ids[i]
            
            # numpy array를 list로 변환 (processor.decode가 list 요구)
            if hasattr(pred_tokens, 'tolist'):
                pred_tokens = pred_tokens.tolist()
            
            # 디코딩
            pred_text = processor.decode(
                pred_tokens, 
                skip_special_tokens=True
            )
            
        except Exception as e:
            # 디코딩 실패 시 에러 처리
            print(f"⚠️  샘플 {i} 디코딩 실패: {e}")
            pred_text = "[DECODE_ERROR]"
        
        # 결과 추가
        results.append({
            "video_id": sample.video_id,
            "question_type": sample.question_type,
            "prediction": pred_text.strip(),
            "ground_truth": sample.answer
        })
    
    return results
