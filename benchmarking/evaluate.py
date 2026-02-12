# benchmarks/evaluate.py
"""Evaluation metrics."""

from typing import List, Dict
import re


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check exact match."""
    return predicted.strip().lower() == ground_truth.strip().lower()


def contains_match(predicted: str, ground_truth: str) -> bool:
    """Check if prediction contains ground truth."""
    return ground_truth.strip().lower() in predicted.strip().lower()


def number_match(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Check if numbers match within tolerance."""
    
    # Extract numbers
    pred_numbers = re.findall(r'[\d,]+\.?\d*', predicted.replace(',', ''))
    gt_numbers = re.findall(r'[\d,]+\.?\d*', ground_truth.replace(',', ''))
    
    if not pred_numbers or not gt_numbers:
        return False
    
    try:
        pred_val = float(pred_numbers[0])
        gt_val = float(gt_numbers[0])
        
        # Check within tolerance
        return abs(pred_val - gt_val) / gt_val < tolerance
    except:
        return False


def calculate_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Calculate accuracy metrics."""
    
    total = len(results)
    if total == 0:
        return {}
    
    exact_matches = sum(1 for r in results if r.get('exact_match', False))
    contains_matches = sum(1 for r in results if r.get('contains_match', False))
    number_matches = sum(1 for r in results if r.get('number_match', False))
    
    return {
        'total': total,
        'exact_match_accuracy': exact_matches / total,
        'contains_accuracy': contains_matches / total,
        'number_accuracy': number_matches / total,
        'avg_confidence': sum(r.get('confidence', 0) for r in results) / total,
        'avg_latency_ms': sum(r.get('latency_ms', 0) for r in results) / total
    }


__all__ = ['exact_match', 'contains_match', 'number_match', 'calculate_accuracy']