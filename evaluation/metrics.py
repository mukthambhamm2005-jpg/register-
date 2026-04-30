"""
Evaluation Metrics Module
=========================
Implements object detection evaluation metrics:
- mAP@0.5 (mean Average Precision at IoU=0.5)
- Precision
- Recall
- F1-Score
- Confusion Matrix

Uses torchmetrics for standardized mAP computation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

import config


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        box1: (N, 4) tensor [x1, y1, x2, y2]
        box2: (M, 4) tensor [x1, y1, x2, y2]
    
    Returns:
        iou: (N, M) tensor of IoU values
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # (N, 1, 4)
    box2 = box2.unsqueeze(0)  # (1, M, 4)
    
    # Compute intersection
    x1_max = torch.max(box1[..., 0], box2[..., 0])
    y1_max = torch.max(box1[..., 1], box2[..., 1])
    x2_min = torch.min(box1[..., 2], box2[..., 2])
    y2_min = torch.min(box1[..., 3], box2[..., 3])
    
    inter_w = (x2_min - x1_max).clamp(min=0)
    inter_h = (y2_min - y1_max).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Compute union
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        recall: Array of recall values
        precision: Array of precision values
    
    Returns:
        Average Precision score
    """
    # Add sentinel values
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    
    # Compute precision envelope
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # Compute AP
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return float(ap)


class DetectionMetrics:
    """
    Computes detection metrics: mAP, Precision, Recall, F1-Score.
    
    Accumulates predictions and targets across all batches,
    then computes metrics at the end.
    
    Args:
        num_classes: Number of object classes
        iou_threshold: IoU threshold for matching predictions to targets
        conf_threshold: Confidence threshold for predictions
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        iou_threshold: float = 0.5,
        conf_threshold: float = config.CONFIDENCE_THRESHOLD
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        
        # Storage for predictions and targets
        self.predictions = []  # List of dicts: {'boxes': (N, 4), 'scores': (N,), 'classes': (N,)}
        self.targets = []      # List of dicts: {'boxes': (M, 4), 'classes': (M,)}
        
        # Per-class statistics
        self.class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    def add_batch(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_classes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor
    ):
        """
        Add a batch of predictions and targets.
        
        Args:
            pred_boxes: (N, 4) predicted boxes [x1, y1, x2, y2]
            pred_scores: (N,) confidence scores
            pred_classes: (N,) predicted class indices
            target_boxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
            target_classes: (M,) ground truth class indices
        """
        # Filter by confidence threshold
        if len(pred_scores) > 0:
            mask = pred_scores >= self.conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_classes = pred_classes[mask]
        
        self.predictions.append({
            'boxes': pred_boxes.cpu(),
            'scores': pred_scores.cpu(),
            'classes': pred_classes.cpu()
        })
        
        self.targets.append({
            'boxes': target_boxes.cpu(),
            'classes': target_classes.cpu()
        })
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing:
            - mAP@0.5: mean Average Precision
            - precision: Overall precision
            - recall: Overall recall
            - f1_score: Overall F1 score
            - per_class_ap: AP for each class
        """
        # Organize predictions by class
        class_predictions = defaultdict(list)
        class_targets = defaultdict(int)
        
        for pred, target in zip(self.predictions, self.targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_classes = pred['classes']
            target_boxes = target['boxes']
            target_classes = target['classes']
            
            # Count targets per class
            for cls in range(self.num_classes):
                class_targets[cls] += (target_classes == cls).sum().item()
            
            # Group predictions by class
            for cls in range(self.num_classes):
                cls_mask = pred_classes == cls
                if cls_mask.any():
                    cls_boxes = pred_boxes[cls_mask]
                    cls_scores = pred_scores[cls_mask]
                    
                    # Sort by confidence (descending)
                    sorted_indices = torch.argsort(cls_scores, descending=True)
                    cls_boxes = cls_boxes[sorted_indices]
                    cls_scores = cls_scores[sorted_indices]
                    
                    class_predictions[cls].append({
                        'boxes': cls_boxes,
                        'scores': cls_scores,
                        'target_boxes': target_boxes[target_classes == cls],
                        'image_idx': len(class_predictions[cls])
                    })
        
        # Compute AP for each class
        per_class_ap = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for cls in range(self.num_classes):
            cls_preds = class_predictions[cls]
            
            if len(cls_preds) == 0:
                # No predictions for this class
                per_class_ap[cls] = 0.0
                total_fn += class_targets[cls]
                continue
            
            # Collect all predictions for this class
            all_scores = []
            all_tp = []
            
            for pred_info in cls_preds:
                pred_boxes = pred_info['boxes']
                target_boxes = pred_info['target_boxes']
                
                if len(target_boxes) == 0:
                    # No targets, all predictions are FP
                    all_scores.extend(pred_info['scores'].tolist())
                    all_tp.extend([0] * len(pred_boxes))
                    continue
                
                # Compute IoU with all targets
                ious = compute_iou(pred_boxes, target_boxes)  # (N, M)
                
                # Match predictions to targets
                matched_targets = set()
                for pred_idx in range(len(pred_boxes)):
                    max_iou, target_idx = ious[pred_idx].max(dim=0)
                    
                    if max_iou >= self.iou_threshold and target_idx.item() not in matched_targets:
                        all_tp.append(1)
                        matched_targets.add(target_idx.item())
                    else:
                        all_tp.append(0)
                    
                    all_scores.append(pred_info['scores'][pred_idx].item())
            
            # Sort by score descending
            sorted_indices = np.argsort(all_scores)[::-1]
            tp_sorted = np.array(all_tp)[sorted_indices]
            
            # Compute cumulative precision and recall
            tp_cumsum = np.cumsum(tp_sorted)
            fp_cumsum = np.cumsum(1 - tp_sorted)
            
            recalls = tp_cumsum / max(class_targets[cls], 1)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
            
            # Compute AP
            ap = compute_ap(recalls, precisions)
            per_class_ap[cls] = ap
            
            # Update totals
            total_tp += tp_sorted.sum()
            total_fp += (1 - tp_sorted).sum()
            total_fn += max(class_targets[cls] - tp_sorted.sum(), 0)
        
        # Compute overall metrics
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-7)
        
        # mAP
        map_score = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        
        return {
            'mAP@0.5': float(map_score),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'per_class_ap': {config.CLASS_NAMES[k]: v for k, v in per_class_ap.items()},
            'total_tp': int(total_tp),
            'total_fp': int(total_fp),
            'total_fn': int(total_fn)
        }
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []
        self.class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    def print_results(self, results: Dict[str, float]):
        """Pretty print evaluation results."""
        print("\n" + "=" * 60)
        print(" 📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n  mAP@0.5:    {results['mAP@0.5']:.4f}")
        print(f"  Precision:  {results['precision']:.4f}")
        print(f"  Recall:     {results['recall']:.4f}")
        print(f"  F1-Score:   {results['f1_score']:.4f}")
        print(f"\n  True Positives:  {results['total_tp']}")
        print(f"  False Positives: {results['total_fp']}")
        print(f"  False Negatives: {results['total_fn']}")
        print(f"\n  Per-Class AP:")
        for class_name, ap in results['per_class_ap'].items():
            print(f"    {class_name:20s}: {ap:.4f}")
        print("=" * 60 + "\n")


class ConfusionMatrix:
    """
    Confusion matrix for object detection.
    
    Tracks:
    - Correct detections (diagonal)
    - Misclassifications (off-diagonal)
    - Background detections (false positives)
    - Missed detections (false negatives)
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, conf_threshold: float = 0.5, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        # Last row/column represents background
    
    def process_batch(self, pred_boxes, pred_scores, pred_classes, target_boxes, target_classes):
        """Process a batch of predictions and targets."""
        # Filter by confidence
        if len(pred_scores) > 0:
            mask = pred_scores >= self.conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_classes = pred_classes[mask]
        
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            return
        
        if len(target_boxes) == 0:
            # All predictions are false positives (background)
            for cls in pred_classes:
                self.matrix[self.num_classes, int(cls)] += 1
            return
        
        if len(pred_boxes) == 0:
            # All targets are missed (false negatives)
            for cls in target_classes:
                self.matrix[int(cls), self.num_classes] += 1
            return
        
        # Compute IoU
        ious = compute_iou(pred_boxes, target_boxes)
        
        # Match predictions to targets
        matched_targets = set()
        
        for pred_idx in range(len(pred_boxes)):
            max_iou, target_idx = ious[pred_idx].max(dim=0)
            
            if max_iou >= self.iou_threshold and target_idx.item() not in matched_targets:
                # Match found
                pred_cls = int(pred_classes[pred_idx])
                target_cls = int(target_classes[target_idx])
                self.matrix[target_cls, pred_cls] += 1
                matched_targets.add(target_idx.item())
            else:
                # False positive (background detection)
                pred_cls = int(pred_classes[pred_idx])
                self.matrix[self.num_classes, pred_cls] += 1
        
        # Unmatched targets are false negatives
        for target_idx in range(len(target_boxes)):
            if target_idx not in matched_targets:
                target_cls = int(target_classes[target_idx])
                self.matrix[target_cls, self.num_classes] += 1
    
    def plot(self, save_path=None):
        """Plot confusion matrix."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create labels
            class_names = [config.CLASS_NAMES[i] for i in range(self.num_classes)]
            labels = class_names + ['Background']
            
            # Normalize by row (recall)
            matrix_norm = self.matrix.astype(np.float32)
            row_sums = matrix_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            matrix_norm = matrix_norm / row_sums
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                matrix_norm,
                annot=True,
                fmt='.2f',
                xticklabels=labels,
                yticklabels=labels,
                cmap='Blues',
                cbar_kws={'label': 'Proportion'}
            )
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"  📊 Confusion matrix saved: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("  ⚠️ matplotlib/seaborn not available for plotting")
    
    def print_matrix(self):
        """Print confusion matrix as text."""
        print("\nConfusion Matrix:")
        class_names = [config.CLASS_NAMES[i] for i in range(self.num_classes)]
        labels = class_names + ['BG']
        
        # Print header
        print("      ", end="")
        for label in labels:
            print(f"{label[:8]:>8}", end="")
        print()
        
        # Print rows
        for i, label in enumerate(labels):
            print(f"{label[:8]:>8}", end="")
            for j in range(len(labels)):
                print(f"{self.matrix[i, j]:>8}", end="")
            print()


if __name__ == "__main__":
    """Demo: Test metrics computation."""
    print("\n" + "=" * 60)
    print(" 📊 METRICS DEMO")
    print("=" * 60 + "\n")
    
    metrics = DetectionMetrics(num_classes=4)
    
    # Simulate some predictions and targets
    for i in range(3):
        # Predictions
        pred_boxes = torch.tensor([
            [10, 10, 50, 50],
            [60, 60, 100, 100],
            [110, 110, 150, 150],
        ], dtype=torch.float32)
        pred_scores = torch.tensor([0.9, 0.8, 0.7])
        pred_classes = torch.tensor([0, 1, 2])
        
        # Targets
        target_boxes = torch.tensor([
            [12, 12, 52, 52],
            [62, 62, 102, 102],
            [200, 200, 240, 240],  # Missed detection
        ], dtype=torch.float32)
        target_classes = torch.tensor([0, 1, 3])
        
        metrics.add_batch(pred_boxes, pred_scores, pred_classes, target_boxes, target_classes)
    
    # Compute metrics
    results = metrics.compute()
    metrics.print_results(results)
    
    # Confusion matrix
    cm = ConfusionMatrix(num_classes=4)
    for i in range(3):
        pred_boxes = torch.tensor([
            [10, 10, 50, 50],
            [60, 60, 100, 100],
            [110, 110, 150, 150],
        ], dtype=torch.float32)
        pred_scores = torch.tensor([0.9, 0.8, 0.7])
        pred_classes = torch.tensor([0, 1, 2])
        
        target_boxes = torch.tensor([
            [12, 12, 52, 52],
            [62, 62, 102, 102],
            [200, 200, 240, 240],
        ], dtype=torch.float32)
        target_classes = torch.tensor([0, 1, 3])
        
        cm.process_batch(pred_boxes, pred_scores, pred_classes, target_boxes, target_classes)
    
    cm.print_matrix()
    
    print("\n✅ Metrics demo complete!")

