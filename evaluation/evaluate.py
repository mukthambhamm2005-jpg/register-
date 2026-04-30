"""
Evaluation Script
=================
Entry point for model evaluation on the test set.

Computes:
- mAP@0.5
- Precision, Recall, F1-Score
- Per-class AP
- Confusion Matrix

Usage:
    python -m evaluation.evaluate
    python main.py --mode evaluate
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

import config
from models.yolo_model import build_model
from preprocessing.dataset_adapter import get_data_loaders
from evaluation.metrics import DetectionMetrics, ConfusionMatrix


def load_trained_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
    iou_threshold: float = config.IOU_THRESHOLD
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print(" 🔍 RUNNING EVALUATION")
    print("=" * 60)
    
    # Initialize metrics
    metrics = DetectionMetrics(
        num_classes=config.NUM_CLASSES,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold
    )
    
    confusion_matrix = ConfusionMatrix(
        num_classes=config.NUM_CLASSES,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # Process test set
    print(f"\n📊 Evaluating on {len(test_loader)} batches...")
    
    for batch_idx, (images, labels_list, paths) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Decode predictions for each image in batch
        for b in range(images.shape[0]):
            # Get predictions for this image
            pred_boxes, pred_scores, pred_classes = model.detection_head.decode_predictions(
                [p[b:b+1] for p in predictions],
                conf_threshold=conf_threshold
            )
            
            # Apply NMS
            if len(pred_boxes) > 0:
                keep_indices = torch.ops.torchvision.nms(pred_boxes, pred_scores, iou_threshold)
                pred_boxes = pred_boxes[keep_indices]
                pred_scores = pred_scores[keep_indices]
                pred_classes = pred_classes[keep_indices]
            
            # Get targets for this image
            targets = labels_list[b]
            
            if len(targets) > 0:
                # Convert normalized targets to pixel coordinates
                target_boxes = targets[:, 1:5]  # [x_center, y_center, width, height]
                target_classes = targets[:, 0].long()
                
                # Convert xywh to x1y1x2y2
                img_size = config.IMAGE_SIZE
                x_center = target_boxes[:, 0] * img_size
                y_center = target_boxes[:, 1] * img_size
                width = target_boxes[:, 2] * img_size
                height = target_boxes[:, 3] * img_size
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                target_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            else:
                target_boxes = torch.zeros((0, 4))
                target_classes = torch.zeros((0,), dtype=torch.long)
            
            # Add to metrics
            metrics.add_batch(
                pred_boxes, pred_scores, pred_classes,
                target_boxes, target_classes
            )
            
            # Add to confusion matrix
            confusion_matrix.process_batch(
                pred_boxes, pred_scores, pred_classes,
                target_boxes, target_classes
            )
    
    # Compute metrics
    results = metrics.compute()
    
    # Print results
    metrics.print_results(results)
    
    # Print confusion matrix
    confusion_matrix.print_matrix()
    
    # Save confusion matrix plot
    cm_path = config.OUTPUT_DIR / 'confusion_matrix.png'
    confusion_matrix.plot(save_path=cm_path)
    
    return results


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print(" 🚗 ROAD DAMAGE DETECTION - MODEL EVALUATION")
    print("=" * 70 + "\n")
    
    # Print configuration
    config.print_config()
    
    # Check device
    device = config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, switching to CPU")
        device = "cpu"
    
    print(f"\n🖥️  Device: {device}")
    
    # Load test data
    print("\n📦 Loading test dataset...")
    _, _, test_loader = get_data_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=0,  # Use 0 for Windows
        image_size=config.IMAGE_SIZE,
        augment_train=False,
        preprocess=True
    )
    
    # Find best checkpoint
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    
    if not checkpoint_path.exists():
        # Try latest checkpoint
        checkpoint_path = config.CHECKPOINT_DIR / "checkpoint_epoch_latest.pth"
        
        if not checkpoint_path.exists():
            # Try any checkpoint
            checkpoints = list(config.CHECKPOINT_DIR.glob("*.pth"))
            if checkpoints:
                checkpoint_path = checkpoints[0]
            else:
                print("\n❌ No checkpoint found! Please train the model first.")
                print(f"   Expected at: {config.CHECKPOINT_DIR}")
                return
    
    # Load model
    model = load_trained_model(checkpoint_path, device)
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD
    )
    
    # Save results
    results_path = config.OUTPUT_DIR / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ROAD DAMAGE DETECTION - EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"mAP@0.5:    {results['mAP@0.5']:.4f}\n")
        f.write(f"Precision:  {results['precision']:.4f}\n")
        f.write(f"Recall:     {results['recall']:.4f}\n")
        f.write(f"F1-Score:   {results['f1_score']:.4f}\n\n")
        f.write("Per-Class AP:\n")
        for class_name, ap in results['per_class_ap'].items():
            f.write(f"  {class_name:20s}: {ap:.4f}\n")
        f.write("\n")
        f.write(f"True Positives:  {results['total_tp']}\n")
        f.write(f"False Positives: {results['total_fp']}\n")
        f.write(f"False Negatives: {results['total_fn']}\n")
    
    print(f"\n📄 Results saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print(" ✅ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

