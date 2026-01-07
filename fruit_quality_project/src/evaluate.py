"""
Evaluation module for fruit quality classification.
Generates metrics, confusion matrices, and prediction samples.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from .dataset import QUALITY_NAMES
from .utils import print_section


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None
) -> Dict:
    """
    Evaluate model and compute all metrics.
    
    Args:
        model: Trained PyTorch model
        dataloader: Test dataloader
        device: Device to evaluate on
        
    Returns:
        Dictionary with all metrics and predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'f1_macro': float(f1_macro),
        'per_class': {
            QUALITY_NAMES[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
            for i in range(len(QUALITY_NAMES))
        },
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    # Print summary
    print_section("Evaluation Results")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (weighted): {precision * 100:.2f}%")
    print(f"Recall (weighted): {recall * 100:.2f}%")
    print(f"F1 Score (weighted): {f1 * 100:.2f}%")
    print(f"F1 Score (macro): {f1_macro * 100:.2f}%")
    print("\nPer-class metrics:")
    for i, name in enumerate(QUALITY_NAMES):
        print(f"  {name}: P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")
    
    return metrics


def generate_confusion_matrix(
    metrics: Dict,
    output_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """
    Generate and save confusion matrix visualization.
    
    Args:
        metrics: Metrics dictionary from evaluate_model
        output_path: Path to save the figure
        title: Title for the plot
    """
    cm = np.array(metrics['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=QUALITY_NAMES,
        yticklabels=QUALITY_NAMES
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[INFO] Confusion matrix saved to {output_path}")


def save_prediction_samples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: str,
    num_samples: int = 20,
    device: torch.device = None
) -> List[Dict]:
    """
    Save sample predictions with images.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        output_dir: Directory to save samples
        num_samples: Number of samples to save
        device: Device for inference
        
    Returns:
        List of prediction details
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    samples = []
    count = 0
    
    # Get original images from dataset
    dataset = dataloader.dataset
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            img_path, true_label = dataset.samples[idx]
            
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            
            # Get prediction
            if dataset.transform:
                img_tensor = dataset.transform(image).unsqueeze(0).to(device)
            else:
                from .dataset import get_transforms
                transform = get_transforms("test")
                img_tensor = transform(image).unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            
            pred_label = pred.item()
            confidence = probs[0][pred_label].item()
            
            # Save image with prediction overlay
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.axis('off')
            
            true_name = QUALITY_NAMES[true_label]
            pred_name = QUALITY_NAMES[pred_label]
            color = 'green' if true_label == pred_label else 'red'
            
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} ({confidence*100:.1f}%)",
                color=color,
                fontsize=12
            )
            
            save_path = os.path.join(output_dir, f"sample_{count:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            samples.append({
                'image_path': img_path,
                'true_label': true_name,
                'predicted_label': pred_name,
                'confidence': float(confidence),
                'correct': true_label == pred_label,
                'output_path': save_path
            })
            
            count += 1
    
    print(f"[INFO] Saved {count} prediction samples to {output_dir}")
    
    return samples


def save_metrics(metrics: Dict, output_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON
    """
    # Remove large arrays for JSON
    metrics_clean = {k: v for k, v in metrics.items() 
                     if k not in ['predictions', 'labels', 'probabilities']}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    
    print(f"[INFO] Metrics saved to {output_path}")


def compare_experiments(
    baseline_metrics: Dict,
    segmented_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Compare baseline vs segmented pipeline results.
    
    Args:
        baseline_metrics: Metrics from baseline model
        segmented_metrics: Metrics from segmented pipeline
        output_path: Optional path to save comparison
        
    Returns:
        Comparison summary string
    """
    print_section("Experiment Comparison")
    
    comparison = """
╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT COMPARISON                      ║
╠══════════════════════════════════════════════════════════════╣
║  Metric              │  Baseline     │  With Segmentation    ║
╠══════════════════════════════════════════════════════════════╣
║  Accuracy            │  {b_acc:>6.2f}%      │  {s_acc:>6.2f}%    ({acc_diff:>+.2f}%)  ║
║  F1 (Macro)          │  {b_f1:>6.2f}%      │  {s_f1:>6.2f}%    ({f1_diff:>+.2f}%)   ║
║  Precision           │  {b_prec:>6.2f}%      │  {s_prec:>6.2f}%    ({prec_diff:>+.2f}%)  ║
║  Recall              │  {b_rec:>6.2f}%      │  {s_rec:>6.2f}%    ({rec_diff:>+.2f}%)   ║
╚══════════════════════════════════════════════════════════════╝
""".format(
        b_acc=baseline_metrics['accuracy'] * 100,
        s_acc=segmented_metrics['accuracy'] * 100,
        acc_diff=(segmented_metrics['accuracy'] - baseline_metrics['accuracy']) * 100,
        b_f1=baseline_metrics['f1_macro'] * 100,
        s_f1=segmented_metrics['f1_macro'] * 100,
        f1_diff=(segmented_metrics['f1_macro'] - baseline_metrics['f1_macro']) * 100,
        b_prec=baseline_metrics['precision_weighted'] * 100,
        s_prec=segmented_metrics['precision_weighted'] * 100,
        prec_diff=(segmented_metrics['precision_weighted'] - baseline_metrics['precision_weighted']) * 100,
        b_rec=baseline_metrics['recall_weighted'] * 100,
        s_rec=segmented_metrics['recall_weighted'] * 100,
        rec_diff=(segmented_metrics['recall_weighted'] - baseline_metrics['recall_weighted']) * 100
    )
    
    # Conclusion
    acc_improvement = (segmented_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    
    if acc_improvement > 2:
        conclusion = "✅ CONCLUSIÓN: La segmentación MEJORA significativamente el desempeño del modelo."
    elif acc_improvement > 0:
        conclusion = "📊 CONCLUSIÓN: La segmentación muestra una ligera mejora en el desempeño."
    elif acc_improvement > -2:
        conclusion = "📊 CONCLUSIÓN: El desempeño es similar con y sin segmentación."
    else:
        conclusion = "⚠️ CONCLUSIÓN: La segmentación no mejora el desempeño en este dataset."
    
    conclusion += "\n   La segmentación es más útil cuando el fondo es variable o hay ruido visual."
    
    full_report = comparison + "\n" + conclusion
    print(full_report)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\n[INFO] Comparison saved to {output_path}")
    
    return full_report
