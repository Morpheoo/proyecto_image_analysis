"""
Main pipeline for fruit quality classification.
Runs the complete training and evaluation workflow.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, get_device, create_directories, print_section
from src.dataset import get_dataloaders, get_sample_images, QUALITY_NAMES
from src.segmentation import segment_image
from src.train import create_model, train_model
from src.evaluate import (
    evaluate_model, generate_confusion_matrix, 
    save_prediction_samples, save_metrics, compare_experiments
)


def check_dataset(data_dir: str) -> bool:
    """Check if dataset exists and has expected structure."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        print("\nPlease download the dataset from Kaggle:")
        print("  1. Install Kaggle API: pip install kaggle")
        print("  2. Download: kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification")
        print("  3. Extract to: {data_dir}")
        return False
    
    # Check for train and test folders
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"[ERROR] Expected 'train' and 'test' folders in {data_dir}")
        return False
    
    # Count images
    total_train = sum(len(list((train_dir / cls).glob("*"))) 
                      for cls in os.listdir(train_dir) if (train_dir / cls).is_dir())
    total_test = sum(len(list((test_dir / cls).glob("*"))) 
                     for cls in os.listdir(test_dir) if (test_dir / cls).is_dir())
    
    print(f"[INFO] Dataset found:")
    print(f"  - Training images: {total_train}")
    print(f"  - Test images: {total_test}")
    
    return total_train > 0 and total_test > 0


def generate_segmentation_samples(data_dir: str, output_dir: str, method: str = "grabcut"):
    """Generate segmentation samples for visualization."""
    from src.utils import load_image, save_image
    
    print_section(f"Generating Segmentation Samples ({method})")
    
    samples = get_sample_images(data_dir, samples_per_class=2)
    
    for class_name, image_paths in samples.items():
        class_output = os.path.join(output_dir, class_name)
        os.makedirs(class_output, exist_ok=True)
        
        for img_path in image_paths[:2]:
            try:
                image = load_image(img_path)
                result = segment_image(image, method=method)
                
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                
                # Save original, mask, and segmented
                save_image(image, os.path.join(class_output, f"{name}_original{ext}"))
                save_image(result['segmented'], os.path.join(class_output, f"{name}_segmented{ext}"))
                
                # Save mask (grayscale)
                import cv2
                cv2.imwrite(os.path.join(class_output, f"{name}_mask.png"), result['mask'])
                
                print(f"  [OK] {class_name}/{filename}: {result['method_info']}")
                
            except Exception as e:
                print(f"  [FAIL] {img_path}: {e}")


def run_experiment(
    name: str,
    data_dir: str,
    model_dir: str,
    output_dir: str,
    use_segmentation: bool,
    segmentation_method: str = "grabcut",
    epochs: int = 15,
    batch_size: int = 32,
    device = None
):
    """Run a single training experiment."""
    print_section(f"Experiment: {name}")
    print(f"Segmentation: {'Enabled - ' + segmentation_method if use_segmentation else 'Disabled'}")
    
    # Create dataloaders
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,  # Windows compatibility
        use_segmentation=use_segmentation,
        segmentation_method=segmentation_method
    )
    
    # Create model
    model = create_model(
        model_name="mobilenetv2",
        num_classes=len(QUALITY_NAMES),
        pretrained=True,
        freeze_backbone=True
    )
    
    # Model save path
    model_suffix = "_segmented" if use_segmentation else "_baseline"
    model_path = os.path.join(model_dir, f"fruit_quality{model_suffix}.pth")
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        learning_rate=0.001,
        save_path=model_path,
        unfreeze_after=epochs // 3,
        device=device
    )
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    # Generate outputs
    exp_output = os.path.join(output_dir, name.lower().replace(" ", "_"))
    os.makedirs(exp_output, exist_ok=True)
    
    # Confusion matrix
    cm_path = os.path.join(exp_output, "confusion_matrix.png")
    generate_confusion_matrix(metrics, cm_path, title=f"Confusion Matrix - {name}")
    
    # Metrics JSON
    metrics_path = os.path.join(exp_output, "metrics.json")
    save_metrics(metrics, metrics_path)
    
    # Prediction samples
    samples_dir = os.path.join(exp_output, "predictions")
    save_prediction_samples(model, test_loader, samples_dir, num_samples=20, device=device)
    
    return metrics, history


def main():
    parser = argparse.ArgumentParser(description="Fruit Quality Classification Pipeline")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Path to outputs directory")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Path to save models")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--segmentation-method", type=str, default="grabcut",
                        choices=["grabcut", "hsv"],
                        help="Segmentation method to use")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (no segmentation) experiment")
    parser.add_argument("--skip-segmented", action="store_true",
                        help="Skip segmented experiment")
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick test with 2 epochs")
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test_mode:
        args.epochs = 2
        args.batch_size = 16
    
    print_section("Fruit Quality Classification Pipeline", char="═", width=70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Segmentation method: {args.segmentation_method}")
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create directories
    create_directories(PROJECT_ROOT)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check dataset
    if not check_dataset(args.data_dir):
        sys.exit(1)
    
    # Generate segmentation samples
    seg_samples_dir = os.path.join(args.output_dir, "segmentation_samples")
    generate_segmentation_samples(args.data_dir, seg_samples_dir, args.segmentation_method)
    
    # Run experiments
    results = {}
    
    if not args.skip_baseline:
        baseline_metrics, _ = run_experiment(
            name="Baseline",
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            use_segmentation=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        results['baseline'] = baseline_metrics
    
    if not args.skip_segmented:
        segmented_metrics, _ = run_experiment(
            name="Segmented",
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            use_segmentation=True,
            segmentation_method=args.segmentation_method,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        results['segmented'] = segmented_metrics
    
    # Compare experiments
    if 'baseline' in results and 'segmented' in results:
        comparison_path = os.path.join(args.output_dir, "experiment_comparison.txt")
        compare_experiments(results['baseline'], results['segmented'], comparison_path)
    
    # Final summary
    print_section("Pipeline Complete", char="═", width=70)
    print("Generated outputs:")
    print(f"  - Segmentation samples: {seg_samples_dir}")
    print(f"  - Models saved in: {args.model_dir}")
    print(f"  - Evaluation results: {args.output_dir}")
    print("\nTo run the Streamlit app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
