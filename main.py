"""
Main Entry Point for Road Damage Detection Project
===================================================

This script serves as the central entry point for:
- Running preprocessing demos
- Training the model
- Evaluating performance
- Running inference
- Launching the Flask web app

Usage:
    python main.py --mode preprocess    # Run preprocessing demo
    python main.py --mode train         # Train the model
    python main.py --mode evaluate      # Evaluate on test set
    python main.py --mode inference     # Run inference on images
    python main.py --mode app           # Launch Flask web app
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import config


def run_preprocessing_demo():
    """Run the complete preprocessing pipeline demo."""
    print("\n" + "=" * 70)
    print(" 🔧 RUNNING PREPROCESSING DEMO")
    print("=" * 70 + "\n")
    
    from preprocessing.preprocessing_demo import main as demo_main
    demo_main()


def run_training():
    """Run model training."""
    print("\n" + "=" * 70)
    print(" 🚀 STARTING MODEL TRAINING")
    print("=" * 70 + "\n")
    
    from training.train import main as train_main
    train_main()


def run_evaluation():
    """Run model evaluation."""
    print("\n" + "=" * 70)
    print(" 📊 RUNNING EVALUATION")
    print("=" * 70 + "\n")
    
    from evaluation.evaluate import main as eval_main
    eval_main()


def run_inference():
    """Run inference on test images."""
    print("\n" + "=" * 70)
    print(" 🔍 RUNNING INFERENCE")
    print("=" * 70 + "\n")
    
    from inference.run_inference import main as inference_main
    inference_main()


def run_app():
    """Launch Flask web application."""
    print("\n" + "=" * 70)
    print(" 🌐 LAUNCHING FLASK WEB APP")
    print("=" * 70 + "\n")
    
    from app.routes import app
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)


def main():
    """Parse arguments and execute selected mode."""
    parser = argparse.ArgumentParser(
        description="Road Damage Detection - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode preprocess
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode inference
  python main.py --mode app
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train", "evaluate", "inference", "app"],
        default="preprocess",
        help="Execution mode (default: preprocess)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    config.print_config()
    
    # Execute selected mode
    mode_functions = {
        "preprocess": run_preprocessing_demo,
        "train": run_training,
        "evaluate": run_evaluation,
        "inference": run_inference,
        "app": run_app,
    }
    
    try:
        mode_functions[args.mode]()
    except KeyError:
        print(f"❌ Unknown mode: {args.mode}")
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error in {args.mode} mode: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

