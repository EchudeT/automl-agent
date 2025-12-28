"""
ActionFormer Training Wrapper Script

This script wraps the ActionFormer training process, allowing AutoML-Agent to
train ActionFormer models without writing complex model code directly.

Usage:
    python train_actionformer_wrapper.py \
        --config_template thumos_i3d \
        --data_path ./data/thumos \
        --learning_rate 0.0001 \
        --epochs 30 \
        --batch_size 2 \
        --output_name automl_run

Author: AutoML-Agent
"""

import argparse
import os
import sys
import subprocess
import re
import yaml
import tempfile
import shutil
from pathlib import Path

# Add ActionFormer library to path
SCRIPT_DIR = Path(__file__).parent.absolute()
ACTIONFORMER_DIR = SCRIPT_DIR.parent / "actionformer_release"

if ACTIONFORMER_DIR.exists():
    sys.path.insert(0, str(ACTIONFORMER_DIR))
    print(f"[Wrapper] Added ActionFormer to path: {ACTIONFORMER_DIR}")
else:
    print(f"[Wrapper] WARNING: ActionFormer directory not found at {ACTIONFORMER_DIR}")
    print(f"[Wrapper] Expected structure: automl-agent/ and actionformer_release/ in same parent directory")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ActionFormer Training Wrapper')

    parser.add_argument('--config_template', type=str, default='thumos_i3d',
                        help='Config template name (e.g., thumos_i3d, anet_i3d)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for optimizer')
    parser.add_argument('--output_name', type=str, default='automl_run',
                        help='Output folder name for checkpoints')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU devices to use (e.g., "0" or "0,1")')

    return parser.parse_args()


def load_and_modify_config(template_name, args):
    """
    Load config template and modify hyperparameters based on args.

    Args:
        template_name: Name of config template (without .yaml extension)
        args: Parsed command line arguments

    Returns:
        Path to modified config file
    """
    # Locate config template
    config_path = ACTIONFORMER_DIR / "configs" / f"{template_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config template not found: {config_path}")

    print(f"[Wrapper] Loading config template: {config_path}")

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify hyperparameters
    print(f"[Wrapper] Modifying hyperparameters:")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  weight_decay: {args.weight_decay}")

    if 'opt' not in config:
        config['opt'] = {}
    config['opt']['learning_rate'] = args.learning_rate
    config['opt']['epochs'] = args.epochs
    config['opt']['weight_decay'] = args.weight_decay

    if 'loader' not in config:
        config['loader'] = {}
    config['loader']['batch_size'] = args.batch_size

    # Update data path if specified
    if args.data_path and 'dataset' in config:
        print(f"  data_path: {args.data_path}")
        # This is dataset-specific, may need adjustment
        if 'feat_folder' in config['dataset']:
            config['dataset']['feat_folder'] = os.path.join(args.data_path, 'features')
        if 'json_file' in config['dataset']:
            config['dataset']['json_file'] = os.path.join(args.data_path, 'annotations.json')

    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.close()

    print(f"[Wrapper] Created temporary config: {temp_config.name}")
    return temp_config.name


def parse_training_output(output_lines):
    """
    Parse training output to extract mAP results.

    Args:
        output_lines: List of output lines from training

    Returns:
        Final mAP value or None if not found
    """
    map_pattern = re.compile(r'(mAP|average mAP).*?(\d+\.\d+)', re.IGNORECASE)

    final_map = None
    for line in output_lines:
        match = map_pattern.search(line)
        if match:
            final_map = float(match.group(2))
            print(f"[Wrapper] Found mAP: {final_map:.4f}")

    return final_map


def run_training(config_path, args):
    """
    Execute ActionFormer training script.

    Args:
        config_path: Path to (modified) config file
        args: Command line arguments

    Returns:
        Tuple of (return_code, final_mAP)
    """
    # Prepare training command
    train_script = ACTIONFORMER_DIR / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(train_script),
        config_path,
        '--output', args.output_name
    ]

    if args.resume:
        cmd.extend(['--resume', args.resume])

    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.devices

    print(f"[Wrapper] Starting training...")
    print(f"[Wrapper] Command: {' '.join(cmd)}")
    print(f"[Wrapper] Working directory: {ACTIONFORMER_DIR}")
    print(f"[Wrapper] GPU devices: {args.devices}")
    print("-" * 80)

    # Run training and capture output
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ACTIONFORMER_DIR),
            env=env
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print in real-time
            output_lines.append(line)

        process.wait()
        return_code = process.returncode

        print("-" * 80)
        print(f"[Wrapper] Training completed with return code: {return_code}")

        # Parse results
        final_map = parse_training_output(output_lines)

        return return_code, final_map

    except Exception as e:
        print(f"[Wrapper] ERROR during training: {e}")
        return -1, None


def main():
    """Main wrapper function."""
    print("=" * 80)
    print("ActionFormer Training Wrapper for AutoML-Agent")
    print("=" * 80)

    args = parse_args()

    try:
        # Step 1: Load and modify config
        config_path = load_and_modify_config(args.config_template, args)

        # Step 2: Run training
        return_code, final_map = run_training(config_path, args)

        # Step 3: Cleanup temporary config
        try:
            os.unlink(config_path)
            print(f"[Wrapper] Cleaned up temporary config")
        except:
            pass

        # Step 4: Print final results for Agent to parse
        print("=" * 80)
        print("[Wrapper] TRAINING SUMMARY")
        print("=" * 80)
        if final_map is not None:
            print(f"FINAL_mAP: {final_map:.4f}")
        else:
            print("FINAL_mAP: N/A (not found in output)")
        print(f"Return Code: {return_code}")
        print("=" * 80)

        sys.exit(return_code)

    except Exception as e:
        print(f"[Wrapper] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
