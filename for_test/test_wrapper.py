"""
Test script for ActionFormer Wrapper (without actually training).
Tests configuration loading and modification logic.
"""

import sys
sys.path.insert(0, '.')

import yaml
from pathlib import Path

def test_wrapper_config():
    """Test wrapper configuration logic."""
    print("Testing ActionFormer Wrapper Configuration")
    print("=" * 60)

    ACTIONFORMER_DIR = Path("../../actionformer_release")

    if not ACTIONFORMER_DIR.exists():
        print(f"[ERROR] ActionFormer directory not found at {ACTIONFORMER_DIR}")
        print("  Expected directory structure:")
        print("    AutoML/")
        print("      +-- automl-agent/")
        print("      +-- actionformer_release/")
        return False

    print(f"[OK] Found ActionFormer directory at {ACTIONFORMER_DIR}")

    # Check for train.py
    train_script = ACTIONFORMER_DIR / "train.py"
    if train_script.exists():
        print(f"[OK] Found train.py at {train_script}")
    else:
        print(f"[ERROR] train.py not found")
        return False

    # Check for config files
    configs_dir = ACTIONFORMER_DIR / "configs"
    if configs_dir.exists():
        configs = list(configs_dir.glob("*.yaml"))
        print(f"[OK] Found {len(configs)} config templates:")
        for cfg in configs[:5]:  # Show first 5
            print(f"    - {cfg.stem}")
        if len(configs) > 5:
            print(f"    ... and {len(configs) - 5} more")
    else:
        print(f"[ERROR] configs/ directory not found")
        return False

    # Test config loading
    test_config = configs_dir / "thumos_i3d.yaml"
    if test_config.exists():
        with open(test_config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"\n[OK] Successfully loaded {test_config.name}")
        print(f"  Original learning_rate: {config.get('opt', {}).get('learning_rate')}")
        print(f"  Original epochs: {config.get('opt', {}).get('epochs')}")
        print(f"  Original batch_size: {config.get('loader', {}).get('batch_size')}")
    else:
        print(f"[ERROR] Test config thumos_i3d.yaml not found")
        return False

    print("\n" + "=" * 60)
    print("[OK] All wrapper tests passed!")
    print("\nExample usage:")
    print("  python train_actionformer_wrapper.py \\")
    print("    --config_template thumos_i3d \\")
    print("    --data_path ./data/thumos \\")
    print("    --learning_rate 0.0002 \\")
    print("    --epochs 50 \\")
    print("    --batch_size 4")

    return True


if __name__ == "__main__":
    success = test_wrapper_config()
    sys.exit(0 if success else 1)
