"""
Test script to verify LLM client configuration.
Usage: python test_llm_config.py
"""

import sys
sys.path.insert(0, '.')

from utils import get_client, print_message
from configs import AVAILABLE_LLMs

def test_llm_config():
    print_message("system", "Testing LLM Configuration")

    # Print available LLMs
    print_message("system", f"Available LLMs: {list(AVAILABLE_LLMs.keys())}")

    # Test default client
    try:
        client = get_client("qwen")
        print_message("system", f"✓ Successfully created client for 'qwen'")
        print_message("system", f"  Base URL: {AVAILABLE_LLMs['qwen'].get('base_url', 'default OpenAI')}")
        print_message("system", f"  Model: {AVAILABLE_LLMs['qwen']['model']}")
    except Exception as e:
        print_message("system", f"✗ Failed to create client: {e}")

    # Test gemini client
    try:
        client = get_client("gemini")
        print_message("system", f"✓ Successfully created client for 'gemini'")
        print_message("system", f"  Base URL: {AVAILABLE_LLMs['gemini']['base_url']}")
        print_message("system", f"  Model: {AVAILABLE_LLMs['gemini']['model']}")
    except Exception as e:
        print_message("system", f"✗ Failed to create gemini client: {e}")

    print_message("system", "Configuration test complete!")

if __name__ == "__main__":
    test_llm_config()
