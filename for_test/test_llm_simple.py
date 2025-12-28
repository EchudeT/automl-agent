"""
Simple test script to verify LLM client configuration.
Usage: python test_llm_simple.py
"""

import sys
import os
sys.path.insert(0, '.')

from openai import OpenAI

# Test configuration
def test_llm_simple():
    print("=" * 60)
    print("Testing LLM Configuration")
    print("=" * 60)

    # Load config
    from configs import AVAILABLE_LLMs

    print(f"\n[OK] Available LLMs: {list(AVAILABLE_LLMs.keys())}")

    # Test qwen/gemini client
    llm_name = "qwen"
    config = AVAILABLE_LLMs[llm_name]

    print(f"\n[OK] Testing '{llm_name}' configuration:")
    print(f"  - Model: {config['model']}")
    print(f"  - Base URL: {config['base_url']}")
    print(f"  - API Key: {config['api_key'][:20]}...")

    try:
        # Create client
        client = OpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        print(f"\n[OK] Successfully created OpenAI client")

        # Test API call with different models
        print(f"\n[OK] Testing API call...")

        # Try multiple model names
        models_to_try = [
            config['model'],
            'gemini-3-flash-preview',
            'gemini/gemini-2.0-flash-exp',
            'gpt-3.5-turbo'
        ]

        success = False
        for model_name in models_to_try:
            try:
                print(f"  Trying model: {model_name}...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": "Say 'Hello, AutoML-Agent!' in one sentence."}
                    ],
                    max_tokens=50,
                    temperature=0.7
                )
                result = response.choices[0].message.content.strip()
                print(f"\n[OK] API Response: {result}")
                print(f"[OK] Working model: {model_name}")
                success = True
                break
            except Exception as e:
                print(f"  Failed with {model_name}: {str(e)[:100]}")
                continue

        if success:
            print(f"\n{'=' * 60}")
            print("[SUCCESS] All tests passed! LLM configuration is working correctly.")
            print("=" * 60)
        else:
            raise Exception("All model attempts failed")

    except Exception as e:
        print(f"\n[ERROR] Error during API call: {e}")
        print(f"\n{'=' * 60}")
        print("[FAILED] Test failed. Please check your API configuration.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    test_llm_simple()
