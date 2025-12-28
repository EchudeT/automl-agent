"""
Test script to verify AutoML-Agent framework can run.
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing AutoML-Agent Framework")
print("=" * 60)

# Test 1: Import core modules
print("\n[Test 1] Importing core modules...")
try:
    from configs import AVAILABLE_LLMs
    print("  [OK] configs imported")

    from utils import get_client, print_message
    print("  [OK] utils imported")

    print("\n[Test 2] Testing LLM client...")
    client = get_client("qwen")
    print("  [OK] LLM client created")

    print("\n[Test 3] Testing simple API call...")
    response = client.chat.completions.create(
        model=AVAILABLE_LLMs["qwen"]["model"],
        messages=[
            {"role": "user", "content": "Say 'AutoML-Agent is ready!' in one sentence."}
        ],
        max_tokens=30,
        temperature=0.7
    )
    result = response.choices[0].message.content.strip()
    print(f"  [OK] API Response: {result}")

    print("\n" + "=" * 60)
    print("[SUCCESS] AutoML-Agent framework is ready!")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
