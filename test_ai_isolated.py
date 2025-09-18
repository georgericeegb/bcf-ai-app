#!/usr/bin/env python3
import sys
import os
import traceback

print("=== AI SERVICE ISOLATION TEST ===")

# Test 1: Basic imports
try:
    import anthropic

    print("✅ anthropic package imported successfully")
    print(f"   anthropic version: {anthropic.__version__}")
except Exception as e:
    print(f"❌ Failed to import anthropic: {e}")
    sys.exit(1)

# Test 2: Environment variable
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found")
    sys.exit(1)
else:
    print(f"✅ ANTHROPIC_API_KEY found (length: {len(api_key)})")

# Test 3: AI Service import
try:
    sys.path.append('/app')  # Cloud Run working directory
    from services.ai_service import AIAnalysisService

    print("✅ AIAnalysisService imported successfully")
except Exception as e:
    print(f"❌ Failed to import AIAnalysisService: {e}")
    print(f"   Traceback: {traceback.format_exc()}")
    sys.exit(1)

# Test 4: AI Service initialization
try:
    ai_service = AIAnalysisService(api_key=api_key)
    print("✅ AIAnalysisService initialized")

    if hasattr(ai_service, 'client') and ai_service.client:
        print("✅ AI client exists")
    else:
        print("❌ AI client is None")

except Exception as e:
    print(f"❌ Failed to initialize AIAnalysisService: {e}")
    print(f"   Traceback: {traceback.format_exc()}")

# Test 5: Simple API call
try:
    if ai_service.client:
        response = ai_service.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        print("✅ API call successful")
    else:
        print("❌ Cannot test API call - client is None")
except Exception as e:
    print(f"❌ API call failed: {e}")

print("=== TEST COMPLETE ===")