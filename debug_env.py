# debug_env.py
# Run this script to debug where the API key is coming from

import os
from dotenv import load_dotenv
import sys


def debug_environment_loading():
    """Debug environment variable loading to find where old key comes from"""

    print("🔍 DEBUGGING ENVIRONMENT VARIABLE LOADING")
    print("=" * 60)

    # Check current working directory
    print(f"📁 Current working directory: {os.getcwd()}")

    # Check if .env file exists in current directory
    env_files = ['.env', '.env.local', '.env.development', '.env.production']

    print("\n📄 Checking for .env files:")
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"✅ Found: {env_file}")

            # Read and show the API key from this file
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith('ANTHROPIC_API_KEY'):
                            key_value = line.split('=', 1)[1] if '=' in line else 'INVALID'
                            print(f"   💡 Contains: ANTHROPIC_API_KEY={key_value[:20]}...")
            except Exception as e:
                print(f"   ❌ Error reading {env_file}: {e}")
        else:
            print(f"❌ Not found: {env_file}")

    print("\n🔧 Checking environment variables BEFORE loading .env:")
    old_key = os.environ.get('ANTHROPIC_API_KEY')
    if old_key:
        print(f"⚠️ ANTHROPIC_API_KEY already in environment: {old_key[:20]}...")
        print("   This might be from a previous run or system environment")
    else:
        print("✅ ANTHROPIC_API_KEY not in environment yet")

    print("\n📥 Loading .env file...")
    # Force reload of .env file
    load_dotenv(override=True)  # This will override existing env vars

    print("\n🔧 Checking environment variables AFTER loading .env:")
    new_key = os.getenv('ANTHROPIC_API_KEY')
    if new_key:
        print(f"✅ ANTHROPIC_API_KEY loaded: {new_key[:20]}...")

        # Check if it changed
        if old_key and old_key != new_key:
            print("🔄 API key was updated from .env file")
        elif old_key and old_key == new_key:
            print("⚠️ API key unchanged - might be cached")
        else:
            print("✅ API key loaded fresh from .env file")

    else:
        print("❌ ANTHROPIC_API_KEY still not found after loading .env")

    print("\n🧪 Testing different loading methods:")

    # Method 1: Load with override
    print("Method 1: load_dotenv(override=True)")
    load_dotenv(override=True)
    key1 = os.getenv('ANTHROPIC_API_KEY')
    print(f"   Result: {key1[:20] if key1 else 'None'}...")

    # Method 2: Load specific file
    print("Method 2: load_dotenv('.env', override=True)")
    load_dotenv('.env', override=True)
    key2 = os.getenv('ANTHROPIC_API_KEY')
    print(f"   Result: {key2[:20] if key2 else 'None'}...")

    # Method 3: Read file manually
    print("Method 3: Manual file reading")
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip().startswith('ANTHROPIC_API_KEY'):
                    manual_key = line.split('=', 1)[1].strip()
                    print(f"   Result: {manual_key[:20]}...")
                    break
            else:
                print("   Result: Not found in .env file")
    except FileNotFoundError:
        print("   Result: .env file not found")
    except Exception as e:
        print(f"   Result: Error reading file: {e}")

    print("\n🌍 All environment variables containing 'ANTHROPIC':")
    for key, value in os.environ.items():
        if 'ANTHROPIC' in key.upper():
            print(f"   {key}: {value[:20]}...")

    print("\n💡 RECOMMENDATIONS:")
    if old_key and new_key and old_key != new_key:
        print("✅ The .env file has been loaded with new key")
        print("   Restart your Flask application to use the new key")
    elif old_key and old_key == new_key:
        print("⚠️ The key in environment matches .env file")
        print("   If you recently changed .env, restart your application")
    elif not new_key:
        print("❌ No API key found in .env file")
        print("   Check that .env file contains: ANTHROPIC_API_KEY=sk-ant-...")
    else:
        print("✅ API key loaded successfully from .env file")


if __name__ == "__main__":
    debug_environment_loading()