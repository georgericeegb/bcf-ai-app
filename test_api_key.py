# test_api_key.py
# Run this script to test your Anthropic API key

import os
from dotenv import load_dotenv
import requests


def test_anthropic_api_key():
    """Test if Anthropic API key is working"""

    print("🔑 Testing Anthropic API Key Configuration")
    print("=" * 50)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')

    # Check if API key exists
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("\nTroubleshooting steps:")
        print("1. Check if .env file exists in your project directory")
        print("2. Make sure .env file contains: ANTHROPIC_API_KEY=sk-ant-...")
        print("3. Restart your application after updating .env")
        return False

    print(f"✅ API Key found: {api_key[:15]}...")
    print(f"📏 API Key length: {len(api_key)} characters")

    # Check API key format
    if not api_key.startswith('sk-ant-'):
        print("❌ Invalid API key format")
        print("   Anthropic API keys should start with 'sk-ant-'")
        print("   Get a valid key from: https://console.anthropic.com/")
        return False

    print("✅ API Key format looks correct")

    # Test API connection
    print("\n🌐 Testing API connection...")

    try:
        # Simple test request to Anthropic API
        headers = {
            'x-api-key': api_key,
            'content-type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': 'claude-3-haiku-20240307',
            'max_tokens': 10,
            'messages': [
                {
                    'role': 'user',
                    'content': 'Say "Hello"'
                }
            ]
        }

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            print("✅ API connection successful!")
            print("🎉 Your API key is working correctly")
            return True
        elif response.status_code == 401:
            print("❌ API authentication failed")
            print("   Your API key is invalid or expired")
            print("   Get a new key from: https://console.anthropic.com/")
            return False
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ API request timed out")
        print("   Check your internet connection")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def create_sample_env_file():
    """Create a sample .env file"""

    sample_content = """# ReportAll API
RAUSA_CLIENT_KEY=your_reportall_key
RAUSA_API_URL=https://reportallusa.com/api/parcels
RAUSA_API_VERSION=9

# Google Cloud Storage  
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
CACHE_BUCKET_NAME=your-bucket-name

# Anthropic API - Replace with your actual API key
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_ACTUAL_API_KEY_HERE

# Flask
SECRET_KEY=your-secret-key
FLASK_ENV=development
"""

    if not os.path.exists('.env'):
        with open('.env.example', 'w') as f:
            f.write(sample_content)
        print("📝 Created .env.example file")
        print("   Copy this to .env and add your actual API key")
    else:
        print("✅ .env file already exists")


if __name__ == "__main__":
    print("🚀 Anthropic API Key Tester")
    print("This script will help you diagnose API key issues\n")

    # Check .env file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found")
        create_sample_env_file()
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your actual Anthropic API key")
        print("3. Run this script again")
    else:
        success = test_anthropic_api_key()

        if success:
            print("\n🎯 Next steps:")
            print("1. Your API key is working correctly")
            print("2. Restart your Flask application")
            print("3. Try the focus area analysis again")
        else:
            print("\n🔧 Next steps:")
            print("1. Get a valid API key from https://console.anthropic.com/")
            print("2. Update your .env file with the new key")
            print("3. Run this script again to verify")
            print("4. Restart your Flask application")