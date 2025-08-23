#!/usr/bin/env python3
"""
Environment Setup Verification Script
Checks if your .env file has all required variables for CRM integration
"""

import os
from pathlib import Path


def verify_env_setup():
    """Verify .env file setup"""

    print("🔍 VERIFYING ENVIRONMENT SETUP")
    print("=" * 50)

    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found in current directory")
        print("💡 Create a .env file with your Monday.com credentials:")
        print("   MONDAY_API_KEY=your_api_key_here")
        print("   MONDAY_BOARD_ID=your_board_id_here")
        return False

    print("✅ .env file found")

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv('.env', override=True)
        print("✅ Loaded .env file successfully")
    except ImportError:
        print("❌ python-dotenv not installed")
        print("💡 Install with: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False

    # Check required variables
    required_vars = {
        'MONDAY_API_KEY': 'Monday.com API Token',
        'MONDAY_BOARD_ID': 'Monday.com Board ID',
        'MONDAY_API_URL': 'Monday.com API URL (optional)'
    }

    missing_vars = []
    found_vars = {}

    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if value:
            # Mask the API key for security
            if 'API_KEY' in var_name:
                display_value = f"{value[:15]}..." if len(value) > 15 else value[:8] + "..."
            else:
                display_value = value

            found_vars[var_name] = display_value
            print(f"✅ {var_name}: {display_value}")
        else:
            missing_vars.append(var_name)
            print(f"❌ {var_name}: Not found")

    # Check .env file content directly to show what's there
    print(f"\n📄 Contents of .env file:")
    try:
        with open('.env', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Mask sensitive values
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if 'API_KEY' in key:
                            masked_value = f"{value[:15]}..." if len(value) > 15 else value[:8] + "..."
                            print(f"   {i:2d}: {key}={masked_value}")
                        else:
                            print(f"   {i:2d}: {line}")
                    else:
                        print(f"   {i:2d}: {line}")
                elif line.startswith('#'):
                    print(f"   {i:2d}: {line}")
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")

    # Validate API key format
    api_key = os.getenv('MONDAY_API_KEY')
    if api_key:
        if len(api_key) < 20:
            print("⚠️  API key seems too short - check if it's complete")
        elif not api_key.replace('_', '').replace('-', '').isalnum():
            print("⚠️  API key contains unexpected characters")
        else:
            print("✅ API key format looks correct")

    # Validate Board ID format
    board_id = os.getenv('MONDAY_BOARD_ID')
    if board_id:
        if not board_id.isdigit():
            print("⚠️  Board ID should be numeric")
        else:
            print("✅ Board ID format looks correct")

    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Required variables: {len(required_vars)}")
    print(f"   Found: {len(found_vars)}")
    print(f"   Missing: {len(missing_vars)}")

    if missing_vars:
        print(f"\n❌ Missing variables: {', '.join(missing_vars)}")
        print(f"\n💡 Add these to your .env file:")
        for var in missing_vars:
            if var == 'MONDAY_API_KEY':
                print(f"   {var}=your_monday_api_token_here")
            elif var == 'MONDAY_BOARD_ID':
                print(f"   {var}=your_board_id_number_here")
            else:
                print(f"   {var}=value_here")
        return False
    else:
        print("✅ All required environment variables are set!")
        return True


def test_monday_connection():
    """Test actual connection to Monday.com"""
    print(f"\n🔗 TESTING MONDAY.COM CONNECTION")
    print("=" * 50)

    try:
        # Import and test the CRM service
        from services.crm_service import CRMService

        crm = CRMService()
        print("✅ CRM Service initialized")

        # Test connection
        result = crm.test_connection()

        if result['success']:
            user = result.get('user', {})
            print(f"✅ Connected successfully!")
            print(f"   User: {user.get('name', 'Unknown')}")
            print(f"   Email: {user.get('email', 'Unknown')}")
            return True
        else:
            print(f"❌ Connection failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False


if __name__ == "__main__":
    print("🔧 MONDAY.COM ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # Step 1: Verify .env setup
    env_ok = verify_env_setup()

    if env_ok:
        # Step 2: Test Monday.com connection
        connection_ok = test_monday_connection()

        if connection_ok:
            print(f"\n🎉 SETUP COMPLETE!")
            print("✅ Environment variables are properly configured")
            print("✅ Monday.com connection is working")
            print("\n💡 You're ready to run CRM imports!")
            print("   Next: Run python test_crm_fixes.py to test data processing")
        else:
            print(f"\n⚠️  SETUP INCOMPLETE")
            print("✅ Environment variables are set")
            print("❌ Monday.com connection failed")
            print("\n💡 Check your API key and board ID in Monday.com")
    else:
        print(f"\n❌ SETUP FAILED")
        print("Fix the .env file issues above and try again")