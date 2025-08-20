#!/usr/bin/env python3
"""
test_monday_api_call.py - Test actual Monday.com API calls

Usage:
    python test_monday_api_call.py [test_type]

Test types:
    connection - Test basic API connection
    create_group - Test creating a group
    create_item - Test creating an item with sample data
    full - Test complete workflow (connection, group, item)

This will help debug actual API communication issues.
"""

import sys
import json
import requests
import os
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed")


def get_api_config():
    """Get API configuration from environment"""
    api_key = os.getenv('MONDAY_API_KEY')
    board_id = os.getenv('MONDAY_BOARD_ID')

    if not api_key:
        print("❌ MONDAY_API_KEY not found in environment variables")
        return None, None, None

    if not board_id:
        print("❌ MONDAY_BOARD_ID not found in environment variables")
        return None, None, None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "API-Version": "2023-10"
    }

    return api_key, board_id, headers


def test_connection():
    """Test basic Monday.com API connection"""
    print("🔗 Testing Monday.com API Connection...")

    api_key, board_id, headers = get_api_config()
    if not api_key:
        return False

    # Simple query to get user info
    query = {
        "query": "query { me { name email } }"
    }

    try:
        response = requests.post(
            "https://api.monday.com/v2",
            json=query,
            headers=headers,
            timeout=10
        )

        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response Data: {json.dumps(data, indent=2)}")

            if 'data' in data and 'me' in data['data']:
                user = data['data']['me']
                print(f"✅ Connection successful!")
                print(f"   User: {user.get('name', 'Unknown')}")
                print(f"   Email: {user.get('email', 'Unknown')}")
                return True
            else:
                print(f"❌ Unexpected response format")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False


def test_create_group():
    """Test creating a group in Monday.com"""
    print("📁 Testing Group Creation...")

    api_key, board_id, headers = get_api_config()
    if not api_key:
        return None

    group_name = f"Test Group - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    mutation = {
        "query": """
        mutation ($boardId: ID!, $groupName: String!) {
            create_group(board_id: $boardId, group_name: $groupName) {
                id
                title
            }
        }
        """,
        "variables": {
            "boardId": board_id,
            "groupName": group_name
        }
    }

    try:
        response = requests.post(
            "https://api.monday.com/v2",
            json=mutation,
            headers=headers,
            timeout=10
        )

        print(f"Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response Data: {json.dumps(data, indent=2)}")

            if 'data' in data and 'create_group' in data['data'] and data['data']['create_group']:
                group = data['data']['create_group']
                group_id = group['id']
                print(f"✅ Group created successfully!")
                print(f"   Group ID: {group_id}")
                print(f"   Group Name: {group['title']}")
                return group_id
            else:
                print(f"❌ Group creation failed")
                if 'errors' in data:
                    print(f"   Errors: {data['errors']}")
                return None
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None


def test_create_item(group_id=None):
    """Test creating an item in Monday.com"""
    print("📋 Testing Item Creation...")

    api_key, board_id, headers = get_api_config()
    if not api_key:
        return False

    # Create a test group if none provided
    if not group_id:
        group_id = test_create_group()
        if not group_id:
            print("❌ Cannot test item creation without a group")
            return False

    # Sample item data based on your CSV analysis
    item_name = "TEST PROPERTY OWNER"
    column_values = {
        "text117": "TEST123456",  # parcel_id
        "numbers6": 100,  # acreage
        "text4": "Test County",  # county_name
        "text_1": "OH",  # state_abbr
        "text1": "123 Test Street",  # address
        "latitude__1": "41.4032",  # latitude
        "longitude__1": "-81.8607",  # longitude
        "numbers85__1": 50000.0,  # land_value
        "text7": "123 Mail St",  # mail_address
        "text49": "Test City",  # mail_city
        "text11": "OH",  # mail_state
        "mzip": "44123"  # mail_zip
    }

    print(f"Creating item with:")
    print(f"  Item Name: {item_name}")
    print(f"  Group ID: {group_id}")
    print(f"  Column Values: {json.dumps(column_values, indent=2)}")

    mutation = {
        "query": """
            mutation ($boardId: ID!, $groupId: String!, $itemName: String!, $columnValues: JSON!) {
                create_item (
                    board_id: $boardId,
                    group_id: $groupId,
                    item_name: $itemName,
                    column_values: $columnValues
                ) {
                    id
                    name
                }
            }
        """,
        "variables": {
            "boardId": board_id,
            "groupId": group_id,
            "itemName": item_name,
            "columnValues": json.dumps(column_values)
        }
    }

    try:
        response = requests.post(
            "https://api.monday.com/v2",
            json=mutation,
            headers=headers,
            timeout=10
        )

        print(f"Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response Data: {json.dumps(data, indent=2)}")

            if 'data' in data and 'create_item' in data['data'] and data['data']['create_item']:
                item = data['data']['create_item']
                print(f"✅ Item created successfully!")
                print(f"   Item ID: {item['id']}")
                print(f"   Item Name: {item['name']}")
                return True
            else:
                print(f"❌ Item creation failed")
                if 'errors' in data:
                    print(f"   Errors: {data['errors']}")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False


def test_board_structure():
    """Test getting board structure to verify field IDs"""
    print("🏗️  Testing Board Structure...")

    api_key, board_id, headers = get_api_config()
    if not api_key:
        return False

    query = {
        "query": f"""
        query {{
            boards(ids: [{board_id}]) {{
                name
                columns {{
                    id
                    title
                    type
                    settings_str
                }}
                groups {{
                    id
                    title
                }}
            }}
        }}
        """
    }

    try:
        response = requests.post(
            "https://api.monday.com/v2",
            json=query,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            if 'data' in data and 'boards' in data['data'] and data['data']['boards']:
                board = data['data']['boards'][0]
                print(f"✅ Board found: {board['name']}")

                print(f"\n📋 Available Columns:")
                for col in board['columns']:
                    print(f"   {col['id']:<25} | {col['title']:<30} | {col['type']}")

                print(f"\n📁 Available Groups:")
                for group in board['groups']:
                    print(f"   {group['id']:<25} | {group['title']}")

                return True
            else:
                print(f"❌ Board not found or no access")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False


def main():
    """Main function"""
    test_type = sys.argv[1] if len(sys.argv) > 1 else "full"

    print(f"🧪 Monday.com API Test Suite")
    print("=" * 60)

    if test_type == "connection":
        success = test_connection()
    elif test_type == "create_group":
        group_id = test_create_group()
        success = group_id is not None
    elif test_type == "create_item":
        success = test_create_item()
    elif test_type == "board":
        success = test_board_structure()
    elif test_type == "full":
        print("Running full test suite...\n")

        # Test 1: Connection
        if not test_connection():
            print("❌ Connection test failed. Cannot continue.")
            sys.exit(1)

        print("\n" + "-" * 40 + "\n")

        # Test 2: Board structure
        if not test_board_structure():
            print("⚠️  Board structure test failed. Continuing anyway.")

        print("\n" + "-" * 40 + "\n")

        # Test 3: Create group
        group_id = test_create_group()
        if not group_id:
            print("❌ Group creation failed. Cannot test item creation.")
            sys.exit(1)

        print("\n" + "-" * 40 + "\n")

        # Test 4: Create item
        success = test_create_item(group_id)

        if success:
            print(f"\n🎉 All tests passed! Your Monday.com integration is working.")
        else:
            print(f"\n❌ Some tests failed. Check the output above for details.")
    else:
        print(f"❌ Unknown test type: {test_type}")
        print("Available tests: connection, create_group, create_item, board, full")
        sys.exit(1)

    if success:
        print(f"\n✅ Test '{test_type}' completed successfully!")
    else:
        print(f"\n❌ Test '{test_type}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()