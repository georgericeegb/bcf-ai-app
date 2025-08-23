# field_type_tester.py - Test individual Monday.com field types

import json
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
current_dir = Path.cwd()
for i in range(5):
    env_path = current_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break
    current_dir = current_dir.parent


class FieldTypeTester:
    def __init__(self):
        self.api_url = os.getenv('MONDAY_API_URL', 'https://api.monday.com/v2')
        self.api_key = os.getenv('MONDAY_API_KEY')
        self.board_id = os.getenv('MONDAY_BOARD_ID')

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "API-Version": "2023-10"
        }

    def test_field_value_formats(self):
        """Test different value formats for Monday.com fields"""

        print("🧪 FIELD VALUE FORMAT TESTER")
        print("=" * 50)

        # Define test cases for different field types
        test_cases = {
            # Text fields
            'text1': [
                ("Simple text", "5300 RIVERSIDE Rd"),
                ("Text with spaces", "  5300 RIVERSIDE Rd  "),
                ("Empty string", ""),
                ("None value", None),
            ],

            # Number fields
            'numbers6': [
                ("Integer", 526),
                ("Float", 526.33),
                ("String number", "526"),
                ("String float", "526.33"),
                ("Zero", 0),
                ("None", None),
                ("Invalid", "not_a_number"),
            ],

            # Coordinate fields
            'latitude__1': [
                ("Float coordinate", 41.4032357967894),
                ("String coordinate", "41.4032357967894"),
                ("Zero coordinate", 0.0),
                ("String zero", "0"),
                ("None", None),
            ],

            # County ID (special text field)
            'county_id__1': [
                ("5-digit FIPS", "39035"),
                ("Integer FIPS", 39035),
                ("Padded FIPS", "39035"),
                ("Short FIPS", "123"),
                ("None", None),
            ],

            # ZIP code field
            'text_mktw4254': [
                ("5-digit ZIP", "44142"),
                ("Integer ZIP", 44142),
                ("Zero ZIP", 0),
                ("String zero ZIP", "0"),
                ("None", None),
            ]
        }

        print("📋 Testing field value formats:")
        print()

        for field_id, test_values in test_cases.items():
            print(f"🔧 Field: {field_id}")

            for test_name, test_value in test_values:
                # Format the value like our CRM service would
                formatted_value = self.format_value_for_field(field_id, test_value)

                # Check JSON serialization
                try:
                    json_str = json.dumps({field_id: formatted_value})
                    json_valid = "✅"
                except Exception as e:
                    json_str = f"ERROR: {e}"
                    json_valid = "❌"

                print(f"   {test_name:20} | {repr(test_value):20} -> {repr(formatted_value):20} | JSON: {json_valid}")

            print()

    def format_value_for_field(self, field_id, value):
        """Format value according to our CRM service logic"""

        # Use the same validation as CRMService
        if value is None:
            return None

        str_val = str(value).strip().lower()
        if str_val in ['', 'null', 'none', 'nan', 'n/a', '0']:
            return None

        # Format based on field type
        if field_id in ['latitude__1', 'longitude__1']:
            try:
                coord_value = float(value)
                if coord_value == 0:
                    return None
                return str(coord_value)
            except:
                return None

        elif field_id == 'county_id__1':
            try:
                return str(int(float(value))).zfill(5)
            except:
                return str(value).zfill(5) if len(str(value)) <= 5 else str(value)[:5]

        elif field_id in ['numbers6', 'dup__of_score__0___3___1']:
            try:
                acreage = float(value)
                if acreage <= 0:
                    return None
                return int(acreage)
            except:
                return None

        elif field_id in ['text_mktw4254', 'mzip']:
            try:
                if str(value).strip() == '0':
                    return '00000'
                zip_value = str(int(float(value)))
                return zip_value.zfill(5)
            except:
                zip_str = str(value).strip()
                return zip_str[:10] if len(zip_str) > 0 else None

        else:
            # Default text handling
            cleaned = str(value).strip()
            return cleaned[:255] if len(cleaned) > 0 else None

    def create_minimal_test_item(self):
        """Create a test item with minimal data to verify basic functionality"""

        print("🧪 CREATING MINIMAL TEST ITEM")
        print("=" * 50)

        # Create test group first
        group_name = "MINIMAL_TEST_GROUP"
        group_id = self.create_test_group(group_name)

        if not group_id:
            print("❌ Failed to create test group")
            return None

        print(f"✅ Created test group: {group_id}")

        # Test with just one field at a time
        test_fields = {
            'text1': '5300 RIVERSIDE Rd',
            'numbers6': 526,
            'county_id__1': '39035',
            'text_mktw4254': '44142',
            'latitude__1': '41.4032',
            'longitude__1': '-81.8607'
        }

        for field_id, test_value in test_fields.items():
            print(f"\n🔧 Testing single field: {field_id} = {repr(test_value)}")

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
                        column_values {
                            id
                            value
                            text
                        }
                    }
                }
                """,
                "variables": {
                    "boardId": self.board_id,
                    "groupId": group_id,
                    "itemName": f"Test_{field_id}",
                    "columnValues": json.dumps({field_id: test_value})
                }
            }

            try:
                response = requests.post(self.api_url, json=mutation, headers=self.headers)
                result = response.json()

                if response.status_code == 200 and 'data' in result:
                    item = result['data']['create_item']
                    print(f"   ✅ Item created: {item['id']}")

                    # Check if the field value was actually set
                    for col_val in item.get('column_values', []):
                        if col_val['id'] == field_id:
                            if col_val.get('value') or col_val.get('text'):
                                print(f"   ✅ Field populated: {col_val.get('text') or col_val.get('value')}")
                            else:
                                print(f"   ⚠️  Field created but empty")
                            break
                    else:
                        print(f"   ❌ Field not found in response")

                elif 'errors' in result:
                    print(f"   ❌ API Error:")
                    for error in result['errors']:
                        print(f"      {error.get('message', 'Unknown error')}")

                else:
                    print(f"   ❌ Unexpected response: {result}")

            except Exception as e:
                print(f"   ❌ Exception: {e}")

    def create_test_group(self, group_name):
        """Create a test group"""
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
                "boardId": self.board_id,
                "groupName": group_name
            }
        }

        try:
            response = requests.post(self.api_url, json=mutation, headers=self.headers)
            result = response.json()

            if 'data' in result and 'create_group' in result['data']:
                return result['data']['create_group']['id']
            return None

        except Exception as e:
            print(f"Error creating group: {e}")
            return None

    def query_existing_items(self):
        """Query existing items to see what's actually in the board"""

        print("🔍 QUERYING EXISTING BOARD ITEMS")
        print("=" * 50)

        query = {
            "query": """
            query ($boardId: ID!) {
                boards(ids: [$boardId]) {
                    items(limit: 5) {
                        id
                        name
                        column_values {
                            id
                            title
                            value
                            text
                        }
                    }
                }
            }
            """,
            "variables": {
                "boardId": self.board_id
            }
        }

        try:
            response = requests.post(self.api_url, json=query, headers=self.headers)
            result = response.json()

            if 'data' in result and 'boards' in result['data']:
                items = result['data']['boards'][0]['items']

                print(f"📋 Found {len(items)} items in board:")

                for item in items:
                    print(f"\n🏠 Item: {item['name']} (ID: {item['id']})")

                    populated_fields = 0
                    for col_val in item['column_values']:
                        if col_val.get('value') or col_val.get('text'):
                            populated_fields += 1
                            print(
                                f"   ✅ {col_val['id']:25} | {col_val.get('title', 'No Title'):30} = {col_val.get('text') or col_val.get('value')}")

                    print(f"   📊 Total populated fields: {populated_fields}/{len(item['column_values'])}")

        except Exception as e:
            print(f"Error querying items: {e}")


def run_field_tests():
    """Run all field tests"""

    # Check environment
    if not os.getenv('MONDAY_API_KEY') or not os.getenv('MONDAY_BOARD_ID'):
        print("❌ Missing Monday.com credentials")
        return

    tester = FieldTypeTester()

    print("🧪 MONDAY.COM FIELD TYPE TESTING")
    print("=" * 60)

    # Test 1: Value formatting
    tester.test_field_value_formats()

    print()
    input("🤔 Press Enter to query existing board items...")

    # Test 2: Query existing items
    tester.query_existing_items()

    print()
    create_test = input("🤔 Create test items? This will add items to your board (y/N): ").lower()

    if create_test in ['y', 'yes']:
        # Test 3: Create minimal test items
        tester.create_minimal_test_item()

    print("\n🎯 Testing complete!")


if __name__ == "__main__":
    run_field_tests()