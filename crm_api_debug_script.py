# crm_api_debug_script.py - Debug CRM API calls and field imports

import pandas as pd
import json
import os
import sys
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime


# Load environment variables
def load_project_env():
    """Load .env file from project root directory"""
    current_dir = Path.cwd()

    for i in range(5):
        env_path = current_dir / '.env'
        if env_path.exists():
            print(f"📁 Found .env file at: {env_path}")
            load_dotenv(env_path, override=True)
            return True
        current_dir = current_dir.parent

    return False


# Load environment variables
print("🔧 Loading environment variables...")
if not load_project_env():
    print("❌ Could not find .env file")
    sys.exit(1)

try:
    from services.crm_service import CRMService
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class CRMAPIDebugger:
    def __init__(self):
        self.api_url = os.getenv('MONDAY_API_URL', 'https://api.monday.com/v2')
        self.api_key = os.getenv('MONDAY_API_KEY')
        self.board_id = os.getenv('MONDAY_BOARD_ID')

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "API-Version": "2023-10"
        }

        self.crm_service = CRMService()

    def debug_single_item_creation(self, sample_parcel, project_type='solar'):
        """Debug the complete process of creating a single CRM item"""

        print("🔍 CRM API DEBUG - SINGLE ITEM CREATION")
        print("=" * 60)

        parcel_id = sample_parcel.get('parcel_id', 'TEST_PARCEL')
        print(f"🏠 Testing parcel: {parcel_id}")
        print()

        # Step 1: Process parcel data
        print("📊 STEP 1: Processing parcel data for CRM...")
        crm_values = self.crm_service.prepare_parcel_for_crm(sample_parcel, project_type)
        owner_name = self.crm_service.proper_case_with_exceptions(sample_parcel.get('owner', 'Test Owner'))

        print(f"   Owner name: {owner_name}")
        print(f"   CRM values generated: {len(crm_values)}")
        print()

        # Step 2: Show the exact API payload
        print("📤 STEP 2: API Payload Construction...")

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
                "groupId": "temp_debug_group",  # We'll use a test group
                "itemName": owner_name,
                "columnValues": json.dumps(crm_values)
            }
        }

        print("🔧 EXACT API PAYLOAD:")
        print("-" * 40)
        print("HEADERS:")
        for key, value in self.headers.items():
            if key == "Authorization":
                print(f"  {key}: Bearer {value[:20]}...")
            else:
                print(f"  {key}: {value}")

        print()
        print("MUTATION QUERY:")
        print(mutation["query"])

        print()
        print("VARIABLES:")
        print(f"  boardId: {mutation['variables']['boardId']}")
        print(f"  groupId: {mutation['variables']['groupId']}")
        print(f"  itemName: {mutation['variables']['itemName']}")

        print()
        print("COLUMN VALUES (JSON):")
        print("  Raw column_values object:")
        for monday_field, value in crm_values.items():
            print(f"    {monday_field:25} = {repr(value)} ({type(value).__name__})")

        print()
        print("  JSON-encoded column_values:")
        json_encoded = json.dumps(crm_values, indent=2)
        print(f"  {json_encoded}")

        print()
        print("  JSON validation:")
        try:
            parsed_back = json.loads(json_encoded)
            print(f"  ✅ Valid JSON - {len(parsed_back)} fields")
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            return None

        # Step 3: Test group creation first
        print()
        print("📁 STEP 3: Creating test group...")
        test_group_name = f"DEBUG_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        group_id = self.create_test_group(test_group_name)

        if not group_id:
            print("❌ Failed to create test group - cannot proceed")
            return None

        print(f"✅ Created test group: {test_group_name} (ID: {group_id})")

        # Step 4: Send the actual API request
        print()
        print("🚀 STEP 4: Sending API request to Monday.com...")

        # Update the mutation with the real group ID
        mutation["variables"]["groupId"] = group_id

        try:
            print("📡 Making API request...")
            response = requests.post(self.api_url, json=mutation, headers=self.headers)

            print(f"📊 Response status: {response.status_code}")
            print(f"📊 Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                print("📊 Response body:")
                print(json.dumps(result, indent=2))

                # Analyze the response
                if 'data' in result and 'create_item' in result['data']:
                    created_item = result['data']['create_item']
                    print()
                    print("✅ ITEM CREATED SUCCESSFULLY!")
                    print(f"   Item ID: {created_item['id']}")
                    print(f"   Item Name: {created_item['name']}")
                    print(f"   Column Values Returned: {len(created_item.get('column_values', []))}")

                    # Check which fields actually got set
                    column_values_returned = created_item.get('column_values', [])
                    print()
                    print("📋 FIELD-BY-FIELD ANALYSIS:")

                    for monday_field, sent_value in crm_values.items():
                        found_in_response = False
                        for col_val in column_values_returned:
                            if col_val['id'] == monday_field:
                                found_in_response = True
                                returned_value = col_val.get('value', '')
                                returned_text = col_val.get('text', '')

                                if returned_value or returned_text:
                                    print(
                                        f"   ✅ {monday_field:25} -> Sent: {repr(sent_value)}, Got: {repr(returned_text or returned_value)}")
                                else:
                                    print(f"   ⚠️  {monday_field:25} -> Sent: {repr(sent_value)}, Got: EMPTY")
                                break

                        if not found_in_response:
                            print(f"   ❌ {monday_field:25} -> Sent: {repr(sent_value)}, NOT IN RESPONSE")

                    return created_item['id']

                elif 'errors' in result:
                    print("❌ API ERRORS:")
                    for error in result['errors']:
                        print(f"   Error: {error.get('message', 'Unknown error')}")
                        if 'extensions' in error:
                            print(f"   Details: {error['extensions']}")
                    return None

                else:
                    print("❌ UNEXPECTED RESPONSE STRUCTURE:")
                    print(json.dumps(result, indent=2))
                    return None

            else:
                print(f"❌ HTTP ERROR {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ REQUEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_test_group(self, group_name):
        """Create a test group for debugging"""
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
            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'create_group' in result['data']:
                return result['data']['create_group']['id']
            else:
                print(f"❌ Group creation failed: {result}")
                return None

        except Exception as e:
            print(f"❌ Error creating test group: {e}")
            return None

    def analyze_board_structure(self):
        """Analyze the Monday.com board structure to verify field IDs"""

        print("🔍 BOARD STRUCTURE ANALYSIS")
        print("=" * 60)

        query = {
            "query": """
            query ($boardId: ID!) {
                boards(ids: [$boardId]) {
                    id
                    name
                    columns {
                        id
                        title
                        type
                        settings_str
                    }
                    groups {
                        id
                        title
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
            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'boards' in result['data'] and result['data']['boards']:
                board = result['data']['boards'][0]

                print(f"📋 Board: {board['name']} (ID: {board['id']})")
                print()

                print("📊 AVAILABLE COLUMNS:")
                columns = board['columns']
                for col in columns:
                    print(f"   {col['id']:25} | {col['title']:30} | {col['type']}")

                print()
                print("📁 AVAILABLE GROUPS:")
                groups = board['groups']
                for group in groups:
                    print(f"   {group['id']:25} | {group['title']}")

                print()
                print("🔍 FIELD MAPPING VERIFICATION:")
                crm_mapping = self.crm_service.crm_field_mapping

                for field_key, monday_field in crm_mapping.items():
                    if field_key == 'owner':  # Skip owner as it's item_name
                        continue

                    found_column = None
                    for col in columns:
                        if col['id'] == monday_field:
                            found_column = col
                            break

                    if found_column:
                        print(f"   ✅ {field_key:25} -> {monday_field:25} ({found_column['type']})")
                    else:
                        print(f"   ❌ {field_key:25} -> {monday_field:25} (NOT FOUND IN BOARD)")

                return board

            else:
                print(f"❌ Board query failed: {result}")
                return None

        except Exception as e:
            print(f"❌ Error analyzing board: {e}")
            return None

    def test_individual_field_updates(self, item_id, crm_values):
        """Test updating individual fields to isolate issues"""

        print()
        print("🔧 INDIVIDUAL FIELD UPDATE TESTS")
        print("=" * 60)

        for monday_field, value in crm_values.items():
            print(f"🧪 Testing field: {monday_field} = {repr(value)}")

            mutation = {
                "query": """
                mutation ($boardId: ID!, $itemId: ID!, $columnId: String!, $value: JSON!) {
                    change_column_value(
                        board_id: $boardId,
                        item_id: $itemId,
                        column_id: $columnId,
                        value: $value
                    ) {
                        id
                        name
                    }
                }
                """,
                "variables": {
                    "boardId": self.board_id,
                    "itemId": item_id,
                    "columnId": monday_field,
                    "value": json.dumps(value) if not isinstance(value, str) else value
                }
            }

            try:
                response = requests.post(self.api_url, json=mutation, headers=self.headers)
                result = response.json()

                if response.status_code == 200 and 'data' in result:
                    print(f"   ✅ Successfully updated {monday_field}")
                else:
                    print(f"   ❌ Failed to update {monday_field}")
                    if 'errors' in result:
                        for error in result['errors']:
                            print(f"      Error: {error.get('message', 'Unknown')}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"   ❌ Exception updating {monday_field}: {e}")


def run_comprehensive_crm_debug():
    """Run comprehensive CRM debugging"""

    # Load sample CSV data
    csv_file_path = "C:/Users/georg/Downloads/OH_Cuyahoga_Parcel_Files_Cuyahoga_OH_parcels_08182025_1815.csv"

    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        return

    try:
        print("📂 Loading CSV data...")
        df = pd.read_csv(csv_file_path)
        sample_parcel = df.iloc[0].to_dict()

        print(f"✅ Loaded sample parcel: {sample_parcel.get('parcel_id', 'Unknown')}")
        print()

        # Initialize debugger
        debugger = CRMAPIDebugger()

        # Step 1: Analyze board structure
        board_info = debugger.analyze_board_structure()
        if not board_info:
            print("❌ Cannot analyze board structure - check API credentials")
            return

        print()
        input("🤔 Press Enter to continue with item creation test...")

        # Step 2: Test item creation
        item_id = debugger.debug_single_item_creation(sample_parcel)

        if item_id:
            print()
            input("🤔 Press Enter to test individual field updates...")

            # Step 3: Test individual field updates
            crm_values = debugger.crm_service.prepare_parcel_for_crm(sample_parcel, 'solar')
            debugger.test_individual_field_updates(item_id, crm_values)

        print()
        print("🎯 DEBUG COMPLETE!")
        print("Check your Monday.com board to see which fields actually populated.")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Verify environment setup
    api_key = os.getenv('MONDAY_API_KEY')
    board_id = os.getenv('MONDAY_BOARD_ID')

    if not api_key or not board_id:
        print("❌ Missing required environment variables")
        print("   MONDAY_API_KEY:", "✅ Set" if api_key else "❌ Missing")
        print("   MONDAY_BOARD_ID:", "✅ Set" if board_id else "❌ Missing")
        sys.exit(1)

    print("✅ Environment variables loaded")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Board ID: {board_id}")
    print()

    run_comprehensive_crm_debug()