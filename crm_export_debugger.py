#!/usr/bin/env python3
"""
crm_export_debugger.py - Debug actual CRM exports step by step

This tool helps debug why values aren't making it into Monday.com
by intercepting and logging the actual export process.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add project directory to path
sys.path.append(str(Path(__file__).parent))


def setup_detailed_logging():
    """Set up detailed logging to see what's happening"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crm_export_debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Enable debug logging for our CRM service
    logger = logging.getLogger('services.crm_service')
    logger.setLevel(logging.DEBUG)

    print("✅ Detailed logging enabled - check 'crm_export_debug.log' for full details")
    return logger


def test_single_parcel_export(csv_file_path, row_index=0):
    """Test exporting a single parcel with full debugging"""

    print(f"🔍 Testing Single Parcel Export")
    print("=" * 60)

    # Set up logging
    logger = setup_detailed_logging()

    try:
        # Import your actual CRM service
        from services.crm_service import CRMService
        import pandas as pd

        # Load parcel data
        df = pd.read_csv(csv_file_path)
        if row_index >= len(df):
            print(f"❌ Row {row_index} not available")
            return False

        parcel_data = df.iloc[row_index].to_dict()

        # Convert NaN values to None
        for key, value in parcel_data.items():
            if pd.isna(value):
                parcel_data[key] = None

        parcel_id = parcel_data.get('parcel_id', 'Unknown')
        print(f"📋 Testing parcel: {parcel_id}")

        # Initialize CRM service
        print(f"🔧 Initializing CRM service...")
        crm_service = CRMService()

        # Test connection first
        print(f"🔗 Testing API connection...")
        connection_test = crm_service.test_connection()
        if not connection_test['success']:
            print(f"❌ Connection failed: {connection_test['error']}")
            return False

        print(f"✅ Connected as: {connection_test['user']['name']}")

        # Create a test group
        print(f"📁 Creating test group...")
        group_name = f"DEBUG - Single Parcel Test - {parcel_id}"
        group_id = crm_service.create_group_in_board(group_name)

        if not group_id:
            print(f"❌ Failed to create test group")
            return False

        print(f"✅ Created group: {group_id}")

        # Process parcel data
        print(f"⚙️  Processing parcel data...")
        crm_values = crm_service.prepare_parcel_for_crm(parcel_data, 'solar')

        print(f"📤 CRM Values Generated:")
        print(f"   Total fields: {len(crm_values)}")

        # Show all values
        for field_id, value in crm_values.items():
            print(f"   {field_id:<25} = {repr(value)} ({type(value).__name__})")

        # Prepare owner name
        owner_name = crm_service.proper_case_with_exceptions(parcel_data.get('owner', 'Unknown Owner'))
        print(f"👤 Owner name: {owner_name}")

        # Show the exact JSON that will be sent
        column_values_json = json.dumps(crm_values)
        print(f"\n📋 JSON Payload (what Monday.com receives):")
        print(column_values_json)

        # Validate JSON
        try:
            json.loads(column_values_json)
            print(f"✅ JSON is valid")
        except json.JSONDecodeError as e:
            print(f"❌ JSON validation error: {e}")
            return False

        # Create the actual CRM item
        print(f"\n🚀 Creating CRM item...")
        success = crm_service.create_crm_item(group_id, owner_name, crm_values)

        if success:
            print(f"✅ CRM item created successfully!")
            print(f"📍 Check your Monday.com board for group: {group_name}")
            return True
        else:
            print(f"❌ CRM item creation failed")
            return False

    except Exception as e:
        print(f"❌ Error during export test: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_field_mappings():
    """Compare your actual CRM service field mappings with what we expect"""

    print(f"🔍 Comparing Field Mappings")
    print("=" * 60)

    try:
        from services.crm_service import CRMService

        crm_service = CRMService()

        print(f"📋 Your CRM Field Mappings ({len(crm_service.crm_field_mapping)} fields):")
        for field_key, monday_field in crm_service.crm_field_mapping.items():
            print(f"   {field_key:<25} -> {monday_field}")

        print(f"\n📋 Your Field Variations:")
        for field_key, variations in crm_service.field_variations.items():
            print(f"   {field_key:<25} : {variations[:3]}{'...' if len(variations) > 3 else ''}")

        return True

    except Exception as e:
        print(f"❌ Error loading CRM service: {e}")
        return False


def test_monday_field_acceptance():
    """Test if Monday.com accepts our field IDs by testing each one individually"""

    print(f"🧪 Testing Monday.com Field Acceptance")
    print("=" * 60)

    try:
        from services.crm_service import CRMService

        crm_service = CRMService()

        # Create a test group
        group_name = f"DEBUG - Field Test - Individual Fields"
        group_id = crm_service.create_group_in_board(group_name)

        if not group_id:
            print(f"❌ Failed to create test group")
            return False

        print(f"✅ Created test group: {group_id}")

        # Test basic field combinations
        test_cases = [
            # Test 1: Just basic text field
            {
                "name": "Basic Text Test",
                "values": {"text117": "TEST123"}
            },
            # Test 2: Number field
            {
                "name": "Number Test",
                "values": {"numbers6": 100}
            },
            # Test 3: Coordinate fields
            {
                "name": "Coordinate Test",
                "values": {"latitude__1": "41.4032", "longitude__1": "-81.8607"}
            },
            # Test 4: Multiple text fields
            {
                "name": "Multiple Text Test",
                "values": {"text4": "Test County", "text_1": "OH", "text1": "123 Test St"}
            },
            # Test 5: Numeric analysis fields
            {
                "name": "Analysis Fields Test",
                "values": {"numeric_mknpptf4": 85.5, "numeric_mknphdv8": 90.2}
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['name']}")
            print(f"   Fields: {list(test_case['values'].keys())}")

            item_name = f"TEST {i} - {test_case['name']}"
            success = crm_service.create_crm_item(group_id, item_name, test_case['values'])

            results.append({
                'test': test_case['name'],
                'fields': test_case['values'],
                'success': success
            })

            if success:
                print(f"   ✅ Success")
            else:
                print(f"   ❌ Failed")

        print(f"\n📊 Test Results Summary:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['test']}")

        print(f"\n📍 Check Monday.com group '{group_name}' to see which items were created")

        return results

    except Exception as e:
        print(f"❌ Error during field testing: {e}")
        import traceback
        traceback.print_exc()
        return []


def check_crm_service_version():
    """Check if you're using the updated CRM service"""

    print(f"🔍 Checking CRM Service Version")
    print("=" * 60)

    try:
        from services.crm_service import CRMService
        import inspect

        crm_service = CRMService()

        # Check for new methods/attributes that indicate the fixed version
        indicators = [
            ('is_valid_value', 'method'),
            ('field_variations', 'attribute'),
            ('crm_field_mapping', 'attribute')
        ]

        print(f"🔍 Checking for updated CRM service indicators:")

        for indicator, item_type in indicators:
            if hasattr(crm_service, indicator):
                print(f"   ✅ {indicator} ({item_type}) - present")

                if indicator == 'field_variations':
                    field_count = len(getattr(crm_service, indicator))
                    print(f"      Field variations count: {field_count}")
                    if field_count < 15:
                        print(f"      ⚠️  Low count - may be old version")

                elif indicator == 'crm_field_mapping':
                    mapping_count = len(getattr(crm_service, indicator))
                    print(f"      CRM field mappings count: {mapping_count}")
                    if mapping_count < 30:
                        print(f"      ⚠️  Low count - may be missing fields")

            else:
                print(f"   ❌ {indicator} ({item_type}) - missing")

        # Check method signatures
        prepare_method = getattr(crm_service, 'prepare_parcel_for_crm', None)
        if prepare_method:
            sig = inspect.signature(prepare_method)
            params = list(sig.parameters.keys())
            print(f"   📋 prepare_parcel_for_crm parameters: {params}")

        return True

    except Exception as e:
        print(f"❌ Error checking CRM service: {e}")
        return False


def main():
    """Main debugging interface"""

    print(f"🔧 CRM Export Live Debugger")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("Usage: python crm_export_debugger.py <command> [csv_file]")
        print("\nCommands:")
        print("  version       - Check if you're using the updated CRM service")
        print("  mappings      - Show your current field mappings")
        print("  test-fields   - Test individual Monday.com fields")
        print("  test-parcel   - Test exporting a single parcel")
        print("\nExamples:")
        print("  python crm_export_debugger.py version")
        print("  python crm_export_debugger.py test-parcel your_file.csv")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'version':
        check_crm_service_version()

    elif command == 'mappings':
        compare_field_mappings()

    elif command == 'test-fields':
        test_monday_field_acceptance()

    elif command == 'test-parcel':
        if len(sys.argv) < 3:
            print("❌ CSV file required for test-parcel command")
            sys.exit(1)

        csv_file = sys.argv[2]
        row_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

        if not Path(csv_file).exists():
            print(f"❌ File not found: {csv_file}")
            sys.exit(1)

        success = test_single_parcel_export(csv_file, row_index)
        if not success:
            sys.exit(1)

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

    print(f"\n✅ Debug completed!")


if __name__ == "__main__":
    main()