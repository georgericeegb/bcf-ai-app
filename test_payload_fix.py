#!/usr/bin/env python3
"""
Test the payload fix - compare what gets processed vs what gets sent
"""

import json
import pandas as pd
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv('.env', override=True)


def test_payload_consistency(csv_file_path, record_index=0):
    """Test that processed fields match what gets sent to Monday.com"""

    print("🔧 TESTING PAYLOAD CONSISTENCY")
    print("=" * 60)

    # Load CSV and get test record
    try:
        df = pd.read_csv(csv_file_path)
        record = df.iloc[record_index].to_dict()
        parcel_id = record.get('parcel_id', 'Unknown')
        print(f"✅ Testing record: Parcel {parcel_id}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Import CRM service
    try:
        from services.crm_service import CRMService
        crm = CRMService()
        print("✅ CRM service loaded")
    except Exception as e:
        print(f"❌ Could not load CRM service: {e}")
        return

    print(f"\n📊 STEP 1: PROCESSING FIELDS")
    print("-" * 40)

    # Step 1: Get processed fields
    processed_fields = crm.prepare_parcel_for_crm(record, 'solar')
    print(f"✅ prepare_parcel_for_crm returned {len(processed_fields)} fields")

    for key, value in processed_fields.items():
        print(f"   {key}: {repr(value)}")

    print(f"\n🔍 STEP 2: JSON SERIALIZATION TEST")
    print("-" * 40)

    # Step 2: Test JSON serialization (what create_crm_item does)
    serializable_fields = {}
    failed_serialization = []

    for key, value in processed_fields.items():
        try:
            json.dumps(value)
            serializable_fields[key] = value
            print(f"✅ {key}: JSON OK")
        except Exception as e:
            failed_serialization.append(f"{key}: {e}")
            print(f"❌ {key}: JSON FAILED - {e}")

    print(f"\n📈 SERIALIZATION RESULTS:")
    print(f"   Original fields: {len(processed_fields)}")
    print(f"   Serializable: {len(serializable_fields)}")
    print(f"   Failed: {len(failed_serialization)}")

    if failed_serialization:
        print(f"\n❌ SERIALIZATION FAILURES:")
        for failure in failed_serialization:
            print(f"   {failure}")

    print(f"\n📦 STEP 3: FINAL PAYLOAD SIMULATION")
    print("-" * 40)

    # Step 3: Create the exact payload that would be sent
    final_payload = {
        "board_id": crm.board_id,
        "group_id": "test_group_id",
        "item_name": crm.proper_case_with_exceptions(record.get('owner', 'Test Owner')),
        "column_values": serializable_fields
    }

    # Test complete payload serialization
    try:
        payload_json = json.dumps(final_payload, indent=2)
        print(f"✅ Complete payload JSON serialization successful")
        print(f"📏 Payload size: {len(payload_json)} characters")

        # Show what would actually be sent
        print(f"\n🚀 WHAT WOULD BE SENT TO MONDAY.COM:")
        print(f"Board ID: {final_payload['board_id']}")
        print(f"Item Name: {final_payload['item_name']}")
        print(f"Fields: {len(final_payload['column_values'])}")

        # Show the column values that would be sent
        print(f"\nColumn Values JSON:")
        column_values_json = json.dumps(final_payload['column_values'])
        print(column_values_json[:500] + "..." if len(column_values_json) > 500 else column_values_json)

    except Exception as e:
        print(f"❌ Complete payload serialization failed: {e}")
        return False

    print(f"\n🎯 CONSISTENCY CHECK")
    print("-" * 40)

    # Compare what we processed vs what would be sent
    processed_count = len(processed_fields)
    sendable_count = len(serializable_fields)

    if processed_count == sendable_count:
        print(f"✅ PERFECT CONSISTENCY: {processed_count} fields processed = {sendable_count} fields sendable")
        return True
    else:
        print(f"⚠️ INCONSISTENCY DETECTED:")
        print(f"   Processed: {processed_count} fields")
        print(f"   Sendable: {sendable_count} fields")
        print(f"   Lost: {processed_count - sendable_count} fields")

        # Show which fields were lost
        lost_fields = set(processed_fields.keys()) - set(serializable_fields.keys())
        if lost_fields:
            print(f"\n❌ LOST FIELDS:")
            for field in lost_fields:
                print(f"   {field}: {repr(processed_fields[field])}")

        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_payload_fix.py <csv_file_path>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    # Test the first record
    success = test_payload_consistency(csv_file, 0)

    if success:
        print(f"\n🎉 PAYLOAD CONSISTENCY TEST PASSED!")
        print("✅ All processed fields will be sent to Monday.com")
        print("💡 Your fix should work - try a real import now!")
    else:
        print(f"\n⚠️ PAYLOAD CONSISTENCY ISSUES DETECTED")
        print("🔧 The fix above should resolve these issues")
        print("💡 Update your CRM service and test again")