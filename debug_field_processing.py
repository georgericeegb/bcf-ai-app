#!/usr/bin/env python3
"""
Debug Field Processing Script
This will show exactly why fields are being rejected
"""

import json
import pandas as pd
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv('.env', override=True)


def debug_single_record(csv_file_path, record_index=0):
    """Debug processing of a single record to see where fields are lost"""

    print("🔍 DETAILED FIELD PROCESSING DEBUG")
    print("=" * 60)

    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Get the record to debug
    if record_index >= len(df):
        record_index = 0

    record = df.iloc[record_index].to_dict()
    parcel_id = record.get('parcel_id', 'Unknown')

    print(f"🏠 Debugging record {record_index + 1}: Parcel {parcel_id}")
    print(f"📋 Available fields in record: {len(record)}")

    # Import CRM service
    try:
        from services.crm_service import CRMService
        crm = CRMService()
        print("✅ CRM service loaded")
    except Exception as e:
        print(f"❌ Could not load CRM service: {e}")
        return

    print(f"\n🎯 FIELD-BY-FIELD ANALYSIS")
    print("=" * 60)

    # Debug each field mapping
    successful_fields = {}
    failed_fields = {}

    for field_key, monday_field in crm.crm_field_mapping.items():
        if field_key == 'owner':
            continue

        print(f"\n🔍 Processing: {field_key} -> {monday_field}")

        # Step 1: Check field variations
        variations = crm.field_variations.get(field_key, [field_key])
        print(f"   Variations to check: {variations}")

        # Step 2: Find which variations exist in the record
        found_variations = [var for var in variations if var in record]
        print(f"   Found in record: {found_variations}")

        if not found_variations:
            failed_fields[field_key] = "No matching field name found"
            print(f"   ❌ FAILED: No field name match")
            continue

        # Step 3: Get the raw value
        raw_value = None
        found_field = None
        for var in found_variations:
            if var in record:
                raw_value = record[var]
                found_field = var
                break

        print(f"   Raw value from '{found_field}': {repr(raw_value)} (type: {type(raw_value)})")

        # Step 4: Check if value is valid
        is_valid = crm.is_valid_value(raw_value, field_key)
        print(f"   Is valid: {is_valid}")

        if not is_valid:
            failed_fields[field_key] = f"Invalid value: {repr(raw_value)}"
            print(f"   ❌ FAILED: Value rejected as invalid")
            continue

        # Step 5: Try formatting
        try:
            formatted_value = crm.format_field_value(field_key, raw_value, monday_field)
            print(f"   Formatted value: {repr(formatted_value)} (type: {type(formatted_value)})")

            if formatted_value is not None:
                successful_fields[field_key] = {
                    'source_field': found_field,
                    'raw_value': raw_value,
                    'formatted_value': formatted_value,
                    'monday_field': monday_field
                }
                print(f"   ✅ SUCCESS: Ready for CRM")
            else:
                failed_fields[field_key] = f"Formatting returned None for: {repr(raw_value)}"
                print(f"   ❌ FAILED: Formatting returned None")

        except Exception as e:
            failed_fields[field_key] = f"Formatting error: {str(e)}"
            print(f"   ❌ FAILED: Formatting error: {e}")

    print(f"\n📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Successful fields: {len(successful_fields)}")
    print(f"❌ Failed fields: {len(failed_fields)}")
    print(f"📈 Success rate: {len(successful_fields) / (len(successful_fields) + len(failed_fields)) * 100:.1f}%")

    if successful_fields:
        print(f"\n✅ SUCCESSFUL FIELDS:")
        for field_key, info in successful_fields.items():
            print(f"   {field_key:20} -> {info['monday_field']:15} = {repr(info['formatted_value'])}")

    if failed_fields:
        print(f"\n❌ FAILED FIELDS:")
        for field_key, reason in failed_fields.items():
            print(f"   {field_key:20} -> {reason}")

    print(f"\n🔧 SPECIFIC MISSING FIELD ANALYSIS")
    print("=" * 60)

    # Check the specific fields mentioned by user
    critical_missing = ['mail_address1', 'county_id', 'address', 'latitude', 'longitude']

    for field_name in critical_missing:
        if field_name in record:
            value = record[field_name]
            print(f"\n🔍 {field_name}:")
            print(f"   Raw value: {repr(value)}")
            print(f"   Type: {type(value)}")
            print(f"   Is null: {pd.isna(value) if hasattr(pd, 'isna') else 'unknown'}")
            print(f"   String repr: '{str(value)}'")

            # Test validation manually
            try:
                str_val = str(value).strip().lower()
                print(f"   String lower: '{str_val}'")

                invalid_values = ['', 'nan', 'none', 'null', '#n/a', 'unknown']
                is_invalid_string = str_val in invalid_values
                print(f"   Is invalid string: {is_invalid_string}")

                if field_name in ['latitude', 'longitude']:
                    try:
                        coord_val = float(value)
                        print(f"   Coordinate value: {coord_val}")
                        print(f"   Is zero: {coord_val == 0.0}")
                        print(f"   Abs > 180: {abs(coord_val) > 180}")
                    except:
                        print(f"   Cannot convert to float")

            except Exception as e:
                print(f"   Error in manual validation: {e}")
        else:
            print(f"\n❌ {field_name}: NOT FOUND in record")

    # Show what would actually be sent to Monday.com
    final_payload = {}
    for field_key, info in successful_fields.items():
        final_payload[info['monday_field']] = info['formatted_value']

    print(f"\n📦 FINAL MONDAY.COM PAYLOAD")
    print("=" * 60)
    print(json.dumps(final_payload, indent=2))

    return successful_fields, failed_fields


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_field_processing.py <csv_file_path>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    # Debug the first record
    debug_single_record(csv_file, 0)

    print(f"\n💡 To debug a different record, modify the record_index parameter")
    print(f"💡 This will show exactly why each field is being included or rejected")