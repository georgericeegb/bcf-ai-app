#!/usr/bin/env python3
"""
crm_debug_payload.py - Debug tool to show exact API payload sent to Monday.com

Usage:
    python crm_debug_payload.py your_parcel_file.csv

This will show you:
1. Raw parcel data from CSV
2. Processed CRM values
3. Exact JSON payload that would be sent to Monday.com API
4. Monday.com mutation query structure
"""

import sys
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Environment variables may not load.")

# Add project directory to path
sys.path.append(str(Path(__file__).parent))


def create_sample_parcel_from_csv(csv_file_path, row_index=0):
    """Load a sample parcel from CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        if row_index >= len(df):
            print(f"❌ Row {row_index} not available. CSV has {len(df)} rows.")
            return None

        parcel_data = df.iloc[row_index].to_dict()

        # Convert any NaN values to None for JSON serialization
        for key, value in parcel_data.items():
            if pd.isna(value):
                parcel_data[key] = None

        return parcel_data

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def create_sample_suitability_analysis():
    """Create sample suitability analysis data"""
    return {
        'is_suitable': True,
        'overall_score': 85.5,
        'slope_score': 90.2,
        'transmission_score': 80.8,
        'slope_degrees': 8.5,
        'transmission_distance': 0.75,
        'transmission_voltage': 230.0,
        'slope_suitable': True,
        'transmission_suitable': True,
        'analysis_notes': 'Excellent slope (8.5°); Good transmission access (0.75 mi, 230 kV)'
    }


def simulate_crm_processing(parcel_data, include_suitability=True):
    """Simulate the CRM processing without requiring actual CRM service"""

    # COMPLETE field mappings matching the fixed CRM service
    field_mappings = {
        'owner': {
            'variations': ['owner', 'owner_name', 'property_owner', 'landowner'],
            'monday_field': 'item_name'
        },
        'county_id': {
            'variations': ['county_id', 'cnty_id', 'fips', 'county_fips'],
            'monday_field': 'county_id__1'
        },
        'county_name': {
            'variations': ['county_name', 'county', 'county_nam'],
            'monday_field': 'text4'
        },
        'state_abbr': {
            'variations': ['state_abbr', 'state', 'st', 'state_code'],
            'monday_field': 'text_1'
        },
        'address': {
            'variations': ['address', 'property_address', 'site_address', 'location'],
            'monday_field': 'text1'
        },
        'muni_name': {
            'variations': ['muni_name', 'municipality', 'city'],
            'monday_field': 'text66'
        },
        'census_zip': {
            'variations': ['census_zip', 'zip', 'zipcode', 'zip_code'],
            'monday_field': 'text_mktw4254'
        },
        'mkt_val_land': {
            'variations': ['mkt_val_land', 'land_value', 'market_value_land'],
            'monday_field': 'numbers85__1'
        },
        'land_use_code': {
            'variations': ['land_use_code', 'land_use_c', 'use_code'],
            'monday_field': 'land_use_code__1'
        },
        'mail_address1': {
            'variations': ['mail_address1', 'mail_address', 'owner_address'],
            'monday_field': 'text7'
        },
        'mail_placename': {
            'variations': ['mail_placename', 'mail_city', 'owner_city'],
            'monday_field': 'text49'
        },
        'mail_statename': {
            'variations': ['mail_statename', 'mail_state', 'owner_state'],
            'monday_field': 'text11'
        },
        'mail_zipcode': {
            'variations': ['mail_zipcode', 'mail_zip', 'owner_zip'],
            'monday_field': 'mzip'
        },
        'parcel_id': {
            'variations': ['parcel_id', 'pin', 'apn', 'parcel_number', 'property_id'],
            'monday_field': 'text117'
        },
        'acreage_calc': {
            'variations': ['acreage_calc', 'acreage', 'acres', 'total_acres'],
            'monday_field': 'numbers6'
        },
        'acreage_adjacent_with_sameowner': {
            'variations': ['acreage_adjacent_with_sameowner', 'adjacent_acres'],
            'monday_field': 'dup__of_score__0___3___1'
        },
        'latitude': {
            'variations': ['latitude', 'lat', 'y', 'y_coord'],
            'monday_field': 'latitude__1'
        },
        'longitude': {
            'variations': ['longitude', 'lon', 'x', 'x_coord'],
            'monday_field': 'longitude__1'
        },
        'elevation': {
            'variations': ['elevation', 'elev', 'altitude', 'height'],
            'monday_field': 'numeric_mktwrwry'
        },
        'legal_desc1': {
            'variations': ['legal_desc1', 'legal_description', 'legal_desc'],
            'monday_field': 'text_mktw1gns'
        },
        'land_cover': {
            'variations': ['land_cover', 'landcover', 'cover'],
            'monday_field': 'long_text__1'
        },
        'county_link': {
            'variations': ['county_link', 'link', 'web_link', 'url'],
            'monday_field': 'text_mktw6bvk'
        },
        'fld_zone': {
            'variations': ['fld_zone', 'flood_zone', 'fema_zone'],
            'monday_field': 'text_mkkbx2zc'
        },
        'zone_subty': {
            'variations': ['zone_subty', 'zone_subtype', 'subtype'],
            'monday_field': 'text_mktwy6h5'
        },
        'avg_slope': {
            'variations': ['avg_slope', 'slope_category', 'slope_class'],
            'monday_field': 'dropdown_mkkzj3m8'
        }
    }

    # Suitability field mappings matching the fixed CRM service
    suitability_mappings = {
        'overall_score': 'numeric_mknpptf4',  # solar_score
        'slope_score': 'numeric_mknphdv8',  # wind_score
        'transmission_score': 'numeric_mknpp74r',  # battery_score
        'slope_degrees': 'numeric_mktx3jgs',  # avg_slope_degrees
        'transmission_distance': 'numbers66__1',  # miles_from_transmission
        'transmission_voltage': 'numbers46__1'  # nearest_transmission_voltage
    }

    def is_valid_value(value):
        """Check if value is valid for CRM"""
        if value is None:
            return False
        str_val = str(value).strip().lower()
        if str_val in ['', 'null', 'none', 'nan', 'n/a']:
            return False
        try:
            import math
            if isinstance(value, (int, float)) and math.isnan(float(value)):
                return False
        except:
            pass
        return True

    def find_field_value(data, variations):
        """Find field value using variations"""
        for variation in variations:
            if variation in data and is_valid_value(data[variation]):
                return data[variation]
        return None

    def format_value(value, field_type):
        """Format value based on field type"""
        if not is_valid_value(value):
            return None

        if field_type == 'text':
            return str(value).strip()[:255]
        elif field_type == 'number':
            try:
                return float(value)
            except:
                return None
        elif field_type == 'integer':
            try:
                return int(float(value))
            except:
                return None
        elif field_type == 'coordinate':
            try:
                coord = float(value)
                return str(coord) if coord != 0 else None
            except:
                return None
        else:
            return str(value).strip()

    # Process standard fields
    crm_values = {}
    owner_name = "Unknown Owner"

    # Get owner name first
    owner_raw = find_field_value(parcel_data, ['owner', 'owner_name', 'property_owner'])
    if owner_raw:
        owner_name = str(owner_raw).strip()

    # Process all fields (including the ones that were missing)
    owner_name = "Unknown Owner"

    for field_key, config in field_mappings.items():
        raw_value = find_field_value(parcel_data, config['variations'])

        if raw_value is not None:
            # Determine field type
            if field_key == 'owner':
                formatted_value = str(raw_value).strip()
                owner_name = formatted_value
                # Don't add owner to crm_values as it goes in item_name
            elif field_key == 'acreage':
                formatted_value = format_value(raw_value, 'integer')
            elif field_key in ['latitude', 'longitude']:
                formatted_value = format_value(raw_value, 'coordinate')
            elif field_key == 'land_value':
                formatted_value = format_value(raw_value, 'number')
            else:
                formatted_value = format_value(raw_value, 'text')

            if formatted_value is not None and field_key != 'owner':
                crm_values[config['monday_field']] = formatted_value

    # Add suitability analysis if requested
    if include_suitability and 'suitability_analysis' in parcel_data:
        analysis = parcel_data['suitability_analysis']
        for analysis_field, monday_field in suitability_mappings.items():
            if analysis_field in analysis:
                raw_value = analysis[analysis_field]
                formatted_value = format_value(raw_value, 'number')
                if formatted_value is not None:
                    crm_values[monday_field] = formatted_value

    return owner_name, crm_values


def create_monday_api_payload(owner_name, crm_values, board_id, group_id):
    """Create the exact Monday.com API payload"""

    # GraphQL mutation
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
            "itemName": owner_name,
            "columnValues": json.dumps(crm_values)
        }
    }

    return mutation


def debug_crm_payload(csv_file_path, row_index=0, include_suitability=True):
    """Debug the complete CRM payload creation process"""

    print(f"🔍 CRM API Payload Debug")
    print("=" * 80)

    # Load sample parcel
    print(f"📁 Loading parcel from CSV (row {row_index})...")
    parcel_data = create_sample_parcel_from_csv(csv_file_path, row_index)

    if not parcel_data:
        return

    # Add sample suitability analysis if requested
    if include_suitability:
        print(f"🧪 Adding sample suitability analysis data...")
        parcel_data['suitability_analysis'] = create_sample_suitability_analysis()

    # Show raw parcel data
    print(f"\n📋 Raw Parcel Data (first 10 fields):")
    print("-" * 60)
    for i, (key, value) in enumerate(list(parcel_data.items())[:10]):
        print(f"  {key:<25} = {value}")
    if len(parcel_data) > 10:
        print(f"  ... and {len(parcel_data) - 10} more fields")

    # Process through CRM mapping
    print(f"\n⚙️  Processing through CRM field mapping...")
    owner_name, crm_values = simulate_crm_processing(parcel_data, include_suitability)

    print(f"\n📤 Processed CRM Values:")
    print("-" * 60)
    print(f"Item Name (owner): {owner_name}")
    print(f"Column Values ({len(crm_values)} fields):")
    for monday_field, value in crm_values.items():
        print(f"  {monday_field:<25} = {value} ({type(value).__name__})")

    # Create API payload
    board_id = os.getenv('MONDAY_BOARD_ID', 'YOUR_BOARD_ID')
    group_id = "sample_group_id"

    api_payload = create_monday_api_payload(owner_name, crm_values, board_id, group_id)

    # Show complete API payload
    print(f"\n🌐 Complete Monday.com API Payload:")
    print("-" * 60)
    print("URL: https://api.monday.com/v2")
    print("Method: POST")
    print("Headers:")
    print(f"  Authorization: Bearer {os.getenv('MONDAY_API_KEY', 'YOUR_API_KEY')[:20]}...")
    print(f"  Content-Type: application/json")
    print(f"  API-Version: 2023-10")

    print(f"\nJSON Payload:")
    print(json.dumps(api_payload, indent=2))

    # Show just the column values JSON
    print(f"\n📋 Column Values JSON (what Monday.com receives):")
    print("-" * 60)
    column_values_json = json.dumps(crm_values, indent=2)
    print(column_values_json)

    # Validate JSON
    try:
        json.loads(column_values_json)
        print(f"\n✅ JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON validation error: {e}")

    # Show field mapping summary
    print(f"\n📊 Field Mapping Summary:")
    print("-" * 40)
    print(f"Owner name: {owner_name}")
    print(f"CRM fields mapped: {len(crm_values)}")
    print(f"Board ID: {board_id}")
    print(f"Group ID: {group_id}")

    # Check for potential issues
    print(f"\n🔍 Potential Issues Check:")
    print("-" * 40)

    issues = []

    if len(crm_values) == 0:
        issues.append("❌ No CRM values mapped - check field variations")
    elif len(crm_values) < 10:
        issues.append(f"⚠️  Only {len(crm_values)} fields mapped - expected ~25+ fields")

    if owner_name in ['Unknown Owner', '', None]:
        issues.append("⚠️  Owner name is empty or default")

    if board_id in ['YOUR_BOARD_ID', '', None]:
        issues.append("❌ Board ID not set in environment variables")

    # Check for NaN values in JSON
    json_str = json.dumps(crm_values)
    if 'NaN' in json_str or 'null' in json_str:
        issues.append("⚠️  Potential NaN or null values in JSON")

    # Check field coverage
    expected_basic_fields = ['text117', 'numbers6', 'text4', 'text_1',
                             'text1']  # parcel_id, acreage, county, state, address
    missing_basic = [field for field in expected_basic_fields if field not in crm_values]
    if missing_basic:
        issues.append(f"⚠️  Missing basic fields: {', '.join(missing_basic)}")

    if not issues:
        print("✅ No obvious issues found!")
        if len(crm_values) >= 20:
            print(f"✅ Excellent field coverage: {len(crm_values)} fields mapped!")
        elif len(crm_values) >= 15:
            print(f"✅ Good field coverage: {len(crm_values)} fields mapped")
    else:
        for issue in issues:
            print(f"  {issue}")

    return {
        'owner_name': owner_name,
        'crm_values': crm_values,
        'api_payload': api_payload,
        'issues': issues
    }


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python crm_debug_payload.py <csv_file_path> [row_index] [include_suitability]")
        print("Example: python crm_debug_payload.py parcels.csv 0 true")
        sys.exit(1)

    csv_file = sys.argv[1]
    row_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    include_suitability = sys.argv[3].lower() in ['true', '1', 'yes'] if len(sys.argv) > 3 else True

    if not Path(csv_file).exists():
        print(f"❌ Error: File does not exist: {csv_file}")
        sys.exit(1)

    result = debug_crm_payload(csv_file, row_index, include_suitability)

    if result and not result['issues']:
        print(f"\n🎉 Payload looks good for Monday.com API!")
    else:
        print(f"\n⚠️  Review issues before sending to CRM API")


if __name__ == "__main__":
    main()