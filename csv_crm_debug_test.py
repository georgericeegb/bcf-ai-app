# csv_crm_debug_test.py - Comprehensive debug script for CSV to CRM issues

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import your CRM service
try:
    from services.crm_service import CRMService

    print("✅ CRM Service imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CRM Service: {e}")
    sys.exit(1)


def load_and_analyze_csv(csv_file_path):
    """Load and analyze the CSV file"""
    print("📁 Loading CSV file...")

    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")

        # Show column names
        print("\n📋 CSV Columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")

        # Show first row data
        print("\n🔍 First row data:")
        first_row = df.iloc[0].to_dict()
        for key, value in first_row.items():
            print(f"  {key}: {repr(value)}")

        return df

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def test_crm_connection():
    """Test the CRM connection"""
    print("\n🔗 Testing CRM connection...")

    try:
        crm_service = CRMService()
        result = crm_service.test_connection()

        if result['success']:
            print("✅ CRM connection successful")
            print(f"👤 User: {result['user']['name']} ({result['user']['email']})")
            return crm_service
        else:
            print(f"❌ CRM connection failed: {result['error']}")
            return None

    except Exception as e:
        print(f"❌ CRM connection error: {e}")
        return None


def test_field_mapping(crm_service, csv_data):
    """Test field mapping for the first CSV row"""
    print("\n🧪 Testing field mapping...")

    # Get first row as dict
    test_parcel = csv_data.iloc[0].to_dict()

    print(f"\n🏠 Testing parcel: {test_parcel.get('parcel_id', 'Unknown')}")
    print(f"👤 Owner: {test_parcel.get('owner', 'Unknown')}")

    # Test the field mapping
    debug_info = crm_service.debug_field_mapping(test_parcel)

    print(f"\n📊 Field Mapping Results:")
    print(f"  Total CSV fields: {debug_info['total_fields_available']}")
    print(f"  Mappable fields: {debug_info['mappable_fields']}")
    print(f"  Unmappable fields: {len(debug_info['unmappable_fields'])}")

    # Show successful mappings
    print(f"\n✅ Successfully Mapped Fields:")
    success_count = 0
    for field_key, analysis in debug_info['field_analysis'].items():
        if analysis['mapping_success']:
            success_count += 1
            print(f"  {success_count:2d}. {field_key} -> {analysis['monday_field']}")
            print(f"      Found in: {analysis['found_in_fields']}")
            print(f"      Value: {repr(analysis['formatted_value'])}")

    # Show failed mappings
    print(f"\n❌ Failed Mappings:")
    fail_count = 0
    for field_key, analysis in debug_info['field_analysis'].items():
        if not analysis['mapping_success']:
            fail_count += 1
            print(f"  {fail_count:2d}. {field_key} -> {analysis['monday_field']}")
            print(
                f"      Checked: {analysis['variations_checked'][:3]}... ({len(analysis['variations_checked'])} total)")
            print(f"      Found fields: {analysis['found_in_fields']}")
            print(f"      Raw value: {repr(analysis['raw_value'])}")

    # Test the prepare_parcel_for_crm method
    print(f"\n🔧 Testing parcel preparation for CRM...")
    crm_values = crm_service.prepare_parcel_for_crm(test_parcel, 'solar')

    print(f"📤 CRM-ready values ({len(crm_values)} fields):")
    for monday_field, value in crm_values.items():
        print(f"  {monday_field}: {repr(value)}")

    return debug_info, crm_values


def analyze_csv_field_matches(csv_data, crm_service):
    """Analyze which CSV fields match CRM field variations"""
    print("\n🔍 Analyzing CSV field matches with CRM variations...")

    csv_columns = set(csv_data.columns)

    print(f"\nField Variation Analysis:")
    for crm_field, variations in crm_service.field_variations.items():
        found_variations = [var for var in variations if var in csv_columns]

        if found_variations:
            print(f"✅ {crm_field}:")
            for var in found_variations:
                sample_value = csv_data[var].iloc[0]
                print(f"    '{var}' = {repr(sample_value)}")
        else:
            print(f"❌ {crm_field}: No matches found")
            print(f"    Looked for: {variations[:3]}... ({len(variations)} total)")


def identify_specific_issues(csv_data):
    """Identify specific data quality issues"""
    print("\n🔍 Identifying data quality issues...")

    issues = []

    # Check for zero coordinates
    if 'latitude' in csv_data.columns and 'longitude' in csv_data.columns:
        zero_coords = csv_data[(csv_data['latitude'] == 0) | (csv_data['longitude'] == 0)]
        if len(zero_coords) > 0:
            issues.append(f"Zero coordinates: {len(zero_coords)} records")

    # Check for missing owner names
    if 'owner' in csv_data.columns:
        missing_owners = csv_data[csv_data['owner'].isna() | (csv_data['owner'] == '')]
        if len(missing_owners) > 0:
            issues.append(f"Missing owners: {len(missing_owners)} records")

    # Check for zero acreage
    acreage_cols = [col for col in csv_data.columns if 'acre' in col.lower()]
    for col in acreage_cols:
        zero_acreage = csv_data[csv_data[col] == 0]
        if len(zero_acreage) > 0:
            issues.append(f"Zero {col}: {len(zero_acreage)} records")

    # Check for null values in key fields
    key_fields = ['parcel_id', 'owner', 'county_id', 'latitude', 'longitude']
    for field in key_fields:
        if field in csv_data.columns:
            null_count = csv_data[field].isna().sum()
            if null_count > 0:
                issues.append(f"Null {field}: {null_count} records")

    if issues:
        print("📋 Data Quality Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ No major data quality issues found")

    return issues


def generate_fix_recommendations(debug_info, issues):
    """Generate specific fix recommendations"""
    print("\n💡 Fix Recommendations:")

    recommendations = []

    # Field mapping recommendations
    failed_fields = [field for field, analysis in debug_info['field_analysis'].items()
                     if not analysis['mapping_success']]

    if failed_fields:
        print(f"\n🔧 Field Mapping Fixes:")
        for i, field in enumerate(failed_fields, 1):
            analysis = debug_info['field_analysis'][field]
            print(f"  {i}. {field}:")

            if not analysis['found_in_fields']:
                print(f"     Problem: No matching CSV column found")
                print(f"     Fix: Add '{field}' variations to field_variations in crm_service.py")
                recommendations.append(f"Add field variations for {field}")
            else:
                print(f"     Problem: Field found but value invalid: {repr(analysis['raw_value'])}")
                print(f"     Fix: Update validation logic for {field}")
                recommendations.append(f"Fix validation for {field}")

    # Data quality recommendations
    if issues:
        print(f"\n🔧 Data Quality Fixes:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
            if "Zero coordinates" in issue:
                recommendations.append("Clean coordinate data - remove or fix zero values")
            elif "Missing owners" in issue:
                recommendations.append("Fill in missing owner names or mark as 'Unknown Owner'")
            elif "Zero" in issue and "acre" in issue:
                recommendations.append("Verify acreage calculations - zero acres may be invalid")

    return recommendations


def main():
    """Main test function"""
    print("🚀 CSV to CRM Import Debug Tool")
    print("=" * 50)

    # Check if CSV file exists
    csv_file = "C:/Users/georg/Downloads/OH_Cuyahoga_Parcel_Files_Cuyahoga_OH_parcels_08182025_1815_cleaned.csv"
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Please ensure the CSV file is in the current directory.")
        return

    # Load CSV data
    csv_data = load_and_analyze_csv(csv_file)
    if csv_data is None:
        return

    # Test CRM connection
    crm_service = test_crm_connection()
    if crm_service is None:
        print("\n❌ Cannot proceed without CRM connection")
        print("Please check your .env file has correct MONDAY_API_KEY and MONDAY_BOARD_ID")
        return

    # Test field mapping
    debug_info, crm_values = test_field_mapping(crm_service, csv_data)

    # Analyze field matches
    analyze_csv_field_matches(csv_data, crm_service)

    # Identify issues
    issues = identify_specific_issues(csv_data)

    # Generate recommendations
    recommendations = generate_fix_recommendations(debug_info, issues)

    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"CSV Records: {len(csv_data)}")
    print(f"Mappable Fields: {debug_info['mappable_fields']}")
    print(f"CRM Values Generated: {len(crm_values)}")
    print(f"Data Quality Issues: {len(issues)}")
    print(f"Recommendations: {len(recommendations)}")

    if debug_info['mappable_fields'] >= 8:  # Minimum viable mappings
        print("\n✅ Field mapping looks good - ready for CRM import")
    else:
        print("\n❌ Field mapping needs fixes before CRM import")

    print("\nNext steps:")
    print("1. Review the failed mappings above")
    print("2. Update the field_variations in crm_service.py")
    print("3. Fix any data quality issues")
    print("4. Re-run this test script")
    print("5. Once all fields map correctly, try the CRM import again")


if __name__ == "__main__":
    main()