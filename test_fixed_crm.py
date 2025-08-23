# test_fixed_crm.py - Test the fixed CRM service

import pandas as pd
import os
import sys
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

try:
    from services.crm_service import CRMService
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_fixed_validation():
    """Test the fixed CRM service validation and field mapping"""

    print("🧪 TESTING FIXED CRM SERVICE")
    print("=" * 60)

    # Load CSV data
    csv_file_path = "C:/Users/georg/Downloads/OH_Cuyahoga_Parcel_Files_Cuyahoga_OH_parcels_08182025_1815.csv"

    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
        sample_parcel = df.iloc[0].to_dict()

        print(f"📊 Testing with parcel: {sample_parcel.get('parcel_id', 'Unknown')}")
        print()

        # Initialize CRM service
        crm_service = CRMService()

        # Test the old vs new processing
        print("🔍 BEFORE/AFTER COMPARISON:")
        print("-" * 60)

        # Process the parcel with new logic
        crm_values = crm_service.prepare_parcel_for_crm(sample_parcel, 'solar')

        print(f"📋 Fields successfully mapped: {len(crm_values)}")
        print()

        # Show what improved
        print("✅ SUCCESSFULLY MAPPED FIELDS:")
        for field_key, monday_field in crm_service.crm_field_mapping.items():
            if field_key == 'owner':
                continue
            if monday_field in crm_values:
                value = crm_values[monday_field]
                print(f"   {field_key:30} -> {monday_field:20} = {repr(value)}")

        print()
        print("❌ STILL MISSING FIELDS:")
        missing_count = 0
        for field_key, monday_field in crm_service.crm_field_mapping.items():
            if field_key == 'owner':
                continue
            if monday_field not in crm_values:
                missing_count += 1
                # Try to understand why it's missing
                raw_value = crm_service.find_field_value(sample_parcel, field_key)
                if raw_value is not None:
                    print(f"   {field_key:30} -> Found: {repr(raw_value)} (formatting failed)")
                else:
                    print(f"   {field_key:30} -> No field found in CSV")

        print()
        print("📊 IMPROVEMENT SUMMARY:")
        total_fields = len(crm_service.crm_field_mapping) - 1  # Exclude owner
        success_rate = (len(crm_values) / total_fields) * 100

        print(f"   Total CRM fields: {total_fields}")
        print(f"   Successfully mapped: {len(crm_values)}")
        print(f"   Still missing: {missing_count}")
        print(f"   Success rate: {success_rate:.1f}%")

        if success_rate >= 70:
            print("   ✅ Good mapping rate - ready for CRM export!")
        elif success_rate >= 50:
            print("   ⚠️  Moderate - some improvement but more fixes needed")
        else:
            print("   ❌ Still needs work")

        # Test with sample analysis data
        print()
        print("🧪 TESTING WITH ANALYSIS DATA:")
        print("-" * 60)

        # Add sample suitability analysis
        sample_parcel_with_analysis = sample_parcel.copy()
        sample_parcel_with_analysis['suitability_analysis'] = {
            'overall_score': 85.5,
            'slope_score': 90.0,
            'transmission_score': 80.0,
            'slope_degrees': 3.2,
            'transmission_distance': 1.5,
            'transmission_voltage': 138000
        }

        crm_values_with_analysis = crm_service.prepare_parcel_for_crm(sample_parcel_with_analysis, 'solar')

        analysis_fields = ['solar_score', 'wind_score', 'battery_score', 'avg_slope_degrees', 'miles_from_transmission',
                           'nearest_transmission_voltage']
        analysis_mapped = 0

        for field_key in analysis_fields:
            monday_field = crm_service.crm_field_mapping.get(field_key)
            if monday_field and monday_field in crm_values_with_analysis:
                analysis_mapped += 1
                value = crm_values_with_analysis[monday_field]
                print(f"   ✅ {field_key:30} -> {value}")
            else:
                print(f"   ❌ {field_key:30} -> Not mapped")

        print(f"\n   Analysis fields mapped: {analysis_mapped}/{len(analysis_fields)}")
        print(f"   Total fields with analysis: {len(crm_values_with_analysis)}")

        improvement = len(crm_values_with_analysis) - len(crm_values)
        if improvement > 0:
            print(f"   ✅ Analysis added {improvement} more fields!")

        print()
        print("🎯 NEXT STEPS:")
        print("-" * 60)
        if len(crm_values) >= 15:
            print("1. ✅ Your CRM integration is working well!")
            print("2. 🚀 Try a real export with a small number of parcels")
            print("3. 📊 Check Monday.com to verify all fields populate correctly")
            print("4. 🔧 Fine-tune any remaining field mappings if needed")
        else:
            print("1. 🔍 Review the 'STILL MISSING FIELDS' section above")
            print("2. 🛠️  Add more field variations or fix data formatting")
            print("3. 🧪 Test with cleaned CSV data")
            print("4. 📞 Reach out for help with specific field issues")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_validation()