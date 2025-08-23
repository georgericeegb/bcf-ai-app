# csv_debug_script.py - Test your CSV field mapping

import pandas as pd
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env file
def load_project_env():
    """Load .env file from project root directory"""
    # Try to find .env file in current directory or parent directories
    current_dir = Path.cwd()

    # Look for .env file in current directory and parent directories (up to 5 levels)
    for i in range(5):
        env_path = current_dir / '.env'
        if env_path.exists():
            print(f"📁 Found .env file at: {env_path}")
            load_dotenv(env_path, override=True)
            return True
        current_dir = current_dir.parent

    # If not found, try common locations
    common_locations = [
        Path('.env'),
        Path('../.env'),
        Path('../../.env'),
        Path.home() / 'BCF_Dev_Land' / '.env'
    ]

    for env_path in common_locations:
        if env_path.exists():
            print(f"📁 Found .env file at: {env_path}")
            load_dotenv(env_path, override=True)
            return True

    return False


# Load environment variables
print("🔧 Loading environment variables...")
if load_project_env():
    print("✅ Successfully loaded .env file")

    # Verify required environment variables
    required_vars = ['MONDAY_API_KEY', 'MONDAY_BOARD_ID']
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show partial value for security
            masked_value = value[:10] + '...' if len(value) > 10 else value
            print(f"   ✅ {var}: {masked_value}")
        else:
            missing_vars.append(var)
            print(f"   ❌ {var}: Not found")

    if missing_vars:
        print(f"⚠️  Warning: Missing required environment variables: {missing_vars}")
        print("   Make sure your .env file contains:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
        print()
else:
    print("❌ Could not find .env file")
    print("   Looked in:")
    print("   - Current directory and parent directories")
    print("   - Common project locations")
    print("   Make sure your .env file exists and contains:")
    print("   MONDAY_API_KEY=your_api_key_here")
    print("   MONDAY_BOARD_ID=your_board_id_here")
    print()

try:
    from services.crm_service import CRMService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're running this script from your project directory")
    print("   and that the services/crm_service.py file exists")
    sys.exit(1)


def debug_csv_field_mapping(csv_file_path):
    """
    Debug CSV field mapping issues
    """
    print("🔍 CSV Field Mapping Debug Tool")
    print("=" * 50)

    try:
        # Read the CSV file
        print(f"📂 Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        print(f"✅ Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"📋 Available columns: {list(df.columns)}")
        print()

        # Take first row as sample
        if len(df) > 0:
            sample_parcel = df.iloc[0].to_dict()
            print(f"🏠 Sample parcel ID: {sample_parcel.get('parcel_id', 'Unknown')}")
            print()

            # Initialize CRM service with better error handling
            try:
                print("🔧 Initializing CRM service...")

                # Check environment variables first
                api_key = os.getenv('MONDAY_API_KEY')
                board_id = os.getenv('MONDAY_BOARD_ID')

                if not api_key:
                    print("❌ MONDAY_API_KEY not found in environment variables")
                    print("   Please check your .env file contains:")
                    print("   MONDAY_API_KEY=your_api_key_here")
                    return

                if not board_id:
                    print("❌ MONDAY_BOARD_ID not found in environment variables")
                    print("   Please check your .env file contains:")
                    print("   MONDAY_BOARD_ID=your_board_id_here")
                    return

                crm_service = CRMService()
                print("✅ CRM Service initialized successfully")

                # Test connection
                print("🔗 Testing Monday.com API connection...")
                connection_test = crm_service.test_connection()

                if connection_test['success']:
                    user_info = connection_test.get('user', {})
                    print(f"✅ Successfully connected to Monday.com")
                    print(f"   User: {user_info.get('name', 'Unknown')}")
                    print(f"   Email: {user_info.get('email', 'Unknown')}")
                else:
                    print(f"❌ Monday.com connection failed: {connection_test['error']}")
                    print("   Please check your API key and try again")
                    return

            except Exception as e:
                print(f"❌ Failed to initialize CRM service: {e}")
                print("   This might be due to:")
                print("   - Missing or invalid MONDAY_API_KEY")
                print("   - Missing or invalid MONDAY_BOARD_ID")
                print("   - Network connectivity issues")
                print("   - Monday.com API issues")
                return

            print()

            # Debug the field mapping
            print("🔧 Analyzing field mapping...")
            debug_info = crm_service.debug_field_mapping(sample_parcel)

            print(f"📊 FIELD MAPPING ANALYSIS")
            print(f"   Total fields available: {debug_info['total_fields_available']}")
            print(f"   Mappable fields: {debug_info['mappable_fields']}")
            print(f"   Unmappable fields: {len(debug_info['unmappable_fields'])}")
            print()

            # Show successful mappings
            print("✅ SUCCESSFUL MAPPINGS:")
            successful_mappings = {k: v for k, v in debug_info['field_analysis'].items()
                                   if v['mapping_success']}

            if successful_mappings:
                for field_key, analysis in successful_mappings.items():
                    print(f"   {field_key:25} -> {analysis['monday_field']:20} = {analysis['formatted_value']}")
            else:
                print("   ❌ No successful mappings found!")

            print()

            # Show failed mappings
            print("❌ FAILED MAPPINGS:")
            failed_mappings = {k: v for k, v in debug_info['field_analysis'].items()
                               if not v['mapping_success']}

            if failed_mappings:
                for field_key, analysis in failed_mappings.items():
                    found_fields = analysis['found_in_fields']
                    if found_fields:
                        raw_val = analysis['raw_value']
                        print(f"   {field_key:25} -> Found '{found_fields[0]}' = {repr(raw_val)} (invalid value)")
                    else:
                        print(f"   {field_key:25} -> No matching field found")
            else:
                print("   ✅ All fields mapped successfully!")

            print()

            # Show data quality issues
            print("🔍 DATA QUALITY ANALYSIS:")
            quality_issues = analyze_data_quality(df)
            for issue, count in quality_issues.items():
                print(f"   {issue}: {count} fields")

            print()

            # Show specific field examples
            print("📝 FIELD VALUE EXAMPLES:")
            for field_key, analysis in debug_info['field_analysis'].items():
                if analysis['found_in_fields']:
                    field_name = analysis['found_in_fields'][0]
                    sample_values = df[field_name].dropna().head(3).tolist()
                    print(f"   {field_name:25} -> {sample_values}")

            # Test CRM preparation
            print()
            print("🧪 TESTING CRM PREPARATION:")
            crm_values = crm_service.prepare_parcel_for_crm(sample_parcel, 'solar')
            print(f"   Generated {len(crm_values)} CRM field values")

            # Show the CRM values
            if crm_values:
                print("   CRM Values Ready for Export:")
                for monday_field, value in crm_values.items():
                    print(f"     {monday_field:25} = {repr(value)}")
            else:
                print("   ❌ No CRM values generated - check data quality issues above")

            # Summary and recommendations
            print()
            print("🎯 SUMMARY & RECOMMENDATIONS:")
            success_rate = (len(successful_mappings) / len(debug_info['field_analysis'])) * 100
            print(f"   Field mapping success rate: {success_rate:.1f}%")

            if success_rate >= 70:
                print("   ✅ Good mapping rate - ready for CRM export")
            elif success_rate >= 50:
                print("   ⚠️  Moderate mapping rate - consider data quality improvements")
            else:
                print("   ❌ Low mapping rate - data quality fixes recommended")

            print(f"   CRM export readiness: {len(crm_values)} fields ready")

            if len(crm_values) >= 10:
                print("   ✅ Sufficient data for meaningful CRM export")
            else:
                print("   ⚠️  Limited data - consider fixing failed mappings")

        else:
            print("❌ CSV file is empty")

    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        import traceback
        traceback.print_exc()


def analyze_data_quality(df):
    """Analyze data quality issues in the DataFrame"""
    issues = {}

    # Check for null/NaN values
    null_counts = df.isnull().sum()
    issues['Fields with null values'] = (null_counts > 0).sum()

    # Check for 'nan' string values
    nan_string_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            nan_strings = df[col].astype(str).str.lower().eq('nan').sum()
            if nan_strings > 0:
                nan_string_count += 1
    issues['Fields with "nan" strings'] = nan_string_count

    # Check for zero values
    zero_count = 0
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  # Numeric columns
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                zero_count += 1
    issues['Numeric fields with zeros'] = zero_count

    # Check for empty strings
    empty_string_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_strings = df[col].astype(str).str.strip().eq('').sum()
            if empty_strings > 0:
                empty_string_count += 1
    issues['Text fields with empty strings'] = empty_string_count

    return issues


def fix_csv_data_quality(csv_file_path, output_path=None):
    """Fix common data quality issues in CSV"""
    print("🔧 FIXING CSV DATA QUALITY ISSUES")
    print("=" * 50)

    try:
        df = pd.read_csv(csv_file_path)
        original_shape = df.shape

        # Fix 1: Replace 'nan' strings with actual NaN
        print("🔄 Replacing 'nan' strings with NaN...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace(['nan', 'NaN', 'NAN', 'null', 'NULL', 'None'], pd.NA)

        # Fix 2: Convert string zeros to numeric where appropriate
        print("🔄 Converting string numbers to numeric...")
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if it looks like numbers
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    # If less than 50% became NaN, it's probably a numeric column
                    if numeric_col.isna().sum() / len(df) < 0.5:
                        df[col] = numeric_col
                except:
                    pass

        # Fix 3: Clean string fields
        print("🔄 Cleaning string fields...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], pd.NA)

        print(f"✅ Data cleaning complete")
        print(f"   Original shape: {original_shape}")
        print(f"   Final shape: {df.shape}")

        # Save cleaned data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"💾 Cleaned data saved to: {output_path}")

        return df

    except Exception as e:
        print(f"❌ Error fixing CSV data: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Use the actual CSV file path from your output
    csv_file_path = "C:/Users/georg/Downloads/OH_Cuyahoga_Parcel_Files_Cuyahoga_OH_parcels_08182025_1815.csv"

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        print("\n🔍 Looking for CSV files in Downloads folder...")

        downloads_dir = Path.home() / "Downloads"
        csv_files = list(downloads_dir.glob("*parcels*.csv"))

        if csv_files:
            print("📁 Found these parcel CSV files:")
            for i, csv_file in enumerate(csv_files, 1):
                print(f"   {i}. {csv_file.name}")

            try:
                choice = input(f"\n🤔 Enter number (1-{len(csv_files)}) or press Enter to skip: ")
                if choice.strip():
                    index = int(choice) - 1
                    if 0 <= index < len(csv_files):
                        csv_file_path = str(csv_files[index])
                        print(f"✅ Using: {csv_file_path}")
                    else:
                        print("❌ Invalid choice")
                        sys.exit(1)
                else:
                    print("❌ No file selected")
                    sys.exit(1)
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input")
                sys.exit(1)
        else:
            print("❌ No parcel CSV files found in Downloads folder")
            print("   Please provide the correct path to your CSV file")
            sys.exit(1)

    print("\n" + "=" * 60)

    # Debug the field mapping
    debug_csv_field_mapping(csv_file_path)

    print("\n" + "=" * 60)
    print("🔧 OPTIONAL DATA QUALITY FIXES")
    print("=" * 60)

    # Ask if user wants to fix data quality issues
    try:
        fix_choice = input("🤔 Do you want to run data quality fixes? (y/N): ").strip().lower()
        if fix_choice in ['y', 'yes']:
            output_path = csv_file_path.replace('.csv', '_cleaned.csv')
            print(f"🔧 Running data quality fixes...")
            cleaned_df = fix_csv_data_quality(csv_file_path, output_path)

            if cleaned_df is not None:
                print(f"\n🎯 Testing field mapping with cleaned data...")
                sample_parcel_cleaned = cleaned_df.iloc[0].to_dict()

                try:
                    crm_service = CRMService()
                    debug_info_cleaned = crm_service.debug_field_mapping(sample_parcel_cleaned)

                    print(f"📊 CLEANED DATA RESULTS:")
                    print(f"   Mappable fields: {debug_info_cleaned['mappable_fields']}")
                    print(
                        f"   Improvement: +{debug_info_cleaned['mappable_fields'] - len([k for k, v in debug_info_cleaned['field_analysis'].items() if v['mapping_success']])} fields")
                except Exception as e:
                    print(f"❌ Error testing cleaned data: {e}")

        else:
            print("⏭️  Skipping data quality fixes")

    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        sys.exit(0)