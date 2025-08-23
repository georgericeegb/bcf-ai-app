#!/usr/bin/env python3
"""
Test CRM Fixes Script
Run this to validate your CRM service fixes before full import
"""

import json
import pandas as pd
from pathlib import Path
import sys
import os

# Add the current directory to Python path to import your fixed CRM service
sys.path.append('.')


def test_crm_fixes():
    """Test the fixed CRM service"""

    print("🧪 TESTING CRM SERVICE FIXES")
    print("=" * 50)

    # Load environment variables from .env file (like your app.py does)
    try:
        from dotenv import load_dotenv
        load_dotenv('.env', override=True)
        print("✅ Loaded environment variables from .env file")

        # Check if the required variables are present
        api_key = os.getenv('MONDAY_API_KEY')
        board_id = os.getenv('MONDAY_BOARD_ID')

        if api_key:
            print(f"✅ Monday API Key found: {api_key[:15]}...")
        else:
            print("⚠️  MONDAY_API_KEY not found in .env file")

        if board_id:
            print(f"✅ Monday Board ID found: {board_id}")
        else:
            print("⚠️  MONDAY_BOARD_ID not found in .env file")

    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    except Exception as e:
        print(f"⚠️  Error loading .env file: {e}")

    # Try to import the fixed CRM service
    try:
        from services.crm_service import CRMService  # Updated import path to match your structure
        print("✅ Successfully imported fixed CRMService")
    except ImportError as e:
        print(f"❌ Could not import CRMService: {e}")
        print("Make sure the fixed crm_service.py is in the services/ directory")
        return False

    # Test initialization
    try:
        crm = CRMService()
        print("✅ CRMService initialized successfully")

        # Test connection to Monday.com
        print("🔗 Testing Monday.com API connection...")
        connection_result = crm.test_connection()

        if connection_result['success']:
            user_info = connection_result.get('user', {})
            print(
                f"✅ Connected to Monday.com as: {user_info.get('name', 'Unknown')} ({user_info.get('email', 'Unknown')})")
        else:
            print(f"❌ Monday.com connection failed: {connection_result.get('error', 'Unknown error')}")
            print("Continuing with offline data processing tests...")

    except Exception as e:
        print(f"⚠️  CRMService initialization issue: {e}")
        print("Continuing with data processing tests...")

        # Create a mock CRM service for testing data processing
        class MockCRMService:
            def __init__(self):
                # Copy the field mappings and methods from the real class
                from crm_service import CRMService
                real_crm = CRMService.__new__(CRMService)
                real_crm.__init__ = lambda: None
                for attr in ['field_variations', 'crm_field_mapping', 'safe_convert_to_string',
                             'is_valid_value', 'find_field_value', 'safe_format_number',
                             'format_field_value', 'prepare_parcel_for_crm', 'proper_case_with_exceptions']:
                    if hasattr(CRMService, attr):
                        setattr(self, attr, getattr(CRMService, attr))
                # Initialize the field mappings manually
                self._init_field_mappings()

            def _init_field_mappings(self):
                self.field_variations = {
                    'owner': ['owner', 'owner_name', 'owner1'],
                    'county_id': ['county_id', 'cnty_id', 'fips'],
                    'county_name': ['county_name', 'county_nam', 'county'],
                    'state_abbr': ['state_abbr', 'state', 'st'],
                    'address': ['address', 'mail_address', 'address1'],
                    'parcel_id': ['parcel_id', 'pin', 'apn', 'id'],
                    'acreage_calc': ['acreage_calc', 'acreage', 'acres'],
                    'latitude': ['latitude', 'lat', 'y'],
                    'longitude': ['longitude', 'long', 'lon', 'x'],
                    'mkt_val_land': ['mkt_val_land', 'market_value_land'],
                    'census_zip': ['census_zip', 'zip', 'zipcode'],
                    'elevation': ['elevation', 'elev']
                }

                self.crm_field_mapping = {
                    'owner': 'item_name',
                    'county_id': 'county_id__1',
                    'county_name': 'text4',
                    'state_abbr': 'text_1',
                    'address': 'text1',
                    'parcel_id': 'text117',
                    'acreage_calc': 'numbers6',
                    'latitude': 'latitude__1',
                    'longitude': 'longitude__1',
                    'mkt_val_land': 'numbers85__1',
                    'census_zip': 'text_mktw4254',
                    'elevation': 'numeric_mktwrwry'
                }

        try:
            crm = MockCRMService()
            print("✅ Created mock CRM service for testing")
        except Exception as e2:
            print(f"❌ Could not create mock CRM service: {e2}")
            return False

    print("\n🧪 TESTING DATA PROCESSING...")

    # Create test parcel data that matches your CSV structure
    test_parcels = [
        {
            'parcel_id': '2936015',
            'owner': 'CLEVELAND CITY',
            'county_id': '39035',
            'county_name': 'Cuyahoga',
            'state_abbr': 'OH',
            'address': '5300 RIVERSIDE Rd',
            'acreage': 526.33,
            'acreage_calc': 526.01,
            'latitude': 41.4032357967894,
            'longitude': -81.8607025742243,
            'elevation': 768.73359579745,
            'mkt_val_land': 0.0,
            'census_zip': '44142'
        },
        {
            # Test parcel with problematic data
            'parcel_id': '1234567',
            'owner': 'TEST OWNER LLC',
            'county_id': 'nan',  # Problematic value
            'county_name': '',  # Empty string
            'state_abbr': 'OH',
            'address': None,  # Null value
            'acreage': 0,  # Zero value
            'acreage_calc': '150.5',  # String number
            'latitude': 0.0,  # Zero coordinate (invalid)
            'longitude': -81.5,
            'elevation': 'nan',  # String nan
            'mkt_val_land': '',  # Empty string
            'census_zip': 0  # Zero zip
        },
        {
            # Test parcel with mixed data types
            'parcel_id': 'ABC123',
            'owner': 'smith, john & jane',
            'county_id': 12345,  # Numeric county ID
            'county_name': 'Test County',
            'state_abbr': 'ohio',  # Lowercase state
            'address': '123 Main St',
            'acreage': 75.25,
            'latitude': 40.123456,
            'longitude': -80.987654,
            'elevation': 1200,
            'mkt_val_land': 50000,
            'census_zip': '44120'  # String zip
        }
    ]

    print(f"Testing with {len(test_parcels)} sample parcels...")

    success_count = 0
    total_fields_mapped = 0

    for i, parcel in enumerate(test_parcels, 1):
        print(f"\n--- Testing Parcel {i}: {parcel.get('parcel_id')} ---")

        try:
            # Test the data processing
            crm_values = crm.prepare_parcel_for_crm(parcel, 'solar')

            # Test owner name formatting
            owner_name = crm.proper_case_with_exceptions(parcel.get('owner', 'Unknown'))

            print(f"✅ Processed successfully")
            print(f"   Owner: {owner_name}")
            print(f"   CRM Fields: {len(crm_values)}")
            print(f"   Fields: {list(crm_values.keys())}")

            # Show some formatted values
            if crm_values:
                print("   Sample values:")
                for key, value in list(crm_values.items())[:5]:  # Show first 5
                    print(f"     {key}: {repr(value)}")

            success_count += 1
            total_fields_mapped += len(crm_values)

        except Exception as e:
            print(f"❌ Error processing parcel {i}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")

    print(f"\n📊 TEST RESULTS:")
    print(f"   Successfully processed: {success_count}/{len(test_parcels)} parcels")
    print(f"   Average fields per parcel: {total_fields_mapped / success_count if success_count > 0 else 0:.1f}")

    if success_count == len(test_parcels):
        print("✅ All tests passed! Your fixes look good.")
        print("\n💡 Next steps:")
        print("   1. Run the diagnostic script on your CSV file")
        print("   2. Set your Monday.com environment variables")
        print("   3. Test with a small batch of real data")
        print("   4. Run full import")
        return True
    else:
        print("⚠️  Some tests failed. Review the errors above.")
        return False


def test_with_csv_file(csv_file_path):
    """Test with actual CSV file"""
    print(f"\n🧪 TESTING WITH ACTUAL CSV: {csv_file_path}")
    print("=" * 50)

    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv('.env', override=True)
    except:
        print("⚠️  Could not load .env file for CSV testing")

    # Test with first 3 rows
    test_rows = df.head(3).to_dict('records')

    try:
        from services.crm_service import CRMService  # Updated import path
        crm = CRMService()
        print("✅ CRM service loaded for CSV testing")
    except Exception as e:
        print(f"⚠️  Could not load CRM service: {e}")
        print("Testing with basic data processing only...")
        return False

    print(f"Testing first {len(test_rows)} rows from CSV...")

    successful_rows = 0
    total_fields = 0

    for i, row in enumerate(test_rows, 1):
        try:
            crm_values = crm.prepare_parcel_for_crm(row, 'solar')
            owner_name = crm.proper_case_with_exceptions(row.get('owner', 'Unknown'))

            print(f"✅ Row {i}: {len(crm_values)} fields mapped")
            print(f"   Parcel: {row.get('parcel_id', 'N/A')}")
            print(f"   Owner: {owner_name}")

            if crm_values:
                print(f"   Sample fields: {list(crm_values.keys())[:5]}")

            successful_rows += 1
            total_fields += len(crm_values)

        except Exception as e:
            print(f"❌ Row {i} failed: {e}")

    if successful_rows > 0:
        avg_fields = total_fields / successful_rows
        print(f"\n📊 CSV Test Results:")
        print(f"   Successful rows: {successful_rows}/{len(test_rows)}")
        print(f"   Average fields per row: {avg_fields:.1f}")

        if successful_rows == len(test_rows):
            print("✅ All CSV rows processed successfully!")
        else:
            print("⚠️  Some CSV rows failed - check data quality")

    return successful_rows > 0


if __name__ == "__main__":
    print("🔧 CRM SERVICE TESTING TOOL")
    print("=" * 50)

    # Run basic tests
    success = test_crm_fixes()

    # If CSV file provided, test with that too
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if Path(csv_file).exists():
            test_with_csv_file(csv_file)
        else:
            print(f"❌ CSV file not found: {csv_file}")
    else:
        print("\n💡 To test with your CSV file, run:")
        print("   python test_crm_fixes.py your_file.csv")

    print(f"\n{'✅ TESTING COMPLETE' if success else '❌ TESTS FAILED'}")