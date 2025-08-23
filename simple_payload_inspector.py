# Create a small test script
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env', override=True)

from services.crm_service import CRMService

# Load just 1 record
df = pd.read_csv(r"C:\Users\georg\Downloads\OH_Cuyahoga_Parcel_Files_Cuyahoga_OH_parcels_08182025_1815_cleaned.csv")
test_parcel = df.iloc[0].to_dict()

# Initialize CRM
crm = CRMService()

# Create group and import 1 parcel
group_name = "Test Import - Single Parcel"
group_id = crm.create_group_in_board(group_name)

if group_id:
    crm_values = crm.prepare_parcel_for_crm(test_parcel, 'solar')
    owner_name = crm.proper_case_with_exceptions(test_parcel.get('owner'))

    print(f"Creating item with {len(crm_values)} fields...")
    success = crm.create_crm_item(group_id, owner_name, crm_values)

    if success:
        print("✅ SUCCESS! Check Monday.com for all 21 fields!")
    else:
        print("❌ Import failed - check logs")