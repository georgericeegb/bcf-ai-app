#!/usr/bin/env python3
"""
CSV Data Diagnostic Script
Run this to identify data quality issues before CRM upload
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def diagnose_csv_data(csv_file_path):
    """Comprehensive CSV data analysis"""

    print("🔍 LOADING CSV DATA...")
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    print("\n" + "=" * 60)
    print("📊 BASIC DATA ANALYSIS")
    print("=" * 60)

    # Basic stats
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Show all column names
    print(f"\n📋 ALL COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\n" + "=" * 60)
    print("🎯 CRM FIELD MAPPING ANALYSIS")
    print("=" * 60)

    # Define the field mappings from the CRM service
    crm_mappings = {
        'owner': ['owner', 'owner_name', 'owner1'],
        'county_id': ['county_id', 'cnty_id', 'fips', 'county_fips'],
        'county_name': ['county_name', 'county_nam', 'county'],
        'state_abbr': ['state_abbr', 'state', 'st'],
        'address': ['address', 'mail_address', 'address1'],
        'parcel_id': ['parcel_id', 'pin', 'apn', 'id'],
        'acreage_calc': ['acreage_calc', 'acreage', 'acres'],
        'latitude': ['latitude', 'lat', 'y'],
        'longitude': ['longitude', 'long', 'lon', 'x'],
        'mkt_val_land': ['mkt_val_land', 'market_value_land', 'land_value'],
        'census_zip': ['census_zip', 'zip', 'zipcode'],
        'elevation': ['elevation', 'elev']
    }

    mapping_results = {}

    for field_key, variations in crm_mappings.items():
        found_columns = [col for col in variations if col in df.columns]

        if found_columns:
            col_name = found_columns[0]  # Use first match
            mapping_results[field_key] = {
                'found': True,
                'column': col_name,
                'total_values': len(df),
                'null_count': df[col_name].isnull().sum(),
                'unique_values': df[col_name].nunique(),
                'data_type': str(df[col_name].dtype)
            }

            # Analyze the actual values
            sample_values = df[col_name].dropna().head(5).tolist()
            mapping_results[field_key]['sample_values'] = sample_values

            # Check for problematic values
            problematic = 0
            if col_name in df.columns:
                series = df[col_name].astype(str).str.lower()
                problematic = series.isin(['nan', 'null', 'none', '', '0', '0.0']).sum()

            mapping_results[field_key]['problematic_values'] = problematic

            print(f"✅ {field_key:15} -> {col_name:20} | "
                  f"Null: {mapping_results[field_key]['null_count']:4d} | "
                  f"Problems: {problematic:4d} | "
                  f"Type: {mapping_results[field_key]['data_type']:10}")
        else:
            mapping_results[field_key] = {'found': False}
            print(f"❌ {field_key:15} -> NOT FOUND")

    print("\n" + "=" * 60)
    print("🔍 DATA QUALITY ISSUES")
    print("=" * 60)

    issues_found = []

    # Check for coordinate issues
    if 'latitude' in mapping_results and mapping_results['latitude']['found']:
        lat_col = mapping_results['latitude']['column']
        lat_zeros = (df[lat_col] == 0).sum()
        lat_nulls = df[lat_col].isnull().sum()
        if lat_zeros > 0:
            issues_found.append(f"⚠️  {lat_zeros} latitude values are zero")
        if lat_nulls > 0:
            issues_found.append(f"⚠️  {lat_nulls} latitude values are null")

    if 'longitude' in mapping_results and mapping_results['longitude']['found']:
        lon_col = mapping_results['longitude']['column']
        lon_zeros = (df[lon_col] == 0).sum()
        lon_nulls = df[lon_col].isnull().sum()
        if lon_zeros > 0:
            issues_found.append(f"⚠️  {lon_zeros} longitude values are zero")
        if lon_nulls > 0:
            issues_found.append(f"⚠️  {lon_nulls} longitude values are null")

    # Check for acreage issues
    if 'acreage_calc' in mapping_results and mapping_results['acreage_calc']['found']:
        acre_col = mapping_results['acreage_calc']['column']
        acre_zeros = (df[acre_col] == 0).sum()
        acre_negatives = (df[acre_col] < 0).sum()
        if acre_zeros > 0:
            issues_found.append(f"⚠️  {acre_zeros} acreage values are zero")
        if acre_negatives > 0:
            issues_found.append(f"⚠️  {acre_negatives} acreage values are negative")

    # Check for string "nan" values
    string_nan_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            nan_count = df[col].astype(str).str.lower().eq('nan').sum()
            if nan_count > 0:
                string_nan_cols.append(f"{col}({nan_count})")

    if string_nan_cols:
        issues_found.append(f"⚠️  String 'nan' values in: {', '.join(string_nan_cols)}")

    # Check for mixed data types
    mixed_type_issues = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains both numbers and text
            sample = df[col].dropna().astype(str).head(100)
            numeric_count = sum(1 for x in sample if x.replace('.', '').replace('-', '').isdigit())
            text_count = len(sample) - numeric_count
            if numeric_count > 0 and text_count > 0:
                mixed_type_issues.append(col)

    if mixed_type_issues:
        issues_found.append(f"⚠️  Mixed data types in: {', '.join(mixed_type_issues)}")

    if not issues_found:
        print("✅ No major data quality issues detected!")
    else:
        for issue in issues_found:
            print(issue)

    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)

    recommendations = []

    # Missing critical fields
    critical_missing = [k for k, v in mapping_results.items()
                        if k in ['parcel_id', 'owner', 'latitude', 'longitude'] and not v['found']]

    if critical_missing:
        recommendations.append(f"❗ Add missing critical fields: {', '.join(critical_missing)}")

    # Data cleaning recommendations
    if any('String \'nan\'' in issue for issue in issues_found):
        recommendations.append("🧹 Replace string 'nan' values with actual nulls: df.replace('nan', pd.NA)")

    if any('zero' in issue.lower() for issue in issues_found):
        recommendations.append("🧹 Review zero values - some may be valid, others should be null")

    if mixed_type_issues:
        recommendations.append("🧹 Convert mixed-type columns to consistent data types")

    # Success prediction
    mappable_count = sum(1 for v in mapping_results.values() if v.get('found', False))
    total_fields = len(mapping_results)
    success_rate = (mappable_count / total_fields) * 100

    recommendations.append(
        f"📊 Estimated CRM success rate: {success_rate:.1f}% ({mappable_count}/{total_fields} fields)")

    if success_rate >= 80:
        recommendations.append("✅ Data looks good for CRM import!")
    elif success_rate >= 60:
        recommendations.append("⚠️  Consider data cleaning before import")
    else:
        recommendations.append("❌ Significant data quality issues - clean before import")

    for rec in recommendations:
        print(rec)

    print("\n" + "=" * 60)
    print("🔧 SAMPLE RECORDS FOR TESTING")
    print("=" * 60)

    # Show a few sample records with key fields
    key_fields = [v['column'] for v in mapping_results.values() if v.get('found', False)][:8]
    if key_fields:
        print("\nSample of your data:")
        sample_df = df[key_fields].head(3)
        print(sample_df.to_string(index=False))

    # Export sample for testing
    sample_file = csv_file_path.replace('.csv', '_sample_for_testing.json')
    if len(df) > 0:
        sample_record = df.iloc[0].to_dict()
        with open(sample_file, 'w') as f:
            json.dump(sample_record, f, indent=2, default=str)
        print(f"\n💾 Sample record saved to: {sample_file}")

    print(f"\n🎯 Analysis complete! Run the fixed CRM service to test import.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python diagnose_csv.py <path_to_csv_file>")
        print("Example: python diagnose_csv.py OH_Cuyahoga_Parcel_Files_Cuyahoga_OH_parcels_08182025_1815_cleaned.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    diagnose_csv_data(csv_file)