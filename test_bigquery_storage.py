#!/usr/bin/env python3
"""
Isolated test script for BigQuery transmission storage
Debug the "name 'row' is not defined" error
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_parcel_data():
    """Create sample parcel data for testing"""

    # Create sample parcels with transmission data
    parcels_data = []

    for i in range(5):
        parcel = {
            'parcel_id': f'test_parcel_{i}',
            'owner': f'Test Owner {i}',
            'acreage': 50 + (i * 10),
            'avg_slope_degrees': 10 + i,
            'tx_nearest_distance': 1.5 + (i * 0.5),  # Miles
            'tx_max_voltage': 138 + (i * 50),  # kV
            'tx_lines_count': 1 + i,
            'geometry': Point(-80.5 - (i * 0.01), 36.2 + (i * 0.01))  # Sample coordinates
        }
        parcels_data.append(parcel)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(parcels_data, geometry='geometry')
    gdf = gdf.set_crs('EPSG:4326')

    logger.info(f"Created test data with {len(gdf)} parcels")
    return gdf


def test_suitability_calculation(parcel_data):
    """Test the suitability calculation that's failing"""

    logger.info("Testing suitability calculation...")

    try:
        # This mimics the calculation from bigquery_transmission_storage.py
        def calculate_simple_suitability_fixed(parcel_row):
            """Fixed version of the suitability calculation"""
            try:
                # Convert all values to float with proper error handling
                def safe_float(value, default=0.0):
                    if pd.isna(value) or value == '' or value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                # Extract and convert values safely from the parcel row
                acreage = safe_float(parcel_row.get('acreage', 0))
                slope = safe_float(parcel_row.get('avg_slope_degrees', 15.0))
                tx_distance = safe_float(parcel_row.get('tx_nearest_distance', 999.0))
                tx_voltage = safe_float(parcel_row.get('tx_max_voltage', 0.0))

                # Handle alternative column names
                if acreage == 0:
                    acreage = safe_float(parcel_row.get('acreage_calc', 0))
                if slope == 15.0:
                    slope = safe_float(parcel_row.get('avg_slope', 15.0))
                if tx_distance == 999.0:
                    tx_distance = safe_float(parcel_row.get('tx_distance_miles', 999.0))
                if tx_voltage == 0.0:
                    tx_voltage = safe_float(parcel_row.get('tx_voltage_kv', 0.0))

                # Calculate scores (same logic as before)
                score = 50

                # Acreage scoring
                if acreage >= 100:
                    score += 20
                elif acreage >= 50:
                    score += 15
                elif acreage >= 20:
                    score += 10
                elif acreage >= 10:
                    score += 5

                # Slope scoring
                if slope <= 5:
                    score += 20
                elif slope <= 15:
                    score += 10
                elif slope <= 25:
                    score += 5

                # Transmission scoring
                if tx_distance <= 1.0:
                    score += 15
                elif tx_distance <= 2.0:
                    score += 10

                if tx_voltage >= 138:
                    score += 5

                final_score = max(0, min(100, score))

                if final_score >= 80:
                    return final_score, 'Excellent', True
                elif final_score >= 65:
                    return final_score, 'Good', True
                elif final_score >= 50:
                    return final_score, 'Fair', False
                else:
                    return final_score, 'Poor', False

            except Exception as e:
                logger.error(f"Suitability calculation error: {e}")
                return 50, 'Fair', False

        # Test calculation on each parcel
        results = []
        for idx, parcel in parcel_data.iterrows():
            try:
                score, category, recommended = calculate_simple_suitability_fixed(parcel)
                result = {
                    'parcel_id': parcel.get('parcel_id'),
                    'score': score,
                    'category': category,
                    'recommended': recommended
                }
                results.append(result)
                logger.info(f"Parcel {parcel.get('parcel_id')}: Score={score}, Category={category}")

            except Exception as e:
                logger.error(f"Error processing parcel {idx}: {e}")

        return results

    except Exception as e:
        logger.error(f"Test suitability calculation failed: {e}")
        return []


def test_bigquery_storage_preparation(parcel_data):
    """Test BigQuery record preparation"""

    logger.info("Testing BigQuery record preparation...")

    analysis_metadata = {
        'state': 'NC',
        'county': 'Watauga',
        'source_file': 'test_file.csv',
        'project_type': 'solar'
    }

    try:
        # Simulate the record preparation from TransmissionAnalysisBQ
        records = []

        for idx, parcel in parcel_data.iterrows():

            # Extract geometry coordinates
            if hasattr(parcel.geometry, 'centroid'):
                centroid = parcel.geometry.centroid
                lat, lon = centroid.y, centroid.x
            else:
                lat, lon = None, None

            # Clean transmission data - handle various field names
            def safe_extract(parcel_data, field_names, default_value):
                """Safely extract value from parcel using multiple possible field names"""
                for field in field_names:
                    if field in parcel_data and pd.notna(parcel_data[field]):
                        return parcel_data[field]
                return default_value

            tx_lines_count = safe_extract(parcel, ['tx_lines_count', 'transmission_lines_count'], 0)
            tx_distance = safe_extract(parcel, ['tx_nearest_distance', 'tx_distance_miles', 'transmission_distance'],
                                       None)
            tx_voltage = safe_extract(parcel, ['tx_max_voltage', 'tx_voltage_kv', 'transmission_voltage'], None)
            tx_owner = safe_extract(parcel, ['tx_closest_owner', 'tx_primary_owner', 'transmission_owner'], 'Unknown')

            # Calculate suitability scoring - FIXED to pass the right parameter
            suitability_score, suitability_category, recommended = test_suitability_calculation_single(parcel)

            record = {
                # Analysis metadata
                'analysis_id': 'test_analysis_123',
                'state': analysis_metadata.get('state', 'Unknown'),
                'county': analysis_metadata.get('county', 'Unknown'),
                'source_file': analysis_metadata.get('source_file', ''),

                # Parcel identification
                'parcel_id': str(parcel.get('parcel_id', f'parcel_{idx}')),
                'owner': str(parcel.get('owner', 'Unknown'))[:100],
                'parcel_index': int(idx),

                # Parcel characteristics
                'acreage': float(safe_extract(parcel, ['acreage', 'acres'], 0)),
                'slope_degrees': float(safe_extract(parcel, ['slope_degrees', 'avg_slope_degrees'], 15)),
                'latitude': float(lat) if lat else None,
                'longitude': float(lon) if lon else None,

                # Clean transmission results
                'tx_lines_count': int(tx_lines_count),
                'tx_nearest_distance_miles': float(tx_distance) if tx_distance and tx_distance < 900 else None,
                'tx_max_voltage_kv': float(tx_voltage) if tx_voltage and tx_voltage > 0 else None,
                'tx_closest_owner': str(tx_owner)[:50],

                # Suitability
                'suitability_score': int(suitability_score),
                'suitability_category': str(suitability_category),
                'recommended_for_outreach': bool(recommended),

                # Analysis status
                'has_transmission_data': bool(tx_lines_count > 0),
                'analysis_quality': 'COMPLETE' if tx_lines_count > 0 else 'NO_TRANSMISSION_FOUND'
            }

            records.append(record)
            logger.info(f"Prepared record for parcel: {record['parcel_id']}")

        logger.info(f"Successfully prepared {len(records)} records for BigQuery")
        return records

    except Exception as e:
        logger.error(f"BigQuery preparation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def test_suitability_calculation_single(parcel_row):
    """Test suitability calculation for a single parcel"""
    try:
        def safe_float(value, default=0.0):
            if pd.isna(value) or value == '' or value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        # Extract values
        acreage = safe_float(parcel_row.get('acreage', 0))
        slope = safe_float(parcel_row.get('avg_slope_degrees', 15.0))
        tx_distance = safe_float(parcel_row.get('tx_nearest_distance', 999.0))
        tx_voltage = safe_float(parcel_row.get('tx_max_voltage', 0.0))

        # Calculate score
        score = 50

        if acreage >= 50:
            score += 15
        if slope <= 15:
            score += 10
        if tx_distance <= 2.0:
            score += 10
        if tx_voltage >= 138:
            score += 5

        final_score = max(0, min(100, score))

        if final_score >= 80:
            return final_score, 'Excellent', True
        elif final_score >= 65:
            return final_score, 'Good', True
        elif final_score >= 50:
            return final_score, 'Fair', False
        else:
            return final_score, 'Poor', False

    except Exception as e:
        logger.error(f"Single suitability calculation error: {e}")
        return 50, 'Fair', False


def main():
    """Main test function"""
    logger.info("Starting BigQuery storage isolation test...")

    try:
        # Create test data
        test_parcels = create_test_parcel_data()

        # Test suitability calculation
        suitability_results = test_suitability_calculation(test_parcels)
        logger.info(f"Suitability test completed: {len(suitability_results)} results")

        # Test BigQuery record preparation
        bq_records = test_bigquery_storage_preparation(test_parcels)
        logger.info(f"BigQuery preparation test completed: {len(bq_records)} records")

        # Print summary
        if bq_records:
            logger.info("Test completed successfully!")
            for record in bq_records:
                logger.info(
                    f"Record: {record['parcel_id']} - Score: {record['suitability_score']} - Category: {record['suitability_category']}")
        else:
            logger.error("Test failed - no records created")

    except Exception as e:
        logger.error(f"Main test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()