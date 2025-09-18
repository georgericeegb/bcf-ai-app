#!/usr/bin/env python3
"""
BigQuery-based transmission analysis storage and retrieval system
Eliminates JSON serialization issues and provides clean data flow
"""

import logging
import pandas as pd
import geopandas as gpd
from google.cloud import bigquery
from datetime import datetime
import uuid
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TransmissionAnalysisBQ:
    """Handle transmission analysis results via BigQuery storage"""

    def __init__(self, project_id='bcfparcelsearchrepository'):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = 'transmission_results'
        self.table_id = 'parcel_transmission_analysis'
        self.table_ref = f"{project_id}.{self.dataset_id}.{self.table_id}"

        # Ensure dataset and table exist
        self._ensure_dataset_and_table()

    def _ensure_dataset_and_table(self):
        """Create dataset and table if they don't exist"""
        try:
            # Create dataset if needed
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {dataset_ref} exists")
            except Exception:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                self.client.create_dataset(dataset)
                logger.info(f"Created dataset {dataset_ref}")

            # Create table if needed
            try:
                self.client.get_table(self.table_ref)
                logger.info(f"Table {self.table_ref} exists")
            except Exception:
                schema = self._get_table_schema()
                table = bigquery.Table(self.table_ref, schema=schema)
                self.client.create_table(table)
                logger.info(f"Created table {self.table_ref}")

        except Exception as e:
            logger.error(f"Error ensuring dataset/table: {e}")
            raise

    # Fix 2: Update the _get_table_schema method in bigquery_transmission_storage.py

    def _get_table_schema(self):
        """Define the BigQuery table schema for transmission analysis results"""
        return [
            # Analysis metadata - FIXED: Use STRING for timestamp to avoid serialization issues
            bigquery.SchemaField("analysis_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("analysis_timestamp", "STRING", mode="REQUIRED"),  # CHANGED to STRING
            bigquery.SchemaField("state", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("county", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("source_file", "STRING", mode="NULLABLE"),

            # Parcel identification
            bigquery.SchemaField("parcel_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("owner", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("parcel_index", "INTEGER", mode="NULLABLE"),

            # Parcel characteristics
            bigquery.SchemaField("acreage", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("slope_degrees", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("latitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("longitude", "FLOAT", mode="NULLABLE"),

            # Transmission analysis results
            bigquery.SchemaField("tx_lines_count", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("tx_nearest_distance_miles", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("tx_max_voltage_kv", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("tx_closest_owner", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("tx_closest_voltage_kv", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("tx_proximity_category", "STRING", mode="NULLABLE"),

            # Suitability scoring
            bigquery.SchemaField("suitability_score", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("suitability_category", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("recommended_for_outreach", "BOOLEAN", mode="NULLABLE"),

            # Analysis status
            bigquery.SchemaField("has_transmission_data", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("analysis_quality", "STRING", mode="NULLABLE")
        ]

    def store_transmission_analysis(self, enhanced_parcels_gdf: gpd.GeoDataFrame,
                                    analysis_metadata: Dict[str, Any]) -> str:
        """Store transmission analysis results in BigQuery"""

        analysis_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        logger.info(f"Storing transmission analysis results: {analysis_id}")
        logger.info(f"Processing {len(enhanced_parcels_gdf)} parcels")

        # Convert GeoDataFrame to clean records
        records = []

        for idx, parcel in enhanced_parcels_gdf.iterrows():
            try:
                # Extract geometry coordinates
                if hasattr(parcel.geometry, 'centroid'):
                    centroid = parcel.geometry.centroid
                    lat, lon = centroid.y, centroid.x
                else:
                    lat, lon = None, None

                # CRITICAL FIX: Debug and properly extract transmission data
                tx_distance_raw = self._safe_extract(parcel, ['tx_nearest_distance', 'tx_distance_miles',
                                                              'transmission_distance'], None)
                tx_voltage_raw = self._safe_extract(parcel, ['tx_max_voltage', 'tx_voltage_kv', 'transmission_voltage'],
                                                    None)
                tx_lines_count = self._safe_extract(parcel, ['tx_lines_count', 'transmission_lines_count'], 0)

                # Debug what we're extracting
                logger.info(f"Parcel {idx} transmission data:")
                logger.info(f"  Raw distance: {tx_distance_raw}")
                logger.info(f"  Raw voltage: {tx_voltage_raw}")
                logger.info(f"  Lines count: {tx_lines_count}")

                # Clean and validate transmission data
                tx_distance_clean = None
                tx_voltage_clean = None

                if tx_distance_raw is not None and tx_distance_raw != 999.0:
                    try:
                        tx_distance_clean = float(tx_distance_raw)
                        if tx_distance_clean >= 999:
                            tx_distance_clean = None
                    except (ValueError, TypeError):
                        tx_distance_clean = None

                if tx_voltage_raw is not None and tx_voltage_raw != 0:
                    try:
                        tx_voltage_clean = float(tx_voltage_raw)
                        if tx_voltage_clean <= 0:
                            tx_voltage_clean = None
                    except (ValueError, TypeError):
                        tx_voltage_clean = None

                # Calculate suitability scoring
                suitability_score, suitability_category, recommended = self._calculate_simple_suitability(parcel)

                record = {
                    # Analysis metadata
                    'analysis_id': analysis_id,
                    'analysis_timestamp': timestamp.isoformat(),
                    'state': analysis_metadata.get('state', 'Unknown'),
                    'county': analysis_metadata.get('county', 'Unknown'),
                    'source_file': analysis_metadata.get('source_file', ''),

                    # Parcel identification
                    'parcel_id': str(parcel.get('parcel_id', f'parcel_{idx}')),
                    'owner': str(parcel.get('owner', 'Unknown'))[:100],
                    'parcel_index': int(idx),

                    # Parcel characteristics
                    'acreage': float(self._safe_extract(parcel, ['acreage', 'acres'], 0)),
                    'slope_degrees': float(self._safe_extract(parcel, ['avg_slope_degrees', 'slope'], 15)),
                    'latitude': float(lat) if lat else None,
                    'longitude': float(lon) if lon else None,

                    # CRITICAL FIX: Store the cleaned transmission data
                    'tx_lines_count': int(tx_lines_count),
                    'tx_nearest_distance_miles': tx_distance_clean,
                    'tx_max_voltage_kv': tx_voltage_clean,
                    'tx_closest_owner': str(
                        self._safe_extract(parcel, ['tx_closest_owner', 'tx_primary_owner'], 'Unknown'))[:50],
                    'tx_closest_voltage_kv': tx_voltage_clean,
                    'tx_proximity_category': str(
                        self._safe_extract(parcel, ['tx_proximity_category', 'proximity_category'], 'UNKNOWN')),

                    # Suitability
                    'suitability_score': int(suitability_score),
                    'suitability_category': str(suitability_category),
                    'recommended_for_outreach': bool(recommended),

                    # Analysis status
                    'has_transmission_data': bool(tx_lines_count > 0 or tx_distance_clean is not None),
                    'analysis_quality': 'COMPLETE' if tx_lines_count > 0 else 'NO_TRANSMISSION_FOUND'
                }

                # Debug the record being stored
                logger.info(f"Storing record with tx_nearest_distance_miles: {record['tx_nearest_distance_miles']}")
                logger.info(f"Storing record with tx_max_voltage_kv: {record['tx_max_voltage_kv']}")
                logger.info(f"Sample parcel data being returned: {parcels[0] if parcels else 'No parcels'}")

                records.append(record)

            except Exception as parcel_error:
                logger.error(f"Error processing parcel {idx}: {parcel_error}")
                continue

        # Insert into BigQuery
        try:
            errors = self.client.insert_rows_json(
                self.client.get_table(self.table_ref),
                records
            )

            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                raise Exception(f"Failed to insert records: {errors}")

            logger.info(f"Successfully stored {len(records)} records with analysis_id: {analysis_id}")

            # CRITICAL: Add debug call to verify storage
            self.debug_stored_data(analysis_id)

            return analysis_id

        except Exception as e:
            logger.error(f"Error inserting records to BigQuery: {e}")
            raise

    def datetime_serializer(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_analysis_results(self, analysis_id: str) -> Dict[str, Any]:
        """Retrieve analysis results from BigQuery for UI display"""

        query = f"""
        SELECT 
            analysis_id,
            analysis_timestamp,
            state,
            county,
            source_file,
            parcel_id,
            owner,
            acreage,
            slope_degrees,
            tx_lines_count,
            tx_nearest_distance_miles,
            tx_max_voltage_kv,
            tx_closest_owner,
            tx_proximity_category,
            suitability_score,
            suitability_category,
            recommended_for_outreach,
            has_transmission_data
        FROM `{self.table_ref}`
        WHERE analysis_id = @analysis_id
        ORDER BY suitability_score DESC, tx_nearest_distance_miles ASC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("analysis_id", "STRING", analysis_id)
            ]
        )

        try:
            query_job = self.client.query(query, job_config=job_config)
            results_df = query_job.result().to_dataframe()

            if len(results_df) == 0:
                return {'success': False, 'error': f'No results found for analysis_id: {analysis_id}'}

            # Convert to clean JSON-serializable format
            parcels = []
            for _, result_row in results_df.iterrows():
                # SIMPLE FIX: Just extract the values and let frontend handle them
                tx_distance = result_row['tx_nearest_distance_miles'] if pd.notna(
                    result_row['tx_nearest_distance_miles']) else None
                tx_voltage = result_row['tx_max_voltage_kv'] if pd.notna(result_row['tx_max_voltage_kv']) else None

                parcels.append({
                    'parcel_id': result_row['parcel_id'],
                    'owner': result_row['owner'],
                    'acreage': int(result_row['acreage']) if pd.notna(result_row['acreage']) else 0,
                    'slope_degrees': float(result_row['slope_degrees']) if pd.notna(result_row['slope_degrees']) and
                                                                           result_row[
                                                                               'slope_degrees'] != 15.0 else None,

                    # CRITICAL: Map to the exact field names the UI expects
                    'tx_distance_miles': tx_distance,
                    'tx_voltage_kv': tx_voltage,
                    'slope_degrees': result_row['slope_degrees'] if pd.notna(result_row['slope_degrees']) else None,

                    'tx_lines_count': int(result_row['tx_lines_count']) if pd.notna(
                        result_row['tx_lines_count']) else 0,
                    'tx_closest_owner': result_row['tx_closest_owner'] if pd.notna(
                        result_row['tx_closest_owner']) else 'Unknown',
                    'suitability_score': int(result_row['suitability_score']) if pd.notna(
                        result_row['suitability_score']) else 0,
                    'suitability_category': result_row['suitability_category'] if pd.notna(
                        result_row['suitability_category']) else 'Unknown',
                    'recommended_for_outreach': bool(result_row['recommended_for_outreach']) if pd.notna(
                        result_row['recommended_for_outreach']) else False
                })

            # Calculate summary
            summary = {
                'total_parcels': len(results_df),
                'excellent': len(results_df[results_df['suitability_category'] == 'Excellent']),
                'good': len(results_df[results_df['suitability_category'] == 'Good']),
                'fair': len(results_df[results_df['suitability_category'] == 'Fair']),
                'poor': len(results_df[results_df['suitability_category'] == 'Poor']),
                'recommended_for_outreach': len(results_df[results_df['recommended_for_outreach'] == True]),
                'average_score': float(results_df['suitability_score'].mean()),
                'parcels_with_transmission': len(results_df[results_df['has_transmission_data'] == True]),
                'location': f"{results_df.iloc[0]['county']}, {results_df.iloc[0]['state']}"
            }

            return {
                'success': True,
                'analysis_id': analysis_id,
                'analysis_timestamp': str(results_df.iloc[0]['analysis_timestamp']),
                'parcels_table': parcels,
                'summary': summary,
                'analysis_metadata': {
                    'analysis_type': 'bigquery_stored',
                    'includes_slope': True,
                    'includes_transmission': True,
                    'scoring_method': 'Multi-factor Scoring via BigQuery'
                }
            }

        except Exception as e:
            logger.error(f"Error retrieving analysis results: {e}")
            return {'success': False, 'error': str(e)}


    def debug_stored_data(self, analysis_id: str):
        """Debug what's actually stored in BigQuery"""
        query = f"""
        SELECT 
            parcel_id,
            tx_nearest_distance_miles,
            tx_max_voltage_kv,
            tx_lines_count,
            has_transmission_data
        FROM `{self.table_ref}`
        WHERE analysis_id = @analysis_id
        LIMIT 3
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("analysis_id", "STRING", analysis_id)
            ]
        )

        try:
            results = self.client.query(query, job_config=job_config).result()
            logger.info("=== BIGQUERY STORAGE DEBUG ===")
            for row in results:
                logger.info(f"Parcel: {row.parcel_id}")
                logger.info(f"  tx_nearest_distance_miles: {row.tx_nearest_distance_miles}")
                logger.info(f"  tx_max_voltage_kv: {row.tx_max_voltage_kv}")
                logger.info(f"  tx_lines_count: {row.tx_lines_count}")
                logger.info(f"  has_transmission_data: {row.has_transmission_data}")
            logger.info("==============================")
        except Exception as e:
            logger.error(f"Debug query failed: {e}")


    def _safe_extract(self, parcel, field_names: List[str], default=None):
        """Safely extract value from parcel using multiple possible field names"""
        for field in field_names:
            if field in parcel and pd.notna(parcel[field]):
                return parcel[field]
        return default

    def _calculate_simple_suitability(self, parcel_data):
        """Calculate suitability score with proper type conversion"""
        try:
            # Convert all values to float with proper error handling
            def safe_float(value, default=0.0):
                if pd.isna(value) or value == '' or value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            # Extract and convert values safely from the parcel data
            acreage = safe_float(parcel_data.get('acreage', 0))
            slope = safe_float(parcel_data.get('avg_slope_degrees', 15.0))
            tx_distance = safe_float(parcel_data.get('tx_nearest_distance', 999.0))
            tx_voltage = safe_float(parcel_data.get('tx_max_voltage', 0.0))

            # Handle alternative column names
            if acreage == 0:
                acreage = safe_float(parcel_data.get('acreage_calc', 0))
            if slope == 15.0:
                slope = safe_float(parcel_data.get('avg_slope', 15.0))
            if tx_distance == 999.0:
                tx_distance = safe_float(parcel_data.get('tx_distance_miles', 999.0))
            if tx_voltage == 0.0:
                tx_voltage = safe_float(parcel_data.get('tx_voltage_kv', 0.0))

            # Calculate scores
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


# Integration functions for web application

def run_transmission_analysis_with_bq_storage(input_file_path: str,
                                              analysis_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Run transmission analysis and store results in BigQuery"""

    try:
        # Import transmission analysis module
        import transmission_analysis_bigquery as tx_analysis

        # Run transmission analysis
        logger.info("Running transmission analysis...")
        result = tx_analysis.run_headless(
            input_file_path=input_file_path,
            buffer_distance_miles=3.0,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        if result['status'] != 'success':
            return {'success': False, 'error': f"Transmission analysis failed: {result.get('message')}"}

        # Load the enhanced results
        from app import download_from_gcs  # Import your existing download function
        import geopandas as gpd

        local_file = download_from_gcs(result['output_file_path'])
        if not local_file:
            return {'success': False, 'error': 'Failed to download analysis results'}

        try:
            enhanced_gdf = gpd.read_file(local_file)
            logger.info(f"Loaded {len(enhanced_gdf)} enhanced parcels")

            # Store in BigQuery
            bq_storage = TransmissionAnalysisBQ()
            analysis_id = bq_storage.store_transmission_analysis(enhanced_gdf, analysis_metadata)

            # Retrieve clean results for UI
            ui_results = bq_storage.get_analysis_results(analysis_id)

            if ui_results['success']:
                return {
                    'success': True,
                    'analysis_id': analysis_id,
                    'message': 'Analysis completed and stored in BigQuery',
                    'analysis_results': ui_results
                }
            else:
                return {'success': False, 'error': 'Failed to retrieve results from BigQuery'}

        finally:
            import os
            if local_file and os.path.exists(local_file):
                os.unlink(local_file)

    except Exception as e:
        logger.error(f"BigQuery transmission analysis failed: {e}")
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
