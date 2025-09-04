# parcel_analysis_integration.py - NEW FILE
"""
Parcel Analysis Integration Service
Runs slope and transmission analysis, then merges results back into parcel data for CRM export
"""

import logging
import geopandas as gpd
import pandas as pd
import os
import tempfile
from typing import Dict, Any, List, Optional

# Import your analysis modules
from bigquery_slope_analysis import run_headless_fixed
from transmission_analysis_bigquery import run_headless as run_transmission_headless

logger = logging.getLogger(__name__)


class ParcelAnalysisIntegrator:
    """Integrates slope and transmission analysis results with parcel data"""

    def __init__(self, project_id: str = 'bcfparcelsearchrepository'):
        self.project_id = project_id

    def run_complete_analysis(self,
                              input_file_path: str,
                              max_slope_degrees: float = 15.0,
                              buffer_distance_miles: float = 1.0) -> Dict[str, Any]:
        """
        Run both slope and transmission analysis, return enriched parcel data

        Args:
            input_file_path: GCS path to parcel file
            max_slope_degrees: Maximum slope threshold for analysis
            buffer_distance_miles: Buffer distance for transmission analysis

        Returns:
            Dict with enriched parcel data and analysis results
        """
        logger.info(f"Starting complete analysis for: {input_file_path}")
        logger.info(f"Slope threshold: {max_slope_degrees}°, Transmission buffer: {buffer_distance_miles} miles")

        try:
            # Step 1: Run slope analysis
            logger.info("Step 1: Running slope analysis...")
            slope_result = run_headless_fixed(
                input_file_path=input_file_path,
                max_slope_degrees=max_slope_degrees,
                output_bucket='bcfparcelsearchrepository',
                project_id=self.project_id
            )

            if slope_result['status'] != 'success':
                logger.error(f"Slope analysis failed: {slope_result.get('message')}")
                return {'status': 'error', 'message': f"Slope analysis failed: {slope_result.get('message')}"}

            logger.info(f"Slope analysis completed: {slope_result['parcels_processed']} parcels")

            # Step 2: Run transmission analysis
            logger.info("Step 2: Running transmission analysis...")
            transmission_result = run_transmission_headless(
                input_file_path=input_file_path,
                buffer_distance_miles=buffer_distance_miles,
                output_bucket='bcfparcelsearchrepository',
                project_id=self.project_id
            )

            if transmission_result['status'] != 'success':
                logger.error(f"Transmission analysis failed: {transmission_result.get('message')}")
                return {'status': 'error',
                        'message': f"Transmission analysis failed: {transmission_result.get('message')}"}

            logger.info(f"Transmission analysis completed: {transmission_result['parcels_processed']} parcels")

            # Step 3: Load and merge analysis results
            logger.info("Step 3: Loading and merging analysis results...")
            enriched_parcels = self.merge_analysis_results(
                input_file_path,
                slope_result,
                transmission_result
            )

            if not enriched_parcels:
                return {'status': 'error', 'message': 'Failed to merge analysis results'}

            logger.info(f"Successfully enriched {len(enriched_parcels)} parcels with analysis data")

            return {
                'status': 'success',
                'enriched_parcels': enriched_parcels,
                'slope_analysis': slope_result,
                'transmission_analysis': transmission_result,
                'parcels_enriched': len(enriched_parcels)
            }

        except Exception as e:
            logger.error(f"Complete analysis failed: {str(e)}")
            return {'status': 'error', 'message': f'Analysis integration failed: {str(e)}'}

    def merge_analysis_results(self,
                               original_file_path: str,
                               slope_result: Dict[str, Any],
                               transmission_result: Dict[str, Any]) -> Optional[List[Dict]]:
        """Load analysis results and merge them back into parcel data"""

        try:
            # Step 1: Download and load slope analysis results
            slope_data = None
            if slope_result.get('output_file_path'):
                logger.info(f"Loading slope data from: {slope_result['output_file_path']}")
                slope_data = self._load_gcs_gpkg(slope_result['output_file_path'])
                if slope_data is not None:
                    logger.info(f"Loaded slope data: {len(slope_data)} parcels with slope information")
                else:
                    logger.warning("Failed to load slope analysis results")

            # Step 2: Download and load transmission analysis results
            transmission_data = None
            if transmission_result.get('output_file_path'):
                logger.info(f"Loading transmission data from: {transmission_result['output_file_path']}")
                transmission_data = self._load_gcs_gpkg(transmission_result['output_file_path'])
                if transmission_data is not None:
                    logger.info(
                        f"Loaded transmission data: {len(transmission_data)} parcels with transmission information")
                else:
                    logger.warning("Failed to load transmission analysis results")

            # Step 3: Load original parcel data for baseline
            logger.info(f"Loading original parcel data from: {original_file_path}")
            original_data = self._load_gcs_gpkg(original_file_path)
            if original_data is None:
                logger.error("Failed to load original parcel data")
                return None

            logger.info(f"Loaded original data: {len(original_data)} parcels")

            # Step 4: Merge all data together
            enriched_parcels = self._merge_parcel_datasets(original_data, slope_data, transmission_data)

            return enriched_parcels

        except Exception as e:
            logger.error(f"Error merging analysis results: {str(e)}")
            return None

    def _load_gcs_gpkg(self, gcs_path: str) -> Optional[gpd.GeoDataFrame]:
        """Download and load GeoPackage from GCS"""
        if not gcs_path or not gcs_path.startswith('gs://'):
            return None

        try:
            # Download to temporary file
            temp_path = self._download_from_gcs(gcs_path)
            if not temp_path:
                return None

            # Load the data
            gdf = gpd.read_file(temp_path)

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            return gdf

        except Exception as e:
            logger.error(f"Error loading GCS file {gcs_path}: {str(e)}")
            return None

    def _download_from_gcs(self, gcs_path: str) -> Optional[str]:
        """Download file from GCS to temporary location"""
        try:
            from google.cloud import storage

            # Parse GCS path
            path_parts = gcs_path.replace('gs://', '').split('/', 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1]

            # Create temp file
            _, file_extension = os.path.splitext(blob_path)
            temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
            os.close(temp_fd)

            # Download
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(temp_path)

            return temp_path

        except Exception as e:
            logger.error(f"Error downloading {gcs_path}: {str(e)}")
            return None

    def _merge_parcel_datasets(self,
                               original: gpd.GeoDataFrame,
                               slope_data: Optional[gpd.GeoDataFrame],
                               transmission_data: Optional[gpd.GeoDataFrame]) -> List[Dict]:
        """Merge original parcel data with slope and transmission analysis results"""

        logger.info("Merging parcel datasets...")
        enriched_parcels = []

        for idx, parcel in original.iterrows():
            # Start with original parcel data
            enriched_parcel = parcel.to_dict()

            # Get parcel identifier for matching
            parcel_id = parcel.get('parcel_id', parcel.get('id', f'parcel_{idx}'))

            # Add slope data if available
            if slope_data is not None:
                slope_info = self._find_matching_parcel(slope_data, parcel_id, parcel)
                if slope_info:
                    # Add slope fields with proper field names for CRM
                    if 'avg_slope_degrees' in slope_info:
                        enriched_parcel['avg_slope'] = slope_info['avg_slope_degrees']
                        enriched_parcel['slope_degrees'] = slope_info['avg_slope_degrees']
                        logger.debug(f"Added slope data to {parcel_id}: {slope_info['avg_slope_degrees']}°")

                    if 'min_slope_degrees' in slope_info:
                        enriched_parcel['min_slope_degrees'] = slope_info['min_slope_degrees']

                    if 'max_slope_degrees' in slope_info:
                        enriched_parcel['max_slope_degrees'] = slope_info['max_slope_degrees']

                    # Add suitability analysis object
                    enriched_parcel['suitability_analysis'] = {
                        'slope_degrees': slope_info.get('avg_slope_degrees'),
                        'slope_category': slope_info.get('slope_category'),
                        'slope_suitability': slope_info.get('reference_suitability'),
                        'analysis_confidence': slope_info.get('analysis_confidence', 'MEDIUM')
                    }
                else:
                    logger.debug(f"No slope data found for {parcel_id}")

            # Add transmission data if available
            if transmission_data is not None:
                tx_info = self._find_matching_parcel(transmission_data, parcel_id, parcel)
                if tx_info:
                    # Add transmission fields with proper field names for CRM
                    if 'tx_nearest_distance' in tx_info:
                        enriched_parcel['transmission_distance'] = tx_info['tx_nearest_distance']
                        enriched_parcel['tx_distance'] = tx_info['tx_nearest_distance']
                        enriched_parcel['tx_nearest_distance'] = tx_info['tx_nearest_distance']
                        logger.debug(
                            f"Added transmission distance to {parcel_id}: {tx_info['tx_nearest_distance']} miles")

                    if 'tx_max_voltage' in tx_info:
                        enriched_parcel['transmission_voltage'] = tx_info['tx_max_voltage']
                        enriched_parcel['tx_voltage'] = tx_info['tx_max_voltage']
                        enriched_parcel['tx_max_voltage'] = tx_info['tx_max_voltage']
                        logger.debug(f"Added transmission voltage to {parcel_id}: {tx_info['tx_max_voltage']} kV")

                    # Add other transmission fields
                    if 'tx_lines_count' in tx_info:
                        enriched_parcel['tx_lines_count'] = tx_info['tx_lines_count']

                    # Add transmission analysis object
                    enriched_parcel['transmission_analysis'] = {
                        'transmission_distance': tx_info.get('tx_nearest_distance'),
                        'transmission_voltage': tx_info.get('tx_max_voltage'),
                        'lines_count': tx_info.get('tx_lines_count', 0),
                        'lines_intersecting': tx_info.get('tx_lines_intersecting', 0)
                    }
                else:
                    logger.debug(f"No transmission data found for {parcel_id}")

            enriched_parcels.append(enriched_parcel)

        logger.info(f"Successfully merged data for {len(enriched_parcels)} parcels")
        return enriched_parcels

    def _find_matching_parcel(self, analysis_data: gpd.GeoDataFrame, parcel_id: str, original_parcel) -> Optional[Dict]:
        """Find matching parcel in analysis results"""
        try:
            # Try exact parcel_id match first
            if 'parcel_id' in analysis_data.columns:
                matches = analysis_data[analysis_data['parcel_id'] == parcel_id]
                if len(matches) > 0:
                    return matches.iloc[0].to_dict()

            # Try other ID fields
            for id_field in ['id', 'pin', 'apn']:
                if id_field in analysis_data.columns:
                    matches = analysis_data[analysis_data[id_field] == parcel_id]
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()

            # Try matching by index if same length datasets
            if len(analysis_data) == len(original_parcel.index):
                try:
                    original_idx = original_parcel.name if hasattr(original_parcel, 'name') else 0
                    if original_idx < len(analysis_data):
                        return analysis_data.iloc[original_idx].to_dict()
                except:
                    pass

            return None

        except Exception as e:
            logger.debug(f"Error finding matching parcel for {parcel_id}: {str(e)}")
            return None


def run_complete_parcel_analysis(input_file_path: str,
                                 max_slope_degrees: float = 15.0,
                                 buffer_distance_miles: float = 1.0,
                                 project_id: str = 'bcfparcelsearchrepository') -> Dict[str, Any]:
    """
    Convenience function to run complete analysis and return enriched parcels

    Usage:
        result = run_complete_parcel_analysis('gs://bucket/path/to/parcels.gpkg')
        if result['status'] == 'success':
            enriched_parcels = result['enriched_parcels']
            # Now export to CRM with enriched data
    """
    integrator = ParcelAnalysisIntegrator(project_id=project_id)
    return integrator.run_complete_analysis(
        input_file_path=input_file_path,
        max_slope_degrees=max_slope_degrees,
        buffer_distance_miles=buffer_distance_miles
    )


# Test function
def test_integration():
    """Test the integration with sample data"""
    test_file = "gs://bcfparcelsearchrepository/PA/Blair/Parcel_Files/BlairCoPA_052620250506.gpkg"

    result = run_complete_parcel_analysis(
        input_file_path=test_file,
        max_slope_degrees=20.0,
        buffer_distance_miles=0.5
    )

    if result['status'] == 'success':
        print(f"Integration test successful! Enriched {result['parcels_enriched']} parcels")

        # Show sample enriched parcel
        if result['enriched_parcels']:
            sample = result['enriched_parcels'][0]
            print(f"Sample parcel fields: {list(sample.keys())}")

            # Check for critical fields
            for field in ['avg_slope', 'transmission_distance', 'transmission_voltage']:
                if field in sample:
                    print(f"✅ {field}: {sample[field]}")
                else:
                    print(f"❌ {field}: NOT FOUND")

        return result['enriched_parcels']
    else:
        print(f"Integration test failed: {result['message']}")
        return None


if __name__ == "__main__":
    # Run test
    test_integration()