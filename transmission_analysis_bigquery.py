"""
Enhanced BigQuery Transmission Analysis - Web API Integration
Fixed run_headless function to work with the web pipeline
"""

import os
import logging
import time
import tempfile
import uuid
import json
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, List, Optional
from datetime import datetime
import gc
from shapely.geometry import Point
from shapely import wkt

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_headless(
        input_file_path: str,
        buffer_distance_miles: float = 1.0,
        output_bucket: str = 'bcfparcelsearchrepository',
        project_id: str = 'bcfparcelsearchrepository',
        output_prefix: str = 'Analysis_Results/Transmission/',
        **kwargs
) -> Dict[str, Any]:
    """
    Run transmission line analysis in headless mode for web API

    Args:
        input_file_path: GCS path to input parcel file (gs://...)
        buffer_distance_miles: Buffer distance around transmission lines in miles
        output_bucket: GCS bucket for output files
        project_id: Google Cloud project ID
        output_prefix: Prefix for output files in bucket

    Returns:
        Dict with status, results, and file paths
    """

    start_time = time.time()
    logger.info(f"🔌 Starting transmission analysis for {input_file_path}")
    logger.info(f"Buffer distance: {buffer_distance_miles} miles")

    try:
        # Validate inputs
        if not input_file_path.startswith('gs://'):
            return {
                'status': 'error',
                'message': 'Input file must be a GCS path (gs://...)'
            }

        if buffer_distance_miles <= 0 or buffer_distance_miles > 10:
            return {
                'status': 'error',
                'message': 'Buffer distance must be between 0 and 10 miles'
            }

        # Extract location info from file path
        location_info = extract_location_from_path(input_file_path)
        state = location_info.get('state', 'Unknown')
        county_name = location_info.get('county_name', 'Unknown')

        logger.info(f"Extracted location: {state}, {county_name}")

        # Download input file from GCS
        logger.info("📥 Downloading input file from GCS...")
        local_input_path = download_from_gcs(input_file_path)

        if not local_input_path:
            return {
                'status': 'error',
                'message': f'Failed to download input file: {input_file_path}'
            }

        try:
            # Load parcel data
            logger.info("📊 Loading parcel data...")
            parcels_gdf = gpd.read_file(local_input_path)
            logger.info(f"Loaded {len(parcels_gdf)} parcels for analysis")

            if len(parcels_gdf) == 0:
                return {
                    'status': 'error',
                    'message': 'Input file contains no parcel data'
                }

            # Initialize transmission analyzer
            logger.info("🔌 Initializing transmission analyzer...")
            analyzer = MultiTransmissionAnalyzer(project_id=project_id)

            # Run the analysis
            logger.info("🚀 Running transmission line analysis...")
            enhanced_parcels = analyzer.analyze_parcels_multi_lines(
                parcels_gdf=parcels_gdf,
                max_distance_miles=buffer_distance_miles,
                max_lines_per_parcel=None  # Get all lines within distance
            )

            logger.info(f"✅ Analysis completed for {len(enhanced_parcels)} parcels")

            # Calculate result statistics
            parcels_processed = len(enhanced_parcels)
            parcels_near_transmission = len(enhanced_parcels[enhanced_parcels['tx_lines_count'] > 0])
            parcels_intersecting = len(enhanced_parcels[enhanced_parcels['tx_lines_intersecting'] > 0])

            # Save results to GCS
            logger.info("💾 Saving results to GCS...")
            save_result = save_multi_transmission_results(
                parcels_gdf=enhanced_parcels,
                input_file_path=input_file_path,
                bucket_name=output_bucket,
                project_id=project_id,
                state=state,
                county_name=county_name
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare result
            result = {
                'status': 'success',
                'message': f'Transmission analysis completed successfully',
                'parcels_processed': parcels_processed,
                'parcels_near_transmission': parcels_near_transmission,
                'parcels_intersecting': parcels_intersecting,
                'buffer_distance_miles': buffer_distance_miles,
                'output_file_path': save_result['gpkg_url'],
                'output_csv_path': save_result['csv_url'],
                'processing_time': f"{processing_time:.2f} seconds",
                'analysis_parameters': {
                    'buffer_distance_miles': buffer_distance_miles,
                    'input_file': input_file_path,
                    'output_file': save_result['gpkg_url'],
                    'state': state,
                    'county': county_name
                },
                'transmission_statistics': {
                    'total_parcels': parcels_processed,
                    'parcels_with_nearby_lines': parcels_near_transmission,
                    'parcels_intersecting_lines': parcels_intersecting,
                    'percentage_near_transmission': round((parcels_near_transmission / parcels_processed) * 100, 1) if parcels_processed > 0 else 0,
                    'percentage_intersecting': round((parcels_intersecting / parcels_processed) * 100, 1) if parcels_processed > 0 else 0
                }
            }

            # Add detailed statistics if available
            if parcels_near_transmission > 0:
                tx_parcels = enhanced_parcels[enhanced_parcels['tx_lines_count'] > 0]

                result['transmission_statistics'].update({
                    'avg_lines_per_parcel': float(tx_parcels['tx_lines_count'].mean()),
                    'max_lines_per_parcel': int(tx_parcels['tx_lines_count'].max()),
                    'avg_distance_to_nearest': float(tx_parcels['tx_nearest_distance'].mean()),
                    'closest_transmission_distance': float(tx_parcels['tx_nearest_distance'].min()),
                    'max_voltage_found': float(tx_parcels['tx_max_voltage'].max()) if tx_parcels['tx_max_voltage'].notna().any() else None
                })

            logger.info(f"🎯 Analysis Summary:")
            logger.info(f"   • {parcels_processed} parcels processed")
            logger.info(f"   • {parcels_near_transmission} parcels near transmission lines ({result['transmission_statistics']['percentage_near_transmission']}%)")
            logger.info(f"   • {parcels_intersecting} parcels intersecting transmission lines ({result['transmission_statistics']['percentage_intersecting']}%)")
            logger.info(f"   • Processing time: {processing_time:.2f} seconds")
            logger.info(f"   • Output file: {save_result['gpkg_url']}")

            return result

        finally:
            # Clean up downloaded file
            try:
                if local_input_path and os.path.exists(local_input_path):
                    os.unlink(local_input_path)
                    logger.info("🧹 Cleaned up temporary input file")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Transmission analysis failed: {str(e)}")

        # Include more detailed error information
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Full error traceback: {error_details}")

        return {
            'status': 'error',
            'message': f'Transmission analysis failed: {str(e)}',
            'processing_time': f"{processing_time:.2f} seconds",
            'error_details': error_details,
            'input_file': input_file_path
        }


# Keep all your existing classes and functions exactly as they are:
# - MultiTransmissionAnalyzer class
# - save_multi_transmission_results function
# - clean_county_name_for_path function
# - extract_location_from_path function
# - upload_multi_results_to_gcs function
# - calculate_file_size_mb function
# - download_from_gcs function

class MultiTransmissionAnalyzer:
    def __init__(self, project_id: str, dataset_id: str = "transmission_analysis"):
        """Initialize BigQuery client for multi-line analysis."""
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.transmission_table = f"{project_id}.{dataset_id}.transmission_lines"

        # Verify table exists
        try:
            table_ref = self.client.get_table(self.transmission_table)
            logger.info(f"Connected to transmission table: {self.transmission_table}")
            logger.info(f"Table has {table_ref.num_rows} transmission lines")
        except Exception as e:
            logger.error(f"Cannot access transmission table {self.transmission_table}: {e}")
            raise

    def analyze_parcels_multi_lines(self, parcels_gdf: gpd.GeoDataFrame,
                                  max_distance_miles: float = 1.0,
                                  max_lines_per_parcel: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Analyze parcels and return ALL transmission lines within buffer distance.

        Args:
            parcels_gdf: GeoDataFrame with parcel data
            max_distance_miles: Maximum distance to search for transmission lines
            max_lines_per_parcel: Maximum lines to return per parcel (None = all)

        Returns:
            Enhanced GeoDataFrame with multiple transmission line records per parcel
        """
        logger.info(f"Starting multi-line transmission analysis for {len(parcels_gdf)} parcels")
        logger.info(f"Buffer distance: {max_distance_miles} miles")
        logger.info(f"Max lines per parcel: {max_lines_per_parcel or 'unlimited'}")

        # Ensure proper CRS (convert to WGS84 for BigQuery)
        if parcels_gdf.crs != 'EPSG:4326':
            logger.info(f"Converting from {parcels_gdf.crs} to EPSG:4326")
            parcels_gdf = parcels_gdf.to_crs('EPSG:4326')

        # Process parcels in batches to avoid query size limits
        batch_size = 50  # Reduced batch size for complex geometries
        all_results = []

        for i in range(0, len(parcels_gdf), batch_size):
            batch_parcels = parcels_gdf.iloc[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch_parcels)} parcels)")

            batch_results = self._process_parcel_batch(batch_parcels, max_distance_miles, max_lines_per_parcel)
            if not batch_results.empty:
                all_results.append(batch_results)

        if not all_results:
            logger.warning("No transmission lines found for any parcels")
            return self._add_empty_transmission_columns(parcels_gdf)

        # Combine all batch results
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined results: {len(combined_results)} parcel-transmission relationships")

        # Process into final format
        enhanced_gdf = self._process_multi_line_results(parcels_gdf, combined_results)
        return enhanced_gdf

    def _process_parcel_batch(self, batch_parcels: gpd.GeoDataFrame,
                             max_distance_miles: float,
                             max_lines_per_parcel: Optional[int]) -> pd.DataFrame:
        """Process a batch of parcels against transmission lines."""

        # Extract parcel data with better geometry handling
        parcels_data = []
        for idx, parcel in batch_parcels.iterrows():
            if parcel.geometry and not parcel.geometry.is_empty:
                try:
                    # Simplify complex geometries to avoid BigQuery issues
                    geom = parcel.geometry
                    if hasattr(geom, 'simplify'):
                        geom = geom.simplify(tolerance=0.0001)  # Simplify to ~10m tolerance

                    # Get WKT representation
                    geometry_wkt = geom.wkt

                    # Validate WKT length (BigQuery has limits)
                    if len(geometry_wkt) > 50000:  # ~50KB limit
                        logger.warning(f"Parcel {idx} geometry too complex, using envelope")
                        geometry_wkt = geom.envelope.wkt

                    # Get centroid for fallback distance calculation
                    centroid = geom.centroid

                    parcels_data.append({
                        'original_index': idx,
                        'parcel_id': parcel.get('parcel_id', f'parcel_{idx}'),
                        'geometry_wkt': geometry_wkt,
                        'centroid_lat': centroid.y,
                        'centroid_lon': centroid.x
                    })

                except Exception as e:
                    logger.warning(f"Error processing parcel {idx} geometry: {e}")
                    continue

        if not parcels_data:
            return pd.DataFrame()

        # Build and execute query
        query = self._build_optimized_batch_query(parcels_data, max_distance_miles, max_lines_per_parcel)

        try:
            logger.info(f"Executing BigQuery for {len(parcels_data)} parcels...")
            query_job = self.client.query(query)
            results_df = query_job.result().to_dataframe()

            logger.info(f"BigQuery returned {len(results_df)} parcel-transmission relationships for this batch")
            return results_df

        except Exception as e:
            logger.error(f"BigQuery batch analysis failed: {str(e)}")
            logger.error(f"Query preview: {query[:1000]}...")
            return pd.DataFrame()

    def _build_optimized_batch_query(self, parcels_data: List[Dict],
                                   max_distance_miles: float,
                                   max_lines_per_parcel: Optional[int]) -> str:
        """Build optimized SQL query for batch processing."""

        # Create parcel UNION with proper geometry handling
        parcel_unions = []
        for i, parcel in enumerate(parcels_data):
            parcel_unions.append(f"""
            SELECT 
                {parcel['original_index']} as parcel_index,
                '{parcel['parcel_id']}' as parcel_id,
                ST_GeogFromText('{parcel['geometry_wkt']}') as parcel_geometry,
                ST_GeogPoint({parcel['centroid_lon']}, {parcel['centroid_lat']}) as parcel_centroid
            """)

        parcels_cte = " UNION ALL ".join(parcel_unions)

        # Distance in meters for BigQuery
        buffer_meters = max_distance_miles * 1609.34

        # Optimized query with proper distance calculation and ranking
        if max_lines_per_parcel:
            limit_clause = f"AND line_rank <= {max_lines_per_parcel}"
        else:
            limit_clause = ""

        query = f"""
        WITH parcels AS (
            {parcels_cte}
        ),
        parcel_transmission_analysis AS (
            SELECT 
                p.parcel_index,
                p.parcel_id,
                t.line_id,
                t.owner as tx_owner,
                CAST(t.voltage_kv AS FLOAT64) as tx_volt,
                t.voltage_class as tx_voltage_class,
                t.line_type as tx_type,
                t.status as tx_status,
                
                -- Distance calculation (boundary to line)
                CASE 
                    WHEN ST_Intersects(p.parcel_geometry, t.geometry) THEN 0.0
                    ELSE ST_Distance(p.parcel_geometry, t.geometry) / 1609.34
                END as tx_dist,
                
                -- Intersection flag
                ST_Intersects(p.parcel_geometry, t.geometry) as intersects_parcel,
                
                -- Centroid distance for comparison
                ST_Distance(p.parcel_centroid, t.geometry) / 1609.34 as centroid_dist,
                
                -- Line characteristics
                ST_Length(t.geometry) / 1609.34 as line_length_miles,
                
                -- Ranking by distance (intersecting first, then closest)
                ROW_NUMBER() OVER (
                    PARTITION BY p.parcel_index 
                    ORDER BY 
                        CASE WHEN ST_Intersects(p.parcel_geometry, t.geometry) THEN 0 ELSE 1 END,
                        ST_Distance(p.parcel_geometry, t.geometry)
                ) as line_rank,
                
                -- Proximity categorization
                CASE 
                    WHEN ST_Intersects(p.parcel_geometry, t.geometry) THEN 'INTERSECTS'
                    WHEN ST_Distance(p.parcel_geometry, t.geometry) / 1609.34 <= 0.25 THEN 'VERY_CLOSE'
                    WHEN ST_Distance(p.parcel_geometry, t.geometry) / 1609.34 <= 0.5 THEN 'CLOSE'
                    ELSE 'WITHIN_BUFFER'
                END as proximity_category
                
            FROM parcels p
            CROSS JOIN `{self.transmission_table}` t
            WHERE 
                -- Spatial filter: within buffer OR intersecting
                (ST_DWithin(p.parcel_geometry, t.geometry, {buffer_meters}) 
                 OR ST_Intersects(p.parcel_geometry, t.geometry))
                -- Additional filters for data quality
                AND t.geometry IS NOT NULL
        )
        SELECT 
            parcel_index,
            parcel_id,
            line_id,
            tx_owner,
            tx_volt,
            tx_voltage_class,
            tx_type,
            tx_status,
            tx_dist,
            intersects_parcel,
            centroid_dist,
            line_length_miles,
            line_rank,
            proximity_category
        FROM parcel_transmission_analysis
        WHERE tx_dist <= {max_distance_miles}  -- Final distance filter
        {limit_clause}
        ORDER BY parcel_index, line_rank
        """

        return query

    def _process_multi_line_results(self, original_gdf: gpd.GeoDataFrame,
                                   results_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Process multiple transmission lines per parcel into structured format."""

        if results_df.empty:
            logger.warning("No transmission lines found within buffer distance")
            return self._add_empty_transmission_columns(original_gdf)

        # Group results by parcel
        parcel_groups = results_df.groupby('parcel_index')

        # Create enhanced dataframe
        enhanced_rows = []

        for idx, parcel in original_gdf.iterrows():
            # Get transmission lines for this parcel
            if idx in parcel_groups.groups:
                parcel_tx_lines = parcel_groups.get_group(idx)

                # Create base parcel info
                parcel_data = parcel.to_dict()

                # Add transmission line summary statistics
                parcel_data.update(self._calculate_transmission_summary(parcel_tx_lines))

                # Add detailed transmission line information
                parcel_data.update(self._format_transmission_details(parcel_tx_lines))

                enhanced_rows.append(parcel_data)
            else:
                # No transmission lines found for this parcel
                parcel_data = parcel.to_dict()
                parcel_data.update(self._get_empty_transmission_data())
                enhanced_rows.append(parcel_data)

        # Create new GeoDataFrame
        enhanced_gdf = gpd.GeoDataFrame(enhanced_rows, crs=original_gdf.crs)

        # Log summary statistics
        self._log_analysis_summary(enhanced_gdf, results_df)

        return enhanced_gdf

    def _add_empty_transmission_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add empty transmission columns to GeoDataFrame when no lines are found."""
        empty_data = self._get_empty_transmission_data()

        # Create a copy to avoid modifying original
        result_gdf = gdf.copy()

        # Add each column with proper length matching
        for col, default_value in empty_data.items():
            if isinstance(default_value, list):
                # For list columns, create empty lists for each row
                result_gdf[col] = [[] for _ in range(len(result_gdf))]
            elif default_value is None:
                # For None values, use pandas-compatible None
                result_gdf[col] = None
            else:
                # For scalar values, assign directly
                result_gdf[col] = default_value

        return result_gdf

    def _calculate_transmission_summary(self, tx_lines_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for transmission lines near a parcel."""

        # Handle potential NaN values
        tx_lines_df = tx_lines_df.fillna({'tx_volt': 0, 'tx_owner': 'Unknown', 'tx_voltage_class': 'Unknown'})

        # Count statistics
        total_lines = len(tx_lines_df)
        intersecting_lines = len(tx_lines_df[tx_lines_df['intersects_parcel'] == True])

        # Distance-based counts
        within_quarter = len(tx_lines_df[tx_lines_df['tx_dist'] <= 0.25])
        within_half = len(tx_lines_df[tx_lines_df['tx_dist'] <= 0.5])

        # Distance statistics
        distances = tx_lines_df['tx_dist']
        nearest_dist = distances.min() if not distances.empty else None
        farthest_dist = distances.max() if not distances.empty else None
        avg_dist = distances.mean() if not distances.empty else None

        # Voltage statistics
        voltages = tx_lines_df[tx_lines_df['tx_volt'] > 0]['tx_volt']
        max_voltage = voltages.max() if not voltages.empty else None
        min_voltage = voltages.min() if not voltages.empty else None

        # Owner and class information
        unique_owners = tx_lines_df['tx_owner'].unique()
        unique_classes = tx_lines_df['tx_voltage_class'].unique()

        # Proximity breakdown
        proximity_counts = tx_lines_df['proximity_category'].value_counts().to_dict()

        return {
            # Count statistics
            'tx_lines_count': total_lines,
            'tx_lines_intersecting': intersecting_lines,
            'tx_lines_within_quarter_mile': within_quarter,
            'tx_lines_within_half_mile': within_half,

            # Distance statistics
            'tx_nearest_distance': nearest_dist,
            'tx_farthest_distance': farthest_dist,
            'tx_avg_distance': avg_dist,

            # Voltage statistics
            'tx_max_voltage': max_voltage,
            'tx_min_voltage': min_voltage,
            'tx_voltage_classes': ', '.join([str(c) for c in unique_classes if c != 'Unknown']),

            # Owner statistics
            'tx_owners': ', '.join([str(o) for o in unique_owners if o != 'Unknown']),
            'tx_owners_count': len([o for o in unique_owners if o != 'Unknown']),

            # Proximity analysis
            'tx_proximity_breakdown': json.dumps(proximity_counts)
        }

    def _format_transmission_details(self, tx_lines_df: pd.DataFrame) -> Dict[str, Any]:
        """Format detailed transmission line information for storage."""

        # Sort by distance (closest first)
        tx_sorted = tx_lines_df.sort_values('line_rank')

        # Handle NaN values
        tx_sorted = tx_sorted.fillna({
            'tx_owner': 'Unknown',
            'tx_volt': 0,
            'tx_voltage_class': 'Unknown',
            'tx_type': 'Unknown'
        })

        # Create detailed arrays
        details = {
            'tx_line_ids': tx_sorted['line_id'].tolist(),
            'tx_distances': [round(float(d), 4) for d in tx_sorted['tx_dist']],
            'tx_owners_list': tx_sorted['tx_owner'].tolist(),
            'tx_voltages': [float(v) for v in tx_sorted['tx_volt']],
            'tx_voltage_classes_list': tx_sorted['tx_voltage_class'].tolist(),
            'tx_types': tx_sorted['tx_type'].tolist(),
            'tx_intersects_flags': tx_sorted['intersects_parcel'].tolist(),
            'tx_proximity_categories': tx_sorted['proximity_category'].tolist()
        }

        # Add primary (closest) line info
        if not tx_sorted.empty:
            closest_line = tx_sorted.iloc[0]
            details.update({
                'tx_primary_line_id': str(closest_line['line_id']),
                'tx_primary_distance': float(closest_line['tx_dist']),
                'tx_primary_owner': str(closest_line['tx_owner']),
                'tx_primary_voltage': float(closest_line['tx_volt']),
                'tx_primary_voltage_class': str(closest_line['tx_voltage_class']),
                'tx_primary_type': str(closest_line['tx_type']),
                'tx_primary_intersects': bool(closest_line['intersects_parcel'])
            })

        # JSON summary for complex storage
        details['tx_details_json'] = json.dumps({
            k: v for k, v in details.items()
            if k.startswith('tx_') and isinstance(v, list)
        })

        return details

    def _get_empty_transmission_data(self) -> Dict[str, Any]:
        """Return empty transmission data structure."""
        return {
            'tx_lines_count': 0,
            'tx_lines_intersecting': 0,
            'tx_lines_within_quarter_mile': 0,
            'tx_lines_within_half_mile': 0,
            'tx_nearest_distance': None,
            'tx_farthest_distance': None,
            'tx_avg_distance': None,
            'tx_max_voltage': None,
            'tx_min_voltage': None,
            'tx_voltage_classes': '',
            'tx_owners': '',
            'tx_owners_count': 0,
            'tx_proximity_breakdown': '{}',
            'tx_line_ids': [],
            'tx_distances': [],
            'tx_owners_list': [],
            'tx_voltages': [],
            'tx_voltage_classes_list': [],
            'tx_types': [],
            'tx_intersects_flags': [],
            'tx_proximity_categories': [],
            'tx_primary_line_id': None,
            'tx_primary_distance': None,
            'tx_primary_owner': None,
            'tx_primary_voltage': None,
            'tx_primary_voltage_class': None,
            'tx_primary_type': None,
            'tx_primary_intersects': False,
            'tx_details_json': '[]'
        }

    def _log_analysis_summary(self, enhanced_gdf: gpd.GeoDataFrame, results_df: pd.DataFrame):
        """Log comprehensive analysis summary."""

        total_parcels = len(enhanced_gdf)
        parcels_with_tx = len(enhanced_gdf[enhanced_gdf['tx_lines_count'] > 0])
        parcels_intersecting = len(enhanced_gdf[enhanced_gdf['tx_lines_intersecting'] > 0])
        total_tx_relationships = len(results_df)

        logger.info("=== MULTI-LINE TRANSMISSION ANALYSIS SUMMARY ===")
        logger.info(f"Total parcels analyzed: {total_parcels}")
        logger.info(f"Parcels with transmission lines nearby: {parcels_with_tx}")
        logger.info(f"Parcels with intersecting transmission lines: {parcels_intersecting}")
        logger.info(f"Total parcel-transmission line relationships: {total_tx_relationships}")

        if parcels_with_tx > 0:
            avg_lines_per_parcel = enhanced_gdf[enhanced_gdf['tx_lines_count'] > 0]['tx_lines_count'].mean()
            max_lines_per_parcel = enhanced_gdf['tx_lines_count'].max()

            logger.info(f"Average transmission lines per parcel (with lines): {avg_lines_per_parcel:.1f}")
            logger.info(f"Maximum transmission lines for any single parcel: {max_lines_per_parcel}")

            # Voltage analysis
            if enhanced_gdf['tx_max_voltage'].notna().any():
                max_voltage_overall = enhanced_gdf['tx_max_voltage'].max()
                logger.info(f"Highest voltage line found: {max_voltage_overall} kV")

            # Distance analysis
            if enhanced_gdf['tx_nearest_distance'].notna().any():
                closest_distance = enhanced_gdf['tx_nearest_distance'].min()
                logger.info(f"Closest transmission line distance: {closest_distance:.4f} miles")


def save_multi_transmission_results(parcels_gdf: gpd.GeoDataFrame, input_file_path: str, bucket_name: str,
                                    project_id: str, state: Optional[str] = None, county_name: Optional[str] = None) -> Dict[str, str]:
    """Save multi-line transmission analysis results with proper error handling."""
    # Create temp directory
    temp_id = str(uuid.uuid4())
    temp_dir = "/tmp/gis_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Determine output format
    output_suffix = '.gpkg' if input_file_path.endswith('.gpkg') else '.gpkg'  # Default to .gpkg
    temp_gpkg_path = os.path.join(temp_dir, f"multi_tx_result_{temp_id}{output_suffix}")
    temp_csv_path = os.path.join(temp_dir, f"multi_tx_result_{temp_id}.csv")

    try:
        # CRITICAL FIX: Create a copy and convert list columns to strings for GPKG compatibility
        logger.info(f"Preparing data for saving ({len(parcels_gdf)} records)...")
        parcels_to_save = parcels_gdf.copy()

        # Convert any list/array columns to strings before saving
        for column in parcels_to_save.columns:
            if column == 'geometry':  # Skip geometry column
                continue

            # Check if column contains lists or arrays
            if len(parcels_to_save) > 0:
                sample_value = parcels_to_save[column].iloc[0]
                if isinstance(sample_value, (list, tuple)):
                    logger.info(f"Converting list column '{column}' to string for GPKG compatibility")
                    # Convert lists to comma-separated strings
                    parcels_to_save[column] = parcels_to_save[column].apply(
                        lambda x: ','.join(map(str, x)) if isinstance(x, (list, tuple)) and x else ''
                    )
                elif parcels_to_save[column].dtype == 'object':
                    # Check for any remaining list values in object columns
                    def convert_lists_to_strings(val):
                        if isinstance(val, (list, tuple)):
                            return ','.join(map(str, val)) if val else ''
                        return val

                    parcels_to_save[column] = parcels_to_save[column].apply(convert_lists_to_strings)

        # Save GeoPackage
        logger.info(f"Saving GeoPackage with {len(parcels_to_save)} records...")
        parcels_to_save.to_file(temp_gpkg_path, driver='GPKG', layer='multi_transmission_analysis')

        # Save CSV (without geometry)
        logger.info("Saving CSV...")
        csv_data = parcels_to_save.drop(columns=['geometry'])
        csv_data.to_csv(temp_csv_path, index=False)

        # Upload to GCS
        logger.info("Uploading to GCS...")
        return upload_multi_results_to_gcs(
            temp_gpkg_path,
            temp_csv_path,
            input_file_path,
            bucket_name,
            parcels_to_save,  # Use the modified dataframe
            output_suffix,
            state,
            county_name
        )

    except Exception as e:
        logger.error(f"Error saving multi-transmission results: {str(e)}")
        import traceback
        logger.error(f"Save error traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup temp files
        for temp_file in [temp_gpkg_path, temp_csv_path]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


def clean_county_name_for_path(county_name):
    """Clean county name for consistent GCS path structure."""
    if not county_name:
        return "Unknown"

    # Remove "County" suffix if present
    if county_name.endswith(" County"):
        return county_name.replace(" County", "")
    elif county_name.endswith("County"):
        return county_name.replace("County", "")

    return county_name


def extract_location_from_path(file_path: str) -> Dict[str, Optional[str]]:
    """Extract state and county information from file path."""
    try:
        if file_path.startswith('gs://bcfparcelsearchrepository/'):
            # Expected format: gs://bcfparcelsearchrepository/STATE/COUNTY_NAME/folder/file.gpkg
            path_parts = file_path.replace('gs://bcfparcelsearchrepository/', '').split('/')
            if len(path_parts) >= 2:
                extracted_county = clean_county_name_for_path(path_parts[1])
                return {
                    'state': path_parts[0],
                    'county_name': extracted_county
                }

        # If we can't extract from path, return empty
        return {'state': None, 'county_name': None}

    except Exception as e:
        logger.warning(f"Failed to extract location from path {file_path}: {e}")
        return {'state': None, 'county_name': None}


def upload_multi_results_to_gcs(temp_gpkg_path: str, temp_csv_path: str, input_file_path: str,
                                bucket_name: str, parcels_gdf: gpd.GeoDataFrame, output_suffix: str,
                                state: Optional[str] = None, county_name: Optional[str] = None) -> Dict[str, str]:
    """Upload multi-line results to GCS with organized structure."""

    # Extract location info from path if not provided
    if not state or not county_name:
        extracted_location = extract_location_from_path(input_file_path)
        state = state or extracted_location.get('state', 'XX')
        county_name = county_name or extracted_location.get('county_name', 'Unknown')

    # Clean county name for consistent path structure
    if county_name:
        county_name = clean_county_name_for_path(county_name)
        logger.info(f"Cleaned county name for path: {county_name}")

    # Create filenames
    original_filename = os.path.basename(input_file_path)
    name_without_ext = os.path.splitext(original_filename)[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gpkg_filename = f"{name_without_ext}_multi_tx_analysis_{timestamp}.gpkg"
    csv_filename = f"{name_without_ext}_multi_tx_analysis_{timestamp}.csv"

    # Organized storage path - now using cleaned county name
    storage_path = f"{state}/{county_name}/Transmission_Files"
    gpkg_blob_path = f"{storage_path}/{gpkg_filename}"
    csv_blob_path = f"{storage_path}/{csv_filename}"

    logger.info(f"Using storage path: {storage_path}")

    try:
        # Initialize storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Upload GPKG
        logger.info(f"Uploading GPKG to {gpkg_blob_path}")
        gpkg_blob = bucket.blob(gpkg_blob_path)
        gpkg_blob.metadata = {
            'analysis_type': 'multi_transmission_analysis',
            'source_file': input_file_path,
            'state': state,
            'county_name': county_name,  # This is now the cleaned name
            'created_by': 'gis_pipeline_multi_line_v2',
            'record_count': str(len(parcels_gdf)),
            'timestamp': timestamp
        }
        gpkg_blob.upload_from_filename(temp_gpkg_path)

        # Upload CSV
        logger.info(f"Uploading CSV to {csv_blob_path}")
        csv_blob = bucket.blob(csv_blob_path)
        csv_blob.metadata = gpkg_blob.metadata.copy()
        csv_blob.upload_from_filename(temp_csv_path)

        logger.info(f"Successfully uploaded multi-line results")

        return {
            'gpkg_url': f"gs://{bucket_name}/{gpkg_blob_path}",
            'csv_url': f"gs://{bucket_name}/{csv_blob_path}",
            'gpkg_filename': gpkg_filename,
            'csv_filename': csv_filename,
            'storage_path': storage_path
        }

    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        raise


def calculate_file_size_mb(gcs_path: str) -> float:
    """Calculate file size in MB from GCS path."""
    try:
        if not gcs_path or not gcs_path.startswith('gs://'):
            return 0.0

        path_parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ''

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            blob.reload()
            return blob.size / (1024 * 1024) if blob.size else 0.0
        return 0.0
    except Exception as e:
        logger.warning(f"Error calculating file size for {gcs_path}: {e}")
        return 0.0


def download_from_gcs(gcs_path):
    """Download a file from Google Cloud Storage to a local temp file."""
    try:
        if not gcs_path.startswith("gs://"):
            return None

        parts = gcs_path[5:].split("/", 1)
        if len(parts) < 2:
            return None

        bucket_name = parts[0]
        blob_path = parts[1]

        _, file_extension = os.path.splitext(blob_path)
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            temp_path = tmp.name

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.error(f"File does not exist in GCS: {gcs_path}")
            return None

        logger.info(f"Downloading {gcs_path} to {temp_path}")
        blob.download_to_filename(temp_path)
        return temp_path

    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        return None


def verify_analysis_dependencies():
    """
    Verify that all dependencies for this analysis module are available
    Add this function to both analysis scripts
    """
    import logging
    logger = logging.getLogger(__name__)

    results = {
        'dependencies': {},
        'bigquery_connectivity': False,
        'gcs_connectivity': False,
        'sample_data_access': False,
        'ready_for_analysis': False
    }

    # Test required Python packages
    required_packages = {
        'pandas': 'pandas',
        'geopandas': 'geopandas',
        'google.cloud.bigquery': 'BigQuery client',
        'google.cloud.storage': 'GCS client',
        'shapely': 'shapely'
    }

    for package_name, description in required_packages.items():
        try:
            __import__(package_name)
            results['dependencies'][package_name] = True
            logger.info(f"✅ {description} available")
        except ImportError as e:
            results['dependencies'][package_name] = False
            logger.error(f"❌ {description} not available: {e}")

    # Test BigQuery connectivity
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        # Test with a simple query
        test_query = "SELECT 1 as test_value"
        query_job = client.query(test_query)
        list(query_job.result())  # Execute query
        results['bigquery_connectivity'] = True
        logger.info("✅ BigQuery connectivity verified")
    except Exception as e:
        results['bigquery_connectivity'] = False
        logger.error(f"❌ BigQuery connectivity failed: {e}")

    # Test GCS connectivity
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket('bcfparcelsearchrepository')
        bucket.exists()  # Test bucket access
        results['gcs_connectivity'] = True
        logger.info("✅ GCS connectivity verified")
    except Exception as e:
        results['gcs_connectivity'] = False
        logger.error(f"❌ GCS connectivity failed: {e}")

    # Determine overall readiness
    critical_deps = ['pandas', 'geopandas', 'google.cloud.bigquery', 'google.cloud.storage']
    deps_ready = all(results['dependencies'].get(dep, False) for dep in critical_deps)

    results['ready_for_analysis'] = (
            deps_ready and
            results['bigquery_connectivity'] and
            results['gcs_connectivity']
    )

    return results


def test_analysis_with_sample_data():
    """
    Test the analysis with a small sample dataset
    Add this function to both analysis scripts, customized for each
    """
    import logging
    import os
    logger = logging.getLogger(__name__)

    try:
        # Get the current script name to determine which analysis to run
        script_name = os.path.basename(__file__)

        # For transmission_analysis_bigquery.py, test transmission analysis
        if 'transmission' in script_name.lower():
            logger.info("🧪 Testing transmission analysis with sample data...")
            test_result = run_headless(
                input_file_path="gs://bcfparcelsearchrepository/PA/Blair/Parcel_Files/BlairCoPA_052620250506.gpkg",
                buffer_distance_miles=0.5,  # Small buffer for quick test
                output_bucket='bcfparcelsearchrepository',
                project_id='bcfparcelsearchrepository'
            )

        # For bigquery_slope_analysis.py, test slope analysis
        elif 'slope' in script_name.lower():
            logger.info("🧪 Testing slope analysis with sample data...")
            test_result = run_headless(
                input_file_path="gs://bcfparcelsearchrepository/PA/Blair/Parcel_Files/BlairCoPA_052620250506.gpkg",
                max_slope_degrees=20.0,  # Lenient slope for quick test
                output_bucket='bcfparcelsearchrepository',
                project_id='bcfparcelsearchrepository'
            )
        else:
            return {'status': 'error', 'message': f'Unknown analysis module: {script_name}'}

        if test_result['status'] == 'success':
            logger.info(f"✅ Sample analysis completed successfully")
            logger.info(f"   Processed: {test_result.get('parcels_processed', 0)} parcels")
            return {
                'status': 'success',
                'message': 'Sample analysis completed successfully',
                'test_result': test_result
            }
        else:
            logger.error(f"❌ Sample analysis failed: {test_result.get('message', 'Unknown error')}")
            return {
                'status': 'error',
                'message': f"Sample analysis failed: {test_result.get('message', 'Unknown error')}",
                'test_result': test_result
            }

    except Exception as e:
        logger.error(f"❌ Sample analysis test failed: {str(e)}")
        return {
            'status': 'error',
            'message': f'Sample analysis test failed: {str(e)}'
        }


# Add this verification command to both scripts at the bottom
if __name__ == "__main__":
    import sys
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Check if this is a verification run
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        print(f"🔍 Verifying {__file__} dependencies and functionality...")

        # Test dependencies
        dep_results = verify_analysis_dependencies()

        print("\n" + "=" * 50)
        print("DEPENDENCY VERIFICATION RESULTS")
        print("=" * 50)

        for package, status in dep_results['dependencies'].items():
            status_str = "✅ AVAILABLE" if status else "❌ MISSING"
            print(f"{package}: {status_str}")

        print(f"\nBigQuery Connectivity: {'✅ WORKING' if dep_results['bigquery_connectivity'] else '❌ FAILED'}")
        print(f"GCS Connectivity: {'✅ WORKING' if dep_results['gcs_connectivity'] else '❌ FAILED'}")
        print(f"Overall Ready: {'✅ YES' if dep_results['ready_for_analysis'] else '❌ NO'}")

        # Test with sample data if everything is ready
        if dep_results['ready_for_analysis']:
            test_response = input("\nRun sample analysis test? [y/N]: ")
            if test_response.lower() in ['y', 'yes']:
                print("\n🧪 Running sample analysis test...")
                sample_result = test_analysis_with_sample_data()

                if sample_result['status'] == 'success':
                    print("✅ Sample analysis test PASSED")
                else:
                    print(f"❌ Sample analysis test FAILED: {sample_result['message']}")

        print("\n🏁 Verification complete!")

        # Exit with appropriate code
        if dep_results['ready_for_analysis']:
            print("✅ Analysis module is ready for use!")
            sys.exit(0)
        else:
            print("❌ Please fix the issues above before using this analysis module.")
            sys.exit(1)
    else:
        test_result = run_headless(
            input_file_path="gs://bcfparcelsearchrepository/PA/Blair/Parcel_Files/BlairCoPA_052620250506.gpkg",
            buffer_distance_miles=1.0,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        print("Test Results:")
        print(json.dumps(test_result, indent=2, default=str))
        pass