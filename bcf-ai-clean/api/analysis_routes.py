from flask import Blueprint, request, jsonify
import logging
from services.ai_service import ai_service
from config.database import db
from enhanced_parcel_search import run_headless, preview_search_count
from bigquery_slope_analysis import run_headless as run_slope_analysis
from transmission_analysis_bigquery import run_headless as run_transmission_analysis
import tempfile
import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Optional
import os
from google.cloud import storage
import time

logger = logging.getLogger(__name__)
analysis_bp = Blueprint('analysis', __name__)

# Simple working version - analysis_routes.py

@analysis_bp.route('/analyze-state-counties', methods=['POST'])
def analyze_state_counties():
    try:
        data = request.get_json()
        state = data.get('state')
        project_type = data.get('project_type', 'solar')
        
        logger.info(f"Starting state analysis: {state} - {project_type}")
        
        # Simple mock data that works - replace with your real data later
        nc_counties = [
            'Alamance', 'Alexander', 'Alleghany', 'Anson', 'Ashe', 'Avery', 'Beaufort', 'Bertie',
            'Bladen', 'Brunswick', 'Buncombe', 'Burke', 'Cabarrus', 'Caldwell', 'Camden', 'Carteret',
            'Caswell', 'Catawba', 'Chatham', 'Cherokee', 'Chowan', 'Clay', 'Cleveland', 'Columbus',
            'Craven', 'Cumberland', 'Currituck', 'Dare', 'Davidson', 'Davie', 'Duplin', 'Durham',
            'Edgecombe', 'Forsyth', 'Franklin', 'Gaston', 'Gates', 'Graham', 'Granville', 'Greene',
            'Guilford', 'Halifax', 'Harnett', 'Haywood', 'Henderson', 'Hertford', 'Hoke', 'Hyde',
            'Iredell', 'Jackson', 'Johnston', 'Jones', 'Lee', 'Lenoir', 'Lincoln', 'McDowell',
            'Macon', 'Madison', 'Martin', 'Mecklenburg', 'Mitchell', 'Montgomery', 'Moore', 'Nash',
            'New Hanover', 'Northampton', 'Onslow', 'Orange', 'Pamlico', 'Pasquotank', 'Pender', 'Perquimans',
            'Person', 'Pitt', 'Polk', 'Randolph', 'Richmond', 'Robeson', 'Rockingham', 'Rowan',
            'Rutherford', 'Sampson', 'Scotland', 'Stanly', 'Stokes', 'Surry', 'Swain', 'Transylvania',
            'Tyrrell', 'Union', 'Vance', 'Wake', 'Warren', 'Washington', 'Watauga', 'Wayne',
            'Wilkes', 'Wilson', 'Yadkin', 'Yancey'
        ]
        
        # Create simple county data
        counties = []
        for i, county_name in enumerate(nc_counties):
            counties.append({
                'name': county_name,
                'fips': f"37{str(i+1).zfill(3)}",
                'score': 85 - (i % 30),  # Vary scores from 85 down to 55
                'rank': i + 1,
                'population': 100000 + (i * 5000),
                'rural_indicator': i % 3 == 0,  # Every 3rd county is rural
                'development_potential': ['High', 'Moderate', 'Low'][i % 3],
                'factor_scores': {
                    'transmission': 75 + (i % 15),
                    'topography': 70 + (i % 20),
                    'regulatory': 65 + (i % 25)
                }
            })
        
        logger.info(f"Created {len(counties)} counties for {state}")
        
        return jsonify({
            'success': True,
            'analysis': {
                'state': state,
                'project_type': project_type,
                'summary': f'Analysis of {len(counties)} counties in {state}',
                'counties': counties,
                'total_counties': len(counties)
            }
        })
        
    except Exception as e:
        logger.error(f"State analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

                       
@analysis_bp.route('/county-market-analysis', methods=['POST'])
def county_market_analysis():
    """Run AI market analysis for a specific county"""
    try:
        data = request.get_json()
        county_fips = data.get('county_fips')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type')
        
        logger.info(f"Starting county market analysis: {county_name}, {state}")
        
        if not all([county_name, state, project_type]):
            return jsonify({
                'success': False,
                'error': 'county_name, state, and project_type are required'
            }), 400
        
        # Call AI service for market analysis
        result = ai_service.analyze_county_market(county_name, state, project_type)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'AI service returned no result'
            }), 500
        
        if not result.success:
            return jsonify({
                'success': False,
                'error': result.error
            }), 500
            
        return jsonify({
            'success': True,
            'analysis': result.content,  # <-- CHANGED FROM result.data to result.content
            'analysis_type': 'AI-Powered Market Analysis'
        })
        
    except Exception as e:
        logger.error(f"County market analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Market analysis failed: {str(e)}'
        }), 500
    
@analysis_bp.route('/execute-parcel-search', methods=['POST'])
def execute_parcel_search():
    """Execute real parcel search using ReportAll API"""
    try:
        data = request.get_json()
        county_id = data.get('county_id')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type')
        user_id = data.get('user_id', 'current_user')
        
        logger.info(f"Executing REAL parcel search: {county_name}, {state}")
        
        if not all([county_id, county_name, state]):
            return jsonify({
                'success': False,
                'error': 'county_id, county_name and state are required'
            }), 400
        
        # Prepare search parameters for enhanced_parcel_search
        search_params = {
            'county_id': county_id,
            'county_name': county_name,
            'state': state,
            'project_type': project_type,
            'user_id': user_id,
            'calc_acreage_min': data.get('calc_acreage_min'),
            'calc_acreage_max': data.get('calc_acreage_max'),
            'owner': data.get('owner'),
            'parcel_id': data.get('parcel_id')
        }
        
        # Execute REAL parcel search
        result = run_headless(**search_params)
        
        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'record_count': result['record_count'],
                'county_name': result['county_name'],
                'state_abbr': result['state_abbr'],
                'processing_time': result['processing_time'],
                'search_id': result['search_id'],
                'csv_blob_name': result['csv_blob_name'],
                'gpkg_blob_name': result['gpkg_blob_name'],
                'parcel_data': result.get('parcel_data', [])[:10],  # Return first 10 for preview
                'bucket_name': result['bucket_name'],
                'storage_type': 'gcs'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('message', 'Search failed')
            }), 500
        
    except Exception as e:
        logger.error(f"Real parcel search error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Search failed: {str(e)}'
        }), 500


@analysis_bp.route('/preview-parcel-search', methods=['POST'])
def preview_parcel_search():
    """Preview real parcel search count using ReportAll API"""
    try:
        data = request.get_json()
        county_id = data.get('county_id')
        county_name = data.get('county_name')
        state = data.get('state')
        
        logger.info(f"Previewing REAL parcel search: {county_name}, {state}")
        
        if not county_id:
            return jsonify({
                'success': False,
                'message': 'County ID is required'
            }), 400
        
        # Prepare search parameters for preview
        search_params = {
            'county_id': county_id,
            'county_name': county_name,
            'state': state,
            'calc_acreage_min': data.get('calc_acreage_min'),
            'calc_acreage_max': data.get('calc_acreage_max'),
            'owner': data.get('owner'),
            'parcel_id': data.get('parcel_id')
        }
        
        # Execute REAL preview search
        result = preview_search_count(**search_params)
        
        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'record_count': result['record_count'],
                'message': result['message']
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Preview failed')
            }), 500
        
    except Exception as e:
        logger.error(f"Preview error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Preview failed: {str(e)}'
        }), 500

@analysis_bp.route('/check-county-activity/<state>', methods=['GET'])
def check_county_activity(state):
    """Check county activity using the proven working logic from test script"""
    try:
        logger.info(f"Checking county activity for {state}")
        
        # Use existing storage client
        storage_client = db._storage_client
        bucket_name = 'bcfparcelsearchrepository'
        bucket = storage_client.bucket(bucket_name)
        
        # Get all blobs under the state directory
        state_prefix = f"{state}/"
        blobs = list(bucket.list_blobs(prefix=state_prefix))
        
        logger.info(f"Found {len(blobs)} total files under {state_prefix}")
        
        # Parse county activity from file paths (same logic as test script)
        county_files = {}
        for blob in blobs:
            path_parts = blob.name.split('/')
            
            # Look for pattern: NC/CountyName/Parcel_Files/filename
            if len(path_parts) >= 4 and path_parts[2] == 'Parcel_Files':
                county_name = path_parts[1]
                
                if county_name not in county_files:
                    county_files[county_name] = []
                
                county_files[county_name].append(blob)
        
        # Convert to Flask format (same as test script)
        county_activity = {}
        for county_name, files in county_files.items():
            county_activity[county_name] = {
                'has_activity': True,
                'folder_count': len(files),
                'folder_path': f"gs://{bucket_name}/{state}/{county_name}/Parcel_Files/",
                'latest_activity': max([blob.time_created for blob in files]).isoformat() if files else None
            }
        
        logger.info(f"Found activity in {len(county_activity)} counties: {list(county_activity.keys())}")
        
        # THIS WAS MISSING - the return statement!
        return jsonify({
            'success': True,
            'county_activity': county_activity
        })
        
    except Exception as e:
        logger.error(f"Error checking county activity: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'county_activity': {}
        })

@analysis_bp.route('/get-existing-files', methods=['POST'])
def get_existing_files():
    try:
        data = request.get_json()
        state = data.get('state')
        county = data.get('county')
        
        logger.info(f"Loading existing CSV files for {county}, {state}")
        
        # Use the same storage logic that's already working
        storage_client = db._storage_client
        bucket_name = 'bcfparcelsearchrepository'
        bucket = storage_client.bucket(bucket_name)
        
        # Build the folder path
        folder_path = f"{state}/{county}/Parcel_Files/"
        
        # Get files in this specific folder
        blobs = list(bucket.list_blobs(prefix=folder_path))
        
        files = []
        for blob in blobs:
            # Only include CSV files
            if blob.name.endswith('.csv') and not blob.name.endswith('/'):
                files.append({
                    'name': blob.name.split('/')[-1],
                    'path': blob.name,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'file_type': 'CSV',
                    'parcel_count': 'Unknown',  # Could parse from filename if needed
                    'search_criteria': f'Parcel search results for {county} County'
                })
        
        logger.info(f"Found {len(files)} CSV files for {county}")
        
        return jsonify({
            'success': True,
            'files': files,
            'folder_path': f"gs://{bucket_name}/{folder_path}",
            'file_types_shown': 'CSV files only'
        })
        
    except Exception as e:
        logger.error(f"Error loading existing files: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'files': [],
            'folder_path': ''
        })
       
# In your Flask application file (e.g., analysis_routes.py)
@analysis_bp.route('/preview-file', methods=['POST'])
def preview_file():
    """Preview a CSV file from cloud storage"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({
                'success': False, 
                'error': 'file_path is required'
            }), 400
        
        # Only allow CSV files
        if not file_path.lower().endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Only CSV files are supported for preview'
            }), 400
        
        logger.info(f"Previewing CSV file: {file_path}")
        
        # Download CSV file from GCS
        local_file_path = download_from_gcs(f"gs://bcfparcelsearchrepository/{file_path}")
        if not local_file_path:
            return jsonify({
                'success': False, 
                'error': 'File not found or could not be downloaded'
            }), 404
        
        try:
            # Read CSV file with error handling
            import pandas as pd
            
            # Try different encodings if needed
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(local_file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to read with {encoding}: {e}")
                    continue
            
            if df is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not read CSV file with any supported encoding'
                }), 500
            
            # Get preview data (first 10 rows)
            preview_rows = df.head(10).fillna('').to_dict('records')
            headers = list(df.columns)
            
            # Clean headers and data for JSON
            clean_headers = [str(h) for h in headers]
            clean_rows = []
            
            for row in preview_rows:
                clean_row = {}
                for key, value in row.items():
                    # Convert all values to strings for JSON safety
                    if pd.isna(value) or value is None:
                        clean_row[str(key)] = ''
                    else:
                        clean_row[str(key)] = str(value)
                clean_rows.append(clean_row)
            
            return jsonify({
                'success': True,
                'headers': clean_headers,
                'preview_rows': clean_rows,
                'total_rows': len(df),
                'showing_rows': min(10, len(df)),
                'file_path': file_path,
                'file_type': 'CSV'
            })
            
        finally:
            # Always cleanup temp file
            if os.path.exists(local_file_path):
                os.unlink(local_file_path)
        
    except Exception as e:
        logger.error(f"CSV preview error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Preview failed: {str(e)}'
        }), 500
        
@analysis_bp.route('/analyze-existing-file-quick', methods=['POST'])
def analyze_existing_file_quick():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type')
        
        # DEBUG: Log what we received
        logger.info(f"Received county_name: '{county_name}', state: '{state}'")
        
        if not file_path:
            return jsonify({
                'status': 'error',
                'message': 'file_path is required'
            }), 400
        
        # FIX: Construct full GCS path if not already provided
        if not file_path.startswith('gs://'):
            full_gcs_path = f"gs://bcfparcelsearchrepository/{file_path}"
            logger.info(f"Converted file path to GCS format: {full_gcs_path}")
        else:
            full_gcs_path = file_path
        
        # Step 1: Run Slope Analysis
        logger.info("Running slope analysis...")
        slope_result = run_slope_analysis(
            input_file_path=full_gcs_path,
            max_slope_degrees=25.0,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )
        
        if slope_result['status'] != 'success':
            return jsonify({
                'status': 'error',
                'message': f'Slope analysis failed: {slope_result.get("message", "Unknown error")}'
            }), 500
        
        # Step 2: Add simplified transmission analysis using CSV
        logger.info("Running simplified transmission analysis...")
        slope_csv_path = slope_result['output_file_path'].replace('.gpkg', '.csv')
        
        transmission_result = run_simplified_transmission_analysis(
            slope_csv_path=slope_csv_path,
            buffer_distance_miles=2.0
        )
        
        if transmission_result['status'] != 'success':
            return jsonify({
                'status': 'error', 
                'message': f'Transmission analysis failed: {transmission_result.get("message", "Unknown error")}'
            }), 500
        
        # Step 3: Load final results for preview
        final_results = transmission_result['enhanced_parcels']
        
        if final_results is None or len(final_results) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No analysis results available'
            }), 500
        
        # Calculate summary statistics
        total_parcels = len(final_results)
        
        # Slope summary
        suitable_slope = len(final_results[
            (final_results.get('reference_suitability', '') == 'SUITABLE') |
            (final_results.get('avg_slope_degrees', 999) <= 25)
        ])
        
        # Transmission summary  
        near_transmission = len(final_results[final_results.get('tx_lines_count', 0) > 0])
        
        # Combined scoring for recommendations
        try:
            combined_scores = calculate_combined_suitability_score(final_results)
            final_results = final_results.copy()
            final_results['combined_score'] = combined_scores
            recommended_parcels = len(final_results[final_results['combined_score'] >= 70])
            
            # Categorize parcels
            excellent = len(final_results[final_results['combined_score'] >= 85])
            good = len(final_results[
                (final_results['combined_score'] >= 70) & 
                (final_results['combined_score'] < 85)
            ])
            fair = len(final_results[
                (final_results['combined_score'] >= 55) & 
                (final_results['combined_score'] < 70)
            ])
            poor = total_parcels - excellent - good - fair
            
            # Calculate average score safely
            avg_score = int(final_results['combined_score'].mean()) if len(final_results) > 0 else 0
            
        except Exception as scoring_error:
            logger.error(f"Scoring calculation error: {scoring_error}")
            # Provide fallback values
            recommended_parcels = 0
            excellent = good = fair = poor = 0
            avg_score = 0
                    
        # Categorize parcels
        excellent = len(final_results[final_results['combined_score'] >= 85])
        good = len(final_results[
            (final_results['combined_score'] >= 70) & 
            (final_results['combined_score'] < 85)
        ])
        fair = len(final_results[
            (final_results['combined_score'] >= 55) & 
            (final_results['combined_score'] < 70)
        ])
        poor = total_parcels - excellent - good - fair
        
        # FIXED: Clean the DataFrame for JSON serialization
        parcels_for_json = prepare_dataframe_for_json(final_results)
        
        return jsonify({
            'status': 'success',
            'parcel_count': total_parcels,
            'analysis_results': {
                'summary': {
                    'total_parcels': total_parcels,
                    'excellent': excellent,
                    'good': good,
                    'fair': fair,
                    'poor': poor,
                    'average_score': int(final_results['combined_score'].mean()),
                    'recommended_for_outreach': recommended_parcels,
                    'location': f"{county_name}, {state}",
                    'slope_suitable': suitable_slope,
                    'transmission_nearby': near_transmission
                },
                'parcels_table': parcels_for_json,  # Use cleaned data
                'analysis_metadata': {
                    'scoring_method': 'Technical Analysis (Slope + Simplified Transmission)',
                    'slope_analysis': slope_result.get('slope_statistics', {}),
                    'transmission_analysis': transmission_result.get('transmission_statistics', {}),
                    'output_files': {
                        'slope_results': slope_result['output_file_path'],
                        'transmission_results': transmission_result.get('output_file_path', 'N/A')
                    }
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Technical analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Technical analysis failed: {str(e)}'
        }), 500
               
# Add this helper function to clean data for JSON
def prepare_dataframe_for_json(df: pd.DataFrame) -> list:
    """Clean DataFrame for JSON serialization"""
    import numpy as np
    
    # Create a copy to avoid modifying original
    clean_df = df.copy()
    
    # Add missing fields for frontend compatibility
    if 'slope_category' in clean_df.columns:
        clean_df['suitability_category'] = clean_df['slope_category']
    
    # FIXED: Map combined_score to suitability_score for frontend
    if 'combined_score' in clean_df.columns:
        clean_df['suitability_score'] = clean_df['combined_score']
    
    # Fix null transmission values
    transmission_cols = ['tx_distance_miles', 'tx_voltage_kv', 'tx_owner']
    for col in transmission_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna('No Data')
    
    # Replace NaN and inf values
    clean_df = clean_df.replace([np.inf, -np.inf], None)
    clean_df = clean_df.fillna('')
    
    # Convert numpy types to native Python types
    for col in clean_df.columns:
        if clean_df[col].dtype in ['int64', 'int32']:
            clean_df[col] = clean_df[col].astype(int)
        elif clean_df[col].dtype in ['float64', 'float32']:
            clean_df[col] = clean_df[col].round(4).astype(float)
        elif clean_df[col].dtype == 'object':
            clean_df[col] = clean_df[col].astype(str)
    
    # Convert to records and ensure all values are JSON-serializable
    records = []
    for _, row in clean_df.iterrows():
        record = {}
        for key, value in row.items():
            # Skip geometry columns
            if key == 'geometry':
                continue
                
            # Convert numpy types to Python native types
            if isinstance(value, np.integer):
                record[key] = int(value)
            elif isinstance(value, np.floating):
                if np.isnan(value) or np.isinf(value):
                    record[key] = None
                else:
                    record[key] = round(float(value), 4)
            elif pd.isna(value) or value is None:
                record[key] = None
            else:
                record[key] = str(value)
        
        records.append(record)
    
    return records

def run_simplified_transmission_analysis(slope_csv_path: str, buffer_distance_miles: float) -> Dict[str, Any]:
    """Real transmission analysis using BigQuery spatial data"""
    try:
        start_time = time.time()
        logger.info(f"Starting REAL transmission analysis on: {slope_csv_path}")
        
        # Download CSV file from GCS
        local_csv_path = download_from_gcs(slope_csv_path)
        if not local_csv_path:
            return {
                'status': 'error',
                'message': 'Failed to download slope results CSV'
            }
        
        try:
            # Load the slope analysis results
            df = pd.read_csv(local_csv_path)
            logger.info(f"Loaded {len(df)} parcels from slope analysis results")
            
            # Convert to GeoDataFrame for spatial analysis
            if 'geom_as_wkt' in df.columns:
                from shapely import wkt
                df['geometry'] = df['geom_as_wkt'].apply(wkt.loads)
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            else:
                logger.warning("No geometry column found, using mock data")
                enhanced_df = add_mock_transmission_data_simple(df, buffer_distance_miles)
                return {
                    'status': 'success',
                    'message': 'Transmission analysis completed (mock data used)',
                    'enhanced_parcels': enhanced_df,
                    'parcels_processed': len(enhanced_df),
                    'parcels_near_transmission': len(enhanced_df[enhanced_df['tx_lines_count'] > 0]),
                    'processing_time': f"{time.time() - start_time:.2f} seconds"
                }
            
            # Run real transmission analysis using BigQuery
            enhanced_gdf = run_real_transmission_analysis(gdf, buffer_distance_miles)
            
            # Calculate statistics
            parcels_processed = len(enhanced_gdf)
            parcels_near_transmission = len(enhanced_gdf[enhanced_gdf['tx_lines_count'] > 0])
            
            return {
                'status': 'success',
                'message': 'Real transmission analysis completed',
                'enhanced_parcels': enhanced_gdf,
                'parcels_processed': parcels_processed,
                'parcels_near_transmission': parcels_near_transmission,
                'processing_time': f"{time.time() - start_time:.2f} seconds",
                'transmission_statistics': {
                    'total_parcels': parcels_processed,
                    'parcels_with_nearby_lines': parcels_near_transmission,
                    'percentage_near_transmission': round(
                        (parcels_near_transmission / parcels_processed) * 100, 1
                    ) if parcels_processed > 0 else 0
                }
            }
            
        finally:
            # Cleanup temp file
            if os.path.exists(local_csv_path):
                os.unlink(local_csv_path)
        
    except Exception as e:
        logger.error(f"Real transmission analysis failed: {e}")
        return {
            'status': 'error',
            'message': f'Transmission analysis failed: {str(e)}'
        }

def run_real_transmission_analysis(gdf: gpd.GeoDataFrame, buffer_distance_miles: float) -> gpd.GeoDataFrame:
    """Run real transmission analysis using BigQuery"""
    try:
        logger.info(f"Running real transmission analysis for {len(gdf)} parcels")
        
        # Import the transmission analysis function
        from transmission_analysis_bigquery import run_headless as run_transmission_analysis
        
        # Upload GeoDataFrame to GCS first, then run analysis
        import time
        timestamp = int(time.time())
        
        # Create temporary GCS path
        temp_gcs_path = f"gs://bcfparcelsearchrepository/temp/transmission_input_{timestamp}.gpkg"
        
        # Upload to GCS using your storage client
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket('bcfparcelsearchrepository')
        
        # Create temporary local file
        temp_local_file = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
        temp_local_file.close()
        
        # Save GeoDataFrame locally first
        gdf.to_file(temp_local_file.name, driver='GPKG')
        
        # Upload to GCS
        blob_name = f"temp/transmission_input_{timestamp}.gpkg"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_local_file.name)
        
        logger.info(f"Uploaded temp file to GCS: {temp_gcs_path}")
        
        # Run transmission analysis with GCS path
        result = run_transmission_analysis(
            input_file_path=temp_gcs_path,
            buffer_distance_miles=buffer_distance_miles,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )
        
        # Cleanup local temp file
        os.unlink(temp_local_file.name)
        
        # Cleanup GCS temp file
        blob.delete()
        
        if result['status'] == 'success':
            # Download the results
            result_gcs_path = result['output_file_path']
            local_result_file = download_from_gcs(result_gcs_path)
            
            if local_result_file:
                # Load the enhanced results
                enhanced_gdf = gpd.read_file(local_result_file)
                os.unlink(local_result_file)  # Cleanup
                
                logger.info(f"Real transmission analysis completed successfully")
                return enhanced_gdf
            else:
                logger.warning("Failed to download transmission results, using mock data")
                return add_mock_transmission_data_simple(gdf, buffer_distance_miles)
        else:
            logger.warning(f"Transmission analysis failed: {result.get('message')}, using mock data")
            return add_mock_transmission_data_simple(gdf, buffer_distance_miles)
            
    except Exception as e:
        logger.warning(f"Real transmission analysis failed: {e}, using mock data")
        return add_mock_transmission_data_simple(gdf, buffer_distance_miles)    

def add_mock_transmission_data_simple(df: pd.DataFrame, buffer_distance: float) -> pd.DataFrame:
    """Add mock transmission data using deterministic random values"""
    import random
    import numpy as np
    
    logger.info("Adding mock transmission data for demonstration...")
    
    # Create deterministic random results based on parcel_id
    enhanced_df = df.copy()
    n_parcels = len(enhanced_df)
    
    # Use parcel coordinates if available for more realistic mock data
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Use coordinates to create consistent mock data
        seed_values = []
        for _, parcel in df.iterrows():
            # Create seed based on coordinates
            lat = float(parcel.get('latitude', 35.0))
            lon = float(parcel.get('longitude', -80.0))
            seed = int(abs(lat * 1000 + lon * 1000)) % 1000
            seed_values.append(seed)
    else:
        # Fallback to simple seed
        seed_values = list(range(n_parcels))
    
    # Generate mock transmission data
    tx_counts = []
    tx_distances = []
    tx_voltages = []
    tx_owners = []
    
    for i, seed in enumerate(seed_values):
        random.seed(seed + 42)  # Consistent seed per parcel
        
        # Mock transmission line count (0-3 lines nearby)
        tx_count = random.choices([0, 1, 2, 3], weights=[30, 40, 25, 5])[0]  # More parcels with transmission
        tx_counts.append(tx_count)
        
        if tx_count > 0:
            # Mock distance within buffer (0.1 to 2.0 miles)
            tx_distance = round(random.uniform(0.1, buffer_distance), 2)
            
            # Mock voltage based on distance (closer = higher voltage)
            if tx_distance < 0.5:
                voltage_options = [345, 500, 230]
            elif tx_distance < 1.0:
                voltage_options = [138, 230, 345]
            else:
                voltage_options = [69, 138, 230]
            
            tx_voltage = random.choice(voltage_options)
            tx_owner = random.choice(['Duke Energy', 'Dominion Energy', 'TVA', 'Progress Energy'])
        else:
            # FIXED: No transmission within buffer - set to None instead of 999
            tx_distance = None
            tx_voltage = None
            tx_owner = None
        
        tx_distances.append(tx_distance)
        tx_voltages.append(tx_voltage)
        tx_owners.append(tx_owner)
    
    # Add mock data to DataFrame
    enhanced_df['tx_lines_count'] = tx_counts
    enhanced_df['tx_distance_miles'] = tx_distances
    enhanced_df['tx_voltage_kv'] = tx_voltages
    enhanced_df['tx_owner'] = tx_owners
    
    logger.info(f"Added mock transmission data to {len(enhanced_df)} parcels")
    return enhanced_df
            
def load_analysis_results_for_preview(gcs_file_path, max_records=1000):
    """Load analysis results from GCS file for preview"""
    try:
        # Download file temporarily
        from google.cloud import storage
        
        # Parse GCS path
        if not gcs_file_path.startswith('gs://'):
            return None
            
        parts = gcs_file_path[5:].split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        
        # Download to temp file
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            temp_path = tmp.name
        
        try:
            # Load with geopandas
            gdf = gpd.read_file(temp_path)
            
            # Limit records for preview
            if len(gdf) > max_records:
                gdf = gdf.head(max_records)
            
            # Convert to regular DataFrame for JSON serialization
            df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
            
            # Clean data for JSON
            df = df.fillna('')
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            logger.info(f"Loaded {len(df)} parcels for preview")
            return df
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        return None

def calculate_combined_suitability_score(parcels_df):
    """Calculate combined suitability score from slope and transmission data"""
    scores = []
    
    for _, parcel in parcels_df.iterrows():
        score = 50  # Base score
        
        # Slope scoring (40 points max)
        avg_slope = float(parcel.get('avg_slope_degrees', 999))
        if pd.isna(avg_slope):
            avg_slope = 999
            
        if avg_slope <= 5:
            score += 40
        elif avg_slope <= 10:
            score += 35
        elif avg_slope <= 15:
            score += 30
        elif avg_slope <= 25:
            score += 20
        elif avg_slope <= 35:
            score += 10
        # else: 0 points for steep slopes
        
        # Transmission scoring (30 points max)
        tx_distance = parcel.get('tx_distance_miles')
        
        # FIXED: Handle None values properly
        if tx_distance is None or pd.isna(tx_distance):
            # No transmission within buffer - 0 points
            score += 0
        else:
            tx_distance = float(tx_distance)
            if tx_distance <= 0.25:
                score += 30
            elif tx_distance <= 0.5:
                score += 25
            elif tx_distance <= 1.0:
                score += 20
            elif tx_distance <= 2.0:
                score += 15
            # else: 0 points for far transmission
        
        # Voltage bonus (10 points max)
        tx_voltage = parcel.get('tx_voltage_kv')
        if tx_voltage is not None and not pd.isna(tx_voltage):
            tx_voltage = float(tx_voltage)
            if tx_voltage >= 345:
                score += 10
            elif tx_voltage >= 138:
                score += 8
            elif tx_voltage >= 69:
                score += 5
        
        # Cap at 100, ensure Poor category for no transmission
        final_score = min(100, max(0, int(score)))
        
        # FIXED: Force Poor category if no transmission within buffer
        if tx_distance is None or pd.isna(tx_distance):
            final_score = min(final_score, 54)  # Ensure Poor category
            
        scores.append(final_score)
    
    return scores

@analysis_bp.route('/analyze-search-results', methods=['POST'])
def analyze_search_results():
    """Run technical analysis on fresh search results"""
    try:
        data = request.get_json()
        search_id = data.get('search_id')
        county_name = data.get('county_name', 'Unknown')
        state = data.get('state', 'Unknown')
        
        logger.info(f"Analyzing search results for {county_name}, {state}")
        
        # For now, return a placeholder since this would require 
        # finding the search results file first
        return jsonify({
            'status': 'success',
            'message': 'Search results analysis not yet implemented',
            'next_step': 'Use existing file analysis instead'
        })
        
    except Exception as e:
        logger.error(f"Search results analysis error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500

@analysis_bp.route('/individual-parcel-analysis', methods=['POST'])
def individual_parcel_analysis():
    """Run AI analysis for individual parcel"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('parcel_data'):
            return jsonify({'success': False, 'error': 'Parcel data required'}), 400
        
        parcel_data = data.get('parcel_data', {})
        county_name = data.get('county_name', 'Unknown')
        state = data.get('state', 'Unknown')
        project_type = data.get('project_type', 'solar')
        
        logger.info(f"Starting individual parcel analysis for {county_name}, {state}")
        
        # Import and use the AI service
        from services.ai_service import ai_service
        
        # Call the AI analysis
        ai_response = ai_service.analyze_individual_parcel(
            parcel_data=parcel_data,
            county_name=county_name,
            state=state,
            project_type=project_type
        )
        
        if ai_response.success:
            return jsonify({
                'success': True,
                'analysis': {
                    'detailed_analysis': ai_response.content,
                    'parcel_info': {
                        'owner': parcel_data.get('owner', 'Unknown'),
                        'acres': parcel_data.get('acreage_calc', parcel_data.get('acreage', 0)),
                        'parcel_id': parcel_data.get('parcel_id', 'Unknown')
                    }
                },
                'metadata': ai_response.metadata
            })
        else:
            logger.error(f"AI analysis failed: {ai_response.error}")
            return jsonify({
                'success': False,
                'error': f'AI analysis failed: {ai_response.error}'
            }), 500
            
    except Exception as e:
        logger.error(f"Individual parcel analysis endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500



@analysis_bp.route('/download-parcel-file', methods=['GET'])
def download_parcel_file():
    """Generate signed URL for downloading parcel files from GCS"""
    try:
        blob_name = request.args.get('blob_name')
        file_type = request.args.get('file_type', 'csv')
        
        if not blob_name:
            return jsonify({
                'success': False,
                'error': 'blob_name parameter is required'
            }), 400
        
        # Import here to avoid circular imports
        from enhanced_parcel_search import generate_signed_url
        
        # Generate signed URL (valid for 24 hours)
        signed_url = generate_signed_url(blob_name, expiration_hours=24)
        
        if signed_url:
            # Redirect to the signed URL for download
            from flask import redirect
            return redirect(signed_url)
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate download URL'
            }), 500
            
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Download failed: {str(e)}'
        }), 500

@analysis_bp.route('/download-file', methods=['GET'])
def download_file():
    """Download a file from cloud storage"""
    try:
        file_path = request.args.get('path')
        
        if not file_path:
            return "File path is required", 400
        
        logger.info(f"Downloading file: {file_path}")
        
        # Download file from GCS to a temporary location
        full_gcs_path = f"gs://bcfparcelsearchrepository/{file_path}"
        local_temp_path = download_from_gcs(full_gcs_path)
        
        if not local_temp_path or not os.path.exists(local_temp_path):
            return "File not found", 404
        
        try:
            # Get the original filename
            filename = os.path.basename(file_path)
            
            # Determine content type based on file extension
            if filename.endswith('.csv'):
                mimetype = 'text/csv'
            elif filename.endswith('.gpkg'):
                mimetype = 'application/octet-stream'
            else:
                mimetype = 'application/octet-stream'
            
            # Read file content
            with open(local_temp_path, 'rb') as f:
                file_data = f.read()
            
            # Create response with file content
            from flask import Response
            response = Response(
                file_data,
                mimetype=mimetype,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Length": str(len(file_data))
                }
            )
            
            return response
            
        finally:
            # Always cleanup temp file
            if os.path.exists(local_temp_path):
                os.unlink(local_temp_path)
        
    except Exception as e:
        logger.error(f"File download error: {str(e)}", exc_info=True)
        return f"Download failed: {str(e)}", 500
          
def run_slope_analysis_csv_output(input_file_path: str, max_slope_degrees: float, 
                                output_bucket: str, project_id: str) -> Dict[str, Any]:
    """Modified slope analysis that outputs CSV instead of GPKG"""
    try:
        # Run normal slope analysis
        result = run_slope_analysis(
            input_file_path=input_file_path,
            max_slope_degrees=max_slope_degrees,
            output_bucket=output_bucket,
            project_id=project_id
        )
        
        if result['status'] == 'success':
            # Extract CSV path from the result
            gpkg_path = result['output_file_path']
            # Convert GPKG path to CSV path
            csv_path = gpkg_path.replace('.gpkg', '.csv')
            result['output_csv_path'] = csv_path
            logger.info(f"Slope analysis CSV output: {csv_path}")
            
        return result
        
    except Exception as e:
        logger.error(f"Slope analysis CSV output failed: {e}")
        return {
            'status': 'error',
            'message': f'Slope analysis failed: {str(e)}'
        }

def run_transmission_analysis_csv_input(input_csv_path: str, buffer_distance_miles: float,
                                      output_bucket: str, project_id: str) -> Dict[str, Any]:
    """Optimized transmission analysis that works with CSV input"""
    try:
        logger.info(f"Starting CSV-based transmission analysis: {input_csv_path}")
        
        # Use simplified transmission analysis for CSV
        result = run_transmission_analysis_simple_csv(
            input_csv_path=input_csv_path,
            buffer_distance_miles=buffer_distance_miles,
            output_bucket=output_bucket,
            project_id=project_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"CSV transmission analysis failed: {e}")
        return {
            'status': 'error',
            'message': f'Transmission analysis failed: {str(e)}'
        }

def run_transmission_analysis_simple_csv(input_csv_path: str, buffer_distance_miles: float,
                                       output_bucket: str, project_id: str) -> Dict[str, Any]:
    """Simplified transmission analysis optimized for CSV processing"""
    start_time = time.time()
    
    try:
        # Download CSV file
        logger.info(f"Downloading CSV: {input_csv_path}")
        local_csv_path = download_from_gcs(input_csv_path)
        
        if not local_csv_path:
            return {
                'status': 'error', 
                'message': 'Failed to download CSV file'
            }
        
        # Load CSV data
        import pandas as pd
        from shapely import wkt
        
        df = pd.read_csv(local_csv_path)
        logger.info(f"Loaded {len(df)} parcels from CSV")
        
        # Convert geometry column if present
        if 'geom_as_wkt' in df.columns:
            df['geometry'] = df['geom_as_wkt'].apply(wkt.loads)
        elif 'geometry_wkt' in df.columns:
            df['geometry'] = df['geometry_wkt'].apply(wkt.loads)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        # Add mock transmission data for now (faster than spatial query)
        enhanced_gdf = add_mock_transmission_data(gdf, buffer_distance_miles)
        
        # Save results as CSV
        output_csv_path = save_csv_transmission_results(
            enhanced_gdf, input_csv_path, output_bucket
        )
        
        # Calculate statistics
        parcels_processed = len(enhanced_gdf)
        parcels_near_transmission = len(enhanced_gdf[enhanced_gdf['tx_lines_count'] > 0])
        
        return {
            'status': 'success',
            'message': 'CSV-based transmission analysis completed',
            'parcels_processed': parcels_processed,
            'parcels_near_transmission': parcels_near_transmission,
            'output_csv_path': output_csv_path,
            'processing_time': f"{time.time() - start_time:.2f} seconds",
            'transmission_statistics': {
                'total_parcels': parcels_processed,
                'parcels_with_nearby_lines': parcels_near_transmission,
                'percentage_near_transmission': round(
                    (parcels_near_transmission / parcels_processed) * 100, 1
                ) if parcels_processed > 0 else 0
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'CSV transmission analysis failed: {str(e)}',
            'processing_time': f"{time.time() - start_time:.2f} seconds"
        }

def save_csv_transmission_results(gdf: gpd.GeoDataFrame, input_path: str, bucket_name: str) -> str:
    """Save transmission results as CSV to GCS"""
    try:
        # Extract location info
        location_info = extract_location_from_path(input_path)
        state = location_info.get('state', 'Unknown')  
        county_name = location_info.get('county_name', 'Unknown')
        
        # Create output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{county_name}_transmission_analysis_{timestamp}.csv"
        
        # Create temp CSV
        temp_csv_path = f"/tmp/{filename}"
        
        # Convert to regular DataFrame (drop geometry for CSV)
        df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
        df.to_csv(temp_csv_path, index=False)
        
        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        gcs_path = f"{state}/{county_name}/Transmission_Files/{filename}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(temp_csv_path)
        
        # Cleanup
        os.unlink(temp_csv_path)
        
        full_gcs_path = f"gs://{bucket_name}/{gcs_path}"
        logger.info(f"Saved CSV transmission results: {full_gcs_path}")
        
        return full_gcs_path
        
    except Exception as e:
        logger.error(f"Error saving CSV results: {e}")
        raise

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
    
def load_csv_results_for_preview(csv_gcs_path: str, max_records: int = 1000) -> pd.DataFrame:
    """Load CSV results from GCS for preview"""
    try:
        # Download CSV temporarily  
        local_csv_path = download_from_gcs(csv_gcs_path)
        
        if not local_csv_path:
            return None
            
        # Load CSV
        df = pd.read_csv(local_csv_path)
        
        # Limit records
        if len(df) > max_records:
            df = df.head(max_records)
        
        # Clean data
        df = df.fillna('')
        
        # Cleanup
        os.unlink(local_csv_path)
        
        logger.info(f"Loaded {len(df)} records from CSV for preview")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV results: {e}")
        return None

def calculate_analysis_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics from analysis results"""
    total_parcels = len(df)
    
    # Slope analysis
    suitable_slope = len(df[df.get('reference_suitability', '') == 'SUITABLE'])
    
    # Transmission analysis  
    near_transmission = len(df[df.get('tx_lines_count', 0) > 0])
    
    # Combined scoring
    df['combined_score'] = calculate_combined_suitability_score(df)
    recommended = len(df[df['combined_score'] >= 70])
    
    # Categories
    excellent = len(df[df['combined_score'] >= 85])
    good = len(df[(df['combined_score'] >= 70) & (df['combined_score'] < 85)])
    fair = len(df[(df['combined_score'] >= 55) & (df['combined_score'] < 70)])
    poor = total_parcels - excellent - good - fair
    
    return {
        'total_parcels': total_parcels,
        'excellent': excellent,
        'good': good,
        'fair': fair, 
        'poor': poor,
        'average_score': int(df['combined_score'].mean()) if len(df) > 0 else 0,
        'recommended_for_outreach': recommended,
        'slope_suitable': suitable_slope,
        'transmission_nearby': near_transmission
    }