# enhanced_parcel_search.py - Fixed GCS integration

import geopandas as gpd
import pandas as pd
import os
import logging
import time
import uuid
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
import tempfile
import json

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def get_gcs_client():
    """
    Initialize Google Cloud Storage client with proper service account credentials
    """
    try:
        # Try different credential methods

        # Method 1: Service account key file path from environment
        service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if service_account_path and os.path.exists(service_account_path):
            credentials = service_account.Credentials.from_service_account_file(service_account_path)
            client = storage.Client(credentials=credentials)
            logger.info("GCS client initialized with service account file")
            return client

        # Method 2: Service account key JSON from environment variable
        service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        if service_account_json:
            try:
                # Parse JSON string
                service_account_info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
                client = storage.Client(credentials=credentials)
                logger.info("GCS client initialized with service account JSON from env var")
                return client
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON: {e}")

        # Method 3: Default credentials (for App Engine, Compute Engine, etc.)
        try:
            client = storage.Client()
            logger.info("GCS client initialized with default credentials")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize with default credentials: {e}")

        # If all methods fail
        logger.error("No valid GCS credentials found")
        return None

    except Exception as e:
        logger.error(f"Error initializing GCS client: {str(e)}")
        return None


def save_files_to_gcs(gdf, df, state_abbr, county_name, gpkg_filename, csv_filename):
    """
    Save files to Google Cloud Storage with proper path structure:
    gs://bcfparcelsearchrepository/{STATE}/{COUNTY}/Parcel_Files/
    """
    try:
        # Get bucket name from environment
        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')

        # Initialize GCS client with proper credentials
        client = get_gcs_client()
        if not client:
            return {
                'success': False,
                'error': 'Failed to initialize Google Cloud Storage client. Please check your credentials.'
            }

        bucket = client.bucket(bucket_name)

        # Create GCS path structure: {STATE}/{COUNTY}/Parcel_Files/
        gcs_folder = f"{state_abbr}/{county_name}/Parcel_Files/"

        logger.info(f"Saving files to GCS bucket: gs://{bucket_name}/{gcs_folder}")

        # Use temporary files for upload
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save GPKG temporarily
            temp_gpkg_path = os.path.join(temp_dir, gpkg_filename)
            gdf.to_file(temp_gpkg_path, driver='GPKG')

            # Save CSV temporarily
            temp_csv_path = os.path.join(temp_dir, csv_filename)
            df.to_csv(temp_csv_path, index=False)

            # Upload GPKG to GCS
            gpkg_blob_name = f"{gcs_folder}{gpkg_filename}"
            gpkg_blob = bucket.blob(gpkg_blob_name)
            gpkg_blob.upload_from_filename(temp_gpkg_path)

            # Upload CSV to GCS
            csv_blob_name = f"{gcs_folder}{csv_filename}"
            csv_blob = bucket.blob(csv_blob_name)
            csv_blob.upload_from_filename(temp_csv_path)

            # Set content types for proper handling
            gpkg_blob.content_type = 'application/geopackage+sqlite3'
            csv_blob.content_type = 'text/csv'
            gpkg_blob.patch()
            csv_blob.patch()

            logger.info(f"Successfully uploaded files to GCS")
            logger.info(f"GPKG: gs://{bucket_name}/{gpkg_blob_name}")
            logger.info(f"CSV: gs://{bucket_name}/{csv_blob_name}")

            # For preview/download, we'll use the blob names and generate signed URLs in the routes
            return {
                'success': True,
                'gpkg_url': f"gs://{bucket_name}/{gpkg_blob_name}",
                'csv_url': f"gs://{bucket_name}/{csv_blob_name}",
                'bucket_path': f"gs://{bucket_name}/{gcs_folder}",
                'gpkg_blob_name': gpkg_blob_name,
                'csv_blob_name': csv_blob_name,
                'bucket_name': bucket_name
            }

    except Exception as e:
        logger.error(f"Error saving files to GCS: {str(e)}")
        return {
            'success': False,
            'error': f'Failed to save to Google Cloud Storage: {str(e)}'
        }


def generate_signed_url(blob_name, bucket_name=None, expiration_hours=24):
    """
    Generate a signed URL for a GCS blob
    """
    try:
        if not bucket_name:
            bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')

        client = get_gcs_client()
        if not client:
            logger.error("Cannot generate signed URL: GCS client not available")
            return None

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.error(f"Blob does not exist: {blob_name}")
            return None

        # Generate signed URL
        expiration = datetime.utcnow() + timedelta(hours=expiration_hours)
        signed_url = blob.generate_signed_url(expiration=expiration, method='GET')

        logger.info(f"Generated signed URL for {blob_name}")
        return signed_url

    except Exception as e:
        logger.error(f"Error generating signed URL for {blob_name}: {str(e)}")
        return None


def run_headless(**search_params):
    """
    Execute parcel search and create both .gpkg and .csv files, saving to GCS
    """

    start_time = time.time()

    try:
        logger.info(f"Starting parcel search with params: {search_params}")

        # Extract parameters
        county_id = search_params.get('county_id')
        pipeline_id = search_params.get('pipeline_id', str(uuid.uuid4()))
        user_id = search_params.get('user_id', 'default_user')

        if not county_id:
            return {
                'status': 'error',
                'message': 'County ID is required for parcel search'
            }

        # Build search criteria for ReportAll API
        search_criteria = {}

        # Add acreage criteria
        if search_params.get('calc_acreage_min'):
            search_criteria['acreage_min'] = search_params['calc_acreage_min']
        if search_params.get('calc_acreage_max'):
            search_criteria['acreage_max'] = search_params['calc_acreage_max']

        # Add owner search
        if search_params.get('owner'):
            search_criteria['owner'] = search_params['owner']

        # Add parcel ID search
        if search_params.get('parcel_id'):
            search_criteria['parcel_id'] = search_params['parcel_id']

        # Execute the ReportAll API search
        logger.info(f"Executing ReportAll search for county {county_id} with criteria: {search_criteria}")

        search_result = execute_reportall_search(county_id, search_criteria)

        if not search_result.get('success'):
            return {
                'status': 'error',
                'message': f"ReportAll search failed: {search_result.get('error', 'Unknown error')}"
            }

        # Get the parcel data
        parcel_data = search_result.get('data', [])
        record_count = len(parcel_data)

        logger.info(f"ReportAll search returned {record_count} parcels")

        if record_count == 0:
            return {
                'status': 'success',
                'message': 'No parcels found matching search criteria',
                'record_count': 0,
                'file_path': '',
                'csv_file_path': '',
                'preview_url': '',
                'download_url': '',
                'parcel_data': [],
                'processing_time': time.time() - start_time,
                'limit': 100
            }

        # Get county and state info
        county_name = search_result.get('county_name', 'Unknown')
        state_abbr = search_result.get('state_abbr', 'XX')

        # Create timestamp and filenames
        timestamp = datetime.now().strftime('%m%d%Y_%H%M')
        base_filename = f"{county_name}_{state_abbr}_parcels_{timestamp}"
        gpkg_filename = f"{base_filename}.gpkg"
        csv_filename = f"{base_filename}.csv"

        # Convert parcel data to GeoDataFrame
        gdf = create_geodataframe_from_parcels(parcel_data)

        if gdf is not None and len(gdf) > 0:
            # Create CSV version (without geometry for CSV)
            df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf

            # Clean the data for JSON serialization
            df_clean = df.copy()
            df_clean = df_clean.fillna('')  # Replace NaN with empty strings
            df_clean = df_clean.replace([float('inf'), float('-inf')], '')  # Replace infinity values

            # Convert any datetime columns to strings
            for col in df_clean.columns:
                if df_clean[col].dtype == 'datetime64[ns]':
                    df_clean[col] = df_clean[col].astype(str)
                elif df_clean[col].dtype == 'object':
                    # Convert any remaining problematic objects to strings
                    df_clean[col] = df_clean[col].astype(str)

            # Save files to Google Cloud Storage
            gcs_result = save_files_to_gcs(
                gdf=gdf,
                df=df,  # Use original df for file saving
                state_abbr=state_abbr,
                county_name=county_name,
                gpkg_filename=gpkg_filename,
                csv_filename=csv_filename
            )

            if not gcs_result['success']:
                return {
                    'status': 'error',
                    'message': f"Failed to save files to cloud storage: {gcs_result['error']}"
                }

            # Prepare result with GCS paths (not signed URLs yet)
            processing_time = time.time() - start_time

            result = {
                'status': 'success',
                'message': f'Successfully found {record_count} parcels',
                'record_count': record_count,
                'file_path': gcs_result['gpkg_url'],
                'file_name': gpkg_filename,
                'csv_file_path': gcs_result['csv_url'],
                'csv_file_name': csv_filename,
                'parcel_data': df_clean.to_dict('records'),  # Use cleaned data
                'total_records': len(df),
                'county_name': county_name,
                'state_abbr': state_abbr,
                'processing_time': round(processing_time, 2),
                'search_id': pipeline_id,
                'storage_type': 'gcs',
                'bucket_path': gcs_result['bucket_path'],
                'bucket_name': gcs_result['bucket_name'],
                'gpkg_blob_name': gcs_result['gpkg_blob_name'],
                'csv_blob_name': gcs_result['csv_blob_name']
            }

            logger.info(f"Parcel search completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Files saved to: {gcs_result['bucket_path']}")
            return result

        else:
            return {
                'status': 'error',
                'message': 'Failed to create spatial data from parcel results'
            }

    except Exception as e:
        logger.error(f"Parcel search failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'status': 'error',
            'message': f'Parcel search failed: {str(e)}'
        }


def execute_reportall_search(county_id, search_criteria):
    """
    Execute the actual ReportAll API search with correct parameters
    """
    try:
        # Get API credentials from environment variables
        api_key = os.getenv('RAUSA_CLIENT_KEY')
        api_url = os.getenv('RAUSA_API_URL', 'https://reportallusa.com/api/parcels')
        api_version = os.getenv('RAUSA_API_VERSION', '9')

        if not api_key:
            return {
                'success': False,
                'error': 'RAUSA_CLIENT_KEY not found in environment variables'
            }

        # Build API request with correct parameter names
        api_params = {
            'client': api_key,
            'v': api_version,
            'county_id': county_id,
            'region': 'county',
            'attribute': 'acreage',
            'format': 'json',
            'rpp': 1000,  # ✅ CORRECT PARAMETER NAME
            'acreage_min': search_criteria['acreage_min']
        }

        # Add search criteria to API params
        # ReportAll requires at least one attribute filter for the search to work
        has_attribute_filter = False

        if search_criteria.get('acreage_min'):
            api_params['acreage_min'] = search_criteria['acreage_min']
            has_attribute_filter = True
        if search_criteria.get('acreage_max'):
            api_params['acreage_max'] = search_criteria['acreage_max']
            has_attribute_filter = True
        if search_criteria.get('owner'):
            api_params['owner'] = search_criteria['owner']
            has_attribute_filter = True
        if search_criteria.get('parcel_id'):
            api_params['parcel_id'] = search_criteria['parcel_id']
            has_attribute_filter = True

        # If no specific filters, add a minimal acreage filter to make the query valid
        if not has_attribute_filter:
            api_params['acreage_min'] = 0.1  # Get all parcels with at least 0.1 acres

        logger.info(f"Making ReportAll API request to {api_url}")

        # Make API request
        response = requests.get(api_url, params=api_params, timeout=60)

        if response.status_code == 200:
            data = response.json()

            logger.info(f"🔍 API Request URL: {response.url}")
            logger.info(f"🔍 Request params sent: {api_params}")

            # Add this right after data = response.json():
            with open('/tmp/api_response_debug.json', 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"🔍 Full API response saved to /tmp/api_response_debug.json")


            # Process the ReportAll API response format
            if data.get('status') == 'OK':
                parcels = data.get('results', [])

                # Get county info from the first parcel
                county_info = {
                    'name': 'Unknown',
                    'state_abbr': 'XX'
                }

                if parcels:
                    first_parcel = parcels[0]
                    county_info = {
                        'name': first_parcel.get('county_name', 'Unknown'),
                        'state_abbr': first_parcel.get('state_abbr', 'XX')
                    }

                logger.info(f"ReportAll API returned {len(parcels)} parcels")

                return {
                    'success': True,
                    'data': parcels,
                    'county_name': county_info['name'],
                    'state_abbr': county_info['state_abbr']
                }
            else:
                return {
                    'success': False,
                    'error': f'API returned error status: {data.get("status")} - {data.get("message", "Unknown error")}'
                }
        else:
            return {
                'success': False,
                'error': f'API request failed with status {response.status_code}: {response.text}'
            }

    except Exception as e:
        logger.error(f"ReportAll API error: {str(e)}")
        return {
            'success': False,
            'error': f'API request failed: {str(e)}'
        }


def create_geodataframe_from_parcels(parcel_data):
    """
    Convert ReportAll parcel data to GeoDataFrame
    """
    try:
        if not parcel_data:
            return None

        # Convert to DataFrame first
        df = pd.DataFrame(parcel_data)

        logger.info(f"Creating GeoDataFrame from {len(df)} parcels")
        logger.info(f"Parcel data columns: {list(df.columns)}")

        # ReportAll provides latitude and longitude columns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Convert lat/lon strings to float and handle any null values
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

            # Remove rows with invalid coordinates
            df = df.dropna(subset=['latitude', 'longitude'])

            if len(df) > 0:
                # Create GeoDataFrame with point geometries
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs='EPSG:4326'
                )
                logger.info(f"Created GeoDataFrame with {len(gdf)} valid parcels")
                return gdf
            else:
                logger.warning("No parcels with valid coordinates found")
                return None
        else:
            # No spatial data available
            logger.warning("No latitude/longitude columns found in parcel data")
            return pd.DataFrame(parcel_data)  # Return as regular DataFrame

    except Exception as e:
        logger.error(f"Error creating GeoDataFrame: {str(e)}")
        return None


def preview_search_count(**search_params):
    """
    Preview search to get estimated count without full data
    """
    try:
        logger.info(f"Preview search with params: {search_params}")

        county_id = search_params.get('county_id')

        if not county_id:
            return {
                'status': 'error',
                'message': 'County ID is required'
            }

        # Build search criteria
        search_criteria = {}
        if search_params.get('calc_acreage_min'):
            search_criteria['acreage_min'] = search_params['calc_acreage_min']
        if search_params.get('calc_acreage_max'):
            search_criteria['acreage_max'] = search_params['calc_acreage_max']
        if search_params.get('owner'):
            search_criteria['owner'] = search_params['owner']
        if search_params.get('parcel_id'):
            search_criteria['parcel_id'] = search_params['parcel_id']

        # Get API credentials
        api_key = os.getenv('RAUSA_CLIENT_KEY')
        api_url = os.getenv('RAUSA_API_URL', 'https://reportallusa.com/api/parcels')
        api_version = os.getenv('RAUSA_API_VERSION', '9')

        api_params = {
            'client': api_key,
            'v': api_version,
            'county_id': county_id,
            'region': 'county',
            'attribute': 'acreage',
            'format': 'json',
            'limit': 1  # Just get 1 record to see total count
        }

        # Add search criteria
        has_filter = False
        if search_criteria.get('acreage_min'):
            api_params['acreage_min'] = search_criteria['acreage_min']
            has_filter = True
        if search_criteria.get('acreage_max'):
            api_params['acreage_max'] = search_criteria['acreage_max']
            has_filter = True
        if search_criteria.get('owner'):
            api_params['owner'] = search_criteria['owner']
            has_filter = True
        if search_criteria.get('parcel_id'):
            api_params['parcel_id'] = search_criteria['parcel_id']
            has_filter = True

        if not has_filter:
            api_params['acreage_min'] = 0.1

        response = requests.get(api_url, params=api_params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                total_count = data.get('count', 0)
                return {
                    'status': 'success',
                    'record_count': total_count,
                    'message': f'Found approximately {total_count} parcels'
                }

        return {
            'status': 'error',
            'message': 'Could not get preview count'
        }

    except Exception as e:
        logger.error(f"Preview search error: {str(e)}")
        return {
            'status': 'error',
            'message': f'Preview search failed: {str(e)}'
        }