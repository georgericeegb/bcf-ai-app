import os
import logging
import json
import requests
import geopandas as gpd
from shapely import wkt
from datetime import datetime
from google.cloud import storage
import tempfile
import time

# Import the config class
from config import config

# Import the new PipelineFileManager
try:
    from pipeline_file_manager import PipelineFileManager

    ENHANCED_FEATURES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Enhanced file management features available")
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Enhanced file management not available - using legacy mode")

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def preview_search_count(**kwargs):
    """Get count of records without downloading them - for cost estimation"""
    try:
        logger.info(f"Starting preview search with parameters: {kwargs}")

        # Get configuration values
        client_key = config.get("RAUSA_CLIENT_KEY")
        api_version = config.get("RAUSA_API_VERSION")
        api_url = config.get("RAUSA_API_URL")

        logger.info(f"🔧 DEBUG - Preview API URL: {api_url}")
        logger.info(f"🔧 DEBUG - Client Key: {'SET' if client_key else 'NOT SET'}")
        logger.info(f"🔧 DEBUG - API Version: {api_version}")

        if not client_key:
            return {"status": "error", "message": "ReportAll API key not configured"}

        # Get search parameters
        county_id = kwargs.get("county_id")
        owner = kwargs.get("owner", "")
        parcel_id = kwargs.get("parcel_id", "")
        calc_acreage_min = kwargs.get("calc_acreage_min", "")
        calc_acreage_max = kwargs.get("calc_acreage_max", "")

        # Build minimal API parameters for count-only
        params = {
            'client': client_key,
            'v': api_version,
            'county_id': county_id,
            'rpp': 1,  # Minimal download - just get the count
            'page': 1
        }

        # Add search criteria
        if parcel_id:
            params['parcel_id'] = parcel_id
            search_type = "parcel_id"
        elif owner:
            params['owner'] = owner
            search_type = "owner"
        else:
            # For general search, use a broad criteria
            params['owner'] = 'SMITH'  # Common name for testing
            search_type = "general"

        # Add acreage filters
        if calc_acreage_min:
            params['calc_acreage_min'] = calc_acreage_min
        if calc_acreage_max:
            params['calc_acreage_max'] = calc_acreage_max

        logger.info(f"🔧 DEBUG - Preview params: {params}")

        # Try different authentication methods
        auth_methods = [
            # Method 1: Original
            {'client': client_key},
            # Method 2: Alternative parameter name
            {'client_key': client_key},
            # Method 3: Both parameters
            {'client': client_key, 'client_key': client_key}
        ]

        for i, auth_method in enumerate(auth_methods, 1):
            try:
                test_params = params.copy()
                test_params.update(auth_method)

                logger.info(f"🔧 Trying authentication method {i}: {list(auth_method.keys())}")

                response = requests.get(api_url, params=test_params, timeout=30)
                logger.info(f"🔧 Method {i} - Status: {response.status_code}")
                logger.info(f"🔧 Method {i} - URL: {response.url}")

                if response.status_code == 200:
                    data = response.json()

                    if data.get('status') == 'OK':
                        total_count = data.get('count', 0)

                        logger.info(f"✅ Authentication method {i} successful!")
                        logger.info(f"📊 Found {total_count} records")

                        return {
                            "status": "success",
                            "record_count": total_count,
                            "search_type": search_type,
                            "message": f"Found {total_count} parcels matching your criteria",
                            "search_criteria": {
                                "county_id": county_id,
                                "owner": owner,
                                "parcel_id": parcel_id,
                                "calc_acreage_min": calc_acreage_min,
                                "calc_acreage_max": calc_acreage_max
                            },
                            "auth_method_used": i
                        }
                    else:
                        logger.warning(f"Method {i} - API returned: {data.get('status')} - {data.get('message')}")
                else:
                    logger.warning(f"Method {i} - HTTP {response.status_code}: {response.text[:200]}")

            except Exception as method_error:
                logger.warning(f"Method {i} failed: {method_error}")
                continue

        # If all methods failed
        return {
            "status": "error",
            "message": "All authentication methods failed. Please check API credentials.",
            "debug_info": {
                "api_url": api_url,
                "client_key_set": bool(client_key),
                "api_version": api_version
            }
        }

    except Exception as e:
        logger.error(f"Error in preview search: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Preview search failed: {str(e)}"}

def run_headless(**kwargs):
    """Enhanced headless version of the parcel search function with metadata tracking."""
    temp_path = None

    try:
        start_time = time.time()
        logger.info(f"Starting enhanced parcel search with parameters: {kwargs}")

        # Get configuration values
        client_key = config.get("RAUSA_CLIENT_KEY")
        api_version = config.get("RAUSA_API_VERSION")
        api_url = config.get("RAUSA_API_URL")

        logger.info(f"Using parcel database API")  # GENERIC LOG
        logger.info(f"Using API version: {api_version}")
        logger.info(f"Client credentials configured: {bool(client_key)}")

        # Get search parameters from kwargs
        county_id = kwargs.get("county_id")
        owner = kwargs.get("owner", "")
        parcel_id = kwargs.get("parcel_id", "")
        calc_acreage_min = kwargs.get("calc_acreage_min", "")
        calc_acreage_max = kwargs.get("calc_acreage_max", "")

        # Enhanced parameters for pipeline tracking
        pipeline_id = kwargs.get("pipeline_id")
        user_id = kwargs.get("user_id", "anonymous")

        if not county_id:
            logger.error("County ID is required")
            return {
                "status": "error",
                "message": "County ID is required",
                "debug_info": f"Received parameters: {list(kwargs.keys())}"
            }
        # ✅ FIXED: Ensure county_id is a string
        county_id = str(county_id)
        logger.info(f"✅ Using county_id: {county_id}")

        logger.info(f"Starting parcel search for County ID: {county_id}")

        # Build API parameters using the working format from direct API test
        params = {
            'client': client_key,  # ✅ Change from 'client_key' to 'client'
            'v': api_version,  # ✅ Change from 'version' to 'v'
            'county_id': county_id,
            'rpp': 1000,  # ✅ Change back to higher number for full search
            'page': 1
        }

        # Add search criteria based on what's provided
        search_type = None

        if parcel_id:
            params['parcel_id'] = parcel_id
            search_type = "parcel_id"
        elif owner:
            params['owner'] = owner
            search_type = "owner"
        else:
            # For general county search, use a broad owner search
            params['owner'] = 'SMITH'  # Use common name to test
            search_type = "owner_default"

        # Add acreage filters if provided
        if calc_acreage_min:
            params['calc_acreage_min'] = calc_acreage_min
        if calc_acreage_max:
            params['calc_acreage_max'] = calc_acreage_max

        logger.info(f"Search type: {search_type}")
        logger.info(f"API parameters: {params}")

        # Query the API with pagination (keeping existing logic)
        all_results = []
        page_count = 0
        max_pages = 10  # Safety limit

        try:
            while page_count < max_pages:
                logger.info(f"Fetching page {params['page']} from parcel database")

                # Add SSL verification disable and better error handling
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                response = requests.get(api_url, params=params, timeout=60, verify=False)  # Added verify=False
                logger.info(f"API response status code: {response.status_code}")
                logger.info(f"API response URL: {response.url}")

                if response.status_code != 200:
                    logger.error(f"API Error - Status: {response.status_code}")
                    error_text = response.text[:500] if response.text else "No response content"
                    logger.error(f"API Error - Response: {error_text}")

                    # Check if it's an authentication error
                    if response.status_code == 401 or response.status_code == 403:
                        return {"status": "error",
                                "message": f"Authentication failed. Please check your API key. Status: {response.status_code}"}

                    return {"status": "error", "message": f"API Error {response.status_code}: {error_text}"}

                # Parse the response
                try:
                    data = response.json()
                    logger.info(
                        f"API response structure: status={data.get('status')}, count={data.get('count')}, page={data.get('page')}, rpp={data.get('rpp')}")

                    # Check for API error
                    if data.get('status') != 'OK':
                        return {"status": "error",
                                "message": f"API returned status: {data.get('status')} - {data.get('message', 'Unknown error')}"}

                    # Get results
                    results = data.get('results', [])
                    total_count = data.get('count', 0)
                    current_page = data.get('page', 1)
                    rpp = data.get('rpp', 10)

                    logger.info(f"Found {len(results)} results on page {current_page}, total available: {total_count}")

                    if len(results) > 0:
                        all_results.extend(results)
                        page_count += 1

                        # Check if we need more pages
                        if len(all_results) < total_count and len(results) == rpp:
                            params['page'] += 1
                            logger.info(f"Fetching next page: {params['page']}")
                            continue
                        else:
                            logger.info("All results collected or reached end of pages")
                            break
                    else:
                        logger.info("No results found")
                        break

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.error(f"Raw response: {response.text[:500]}...")
                    return {"status": "error", "message": f"Invalid JSON response: {str(e)}"}

            # Process the results
            if not all_results:
                return {
                    "status": "warning",
                    "message": f"No parcels found for the specified search criteria",
                    "search_parameters": params,
                    "search_type": search_type
                }

            logger.info(f"Processing {len(all_results)} total parcels")

            # Check for geometry data
            valid_results = []
            for res in all_results:
                if 'geom_as_wkt' in res and res['geom_as_wkt']:
                    valid_results.append(res)
                else:
                    logger.debug(f"Skipping parcel {res.get('parcel_id', 'unknown')} - missing geometry")

            logger.info(f"Found {len(valid_results)} parcels with valid geometry")

            if not valid_results:
                return {
                    "status": "warning",
                    "message": f"No parcels with valid geometry found",
                    "total_parcels": len(all_results),
                    "search_type": search_type
                }

            # Create GeoDataFrame from valid results
            logger.info("Creating GeoDataFrame from valid results")
            geometries = []
            for res in valid_results:
                try:
                    geom = wkt.loads(res['geom_as_wkt'])
                    geometries.append(geom)
                except Exception as e:
                    logger.warning(f"Failed to parse geometry for parcel {res.get('parcel_id', 'unknown')}: {str(e)}")
                    geometries.append(None)

            # Create the GeoDataFrame
            gdf = gpd.GeoDataFrame(valid_results, geometry=geometries)
            gdf.crs = "EPSG:4326"

            # Extract the county name and state abbreviation
            county_name = "Unknown"
            state_abbr = "XX"

            if len(gdf) > 0:
                if 'county_name' in gdf.columns:
                    county_name = gdf['county_name'].iloc[0]
                if 'state_abbr' in gdf.columns:
                    state_abbr = gdf['state_abbr'].iloc[0]

            logger.info(f"County name: {county_name}, State: {state_abbr}")

            # Save to cloud storage with enhanced metadata
            if config.is_cloud_environment():
                bucket_name = config.get("BUCKET_NAME")
                if not bucket_name:
                    logger.error("No bucket name configured for cloud storage")
                    return {"status": "error", "message": "No bucket name configured"}

                logger.info(f"Saving to Google Cloud Storage bucket: {bucket_name}")

                # Create temp file
                import os, uuid
                temp_dir = "/tmp"
                current_timestamp = int(time.time())
                unique_id = str(uuid.uuid4())
                temp_path = os.path.join(temp_dir, f"parcel_{current_timestamp}_{unique_id}.gpkg")

                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

                    # Save to temporary file
                    logger.info(f"Saving to temporary file: {temp_path}")
                    gdf.to_file(temp_path, driver='GPKG')

                    # Enhanced file saving with metadata
                    if ENHANCED_FEATURES_AVAILABLE and pipeline_id:
                        logger.info("Using enhanced file management with metadata")
                        result = save_with_enhanced_metadata(
                            temp_path=temp_path,
                            state_abbr=state_abbr,
                            county_name=county_name,
                            pipeline_id=pipeline_id,
                            search_criteria={
                                'county_id': county_id,
                                'owner': owner,
                                'parcel_id': parcel_id,
                                'calc_acreage_min': calc_acreage_min,
                                'calc_acreage_max': calc_acreage_max
                            },
                            user_id=user_id,
                            record_count=len(gdf),
                            search_type=search_type
                        )

                        if result['success']:
                            # Return enhanced result
                            end_time = time.time()
                            elapsed_time = end_time - start_time

                            return {
                                "status": "success",
                                "message": f"Found {len(gdf)} parcels for {county_name}, {state_abbr}",
                                "record_count": len(gdf),
                                "file_path": result['gcs_path'],
                                "file_name": result['filename'],
                                "storage_path": result.get('directory', f"parcel_files/{state_abbr}/{county_name}"),
                                "county_name": county_name,
                                "state_abbr": state_abbr,
                                "search_type": search_type,
                                "processing_time": round(elapsed_time, 2),
                                "search_id": result.get('base_search_id'),
                                "pipeline_id": pipeline_id,
                                "enhanced_metadata": True,
                                "parameters": {
                                    "county_id": county_id,
                                    "owner": owner,
                                    "parcel_id": parcel_id,
                                    "calc_acreage_min": calc_acreage_min,
                                    "calc_acreage_max": calc_acreage_max
                                }
                            }
                        else:
                            logger.warning(f"Enhanced file save failed: {result.get('error')}, falling back to legacy")

                    # Legacy file saving (existing logic as fallback)
                    logger.info("Using legacy file management")

                    # Generate timestamp for filename
                    timestamp = datetime.now().strftime("_%m%d%Y%H%M")
                    file_name = f"{county_name}Co{state_abbr}{timestamp}.gpkg"

                    # Upload to Google Cloud Storage (existing logic)
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)

                    # Create a storage path
                    storage_path = f"{state_abbr}/{county_name}/Parcel_Files"
                    blob_path = f"{storage_path}/{file_name}"
                    blob = bucket.blob(blob_path)

                    logger.info(f"Uploading to Google Cloud Storage: {blob_path}")
                    blob.upload_from_filename(temp_path)

                    # Generate public URL
                    public_url = f"gs://{bucket_name}/{blob_path}"

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    logger.info(f"Parcel search completed successfully in {elapsed_time:.2f} seconds")

                    return {
                        "status": "success",
                        "message": f"Found {len(gdf)} parcels for {county_name}, {state_abbr}",
                        "record_count": len(gdf),
                        "file_path": public_url,
                        "file_name": file_name,
                        "storage_path": storage_path,
                        "county_name": county_name,
                        "state_abbr": state_abbr,
                        "search_type": search_type,
                        "processing_time": round(elapsed_time, 2),
                        "enhanced_metadata": False,
                        "parameters": {
                            "county_id": county_id,
                            "owner": owner,
                            "parcel_id": parcel_id,
                            "calc_acreage_min": calc_acreage_min,
                            "calc_acreage_max": calc_acreage_max
                        }
                    }
                except Exception as e:
                    logger.error(f"Error in cloud storage process: {str(e)}", exc_info=True)
                    return {"status": "error", "message": f"Error saving to cloud storage: {str(e)}"}
                finally:
                    # Clean up the temporary file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                            logger.info(f"Removed temporary file: {temp_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying the API: {str(e)}")
            return {"status": "error", "message": f"Error querying the API: {str(e)}"}

    except Exception as e:
        logger.error(f"Unexpected error in parcel search: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}


def save_with_enhanced_metadata(temp_path, state_abbr, county_name, pipeline_id,
                                search_criteria, user_id, record_count, search_type):
    """Save file using the enhanced PipelineFileManager with rich metadata."""
    try:
        file_manager = PipelineFileManager()

        # Clean county name for file path
        county_clean = county_name.replace(' County', '').replace(' ', '_').lower()

        result = file_manager.save_file_with_metadata(
            local_file_path=temp_path,
            state=state_abbr,
            county=county_clean,
            analysis_type='parcel_files',
            pipeline_id=pipeline_id,
            search_criteria=search_criteria,
            user_id=user_id
        )

        if result['success']:
            logger.info(f"Enhanced file save successful: {result['gcs_path']}")

            # Add additional metadata specific to parcel search
            try:
                storage_client = storage.Client()
                bucket_name = result['gcs_path'].split('/')[2]  # Extract bucket from gs://bucket/path
                blob_path = '/'.join(result['gcs_path'].split('/')[3:])  # Extract path

                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                # Add parcel-specific metadata
                enhanced_metadata = blob.metadata or {}
                enhanced_metadata.update({
                    'record_count': str(record_count),
                    'search_type': search_type,
                    'api_source': 'ReportAll',
                    'data_version': 'v1.0'
                })

                blob.metadata = enhanced_metadata
                blob.patch()

                logger.info("Enhanced metadata updated successfully")

            except Exception as meta_error:
                logger.warning(f"Failed to add enhanced metadata: {meta_error}")

        return result

    except Exception as e:
        logger.error(f"Error in enhanced file save: {e}")
        return {'success': False, 'error': str(e)}


# Legacy function name for backwards compatibility
def run_parcel_search(**kwargs):
    """Legacy function name - calls the enhanced version."""
    return run_headless(**kwargs)