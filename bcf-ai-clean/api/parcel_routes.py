from flask import Blueprint, request, jsonify, redirect
from api.auth_routes import login_required
from config.database import db
from utils.file_helpers import download_from_gcs, format_file_size
import logging
import os
from datetime import timedelta

logger = logging.getLogger(__name__)
parcel_bp = Blueprint('parcel', __name__)

@parcel_bp.route('/preview-search', methods=['POST'])
@login_required
def preview_parcel_search():
    """Preview parcel search count"""
    try:
        # Import here to avoid circular imports
        import enhanced_parcel_search
        
        data = request.get_json()
        logger.info(f"Parcel search preview request: {data}")

        # Map frontend parameters to backend parameters
        search_params = {
            'county_id': data.get('county_id'),
            'calc_acreage_min': data.get('calc_acreage_min'),
            'calc_acreage_max': data.get('calc_acreage_max'),
            'owner': data.get('owner'),
            'parcel_id': data.get('parcel_id')
        }

        # Remove None values
        search_params = {k: v for k, v in search_params.items() if v is not None}

        result = enhanced_parcel_search.preview_search_count(**search_params)

        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'record_count': result.get('record_count', 0),
                'message': result.get('message', 'Preview completed')
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Preview failed')
            }), 400

    except Exception as e:
        logger.error(f"Preview parcel search error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Preview failed: {str(e)}'
        }), 500

@parcel_bp.route('/execute-search', methods=['POST'])
@login_required
def execute_parcel_search():
    """Execute parcel search"""
    try:
        # Import here to avoid circular imports
        import enhanced_parcel_search
        
        data = request.get_json()
        logger.info(f"Execute parcel search request: {data}")

        # Map all search parameters
        search_params = {
            'county_id': data.get('county_id'),
            'calc_acreage_min': data.get('calc_acreage_min'),
            'calc_acreage_max': data.get('calc_acreage_max'),
            'owner': data.get('owner'),
            'parcel_id': data.get('parcel_id'),
            'user_id': data.get('user_id', request.session.get('username', 'default_user')),
            'project_type': data.get('project_type', 'solar'),
            'county_name': data.get('county_name', 'Unknown'),
            'state': data.get('state', 'XX')
        }

        # Remove None values but keep empty strings
        search_params = {k: v for k, v in search_params.items() if v is not None}

        result = enhanced_parcel_search.run_headless(**search_params)

        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'record_count': result.get('record_count', 0),
                'message': result.get('message', 'Search completed successfully'),
                'search_id': result.get('search_id'),
                'county_name': result.get('county_name'),
                'state_abbr': result.get('state_abbr'),
                'processing_time': result.get('processing_time'),
                'file_path': result.get('file_path'),
                'csv_file_path': result.get('csv_file_path'),
                'parcel_data': result.get('parcel_data', [])[:50],  # Limit for preview
                'total_records': result.get('total_records', 0),
                'bucket_name': result.get('bucket_name'),
                'gpkg_blob_name': result.get('gpkg_blob_name'),
                'csv_blob_name': result.get('csv_blob_name'),
                'bucket_path': result.get('bucket_path')
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Search failed')
            }), 400

    except Exception as e:
        logger.error(f"Execute parcel search error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Search failed: {str(e)}'
        }), 500

@parcel_bp.route('/existing-files', methods=['POST'])
@login_required
def get_existing_files():
    """Get existing parcel files from cloud storage"""
    try:
        data = request.get_json()
        state = data.get('state')
        county = data.get('county')
        county_fips = data.get('county_fips')

        logger.info(f"Getting existing files for {county}, {state}")

        # Get storage client
        storage_client = db.get_storage_client()
        if not storage_client:
            logger.warning("Cloud storage not available")
            return jsonify({
                'success': True,
                'files': [],
                'folder_path': f"Cloud storage not available",
                'message': 'Cloud storage not configured - no existing files found'
            })

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        logger.info(f"Searching bucket: {bucket_name}")

        bucket = storage_client.bucket(bucket_name)

        # Try multiple path variations
        path_variations = [
            f"{state}/{county}/Parcel_Files/",
            f"{state.upper()}/{county}/Parcel_Files/",
            f"{state}/{county.upper()}/Parcel_Files/",
            f"{state.upper()}/{county.upper()}/Parcel_Files/"
        ]

        files = []
        for path_prefix in path_variations:
            logger.info(f"Checking path: {path_prefix}")
            blobs = list(bucket.list_blobs(prefix=path_prefix))

            for blob in blobs:
                if blob.name.endswith('/'):  # Skip folder entries
                    continue

                # Include both CSV and GPKG files
                if not blob.name.lower().endswith(('.csv', '.gpkg')):
                    continue

                file_info = {
                    'name': blob.name.split('/')[-1],
                    'path': f"gs://{bucket_name}/{blob.name}",
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'type': 'CSV' if blob.name.lower().endswith('.csv') else 'GPKG',
                    'parcel_count': _estimate_parcel_count_from_filename(blob.name),
                    'search_criteria': _extract_search_criteria_from_filename(blob.name)
                }
                files.append(file_info)

            if files:  # Found files with this path variation
                logger.info(f"Found {len(files)} files with path: {path_prefix}")
                break

        return jsonify({
            'success': True,
            'files': files,
            'folder_path': f"gs://{bucket_name}/{state}/{county}/",
            'message': f'Found {len(files)} files' if files else 'No files found'
        })

    except Exception as e:
        logger.error(f"Error getting existing files: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get existing files: {str(e)}'
        })

@parcel_bp.route('/download')
def download_file():
    """Download a file from GCS"""
    try:
        file_path = request.args.get('path')
        if not file_path or not file_path.startswith('gs://'):
            return jsonify({'error': 'Invalid file path'}), 400

        # Parse GCS path
        path_parts = file_path.replace('gs://', '').split('/', 1)
        if len(path_parts) < 2:
            return jsonify({'error': 'Invalid GCS path format'}), 400

        bucket_name = path_parts[0]
        blob_name = path_parts[1]

        # Get storage client
        storage_client = db.get_storage_client()
        if not storage_client:
            return jsonify({'error': 'Cloud storage not available'}), 500

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return jsonify({'error': 'File not found'}), 404

        # Generate signed URL for download
        download_url = blob.generate_signed_url(
            expiration=timedelta(hours=1),
            method='GET'
        )

        return redirect(download_url)

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@parcel_bp.route('/preview-file', methods=['POST'])
@login_required
def preview_file():
    """Preview first 50 rows of a CSV file from GCS"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path or not file_path.startswith('gs://'):
            return jsonify({'error': 'Invalid file path'}), 400

        # Parse GCS path
        path_parts = file_path.replace('gs://', '').split('/', 1)
        if len(path_parts) < 2:
            return jsonify({'error': 'Invalid GCS path format'}), 400

        bucket_name = path_parts[0]
        blob_name = path_parts[1]

        # Get storage client
        storage_client = db.get_storage_client()
        if not storage_client:
            return jsonify({'error': 'Cloud storage not available'}), 500

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return jsonify({'error': 'File not found'}), 404

        # Download file content
        file_content = blob.download_as_text()

        # Parse CSV and get first 50 rows
        import csv
        import io

        csv_reader = csv.DictReader(io.StringIO(file_content))

        # Get headers
        headers = csv_reader.fieldnames

        # Get first 50 rows
        preview_rows = []
        for i, row in enumerate(csv_reader):
            if i >= 50:  # Limit to 50 rows
                break
            preview_rows.append(row)

        # Get total row count estimate
        total_rows = len(file_content.split('\n')) - 1  # Subtract header row

        return jsonify({
            'success': True,
            'headers': headers,
            'preview_rows': preview_rows,
            'total_rows': total_rows,
            'showing_rows': len(preview_rows),
            'file_path': file_path
        })

    except Exception as e:
        logger.error(f"Preview error: {e}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@parcel_bp.route('/county-activity/<state>')
@login_required
def check_county_activity(state):
    """Check which counties have existing folders in Cloud Storage"""
    try:
        logger.info(f"Checking county activity for {state}")

        storage_client = db.get_storage_client()
        if not storage_client:
            return jsonify({
                'success': True,
                'county_activity': {},
                'total_counties': 0,
                'active_counties': 0,
                'message': 'Cloud storage not available'
            })

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = storage_client.bucket(bucket_name)

        # Look for state-specific folder structure like "STATE/CountyName/"
        state_prefix = f"{state}/"
        logger.info(f"Looking for folders with prefix: {state_prefix}")

        # List all folders under the state prefix
        blobs = bucket.list_blobs(prefix=state_prefix, delimiter='/')
        state_folders = []

        # Get the "directories" under the state folder
        for page in blobs.pages:
            if hasattr(page, 'prefixes'):
                state_folders.extend(page.prefixes)

        logger.info(f"Found {len(state_folders)} folders under {state_prefix}")

        # Get county list for this state
        from utils.county_helpers import load_counties_from_file
        state_counties = load_counties_from_file(state)
        county_activity = {}

        for county in state_counties:
            county_name = county['name']
            has_activity = False
            matching_folders = []

            # Check for exact matches with the state/county pattern
            expected_patterns = [
                f"{state}/{county_name}/",
                f"{state}/{county_name.replace(' ', '')}/",
                f"{state}/{county_name.upper()}/",
                f"{state}/{county_name.lower()}/"
            ]

            for folder in state_folders:
                folder_clean = folder.rstrip('/')

                for pattern in expected_patterns:
                    pattern_clean = pattern.rstrip('/')
                    if folder_clean.lower() == pattern_clean.lower():
                        matching_folders.append(folder)
                        has_activity = True
                        logger.info(f"Found activity: {folder} matches {county_name}")
                        break

                if has_activity:
                    break

            county_activity[county.get('fips', county_name)] = {
                'county_name': county_name,
                'has_activity': has_activity,
                'matching_folders': matching_folders
            }

        active_counties = sum(1 for c in county_activity.values() if c['has_activity'])
        logger.info(f"Activity summary: {active_counties}/{len(state_counties)} counties have past work")

        return jsonify({
            'success': True,
            'county_activity': county_activity,
            'total_counties': len(state_counties),
            'active_counties': active_counties
        })

    except Exception as e:
        logger.error(f"Error checking county activity for {state}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def _estimate_parcel_count_from_filename(filename):
    """Try to extract parcel count from filename patterns"""
    import re

    # Look for patterns like "parcels_1234" or similar
    match = re.search(r'(\d+)(?:_parcels?|parcels?)', filename.lower())
    if match:
        return int(match.group(1))

    # Default estimate based on file type
    if filename.lower().endswith('.csv'):
        return "~100-500"
    elif filename.lower().endswith('.gpkg'):
        return "~50-200"
    else:
        return "Unknown"

def _extract_search_criteria_from_filename(filename):
    """Extract search criteria hints from filename"""
    criteria_hints = []

    filename_lower = filename.lower()

    if 'acres' in filename_lower:
        criteria_hints.append('Acreage-based search')
    if 'owner' in filename_lower:
        criteria_hints.append('Owner search')
    if any(word in filename_lower for word in ['solar', 'wind', 'battery']):
        criteria_hints.append('Project-specific search')

    return '; '.join(criteria_hints) if criteria_hints else 'Standard parcel search'