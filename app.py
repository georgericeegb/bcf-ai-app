 # app.py - Main Flask Application
import os
from dotenv import load_dotenv

# Force clear any existing environment variable
if 'ANTHROPIC_API_KEY' in os.environ:
    print(f"🔄 Removing old system key: {os.environ['ANTHROPIC_API_KEY'][:20]}...")
    del os.environ['ANTHROPIC_API_KEY']

# Force load from .env file
print("📁 Loading .env file with override...")
load_dotenv('.env', override=True)

# Verify the new key
new_key = os.getenv('ANTHROPIC_API_KEY')
if new_key:
    print(f"✅ New key loaded: {new_key[:20]}...")
else:
    print("❌ Failed to load key from .env")

from services.ai_service import AIAnalysisService
from models.project_config import ProjectConfig
from datetime import datetime
from config import config, get_county_id
import uuid
from flask import send_file, abort
import tempfile
import math
import json
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
import pandas as pd
import requests
from google.cloud import storage
import io
import geopandas as gpd

# Load environment variables
load_dotenv()

app = Flask(__name__)

from enhanced_parcel_search import run_headless
from bigquery_slope_analysis import run_headless_fixed as run_slope_analysis
from transmission_analysis_bigquery import run_headless as run_transmission_analysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Check what API key is loaded
api_key = os.getenv('ANTHROPIC_API_KEY')
print(f"🔑 API Key loaded in Flask: {api_key[:15] if api_key else 'None'}...")
print(f"🔑 API Key length: {len(api_key) if api_key else 0}")



# Initialize AI service
ai_service = AIAnalysisService(
    api_key=os.getenv('ANTHROPIC_API_KEY')  # ✅ SECURE
)

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')


@app.route('/api/tier-info', methods=['GET'])
def get_tier_info():
    """Get information about analysis tiers and their criteria"""
    try:
        tier_info = {
            "state": {
                "description": ProjectConfig.get_tier_description("state"),
                "criteria": ProjectConfig.get_tier_criteria("state"),
                "purpose": "Market entry screening and state-level opportunity assessment",
                "typical_outcomes": ["Market entry decision", "State ranking", "Resource screening"]
            },
            "county": {
                "description": ProjectConfig.get_tier_description("county"),
                "criteria": ProjectConfig.get_tier_criteria("county"),
                "purpose": "Site identification and county-level feasibility analysis",
                "typical_outcomes": ["County shortlist", "Site selection", "Infrastructure assessment"]
            },
            "local": {
                "description": ProjectConfig.get_tier_description("local"),
                "criteria": ProjectConfig.get_tier_criteria("local"),
                "purpose": "Project development and detailed site analysis",
                "typical_outcomes": ["Go/no-go decision", "Development plan", "Investment decision"]
            }
        }

        return jsonify({
            'success': True,
            'tier_info': tier_info,
            'workflow': [
                "1. State Level: Identify promising markets",
                "2. County Level: Select specific development regions",
                "3. Local Level: Confirm site viability and proceed to development"
            ]
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommend-variables', methods=['POST'])
def recommend_variables():
    """Get AI recommendations for critical project variables"""
    try:
        data = request.json
        project_type = data.get('project_type')
        analysis_level = data.get('analysis_level', 'state')
        location = data.get('location', 'United States')
        selected_criteria = data.get('selected_criteria', [])  # NEW: Add this line

        # Get AI recommendations with selected criteria
        recommendations = ai_service.recommend_project_variables(
            project_type=project_type,
            analysis_level=analysis_level,
            location=location,
            selected_criteria=selected_criteria  # NEW: Pass criteria to AI service
        )

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze-variable', methods=['POST'])
def analyze_variable():
    """Deep dive analysis on a specific variable"""
    try:
        data = request.json
        project_config = ProjectConfig.from_dict(data.get('project_config', {}))
        variable_name = data.get('variable_name')
        analysis_criteria = data.get('analysis_criteria', [])

        # Perform deep analysis
        analysis = ai_service.analyze_variable_deeply(
            project_config=project_config,
            variable_name=variable_name,
            criteria=analysis_criteria
        )

        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/comprehensive-analysis', methods=['POST'])
def comprehensive_analysis():
    """Run comprehensive project analysis"""
    try:
        data = request.json
        project_config = ProjectConfig.from_dict(data.get('project_config', {}))

        # Perform comprehensive analysis
        analysis = ai_service.comprehensive_project_analysis(project_config)

        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({'status': 'healthy'})

@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache usage statistics"""
    try:
        stats = ai_service.get_cache_stats()
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache for specific location"""
    try:
        data = request.json
        location = data.get('location')
        analysis_level = data.get('analysis_level')

        if not location:
            return jsonify({
                'success': False,
                'error': 'Location is required'
            }), 400

        deleted_count = ai_service.clear_location_cache(location, analysis_level)

        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Cleared {deleted_count} cached responses for {location}',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/cache-dashboard')
def cache_dashboard():
    """Simple cache management dashboard"""
    return render_template('cache_dashboard.html')


@app.route('/api/analyze-focus-areas', methods=['POST'])
def analyze_focus_areas():
    """Analyze selected focus areas for the chosen tier and location"""
    try:
        data = request.json
        project_type = data.get('project_type')
        analysis_level = data.get('analysis_level')
        location = data.get('location')
        selected_criteria = data.get('selected_criteria', [])

        # Get focus area analysis
        analysis = ai_service.analyze_focus_areas(
            project_type=project_type,
            analysis_level=analysis_level,
            location=location,
            selected_criteria=selected_criteria
        )

        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-next-step', methods=['POST'])
def analyze_next_step():
    """Get detailed implementation guidance for a specific next step"""
    try:
        data = request.json
        next_step = data.get('next_step')
        project_type = data.get('project_type')
        analysis_level = data.get('analysis_level')
        location = data.get('location')
        category = data.get('category', 'General')
        step_id = data.get('step_id', '')

        if not next_step:
            return jsonify({
                'success': False,
                'error': 'Next step is required'
            }), 400

        # Get detailed implementation guidance
        guidance = ai_service.analyze_next_step_implementation(
            next_step=next_step,
            project_type=project_type,
            analysis_level=analysis_level,
            location=location,
            category=category
        )

        return jsonify({
            'success': True,
            'guidance': guidance,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Helper functions
def get_min_acreage_for_project(project_type):
    """Get minimum acreage recommendations for renewable energy projects"""
    acreage_requirements = {
        'solar': 10,  # 10+ acres for utility-scale solar (5-8 acres per MW)
        'wind': 50  # 50+ acres for wind (more spacing needed between turbines)
    }
    return acreage_requirements.get(project_type.lower(), 10)


def get_state_name_from_fips(state_fips):
    """Convert state FIPS code to state name"""
    fips_to_state = {
        '06': 'California',
        '48': 'Texas',
        '04': 'Arizona',
        '32': 'Nevada',
        '35': 'New Mexico',
        '08': 'Colorado',
        '56': 'Wyoming',
        '49': 'Utah',
        '20': 'Kansas',
        '19': 'Iowa',
        '17': 'Illinois',
        '39': 'Ohio',
        '42': 'Pennsylvania',
        '36': 'New York',
        '12': 'Florida'
    }
    return fips_to_state.get(state_fips, 'Unknown')

def generate_parcel_next_steps(project_type, parcel_count, search_type):
    """Generate next steps recommendations based on parcel search results and search type"""
    if parcel_count == 0:
        if search_type == 'parcel_id':
            return [
                "Verify the parcel ID is correct and formatted properly",
                "Check if the parcel exists in this county",
                "Try searching by owner name instead"
            ]
        elif search_type == 'owner':
            return [
                "Try variations of the owner name (with/without suffixes)",
                "Search for partial names or common abbreviations",
                "Consider expanding search to adjacent counties"
            ]
        else:
            return [
                "Consider expanding search to adjacent counties",
                "Review minimum acreage requirements - may be too restrictive",
                "Try searching by owner names in the area"
            ]
    elif parcel_count < 5:
        return [
            "Prioritize parcels with best grid proximity and access",
            "Conduct preliminary site visits for all candidates",
            "Begin outreach to property owners for development interest"
        ]
    elif parcel_count < 25:
        return [
            "Filter parcels by proximity to transmission infrastructure",
            "Use GIS analysis to screen for environmental constraints",
            "Develop scoring matrix for parcel prioritization"
        ]
    else:
        return [
            "Apply additional filtering criteria (zoning, slopes, etc.)",
            "Use automated screening tools for large parcel sets",
            "Consider batch analysis for regional development strategy"
        ]

@app.route('/api/parcel-search/config', methods=['GET'])
def get_parcel_search_config():
    """Get parcel search configuration - updated to use FIPS mapping"""
    try:
        # Check if ReportAll API is configured
        api_configured = bool(config.get('RAUSA_CLIENT_KEY'))

        # Get list of supported states and counties from our FIPS mapping
        supported_counties = {}
        total_counties = 0

        for state_abbr, counties in STATE_COUNTIES_MAP.items():
            state_name = get_state_name_from_abbr(state_abbr)
            supported_counties[state_name] = {
                'abbr': state_abbr,
                'counties': [county['name'] for county in counties],
                'count': len(counties)
            }
            total_counties += len(counties)

        return jsonify({
            'success': True,
            'api_configured': api_configured,
            'supported_counties': supported_counties,
            'total_counties': total_counties,
            'total_states': len(supported_counties),
            'default_acreage': {
                'solar': {'min': 10, 'typical_max': 1000},
                'wind': {'min': 50, 'typical_max': 5000}
            },
            'search_info': {
                'data_source': 'ReportAll USA',
                'update_frequency': 'Monthly',
                'coverage': f'US nationwide county-level parcel data ({total_counties} counties)'
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Configuration error: {str(e)}'
        }), 500


@app.route('/api/parcel-search/preview', methods=['POST'])
def preview_parcel_search():
    """Preview parcel search to get count estimate before full search"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No request data provided'
            }), 400

        location = data.get('location', '')
        project_type = data.get('project_type', 'solar')
        analysis_level = data.get('analysis_level', 'county')

        # Get search parameters
        min_acreage = data.get('min_acreage') or data.get('calc_acreage_min')
        max_acreage = data.get('max_acreage') or data.get('calc_acreage_max')
        owner = data.get('owner', '').strip()
        parcel_id = data.get('parcel_id', '').strip()

        # Parse location to get county name - improved error handling
        county_name = None

        if analysis_level == 'state':
            # For state level, county should be provided separately
            county_name = data.get('selected_county', '').strip()
        elif analysis_level == 'county' and location and ',' in location:
            parts = location.split(',')
            if len(parts) >= 1:
                county_name = parts[0].strip()
        elif analysis_level == 'local' and location and ',' in location:
            parts = location.split(',')
            if len(parts) >= 2:
                county_name = parts[1].strip()

        if not county_name:
            return jsonify({
                'success': False,
                'error': f'County name could not be determined. Analysis level: {analysis_level}, Location: {location}, Selected county: {data.get("selected_county", "Not provided")}'
            }), 400

        # Get county ID for API - using the legacy function name that should work
        county_id = get_county_id(county_name)
        if not county_id:
            return jsonify({
                'success': False,
                'error': f'County ID not found for {county_name}. Please check the county name spelling.'
            }), 400

        # Build search parameters - must have at least one search criteria
        search_params = {
            'county_id': county_id,
        }

        # Validate that at least one search criteria is provided
        search_criteria_provided = False

        if min_acreage:
            try:
                search_params['calc_acreage_min'] = str(min_acreage)
                search_criteria_provided = True
            except:
                pass

        if max_acreage:
            try:
                search_params['calc_acreage_max'] = str(max_acreage)
            except:
                pass

        if owner:
            search_params['owner'] = owner
            search_criteria_provided = True

        if parcel_id:
            search_params['parcel_id'] = parcel_id
            search_criteria_provided = True

        if not search_criteria_provided:
            return jsonify({
                'success': False,
                'error': 'At least one search criteria is required: minimum acreage, owner name, or parcel ID'
            }), 400

        # Import and use the parcel search module
        from enhanced_parcel_search import preview_search_count

        result = preview_search_count(**search_params)

        if result.get('status') == 'success':
            return jsonify({
                'success': True,
                'county_name': county_name,
                'county_id': county_id,
                'estimated_parcels': result.get('record_count', 0),
                'search_criteria': search_params,
                'message': f"Found approximately {result.get('record_count', 0)} parcels matching your criteria in {county_name}"
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('message', 'Preview search failed'),
                'debug_info': result.get('debug_info', {}),
                'search_params': search_params
            }), 500

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'Preview search error: {str(e)}',
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()[:500]  # Limit traceback length
        }), 500


@app.route('/api/parcel-search/execute', methods=['POST'])
def execute_parcel_search():
    """Execute full parcel search and return results"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No request data provided'
            }), 400

        location = data.get('location', '')
        project_type = data.get('project_type', 'solar')
        analysis_level = data.get('analysis_level', 'county')

        # Get search parameters
        min_acreage = data.get('min_acreage') or data.get('calc_acreage_min')
        max_acreage = data.get('max_acreage') or data.get('calc_acreage_max')
        owner = data.get('owner', '').strip()
        parcel_id = data.get('parcel_id', '').strip()

        # Generate unique pipeline ID for this search
        pipeline_id = str(uuid.uuid4())

        # Parse location to get county name - improved error handling
        county_name = None

        if analysis_level == 'state':
            # For state level, county should be provided separately
            county_name = data.get('selected_county', '').strip()
        elif analysis_level == 'county' and location and ',' in location:
            parts = location.split(',')
            if len(parts) >= 1:
                county_name = parts[0].strip()
        elif analysis_level == 'local' and location and ',' in location:
            parts = location.split(',')
            if len(parts) >= 2:
                county_name = parts[1].strip()

        if not county_name:
            return jsonify({
                'success': False,
                'error': f'County name could not be determined. Analysis level: {analysis_level}, Location: {location}, Selected county: {data.get("selected_county", "Not provided")}'
            }), 400

        # Get county ID for API - using the legacy function name
        county_id = get_county_id(county_name)
        if not county_id:
            return jsonify({
                'success': False,
                'error': f'County ID not found for {county_name}. Please check the county name spelling.'
            }), 400

        # Set search parameters
        search_params = {
            'county_id': county_id,
            'pipeline_id': pipeline_id,
            'user_id': 'renewable_energy_analyzer'
        }

        # Validate that at least one search criteria is provided
        search_criteria_provided = False
        search_type = 'general'

        if min_acreage:
            try:
                search_params['calc_acreage_min'] = str(min_acreage)
                search_criteria_provided = True
                search_type = 'acreage'
            except:
                pass

        if max_acreage:
            try:
                search_params['calc_acreage_max'] = str(max_acreage)
            except:
                pass

        if owner:
            search_params['owner'] = owner
            search_criteria_provided = True
            search_type = 'owner'

        if parcel_id:
            search_params['parcel_id'] = parcel_id
            search_criteria_provided = True
            search_type = 'parcel_id'

        if not search_criteria_provided:
            return jsonify({
                'success': False,
                'error': 'At least one search criteria is required: minimum acreage, owner name, or parcel ID'
            }), 400

        # Import and execute the parcel search
        logger.info(f"Executing parcel search with params: {search_params}")
        result = run_headless(**search_params)

        if result.get('status') == 'success':
            return jsonify({
                'success': True,
                'message': result.get('message', f'Found {result.get("record_count", 0)} parcels'),
                'parcel_count': result.get('record_count', 0),
                'file_path': result.get('file_path', ''),
                'file_name': result.get('file_name', ''),
                # Add these CSV-specific fields:
                'csv_file_path': result.get('csv_file_path', ''),
                'csv_file_name': result.get('csv_file_name', ''),
                'parcel_data': result.get('parcel_data', []),
                'county_name': result.get('county_name', county_name),
                'state_abbr': result.get('state_abbr', ''),
                'processing_time': result.get('processing_time', 0),
                'search_id': result.get('search_id'),
                'pipeline_id': pipeline_id,
                'search_type': search_type,
                'storage_type': result.get('storage_type', 'local'),
                'search_parameters': {
                    'county_name': county_name,
                    'county_id': county_id,
                    'min_acreage': min_acreage,
                    'max_acreage': max_acreage,
                    'owner': owner,
                    'parcel_id': parcel_id
                },
                'download_info': {
                    'format': 'CSV (.csv) and GeoPackage (.gpkg)',
                    'csv_format': 'Tabular data for analysis',
                    'gpkg_format': 'Spatial data for mapping',
                    'size_estimate': f"~{result.get('record_count', 0) * 1}KB"
                },
                'next_steps': generate_parcel_next_steps(project_type, result.get('record_count', 0), search_type)
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('message', 'Parcel search failed'),
                'details': result,
                'search_params': search_params
            }), 500

    except Exception as e:
        import traceback
        logger.error(f"Parcel search execution error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Parcel search execution error: {str(e)}',
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()[:500]  # Limit traceback length
        }), 500

@app.route('/api/counties/<state>', methods=['GET'])
def get_counties_for_state_api(state):
    """Get list of counties for a given state - updated to use FIPS mapping"""
    try:
        # Import the new functions
        from config import get_counties_for_state, get_state_name_from_abbr

        # Convert full state name to abbreviation if needed
        state_abbr_mapping = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
            'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
            'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'Idaho',
            'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
            'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
            'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
            'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
            'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
            'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
            'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
        }

        # Try to get state abbreviation
        state_abbr = state_abbr_mapping.get(state, state.upper())

        # Get counties for this state
        counties = get_counties_for_state(state_abbr)

        if counties and len(counties) > 0:
            county_names = [county['name'] for county in counties if county and 'name' in county]
            return jsonify({
                'success': True,
                'state': state,
                'state_abbr': state_abbr,
                'counties': sorted(county_names),
                'count': len(county_names)
            })
        else:
            return jsonify({
                'success': True,
                'state': state,
                'state_abbr': state_abbr,
                'counties': [],
                'count': 0,
                'message': f'No counties found for {state}'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error loading counties for {state}: {str(e)}',
            'error_type': type(e).__name__
        }), 500


@app.route('/parcel_downloads/<path:filename>')
def serve_parcel_file(filename):
    """Serve parcel files from cloud storage"""
    try:
        from google.cloud import storage
        from config import config

        bucket_name = config.get("BUCKET_NAME", "bcfparcelsearchrepository")

        # Parse the filename to get the cloud path
        # Expected format: STATE/COUNTY/filename.gpkg
        # Cloud path format: STATE/COUNTY/Parcel_Files/filename.gpkg

        parts = filename.split('/')
        if len(parts) < 3:
            abort(404)  # Invalid path format

        state = parts[0]
        county = parts[1]
        file_name = parts[2]

        # Construct cloud storage path
        cloud_path = f"{state}/{county}/Parcel_Files/{file_name}"

        logger.info(f"Attempting to serve file from cloud: gs://{bucket_name}/{cloud_path}")

        # Access cloud storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(cloud_path)

        if not blob.exists():
            logger.error(f"File not found in cloud storage: {cloud_path}")
            abort(404)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)

        try:
            # Download to temporary file
            blob.download_to_filename(temp_file.name)

            # Send file
            return send_file(
                temp_file.name,
                as_attachment=True,
                download_name=file_name,
                mimetype='application/octet-stream'
            )
        finally:
            # Note: We can't delete the temp file here because send_file needs it
            # The file will be cleaned up by the OS eventually
            pass

    except Exception as e:
        logger.error(f"Error serving cloud file {filename}: {str(e)}")
        abort(500)


@app.route('/api/parcel-preview/<path:filename>')
def preview_parcel_data(filename):
    """Get preview data from a parcel file in cloud storage"""
    try:
        from google.cloud import storage
        from config import config
        import geopandas as gpd

        bucket_name = config.get("BUCKET_NAME", "bcfparcelsearchrepository")

        # Parse the filename to get the cloud path
        parts = filename.split('/')
        if len(parts) < 3:
            return jsonify({'success': False, 'error': 'Invalid file path format'}), 400

        state = parts[0]
        county = parts[1]
        file_name = parts[2]

        # Construct cloud storage path
        cloud_path = f"{state}/{county}/Parcel_Files/{file_name}"

        logger.info(f"Previewing file from cloud: gs://{bucket_name}/{cloud_path}")

        # Access cloud storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(cloud_path)

        if not blob.exists():
            return jsonify({'success': False, 'error': 'File not found in cloud storage'}), 404

        # Download to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
        blob.download_to_filename(temp_file.name)

        try:
            # Read the GeoPackage file
            gdf = gpd.read_file(temp_file.name)

            # Convert to regular DataFrame for JSON serialization
            df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf

            # Get basic statistics
            total_parcels = len(df)

            # Calculate acreage statistics if available
            acreage_stats = {}
            acreage_col = None
            for col in ['calc_acreage', 'acreage', 'acres', 'total_acreage']:
                if col in df.columns:
                    acreage_col = col
                    break

            if acreage_col and df[acreage_col].dtype in ['float64', 'int64']:
                acreage_stats = {
                    'min': float(df[acreage_col].min()),
                    'max': float(df[acreage_col].max()),
                    'mean': float(df[acreage_col].mean()),
                    'total': float(df[acreage_col].sum())
                }

            # Get sample data (first 20 records)
            preview_data = df.head(20).fillna('').to_dict('records')

            # Get column information
            columns = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]

            # Get file size from blob
            file_size_mb = round(blob.size / (1024 * 1024), 2) if blob.size else 0

            return jsonify({
                'success': True,
                'total_parcels': total_parcels,
                'preview_data': preview_data,
                'columns': columns,
                'acreage_stats': acreage_stats,
                'acreage_column': acreage_col,
                'storage_type': 'cloud',
                'cloud_path': cloud_path,
                'file_info': {
                    'name': file_name,
                    'size_mb': file_size_mb,
                    'bucket': bucket_name
                }
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except Exception as e:
        logger.error(f"Error previewing cloud file {filename}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error reading file from cloud storage: {str(e)}'
        }), 500


@app.route('/api/cloud-storage/status')
def cloud_storage_status():
    """Check cloud storage configuration and access"""
    try:
        from google.cloud import storage
        from config import config

        bucket_name = config.get("BUCKET_NAME", "bcfparcelsearchrepository")

        # Test bucket access
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()  # This will fail if bucket doesn't exist or no access

        # List some files to verify access
        blobs = list(bucket.list_blobs(max_results=5))

        return jsonify({
            'success': True,
            'bucket_name': bucket_name,
            'bucket_accessible': True,
            'sample_files': [blob.name for blob in blobs]
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'bucket_name': config.get("BUCKET_NAME", "Not configured")
        }), 500

def normalize_parcel_data(parcel):
    """Convert your ReportAll parcel data to expected format"""
    return {
        'parcel_id': parcel.get('parcel_id', ''),
        'address': parcel.get('address', ''),
        'acreage_calc': float(parcel.get('acreage', 0) or parcel.get('acreage_calc', 0)),
        'latitude': float(parcel.get('latitude', 40.5)),
        'longitude': float(parcel.get('longitude', -78.4)),
        'elevation': float(parcel.get('elevation', 1200)),
        'land_use_class': parcel.get('land_use_class', 'Residential'),
        'land_cover': parcel.get('land_cover', '{}'),
        'crop_cover': parcel.get('crop_cover', '{}'),
        'fld_zone': parcel.get('fld_zone', None),
        'mkt_val_land': float(parcel.get('mkt_val_land', 0)),
        'mkt_val_tot': float(parcel.get('mkt_val_tot', 0)),
        'owner': parcel.get('owner', ''),
        'trans_date': parcel.get('trans_date', None),
        'acreage_adjacent_with_sameowner': float(parcel.get('acreage_adjacent_with_sameowner', 0))
    }


def calculate_suitability_scores(parcel):
    """
    Main suitability scoring function - implements the algorithm from the React component
    """

    # Pennsylvania solar irradiance data
    PA_SOLAR_IRRADIANCE = 4.2  # Blair County average

    # 1. Solar Resource Score (25% weight)
    solar_score = calculate_solar_resource_score(parcel, PA_SOLAR_IRRADIANCE)

    # 2. Physical Characteristics Score (20% weight)
    physical_score = calculate_physical_characteristics_score(parcel)

    # 3. Land Use Compatibility Score (20% weight)
    land_use_score = calculate_land_use_compatibility_score(parcel)

    # 4. Economic Viability Score (15% weight)
    economic_score = calculate_economic_viability_score(parcel)

    # 5. Grid Access Score (10% weight)
    grid_score = calculate_grid_access_score(parcel)

    # 6. Environmental Constraints Score (10% weight)
    environmental_score = calculate_environmental_constraints_score(parcel)

    # Calculate weighted total score
    total_score = (
            solar_score * 0.25 +
            physical_score * 0.20 +
            land_use_score * 0.20 +
            economic_score * 0.15 +
            grid_score * 0.10 +
            environmental_score * 0.10
    )

    # Determine rating
    if total_score >= 80:
        rating = 'Excellent'
    elif total_score >= 70:
        rating = 'Very Good'
    elif total_score >= 60:
        rating = 'Good'
    elif total_score >= 50:
        rating = 'Fair'
    else:
        rating = 'Poor'

    # Calculate recommended capacity (0.5 MW per acre, 80% usable land)
    acreage = parcel.get('acreage_calc', 200)
    recommended_capacity = math.floor(acreage * 0.8 * 0.5)

    return {
        'solar_resource': round(solar_score),
        'physical': round(physical_score),
        'land_use': round(land_use_score),
        'economic': round(economic_score),
        'grid_access': round(grid_score),
        'environmental': round(environmental_score),
        'total_score': round(total_score),
        'rating': rating,
        'recommended_capacity': recommended_capacity
    }


def parse_land_cover(land_cover_string):
    """Parse land cover data from string format"""
    if not land_cover_string or land_cover_string == 'null':
        return {}
    try:
        # Handle the format from your CSV: "{'Forest': 100.5, 'Crop': 25.3}"
        clean_string = land_cover_string.replace("'", '"')
        return json.loads(clean_string)
    except:
        return {}


def calculate_solar_resource_score(parcel, base_irradiance):
    """Calculate solar resource potential score (0-100)"""
    score = 0

    # Base solar irradiance (40% of score)
    score += (base_irradiance / 5.5) * 40

    # Elevation optimization (20% of score)
    elevation = parcel.get('elevation', 1200)
    if 1000 <= elevation <= 1500:
        score += 20
    elif 1500 < elevation <= 2000:
        score += 15
    elif elevation > 2000:
        score += 5
    else:
        score += 10

    # Latitude optimization (20% of score)
    latitude = parcel.get('latitude', 40.5)
    if 40 <= latitude <= 41:
        score += 20

    # Land cover for solar access (20% of score)
    land_cover = parse_land_cover(parcel.get('land_cover', '{}'))
    total_acres = parcel.get('acreage_calc', 200)

    open_area = (land_cover.get('Pasture/Hay', 0) +
                 land_cover.get('Cultivated Crops', 0) +
                 land_cover.get('Grassland/Herbaceous', 0) +
                 land_cover.get('Developed Open Space', 0))

    forest_area = (land_cover.get('Deciduous Forest', 0) +
                   land_cover.get('Mixed Forest', 0))

    cover_score = (open_area / total_acres) * 20 - (forest_area / total_acres) * 10
    score += max(0, cover_score)

    return min(100, max(0, score))


def calculate_physical_characteristics_score(parcel):
    """Calculate physical characteristics score (0-100)"""
    score = 0

    # Parcel size (40% of score)
    acreage = parcel.get('acreage_calc', 200)
    if acreage >= 200:
        score += 40
    elif acreage >= 100:
        score += 35
    elif acreage >= 50:
        score += 25
    else:
        score += 10

    # Adjacent land availability (30% of score)
    adjacent_acres = parcel.get('acreage_adjacent_with_sameowner', acreage)
    if adjacent_acres > acreage * 1.5:
        score += 30
    elif adjacent_acres > acreage:
        score += 20
    else:
        score += 10

    # Flood zone assessment (30% of score)
    flood_zone = parcel.get('fld_zone')
    if not flood_zone or flood_zone == 'null':
        score += 30
    elif 'X' in str(flood_zone):
        score += 25
    elif 'A' in str(flood_zone):
        score += 10
    else:
        score += 5

    return min(100, score)


def calculate_land_use_compatibility_score(parcel):
    """Calculate land use compatibility score (0-100)"""
    score = 0

    # Land use class (50% of score)
    land_use = parcel.get('land_use_class', '')
    if land_use in ['Commercial', 'Industrial']:
        score += 50
    elif land_use == 'Residential':
        score += 25
    elif land_use == 'Tax Exempt':
        score += 15
    else:
        score += 30

    # Development level (30% of score)
    land_cover = parse_land_cover(parcel.get('land_cover', '{}'))
    total_acres = parcel.get('acreage_calc', 200)

    developed_area = (land_cover.get('Developed Low Intensity', 0) +
                      land_cover.get('Developed Medium Intensity', 0) +
                      land_cover.get('Developed High Intensity', 0))

    if developed_area / total_acres < 0.1:
        score += 30
    elif developed_area / total_acres < 0.3:
        score += 20
    else:
        score += 10

    # Agricultural compatibility (20% of score)
    agric_area = (land_cover.get('Pasture/Hay', 0) +
                  land_cover.get('Cultivated Crops', 0))

    if agric_area / total_acres > 0.5:
        score += 20
    elif agric_area / total_acres > 0.2:
        score += 15
    else:
        score += 10

    return min(100, score)


def calculate_economic_viability_score(parcel):
    """Calculate economic viability score (0-100)"""
    score = 0

    # Land value assessment (40% of score)
    land_value = parcel.get('mkt_val_land', 0)
    acreage = parcel.get('acreage_calc', 200)
    value_per_acre = land_value / acreage if acreage > 0 else 0

    if value_per_acre < 2000:
        score += 40
    elif value_per_acre < 5000:
        score += 30
    elif value_per_acre < 10000:
        score += 20
    else:
        score += 10

    # Project size economics (35% of score)
    if acreage >= 250:
        score += 35
    elif acreage >= 200:
        score += 30
    elif acreage >= 150:
        score += 25
    else:
        score += 15

    # Ownership stability (25% of score)
    trans_date = parcel.get('trans_date')
    if trans_date:
        try:
            from datetime import datetime
            trans_year = datetime.strptime(trans_date, '%Y-%m-%d').year
            years_owned = datetime.now().year - trans_year
            if years_owned > 10:
                score += 25
            elif years_owned > 5:
                score += 20
            else:
                score += 15
        except:
            score += 15
    else:
        score += 15

    return min(100, score)


def calculate_grid_access_score(parcel):
    """Calculate grid access potential score (0-100)"""
    score = 0

    # Development proximity (60% of score)
    land_cover = parse_land_cover(parcel.get('land_cover', '{}'))
    total_acres = parcel.get('acreage_calc', 200)

    developed_area = (land_cover.get('Developed Low Intensity', 0) +
                      land_cover.get('Developed Medium Intensity', 0) +
                      land_cover.get('Developed High Intensity', 0) +
                      land_cover.get('Developed Open Space', 0))

    if developed_area / total_acres > 0.05:
        score += 60
    else:
        score += 30

    # Geographic accessibility (40% of score)
    elevation = parcel.get('elevation', 1200)
    if elevation < 1500:
        score += 40
    elif elevation < 2000:
        score += 30
    else:
        score += 20

    return min(100, score)


def calculate_environmental_constraints_score(parcel):
    """Calculate environmental constraints score (0-100)"""
    score = 100  # Start perfect, subtract penalties

    # Water/wetland proximity (30% penalty)
    land_cover = parse_land_cover(parcel.get('land_cover', '{}'))
    total_acres = parcel.get('acreage_calc', 200)

    water_area = land_cover.get('Open Water', 0) + land_cover.get('Woody Wetlands', 0)
    if water_area / total_acres > 0.1:
        score -= 30
    elif water_area / total_acres > 0.05:
        score -= 15

    # Forest coverage penalty (40% penalty)
    forest_area = (land_cover.get('Deciduous Forest', 0) +
                   land_cover.get('Mixed Forest', 0))

    if forest_area / total_acres > 0.7:
        score -= 40
    elif forest_area / total_acres > 0.5:
        score -= 25
    elif forest_area / total_acres > 0.3:
        score -= 15

    # Slope estimation from elevation (30% penalty)
    elevation = parcel.get('elevation', 1200)
    if elevation > 2200:
        score -= 30
    elif elevation > 1800:
        score -= 15

    return max(0, score)


def save_analysis_to_gcs(result_data, project_id):
    """Save analysis results to your GCS bucket"""
    try:
        # Use your existing GCS client/bucket logic
        # This would integrate with your existing cloud storage system
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"suitability_analysis_{project_id}_{timestamp}.json"

        # Save logic here - adapt to your existing GCS implementation
        # bucket_client.upload_string(json.dumps(result_data), filename)

        logger.info(f"Saved suitability analysis to GCS: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save to GCS: {str(e)}")
        return None


def load_parcels_from_gcs(gcs_path):
    """Load parcels from your GCS storage"""
    try:
        # Use your existing GCS loading logic
        # This would integrate with your existing cloud storage system
        pass
    except Exception as e:
        logger.error(f"Failed to load from GCS: {str(e)}")
        return []


@app.route('/preview/<path:file_path>')
def preview_file(file_path):
    """
    Preview CSV/GPKG file content in browser before download
    """
    try:
        # Parse the GCS path or signed URL
        if file_path.startswith('gs://'):
            # Direct GCS path
            bucket_name = file_path.split('/')[2]
            blob_name = '/'.join(file_path.split('/')[3:])

            # Get file from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return jsonify({'error': 'File not found'}), 404

            # Download file content
            file_content = blob.download_as_bytes()

        else:
            # Assume it's a signed URL
            response = requests.get(file_path)
            if response.status_code != 200:
                return jsonify({'error': 'Could not access file'}), 404
            file_content = response.content

        # Determine file type and create preview
        if file_path.endswith('.csv'):
            # Preview CSV as HTML table
            df = pd.read_csv(io.BytesIO(file_content))

            # Limit to first 100 rows for preview
            preview_df = df.head(100)

            # Create HTML table
            html_table = preview_df.to_html(
                classes='table table-striped table-hover',
                table_id='parcel-preview-table',
                escape=False,
                index=False
            )

            # Render preview template
            return render_template('file_preview.html',
                                   file_name=file_path.split('/')[-1],
                                   file_type='CSV',
                                   total_records=len(df),
                                   preview_records=len(preview_df),
                                   table_html=html_table,
                                   download_url=url_for('download_file', file_path=file_path),
                                   columns=list(df.columns))

        elif file_path.endswith('.gpkg'):
            # For GPKG files, show metadata and offer direct download
            import geopandas as gpd

            # Read GPKG metadata
            gdf = gpd.read_file(io.BytesIO(file_content))

            # Create summary information
            summary = {
                'total_features': len(gdf),
                'geometry_type': gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else 'Unknown',
                'crs': str(gdf.crs),
                'bounds': gdf.total_bounds.tolist() if len(gdf) > 0 else None,
                'columns': list(gdf.columns)
            }

            # Show first few rows as table (without geometry)
            preview_df = gdf.drop(columns=['geometry']).head(20)
            html_table = preview_df.to_html(
                classes='table table-striped table-hover',
                escape=False,
                index=False
            )

            return render_template('file_preview.html',
                                   file_name=file_path.split('/')[-1],
                                   file_type='GeoPackage',
                                   total_records=len(gdf),
                                   preview_records=len(preview_df),
                                   table_html=html_table,
                                   download_url=url_for('download_file', file_path=file_path),
                                   summary=summary,
                                   columns=list(gdf.columns))

        else:
            return jsonify({'error': 'Unsupported file type for preview'}), 400

    except Exception as e:
        logger.error(f"Error previewing file: {str(e)}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500


@app.route('/download/<path:file_path>')
def download_file(file_path):
    """
    Download file directly
    """
    try:
        if file_path.startswith('http'):
            # Redirect to signed URL for direct download
            return redirect(file_path)
        else:
            # Handle GCS path
            bucket_name = file_path.split('/')[2] if file_path.startswith('gs://') else 'bcfparcelsearchrepository'
            blob_name = '/'.join(file_path.split('/')[3:]) if file_path.startswith('gs://') else file_path

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return jsonify({'error': 'File not found'}), 404

            # Generate download URL
            download_url = blob.generate_signed_url(
                expiration=datetime.now() + pd.Timedelta(hours=1),
                method='GET'
            )

            return redirect(download_url)

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/api/parcel-data-preview')
def parcel_data_preview():
    """
    API endpoint to get parcel data preview as JSON
    """
    try:
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({'error': 'file_path parameter required'}), 400

        # Get CSV data
        if file_path.startswith('http'):
            response = requests.get(file_path)
            file_content = response.content
        else:
            # Handle GCS path
            client = storage.Client()
            bucket_name = file_path.split('/')[2] if file_path.startswith('gs://') else 'bcfparcelsearchrepository'
            blob_name = '/'.join(file_path.split('/')[3:]) if file_path.startswith('gs://') else file_path

            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            file_content = blob.download_as_bytes()

        # Read CSV
        df = pd.read_csv(io.BytesIO(file_content))

        # Return preview data
        preview_data = {
            'total_records': len(df),
            'columns': list(df.columns),
            'sample_data': df.head(50).to_dict('records'),
            'summary_stats': {
                'acreage_stats': df['acreage'].describe().to_dict() if 'acreage' in df.columns else None,
                'unique_owners': df['owner'].nunique() if 'owner' in df.columns else None,
                'counties': df['county_name'].unique().tolist() if 'county_name' in df.columns else []
            }
        }

        return jsonify(preview_data)

    except Exception as e:
        logger.error(f"Error getting parcel preview: {str(e)}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@app.route('/api/parcel/analyze_suitability', methods=['POST'])
def analyze_parcel_suitability():
    """
    Analyze renewable energy suitability for parcels
    Performs slope and transmission line analysis
    """
    try:
        data = request.get_json()

        # Get parcel data from the request - FIX HERE
        if 'parcels' in data and data['parcels']:
            parcels = data['parcels']
        elif 'search_results' in data and data['search_results'].get('parcel_data'):
            parcels = data['search_results']['parcel_data']
        else:
            return jsonify({'error': 'No parcel data provided in request'}), 400

        if not parcels or len(parcels) == 0:
            return jsonify({'error': 'No parcels found in data'}), 400

        logger.info(f"Starting suitability analysis for {len(parcels)} parcels")

        # Convert parcels to GeoDataFrame for analysis
        parcel_gdf = convert_parcels_to_geodataframe(parcels)

        if parcel_gdf is None or len(parcel_gdf) == 0:
            return jsonify({'error': 'Failed to convert parcels to spatial data'}), 400

        # Save parcels to temporary file for analysis modules
        temp_file_path = save_parcels_to_temp_gcs(parcel_gdf)

        if not temp_file_path:
            return jsonify({'error': 'Failed to prepare parcel data for analysis'}), 500

        # Run both analyses
        analysis_results = run_combined_suitability_analysis(
            temp_file_path,
            data.get('project_type', 'solar')
        )

        if analysis_results['status'] != 'success':
            return jsonify({'error': analysis_results['message']}), 500

        # Calculate suitability scores
        scored_parcels = calculate_parcel_suitability_scores(
            parcels,
            analysis_results,
            data.get('project_type', 'solar')
        )

        # Prepare response
        result_data = {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'total_parcels': len(scored_parcels),
            'suitable_parcels': len([p for p in scored_parcels if p['is_suitable']]),
            'analysis_parameters': {
                'slope_threshold': analysis_results.get('slope_threshold', 15),
                'transmission_buffer': analysis_results.get('transmission_buffer', 1.0),
                'voltage_range': '100-350 kV'
            },
            'parcels': scored_parcels
        }

        return jsonify({
            'status': 'success',
            'message': f'Analyzed {len(parcels)} parcels successfully',
            'data': result_data
        })

    except Exception as e:
        logger.error(f"Suitability analysis failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def convert_parcels_to_geodataframe(parcels):
    """Convert parcel list to GeoDataFrame"""
    try:
        from shapely.geometry import Point

        # Prepare data for GeoDataFrame
        gdf_data = []

        for i, parcel in enumerate(parcels):
            # Extract coordinates
            lat = float(parcel.get('latitude', parcel.get('lat', 0)))
            lon = float(parcel.get('longitude', parcel.get('lon', 0)))

            if lat == 0 or lon == 0:
                logger.warning(f"Parcel {i} has invalid coordinates: lat={lat}, lon={lon}")
                continue

            # Create point geometry
            geometry = Point(lon, lat)

            # Prepare parcel data
            parcel_data = {
                'parcel_id': parcel.get('parcel_id', f'PARCEL_{i:06d}'),
                'acres': float(parcel.get('acreage', parcel.get('acreage_calc', 0))),
                'owner': parcel.get('owner', 'Unknown'),
                'state': parcel.get('state_abbr', parcel.get('state', 'Unknown')),
                'county': parcel.get('county_name', parcel.get('county', 'Unknown')),
                'geometry': geometry
            }

            gdf_data.append(parcel_data)

        if not gdf_data:
            logger.error("No valid parcels found for analysis")
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(gdf_data, crs='EPSG:4326')
        logger.info(f"Created GeoDataFrame with {len(gdf)} parcels")

        return gdf

    except Exception as e:
        logger.error(f"Error converting parcels to GeoDataFrame: {e}")
        return None


def save_parcels_to_temp_gcs(parcel_gdf):
    """Save parcels to temporary GCS file for analysis"""
    try:
        from google.cloud import storage
        import uuid

        # Generate unique filename
        temp_filename = f"temp_suitability_analysis_{uuid.uuid4().hex[:8]}.gpkg"
        bucket_name = "bcfparcelsearchrepository"
        blob_path = f"temp_analysis/{temp_filename}"

        # Save to local temp file first
        temp_local_path = f"/tmp/{temp_filename}"
        parcel_gdf.to_file(temp_local_path, driver='GPKG')

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(temp_local_path)

        # Clean up local file
        os.unlink(temp_local_path)

        gcs_path = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"Saved parcels to temporary GCS file: {gcs_path}")

        return gcs_path

    except Exception as e:
        logger.error(f"Error saving parcels to GCS: {e}")
        return None


def run_combined_suitability_analysis(temp_file_path, project_type):
    """Run both slope and transmission analysis"""
    try:
        results = {
            'status': 'success',
            'slope_analysis': None,
            'transmission_analysis': None,
            'slope_threshold': 15.0,  # Default slope threshold
            'transmission_buffer': 1.0  # 1 mile buffer
        }

        # Set project-specific parameters
        if project_type.lower() == 'wind':
            results['slope_threshold'] = 20.0  # Wind can handle steeper slopes
            results['transmission_buffer'] = 2.0  # Larger buffer for wind

        logger.info("Starting slope analysis...")

        # Run slope analysis
        slope_result = run_slope_analysis(
            input_file_path=temp_file_path,
            max_slope_degrees=results['slope_threshold'],
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        if slope_result['status'] == 'success':
            results['slope_analysis'] = slope_result
            logger.info(f"Slope analysis completed: {slope_result.get('parcels_suitable_slope', 0)} suitable parcels")
        else:
            logger.error(f"Slope analysis failed: {slope_result.get('message', 'Unknown error')}")
            results['slope_analysis'] = {'status': 'failed', 'message': slope_result.get('message', 'Unknown error')}

        logger.info("Starting transmission analysis...")

        # Run transmission analysis
        transmission_result = run_transmission_analysis(
            input_file_path=temp_file_path,
            buffer_distance_miles=results['transmission_buffer'],
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        if transmission_result['status'] == 'success':
            results['transmission_analysis'] = transmission_result
            logger.info(
                f"Transmission analysis completed: {transmission_result.get('parcels_near_transmission', 0)} parcels near transmission")
        else:
            logger.error(f"Transmission analysis failed: {transmission_result.get('message', 'Unknown error')}")
            results['transmission_analysis'] = {'status': 'failed',
                                                'message': transmission_result.get('message', 'Unknown error')}

        return results

    except Exception as e:
        logger.error(f"Combined analysis failed: {e}")
        return {
            'status': 'error',
            'message': f'Combined analysis failed: {str(e)}'
        }


def calculate_parcel_suitability_scores(original_parcels, analysis_results, project_type):
    """Calculate suitability scores for each parcel"""
    try:
        scored_parcels = []

        # Load analysis results if available
        slope_data = {}
        transmission_data = {}

        # Load slope analysis results
        if (analysis_results.get('slope_analysis') and
                analysis_results['slope_analysis'].get('status') == 'success'):

            slope_file_path = analysis_results['slope_analysis'].get('output_file_path')
            if slope_file_path:
                slope_data = load_analysis_results_from_gcs(slope_file_path)

        # Load transmission analysis results
        if (analysis_results.get('transmission_analysis') and
                analysis_results['transmission_analysis'].get('status') == 'success'):

            transmission_file_path = analysis_results['transmission_analysis'].get('output_file_path')
            if transmission_file_path:
                transmission_data = load_analysis_results_from_gcs(transmission_file_path)

        logger.info(f"Loaded slope data for {len(slope_data)} parcels")
        logger.info(f"Loaded transmission data for {len(transmission_data)} parcels")

        # Process each parcel
        for i, parcel in enumerate(original_parcels):
            parcel_id = parcel.get('parcel_id', f'PARCEL_{i:06d}')

            # Get analysis results for this parcel
            slope_info = slope_data.get(parcel_id, {})
            transmission_info = transmission_data.get(parcel_id, {})

            # Calculate scores
            slope_score, slope_suitable = calculate_slope_score(slope_info, analysis_results['slope_threshold'])
            transmission_score, transmission_suitable = calculate_transmission_score(transmission_info, project_type)

            # Overall suitability (both criteria must be met)
            is_suitable = slope_suitable and transmission_suitable
            overall_score = (slope_score + transmission_score) / 2

            # Create scored parcel
            scored_parcel = {
                **parcel,  # Include all original data
                'parcel_id': parcel_id,
                'suitability_analysis': {
                    'is_suitable': is_suitable,
                    'overall_score': round(overall_score, 1),
                    'slope_score': round(slope_score, 1),
                    'transmission_score': round(transmission_score, 1),
                    'slope_suitable': slope_suitable,
                    'transmission_suitable': transmission_suitable,
                    'slope_degrees': slope_info.get('avg_slope_degrees', 'Unknown'),
                    'transmission_distance': transmission_info.get('tx_nearest_distance', 'Unknown'),
                    'transmission_voltage': transmission_info.get('tx_max_voltage', 'Unknown'),
                    'analysis_notes': generate_analysis_notes(slope_info, transmission_info, is_suitable)
                },
                'is_suitable': is_suitable  # Top-level flag for easy filtering
            }

            scored_parcels.append(scored_parcel)

        # Sort by suitability and score
        scored_parcels.sort(key=lambda x: (x['is_suitable'], x['suitability_analysis']['overall_score']), reverse=True)

        logger.info(f"Completed suitability scoring for {len(scored_parcels)} parcels")

        return scored_parcels

    except Exception as e:
        logger.error(f"Error calculating suitability scores: {e}")
        return original_parcels  # Return original data if scoring fails


def load_analysis_results_from_gcs(gcs_file_path):
    """Load analysis results from GCS file"""
    try:
        if not gcs_file_path or not gcs_file_path.startswith('gs://'):
            return {}

        # Download file to temp location
        temp_file = download_from_gcs(gcs_file_path)
        if not temp_file:
            return {}

        # Read the file
        if gcs_file_path.endswith('.gpkg'):
            gdf = gpd.read_file(temp_file)
        elif gcs_file_path.endswith('.csv'):
            gdf = pd.read_csv(temp_file)
        else:
            logger.error(f"Unsupported file format: {gcs_file_path}")
            return {}

        # Convert to dictionary keyed by parcel_id
        result_data = {}
        for _, row in gdf.iterrows():
            parcel_id = row.get('parcel_id', '')
            if parcel_id:
                result_data[parcel_id] = row.to_dict()

        # Clean up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

        return result_data

    except Exception as e:
        logger.error(f"Error loading analysis results from {gcs_file_path}: {e}")
        return {}

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

        from google.cloud import storage
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


def test_suitability_analysis():
    """Test function to verify the suitability analysis works"""
    try:
        # Create test parcel data
        test_parcels = [
            {
                'parcel_id': 'TEST_001',
                'acreage': 50.0,
                'owner': 'Test Owner 1',
                'latitude': 40.5,
                'longitude': -78.4,
                'state_abbr': 'PA',
                'county_name': 'Blair'
            },
            {
                'parcel_id': 'TEST_002',
                'acreage': 100.0,
                'owner': 'Test Owner 2',
                'latitude': 40.51,
                'longitude': -78.41,
                'state_abbr': 'PA',
                'county_name': 'Blair'
            }
        ]

        # Test the conversion function
        gdf = convert_parcels_to_geodataframe(test_parcels)
        if gdf is not None and len(gdf) > 0:
            print(f"✅ Successfully converted {len(gdf)} test parcels to GeoDataFrame")
            return True
        else:
            print("❌ Failed to convert test parcels")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


## Step 6: Debugging Tips

def debug_analysis_modules():
    """Check if analysis modules are working"""
    try:
        # Test slope analysis module
        from bigquery_slope_analysis import verify_analysis_dependencies as verify_slope
        slope_status = verify_slope()
        print(f"Slope Analysis Dependencies: {'✅ Ready' if slope_status['ready_for_analysis'] else '❌ Not Ready'}")

        # Test transmission analysis module
        from transmission_analysis_bigquery import verify_analysis_dependencies as verify_transmission
        transmission_status = verify_transmission()
        print(
            f"Transmission Analysis Dependencies: {'✅ Ready' if transmission_status['ready_for_analysis'] else '❌ Not Ready'}")

        return slope_status['ready_for_analysis'] and transmission_status['ready_for_analysis']

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure your analysis modules are in the correct location")
        return False
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False


## Step 7: Error Handling Improvements

class SuitabilityAnalysisError(Exception):
    """Custom exception for suitability analysis errors"""
    pass


def validate_parcel_data(parcels):
    """Validate parcel data before analysis"""
    if not parcels or len(parcels) == 0:
        raise SuitabilityAnalysisError("No parcel data provided")

    required_fields = ['parcel_id', 'latitude', 'longitude']
    missing_fields = []

    for i, parcel in enumerate(parcels[:5]):  # Check first 5 parcels
        for field in required_fields:
            if field not in parcel or not parcel[field]:
                missing_fields.append(f"Parcel {i}: missing {field}")

    if missing_fields:
        raise SuitabilityAnalysisError(f"Invalid parcel data: {', '.join(missing_fields)}")

    return True


## Step 8: Performance Optimization

def optimize_analysis_for_large_datasets(parcels, max_batch_size=100):
    """Handle large datasets by batching"""
    if len(parcels) <= max_batch_size:
        return [parcels]  # Single batch

    # Split into batches
    batches = []
    for i in range(0, len(parcels), max_batch_size):
        batch = parcels[i:i + max_batch_size]
        batches.append(batch)

    logger.info(f"Split {len(parcels)} parcels into {len(batches)} batches for processing")
    return batches


## Step 9: Configuration Options

SUITABILITY_CONFIG = {
    'slope_thresholds': {
        'solar': 15.0,  # degrees
        'wind': 20.0  # degrees
    },
    'transmission_buffers': {
        'solar': 1.0,  # miles
        'wind': 2.0  # miles
    },
    'voltage_range': {
        'min': 100,  # kV
        'max': 350  # kV
    },
    'scoring_weights': {
        'slope': 0.4,  # 40% weight
        'transmission': 0.4,  # 40% weight
        'proximity': 0.2  # 20% weight
    }
}


def get_project_config(project_type):
    """Get configuration for specific project type"""
    return {
        'slope_threshold': SUITABILITY_CONFIG['slope_thresholds'].get(project_type.lower(), 15.0),
        'transmission_buffer': SUITABILITY_CONFIG['transmission_buffers'].get(project_type.lower(), 1.0),
        'voltage_min': SUITABILITY_CONFIG['voltage_range']['min'],
        'voltage_max': SUITABILITY_CONFIG['voltage_range']['max']
    }


def calculate_slope_score(slope_info, threshold):
    """Calculate slope suitability score"""
    if not slope_info:
        return 50, False  # Unknown slope, not suitable

    avg_slope = slope_info.get('avg_slope_degrees', 999)

    if avg_slope == 'Unknown' or avg_slope == 999:
        return 50, False

    try:
        slope_value = float(avg_slope)

        # Score based on slope (lower is better)
        if slope_value <= threshold * 0.5:  # Very gentle
            score = 100
        elif slope_value <= threshold * 0.75:  # Gentle
            score = 80
        elif slope_value <= threshold:  # Acceptable
            score = 60
        else:  # Too steep
            score = 20

        suitable = slope_value <= threshold

        return score, suitable

    except (ValueError, TypeError):
        return 50, False


def calculate_transmission_score(transmission_info, project_type):
    """Calculate transmission line proximity score"""
    if not transmission_info:
        return 20, False  # No transmission data, not suitable

    # Get distance to nearest transmission line
    distance = transmission_info.get('tx_nearest_distance', 999)
    max_voltage = transmission_info.get('tx_max_voltage', 0)

    if distance == 'Unknown' or distance == 999:
        return 20, False

    try:
        distance_value = float(distance)
        voltage_value = float(max_voltage) if max_voltage != 'Unknown' else 0

        # Check voltage range (100-350 kV)
        voltage_suitable = 100 <= voltage_value <= 350

        # Score based on distance (closer is better, but not too close)
        if distance_value <= 0.25:  # Very close (within 0.25 miles)
            score = 100 if voltage_suitable else 40
        elif distance_value <= 0.5:  # Close (within 0.5 miles)
            score = 90 if voltage_suitable else 35
        elif distance_value <= 1.0:  # Moderate (within 1 mile)
            score = 70 if voltage_suitable else 30
        elif distance_value <= 2.0:  # Far (within 2 miles)
            score = 50 if voltage_suitable else 20
        else:  # Too far
            score = 20

        # For wind projects, allow slightly larger distances
        if project_type.lower() == 'wind' and distance_value <= 3.0:
            score = max(score, 40)

        suitable = voltage_suitable and distance_value <= (2.0 if project_type.lower() == 'wind' else 1.5)

        return score, suitable

    except (ValueError, TypeError):
        return 20, False


def generate_analysis_notes(slope_info, transmission_info, is_suitable):
    """Generate human-readable analysis notes"""
    notes = []

    # Slope notes
    if slope_info:
        slope = slope_info.get('avg_slope_degrees', 'Unknown')
        if slope != 'Unknown':
            try:
                slope_val = float(slope)
                if slope_val <= 5:
                    notes.append(f"Excellent slope ({slope_val:.1f}°)")
                elif slope_val <= 10:
                    notes.append(f"Good slope ({slope_val:.1f}°)")
                elif slope_val <= 15:
                    notes.append(f"Acceptable slope ({slope_val:.1f}°)")
                else:
                    notes.append(f"Steep slope ({slope_val:.1f}°)")
            except:
                notes.append("Slope data available")
    else:
        notes.append("No slope data")

    # Transmission notes
    if transmission_info:
        distance = transmission_info.get('tx_nearest_distance', 'Unknown')
        voltage = transmission_info.get('tx_max_voltage', 'Unknown')

        if distance != 'Unknown' and voltage != 'Unknown':
            try:
                dist_val = float(distance)
                volt_val = float(voltage)
                notes.append(f"Transmission: {dist_val:.2f} mi, {volt_val:.0f} kV")
            except:
                notes.append("Transmission data available")
        else:
            notes.append("Limited transmission data")
    else:
        notes.append("No transmission data")

    # Overall assessment
    if is_suitable:
        notes.append("✓ Suitable for development")
    else:
        notes.append("✗ Requires further evaluation")

    return "; ".join(notes)

if __name__ == '__main__':
    # For Cloud Run, use PORT env variable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)