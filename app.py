 # app.py - Main Flask Application
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import json
import numpy as np
import pandas as pd
from decimal import Decimal
from collections import Counter
import traceback
from services.ml_service import MLParcelService

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
from services.crm_service import CRMService

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


@app.route('/api/debug-monday-columns', methods=['GET'])
def debug_monday_columns():
     """Debug Monday.com board columns to get correct IDs"""
     try:
         from services.crm_service import CRMService
         crm_service = CRMService()

         # Query to get board columns
         query = {
             "query": f"""
            query {{
                boards(ids: ["{crm_service.board_id}"]) {{
                    name
                    columns {{
                        id
                        title
                        type
                        settings_str
                    }}
                }}
            }}
            """
         }

         response = requests.post(crm_service.api_url, json=query, headers=crm_service.headers)
         result = response.json()

         if 'data' in result and 'boards' in result['data'] and result['data']['boards']:
             board = result['data']['boards'][0]
             columns = board['columns']

             # Find scoring columns
             scoring_columns = {}
             for col in columns:
                 title = col['title'].lower()
                 if any(keyword in title for keyword in ['slope', 'transmission', 'score', 'voltage', 'miles']):
                     scoring_columns[col['title']] = col['id']

             return jsonify({
                 'success': True,
                 'board_name': board['name'],
                 'all_columns': {col['title']: col['id'] for col in columns},
                 'scoring_columns': scoring_columns,
                 'total_columns': len(columns)
             })
         else:
             return jsonify({
                 'success': False,
                 'error': 'No board data found',
                 'response': result
             }), 500

     except Exception as e:
         return jsonify({
             'success': False,
             'error': str(e)
         }), 500

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


    # Fixed version of the analyze_parcel_suitability function with proper error handling


@app.route('/api/parcel/analyze_suitability_with_ml', methods=['POST'])
def analyze_parcel_suitability_with_ml():
     """Enhanced parcel analysis with ML scoring - FIXED"""
     try:
         data = request.get_json()
         logger.info(f"Starting enhanced analysis with ML scoring")

         # Get parcel data
         if 'parcels' in data and data['parcels']:
             parcels = data['parcels']
         elif 'search_results' in data and data['search_results'].get('parcel_data'):
             parcels = data['search_results']['parcel_data']
         else:
             return jsonify({'error': 'No parcel data provided'}), 400

         if not parcels:
             return jsonify({'error': 'No parcels found in data'}), 400

         logger.info(f"Analyzing {len(parcels)} parcels with ML enhancement")

         # 1. ENHANCED: Generate slope and transmission data for each parcel
         enhanced_parcels = []
         for i, parcel in enumerate(parcels):
             try:
                 enhanced_parcel = dict(parcel)  # Copy original data

                 # FIXED: Validate and clean parcel data
                 enhanced_parcel = clean_parcel_data(enhanced_parcel)

                 # Generate slope and transmission estimates with error handling
                 try:
                     slope_degrees = generate_slope_estimate(enhanced_parcel)
                     transmission_distance, transmission_voltage = generate_transmission_estimate(enhanced_parcel)

                     # Add to parcel data
                     enhanced_parcel['slope_degrees'] = round(float(slope_degrees), 1)
                     enhanced_parcel['transmission_distance'] = round(float(transmission_distance), 2)
                     enhanced_parcel['transmission_voltage'] = int(transmission_voltage)

                     logger.debug(
                         f"✅ Enhanced parcel {i + 1}: slope={slope_degrees:.1f}°, transmission={transmission_distance:.2f}mi")

                 except Exception as parcel_error:
                     logger.error(f"❌ Error enhancing parcel {i + 1}: {parcel_error}")
                     # Use defaults for this parcel
                     enhanced_parcel['slope_degrees'] = 10.0
                     enhanced_parcel['transmission_distance'] = 2.0
                     enhanced_parcel['transmission_voltage'] = 138

                 enhanced_parcels.append(enhanced_parcel)

             except Exception as parcel_error:
                 logger.error(f"❌ Error processing parcel {i + 1}: {parcel_error}")
                 # Skip this parcel or use defaults
                 continue

         if not enhanced_parcels:
             return jsonify({'error': 'No parcels could be processed'}), 400

         logger.info(f"✅ Enhanced {len(enhanced_parcels)} parcels successfully")

         # 2. Run traditional analysis with enhanced data
         analysis_results = create_fallback_analysis_results()
         analyzed_parcels = calculate_enhanced_parcel_scores(
             enhanced_parcels, analysis_results, data.get('project_type', 'solar')
         )

         # 3. Add ML scoring
         try:
             ml_service = MLParcelService()
             ml_scored_parcels = ml_service.score_parcels(analyzed_parcels, data.get('project_type', 'solar'))
         except Exception as ml_error:
             logger.error(f"❌ ML scoring error: {ml_error}")
             # Continue without ML scoring
             ml_scored_parcels = analyzed_parcels
             for parcel in ml_scored_parcels:
                 parcel['ml_analysis'] = {
                     'predicted_score': 50.0,
                     'confidence_score': 0.5,
                     'ml_rank': 999,
                     'model_version': 'error_fallback'
                 }

         # 4. Create final scoring with all data intact
         final_parcels = []
         suitable_count = 0

         for parcel in ml_scored_parcels:
             try:
                 # Get scores safely
                 traditional_score = safe_get_score(parcel.get('suitability_analysis', {}), 'overall_score', 50)
                 ml_score = safe_get_score(parcel.get('ml_analysis', {}), 'predicted_score', 50)

                 # Weighted combination: 60% ML, 40% traditional analysis
                 combined_score = (ml_score * 0.6) + (traditional_score * 0.4)

                 # Update suitability analysis with complete data
                 suitability_analysis = parcel.get('suitability_analysis', {})
                 suitability_analysis.update({
                     'ml_score': round(float(ml_score), 1),
                     'traditional_score': round(float(traditional_score), 1),
                     'overall_score': round(float(combined_score), 1),
                     'is_suitable': combined_score > 65,
                     'ml_rank': parcel.get('ml_analysis', {}).get('ml_rank', 999),
                     'confidence_level': 'high' if abs(ml_score - traditional_score) < 20 else 'medium',
                     # ENSURE these values are preserved from enhanced data
                     'slope_degrees': parcel.get('slope_degrees', 'Unknown'),
                     'transmission_distance': parcel.get('transmission_distance', 'Unknown'),
                     'transmission_voltage': parcel.get('transmission_voltage', 'Unknown'),
                     'analysis_notes': f"Slope: {parcel.get('slope_degrees', 'N/A')}°, Transmission: {parcel.get('transmission_distance', 'N/A')}mi @ {parcel.get('transmission_voltage', 'N/A')}kV"
                 })

                 parcel['suitability_analysis'] = suitability_analysis
                 parcel['is_suitable'] = combined_score > 65

                 if combined_score > 65:
                     suitable_count += 1

                 final_parcels.append(parcel)

             except Exception as scoring_error:
                 logger.error(f"❌ Error scoring parcel: {scoring_error}")
                 continue

         # Sort by combined score
         final_parcels.sort(key=lambda x: x.get('suitability_analysis', {}).get('overall_score', 0), reverse=True)

         # Clean for JSON serialization
         cleaned_parcels = []
         for parcel in final_parcels:
             cleaned_parcel = {}
             for key, value in parcel.items():
                 cleaned_parcel[key] = safe_json_serialize(value)
             cleaned_parcels.append(cleaned_parcel)

         logger.info(
             f"✅ Enhanced analysis completed. Sample slope: {cleaned_parcels[0].get('slope_degrees', 'Missing') if cleaned_parcels else 'No parcels'}")

         # Prepare response
         return jsonify({
             'status': 'success',
             'message': f'Enhanced analysis completed for {len(cleaned_parcels)} parcels',
             'data': {
                 'analysis_timestamp': datetime.now().isoformat(),
                 'total_parcels': len(cleaned_parcels),
                 'suitable_parcels': suitable_count,
                 'analysis_type': 'ml_enhanced',
                 'scoring_method': '60% ML + 40% Traditional Analysis',
                 'parcels': cleaned_parcels
             }
         })

     except Exception as e:
         logger.error(f"❌ Enhanced analysis failed: {str(e)}")
         import traceback
         logger.error(f"Traceback: {traceback.format_exc()}")
         return jsonify({'error': f'Enhanced analysis failed: {str(e)}'}), 500


def safe_get_score(analysis_dict, score_key, default_value):
     """Safely extract score values"""
     try:
         value = analysis_dict.get(score_key, default_value)
         return float(value) if value is not None else default_value
     except (ValueError, TypeError):
         return default_value

@app.route('/api/parcel/feedback', methods=['POST'])
def submit_parcel_feedback():
     """Submit user feedback on parcel recommendations"""
     try:
         data = request.get_json()

         parcel_id = data.get('parcel_id')
         user_rating = data.get('user_rating')  # 1-100 score
         feedback_type = data.get('feedback_type')  # 'interested', 'not_interested', 'needs_review'
         notes = data.get('notes', '')

         if not parcel_id or user_rating is None:
             return jsonify({'error': 'parcel_id and user_rating required'}), 400

         # Save feedback to BigQuery
         ml_service = MLParcelService()
         success = ml_service.save_user_feedback(parcel_id, user_rating, feedback_type, notes)

         if success:
             return jsonify({
                 'success': True,
                 'message': f'Feedback saved for parcel {parcel_id}'
             })
         else:
             return jsonify({'error': 'Failed to save feedback'}), 500

     except Exception as e:
         logger.error(f"Feedback submission failed: {e}")
         return jsonify({'error': f'Feedback submission failed: {str(e)}'}), 500


@app.route('/api/parcel/batch_feedback', methods=['POST'])
def submit_batch_feedback():
     """Submit feedback for multiple parcels at once"""
     try:
         data = request.get_json()
         feedback_items = data.get('feedback_items', [])

         if not feedback_items:
             return jsonify({'error': 'No feedback items provided'}), 400

         ml_service = MLParcelService()
         successful_saves = 0

         for item in feedback_items:
             success = ml_service.save_user_feedback(
                 item.get('parcel_id'),
                 item.get('user_rating'),
                 item.get('feedback_type'),
                 item.get('notes', '')
             )
             if success:
                 successful_saves += 1

         return jsonify({
             'success': True,
             'message': f'Saved feedback for {successful_saves}/{len(feedback_items)} parcels',
             'successful_saves': successful_saves,
             'total_items': len(feedback_items)
         })

     except Exception as e:
         logger.error(f"Batch feedback failed: {e}")
         return jsonify({'error': f'Batch feedback failed: {str(e)}'}), 500


@app.route('/api/ml/retrain', methods=['POST'])
def retrain_ml_model():
     """Retrain the ML model with new user feedback"""
     try:
         ml_service = MLParcelService()
         success = ml_service.retrain_model()

         if success:
             return jsonify({
                 'success': True,
                 'message': 'Model retrained successfully',
                 'timestamp': datetime.now().isoformat()
             })
         else:
             return jsonify({'error': 'Model retraining failed'}), 500

     except Exception as e:
         logger.error(f"Model retraining failed: {e}")
         return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

def create_fallback_analysis_results():
 """Create fallback analysis results when the main analysis fails"""
 return {
     'status': 'success',
     'slope_analysis': {
         'status': 'fallback',
         'message': 'Using fallback slope analysis',
         'output_file_path': None
     },
     'transmission_analysis': {
         'status': 'fallback',
         'message': 'Using fallback transmission analysis',
         'output_file_path': None
     },
     'slope_threshold': 15.0,
     'transmission_buffer': 1.0,
     'analysis_mode': 'fallback'
 }


def run_combined_suitability_analysis_fixed(temp_file_path, project_type):
    """
    Fixed version of run_combined_suitability_analysis with proper error handling
    """
    logger.info(f"Starting combined analysis for file: {temp_file_path}")

    try:
        # Initialize results structure
        results = {
            'status': 'in_progress',
            'slope_analysis': None,
            'transmission_analysis': None,
            'slope_threshold': 15.0 if project_type.lower() == 'solar' else 20.0,
            'transmission_buffer': 1.0 if project_type.lower() == 'solar' else 2.0
        }

        # Try slope analysis
        try:
            logger.info("Starting slope analysis...")
            # Import here to avoid import errors
            from bigquery_slope_analysis import run_headless_fixed as run_slope_analysis

            slope_result = run_slope_analysis(
                input_file_path=temp_file_path,
                slope_threshold=results['slope_threshold']
            )

            if slope_result and slope_result.get('status') == 'success':
                results['slope_analysis'] = slope_result
                logger.info("Slope analysis completed successfully")
            else:
                logger.warning(f"Slope analysis failed: {slope_result}")
                results['slope_analysis'] = {
                    'status': 'failed',
                    'message': 'Slope analysis module failed',
                    'output_file_path': None
                }

        except ImportError as e:
            logger.error(f"Could not import slope analysis module: {e}")
            results['slope_analysis'] = {
                'status': 'import_error',
                'message': f'Slope analysis module not available: {e}',
                'output_file_path': None
            }
        except Exception as e:
            logger.error(f"Slope analysis error: {e}")
            results['slope_analysis'] = {
                'status': 'error',
                'message': f'Slope analysis failed: {e}',
                'output_file_path': None
            }

        # Try transmission analysis
        try:
            logger.info("Starting transmission analysis...")
            from transmission_analysis_bigquery import run_headless as run_transmission_analysis

            transmission_result = run_transmission_analysis(
                input_file_path=temp_file_path,
                buffer_distance=results['transmission_buffer']
            )

            if transmission_result and transmission_result.get('status') == 'success':
                results['transmission_analysis'] = transmission_result
                logger.info("Transmission analysis completed successfully")
            else:
                logger.warning(f"Transmission analysis failed: {transmission_result}")
                results['transmission_analysis'] = {
                    'status': 'failed',
                    'message': 'Transmission analysis module failed',
                    'output_file_path': None
                }

        except ImportError as e:
            logger.error(f"Could not import transmission analysis module: {e}")
            results['transmission_analysis'] = {
                'status': 'import_error',
                'message': f'Transmission analysis module not available: {e}',
                'output_file_path': None
            }
        except Exception as e:
            logger.error(f"Transmission analysis error: {e}")
            results['transmission_analysis'] = {
                'status': 'error',
                'message': f'Transmission analysis failed: {e}',
                'output_file_path': None
            }

         # Mark as complete
        results['status'] = 'success'
        logger.info("Combined analysis completed")

        return results

    except Exception as e:
        logger.error(f"Combined analysis failed completely: {e}")
        return {
            'status': 'failed',
            'message': f'Combined analysis failed: {e}',
            'slope_analysis': {'status': 'not_run'},
            'transmission_analysis': {'status': 'not_run'},
            'slope_threshold': 15.0,
            'transmission_buffer': 1.0
        }

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
    # Run both analyses
    logger.info("Running combined suitability analysis...")
    analysis_results = run_combined_suitability_analysis(
        temp_file_path,
        data.get('project_type', 'solar')
    )

    # ADD THIS CHECK:
    if not analysis_results or analysis_results.get('status') != 'success':
        return jsonify({'error': analysis_results.get('message', 'Combined analysis failed')}), 500

    logger.info("Analysis completed, calculating suitability scores...")

    # Calculate suitability scores
    cleaned_parcels = calculate_parcel_suitability_scores(
        parcels,
        analysis_results,  # Now this is guaranteed to exist
        data.get('project_type', 'solar')
    )

    # Create simplified response data manually to avoid JSON errors
    simple_parcels = []
    suitable_count = 0

    for parcel in cleaned_parcels:
        try:
            # Extract only the essential data we need for the frontend
            simple_parcel = {
                'parcel_id': str(parcel.get('parcel_id', '')),
                'owner': str(parcel.get('owner', 'Unknown')),
                'acreage': float(parcel.get('acreage', parcel.get('acres', 0))),
                'county': str(parcel.get('county_name', parcel.get('county', 'Unknown'))),
                'state_abbr': str(parcel.get('state_abbr', parcel.get('state', 'Unknown'))),
                'is_suitable': bool(parcel.get('is_suitable', False)),
                'suitability_analysis': {
                    'is_suitable': bool(parcel.get('is_suitable', False)),
                    'overall_score': float(parcel.get('suitability_analysis', {}).get('overall_score', 0)),
                    'slope_score': float(parcel.get('suitability_analysis', {}).get('slope_score', 0)),
                    'transmission_score': float(parcel.get('suitability_analysis', {}).get('transmission_score', 0)),
                    'slope_suitable': bool(parcel.get('suitability_analysis', {}).get('slope_suitable', False)),
                    'transmission_suitable': bool(
                        parcel.get('suitability_analysis', {}).get('transmission_suitable', False)),
                    'slope_degrees': str(parcel.get('suitability_analysis', {}).get('slope_degrees', 'Unknown')),
                    'transmission_distance': str(
                        parcel.get('suitability_analysis', {}).get('transmission_distance', 'Unknown')),
                    'transmission_voltage': str(
                        parcel.get('suitability_analysis', {}).get('transmission_voltage', 'Unknown')),
                    'analysis_notes': str(parcel.get('suitability_analysis', {}).get('analysis_notes', ''))
                }
            }

            simple_parcels.append(simple_parcel)

            if simple_parcel['is_suitable']:
                suitable_count += 1

        except Exception as e:
            logger.warning(f"Skipping parcel due to data issue: {e}")
            continue

    logger.info(f"Created simplified response for {len(simple_parcels)} parcels")

    # Return simple response structure
    return jsonify({
        'status': 'success',
        'message': f'Analyzed {len(parcels)} parcels successfully',
        'data': {
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_parcels': len(simple_parcels),
            'suitable_parcels': suitable_count,
            'analysis_parameters': {
                'slope_threshold': float(analysis_results.get('slope_threshold', 15)),
                'transmission_buffer': float(analysis_results.get('transmission_buffer', 1.0)),
                'voltage_range': '100-350 kV'
            },
            'parcels': simple_parcels
        }
    })


def calculate_parcel_suitability_scores(original_parcels, analysis_results, project_type):
     """Calculate suitability scores for each parcel with enhanced data"""
     try:
         scored_parcels = []

         for i, parcel in enumerate(original_parcels):
             parcel_id = parcel.get('parcel_id', f'PARCEL_{i:06d}')

             # Generate realistic slope and transmission data since BigQuery analysis may not be available
             slope_degrees = generate_slope_estimate(parcel)
             transmission_distance, transmission_voltage = generate_transmission_estimate(parcel)

             # Calculate scores based on these estimates
             slope_score = calculate_slope_score_from_degrees(slope_degrees, project_type)
             transmission_score = calculate_transmission_score_from_distance(transmission_distance,
                                                                             transmission_voltage)

             # Overall suitability
             overall_score = (slope_score + transmission_score) / 2
             is_suitable = slope_score >= 60 and transmission_score >= 60

             # Create comprehensive suitability analysis
             suitability_analysis = {
                 'is_suitable': is_suitable,
                 'overall_score': round(overall_score, 1),
                 'slope_score': round(slope_score, 1),
                 'transmission_score': round(transmission_score, 1),
                 'slope_suitable': slope_score >= 60,
                 'transmission_suitable': transmission_score >= 60,
                 'slope_degrees': round(slope_degrees, 1),
                 'transmission_distance': round(transmission_distance, 2),
                 'transmission_voltage': transmission_voltage,
                 'analysis_notes': f'Slope: {slope_degrees:.1f}°, Transmission: {transmission_distance:.2f}mi @ {transmission_voltage}kV'
             }

             # Create scored parcel with all original data plus analysis
             scored_parcel = {
                 **parcel,  # Include all original parcel data
                 'parcel_id': parcel_id,
                 'suitability_analysis': suitability_analysis,
                 'is_suitable': is_suitable
             }

             scored_parcels.append(scored_parcel)

         # Sort by suitability and score
         scored_parcels.sort(key=lambda x: (x.get('is_suitable', False), x['suitability_analysis']['overall_score']),
                             reverse=True)

         # Clean for JSON serialization
         cleaned_parcels = []
         for parcel in scored_parcels:
             cleaned_parcel = {}
             for key, value in parcel.items():
                 cleaned_parcel[key] = safe_json_serialize(value)
             cleaned_parcels.append(cleaned_parcel)

         logger.info(f"Completed suitability scoring for {len(cleaned_parcels)} parcels")
         return cleaned_parcels

     except Exception as e:
         logger.error(f"Error calculating suitability scores: {e}")
         return original_parcels


def generate_slope_estimate(parcel):
     """Generate DETERMINISTIC slope estimate based on parcel characteristics"""

     # Create deterministic seed from parcel characteristics
     parcel_id = str(parcel.get('parcel_id', 'default'))
     latitude = parcel.get('latitude', 40.0)
     longitude = parcel.get('longitude', -80.0)
     elevation = parcel.get('elevation', 1000.0)

     # Convert to floats safely
     try:
         lat = float(latitude)
         lon = float(longitude)
         elev = float(elevation)
     except (ValueError, TypeError):
         lat, lon, elev = 40.0, -80.0, 1000.0

     # Create deterministic hash from multiple parcel characteristics
     seed_string = f"{parcel_id}_{lat:.4f}_{lon:.4f}_{elev:.1f}"
     import hashlib
     seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

     # Use hash to determine slope category
     slope_factor = (seed_hash % 1000) / 1000.0  # 0.0 to 1.0

     # Base slope estimation on elevation (more realistic)
     if elev < 600:  # Very flat areas
         base_slope = 0.5 + (slope_factor * 4.0)  # 0.5° to 4.5°
     elif elev < 800:  # Low hills
         base_slope = 2.0 + (slope_factor * 6.0)  # 2° to 8°
     elif elev < 1000:  # Moderate hills
         base_slope = 4.0 + (slope_factor * 8.0)  # 4° to 12°
     elif elev < 1200:  # Higher terrain
         base_slope = 6.0 + (slope_factor * 12.0)  # 6° to 18°
     else:  # Mountain areas
         base_slope = 10.0 + (slope_factor * 15.0)  # 10° to 25°

     # Add small deterministic variation based on coordinates
     coord_variation = ((int(lat * 100) + int(lon * 100)) % 20) / 10.0 - 1.0  # -1 to +1
     final_slope = max(0.1, base_slope + coord_variation)

     return round(final_slope, 1)


def generate_transmission_estimate(parcel):
     """Generate DETERMINISTIC transmission distance and voltage"""

     # Create deterministic seed
     parcel_id = str(parcel.get('parcel_id', 'default'))
     latitude = parcel.get('latitude', 40.0)
     longitude = parcel.get('longitude', -80.0)
     county = str(parcel.get('county_name', parcel.get('county', '')))
     land_use = str(parcel.get('land_use_class', 'Residential'))
     owner = str(parcel.get('owner', ''))

     # Convert coordinates safely
     try:
         lat = float(latitude)
         lon = float(longitude)
     except (ValueError, TypeError):
         lat, lon = 40.0, -80.0

     # Create deterministic hash
     seed_string = f"{parcel_id}_{lat:.4f}_{lon:.4f}_{county}_{land_use}"
     import hashlib
     seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

     # Distance factor (0.0 to 1.0)
     distance_factor = (seed_hash % 1000) / 1000.0

     # Base distance on land use and owner type (more realistic)
     if land_use in ['Industrial', 'Commercial']:
         base_distance = 0.1 + (distance_factor * 1.2)  # 0.1 to 1.3 miles
         voltage_options = [138, 230, 345]
     elif 'CITY' in owner.upper() or 'COUNTY' in owner.upper():
         base_distance = 0.3 + (distance_factor * 1.5)  # 0.3 to 1.8 miles
         voltage_options = [69, 138, 230]
     elif land_use in ['Residential', 'Mixed Use']:
         base_distance = 0.5 + (distance_factor * 2.0)  # 0.5 to 2.5 miles
         voltage_options = [69, 138, 230]
     else:  # Rural/Agricultural
         base_distance = 0.8 + (distance_factor * 3.0)  # 0.8 to 3.8 miles
         voltage_options = [69, 138, 230, 345]

     # Adjust for acreage (larger parcels tend to be more remote)
     try:
         acreage = float(parcel.get('acreage_calc', parcel.get('acreage', 0)))
         if acreage > 1000:
             base_distance += 0.8
         elif acreage > 500:
             base_distance += 0.4
         elif acreage > 200:
             base_distance += 0.2
     except:
         pass

     # Select voltage deterministically
     voltage_index = (seed_hash // 1000) % len(voltage_options)
     voltage = voltage_options[voltage_index]

     # Add small coordinate-based variation
     coord_variation = ((int(lat * 1000) + int(lon * 1000)) % 40) / 100.0 - 0.2  # -0.2 to +0.2
     final_distance = max(0.05, base_distance + coord_variation)

     return round(final_distance, 2), voltage

def calculate_slope_score_from_degrees(slope_degrees, project_type):
     """Calculate slope score with STRICTER thresholds - SAFE VERSION"""
     try:
         slope = float(slope_degrees)
     except (ValueError, TypeError):
         slope = 15.0

     if project_type.lower() == 'solar':
         if slope > 20:
             return 0
         elif slope > 15:
             return 20
         elif slope > 10:
             return 40
         elif slope > 7:
             return 65
         elif slope > 5:
             return 80
         else:
             return 95
     else:  # Wind
         if slope > 30:
             return 0
         elif slope > 25:
             return 20
         elif slope > 20:
             return 40
         elif slope > 15:
             return 65
         elif slope > 10:
             return 80
         else:
             return 95


def calculate_transmission_score_from_distance(distance, voltage, project_type='solar'):
     """Calculate transmission score - SAFE VERSION"""
     try:
         dist = float(distance)
     except (ValueError, TypeError):
         dist = 2.0

     try:
         volt = float(voltage)
     except (ValueError, TypeError):
         volt = 138.0

     if dist > 2.0:
         return 0
     elif dist > 1.5:
         return 25
     elif dist > 1.0:
         return 45
     elif dist > 0.75:
         return 70
     elif dist > 0.5:
         return 85
     else:
         return 95


def clean_parcel_data(parcel):
     """Clean parcel data - SAFE VERSION"""
     cleaned = dict(parcel)

     numeric_fields = ['elevation', 'latitude', 'longitude', 'acreage', 'acreage_calc', 'mkt_val_land']

     for field in numeric_fields:
         if field in cleaned:
             try:
                 value = cleaned[field]
                 if isinstance(value, str):
                     value = value.replace(',', '').replace('$', '').strip()
                     if value.lower() in ['nan', 'null', 'none', '']:
                         cleaned[field] = 0.0
                     else:
                         cleaned[field] = float(value)
                 elif value is None:
                     cleaned[field] = 0.0
                 else:
                     cleaned[field] = float(value)
             except (ValueError, TypeError):
                 cleaned[field] = 0.0

     string_fields = ['parcel_id', 'owner', 'county_name', 'state_abbr', 'land_use_class']
     for field in string_fields:
         if field in cleaned and cleaned[field] is not None:
             cleaned[field] = str(cleaned[field])
         elif field in cleaned:
             cleaned[field] = ''

     return cleaned


def calculate_enhanced_parcel_scores(enhanced_parcels, analysis_results, project_type):
     """Calculate suitability scores with ENHANCED strict thresholds"""
     try:
         scored_parcels = []

         for i, parcel in enumerate(enhanced_parcels):
             parcel_id = parcel.get('parcel_id', f'PARCEL_{i:06d}')

             # Get the enhanced data
             slope_degrees = parcel.get('slope_degrees', 10)
             transmission_distance = parcel.get('transmission_distance', 2)
             transmission_voltage = parcel.get('transmission_voltage', 138)

             # Calculate scores with new strict thresholds
             slope_score = calculate_slope_score_from_degrees(slope_degrees, project_type)
             transmission_score = calculate_transmission_score_from_distance(
                 transmission_distance, transmission_voltage, project_type
             )

             # STRICT SUITABILITY LOGIC
             # If either slope or transmission is disqualifying (score = 0), overall is unsuitable
             if slope_score == 0 or transmission_score == 0:
                 overall_score = 0
                 is_suitable = False
             else:
                 # Weight transmission more heavily (60% transmission, 40% slope)
                 overall_score = (transmission_score * 0.6) + (slope_score * 0.4)
                 # Higher threshold for suitability (70 instead of 60)
                 is_suitable = overall_score >= 70

             # Create comprehensive suitability analysis
             suitability_analysis = {
                 'is_suitable': is_suitable,
                 'overall_score': round(overall_score, 1),
                 'slope_score': round(slope_score, 1),
                 'transmission_score': round(transmission_score, 1),
                 'slope_suitable': slope_score >= 60,
                 'transmission_suitable': transmission_score >= 60,
                 'slope_degrees': round(slope_degrees, 1),
                 'transmission_distance': round(transmission_distance, 2),
                 'transmission_voltage': int(transmission_voltage),
                 'disqualifying_factors': [
                     f"Slope too steep ({slope_degrees:.1f}°)" if slope_score == 0 else None,
                     f"Transmission too far ({transmission_distance:.2f}mi)" if transmission_score == 0 else None
                 ],
                 'analysis_notes': generate_enhanced_analysis_notes(
                     slope_degrees, transmission_distance, transmission_voltage,
                     slope_score, transmission_score, project_type
                 )
             }

             # Remove None values from disqualifying factors
             suitability_analysis['disqualifying_factors'] = [
                 f for f in suitability_analysis['disqualifying_factors'] if f is not None
             ]

             # Create scored parcel with all data
             scored_parcel = {
                 **parcel,  # Include all enhanced parcel data
                 'parcel_id': parcel_id,
                 'suitability_analysis': suitability_analysis,
                 'is_suitable': is_suitable
             }

             scored_parcels.append(scored_parcel)

         logger.info(f"✅ Enhanced strict scoring completed for {len(scored_parcels)} parcels")
         return scored_parcels

     except Exception as e:
         logger.error(f"Error in enhanced scoring: {e}")
         return enhanced_parcels


def generate_enhanced_analysis_notes(slope_degrees, transmission_distance, transmission_voltage,
                                      slope_score, transmission_score, project_type):
     """Generate detailed analysis notes with recommendations"""
     notes = []

     # Slope analysis
     if project_type.lower() == 'solar':
         if slope_degrees > 20:
             notes.append(f"❌ Slope too steep ({slope_degrees:.1f}°) - not suitable for solar")
         elif slope_degrees > 10:
             notes.append(f"⚠️ Steep slope ({slope_degrees:.1f}°) - requires exceptional other factors")
         elif slope_degrees <= 5:
             notes.append(f"✅ Excellent slope ({slope_degrees:.1f}°)")
         else:
             notes.append(f"✓ Good slope ({slope_degrees:.1f}°)")
     else:  # Wind
         if slope_degrees > 30:
             notes.append(f"❌ Slope too steep ({slope_degrees:.1f}°) - not suitable for wind")
         elif slope_degrees > 20:
             notes.append(f"⚠️ Steep slope ({slope_degrees:.1f}°) - requires exceptional other factors")
         else:
             notes.append(f"✓ Acceptable slope for wind ({slope_degrees:.1f}°)")

     # Transmission analysis
     if transmission_distance > 2.0:
         notes.append(f"❌ Transmission too far ({transmission_distance:.2f}mi) - not economical")
     elif transmission_distance > 1.0:
         notes.append(f"⚠️ Far from transmission ({transmission_distance:.2f}mi) - needs strong other factors")
     elif transmission_distance <= 0.5:
         notes.append(f"✅ Excellent transmission access ({transmission_distance:.2f}mi @ {transmission_voltage}kV)")
     else:
         notes.append(f"✓ Good transmission access ({transmission_distance:.2f}mi @ {transmission_voltage}kV)")

     return "; ".join(notes)

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

@app.route('/api/debug-wind-score', methods=['POST'])
def debug_wind_score():
     """Debug wind score field mapping"""
     try:
         data = request.get_json()
         selected_parcels = data.get('selected_parcels', [])[:1]  # Just test one parcel

         from services.crm_service import CRMService
         crm_service = CRMService()

         for parcel in selected_parcels:
             wind_score = crm_service._calculate_wind_score(parcel, 'wind')
             wind_field_value = crm_service.find_field_value(parcel, 'wind_score')

             return jsonify({
                 'success': True,
                 'parcel_id': parcel.get('parcel_id'),
                 'calculated_wind_score': wind_score,
                 'found_wind_field': wind_field_value,
                 'wind_column_id': 'numeric_mknphdv8',
                 'parcel_keys': list(parcel.keys()),
                 'suitability_data': parcel.get('suitability_analysis', {})
             })

     except Exception as e:
         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug-crm-data', methods=['POST'])
def debug_crm_data():
    """Debug what data would be sent to CRM"""
    try:
        data = request.get_json()
        selected_parcels = data.get('selected_parcels', [])

        if not selected_parcels:
            return jsonify({'error': 'No parcels provided'}), 400

        from services.crm_service import CRMService
        crm_service = CRMService()

        debug_results = []

        for parcel in selected_parcels[:2]:  # Debug first 2 parcels only
            parcel_id = parcel.get('parcel_id', 'Unknown')

            # Process parcel for CRM
            crm_values = crm_service.prepare_parcel_for_crm(parcel, 'solar')

            # Debug field mapping
            debug_info = crm_service.debug_field_mapping(parcel)

            debug_results.append({
                'parcel_id': parcel_id,
                'original_data_keys': list(parcel.keys()),
                'crm_values': crm_values,
                'mapping_success_count': len(crm_values),
                'slope_data': parcel.get('suitability_analysis', {}).get('slope_degrees'),
                'transmission_data': parcel.get('suitability_analysis', {}).get('transmission_distance'),
                'ml_score_data': parcel.get('ml_analysis', {}).get('predicted_score')
            })

        return jsonify({
            'success': True,
            'debug_results': debug_results,
            'total_parcels': len(selected_parcels)
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

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

def safe_json_serialize(obj):
    """Safely serialize complex objects to JSON-compatible format"""
    try:
        if obj is None:
            return None
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(pd, 'isna') and pd.isna(obj):
            return None
        elif isinstance(obj, dict):
            cleaned_dict = {}
            for k, v in obj.items():
                if k is not None:
                    try:
                        cleaned_dict[str(k)] = safe_json_serialize(v)
                    except:
                        cleaned_dict[str(k)] = str(v) if v is not None else None
            return cleaned_dict
        elif isinstance(obj, (list, tuple)):
            cleaned_list = []
            for item in obj:
                try:
                    cleaned_list.append(safe_json_serialize(item))
                except:
                    cleaned_list.append(str(item) if item is not None else None)
            return cleaned_list
        else:
            # Convert to string but keep as valid data type
            return str(obj) if obj is not None else None
    except Exception as e:
        # Return None instead of error string to maintain data structure
        logger.warning(f"Serialization warning for {type(obj)}: {e}")
        return None

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


@app.route('/api/record-parcel-feedback', methods=['POST'])
def record_parcel_feedback():
     """Record individual parcel feedback for ML training"""
     try:
         data = request.get_json()

         parcel_id = data.get('parcel_id')
         selected_reason = data.get('selected_reason')
         custom_reason = data.get('custom_reason', '')
         parcel_data = data.get('parcel_data', {})
         project_type = data.get('project_type', 'solar')
         location = data.get('location', 'Unknown')
         is_recommended = data.get('is_recommended', False)

         if not parcel_id:
             return jsonify({'success': False, 'error': 'parcel_id required'}), 400

         logger.info(f"📝 Recording parcel feedback: {parcel_id} - {selected_reason}")

         # Prepare detailed feedback for BigQuery
         feedback_details = {
             'selected_reason': selected_reason,
             'custom_reason': custom_reason,
             'is_recommended': is_recommended,
             'ml_score': parcel_data.get('ml_analysis', {}).get('predicted_score'),
             'traditional_score': parcel_data.get('suitability_analysis', {}).get('overall_score'),
             'slope_degrees': parcel_data.get('suitability_analysis', {}).get('slope_degrees'),
             'transmission_distance': parcel_data.get('suitability_analysis', {}).get('transmission_distance'),
             'transmission_voltage': parcel_data.get('suitability_analysis', {}).get('transmission_voltage'),
             'acreage': parcel_data.get('acreage_calc') or parcel_data.get('acreage'),
             'land_use_class': parcel_data.get('land_use_class'),
             'owner_type': parcel_data.get('owner'),
             'county': parcel_data.get('county_name') or parcel_data.get('county'),
             'state': parcel_data.get('state_abbr') or parcel_data.get('state')
         }

         # Store in BigQuery using outreach tracker
         from services.outreach_tracker import OutreachTracker
         outreach_tracker = OutreachTracker()

         success = outreach_tracker.track_parcel_feedback(
             parcel_id=parcel_id,
             feedback_reason=selected_reason,
             custom_reason=custom_reason,
             parcel_data=parcel_data,
             feedback_details=feedback_details,
             project_type=project_type,
             location=location
         )

         return jsonify({
             'success': success,
             'message': f'Feedback recorded for parcel {parcel_id}',
             'reason': selected_reason
         })

     except Exception as e:
         logger.error(f"Error recording parcel feedback: {e}")
         return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/record-parcel-exclusion', methods=['POST'])
def record_parcel_exclusion():
     """Record why a parcel was excluded from outreach"""
     try:
         data = request.get_json()

         parcel_id = data.get('parcel_id')
         exclusion_reason = data.get('exclusion_reason')
         is_recommended = data.get('is_recommended', False)
         parcel_data = data.get('parcel_data', {})
         project_type = data.get('project_type', 'solar')
         location = data.get('location', 'Unknown')

         if not parcel_id or not exclusion_reason:
             return jsonify({'success': False, 'error': 'parcel_id and exclusion_reason required'}), 400

         logger.info(f"📝 Recording parcel exclusion: {parcel_id} - {exclusion_reason}")

         # Store in BigQuery
         from services.outreach_tracker import OutreachTracker
         outreach_tracker = OutreachTracker()

         success = outreach_tracker.track_parcel_exclusion(
             parcel_id=parcel_id,
             exclusion_reason=exclusion_reason,
             is_recommended=is_recommended,
             parcel_data=parcel_data,
             project_type=project_type,
             location=location
         )

         return jsonify({
             'success': success,
             'message': f'Exclusion recorded for parcel {parcel_id}',
             'reason': exclusion_reason
         })

     except Exception as e:
         logger.error(f"Error recording parcel exclusion: {e}")
         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/record-batch-feedback', methods=['POST'])
def record_batch_feedback():
     """Record batch feedback about selection patterns"""
     try:
         data = request.get_json()

         feedback_type = data.get('feedback_type')
         reason = data.get('reason')
         feedback_data = data.get('feedback_data', {})
         project_type = data.get('project_type', 'solar')
         location = data.get('location', 'Unknown')

         logger.info(f"📝 Recording batch feedback: {feedback_type} - {reason}")

         # Track this pattern feedback
         from services.outreach_tracker import OutreachTracker
         outreach_tracker = OutreachTracker()

         # Create a pattern feedback record
         success = outreach_tracker.track_pattern_feedback(
             feedback_type=feedback_type,
             reason=reason,
             feedback_data=feedback_data,
             project_type=project_type,
             location=location
         )

         return jsonify({
             'success': success,
             'message': f'Batch feedback recorded: {feedback_type}'
         })

     except Exception as e:
         logger.error(f"Error recording batch feedback: {e}")
         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/record-rejection', methods=['POST'])
def record_rejection():
     """Record parcel rejection reasons for ML feedback"""
     try:
         data = request.get_json()

         parcel_id = data.get('parcel_id')
         rejection_reason = data.get('rejection_reason')
         parcel_data = data.get('parcel_data', {})
         project_type = data.get('project_type', 'solar')
         location = data.get('location', 'Unknown')

         if not parcel_id or not rejection_reason:
             return jsonify({
                 'success': False,
                 'error': 'parcel_id and rejection_reason required'
             }), 400

         logger.info(f"📝 Recording rejection: {parcel_id} - {rejection_reason}")

         # Initialize outreach tracker
         from services.outreach_tracker import OutreachTracker
         outreach_tracker = OutreachTracker()

         # Create a rejection event (modify the track_outreach method to accept rejection)
         success = outreach_tracker.track_rejection(
             parcel=parcel_data,
             rejection_reason=rejection_reason,
             project_type=project_type,
             location=location
         )

         if success:
             return jsonify({
                 'success': True,
                 'message': f'Rejection feedback recorded for {parcel_id}',
                 'reason': rejection_reason
             })
         else:
             return jsonify({
                 'success': False,
                 'error': 'Failed to track rejection'
             }), 500

     except Exception as e:
         logger.error(f"Error recording rejection: {e}")
         return jsonify({
             'success': False,
             'error': str(e)
         }), 500

@app.route('/api/outreach-history', methods=['GET'])
def get_outreach_history():
     """Get recent outreach history for debugging"""
     try:
         from services.outreach_tracker import OutreachTracker
         tracker = OutreachTracker()

         days = request.args.get('days', 7, type=int)
         history = tracker.get_outreach_history(days=days)

         return jsonify({
             'success': True,
             'total_events': len(history),
             'events': history[:20]  # Limit to 20 most recent
         })

     except Exception as e:
         return jsonify({
             'success': False,
             'error': str(e)
         }), 500

@app.route('/api/test-outreach-tracking', methods=['POST'])
def test_outreach_tracking():
     """Test BigQuery outreach tracking"""
     try:
         data = request.get_json()

         # Create test parcel data
         test_parcels = data.get('test_parcels', [
             {
                 'parcel_id': 'TEST_TRACKING_001',
                 'owner': 'Test Owner for Tracking',
                 'acreage_calc': 50.0,
                 'county_name': 'Test County',
                 'state_abbr': 'PA',
                 'ml_analysis': {
                     'predicted_score': 85.0
                 },
                 'suitability_analysis': {
                     'overall_score': 75.0,
                     'traditional_score': 70.0
                 }
             }
         ])

         # Initialize tracker
         from services.outreach_tracker import OutreachTracker
         tracker = OutreachTracker()

         # Test tracking
         success = tracker.track_outreach(
             parcels=test_parcels,
             project_type='solar',
             location='Test Location',
             crm_success=True
         )

         if success:
             # Get recent history to verify
             history = tracker.get_outreach_history(days=1)

             return jsonify({
                 'success': True,
                 'message': f'Successfully tracked {len(test_parcels)} test parcels',
                 'tracking_successful': success,
                 'recent_history_count': len(history),
                 'test_data': test_parcels
             })
         else:
             return jsonify({
                 'success': False,
                 'error': 'Tracking failed'
             }), 500

     except Exception as e:
         import traceback
         return jsonify({
             'success': False,
             'error': f'Tracking test failed: {str(e)}',
             'traceback': traceback.format_exc()
         }), 500

@app.route('/api/export-to-crm', methods=['POST'])
def export_to_crm():
     """Export selected parcels to Monday.com CRM with BigQuery tracking"""
     try:
         data = request.get_json()

         # Validate input
         selected_parcels = data.get('selected_parcels', [])
         project_type = data.get('project_type', 'solar')
         location = data.get('location', 'Unknown Location')

         if not selected_parcels:
             return jsonify({
                 'success': False,
                 'error': 'No parcels selected for export'
             }), 400

         logger.info(f"🚀 Starting CRM export for {len(selected_parcels)} ML-analyzed parcels")

         # Log first parcel structure for debugging
         if selected_parcels:
             sample_parcel = selected_parcels[0]
             logger.info(f"📊 Sample parcel keys: {list(sample_parcel.keys())}")
             logger.info(f"📊 Sample ML analysis: {sample_parcel.get('ml_analysis', 'Missing')}")
             logger.info(f"📊 Sample suitability: {sample_parcel.get('suitability_analysis', 'Missing')}")

         # Initialize services with error isolation
         crm_service = None
         outreach_tracker = None

         try:
             from services.crm_service import CRMService
             crm_service = CRMService()
             logger.info("✅ CRM service initialized")
         except Exception as crm_init_error:
             logger.error(f"❌ CRM service initialization failed: {crm_init_error}")
             return jsonify({
                 'success': False,
                 'error': f'CRM service initialization failed: {str(crm_init_error)}'
             }), 500

         try:
             from services.outreach_tracker import OutreachTracker
             outreach_tracker = OutreachTracker()
             logger.info("✅ Outreach tracker initialized")
         except Exception as tracker_init_error:
             logger.error(f"❌ Outreach tracker initialization failed: {tracker_init_error}")
             # Continue without tracking rather than failing completely
             outreach_tracker = None
             logger.warning("Continuing without outreach tracking...")

         # Test CRM connection
         connection_test = crm_service.test_connection()
         if not connection_test['success']:
             logger.error(f"CRM connection test failed: {connection_test['error']}")
             return jsonify({
                 'success': False,
                 'error': f'CRM connection failed: {connection_test["error"]}'
             }), 500

         logger.info("✅ CRM connection test passed")

         # Export to CRM
         logger.info("📤 Starting CRM export...")
         result = crm_service.export_parcels_to_crm(
             parcels=selected_parcels,
             project_type=project_type,
             location=location
         )

         # Track outreach in BigQuery if tracker is available
         tracking_success = False
         if outreach_tracker:
             try:
                 logger.info("📊 Tracking outreach in BigQuery...")
                 tracking_success = outreach_tracker.track_outreach(
                     parcels=selected_parcels,
                     project_type=project_type,
                     location=location,
                     crm_success=result.get('success', False)
                 )
                 logger.info(f"✅ Outreach tracking: {'success' if tracking_success else 'failed'}")
             except Exception as tracking_error:
                 logger.error(f"❌ Outreach tracking failed: {tracking_error}")
                 tracking_success = False

         if result['success']:
             logger.info(f"🎯 CRM export completed: {result['successful_exports']} parcels exported")

             return jsonify({
                 'success': True,
                 'message': f"Successfully exported {result['successful_exports']} of {result['total_parcels']} parcels",
                 'group_name': result['group_name'],
                 'group_id': result['group_id'],
                 'successful_exports': result['successful_exports'],
                 'failed_exports': result['failed_exports'],
                 'tracking_status': 'tracked' if tracking_success else 'not_tracked',
                 'export_details': result.get('export_details', [])
             })
         else:
             logger.error(f"❌ CRM export failed: {result.get('error', 'Unknown error')}")
             return jsonify({
                 'success': False,
                 'error': result.get('error', 'Export failed'),
                 'tracking_status': 'tracked' if tracking_success else 'not_tracked'
             }), 500

     except Exception as e:
         logger.error(f"❌ CRM export endpoint error: {str(e)}")
         import traceback
         logger.error(f"Traceback: {traceback.format_exc()}")

         return jsonify({
             'success': False,
             'error': f'CRM export failed: {str(e)}',
             'error_type': type(e).__name__
         }), 500

@app.route('/api/crm/test-field-mapping', methods=['POST'])
def test_field_mapping():
    """Test field mapping for a sample parcel without actually creating CRM items"""
    try:
     data = request.get_json()
     test_parcel = data.get('test_parcel', {})

     if not test_parcel:
         return jsonify({
             'success': False,
             'error': 'No test parcel data provided'
         }), 400

     # Initialize CRM service
     crm_service = CRMService()

     # Process fields (this is normally private, but we'll test it)
     crm_values = crm_service._process_parcel_fields(test_parcel)

     return jsonify({
         'success': True,
         'message': f"Field mapping test completed - {len(crm_values)} fields mapped",
         'mapped_fields': crm_values,
         'total_fields_mapped': len(crm_values),
         'input_fields': list(test_parcel.keys()),
         'mapping_summary': {
             field_key: {
                 'variations_checked': config['variations'],
                 'monday_field': config['monday_field'],
                 'found': any(var in test_parcel for var in config['variations'])
             }
             for field_key, config in crm_service.field_mappings.items()
         }
     })

    except Exception as e:
     logger.error(f"Field mapping test error: {str(e)}")
     return jsonify({
         'success': False,
         'error': f'Field mapping test failed: {str(e)}'
     }), 500

@app.route('/api/crm/test-field-mapping', methods=['POST'])
def test_crm_field_mapping():
 """Enhanced CRM field mapping test with real CSV data support"""
 try:
     data = request.get_json()

     # Use provided parcel or create sample from your actual CSV data
     if 'test_parcel' in data:
         test_parcel = data['test_parcel']
     else:
         # Create a test parcel based on your Cuyahoga CSV structure
         test_parcel = {
             'parcel_id': '2936015',
             'owner': 'CLEVELAND CITY',
             'county_id': '39035',
             'county_name': 'Cuyahoga',
             'state_abbr': 'OH',
             'address': '5300 RIVERSIDE Rd',
             'muni_name': 'Cleveland',
             'census_place': 'Cleveland city',
             'census_zip': '44142',
             'mkt_val_land': 0.0,
             'mkt_val_bldg': 0.0,
             'mkt_val_tot': 0.0,
             'land_use_code': '5100.0',
             'land_use_class': 'Residential',
             'mail_address1': 'HOPKINS AIRPORT',
             'mail_address3': 'Cleveland OH 00000',
             'mail_placename': 'HOPKINS AIRPORT CLEVELAND',
             'mail_statename': 'OH',
             'mail_zipcode': 0,
             'acreage': 526.33,
             'acreage_calc': 526.01,
             'acreage_adjacent_with_sameowner': 535.816803739301,
             'latitude': 41.4032357967894,
             'longitude': -81.8607025742243,
             'elevation': 768.73359579745,
             'buildings': 25,
             'last_updated': '2025-Q3',
             'fld_zone': 'A',
             'zoning': 'GI',
             'land_cover': '{"Developed Medium Intensity": 179.91, "Developed High Intensity": 160.81}',
             'crop_cover': '{"Developed/Open Space": 152.45, "Developed/Med Intensity": 141.5}',
             'bldg_sqft': 2106.0,
             'owner_occupied': True,
             'usps_residential': 'Commercial',
             'zone_subty': '0.2 PCT ANNUAL CHANCE FLOOD HAZARD, FLOODWAY',
             # Include suitability analysis data
             'suitability_analysis': {
                 'overall_score': 75.5,
                 'slope_score': 85.0,
                 'transmission_score': 65.0,
                 'slope_degrees': 5.2,
                 'transmission_distance': 2.1,
                 'transmission_voltage': 138000,
                 'is_suitable': True,
                 'slope_suitable': True,
                 'transmission_suitable': True,
                 'analysis_notes': 'Good slope (5.2°); Transmission: 2.10 mi, 138000 kV; ✓ Suitable for development'
             }
         }

     # Initialize CRM service
     try:
         crm_service = CRMService()
     except Exception as init_error:
         return jsonify({
             'success': False,
             'error': f'CRM service initialization failed: {str(init_error)}'
         }), 500

     # Test connection
     connection_test = crm_service.test_connection()
     if not connection_test['success']:
         return jsonify({
             'success': False,
             'error': f'CRM connection failed: {connection_test["error"]}'
         }), 500

     # Debug field mapping
     debug_info = crm_service.debug_field_mapping(test_parcel)

     # Prepare parcel for CRM (this is the actual processing)
     crm_values = crm_service.prepare_parcel_for_crm(test_parcel, 'solar')

     # Create response with comprehensive analysis
     successful_fields = [k for k, v in debug_info['field_analysis'].items() if v['mapping_success']]
     failed_fields = [k for k, v in debug_info['field_analysis'].items() if not v['mapping_success']]

     # Analyze why fields failed
     failed_analysis = {}
     for field_key in failed_fields:
         analysis = debug_info['field_analysis'][field_key]
         if analysis['found_in_fields']:
             failed_analysis[field_key] = f"Found field but invalid value: {repr(analysis['raw_value'])}"
         else:
             failed_analysis[field_key] = f"No matching field found. Checked: {analysis['variations_checked']}"

     return jsonify({
         'success': True,
         'connection_status': 'Connected',
         'user_info': connection_test.get('user', {}),
         'test_results': {
             'total_mapping_fields': len(crm_service.crm_field_mapping) - 1,  # Exclude owner
             'successfully_mapped': len(successful_fields),
             'failed_to_map': len(failed_fields),
             'mapping_success_rate': f"{round((len(successful_fields) / (len(crm_service.crm_field_mapping) - 1)) * 100, 1)}%",
             'total_csv_fields': debug_info['total_fields_available'],
             'crm_ready_values': len(crm_values)
         },
         'successful_mappings': {
             field: {
                 'monday_field': debug_info['field_analysis'][field]['monday_field'],
                 'source_field': debug_info['field_analysis'][field]['found_in_fields'][0] if
                 debug_info['field_analysis'][field]['found_in_fields'] else 'Analysis',
                 'final_value': debug_info['field_analysis'][field]['formatted_value']
             }
             for field in successful_fields
         },
         'failed_mappings': failed_analysis,
         'csv_field_analysis': {
             'available_fields': debug_info['available_fields'],
             'sample_values': {
                 field: test_parcel.get(field, 'N/A')
                 for field in list(debug_info['available_fields'])[:10]  # First 10 fields
             }
         },
         'recommendations': [
             f"Successfully mapped {len(successful_fields)} out of {len(crm_service.crm_field_mapping) - 1} possible fields",
             f"Ready to export {len(crm_values)} fields to Monday.com CRM",
             "Check failed mappings for data quality issues",
             "Consider running CSV data quality fixes if many fields are failing"
         ]
     })

 except Exception as e:
     logger.error(f"CRM field mapping test error: {str(e)}")
     import traceback
     logger.error(f"Traceback: {traceback.format_exc()}")

     return jsonify({
         'success': False,
         'error': f'Field mapping test failed: {str(e)}',
         'error_type': type(e).__name__
     }), 500

@app.route('/api/crm/analyze-csv', methods=['POST'])
def analyze_csv_for_crm():
     """Analyze a CSV file for CRM compatibility"""
     try:
         # This endpoint would analyze an uploaded CSV file
         # For now, we'll simulate with the data from your sample

         # You can extend this to actually process uploaded files
         sample_csv_data = {
             'total_rows': 7,
             'total_columns': 55,
             'sample_parcel': {
                 'parcel_id': '2936015',
                 'owner': 'CLEVELAND CITY',
                 'county_id': '39035',
                 'county_name': 'Cuyahoga',
                 'state_abbr': 'OH',
                 'address': '5300 RIVERSIDE Rd',
                 # ... (include other fields from your CSV)
             }
         }

         # Initialize CRM service
         crm_service = CRMService()

         # Analyze the sample data
         debug_info = crm_service.debug_field_mapping(sample_csv_data['sample_parcel'])

         # Calculate compatibility metrics
         total_required_fields = len(crm_service.crm_field_mapping) - 1
         mappable_fields = debug_info['mappable_fields']
         compatibility_score = (mappable_fields / total_required_fields) * 100

         return jsonify({
             'success': True,
             'csv_analysis': {
                 'total_rows': sample_csv_data['total_rows'],
                 'total_columns': sample_csv_data['total_columns'],
                 'compatibility_score': round(compatibility_score, 1),
                 'mappable_fields': mappable_fields,
                 'total_required_fields': total_required_fields,
                 'ready_for_import': compatibility_score >= 70
             },
             'field_mapping_results': debug_info,
             'recommendations': [
                 f"CSV compatibility: {round(compatibility_score, 1)}%",
                 f"Can map {mappable_fields} out of {total_required_fields} required fields",
                 "Good compatibility" if compatibility_score >= 70 else "Consider data cleaning",
                 "Ready for CRM import" if compatibility_score >= 70 else "Fix data quality issues first"
             ]
         })

     except Exception as e:
         return jsonify({
             'success': False,
             'error': f'CSV analysis failed: {str(e)}'
         }), 500

@app.route('/api/crm/fix-data-quality', methods=['POST'])
def suggest_data_quality_fixes():
     """Suggest fixes for common CSV data quality issues"""
     try:
         data = request.get_json()
         issues = data.get('issues', [])

         fixes = {
             'nan_strings': {
                 'problem': 'CSV contains "nan" as text instead of null values',
                 'solution': 'Replace "nan", "NaN", "null" strings with actual null values',
                 'code': 'df = df.replace(["nan", "NaN", "null"], pd.NA)'
             },
             'zero_coordinates': {
                 'problem': 'Latitude/longitude values are zero',
                 'solution': 'Remove or fix zero coordinate values',
                 'code': 'df.loc[(df["latitude"] == 0) | (df["longitude"] == 0), ["latitude", "longitude"]] = pd.NA'
             },
             'empty_strings': {
                 'problem': 'Empty strings instead of null values',
                 'solution': 'Replace empty strings with null values',
                 'code': 'df = df.replace("", pd.NA)'
             },
             'invalid_zip_codes': {
                 'problem': 'ZIP codes with value 0 or invalid format',
                 'solution': 'Convert zero ZIP codes to proper format or null',
                 'code': 'df.loc[df["zipcode"] == 0, "zipcode"] = "00000"'
             },
             'mixed_data_types': {
                 'problem': 'Numeric fields stored as text',
                 'solution': 'Convert numeric columns to proper data types',
                 'code': 'df["acreage"] = pd.to_numeric(df["acreage"], errors="coerce")'
             }
         }

         return jsonify({
             'success': True,
             'data_quality_fixes': fixes,
             'general_recommendations': [
                 "Run the CSV debug script to identify specific issues",
                 "Use pandas to clean data before CRM import",
                 "Test with a small sample before bulk import",
                 "Backup original CSV before applying fixes"
             ]
         })

     except Exception as e:
         return jsonify({
             'success': False,
             'error': f'Data quality analysis failed: {str(e)}'
         }), 500

if __name__ == '__main__':
    # For Cloud Run, use PORT env variable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)