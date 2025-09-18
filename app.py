import os
import sys
import codecs

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ .env file loaded successfully")

    # Verify critical variables are loaded
    required_vars = ['GOOGLE_APPLICATION_CREDENTIALS', 'FLASK_SECRET_KEY']
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Found")
        else:
            print(f"❌ {var}: Missing")

except ImportError:
    print("❌ python-dotenv not installed. Run: pip install python-dotenv")
except Exception as e:
    print(f"❌ Error loading .env file: {e}")



# STEP 1: Fix encoding first (before any logging)
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        import subprocess

        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach(), errors='replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach(), errors='replace')

# STEP 2: Set up logging BEFORE any other imports
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# STEP 3: Now import everything else
from datetime import datetime
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import traceback
import json
import pandas as pd
import geopandas as gpd

try:
    from google.cloud import storage, bigquery

    logger.info("Google Cloud libraries imported successfully")
except ImportError as e:
    logger.warning(f"Google Cloud libraries not available: {e}")
    storage = None
    bigquery = None

# Import custom modules
from services.census_api import CensusCountyAPI
from bigquery_transmission_storage import TransmissionAnalysisBQ

try:
    import transmission_analysis_bigquery as tx_analysis

    logger.info("Transmission analysis module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import transmission analysis: {e}")
    tx_analysis = None

try:
    import bigquery_slope_analysis as slope_analysis

    logger.info("Slope analysis module imported successfully")
except ImportError:
    try:
        import slope_analysis_bigquery as slope_analysis

        logger.info("Slope analysis module imported as slope_analysis_bigquery")
    except ImportError:
        try:
            import slope_analysis

            logger.info("Slope analysis module imported as slope_analysis")
        except ImportError as e:
            logger.error(f"Could not import any slope analysis module: {e}")
            slope_analysis = None

try:
    import enhanced_parcel_search
    logger.info("Enhanced parcel search module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import enhanced parcel search: {e}")
    enhanced_parcel_search = None

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'Lisboa!2022')

USERS = {
    'demo': generate_password_hash('bcf2025'),
    'admin': generate_password_hash(os.getenv('ADMIN_PASSWORD', 'admin123')),
}


def verify_environment():
    """Verify all required environment variables and services"""
    required_env_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'FLASK_SECRET_KEY',
        'ANTHROPIC_API_KEY'
    ]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")

    # Test BigQuery connectivity
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        client.query("SELECT 1").result()
        logger.info("BigQuery connectivity verified")
    except Exception as e:
        logger.error(f"BigQuery connectivity failed: {e}")

    # Test GCS connectivity
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket('bcfparcelsearchrepository')
        bucket.exists()
        logger.info("GCS connectivity verified")
    except Exception as e:
        logger.error(f"GCS connectivity failed: {e}")


# Call verification on startup
verify_environment()


class BigQueryCountiesManager:
    def __init__(self, api_key=None):
        import os
        print("=== INITIALIZING CLEAN AI SERVICE ===")
        try:
            AI_SERVICE = CleanAIService()
            if AI_SERVICE.client:
                test_result = AI_SERVICE.test_connection()
                print(f"AI service test: {test_result}")
                AI_AVAILABLE = test_result.get('success', False)
                print(f"✅ AI Available: {AI_AVAILABLE}")
            else:
                AI_AVAILABLE = False
                print("❌ AI service created but no client")
        except Exception as e:
            print(f"❌ Clean AI service failed: {e}")
            AI_SERVICE = None
            AI_AVAILABLE = False

    def get_state_counties(self, state_code: str):
        """Get counties for a state from BigQuery"""
        if not self.client:
            return []

        try:
            from google.cloud import bigquery
            query = f"""
            SELECT
                county_name,
                county_fips,
                population,
                population_density,
                rural_indicator,
                population_tier
            FROM `{self.table_ref}`
            WHERE state_code = @state_code
            ORDER BY population DESC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper())
                ]
            )

            results = self.client.query(query, job_config=job_config)
            counties = []

            for row in results:
                counties.append({
                    'name': row.county_name,
                    'fips': row.county_fips,
                    'population': row.population,
                    'population_density': row.population_density,
                    'rural_indicator': row.rural_indicator,
                    'population_tier': row.population_tier,
                    'state_code': state_code.upper()
                })

            logger.info(f"Retrieved {len(counties)} counties for {state_code} from BigQuery")
            return counties

        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            return []

    def counties_exist_for_state(self, state_code: str) -> bool:
        """Check if counties already exist for a state"""
        if not self.client:
            return False

        try:
            from google.cloud import bigquery
            query = f"""
            SELECT COUNT(*) as county_count
            FROM `{self.table_ref}`
            WHERE state_code = @state_code
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper())
                ]
            )

            results = self.client.query(query, job_config=job_config)
            count = list(results)[0].county_count
            return count > 0
        except Exception as e:
            logger.error(f"Error checking counties existence: {e}")
            return False


class RenewableEnergyIntelligence:
    def __init__(self):
        try:
            from google.cloud import bigquery
            self.client = bigquery.Client()
            self.dataset_id = "renewable_energy"
            self.table_id = "county_renewable_intelligence"
            self.table_ref = f"{self.client.project}.{self.dataset_id}.{self.table_id}"
            logger.info("Renewable Energy Intelligence manager initialized")
        except Exception as e:
            logger.warning(f"BigQuery initialization failed: {e}")
            self.client = None

    def get_analysis_freshness(self, state_code: str, project_type: str, max_age_days: int = 30) -> dict:
        """Check how fresh the existing analysis is for a state/project type"""
        if not self.client:
            return {'needs_refresh': True, 'reason': 'BigQuery unavailable'}

        try:
            from google.cloud import bigquery
            query = f"""
            SELECT
                COUNT(*) as total_counties,
                COUNT({project_type.lower()}_last_analyzed) as analyzed_counties,
                MIN({project_type.lower()}_last_analyzed) as oldest_analysis,
                MAX({project_type.lower()}_last_analyzed) as newest_analysis
            FROM `{self.table_ref}`
            WHERE state_code = @state_code
            AND {project_type.lower()}_last_analyzed >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @max_age_days DAY)
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper()),
                    bigquery.ScalarQueryParameter("max_age_days", "INT64", max_age_days)
                ]
            )

            results = self.client.query(query, job_config=job_config)
            row = list(results)[0]

            total_counties = row.total_counties or 0
            analyzed_counties = row.analyzed_counties or 0
            coverage_pct = (analyzed_counties / total_counties * 100) if total_counties > 0 else 0

            needs_refresh = coverage_pct < 80

            return {
                'needs_refresh': needs_refresh,
                'coverage_pct': coverage_pct,
                'total_counties': total_counties,
                'analyzed_counties': analyzed_counties,
                'reason': f'Coverage: {coverage_pct:.1f}%' if not needs_refresh else 'Needs fresh analysis'
            }

        except Exception as e:
            logger.error(f"Freshness check failed: {e}")
            return {'needs_refresh': True, 'reason': f'Error: {str(e)}'}

    def get_county_intelligence(self, state_code: str, county_name: str = None, project_type: str = None):
        """Retrieve existing renewable energy intelligence for counties"""
        return []

    def save_analysis_results(self, state_code: str, project_type: str, analysis_results: dict):
        """Save AI analysis results to the intelligence database"""
        logger.info(
            f"Would save {project_type} analysis for {state_code} with {len(analysis_results.get('county_rankings', []))} counties")
        return True


# Initialize services
try:
    RE_INTELLIGENCE = RenewableEnergyIntelligence()
    logger.info("RE_INTELLIGENCE initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RE_INTELLIGENCE: {e}")
    RE_INTELLIGENCE = None

try:
    CENSUS_API = CensusCountyAPI()
    logger.info("Census API initialized")
except Exception as e:
    CENSUS_API = None
    logger.warning(f"Census API failed: {e}")

try:
    BQ_COUNTIES = BigQueryCountiesManager()
    BQ_AVAILABLE = BQ_COUNTIES.client is not None
    logger.info(f"BigQuery available: {BQ_AVAILABLE}")
except Exception as e:
    BQ_COUNTIES = None
    BQ_AVAILABLE = False
    logger.warning(f"BigQuery not available: {e}")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)

    return decorated_function


def get_gcs_client():
    """Get Google Cloud Storage client with proper error handling"""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        try:
            bucket = client.bucket(bucket_name)
            bucket.reload()
            logger.info(f"GCS client initialized successfully for bucket: {bucket_name}")
            return client
        except Exception as bucket_error:
            logger.warning(f"GCS bucket access failed: {bucket_error}")
            logger.info("Continuing without cloud storage - some features may be limited")
            return None
    except ImportError:
        logger.warning("Google Cloud Storage not available - install google-cloud-storage")
        return None
    except Exception as e:
        logger.error(f"GCS client initialization failed: {e}")
        return None


def download_from_gcs(gcs_path):
    """Download file from GCS with proper error handling"""
    try:
        if not gcs_path.startswith("gs://"):
            logger.error(f"Invalid GCS path: {gcs_path}")
            return None

        parts = gcs_path[5:].split("/", 1)
        if len(parts) < 2:
            logger.error(f"Invalid GCS path format: {gcs_path}")
            return None

        bucket_name = parts[0]
        blob_path = parts[1]

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.error(f"File not found in GCS: {gcs_path}")
            return None

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp:
            blob.download_to_filename(tmp.name)
            logger.info(f"Downloaded {gcs_path} to {tmp.name}")
            return tmp.name

    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        return None


def load_counties_from_file(state_code):
    """Load counties from the counties file"""
    try:
        import os
        import json

        counties_file = os.path.join(os.path.dirname(__file__), 'counties-trimmed.json')

        if not os.path.exists(counties_file):
            logger.error(f"Counties file not found: {counties_file}")
            return []

        with open(counties_file, 'r') as f:
            all_counties = json.load(f)

        state_counties = []
        for county in all_counties:
            if county.get('state') == state_code.upper():
                state_counties.append({
                    'name': county.get('county', '').replace(' County', ''),
                    'fips': county.get('fips', ''),
                    'full_name': county.get('county', '')
                })

        logger.info(f"Loaded {len(state_counties)} counties for {state_code}")
        return state_counties

    except Exception as e:
        logger.error(f"Error loading counties: {e}")
        return []


# Routes
@app.route('/')
def index():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login_page'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if username in USERS and check_password_hash(USERS[username], password):
            session['logged_in'] = True
            session['username'] = username
            session['login_time'] = datetime.now().isoformat()
            return jsonify({'success': True, 'message': f'Welcome {username}'})
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'POST credentials to login', 'demo': 'demo/bcf2025'})


@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})


@app.route('/login-page')
def login_page():
    try:
        logger.info("Serving login page")
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error serving login page: {e}")
        return f"Error loading login page: {str(e)}", 500


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'templates_folder': app.template_folder,
        'static_folder': app.static_folder
    })


@app.route('/api/test-simple', methods=['POST'])
def test_simple():
    try:
        logger.info("Simple test started")
        return jsonify({"success": True, "message": "Simple test works"})
    except Exception as e:
        logger.error(f"Simple test failed: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/analyze-existing-file-bq', methods=['POST'])
@login_required
def analyze_existing_file_with_bigquery():
    """Run transmission analysis and store results in BigQuery"""
    logger.info("=== FUNCTION STARTED ===")

    local_file = None  # Initialize variable at function start

    try:
        logger.info("Parsing request data...")
        data = request.get_json()
        logger.info(f"Request data: {data}")

        # Extract parameters BEFORE validating them
        file_path = data.get('file_path')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type')

        # Validate required parameters
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'file_path is required'
            }), 400

        # Configuration
        project_id = 'bcfparcelsearchrepository'
        output_bucket = 'bcfparcelsearchrepository'

        logger.info(f"Starting BigQuery-based analysis for {file_path}")

        # Check if transmission analysis module is available
        if tx_analysis is None:
            return jsonify({
                'success': False,
                'error': 'Transmission analysis module not available'
            }), 500

        # Run transmission analysis
        try:
            logger.info("Running transmission analysis...")
            transmission_result = tx_analysis.run_headless(
                input_file_path=file_path,
                buffer_distance_miles=3.0,
                output_bucket=output_bucket,
                project_id=project_id
            )

            logger.info(f"Transmission analysis result: {transmission_result.get('status')}")

        except Exception as e:
            logger.error(f"Transmission analysis error: {str(e)}")
            logger.error(f"Transmission analysis traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': f'Transmission analysis failed: {str(e)}'
            }), 500

        # Check transmission analysis results
        if transmission_result.get('status') != 'success':
            error_msg = transmission_result.get('message', 'Unknown transmission analysis error')
            return jsonify({
                'success': False,
                'error': f"Transmission analysis failed: {error_msg}"
            }), 500

        logger.info(f"Transmission analysis completed: {transmission_result.get('output_file_path')}")

        # Download and load enhanced results
        try:
            logger.info("Downloading enhanced results file...")
            local_file = download_from_gcs(transmission_result.get('output_file_path'))
            if not local_file:
                return jsonify({
                    'success': False,
                    'error': 'Failed to download enhanced file from GCS'
                }), 500

            # Load enhanced parcels
            logger.info("Loading enhanced parcels data...")
            enhanced_parcels_gdf = gpd.read_file(local_file)
            logger.info(f"Loaded {len(enhanced_parcels_gdf)} enhanced parcels")

            # ADD SLOPE ANALYSIS HERE
            if slope_analysis is not None:
                logger.info("Running slope analysis on transmission results...")
                try:
                    slope_result = slope_analysis.run_headless(
                        input_file_path=transmission_result.get('output_file_path'),
                        max_slope_degrees=25.0,  # Reasonable limit for solar
                        output_bucket='bcfparcelsearchrepository',
                        project_id='bcfparcelsearchrepository'
                    )

                    if slope_result.get('status') == 'success':
                        logger.info("Slope analysis completed successfully")
                        # Download the slope-enhanced file
                        slope_local_file = download_from_gcs(slope_result.get('output_file_path'))
                        if slope_local_file:
                            enhanced_parcels_gdf = gpd.read_file(slope_local_file)
                            logger.info(f"Loaded slope-enhanced data with {len(enhanced_parcels_gdf)} parcels")
                            # Cleanup slope temp file
                            if os.path.exists(slope_local_file):
                                os.unlink(slope_local_file)
                        else:
                            logger.warning("Could not download slope-enhanced file, continuing with transmission data only")
                    else:
                        logger.warning(f"Slope analysis failed: {slope_result.get('message', 'Unknown error')}")
                except Exception as slope_error:
                    logger.warning(f"Slope analysis error: {slope_error} - continuing without slope data")
            else:
                logger.warning("Slope analysis module not available")

        except Exception as e:
            logger.error(f"Error loading enhanced file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load enhanced file: {str(e)}'
            }), 500

        # Clean numeric data for BigQuery compatibility
        try:
            logger.info("Cleaning numeric data types for BigQuery compatibility...")

            def safe_numeric_convert(df, column_name, default_value):
                """Safely convert column to numeric"""
                if column_name in df.columns:
                    original_type = str(df[column_name].dtype)
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(default_value)
                    logger.info(f"Converted {column_name} from {original_type} to numeric")
                return df

            # Convert all potentially problematic columns
            numeric_columns = [
                ('acreage', 0.0),
                ('acreage_calc', 0.0),
                ('avg_slope_degrees', 15.0),
                ('avg_slope', 15.0),
                ('tx_nearest_distance', 999.0),
                ('tx_max_voltage', 0.0),
                ('tx_distance_miles', 999.0),
                ('tx_voltage_kv', 0.0),
                ('tx_lines_count', 0),
                ('tx_lines_intersecting', 0)
            ]

            for col_name, default_val in numeric_columns:
                enhanced_parcels_gdf = safe_numeric_convert(enhanced_parcels_gdf, col_name, default_val)

            logger.info("Data type cleaning completed successfully")

        except Exception as e:
            logger.error(f"Error cleaning numeric data: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Data cleaning failed: {str(e)}'
            }), 500

        # Store results in BigQuery
        try:
            logger.info("Storing results in BigQuery...")
            from bigquery_transmission_storage import TransmissionAnalysisBQ

            bq_storage = TransmissionAnalysisBQ()

            analysis_metadata = {
                'state': state or 'Unknown',
                'county': county_name or 'Unknown',
                'source_file': file_path,
                'project_type': project_type or 'solar'
            }

            analysis_id = bq_storage.store_transmission_analysis(enhanced_parcels_gdf, analysis_metadata)
            bq_storage.debug_stored_data(analysis_id)
            logger.info(f"Results stored with analysis_id: {analysis_id}")

        except Exception as e:
            logger.error(f"BigQuery storage error: {str(e)}")
            logger.error(f"BigQuery storage traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': f'BigQuery storage failed: {str(e)}'
            }), 500

        # Retrieve clean results for UI
        try:
            logger.info("Retrieving clean results for UI...")
            ui_results = bq_storage.get_analysis_results(analysis_id)

            if not ui_results.get('success'):
                return jsonify({
                    'success': False,
                    'error': f"Failed to retrieve results: {ui_results.get('error')}"
                }), 500

            total_parcels = ui_results.get('summary', {}).get('total_parcels', 0)
            logger.info(f"UI results prepared: {total_parcels} parcels")

            return jsonify({
                'success': True,
                'message': 'Analysis completed successfully via BigQuery',
                'analysis_id': analysis_id,
                'parcel_count': total_parcels,
                'analysis_results': ui_results,
                'transmission_analysis': transmission_result,
                'storage_method': 'bigquery'
            })

        except Exception as e:
            logger.error(f"Error retrieving UI results: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to retrieve UI results: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"BigQuery analysis failed with unexpected error: {str(e)}")
        logger.error(f"Unexpected error traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'details': traceback.format_exc(),
            'storage_method': 'bigquery'
        }), 500

    finally:
        # FIXED: Cleanup downloaded file with proper variable checking
        try:
            if local_file is not None and os.path.exists(local_file):
                os.unlink(local_file)
                logger.info("Cleaned up temporary file")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file: {e}")


@app.route('/api/transmission-test-public', methods=['POST'])
def transmission_test_public():
    """Public transmission analysis test (no auth required)"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path:
            return jsonify({'error': 'file_path required'}), 400

        logger.info(f"Testing transmission analysis for: {file_path}")

        # Run the working transmission analysis
        result = tx_analysis.run_headless(
            input_file_path=file_path,
            buffer_distance_miles=2.0,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        return jsonify({
            'success': result['status'] == 'success',
            'message': result.get('message', ''),
            'output_file': result.get('output_file_path', ''),
            'parcels_processed': result.get('parcels_processed', 0),
            'parcels_near_transmission': result.get('parcels_near_transmission', 0),
            'transmission_working': True
        })

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze-state-counties', methods=['POST'])
@login_required
def analyze_state_counties():
    try:
        data = request.get_json()
        state = data.get('state')
        project_type = data.get('project_type')

        if not state or not project_type:
            return jsonify({'success': False, 'error': 'Missing state or project_type'}), 400

        logger.info(f"Starting analysis: {state} for {project_type}")

        try:
            # Load counties from file
            counties = load_counties_from_file(state)

            if not counties:
                return jsonify({'success': False, 'error': f'No counties found for {state}'}), 500

            # Create simple deterministic analysis
            analyzed_counties = []
            for i, county in enumerate(counties):
                # Simple scoring based on position and name characteristics
                base_score = 75 - (i * 0.5)  # Gradually declining scores

                county_name_lower = county['name'].lower()

                # Adjust score based on county name characteristics
                if any(word in county_name_lower for word in ['center', 'central']):
                    base_score += 5
                if any(word in county_name_lower for word in ['mountain', 'highland']):
                    base_score -= 10 if project_type == 'solar' else 5
                if any(word in county_name_lower for word in ['wake', 'mecklenburg', 'guilford']):
                    base_score += 3  # Major counties

                analyzed_counties.append({
                    'name': county['name'],
                    'fips': county.get('fips', ''),
                    'score': round(max(35, min(95, base_score)), 1),
                    'rank': i + 1,
                    'strengths': [f'{project_type.title()} resource potential', 'Infrastructure access'],
                    'challenges': ['Requires site-specific analysis'],
                    'resource_quality': 'Good',
                    'policy_environment': 'Supportive',
                    'development_potential': 'Medium' if base_score >= 65 else 'Limited'
                })

            # Sort by score
            analyzed_counties.sort(key=lambda x: x['score'], reverse=True)

            # Update ranks
            for i, county in enumerate(analyzed_counties):
                county['rank'] = i + 1

            logger.info(f"Simple analysis completed: {len(analyzed_counties)} counties")

            return jsonify({
                'success': True,
                'analysis': {
                    'state': state,
                    'project_type': project_type,
                    'counties': analyzed_counties,
                    'total_counties': len(analyzed_counties),
                    'analysis_summary': f'County analysis for {project_type} development in {state}',
                    'state_renewable_outlook': f'{state} shows varied {project_type} potential across counties',
                    'comprehensive_analysis': True,
                    'ai_powered': False,
                    'cached_data': False,
                    'data_sources': ['County Database', 'Deterministic Analysis'],
                    'analysis_quality': 'Deterministic county ranking',
                    'coverage': f"Analysis for {len(analyzed_counties)} counties"
                }
            })

        except Exception as analysis_error:
            logger.error(f"Simple analysis failed: {analysis_error}")
            return jsonify({'success': False, 'error': f'Analysis failed: {str(analysis_error)}'}), 500

    except Exception as e:
        logger.error(f"Analysis route error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'}), 500


AI_SERVICE = None
ENHANCED_PARCEL_AI = None

print("=== STARTING CLEAN AI SERVICE INITIALIZATION ===")


class WorkingAIService:
    def __init__(self):
        import os

        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None

        print(f"API key check: exists={bool(self.api_key)}, length={len(self.api_key) if self.api_key else 0}")

        if not self.api_key or not self.api_key.startswith('sk-ant-'):
            print("ERROR: Invalid or missing API key")
            return

        # Clear proxy variables temporarily
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        original_proxies = {}
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("SUCCESS: Anthropic client created")

            # Test immediately
            test_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            print("SUCCESS: AI connection test passed")

        except Exception as e:
            print(f"ERROR: AI client creation failed: {e}")
            self.client = None

        # Restore proxy variables
        for var, value in original_proxies.items():
            os.environ[var] = value

    def test_connection(self):
        if not self.client:
            return {"success": False, "error": "Client not initialized"}

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return {"success": True, "message": "Connection working"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_state_counties(self, state, project_type, counties):
        if not self.client:
            logger.error("No AI client - using fallback")
            return self._create_fallback(state, project_type, counties)

        try:
            county_names = [c.get('name', f'County_{i}') for i, c in enumerate(counties)]

            prompt = f"""Analyze {state} counties for {project_type} energy development.

Counties: {', '.join(county_names)}

Return this exact JSON structure:
{{
    "analysis_summary": "Market overview for {state} {project_type} development",
    "county_rankings": [
        {{
            "name": "CountyName",
            "score": 85,
            "rank": 1,
            "strengths": ["Good resource", "Strong infrastructure"],
            "challenges": ["Zoning restrictions"],
            "resource_quality": "Excellent",
            "policy_environment": "Supportive", 
            "development_activity": "High",
            "summary": "Strong development potential"
        }}
    ]
}}"""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            import re

            response_text = response.content[0].text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                analysis_data = json.loads(json_match.group())

                # Add FIPS codes
                for ranking in analysis_data.get('county_rankings', []):
                    county_name = ranking.get('name', '')
                    for county in counties:
                        if county.get('name', '').lower() in county_name.lower():
                            ranking['fips'] = county.get('fips', f'{state}999')
                            break

                print(f"SUCCESS: AI analyzed {len(analysis_data.get('county_rankings', []))} counties")
                return analysis_data

        except Exception as e:
            print(f"ERROR: AI analysis failed: {e}")

        return self._create_fallback(state, project_type, counties)

    def _create_fallback(self, state, project_type, counties):
        print("WARNING: Using fallback analysis")

        analyzed_counties = []
        for i, county in enumerate(counties):
            analyzed_counties.append({
                'name': county.get('name', f'County_{i}'),
                'fips': county.get('fips', ''),
                'score': 75 - (i * 2),
                'rank': i + 1,
                'strengths': [f'{project_type.title()} potential', 'Infrastructure access'],
                'challenges': ['Requires detailed analysis'],
                'resource_quality': 'Good',
                'policy_environment': 'Supportive',
                'development_activity': 'Medium',
                'summary': f'Standard {project_type} development potential'
            })

        return {
            'analysis_summary': f'{state} {project_type} development analysis (FALLBACK)',
            'county_rankings': analyzed_counties
        }


# Initialize the working AI service
try:
    AI_SERVICE = WorkingAIService()
    AI_AVAILABLE = AI_SERVICE.client is not None
    print(f"FINAL RESULT: AI_AVAILABLE = {AI_AVAILABLE}")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    AI_SERVICE = None
    AI_AVAILABLE = False

print("=== AI SERVICE INITIALIZATION COMPLETE ===")

# Set availability flag
AI_AVAILABLE = AI_SERVICE is not None and hasattr(AI_SERVICE, 'client') and AI_SERVICE.client is not None
print(f"=== AI INITIALIZATION COMPLETE - Available: {AI_AVAILABLE} ===")
logger.info(f"=== AI INITIALIZATION COMPLETE - Available: {AI_AVAILABLE} ===")

try:
    from services.enhanced_parcel_ai_service import EnhancedParcelAIService
    ENHANCED_PARCEL_AI = EnhancedParcelAIService()
    logger.info("Enhanced Parcel AI Service initialized successfully")
except ImportError as e:
    logger.error(f"Failed to import Enhanced Parcel AI Service: {e}")
    ENHANCED_PARCEL_AI = None
except Exception as e:
    logger.error(f"Failed to initialize Enhanced Parcel AI Service: {e}")
    ENHANCED_PARCEL_AI = None

@app.route('/api/get-existing-files', methods=['POST'])
def get_existing_files():
    try:
        data = request.get_json()
        state = data.get('state')
        county = data.get('county')
        county_fips = data.get('county_fips')

        logger.info(f"Getting existing files for {county}, {state}")

        # Try to get GCS client
        client = get_gcs_client()
        if not client:
            logger.warning("GCS not available - returning empty file list")
            return jsonify({
                'success': True,
                'files': [],
                'folder_path': f"Cloud storage not available",
                'message': 'Cloud storage not configured - no existing files found'
            })

        # Use the correct bucket name from environment
        bucket_name = os.getenv('CACHE_BUCKET_NAME') or os.getenv('BUCKET_NAME', 'bcfparcelsearchrepository')

        logger.info(f"Searching bucket: {bucket_name}")
        logger.info(f"Looking for: {state}/{county}/")

        try:
            bucket = client.bucket(bucket_name)

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
                        'parcel_count': estimate_parcel_count_from_filename(blob.name),
                        'search_criteria': extract_search_criteria_from_filename(blob.name)
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

        except Exception as bucket_error:
            logger.error(f"Bucket access error: {bucket_error}")
            return jsonify({
                'success': True,
                'files': [],
                'folder_path': f"gs://{bucket_name}/",
                'message': f'Bucket access failed: {str(bucket_error)}'
            })

    except Exception as e:
        logger.error(f"Error getting existing files: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get existing files: {str(e)}'
        })


@app.route('/api/download-file')
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

        # Get GCS client
        client = get_gcs_client()
        if not client:
            return jsonify({'error': 'Cloud storage not available'}), 500

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return jsonify({'error': 'File not found'}), 404

        # Generate signed URL for download
        from datetime import timedelta
        download_url = blob.generate_signed_url(
            expiration=timedelta(hours=1),
            method='GET'
        )

        return redirect(download_url)

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/api/preview-file', methods=['POST'])
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

        # Get GCS client
        client = get_gcs_client()
        if not client:
            return jsonify({'error': 'Cloud storage not available'}), 500

        bucket = client.bucket(bucket_name)
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


@app.route('/api/check-county-activity/<state>', methods=['GET'])
@login_required
def check_county_activity(state):
    """Check which counties have existing folders in Cloud Storage"""
    try:
        logger.info(f"Checking county activity for {state}")

        client = get_gcs_client()
        if not client:
            return jsonify({
                'success': True,
                'county_activity': {},
                'total_counties': 0,
                'active_counties': 0,
                'message': 'Cloud storage not available'
            })

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = client.bucket(bucket_name)

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


@app.route('/api/debug-ai-status', methods=['GET'])
def debug_ai_status():
    """Debug AI service status in production"""
    try:
        import os
        import traceback

        # Check environment variable
        api_key = os.getenv('ANTHROPIC_API_KEY')

        debug_info = {
            'anthropic_key_exists': bool(api_key),
            'anthropic_key_length': len(api_key) if api_key else 0,
            'anthropic_key_format_valid': api_key.startswith('sk-ant-') if api_key else False,
            'anthropic_key_preview': api_key[:10] + '...' if api_key else None,
            'ai_service_initialized': AI_SERVICE is not None,
            'ai_client_exists': hasattr(AI_SERVICE,
                                        'client') and AI_SERVICE.client is not None if AI_SERVICE else False,
            'environment': os.getenv('GAE_ENV', 'cloud_run'),
            'working_directory': os.getcwd(),
        }

        # Test connection if available
        if AI_SERVICE and hasattr(AI_SERVICE, 'client') and AI_SERVICE.client:
            try:
                test_result = AI_SERVICE.test_connection()
                debug_info['connection_test'] = test_result
            except Exception as e:
                debug_info['connection_test'] = {'success': False, 'error': str(e)}
        elif AI_SERVICE:
            debug_info['ai_service_error'] = 'AI service exists but client is None'
        else:
            debug_info['ai_service_error'] = 'AI_SERVICE is None'

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/debug-gcs')
def debug_gcs():
    """Debug GCS connectivity"""
    try:
        client = get_gcs_client()
        if not client:
            return jsonify({
                'gcs_available': False,
                'error': 'Could not initialize GCS client'
            })

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = client.bucket(bucket_name)

        # List some files to test access
        blobs = list(bucket.list_blobs(max_results=5))

        return jsonify({
            'gcs_available': True,
            'bucket_name': bucket_name,
            'sample_files': [blob.name for blob in blobs],
            'total_sample_files': len(blobs)
        })

    except Exception as e:
        return jsonify({
            'gcs_available': False,
            'error': str(e)
        })


@app.route('/api/debug-gcs-simple')
def debug_gcs_simple():
    """Simple GCS bucket diagnostic"""
    try:
        client = get_gcs_client()
        if not client:
            return jsonify({'error': 'GCS client not available'})

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = client.bucket(bucket_name)

        # Get just the first 20 files/folders
        blobs = list(bucket.list_blobs(max_results=20))

        return jsonify({
            'success': True,
            'bucket_name': bucket_name,
            'total_files_sample': len(blobs),
            'sample_paths': [blob.name for blob in blobs],
            'has_nc_files': any('NC' in blob.name.upper() for blob in blobs),
            'has_state_folders': any('/' in blob.name for blob in blobs)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def estimate_parcel_count_from_filename(filename):
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


def extract_search_criteria_from_filename(filename):
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


# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500


@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found', 'url': request.url}), 404


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/api/preview-parcel-search', methods=['POST'])
@login_required
def parcel_search_preview():
    """Preview parcel search results"""
    try:
        if not enhanced_parcel_search:
            return jsonify({
                'success': False,
                'error': 'Parcel search module not available'
            }), 500

        data = request.get_json()
        logger.info(f"Parcel search preview request: {data}")

        # Call the preview function
        result = enhanced_parcel_search.preview_search_count(**data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Parcel search preview error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Preview failed: {str(e)}'
        }), 500


# Add these routes to your app.py (after your existing parcel search routes)

@app.route('/api/preview-parcel-search', methods=['POST'])
@login_required
def preview_parcel_search():
    """Frontend-compatible preview route"""
    try:
        if not enhanced_parcel_search:
            return jsonify({
                'success': False,
                'message': 'Parcel search module not available'
            }), 500

        data = request.get_json()

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

        logger.info(f"Preview search request: {search_params}")

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


@app.route('/api/execute-parcel-search', methods=['POST'])
@login_required
def execute_parcel_search():
    """Frontend-compatible execute route"""
    try:
        if not enhanced_parcel_search:
            return jsonify({
                'success': False,
                'message': 'Parcel search module not available'
            }), 500

        data = request.get_json()

        # Map all search parameters
        search_params = {
            'county_id': data.get('county_id'),
            'calc_acreage_min': data.get('calc_acreage_min'),
            'calc_acreage_max': data.get('calc_acreage_max'),
            'owner': data.get('owner'),
            'parcel_id': data.get('parcel_id'),
            'user_id': data.get('user_id', session.get('username', 'default_user')),
            'project_type': data.get('project_type', 'solar'),
            'county_name': data.get('county_name', 'Unknown'),
            'state': data.get('state', 'XX')
        }

        # Remove None values but keep empty strings
        search_params = {k: v for k, v in search_params.items() if v is not None}

        logger.info(f"Execute parcel search request: {search_params}")

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


@app.route('/api/download-parcel-file', methods=['GET'])
@login_required
def download_parcel_file():
    """Download parcel files from Google Cloud Storage"""
    try:
        blob_name = request.args.get('blob_name')
        file_type = request.args.get('file_type', 'csv')

        if not blob_name:
            return jsonify({'error': 'Missing blob_name parameter'}), 400

        # Use existing get_gcs_client function
        client = get_gcs_client()
        if not client:
            return jsonify({'error': 'Cloud storage not available'}), 500

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return jsonify({'error': 'File not found'}), 404

        # Generate signed URL for download (reuse existing pattern)
        from datetime import timedelta
        download_url = blob.generate_signed_url(
            expiration=timedelta(hours=1),
            method='GET'
        )

        return redirect(download_url)

    except Exception as e:
        logger.error(f"Download file error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/api/list-user-searches', methods=['GET'])
@login_required
def list_user_searches():
    """List previous parcel searches for a user"""
    try:
        user_id = request.args.get('user_id', session.get('username', 'default_user'))
        county_name = request.args.get('county_name')
        state = request.args.get('state')

        client = get_gcs_client()
        if not client:
            return jsonify({
                'success': True,
                'files': [],
                'total_files': 0,
                'message': 'Cloud storage not available'
            })

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = client.bucket(bucket_name)

        # Build search prefix
        if state and county_name:
            prefix = f"{state}/{county_name}/Parcel_Files/"
        else:
            prefix = ""

        blobs = bucket.list_blobs(prefix=prefix)

        files = []
        for blob in blobs:
            if blob.name.endswith('.csv'):
                file_info = {
                    'name': blob.name.split('/')[-1],
                    'blob_name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'file_type': 'csv'
                }
                files.append(file_info)

        return jsonify({
            'success': True,
            'files': files,
            'total_files': len(files)
        })

    except Exception as e:
        logger.error(f"List user searches error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to list searches: {str(e)}'
        }), 500


@app.route('/api/delete-search-file', methods=['DELETE'])
@login_required
def delete_search_file():
    """Delete a parcel search file from GCS"""
    try:
        data = request.get_json()
        blob_name = data.get('blob_name')

        if not blob_name:
            return jsonify({'error': 'Missing blob_name'}), 400

        client = get_gcs_client()
        if not client:
            return jsonify({'error': 'Cloud storage not available'}), 500

        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return jsonify({'error': 'File not found'}), 404

        # Delete the blob
        blob.delete()

        # Also try to delete corresponding GPKG file
        if blob_name.endswith('.csv'):
            gpkg_blob_name = blob_name.replace('.csv', '.gpkg')
            gpkg_blob = bucket.blob(gpkg_blob_name)
            if gpkg_blob.exists():
                gpkg_blob.delete()

        return jsonify({
            'success': True,
            'message': 'File deleted successfully'
        })

    except Exception as e:
        logger.error(f"Delete search file error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Delete failed: {str(e)}'
        }), 500


@app.route('/api/analyze-search-results', methods=['POST'])
@login_required
def analyze_search_results():
    """Analyze parcel search results with AI (enhanced analysis)"""
    try:
        data = request.get_json()
        search_id = data.get('search_id')
        parcel_data = data.get('parcel_data', [])
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type')

        if not parcel_data:
            return jsonify({
                'success': False,
                'error': 'No parcel data provided for analysis'
            }), 400

        logger.info(f"Analyzing search results: {len(parcel_data)} parcels")

        # For now, create a simple analysis structure
        # This is where you'd integrate with your existing analysis code
        analysis_results = create_simple_parcel_analysis(parcel_data, county_name, state, project_type)

        return jsonify({
            'success': True,
            'analysis_results': analysis_results,
            'message': f'Analysis completed for {len(parcel_data)} parcels'
        })

    except Exception as e:
        logger.error(f"Analyze search results error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


# Replace the analyze_existing_search_file function I provided with this simpler version:

@app.route('/api/analyze-existing-search-file', methods=['POST'])
@login_required
def analyze_existing_search_file():
    """Analyze an existing search file from GCS using your existing BigQuery analysis"""
    try:
        data = request.get_json()
        blob_name = data.get('blob_name')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type')

        if not blob_name:
            return jsonify({
                'success': False,
                'error': 'blob_name is required'
            }), 400

        # Construct GCS path
        bucket_name = os.getenv('CACHE_BUCKET_NAME', 'bcfparcelsearchrepository')
        gcs_path = f"gs://{bucket_name}/{blob_name}"

        logger.info(f"Analyzing existing file: {gcs_path}")

        # Create a new request context to call your existing function
        from flask import Flask
        with app.test_request_context(
                '/api/analyze-existing-file-bq',
                method='POST',
                json={
                    'file_path': gcs_path,
                    'county_name': county_name,
                    'state': state,
                    'project_type': project_type
                }
        ):
            # Call your existing function directly
            return analyze_existing_file_with_bigquery()

    except Exception as e:
        logger.error(f"Analyze existing search file error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/api/analyze-existing-file-comprehensive', methods=['POST'])
def analyze_existing_file_comprehensive():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        county_name = data.get('county_name', 'Unknown')
        state = data.get('state', 'Unknown')
        project_type = data.get('project_type', 'solar')

        logger.info(f"Starting comprehensive analysis for {file_path}")

        # STEP 1: Run transmission analysis first
        logger.info("Running transmission analysis...")
        try:
            # Import the module and call the correct function
            from transmission_analysis_bigquery import run_headless as transmission_run_headless

            transmission_result = transmission_run_headless(
                input_file_path=file_path,
                buffer_distance_miles=data.get('buffer_distance_miles', 2.0),
                output_bucket='bcfparcelsearchrepository',
                project_id='bcfparcelsearchrepository'
            )

            logger.info(f"Transmission analysis result: {transmission_result['status']}")

            if transmission_result['status'] == 'success':
                transmission_enhanced_file = transmission_result['output_file_path']
                logger.info(f"Transmission analysis successful: {transmission_enhanced_file}")
            else:
                logger.warning(f"Transmission analysis failed: {transmission_result.get('message', 'Unknown error')}")
                transmission_enhanced_file = file_path  # Use original file

        except Exception as e:
            logger.error(f"Transmission analysis failed: {str(e)}")
            transmission_enhanced_file = file_path  # Use original file
            transmission_result = {'status': 'error', 'message': str(e)}

        # STEP 2: Run slope analysis on the transmission-enhanced data
        logger.info("Running slope analysis...")
        try:
            # Import the module and call the correct function
            from bigquery_slope_analysis import run_headless as slope_run_headless

            slope_result = slope_run_headless(
                input_file_path=transmission_enhanced_file,
                max_slope_degrees=data.get('max_slope_degrees', 25.0),
                output_bucket='bcfparcelsearchrepository',
                project_id='bcfparcelsearchrepository'
            )

            logger.info(f"Slope analysis result: {slope_result['status']}")

            if slope_result['status'] != 'success':
                return jsonify({
                    'status': 'error',
                    'message': f"Slope analysis failed: {slope_result.get('message', 'Unknown error')}"
                }), 500

        except Exception as e:
            logger.error(f"Slope analysis failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f"Slope analysis failed: {str(e)}"
            }), 500

        # STEP 3: Load the final results and format for frontend
        logger.info("Processing results for frontend...")

        try:
            # Load the final result file
            final_results_gdf = load_analysis_results(slope_result['output_file_path'])

            if final_results_gdf is None or len(final_results_gdf) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'No results found in output file'
                }), 500

            # Convert to format expected by frontend
            analysis_results = format_results_for_frontend(
                final_results_gdf,
                county_name,
                state,
                transmission_result,
                slope_result
            )

            return jsonify({
                'status': 'success',
                'analysis_results': analysis_results,
                'parcel_count': len(final_results_gdf),
                'transmission_analysis': transmission_result,
                'slope_analysis': slope_result
            })

        except Exception as e:
            logger.error(f"Results processing failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f"Results processing failed: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500


def load_analysis_results(file_path: str):
    """Load analysis results from GCS file"""
    try:
        import geopandas as gpd
        import tempfile
        from google.cloud import storage

        # Download file from GCS
        if file_path.startswith('gs://'):
            # Parse GCS path
            path_parts = file_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1]

            # Download to temp file
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Create temp file with appropriate extension
            suffix = '.gpkg' if blob_path.endswith('.gpkg') else '.csv'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_path = tmp.name

            blob.download_to_filename(temp_path)

            # Load the data
            if temp_path.endswith('.gpkg'):
                gdf = gpd.read_file(temp_path)
            else:
                # Handle CSV with geometry
                import pandas as pd
                from shapely import wkt
                df = pd.read_csv(temp_path)
                if 'geometry' in df.columns:
                    df['geometry'] = df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df, geometry='geometry')
                else:
                    gdf = gpd.GeoDataFrame(df)

            # Cleanup temp file
            import os
            os.unlink(temp_path)

            return gdf
        else:
            return gpd.read_file(file_path)

    except Exception as e:
        logger.error(f"Failed to load analysis results: {str(e)}")
        return None


def format_results_for_frontend(gdf, county_name: str, state: str,
                                transmission_result: dict, slope_result: dict):
    """Format analysis results for frontend display"""
    try:
        # Convert GeoDataFrame to list of dictionaries
        parcels_data = []

        for idx, row in gdf.iterrows():
            parcel_dict = {}

            # Convert all columns to JSON-serializable format
            for col in gdf.columns:
                if col == 'geometry':
                    continue  # Skip geometry for JSON

                value = row[col]

                # Handle different data types
                if pd.isna(value) or value is None:
                    parcel_dict[col] = None
                elif isinstance(value, (list, tuple)):
                    parcel_dict[col] = str(value)  # Convert lists to strings
                elif hasattr(value, 'item'):  # NumPy types
                    parcel_dict[col] = value.item()
                else:
                    parcel_dict[col] = value

            parcels_data.append(parcel_dict)

        # Calculate summary statistics
        total_parcels = len(parcels_data)

        # Count by suitability category if available
        excellent = len([p for p in parcels_data if p.get('suitability_category') == 'Excellent'])
        good = len([p for p in parcels_data if p.get('suitability_category') == 'Good'])
        fair = len([p for p in parcels_data if p.get('suitability_category') == 'Fair'])
        poor = len([p for p in parcels_data if p.get('suitability_category') == 'Poor'])

        # Calculate average score if available
        scores = [p.get('suitability_score') for p in parcels_data if p.get('suitability_score') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Count recommended for outreach
        recommended = len([p for p in parcels_data if p.get('recommended_for_outreach') == True])

        return {
            'parcels_table': parcels_data,
            'summary': {
                'total_parcels': total_parcels,
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor,
                'average_score': round(avg_score, 1),
                'recommended_for_outreach': recommended,
                'location': f"{county_name}, {state}"
            },
            'analysis_metadata': {
                'scoring_method': 'Comprehensive Analysis (Slope + Transmission)',
                'transmission_status': transmission_result.get('status', 'unknown'),
                'slope_status': slope_result.get('status', 'unknown'),
                'generated_at': pd.Timestamp.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to format results: {str(e)}")
        raise

def analyze_existing_file_with_bigquery_internal(file_path, county_name, state, project_type):
    """Internal function to reuse existing BigQuery analysis logic"""

    # Create a mock request object for the existing function
    class MockRequest:
        def get_json(self):
            return {
                'file_path': file_path,
                'county_name': county_name,
                'state': state,
                'project_type': project_type
            }

    # Temporarily replace the request object
    original_request = request

    try:
        # Call your existing function logic directly
        return analyze_existing_file_with_bigquery()
    except Exception as e:
        logger.error(f"Internal BigQuery analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'BigQuery analysis failed: {str(e)}'
        }), 500


def create_simple_parcel_analysis(parcel_data, county_name, state, project_type):
    """Create a simple analysis structure for frontend display"""
    # This is a placeholder - you can enhance this with your existing analysis logic
    total_parcels = len(parcel_data)

    # Simple categorization based on acreage
    excellent = sum(1 for p in parcel_data if float(p.get('acreage', 0)) >= 50)
    good = sum(1 for p in parcel_data if 20 <= float(p.get('acreage', 0)) < 50)
    fair = sum(1 for p in parcel_data if 5 <= float(p.get('acreage', 0)) < 20)
    poor = total_parcels - excellent - good - fair

    # Create enhanced parcel data with simple scoring
    enhanced_parcels = []
    for i, parcel in enumerate(parcel_data):
        acreage = float(parcel.get('acreage', 0))

        # Simple scoring logic
        if acreage >= 50:
            score = 85 + (min(acreage, 200) / 200 * 10)
            category = 'Excellent'
        elif acreage >= 20:
            score = 70 + (acreage / 50 * 15)
            category = 'Good'
        elif acreage >= 5:
            score = 55 + (acreage / 20 * 15)
            category = 'Fair'
        else:
            score = 35 + (acreage / 5 * 20)
            category = 'Poor'

        enhanced_parcel = parcel.copy()
        enhanced_parcel.update({
            'suitability_score': round(score, 1),
            'suitability_category': category,
            'recommended_for_outreach': score >= 70,
            'avg_slope': 15.0,  # Placeholder - you can integrate real slope analysis
            'tx_distance': 'Pending',  # Placeholder - you can integrate real transmission analysis
            'tx_voltage': 'Pending'
        })
        enhanced_parcels.append(enhanced_parcel)

    return {
        'parcels_table': enhanced_parcels,
        'summary': {
            'total_parcels': total_parcels,
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor,
            'average_score': round(sum(float(p['suitability_score']) for p in enhanced_parcels) / total_parcels, 1),
            'recommended_for_outreach': sum(1 for p in enhanced_parcels if p['recommended_for_outreach']),
            'location': f"{county_name}, {state}"
        },
        'analysis_metadata': {
            'scoring_method': 'Simple acreage-based scoring',
            'analysis_date': datetime.now().isoformat(),
            'project_type': project_type
        }
    }

@app.route('/api/execute-parcel-search', methods=['POST'])
@login_required
def parcel_search_run():
    """Execute parcel search"""
    try:
        if not enhanced_parcel_search:
            return jsonify({
                'success': False,
                'error': 'Parcel search module not available'
            }), 500

        data = request.get_json()
        logger.info(f"Parcel search run request: {data}")

        # Call the run_headless function
        result = enhanced_parcel_search.run_headless(**data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Parcel search run error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Search failed: {str(e)}'
        }), 500


@app.route('/api/parcel/files/signed_url', methods=['POST'])
@login_required
def generate_file_signed_url():
    """Generate signed URL for file download"""
    try:
        data = request.get_json()
        blob_name = data.get('blob_name')
        bucket_name = data.get('bucket_name', 'bcfparcelsearchrepository')

        if not blob_name:
            return jsonify({'error': 'blob_name required'}), 400

        if not enhanced_parcel_search:
            return jsonify({'error': 'Parcel search module not available'}), 500

        # Generate signed URL
        signed_url = enhanced_parcel_search.generate_signed_url(blob_name, bucket_name)

        if signed_url:
            return jsonify({
                'success': True,
                'signed_url': signed_url
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate signed URL'
            }), 500

    except Exception as e:
        logger.error(f"Signed URL generation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to generate signed URL: {str(e)}'
        }), 500


@app.route('/api/individual-parcel-analysis', methods=['POST'])
@login_required
def individual_parcel_analysis():
    """
    Run detailed AI analysis on a single parcel
    """
    try:
        if not ENHANCED_PARCEL_AI:
            return jsonify({
                'success': False,
                'error': 'AI analysis service not available. Check ANTHROPIC_API_KEY configuration.'
            }), 500

        data = request.get_json()

        parcel_data = data.get('parcel_data')
        county_name = data.get('county_name', 'Unknown')
        state = data.get('state', 'Unknown')
        project_type = data.get('project_type', 'solar')

        if not parcel_data:
            return jsonify({
                'success': False,
                'error': 'parcel_data is required'
            }), 400

        logger.info(f"Running AI analysis for parcel: {parcel_data.get('parcel_id', 'Unknown')}")

        # Call the enhanced parcel AI service
        ai_analysis = ENHANCED_PARCEL_AI.analyze_single_parcel_detailed(
            parcel_data=parcel_data,
            project_type=project_type,
            location=f"{county_name}, {state}"
        )

        # Format the response for the frontend
        formatted_analysis = {
            'detailed_analysis': format_ai_analysis_for_display(ai_analysis),
            'analysis_type': 'AI-Powered Individual Parcel Analysis',
            'generated_at': datetime.now().isoformat(),
            'parcel_id': parcel_data.get('parcel_id', 'Unknown'),
            'location': f"{county_name}, {state}",
            'project_type': project_type,
            'ai_score': ai_analysis.get('ai_suitability_score', 50),
            'ai_category': ai_analysis.get('ai_suitability_category', 'FAIR'),
            'ai_strengths': ai_analysis.get('ai_strengths', []),
            'ai_challenges': ai_analysis.get('ai_challenges', []),
            'ai_next_steps': ai_analysis.get('ai_next_steps', [])
        }

        return jsonify({
            'success': True,
            'analysis': formatted_analysis,
            'message': 'AI analysis completed successfully'
        })

    except Exception as e:
        logger.error(f"Individual parcel AI analysis error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'AI analysis failed: {str(e)}'
        }), 500


def format_ai_analysis_for_display(ai_analysis):
    """Format AI analysis results for frontend display"""

    # Get the key analysis components
    score = ai_analysis.get('ai_suitability_score', 50)
    category = ai_analysis.get('ai_suitability_category', 'FAIR')
    complexity = ai_analysis.get('ai_development_complexity', 'MEDIUM')
    priority = ai_analysis.get('ai_investment_priority', 'MEDIUM')
    timeline = ai_analysis.get('ai_development_timeline', 'STANDARD')

    strengths = ai_analysis.get('ai_strengths', ['Standard development factors'])
    challenges = ai_analysis.get('ai_challenges', ['Requires detailed assessment'])
    next_steps = ai_analysis.get('ai_next_steps', ['Site verification needed'])

    technical_notes = ai_analysis.get('ai_technical_notes', 'Technical analysis completed')
    economic_notes = ai_analysis.get('ai_economic_notes', 'Economic factors assessed')
    risk_assessment = ai_analysis.get('ai_risk_assessment', 'Standard development risks apply')

    # Create formatted analysis text
    analysis_text = f"""
DEVELOPMENT SUITABILITY ANALYSIS

🎯 Overall Assessment: {score}/100 ({category})
📊 Development Complexity: {complexity}
💡 Investment Priority: {priority}
⏱️ Development Timeline: {timeline}

🟢 KEY STRENGTHS:
{chr(10).join([f"• {strength}" for strength in strengths])}

🟡 CHALLENGES TO ADDRESS:
{chr(10).join([f"• {challenge}" for challenge in challenges])}

📋 RECOMMENDED NEXT STEPS:
{chr(10).join([f"• {step}" for step in next_steps])}

🔧 TECHNICAL CONSIDERATIONS:
{technical_notes}

💰 ECONOMIC FACTORS:
{economic_notes}

⚠️ RISK ASSESSMENT:
{risk_assessment}

---
Analysis generated using AI-powered renewable energy development assessment
    """.strip()

    return analysis_text


def get_ai_setup_recommendations(test_results):
    """Get setup recommendations based on test results"""
    recommendations = []

    if not test_results.get('anthropic_key_configured'):
        recommendations.append('Set ANTHROPIC_API_KEY in your environment variables')

    if not test_results.get('ai_service_available'):
        recommendations.append('Ensure ai_service.py is in services/ directory')

    if not test_results.get('enhanced_parcel_ai_available'):
        recommendations.append('Ensure enhanced_parcel_ai_service.py is in services/ directory')

    if not test_results.get('ai_service_functional'):
        error = test_results.get('ai_service_error', 'Unknown error')
        recommendations.append(f'Fix AI service initialization error: {error}')

    if not recommendations:
        recommendations.append('AI services are properly configured and functional')

    return recommendations

@app.route('/api/test-env-vars', methods=['GET'])
def test_env_vars():
    """Test if Flask can read environment variables"""
    return jsonify({
        'anthropic_key_available': bool(os.getenv('ANTHROPIC_API_KEY')),
        'anthropic_key_starts_with_sk': os.getenv('ANTHROPIC_API_KEY', '').startswith('sk-'),
        'anthropic_key_length': len(os.getenv('ANTHROPIC_API_KEY', '')),
        'flask_secret_available': bool(os.getenv('FLASK_SECRET_KEY')),
        'working_directory': os.getcwd()
    })


@app.route('/api/county-market-analysis', methods=['POST'])
@login_required
def county_market_analysis():
    """Run AI market analysis for a specific county"""
    try:
        if not AI_SERVICE:
            logger.error("AI_SERVICE is None")
            return jsonify({
                'success': True,  # Still return success with fallback
                'analysis': create_fallback_county_analysis('Unknown', 'Unknown', 'solar'),
                'analysis_type': 'Fallback Analysis - AI Service Unavailable',
                'error_note': 'AI_SERVICE not initialized'
            })

        # Test the AI service connection first
        connection_test = AI_SERVICE.test_connection()
        if not connection_test.get('success'):
            logger.error(f"AI service connection failed: {connection_test.get('error')}")

            # Use fallback but inform user
            data = request.get_json()
            county_name = data.get('county_name', 'Unknown')
            state = data.get('state', 'Unknown')
            project_type = data.get('project_type', 'solar')

            return jsonify({
                'success': True,
                'analysis': create_fallback_county_analysis(county_name, state, project_type),
                'analysis_type': 'Fallback Analysis - AI Service Connection Failed',
                'error_note': f'AI connection error: {connection_test.get("error")}'
            })

        # If we get here, AI service should work
        data = request.get_json()
        county_fips = data.get('county_fips')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type', 'solar')

        if not county_name or not state:
            return jsonify({
                'success': False,
                'error': 'county_name and state are required'
            }), 400

        logger.info(f"Running AI county market analysis for {county_name}, {state}")

        # Create a single county "list" for the AI analysis
        county_data = [{
            'name': county_name,
            'fips': county_fips,
            'state_code': state
        }]

        # Call your existing AI service
        ai_result = AI_SERVICE.analyze_state_counties(state, project_type, county_data)

        if ai_result and ai_result.get('county_rankings'):
            # Get the analysis for our specific county
            county_analysis = ai_result['county_rankings'][0] if ai_result['county_rankings'] else None

            if county_analysis:
                # Format the comprehensive analysis
                formatted_analysis = format_county_analysis_for_display(
                    county_analysis,
                    ai_result.get('analysis_summary', ''),
                    project_type,
                    f"{county_name}, {state}"
                )

                return jsonify({
                    'success': True,
                    'analysis': formatted_analysis,
                    'county_name': county_name,
                    'state': state,
                    'project_type': project_type,
                    'analysis_type': 'AI-Powered County Market Analysis'
                })

        # Fallback if AI analysis fails
        fallback = create_fallback_county_analysis(county_name, state, project_type)
        return jsonify({
            'success': True,
            'analysis': fallback,
            'county_name': county_name,
            'state': state,
            'project_type': project_type,
            'analysis_type': 'Fallback Analysis - AI Analysis Failed',
            'note': 'AI analysis failed - using fallback content'
        })

    except Exception as e:
        logger.error(f"County market analysis error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Return fallback analysis on error
        data = request.get_json() or {}
        fallback = create_fallback_county_analysis(
            data.get('county_name', 'Unknown'),
            data.get('state', 'Unknown'),
            data.get('project_type', 'solar')
        )

        return jsonify({
            'success': True,  # Still return success with fallback
            'analysis': fallback,
            'analysis_type': 'Fallback Analysis - Exception Occurred',
            'error_note': f'Exception: {str(e)}'
        })


def format_county_analysis_for_display(county_analysis, state_summary, project_type, location):
    """Format AI county analysis for frontend display"""

    score = county_analysis.get('score', 50)
    rank = county_analysis.get('rank', 'Unknown')
    strengths = county_analysis.get('strengths', [])
    challenges = county_analysis.get('challenges', [])
    resource_quality = county_analysis.get('resource_quality', 'Good')
    policy_environment = county_analysis.get('policy_environment', 'Neutral')
    development_activity = county_analysis.get('development_activity', 'Low')
    summary = county_analysis.get('summary', 'County analysis completed')

    # Create comprehensive analysis text
    analysis_text = f"""
AI-POWERED COUNTY MARKET ANALYSIS

🎯 Overall Development Score: {score}/100 (Rank #{rank} in state)
📍 Location: {location}
⚡ Project Focus: {project_type.title()} Energy Development

MARKET ASSESSMENT OVERVIEW:
{state_summary}

COUNTY-SPECIFIC ANALYSIS:
{summary}

🟢 KEY STRENGTHS & OPPORTUNITIES:
{chr(10).join([f"• {strength}" for strength in strengths])}

🟡 CHALLENGES & CONSIDERATIONS:
{chr(10).join([f"• {challenge}" for challenge in challenges])}

📊 MARKET FACTORS:
• Resource Quality: {resource_quality}
• Policy Environment: {policy_environment}  
• Development Activity: {development_activity}

🚀 STRATEGIC RECOMMENDATIONS:

IMMEDIATE OPPORTUNITIES:
• Focus on {project_type} development advantages identified above
• Leverage {resource_quality.lower()} resource quality for competitive positioning
• Navigate {policy_environment.lower()} regulatory environment strategically

DEVELOPMENT PATHWAY:
• Conduct detailed feasibility studies for priority sites
• Engage with local stakeholders and permitting authorities
• Assess grid interconnection opportunities and constraints
• Evaluate land acquisition strategies and costs

COMPETITIVE POSITIONING:
• Ranking #{rank} suggests {"strong competitive position" if int(str(rank).replace('#', '')) <= 10 else "moderate competitive position"}
• {resource_quality} resource quality provides {"significant" if resource_quality == "Excellent" else "reasonable"} technical advantages
• {policy_environment} policy environment {"supports" if "Supportive" in policy_environment else "allows for"} development activities

NEXT STEPS FOR MARKET ENTRY:
• Validate market assumptions with local data and contacts
• Identify and prioritize specific development sites  
• Assess competition and market saturation levels
• Develop relationships with key local stakeholders

---
Analysis generated using AI-powered renewable energy market intelligence
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """.strip()

    return analysis_text


def create_fallback_county_analysis(county_name='Unknown', state='Unknown', project_type='solar'):
    """Create fallback analysis when AI is unavailable"""

    analysis_text = f"""
COUNTY MARKET ANALYSIS - {county_name.upper()}, {state.upper()}

🎯 Project Type: {project_type.title()} Energy Development
📍 Analysis Level: County Market Assessment
⚙️ Analysis Method: Deterministic Market Factors

DEVELOPMENT ASSESSMENT:

MARKET FUNDAMENTALS:
• {county_name} County represents a potential {project_type} development opportunity
• State-level policies in {state} generally support renewable energy development
• County-level factors require detailed site-specific evaluation

TYPICAL DEVELOPMENT FACTORS:
• Resource Availability: {state} generally has adequate renewable energy resources
• Grid Infrastructure: Most counties have some level of transmission access
• Land Availability: Rural counties typically offer development opportunities
• Regulatory Environment: Varies by local jurisdiction and project scale

STRATEGIC CONSIDERATIONS:

OPPORTUNITY ASSESSMENT:
• Market entry feasibility depends on local resource quality
• Competition levels vary significantly by region within state
• Development timeline affected by permitting and interconnection processes

RECOMMENDED NEXT STEPS:
• Conduct detailed resource assessment for {project_type} development
• Engage with local utilities for interconnection discussions  
• Research county-specific zoning and permitting requirements
• Identify potential development sites and land acquisition opportunities
• Assess local community support and potential opposition

DEVELOPMENT PATHWAY:
• Phase 1: Market validation and resource confirmation
• Phase 2: Site identification and preliminary feasibility
• Phase 3: Detailed site assessment and permitting strategy
• Phase 4: Project development and construction planning

RISK FACTORS:
• Resource quality may vary significantly within county boundaries
• Local regulations and community acceptance uncertain without research
• Grid interconnection capacity and costs require utility engagement
• Land acquisition complexity depends on ownership patterns and pricing

---
Note: This is a basic market analysis framework. For detailed AI-powered market intelligence including specific scores, competitive analysis, and strategic recommendations, please configure ANTHROPIC_API_KEY in your environment.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """.strip()

    return analysis_text


# In your app.py, add this test route to verify AI service
@app.route('/api/test-ai-service', methods=['GET'])
@login_required
def test_ai_service():
    """Test AI service connectivity"""
    try:
        if not AI_SERVICE:
            return jsonify({
                'success': False,
                'error': 'AI_SERVICE not initialized',
                'has_anthropic_key': bool(os.getenv('ANTHROPIC_API_KEY')),
                'anthropic_key_length': len(os.getenv('ANTHROPIC_API_KEY', ''))
            })

        # Test the service with a simple call
        test_result = AI_SERVICE._get_fallback_variables('solar', 'county')

        return jsonify({
            'success': True,
            'ai_service_available': True,
            'test_result': bool(test_result),
            'message': 'AI service is working properly'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'has_anthropic_key': bool(os.getenv('ANTHROPIC_API_KEY'))
        })


@app.route('/api/test-ai-isolation', methods=['GET'])
def test_ai_isolation():
    """Run isolation test for AI service"""
    try:
        import subprocess
        import sys

        # Run the isolation test
        result = subprocess.run([
            sys.executable, '/app/test_ai_isolated.py'
        ], capture_output=True, text=True, timeout=30)

        return jsonify({
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/test-ai-connectivity', methods=['GET'])
@login_required
def test_ai_connectivity():
    """Test AI service connectivity"""
    try:
        import os

        results = {
            'ai_service_available': AI_SERVICE is not None,
            'ai_service_class_name': AI_SERVICE.__class__.__name__ if AI_SERVICE else None,
            'anthropic_key_configured': bool(os.getenv('ANTHROPIC_API_KEY')),
            'anthropic_key_length': len(os.getenv('ANTHROPIC_API_KEY', '')),
            'anthropic_key_format': os.getenv('ANTHROPIC_API_KEY', '')[:10] + '...' if os.getenv(
                'ANTHROPIC_API_KEY') else None,
            'flask_debug': app.debug,
            'timestamp': datetime.now().isoformat()
        }

        # Test basic AI service if available
        if AI_SERVICE:
            try:
                if hasattr(AI_SERVICE, 'client') and AI_SERVICE.client:
                    # Test the connection
                    test_result = AI_SERVICE.test_connection()
                    results['ai_service_functional'] = test_result.get('success', False)
                    results['ai_test_result'] = test_result
                else:
                    results['ai_service_functional'] = False
                    results['ai_service_error'] = 'AI service client is None'
            except Exception as e:
                results['ai_service_functional'] = False
                results['ai_service_error'] = str(e)
        else:
            results['ai_service_functional'] = False
            results['ai_service_error'] = 'AI_SERVICE is None'

        return jsonify({
            'success': True,
            'connectivity_test': results
        })

    except Exception as e:
        logger.error(f"AI connectivity test error: {e}")
        return jsonify({
            'success': False,
            'error': f'Connectivity test failed: {str(e)}'
        }), 500


@app.route('/api/debug-dependencies', methods=['GET'])
def debug_dependencies():
    """Check for missing dependencies"""
    dependencies = {}

    try:
        import anthropic
        dependencies['anthropic'] = f"✅ {anthropic.__version__}"
    except ImportError as e:
        dependencies['anthropic'] = f"❌ {str(e)}"

    try:
        from services.ai_service import AIAnalysisService
        dependencies['ai_service'] = "✅ Importable"
    except ImportError as e:
        dependencies['ai_service'] = f"❌ {str(e)}"

    try:
        from services.cache_service import AIResponseCache
        dependencies['cache_service'] = "✅ Importable"
    except ImportError as e:
        dependencies['cache_service'] = f"❌ {str(e)}"

    try:
        from models.project_config import ProjectConfig
        dependencies['project_config'] = "✅ Importable"
    except ImportError as e:
        dependencies['project_config'] = f"❌ {str(e)}"

    return jsonify({
        'dependencies': dependencies,
        'working_directory': os.getcwd(),
        'python_path': sys.path,
        'ai_service_status': AI_SERVICE is not None
    })


@app.route('/api/debug-file-structure', methods=['GET'])
def debug_file_structure():
    """Check what files exist in the container"""
    import os
    import glob

    file_structure = {}

    # Check working directory contents
    working_dir = os.getcwd()
    file_structure['working_directory'] = working_dir
    file_structure['root_files'] = os.listdir(working_dir)

    # Check for services directory
    services_dir = os.path.join(working_dir, 'services')
    file_structure['services_exists'] = os.path.exists(services_dir)
    if os.path.exists(services_dir):
        file_structure['services_files'] = os.listdir(services_dir)

    # Check for models directory
    models_dir = os.path.join(working_dir, 'models')
    file_structure['models_exists'] = os.path.exists(models_dir)
    if os.path.exists(models_dir):
        file_structure['models_files'] = os.listdir(models_dir)

    # Check for specific files
    critical_files = [
        'services/ai_service.py',
        'services/cache_service.py',
        'models/project_config.py',
        'services/__init__.py',
        'models/__init__.py'
    ]

    file_structure['critical_files'] = {}
    for file_path in critical_files:
        full_path = os.path.join(working_dir, file_path)
        file_structure['critical_files'][file_path] = os.path.exists(full_path)

    return jsonify(file_structure)


@app.route('/api/create-missing-files', methods=['POST'])
@login_required
def create_missing_files():
    """Temporarily create missing files"""
    try:
        import os

        # Create directories
        os.makedirs('services', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Create __init__.py files
        with open('services/__init__.py', 'w') as f:
            f.write('# Services module\n')

        with open('models/__init__.py', 'w') as f:
            f.write('# Models module\n')

        # Create minimal project_config.py
        project_config_content = '''
class ProjectConfig:
    @staticmethod
    def get_tier_criteria(analysis_level):
        return ["Resource Quality", "Market Opportunity", "Technical Feasibility", "Regulatory Environment"]

    @staticmethod  
    def get_tier_description(analysis_level):
        descriptions = {
            'state': 'State-level market entry and opportunity assessment',
            'county': 'County-level development feasibility analysis', 
            'site': 'Site-specific technical and commercial evaluation'
        }
        return descriptions.get(analysis_level, f"{analysis_level.title()}-level renewable energy analysis")
'''

        with open('models/project_config.py', 'w') as f:
            f.write(project_config_content)

        # Create minimal cache_service.py
        cache_service_content = '''
import logging
logger = logging.getLogger(__name__)

class AIResponseCache:
    def __init__(self):
        logger.info("Cache service initialized (minimal version)")

    def get_cached_response(self, **kwargs):
        return None

    def store_response(self, **kwargs):
        return True

    def get_cache_stats(self):
        return {"status": "minimal_cache"}

    def clear_location_cache(self, location, analysis_level=None):
        return 0
'''

        with open('services/cache_service.py', 'w') as f:
            f.write(cache_service_content)

        return jsonify({
            'success': True,
            'message': 'Missing files created',
            'files_created': [
                'services/__init__.py',
                'models/__init__.py',
                'models/project_config.py',
                'services/cache_service.py'
            ]
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug-import-errors', methods=['GET'])
def debug_import_errors():
    """Test imports individually to find specific errors"""
    import sys
    import traceback

    results = {}

    # Test each import individually
    test_imports = [
        ('services', 'import services'),
        ('models', 'import models'),
        ('services.ai_service', 'from services.ai_service import AIAnalysisService'),
        ('services.cache_service', 'from services.cache_service import AIResponseCache'),
        ('models.project_config', 'from models.project_config import ProjectConfig')
    ]

    for name, import_statement in test_imports:
        try:
            exec(import_statement)
            results[name] = {'success': True, 'error': None}
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    # Also check if modules are in sys.modules after successful imports
    modules_in_sys = {}
    for module_name in ['services', 'models', 'services.ai_service', 'services.cache_service',
                        'models.project_config']:
        modules_in_sys[module_name] = module_name in sys.modules

    return jsonify({
        'import_tests': results,
        'modules_in_sys': modules_in_sys,
        'sys_path': sys.path[:5]  # First 5 entries
    })


# Temporary bypass: Create minimal AI service class directly in app.py
class MinimalAIService:
    def __init__(self, api_key=None):
        import os
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = None

        if self.api_key and self.api_key.startswith('sk-ant-'):
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("✅ Minimal AI service created successfully")
            except Exception as e:
                print(f"❌ Minimal AI client creation failed: {e}")
                self.client = None
        else:
            print("❌ Invalid or missing API key")

    def test_connection(self):
        """Test the connection"""
        if not self.client:
            return {"success": False, "error": "Client not initialized"}

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return {"success": True, "message": "Connection working"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_state_counties(self, state, project_type, counties):
        """Fallback analysis"""
        analyzed_counties = []
        for i, county in enumerate(counties):
            analyzed_counties.append({
                'name': county.get('name', f'County_{i}'),
                'fips': county.get('fips', ''),
                'score': 75 - (i * 2),
                'rank': i + 1,
                'strengths': [f'{project_type.title()} potential', 'Infrastructure access'],
                'challenges': ['Requires detailed analysis'],
                'resource_quality': 'Good',
                'policy_environment': 'Supportive',
                'development_activity': 'Medium'
            })

        return {
            'analysis_summary': f'{state} {project_type} development analysis',
            'county_rankings': analyzed_counties
        }


# Replace your current AI_SERVICE initialization with this:
print("=== ATTEMPTING MINIMAL AI SERVICE ===")
try:
    AI_SERVICE = MinimalAIService()
    if AI_SERVICE.client:
        test_result = AI_SERVICE.test_connection()
        print(f"Minimal AI service test: {test_result}")
        AI_AVAILABLE = test_result.get('success', False)
    else:
        AI_AVAILABLE = False
        print("Minimal AI service created but no client")
except Exception as e:
    print(f"Even minimal AI service failed: {e}")
    AI_SERVICE = None
    AI_AVAILABLE = False

print(f"AI_AVAILABLE: {AI_AVAILABLE}")


@app.route('/api/test-anthropic-direct', methods=['GET'])
def test_anthropic_direct():
    """Test Anthropic client creation directly"""
    try:
        import os
        import anthropic
        import traceback

        print("=== DIRECT ANTHROPIC TEST ===")

        api_key = os.getenv('ANTHROPIC_API_KEY')
        print(f"API key exists: {bool(api_key)}")
        print(f"API key length: {len(api_key) if api_key else 0}")

        if not api_key:
            return jsonify({'success': False, 'error': 'No API key'})

        # Create client with minimal parameters
        print("Creating Anthropic client with only api_key...")
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Direct client creation successful")

        # Test it
        print("Testing client...")
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        print("✅ Direct client test successful")

        return jsonify({
            'success': True,
            'message': 'Direct Anthropic client works perfectly',
            'anthropic_version': anthropic.__version__,
            'response_text': response.content[0].text
        })

    except Exception as e:
        print(f"❌ Direct Anthropic test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/debug-environment', methods=['GET'])
def debug_environment():
    """Check for proxy-related environment variables"""
    import os

    proxy_vars = {}
    all_env_vars = dict(os.environ)

    # Check for common proxy environment variables
    proxy_keys = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
        'NO_PROXY', 'no_proxy', 'ALL_PROXY', 'all_proxy',
        'GOOGLE_CLOUD_PROJECT', 'GAE_ENV', 'CLOUD_RUN_SERVICE'
    ]

    for key in proxy_keys:
        proxy_vars[key] = all_env_vars.get(key, 'Not set')

    # Also check for any environment variable containing 'proxy'
    proxy_related = {k: v for k, v in all_env_vars.items()
                     if 'proxy' in k.lower() or 'PROXY' in k}

    return jsonify({
        'common_proxy_vars': proxy_vars,
        'all_proxy_related': proxy_related,
        'total_env_vars': len(all_env_vars),
        'anthropic_version': '0.28.0'  # from your requirements
    })

@app.route('/api/test-anthropic-fixed', methods=['GET'])
def test_anthropic_fixed():
    """Test Anthropic with explicit proxy handling"""
    try:
        import os
        import anthropic

        # Clear proxy environment variables temporarily
        original_proxies = {}
        proxy_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']

        for key in proxy_keys:
            if key in os.environ:
                original_proxies[key] = os.environ[key]
                del os.environ[key]

        api_key = os.getenv('ANTHROPIC_API_KEY')

        print("Creating Anthropic client with proxy environment cleared...")

        # Create client with only api_key
        client = anthropic.Anthropic(api_key=api_key)

        print("✅ Fixed client creation successful")

        # Test it
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )

        # Restore original proxy settings
        for key, value in original_proxies.items():
            os.environ[key] = value

        print("✅ Fixed client test successful")

        return jsonify({
            'success': True,
            'message': 'Fixed Anthropic client works!',
            'anthropic_version': anthropic.__version__,
            'response_text': response.content[0].text,
            'cleared_proxies': list(original_proxies.keys())
        })

    except Exception as e:
        # Restore proxy settings even on error
        for key, value in original_proxies.items():
            os.environ[key] = value

        return jsonify({
            'success': False,
            'error': str(e),
            'cleared_proxies': list(original_proxies.keys()) if 'original_proxies' in locals() else []
        })


@app.route('/api/debug-anthropic-version', methods=['GET'])
def debug_anthropic_version():
    """Check actual anthropic version in deployment"""
    try:
        import anthropic
        import sys
        import pkg_resources

        # Get version info
        version_info = {
            'anthropic_version': anthropic.__version__,
            'anthropic_file_location': anthropic.__file__,
            'python_version': sys.version,
            'installed_packages': {}
        }

        # Check installed packages
        try:
            for package in ['anthropic', 'requests', 'httpx']:
                try:
                    version_info['installed_packages'][package] = pkg_resources.get_distribution(package).version
                except:
                    version_info['installed_packages'][package] = 'Not found'
        except:
            pass

        return jsonify(version_info)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Could not check anthropic version'
        })


@app.route('/api/test-anthropic-bypass', methods=['GET'])
def test_anthropic_bypass():
    """Test anthropic with manual client setup"""
    try:
        import os
        import requests
        import json

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'No API key'})

        # Make direct API call instead of using the client
        headers = {
            'x-api-key': api_key,
            'content-type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': 'claude-3-haiku-20240307',
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'test'}]
        }

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'message': 'Direct API call works!',
                'response_text': result['content'][0]['text']
            })
        else:
            return jsonify({
                'success': False,
                'error': f'API call failed: {response.status_code}',
                'response': response.text
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


class DirectAnthropicAI:
    def __init__(self):
        import os
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.base_url = 'https://api.anthropic.com/v1/messages'

    def test_connection(self):
        """Test with direct API call"""
        if not self.api_key:
            return {"success": False, "error": "No API key"}

        try:
            import requests

            headers = {
                'x-api-key': self.api_key,
                'content-type': 'application/json',
                'anthropic-version': '2023-06-01'
            }

            data = {
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 10,
                'messages': [{'role': 'user', 'content': 'test'}]
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                return {"success": True, "message": "Direct API working"}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_state_counties(self, state, project_type, counties):
        """AI analysis using direct API calls"""
        if not self.api_key:
            return None

        try:
            import requests
            import json

            # Create the same prompt as your AI service
            county_names = [c.get('name', f'County_{i}') for i, c in enumerate(counties)]

            prompt = f"""
            You are a renewable energy market analyst. Analyze and rank ALL counties in {state} for {project_type} energy development potential.

            Counties to analyze: {', '.join(county_names)}

            For EACH county listed above, provide a score 0-100 and return as JSON:
            {{
                "analysis_summary": "Brief overview of {state}'s {project_type} energy landscape",
                "county_rankings": [
                    {{
                        "name": "County Name",
                        "score": 85,
                        "rank": 1,
                        "strengths": ["Strong infrastructure"],
                        "challenges": ["Limited land"],
                        "resource_quality": "Excellent",
                        "policy_environment": "Supportive",
                        "development_activity": "High"
                    }}
                ]
            }}
            """

            headers = {
                'x-api-key': self.api_key,
                'content-type': 'application/json',
                'anthropic-version': '2023-06-01'
            }

            data = {
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 4000,
                'messages': [{'role': 'user', 'content': prompt}]
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)

            if response.status_code == 200:
                result = response.json()
                response_text = result['content'][0]['text']

                # Parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())

                    # Add missing FIPS codes
                    returned_counties = analysis_data.get('county_rankings', [])
                    for ranking in returned_counties:
                        county_name = ranking.get('name', '')
                        for county in counties:
                            if county.get('name', '').lower() in county_name.lower():
                                ranking['fips'] = county.get('fips', f'{state}999')
                                break

                    return analysis_data

            return None

        except Exception as e:
            print(f"Direct API analysis error: {e}")
            return None


# Replace your AI service initialization with:
print("=== CREATING DIRECT API AI SERVICE ===")
try:
    AI_SERVICE = DirectAnthropicAI()
    test_result = AI_SERVICE.test_connection()
    print(f"Direct API test: {test_result}")
    AI_AVAILABLE = test_result.get('success', False)
    print(f"Direct API AI available: {AI_AVAILABLE}")
except Exception as e:
    print(f"Direct API AI failed: {e}")
    AI_SERVICE = None
    AI_AVAILABLE = False

app.route('/api/debug-api-key', methods=['GET'])


def debug_api_key():
    """Debug API key configuration"""
    import os

    api_key = os.getenv('ANTHROPIC_API_KEY')

    return jsonify({
        'api_key_exists': bool(api_key),
        'api_key_length': len(api_key) if api_key else 0,
        'api_key_starts_correctly': api_key.startswith('sk-ant-') if api_key else False,
        'api_key_preview': api_key[:20] + '...' if api_key else None,
        'environment_vars': {
            'ANTHROPIC_API_KEY': 'SET' if api_key else 'NOT SET',
            'HTTP_PROXY': os.getenv('HTTP_PROXY', 'NOT SET'),
            'HTTPS_PROXY': os.getenv('HTTPS_PROXY', 'NOT SET')
        }
    })


class CleanAIService:
    def __init__(self):
        """Clean AI service initialization"""
        import os

        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None

        # Clear any proxy variables that might interfere
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        self._original_proxies = {}

        for var in proxy_vars:
            if var in os.environ:
                self._original_proxies[var] = os.environ[var]
                del os.environ[var]

        if self.api_key and self.api_key.startswith('sk-ant-'):
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✅ Clean AI service initialized successfully")
            except Exception as e:
                logger.error(f"❌ AI client creation failed: {e}")
                self.client = None
        else:
            logger.error("❌ Invalid or missing API key")

        # Restore proxy variables
        for var, value in self._original_proxies.items():
            os.environ[var] = value

    def test_connection(self):
        """Test API connection"""
        if not self.client:
            return {"success": False, "error": "Client not initialized"}

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return {"success": True, "message": "Connection working"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_state_counties(self, state, project_type, counties):
        """AI-powered county analysis"""
        if not self.client:
            logger.error("No AI client available - using fallback")
            return self._create_fallback_analysis(state, project_type, counties)

        try:
            county_names = [c.get('name', f'County_{i}') for i, c in enumerate(counties)]

            prompt = f"""
            Analyze and rank ALL counties in {state} for {project_type} energy development potential.

            Counties: {', '.join(county_names)}

            Return JSON with this exact structure:
            {{
                "analysis_summary": "Brief {state} {project_type} market overview",
                "county_rankings": [
                    {{
                        "name": "CountyName",
                        "score": 85,
                        "rank": 1,
                        "strengths": ["advantage1", "advantage2"],
                        "challenges": ["challenge1"],
                        "resource_quality": "Excellent",
                        "policy_environment": "Supportive",
                        "development_activity": "High",
                        "summary": "Brief explanation"
                    }}
                ]
            }}
            """

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            import re

            response_text = response.content[0].text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                analysis_data = json.loads(json_match.group())

                # Add FIPS codes
                for ranking in analysis_data.get('county_rankings', []):
                    county_name = ranking.get('name', '')
                    for county in counties:
                        if county.get('name', '').lower() in county_name.lower():
                            ranking['fips'] = county.get('fips', f'{state}999')
                            break

                logger.info(f"✅ AI analysis completed for {len(analysis_data.get('county_rankings', []))} counties")
                return analysis_data

        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")

        return self._create_fallback_analysis(state, project_type, counties)

    def _create_fallback_analysis(self, state, project_type, counties):
        """Fallback when AI fails"""
        logger.warning("Using fallback analysis")

        analyzed_counties = []
        for i, county in enumerate(counties):
            analyzed_counties.append({
                'name': county.get('name', f'County_{i}'),
                'fips': county.get('fips', ''),
                'score': 75 - (i * 2),
                'rank': i + 1,
                'strengths': [f'{project_type.title()} potential', 'Infrastructure access'],
                'challenges': ['Requires detailed analysis'],
                'resource_quality': 'Good',
                'policy_environment': 'Supportive',
                'development_activity': 'Medium',
                'summary': f'Standard {project_type} development potential'
            })

        return {
            'analysis_summary': f'{state} {project_type} development analysis (fallback)',
            'county_rankings': analyzed_counties
        }


@app.route('/api/debug-api-key', methods=['GET'])
def debug_api_key():
    """Debug API key configuration"""
    import os

    api_key = os.getenv('ANTHROPIC_API_KEY')

    return jsonify({
        'api_key_exists': bool(api_key),
        'api_key_length': len(api_key) if api_key else 0,
        'api_key_starts_correctly': api_key.startswith('sk-ant-') if api_key else False,
        'api_key_preview': api_key[:20] + '...' if api_key else None,
        'environment_vars': {
            'ANTHROPIC_API_KEY': 'SET' if api_key else 'NOT SET',
            'HTTP_PROXY': os.getenv('HTTP_PROXY', 'NOT SET'),
            'HTTPS_PROXY': os.getenv('HTTPS_PROXY', 'NOT SET')
        }
    })


@app.route('/api/debug-analysis-data/<county_name>', methods=['GET'])
def debug_analysis_data(county_name):
    """Debug endpoint to check what data is available"""
    try:
        from google.cloud import bigquery
        client = bigquery.Client()

        # Check transmission data
        tx_query = f"""
        SELECT COUNT(*) as tx_count,
               ST_X(ST_CENTROID(ST_UNION_AGG(geometry))) as center_x,
               ST_Y(ST_CENTROID(ST_UNION_AGG(geometry))) as center_y
        FROM `{client.project}.transmission_analysis.transmission_lines`
        """

        tx_result = list(client.query(tx_query).result())[0]

        # Check slope data
        slope_query = f"""
        SELECT COUNT(*) as slope_count
        FROM `{client.project}.spatial_analysis.slope_grid`
        """

        slope_result = list(client.query(slope_query).result())[0]

        return jsonify({
            'transmission_lines': tx_result.tx_count,
            'transmission_center': f"{tx_result.center_x:.4f}, {tx_result.center_y:.4f}",
            'slope_grid_cells': slope_result.slope_count,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting BCF.ai app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
