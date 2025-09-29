import os
from pathlib import Path
from dotenv import load_dotenv

# Force load .env file from project root
project_root = Path(__file__).parent
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"Loaded .env from: {env_path}")
    # Verify the key was loaded
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"ANTHROPIC_API_KEY loaded: {api_key[:15]}...")
    else:
        print("WARNING: ANTHROPIC_API_KEY not found in .env")
else:
    print(f"WARNING: .env file not found at {env_path}")

import logging
import sys
from datetime import datetime
from flask import Flask, render_template, session, redirect, url_for, jsonify, request

# Local imports
from config.settings import config
from config.database import db
from services.ai_service import ai_service
from api.auth_routes import auth_bp
from api.county_routes import county_bp
from api.parcel_routes import parcel_bp
from api.analysis_routes import analysis_bp
from api.crm_routes import crm_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY
    
    print("=== REGISTERING BLUEPRINTS ===")
    
    try:
        print("Registering auth_bp...")
        app.register_blueprint(auth_bp, url_prefix='/auth')
        print("✅ auth_bp registered")
    except Exception as e:
        print(f"❌ auth_bp failed: {e}")
    
    try:
        print("Registering analysis_bp...")
        app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
        print("✅ analysis_bp registered")
    except Exception as e:
        print(f"❌ analysis_bp failed: {e}")

    try:
        print("Registering crm_bp...")
        app.register_blueprint(crm_bp, url_prefix='/api/crm')
        print("✅ crm_bp registered")
    except Exception as e:
        print(f"❌ crm_bp failed: {e}")
    
    # Print all registered routes
    print("\n=== ALL REGISTERED ROUTES ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint} [{', '.join(rule.methods)}]")
    print("===============================\n")
    
    return app

app = create_app()

# Add this right after creating the app, before the error handlers
@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    return jsonify({'status': 'working', 'method': request.method})

# Initialize services on startup
@app.before_request
def initialize_services():
    """Initialize all services and verify connections"""
    if not hasattr(app, '_services_initialized'):
        logger.info("Initializing BCF.ai services...")
       
        # Test database connections
        db_status = db.test_connections()
        if db_status['bigquery']:
            logger.info("✅ BigQuery connection verified")
        else:
            logger.warning("⚠️  BigQuery connection failed")
       
        if db_status['storage']:
            logger.info("✅ Cloud Storage connection verified")
        else:
            logger.warning("⚠️  Cloud Storage connection failed")
       
        if db_status['errors']:
            for error in db_status['errors']:
                logger.error(f"Database error: {error}")
       
        # Test AI service
        ai_status = ai_service.test_connection()
        if ai_status.success:
            logger.info("✅ AI service connection verified")
        else:
            logger.warning(f"⚠️  AI service connection failed: {ai_status.error}")
        
        # Mark as initialized to prevent re-running
        app._services_initialized = True

# Core routes
@app.route('/')
def index():
    """Main application page"""
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('auth.login_page'))
    return render_template('index.html', user=session.get('username', 'User'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    # Test all services
    db_status = db.test_connections()
    ai_status = ai_service.test_connection()
    
    health_data = {
        'status': 'healthy' if all([
            db_status['bigquery'], 
            db_status['storage'], 
            ai_status.success
        ]) else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'bigquery': db_status['bigquery'],
            'storage': db_status['storage'],
            'ai_service': ai_status.success
        },
        'version': '2.0.0'
    }
    
    status_code = 200 if health_data['status'] == 'healthy' else 503
    return jsonify(health_data), status_code

@app.route('/system-status')
def system_status():
    """Detailed system status for debugging"""
    return jsonify({
        'config': {
            'debug': config.DEBUG,
            'environment': 'production' if config.is_production else 'development',
            'ai_configured': bool(config.ANTHROPIC_API_KEY),
            'gcp_configured': bool(config.GOOGLE_APPLICATION_CREDENTIALS)
        },
        'services': {
            'ai_service': ai_service.get_service_status(),
            'database': db.test_connections()
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': 'An unexpected error occurred'}), 500

# CORS headers
@app.after_request
def after_request(response):
    # Prevent caching of API responses
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    # Existing CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    logger.info("Starting BCF.ai application...")
    logger.info(f"Configuration: Debug={config.DEBUG}, Port={config.PORT}")
    
    # Validate configuration
    config_errors = config.validate()
    if config_errors:
        logger.error("Configuration validation failed:")
        for error in config_errors:
            logger.error(f"  - {error}")
        if config.is_production:
            sys.exit(1)
        else:
            logger.warning("Continuing with invalid configuration in development mode")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )