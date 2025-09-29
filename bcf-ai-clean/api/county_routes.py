from flask import Blueprint, request, jsonify
from services.ai_service import ai_service
from api.auth_routes import login_required
from utils.county_helpers import load_counties_from_file, get_state_name
import logging

logger = logging.getLogger(__name__)
county_bp = Blueprint('county', __name__)

@county_bp.route('/analyze-state', methods=['POST'])
@login_required
def analyze_state():
    try:
        data = request.get_json()
        state = data.get('state')
        project_type = data.get('project_type')

        if not state or not project_type:
            return jsonify({'success': False, 'error': 'Missing state or project_type'}), 400

        logger.info(f"Starting analysis: {state} for {project_type}")

        # Load counties from file
        counties = load_counties_from_file(state)
        if not counties:
            return jsonify({'success': False, 'error': f'No counties found for {state}'}), 500

        # Use AI service for analysis
        try:
            analysis_result = ai_service.analyze_state_counties(state, project_type, counties)
            if analysis_result:
                return jsonify({
                    'success': True,
                    'analysis': {
                        'state': state,
                        'project_type': project_type,
                        'counties': analysis_result.get('county_rankings', []),
                        'analysis_summary': analysis_result.get('analysis_summary', ''),
                        'total_counties': len(analysis_result.get('county_rankings', [])),
                        'ai_powered': True,
                        'data_sources': ['AI Analysis', 'County Database']
                    }
                })
        except Exception as ai_error:
            logger.error(f"AI analysis failed: {ai_error}")
            # Will fall through to fallback

        # Fallback analysis
        logger.info("Using fallback analysis")
        fallback_result = ai_service._create_fallback_analysis(state, project_type, counties)
        
        return jsonify({
            'success': True,
            'analysis': {
                'state': state,
                'project_type': project_type,
                'counties': fallback_result.get('county_rankings', []),
                'analysis_summary': fallback_result.get('analysis_summary', ''),
                'total_counties': len(fallback_result.get('county_rankings', [])),
                'ai_powered': False,
                'data_sources': ['Fallback Analysis']
            }
        })

    except Exception as e:
        logger.error(f"Analysis route error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@county_bp.route('/market-analysis', methods=['POST'])
@login_required
def market_analysis():
    try:
        data = request.get_json()
        county_fips = data.get('county_fips')
        county_name = data.get('county_name')
        state = data.get('state')
        project_type = data.get('project_type', 'solar')

        if not county_name or not state:
            return jsonify({'success': False, 'error': 'county_name and state are required'}), 400

        logger.info(f"Running AI county market analysis for {county_name}, {state}")

        # Use AI service
        ai_result = ai_service.analyze_county_market(county_name, state, project_type, county_fips)

        if ai_result.success:
            return jsonify({
                'success': True,
                'analysis': ai_result.content,
                'county_name': county_name,
                'state': state,
                'project_type': project_type,
                'analysis_type': 'AI-Powered County Market Analysis',
                'metadata': ai_result.metadata
            })
        else:
            # Fallback analysis
            fallback = f"Market analysis for {county_name} County, {state} could not be completed with AI service. Error: {ai_result.error}"
            return jsonify({
                'success': True,
                'analysis': fallback,
                'analysis_type': 'Fallback Analysis - AI Analysis Failed',
                'error_note': ai_result.error
            })

    except Exception as e:
        logger.error(f"County market analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500