# api/crm_routes.py
import logging
from flask import Blueprint, request, jsonify
from services.crm_service import CRMService

logger = logging.getLogger(__name__)

crm_bp = Blueprint('crm', __name__)

@crm_bp.route('/export-to-crm', methods=['POST'])
def export_to_crm():
    """Export selected parcels to Monday.com CRM"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['parcels', 'project_type', 'location']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        parcels = data['parcels']
        project_type = data['project_type']
        location = data['location']
        
        logger.info(f"Exporting {len(parcels)} parcels to CRM for {location}")
        
        # Initialize CRM service
        crm_service = CRMService()
        
        # Test connection first
        connection_test = crm_service.test_connection()
        if not connection_test['success']:
            return jsonify({
                'success': False,
                'error': f'CRM connection failed: {connection_test["error"]}'
            }), 500
        
        # Export parcels to CRM
        result = crm_service.export_parcels_to_crm(parcels, project_type, location)
        
        if result['success']:
            logger.info(f"✅ Successfully exported {result['successful_exports']} parcels to CRM")
            return jsonify({
                'success': True,
                'message': f'Successfully exported {result["successful_exports"]} parcels to CRM',
                'group_name': result['group_name'],
                'group_id': result['group_id'],
                'total_parcels': result['total_parcels'],
                'successful_exports': result['successful_exports'],
                'failed_exports': result['failed_exports'],
                'critical_field_rates': result.get('critical_field_success_rates', {}),
                'export_details': result.get('export_details', [])
            }), 200
        else:
            logger.error(f"❌ CRM export failed: {result.get('error')}")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error occurred')
            }), 500
            
    except Exception as e:
        logger.error(f"💥 CRM export error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@crm_bp.route('/test-connection', methods=['GET'])
def test_crm_connection():
    """Test Monday.com CRM connection"""
    try:
        crm_service = CRMService()
        result = crm_service.test_connection()
        return jsonify(result), 200 if result['success'] else 500
    except Exception as e:
        logger.error(f"CRM connection test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500