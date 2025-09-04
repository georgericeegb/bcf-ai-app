# Debug version to trace the three missing fields

import logging
import json

logger = logging.getLogger(__name__)


class CRMFieldDebugger:
    """Debugging utility to trace the three critical fields"""

    def __init__(self, crm_service):
        self.crm_service = crm_service
        self.critical_fields = ['avg_slope', 'transmission_distance', 'transmission_voltage']
        self.debug_log = []

    def debug_parcel_data(self, parcel, parcel_index=0):
        """Comprehensive debugging of a single parcel's data"""
        parcel_id = parcel.get('parcel_id', f'Parcel_{parcel_index}')

        debug_result = {
            'parcel_id': parcel_id,
            'parcel_index': parcel_index,
            'raw_parcel_structure': self._analyze_parcel_structure(parcel),
            'critical_fields_analysis': {},
            'all_available_keys': list(parcel.keys()) if isinstance(parcel, dict) else [],
            'nested_objects_found': []
        }

        print(f"\n{'=' * 60}")
        print(f"DEBUGGING PARCEL: {parcel_id}")
        print(f"{'=' * 60}")

        # Check the raw parcel structure
        print(f"\n1. RAW PARCEL STRUCTURE:")
        print(f"   Type: {type(parcel)}")
        print(f"   Total keys: {len(parcel.keys()) if isinstance(parcel, dict) else 0}")

        # Show all top-level keys
        if isinstance(parcel, dict):
            print(f"   Top-level keys: {sorted(parcel.keys())}")

            # Check for nested objects that might contain our data
            nested_objects = []
            for key, value in parcel.items():
                if isinstance(value, dict):
                    nested_objects.append((key, list(value.keys())))

            if nested_objects:
                print(f"\n   NESTED OBJECTS FOUND:")
                for obj_name, obj_keys in nested_objects:
                    print(f"     {obj_name}: {obj_keys}")
                    debug_result['nested_objects_found'].append({
                        'name': obj_name,
                        'keys': obj_keys
                    })

        # Now test each critical field
        print(f"\n2. CRITICAL FIELDS ANALYSIS:")

        for field_name in self.critical_fields:
            field_result = self._debug_single_field(parcel, field_name)
            debug_result['critical_fields_analysis'][field_name] = field_result

            print(f"\n   {field_name.upper()}:")
            print(f"     Monday field: {field_result['monday_field']}")
            print(f"     Direct search result: {field_result['direct_search']}")
            print(f"     Nested search results: {len(field_result['nested_search_results'])}")

            if field_result['nested_search_results']:
                for location, value in field_result['nested_search_results']:
                    print(f"       Found in {location}: {value}")

            print(f"     Final extracted value: {field_result['extracted_value']}")
            print(f"     Formatted for CRM: {field_result['formatted_value']}")

            if field_result['extracted_value'] is None:
                print(f"     ❌ MISSING: No value found for {field_name}")
            else:
                print(f"     ✅ FOUND: {field_name} = {field_result['extracted_value']}")

        return debug_result

    def _analyze_parcel_structure(self, parcel):
        """Analyze the structure of the parcel data"""
        if not isinstance(parcel, dict):
            return {'type': str(type(parcel)), 'is_dict': False}

        structure = {
            'type': 'dict',
            'is_dict': True,
            'total_keys': len(parcel.keys()),
            'keys_by_type': {},
            'potential_analysis_objects': []
        }

        # Categorize keys by value type
        for key, value in parcel.items():
            value_type = str(type(value).__name__)
            if value_type not in structure['keys_by_type']:
                structure['keys_by_type'][value_type] = []
            structure['keys_by_type'][value_type].append(key)

            # Look for objects that might contain analysis results
            if isinstance(value, dict) and any(analysis_word in key.lower() for analysis_word in
                                               ['analysis', 'suitability', 'ml', 'score', 'transmission', 'slope',
                                                'terrain']):
                structure['potential_analysis_objects'].append({
                    'key': key,
                    'sub_keys': list(value.keys())
                })

        return structure

    def _debug_single_field(self, parcel, field_name):
        """Debug extraction of a single field"""
        monday_field = self.crm_service.crm_field_mapping.get(field_name, 'UNKNOWN')

        result = {
            'field_name': field_name,
            'monday_field': monday_field,
            'field_variations': self.crm_service.field_variations.get(field_name, []),
            'direct_search': None,
            'nested_search_results': [],
            'extracted_value': None,
            'formatted_value': None,
            'extraction_method_used': None
        }

        # Method 1: Direct field search
        direct_value = self.crm_service.find_field_value(parcel, field_name)
        result['direct_search'] = direct_value

        if direct_value is not None:
            result['extraction_method_used'] = 'direct_search'
            result['extracted_value'] = direct_value

        # Method 2: Check all nested objects
        if isinstance(parcel, dict):
            nested_objects = [
                ('suitability_analysis', parcel.get('suitability_analysis', {})),
                ('ml_analysis', parcel.get('ml_analysis', {})),
                ('analysis_results', parcel.get('analysis_results', {})),
                ('terrain_analysis', parcel.get('terrain_analysis', {})),
                ('topographic_data', parcel.get('topographic_data', {})),
                ('transmission_analysis', parcel.get('transmission_analysis', {})),
                ('grid_analysis', parcel.get('grid_analysis', {})),
                ('infrastructure_data', parcel.get('infrastructure_data', {}))
            ]

            for obj_name, obj_data in nested_objects:
                if isinstance(obj_data, dict):
                    for variation in result['field_variations']:
                        if variation in obj_data:
                            value = obj_data[variation]
                            if self.crm_service.is_valid_value(value, field_name):
                                result['nested_search_results'].append((f"{obj_name}.{variation}", value))
                                if result['extracted_value'] is None:
                                    result['extraction_method_used'] = f'nested_{obj_name}'
                                    result['extracted_value'] = value

        # Method 3: Use the specific extraction methods
        if result['extracted_value'] is None:
            if field_name == 'avg_slope':
                extracted = self.crm_service._extract_slope_score(parcel)
            elif field_name == 'transmission_distance':
                extracted = self.crm_service._extract_transmission_distance(parcel)
            elif field_name == 'transmission_voltage':
                extracted = self.crm_service._extract_transmission_voltage(parcel)
            else:
                extracted = None

            if extracted is not None:
                result['extraction_method_used'] = 'dedicated_method'
                result['extracted_value'] = extracted

        # Format the value
        if result['extracted_value'] is not None:
            result['formatted_value'] = self.crm_service.format_field_value(
                field_name, result['extracted_value'], monday_field
            )

        return result

    def debug_full_parcel_batch(self, parcels, max_parcels_to_debug=3):
        """Debug a batch of parcels"""
        print(f"\n{'=' * 80}")
        print(f"DEBUGGING PARCEL BATCH ({len(parcels)} parcels total)")
        print(f"Will debug first {min(max_parcels_to_debug, len(parcels))} parcels in detail")
        print(f"{'=' * 80}")

        batch_summary = {
            'total_parcels': len(parcels),
            'debugged_parcels': 0,
            'field_success_rates': {},
            'detailed_results': []
        }

        # Initialize success counters
        for field in self.critical_fields:
            batch_summary['field_success_rates'][field] = {
                'found': 0,
                'missing': 0,
                'percentage': 0
            }

        # Debug individual parcels in detail
        for i, parcel in enumerate(parcels[:max_parcels_to_debug]):
            parcel_debug = self.debug_parcel_data(parcel, i)
            batch_summary['detailed_results'].append(parcel_debug)
            batch_summary['debugged_parcels'] += 1

        # Quick scan of all parcels for field availability
        print(f"\n{'=' * 60}")
        print(f"QUICK SCAN OF ALL {len(parcels)} PARCELS")
        print(f"{'=' * 60}")

        for i, parcel in enumerate(parcels):
            for field_name in self.critical_fields:
                if field_name == 'avg_slope':
                    value = self.crm_service._extract_slope_score(parcel)
                elif field_name == 'transmission_distance':
                    value = self.crm_service._extract_transmission_distance(parcel)
                elif field_name == 'transmission_voltage':
                    value = self.crm_service._extract_transmission_voltage(parcel)
                else:
                    value = self.crm_service.find_field_value(parcel, field_name)

                if value is not None:
                    batch_summary['field_success_rates'][field_name]['found'] += 1
                else:
                    batch_summary['field_success_rates'][field_name]['missing'] += 1

        # Calculate percentages
        for field_name in self.critical_fields:
            total = batch_summary['field_success_rates'][field_name]['found'] + \
                    batch_summary['field_success_rates'][field_name]['missing']
            if total > 0:
                percentage = (batch_summary['field_success_rates'][field_name]['found'] / total) * 100
                batch_summary['field_success_rates'][field_name]['percentage'] = round(percentage, 1)

        print(f"\nFIELD AVAILABILITY SUMMARY:")
        for field_name in self.critical_fields:
            stats = batch_summary['field_success_rates'][field_name]
            print(f"  {field_name}: {stats['found']}/{len(parcels)} parcels ({stats['percentage']}%)")

        return batch_summary

    def debug_sample_parcel_json(self, parcel, filename=None):
        """Export a sample parcel's JSON structure for analysis"""
        if filename is None:
            filename = f"sample_parcel_{parcel.get('parcel_id', 'unknown')}.json"

        # Create a clean version for JSON export
        clean_parcel = {}

        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)  # Convert complex objects to string
            else:
                try:
                    json.dumps(obj)  # Test if it's JSON serializable
                    return obj
                except:
                    return str(obj)

        clean_parcel = clean_for_json(parcel)

        try:
            with open(filename, 'w') as f:
                json.dump(clean_parcel, f, indent=2, default=str)
            print(f"Sample parcel JSON exported to: {filename}")
        except Exception as e:
            print(f"Failed to export JSON: {e}")
            print(f"Parcel keys: {list(parcel.keys()) if isinstance(parcel, dict) else 'Not a dict'}")

        return clean_parcel


# Enhanced CRM service with debugging hooks
def debug_crm_export(crm_service, parcels, project_type, location):
    """Debug version of CRM export that traces the three critical fields"""

    debugger = CRMFieldDebugger(crm_service)

    print(f"\n🔍 DEBUGGING CRM EXPORT")
    print(f"Project: {project_type} in {location}")
    print(f"Parcels to process: {len(parcels)}")

    if not parcels:
        print("❌ No parcels provided for debugging")
        return None

    # Debug the parcel batch
    batch_debug = debugger.debug_full_parcel_batch(parcels, max_parcels_to_debug=2)

    # Export sample parcel structure
    if parcels:
        sample_parcel = debugger.debug_sample_parcel_json(parcels[0])

    print(f"\n📋 DEBUGGING SUMMARY:")
    print(f"Total parcels analyzed: {len(parcels)}")

    missing_fields = []
    for field_name in debugger.critical_fields:
        stats = batch_debug['field_success_rates'][field_name]
        if stats['found'] == 0:
            missing_fields.append(field_name)
            print(f"❌ {field_name}: COMPLETELY MISSING from all parcels")
        elif stats['percentage'] < 50:
            print(f"⚠️  {field_name}: Only found in {stats['percentage']}% of parcels")
        else:
            print(f"✅ {field_name}: Found in {stats['percentage']}% of parcels")

    if missing_fields:
        print(f"\n🚨 CRITICAL ISSUE: These fields are missing from ALL parcels:")
        for field in missing_fields:
            print(f"   - {field}")
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Check if your slope/transmission modules are actually being called")
        print(f"   2. Verify the modules are adding data to the parcel objects")
        print(f"   3. Check the field names they're using")

    return batch_debug


# Usage example function
def test_critical_fields_extraction():
    """Test function to verify the three fields are being extracted"""

    # Sample parcel data structure - modify this to match your actual data
    sample_parcel = {
        'parcel_id': 'TEST123',
        'owner': 'Test Owner',
        'acreage_calc': 25.5,
        'latitude': 35.7796,
        'longitude': -78.6382,

        # These might be where your modules put the data:
        'suitability_analysis': {
            'slope_degrees': 5.2,
            'transmission_distance': 1.5,
            'transmission_voltage': 230
        },

        # Or they might be at the top level:
        'avg_slope': 5.2,
        'transmission_distance': 1.5,
        'transmission_voltage': 230,

        # Or in ML analysis results:
        'ml_analysis': {
            'slope_degrees': 5.2,
            'nearest_transmission_line': 1.5,
            'tx_voltage': 230
        }
    }

    # Test with your CRM service
    from your_crm_service import CRMService  # Adjust import as needed
    crm_service = CRMService()

    debugger = CRMFieldDebugger(crm_service)
    result = debugger.debug_parcel_data(sample_parcel)

    return result