# services/crm_service.py - FIXED: Complete field mapping coverage

import os
import json
import requests
import time
import math
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CRMService:
    def __init__(self):
        self.api_url = os.getenv('MONDAY_API_URL', 'https://api.monday.com/v2')
        self.api_key = os.getenv('MONDAY_API_KEY')
        self.board_id = os.getenv('MONDAY_BOARD_ID')

        if not self.api_key:
            raise ValueError("MONDAY_API_KEY not found in environment variables")
        if not self.board_id:
            raise ValueError("MONDAY_BOARD_ID not found in environment variables")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "API-Version": "2023-10"
        }

        # COMPLETE field variations - covering ALL crm_field_mapping keys
        self.field_variations = {
            'owner': [
                'owner', 'owner_name', 'owner1', 'ownername', 'owner_nam', 'owner_full',
                'property_owner', 'landowner', 'deed_holder', 'title_holder'
            ],
            'county_id': [
                'county_id', 'cnty_id', 'fips', 'county_fips', 'fips_code', 'county_code',
                'cnty_fips', 'co_id', 'county_num'
            ],
            'county_name': [
                'county_name', 'county_nam', 'countyname', 'county', 'cnty_name', 'co_name',
                'county_desc', 'cnty', 'county_full'
            ],
            'state_abbr': [
                'state_abbr', 'state', 'st', 'state_abb', 'state_code', 'state_cd',
                'st_abbr', 'state_name', 'statename'
            ],
            'address': [
                'address', 'mail_address', 'address1', 'addr', 'addr1', 'prop_address',
                'property_address', 'site_address', 'location', 'street_address',
                'physical_address', 'addr_full', 'full_address', 'location_address'
            ],
            'muni_name': [
                'muni_name', 'municipality', 'muni', 'municipal', 'muni_nam', 'city',
                'township', 'village', 'borough', 'town', 'place_name', 'locality'
            ],
            'census_zip': [
                'census_zip', 'zip', 'zipcode', 'zip_code', 'postal_code', 'zip5',
                'mail_zip', 'zip_cd', 'zipcd', 'postcode'
            ],
            'mkt_val_land': [
                'mkt_val_land', 'market_value_land', 'land_value', 'mkt_val_la', 'market_val',
                'land_market_value', 'assessed_land', 'land_assessed', 'land_val',
                'mkt_land', 'market_land', 'appraised_land'
            ],
            'land_use_code': [
                'land_use_code', 'land_use_c', 'use_code', 'landuse', 'land_use',
                'use_cd', 'property_use', 'zoning', 'zone_code', 'use_type'
            ],
            'mail_address1': [
                'mail_address1', 'mail_addre', 'mail_address', 'mail_addr', 'mail_add1',
                'mailing_address', 'billing_address', 'owner_address', 'mail_line1'
            ],
            'mail_placename': [
                'mail_placename', 'mail_place', 'mail_city', 'mail_plac', 'owner_city',
                'billing_city', 'mail_municipality', 'mail_town'
            ],
            'mail_statename': [
                'mail_statename', 'mail_state', 'mail_st', 'mail_stat', 'owner_state',
                'billing_state', 'mail_state_name', 'mail_state_abbr'
            ],
            'mail_zipcode': [
                'mail_zipcode', 'mail_zipco', 'mail_zip', 'mail_zip_c', 'owner_zip',
                'billing_zip', 'mail_postal', 'mail_zip_code'
            ],
            'parcel_id': [
                'parcel_id', 'pin', 'apn', 'parcel_pin', 'parcel_num', 'parcel_number',
                'assessor_parcel_number', 'tax_id', 'property_id', 'map_parcel',
                'parcel_key', 'unique_id', 'id'
            ],
            'acreage_calc': [
                'acreage_calc', 'acreage_ca', 'acres_calc', 'acreage', 'acres', 'calc_acres',
                'calculated_acres', 'total_acres', 'parcel_acres', 'area_acres',
                'size_acres', 'acre', 'acreage_total'
            ],
            'acreage_adjacent_with_sameowner': [
                'acreage_adjacent_with_sameowner', 'acreage_ad', 'adj_acres', 'adjacent_acres',
                'adjacent_acreage', 'contiguous_acres', 'same_owner_acres', 'connected_acres'
            ],
            'latitude': [
                'latitude', 'lat', 'y', 'coord_y', 'y_coord', 'lat_dd', 'latitude_dd',
                'centroid_y', 'center_lat', 'y_coordinate'
            ],
            'longitude': [
                'longitude', 'long', 'lon', 'x', 'coord_x', 'x_coord', 'lon_dd', 'longitude_dd',
                'centroid_x', 'center_lon', 'x_coordinate'
            ],
            # FIXED: 'evalation' -> 'elevation' (correcting the typo)
            'elevation': [
                'elevation', 'elev', 'elevatio', 'altitude', 'height',
                'elevation_ft', 'elev_ft', 'ground_elevation', 'evalation'  # Include the typo as fallback
            ],
            'legal_desc1': [
                'legal_desc1', 'legal_description', 'legal_desc', 'legal_des', 'legal',
                'deed_description', 'metes_bounds', 'legal_text', 'description'
            ],
            'land_cover': [
                'land_cover', 'landcover', 'cover', 'land_cove', 'vegetation',
                'cover_type', 'land_type', 'surface_cover', 'nlcd'
            ],
            'county_link': [
                'county_link', 'link', 'web_link', 'url', 'website', 'online_link',
                'assessor_link', 'property_link', 'record_link'
            ],
            'fld_zone': [
                'fld_zone', 'flood_zone', 'fema_zone', 'flood_zon', 'floodzone',
                'flood_hazard', 'fema_flood', 'flood_risk', 'hazard_zone'
            ],
            'zone_subty': [
                'zone_subty', 'zone_subtype', 'subtype', 'zone_subt', 'flood_subtype',
                'hazard_subtype', 'zone_detail', 'flood_detail'
            ],
            # Additional analysis fields
            'solar_score': [
                'solar_score', 'overall_score', 'total_score', 'suitability_score'
            ],
            'wind_score': [
                'wind_score', 'slope_score'
            ],
            'battery_score': [
                'battery_score', 'transmission_score'
            ],
            'avg_slope_degrees': [
                'avg_slope_degrees', 'slope_degrees', 'slope'
            ],
            'miles_from_transmission': [
                'miles_from_transmission', 'transmission_distance', 'tx_distance'
            ],
            'nearest_transmission_voltage': [
                'nearest_transmission_voltage', 'transmission_voltage', 'tx_voltage'
            ],
            'avg_slope': [
                'avg_slope', 'slope_category', 'slope_class'
            ],
            'transmission_distance': [
                'transmission_distance', 'tx_distance', 'miles_from_transmission'
            ],
            'transmission_voltage': [
                'transmission_voltage', 'tx_voltage', 'nearest_transmission_voltage'
            ]
        }

        # Your existing CRM field mapping - keeping exactly as you have it
        self.crm_field_mapping = {
            'owner': 'item_name',
            'county_id': 'county_id__1',
            'county_name': 'text4',
            'state_abbr': 'text_1',
            'address': 'text1',
            'muni_name': 'text66',
            'census_zip': 'text_mktw4254',
            'mkt_val_land': 'numbers85__1',
            'land_use_code': 'land_use_code__1',
            'mail_address1': 'text7',
            'mail_placename': 'text49',
            'mail_statename': 'text11',
            'mail_zipcode': 'mzip',
            'parcel_id': 'text117',
            'acreage_calc': 'numbers6',
            'acreage_adjacent_with_sameowner': 'dup__of_score__0___3___1',
            'latitude': 'latitude__1',
            'longitude': 'longitude__1',
            'solar_score': 'numeric_mknpptf4',
            'wind_score': 'numeric_mknphdv8',
            'battery_score': 'numeric_mknpp74r',
            'avg_slope_degrees': 'numeric_mktx3jgs',
            'miles_from_transmission': 'numbers66__1',
            'nearest_transmission_voltage': 'numbers46__1',
            'elevation': 'numeric_mktwrwry',  # FIXED: changed from 'evalation' to 'elevation'
            'legal_desc1': 'text_mktw1gns',
            'land_cover': 'long_text__1',
            'county_link': 'text_mktw6bvk',
            'fld_zone': 'text_mkkbx2zc',
            'zone_subty': 'text_mktwy6h5',
            'avg_slope': 'dropdown_mkkzj3m8',
            'transmission_distance': 'numbers66__1',
            'transmission_voltage': 'numbers46__1'
        }

    def is_valid_value(self, value):
        """Check if value is valid for CRM"""
        if value is None:
            return False

        # String checks
        str_val = str(value).strip().lower()
        if str_val in ['', 'null', 'none', 'nan', 'n/a']:
            return False

        # Numeric NaN check
        try:
            if isinstance(value, (int, float)) and math.isnan(float(value)):
                return False
        except (ValueError, TypeError, OverflowError):
            pass

        return True

    def find_field_value(self, parcel, field_key):
        """Find value using field variations with comprehensive logging"""
        variations = self.field_variations.get(field_key, [field_key])

        logger.debug(f"Looking for field '{field_key}' in variations: {variations}")

        for variation in variations:
            if variation in parcel:
                value = parcel[variation]
                if self.is_valid_value(value):
                    logger.debug(f"✓ Found value for '{field_key}' in field '{variation}': {value}")
                    return value
                else:
                    logger.debug(f"✗ Found field '{variation}' but value is invalid: {value}")

        logger.debug(f"✗ No valid value found for field '{field_key}' in any variation")
        return None

    def format_field_value(self, field_key, value, monday_field):
        """Format field values for Monday.com with comprehensive type handling"""
        try:
            if not self.is_valid_value(value):
                return None

            logger.debug(f"Formatting field {field_key} (value: {value}) for Monday field {monday_field}")

            # Handle different Monday.com field types
            if monday_field in ['latitude__1', 'longitude__1']:
                # Coordinates as strings
                try:
                    coord_value = float(value)
                    if coord_value == 0:  # Allow zero coordinates but not NaN
                        return None
                    return str(coord_value)
                except (ValueError, TypeError, OverflowError):
                    return None

            elif monday_field == 'county_id__1':
                # County FIPS code - ensure 5 digits
                try:
                    return str(int(float(value))).zfill(5)
                except (ValueError, TypeError, OverflowError):
                    return str(value).zfill(5) if len(str(value)) <= 5 else str(value)[:5]

            elif monday_field in ['numbers6', 'dup__of_score__0___3___1']:  # acreage fields
                # Integer acreage values
                try:
                    acreage = float(value)
                    if acreage < 0:  # Allow zero acreage
                        return None
                    return int(acreage)
                except (ValueError, TypeError, OverflowError):
                    return None

            elif monday_field in ['numbers85__1', 'numeric_mktwrwry']:  # money and elevation fields
                # Numeric fields with decimals
                try:
                    numeric_value = float(value)
                    return round(numeric_value, 2)  # Allow zero values
                except (ValueError, TypeError, OverflowError):
                    return None

            elif monday_field in ['numbers66__1', 'numbers46__1']:  # transmission fields
                # Transmission distance and voltage fields
                try:
                    numeric_value = float(value)
                    if numeric_value < 0:
                        return None
                    return round(numeric_value, 3)
                except (ValueError, TypeError, OverflowError):
                    return None

            elif monday_field in ['numeric_mknpptf4', 'numeric_mknphdv8', 'numeric_mknpp74r', 'numeric_mktx3jgs']:
                # Analysis score fields
                try:
                    score_value = float(value)
                    if score_value < 0:
                        return None
                    return round(score_value, 2)
                except (ValueError, TypeError, OverflowError):
                    return None

            elif monday_field in ['text_mktw4254', 'mzip']:  # zipcode fields
                # ZIP code formatting
                if str(value) == '0':  # Handle explicit zero
                    return '00000'
                try:
                    zip_value = str(int(float(value)))
                    return zip_value.zfill(5)
                except (ValueError, TypeError, OverflowError):
                    # Handle non-numeric zip codes
                    zip_str = str(value).strip()
                    if len(zip_str) > 0:
                        return zip_str[:10]  # Limit length
                    return None

            elif monday_field == 'land_use_code__1':
                # Land use code - as string, allow numbers
                return str(value).strip()

            elif monday_field == 'long_text__1':  # land_cover JSON field
                # Handle JSON land cover data
                try:
                    if isinstance(value, str) and value.startswith('{'):
                        # Try to parse and reformat JSON
                        import json
                        parsed = json.loads(value.replace("'", '"'))
                        return json.dumps(parsed)
                    else:
                        return str(value)
                except:
                    return str(value)

            elif monday_field == 'dropdown_mkkzj3m8':  # avg_slope dropdown
                # Handle dropdown fields
                return str(value).strip()

            elif monday_field in ['text7', 'text49', 'text11']:  # mail address fields
                # Mail address fields - clean but preserve case
                cleaned = str(value).strip()
                return cleaned[:100] if len(cleaned) > 0 else None

            elif monday_field in ['text1', 'text4', 'text66', 'text117', 'text_1']:  # Standard text fields
                # Standard text fields - clean and limit length
                cleaned = str(value).strip()
                return cleaned[:255] if len(cleaned) > 0 else None

            elif monday_field in ['text_mktw1gns', 'text_mktw6bvk', 'text_mkkbx2zc', 'text_mktwy6h5']:
                # Special text fields (legal, links, flood zones)
                cleaned = str(value).strip()
                return cleaned[:1000] if len(cleaned) > 0 else None

            else:
                # Default string handling
                cleaned = str(value).strip()
                return cleaned[:255] if len(cleaned) > 0 else None

        except Exception as e:
            logger.error(f"Error formatting field {field_key} with value {value}: {e}")
            return None

    def prepare_parcel_for_crm(self, parcel, project_type):
        """FIXED: Process ALL fields in crm_field_mapping"""
        values = {}
        processing_stats = {'found': 0, 'missing': 0, 'formatted': 0, 'rejected': 0}

        logger.info(f"Processing parcel {parcel.get('parcel_id', 'Unknown')} for CRM")
        logger.debug(f"Available parcel fields: {list(parcel.keys())}")

        # Process ALL fields in the CRM mapping (no exceptions)
        for field_key, monday_field in self.crm_field_mapping.items():
            try:
                # Skip the owner field as it goes in item_name, not column_values
                if field_key == 'owner':
                    continue

                # Find the value using field variations
                raw_value = self.find_field_value(parcel, field_key)

                if raw_value is not None:
                    processing_stats['found'] += 1

                    # Format the value
                    formatted_value = self.format_field_value(field_key, raw_value, monday_field)

                    if formatted_value is not None:
                        processing_stats['formatted'] += 1
                        values[monday_field] = formatted_value
                        logger.debug(f"✓ Mapped {field_key} -> {monday_field}: {formatted_value}")
                    else:
                        processing_stats['rejected'] += 1
                        logger.debug(f"✗ Rejected {field_key}: {raw_value} (formatting failed)")
                else:
                    processing_stats['missing'] += 1
                    logger.debug(f"- Missing {field_key}")

            except Exception as e:
                processing_stats['rejected'] += 1
                logger.error(f"Error processing field {field_key}: {e}")

        # Handle suitability analysis scores (same as before but with better logging)
        if parcel.get('suitability_analysis'):
            analysis = parcel['suitability_analysis']
            logger.debug("Processing suitability analysis scores")

            # Map analysis fields to CRM fields
            analysis_mappings = {
                'overall_score': 'numeric_mknpptf4',  # solar_score
                'slope_score': 'numeric_mknphdv8',  # wind_score
                'transmission_score': 'numeric_mknpp74r',  # battery_score
                'slope_degrees': 'numeric_mktx3jgs',  # avg_slope_degrees
                'transmission_distance': 'numbers66__1',  # miles_from_transmission
                'transmission_voltage': 'numbers46__1'  # nearest_transmission_voltage
            }

            for analysis_field, monday_field in analysis_mappings.items():
                if analysis_field in analysis:
                    raw_value = analysis[analysis_field]
                    formatted_value = self.format_field_value(f"analysis_{analysis_field}", raw_value, monday_field)

                    if formatted_value is not None:
                        values[monday_field] = formatted_value
                        processing_stats['formatted'] += 1
                        logger.debug(f"✓ Analysis {analysis_field} -> {monday_field}: {formatted_value}")
                    else:
                        processing_stats['rejected'] += 1
                        logger.debug(f"✗ Analysis {analysis_field}: {raw_value} (rejected)")

        logger.info(f"CRM field processing complete: {processing_stats}")
        logger.info(f"Generated {len(values)} CRM field values for parcel {parcel.get('parcel_id', 'Unknown')}")

        return values

    def proper_case_with_exceptions(self, name):
        """Convert text to proper case while preserving special cases."""
        if not name:
            return "Unknown Owner"
        special_cases = ['LLC', 'Inc', 'Corp', 'Ltd', 'LLP', 'PA', 'OF', 'AND', 'THE']
        words = str(name).split()
        formatted_words = []

        for word in words:
            if word.upper() in [case.upper() for case in special_cases]:
                formatted_words.append(word.upper())
            else:
                formatted_words.append(word.capitalize())

        return ' '.join(formatted_words)

    def generate_group_name(self, location, project_type):
        """Generate a standardized group name with today's date."""
        return f"{location} - {project_type.title()} Project - {datetime.now().strftime('%Y-%m-%d')}"

    def create_group_in_board(self, group_name):
        """Create a new group in the Monday.com board."""
        mutation = {
            "query": """
            mutation ($boardId: ID!, $groupName: String!) {
                create_group(board_id: $boardId, group_name: $groupName) {
                    id
                    title
                }
            }
            """,
            "variables": {
                "boardId": self.board_id,
                "groupName": group_name
            }
        }

        try:
            response = requests.post(self.api_url, json=mutation, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'create_group' in result['data']:
                return result['data']['create_group']['id']
            else:
                logger.error(f"Unexpected response when creating group: {result}")
                return None

        except Exception as e:
            logger.error(f"Error creating group: {e}")
            return None

    def export_parcels_to_crm(self, parcels, project_type, location):
        """Export selected parcels to Monday.com CRM"""
        try:
            if not parcels:
                return {'success': False, 'error': 'No parcels provided'}

            logger.info(f"Starting CRM export of {len(parcels)} parcels")

            # Create group for this export
            group_name = self.generate_group_name(location, project_type)
            logger.info(f"Creating CRM group: {group_name}")

            group_id = self.create_group_in_board(group_name)
            if not group_id:
                return {'success': False, 'error': 'Failed to create group in CRM'}

            successful_exports = 0
            failed_exports = 0
            export_details = []

            for i, parcel in enumerate(parcels):
                try:
                    parcel_id = parcel.get('parcel_id', f'Parcel_{i + 1}')
                    logger.info(f"Processing parcel {i + 1}/{len(parcels)}: {parcel_id}")

                    # Prepare parcel data for CRM
                    crm_values = self.prepare_parcel_for_crm(parcel, project_type)

                    if len(crm_values) == 0:
                        logger.warning(f"No CRM values generated for parcel {parcel_id}")

                    # Owner name for the item
                    owner_name = self.proper_case_with_exceptions(parcel.get('owner', 'Unknown Owner'))

                    # Create item in Monday.com
                    success = self.create_crm_item(group_id, owner_name, crm_values)

                    if success:
                        successful_exports += 1
                        export_details.append({
                            'parcel_id': parcel_id,
                            'owner': owner_name,
                            'status': 'success',
                            'fields_mapped': len(crm_values)
                        })
                        logger.info(f"Successfully exported parcel {parcel_id} with {len(crm_values)} fields")
                    else:
                        failed_exports += 1
                        export_details.append({
                            'parcel_id': parcel_id,
                            'owner': owner_name,
                            'status': 'failed',
                            'error': 'CRM creation failed'
                        })

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing parcel {i + 1}: {str(e)}")
                    failed_exports += 1
                    export_details.append({
                        'parcel_id': parcel.get('parcel_id', f'Parcel_{i + 1}'),
                        'owner': parcel.get('owner', 'Unknown'),
                        'status': 'failed',
                        'error': str(e)
                    })

            logger.info(f"CRM export completed: {successful_exports} successful, {failed_exports} failed")

            return {
                'success': True,
                'group_name': group_name,
                'group_id': group_id,
                'total_parcels': len(parcels),
                'successful_exports': successful_exports,
                'failed_exports': failed_exports,
                'export_details': export_details
            }

        except Exception as e:
            logger.error(f"Error during CRM export: {str(e)}")
            return {'success': False, 'error': str(e)}

    def create_crm_item(self, group_id, item_name, column_values):
        """Create a single item in Monday.com."""
        mutation = {
            "query": """
                mutation ($boardId: ID!, $groupId: String!, $itemName: String!, $columnValues: JSON!) {
                    create_item (
                        board_id: $boardId,
                        group_id: $groupId,
                        item_name: $itemName,
                        column_values: $columnValues
                    ) {
                        id
                        name
                    }
                }
            """,
            "variables": {
                "boardId": self.board_id,
                "groupId": group_id,
                "itemName": item_name,
                "columnValues": json.dumps(column_values)
            }
        }

        try:
            response = requests.post(self.api_url, json=mutation, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'create_item' in result['data']:
                logger.info(f"Successfully created CRM item: {item_name}")
                return True
            else:
                logger.error(f"Failed to create CRM item for {item_name}: {result}")
                return False

        except Exception as e:
            logger.error(f"Error creating CRM item for {item_name}: {str(e)}")
            return False

    def test_connection(self):
        """Test the Monday.com API connection."""
        query = {
            "query": "query { me { name email } }"
        }

        try:
            response = requests.post(self.api_url, json=query, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'me' in result['data']:
                return {'success': True, 'user': result['data']['me']}
            else:
                return {'success': False, 'error': 'Invalid API response'}

        except Exception as e:
            return {'success': False, 'error': str(e)}