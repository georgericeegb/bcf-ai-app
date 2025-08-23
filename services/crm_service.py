# services/crm_service.py - FIXED VERSION

import os
import json
import requests
import time
import math
from datetime import datetime
import logging
import pandas as pd
import numpy as np

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

        # FIXED: Enhanced field variations
        self.field_variations = {
            'owner': [
                'owner', 'owner_name', 'owner1', 'ownername', 'owner_nam', 'owner_full',
                'property_owner', 'landowner', 'deed_holder', 'title_holder'
            ],
            'county_id': [
                'county_id', 'cnty_id', 'fips', 'county_fips', 'fips_code', 'county_code',
                'cnty_fips', 'co_id', 'county_num', 'cty_row_id'
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
                'mail_zip', 'zip_cd', 'zipcd', 'postcode', 'addr_zip', 'addr_zipplusfour'
            ],
            'mkt_val_land': [
                'mkt_val_land', 'market_value_land', 'land_value', 'mkt_val_la', 'market_val',
                'land_market_value', 'assessed_land', 'land_assessed', 'land_val',
                'mkt_land', 'market_land', 'appraised_land'
            ],
            'land_use_code': [
                'land_use_code', 'land_use_c', 'use_code', 'landuse', 'land_use',
                'use_cd', 'property_use', 'zoning', 'zone_code', 'use_type', 'land_use_class'
            ],
            'mail_address1': [
                'mail_address1', 'mail_addre', 'mail_address', 'mail_addr', 'mail_add1',
                'mailing_address', 'billing_address', 'owner_address', 'mail_line1', 'mail_address3'
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
            'elevation': [
                'elevation', 'elev', 'elevatio', 'altitude', 'height',
                'elevation_ft', 'elev_ft', 'ground_elevation', 'evalation'
            ],
            'legal_desc1': [
                'legal_desc1', 'legal_description', 'legal_desc', 'legal_des', 'legal',
                'deed_description', 'metes_bounds', 'legal_text', 'description'
            ],
            'land_cover': [
                'land_cover', 'landcover', 'cover', 'land_cove', 'vegetation',
                'cover_type', 'land_type', 'surface_cover', 'nlcd', 'crop_cover'
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
            'avg_slope': ['avg_slope', 'slope_degrees', 'slope', 'avg_slope_degrees'],
            'transmission_distance': ['transmission_distance', 'tx_distance', 'tx_nearest_distance',
                                      'transmission_dist'],
            'transmission_voltage': ['transmission_voltage', 'tx_voltage', 'tx_max_voltage', 'voltage'],
            'ml_score': ['ml_score', 'predicted_score', 'ai_score'],
            'traditional_score': ['traditional_score', 'slope_score', 'conventional_score'],
            'combined_score': ['combined_score', 'overall_score', 'final_score'],
            'solar_score': ['solar_score', 'solar_suitability', 'pv_score'],
            'wind_score': ['wind_score', 'wind_suitability'],
            'battery_score': ['battery_score', 'storage_score', 'bess_score'],
            'legal_desc1': ['legal_desc1', 'legal_description', 'legal_desc', 'legal_des', 'legal']
        }

        self.crm_field_mapping = {
            'owner': 'item_name',  # This maps to the "Name" field
            'county_id': 'county_id__1',
            'county_name': 'text4',  # "Project County"
            'state_abbr': 'text_1',  # "Project State"
            'address': 'text1',  # "Site Address"
            'muni_name': 'text66',  # "Site City"
            'census_zip': 'text_mktw4254',  # "Site Zip"
            'mkt_val_land': 'numbers85__1',  # "mkt val land"
            'land_use_code': 'land_use_code__1',  # "land use code"
            'mail_address1': 'text7',  # "Mailing address"
            'mail_placename': 'text49',  # "MCity"
            'mail_statename': 'text11',  # "MState"
            'mail_zipcode': 'mzip',  # "MZip"
            'parcel_id': 'text117',  # "Parcel ID"
            'acreage_calc': 'numbers6',  # "Acres"
            'acreage_adjacent_with_sameowner': 'dup__of_score__0___3___1',  # "C-acres"
            'latitude': 'latitude__1',  # "Latitude"
            'longitude': 'longitude__1',  # "Longitude"
            'elevation': 'numeric_mktwrwry',  # "Elavation"
            'land_cover': 'long_text__1',  # "Land cover"
            'county_link': 'text_mktw6bvk',  # "County link"
            'fld_zone': 'text_mkkbx2zc',  # "Flood zone"
            'zone_subty': 'text_mktwy6h5',  # "Flood zone subtype"
            'legal_desc1': 'text_mktw1gns',  # "Site Legal Desc"

            # SCORING FIELDS - using your actual column IDs
            'avg_slope': 'numeric_mktx3jgs',  # "Avg Slope"
            'transmission_distance': 'numbers66__1',  # "Miles to closest tx"
            'transmission_voltage': 'numbers46__1',  # "kV of closest tx"
            'solar_score': 'numeric_mknpptf4',  # "Solar score"
            'wind_score': 'numeric_mknphdv8',  # "Wind score"
            'battery_score': 'numeric_mknpp74r',  # "Battery score"
            'ml_score': 'numeric_mkv240mj',  # Need to add "ML Score" column
            'traditional_score': 'numeric_mkv25w3h',  # Need to add "Traditional Score" column
            'combined_score': 'numeric_mkv2zqj5',  # Need to add "Combined Score" column
        }

    def safe_convert_to_string(self, value):
        """Safely convert any value to string"""
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            if pd.isna(value) or np.isnan(value) or np.isinf(value):
                return ""
            return str(value)
        return str(value).strip()

    def is_valid_value(self, value, field_key=None):
        """FIXED: More lenient validation"""
        if value is None:
            return False

        # Convert to string safely
        try:
            str_val = self.safe_convert_to_string(value).lower()
        except:
            return False

        # Invalid value patterns (basic only)
        if str_val in ['', 'nan', 'none', 'null', '#n/a', 'unknown']:
            return False

        # Check for pandas/numpy NaN more safely
        try:
            if pd.isna(value):
                return False
        except:
            pass

        # FIXED: Less strict coordinate validation
        if field_key in ['latitude', 'longitude']:
            try:
                coord_val = float(value)
                # Only reject if exactly zero or clearly invalid
                if coord_val == 0.0 or abs(coord_val) > 180:
                    return False
            except:
                return False

        # FIXED: Allow zero for numeric fields that might legitimately be zero
        if field_key in ['mkt_val_land', 'mkt_val_bldg', 'elevation', 'mail_zipcode']:
            # Don't reject zero values for these fields
            pass

        return len(str_val) > 0

    def find_field_value(self, parcel, field_key):
        """Find field value with comprehensive search"""
        variations = self.field_variations.get(field_key, [field_key])

        for variation in variations:
            if variation in parcel:
                value = parcel[variation]
                if self.is_valid_value(value, field_key):
                    return value

        return None

    def safe_format_number(self, value, allow_zero=True):
        """Safely format numeric values"""
        try:
            if pd.isna(value):
                return None

            num_val = float(value)

            # Check for invalid numbers
            if np.isnan(num_val) or np.isinf(num_val):
                return None

            # Check if zero is allowed
            if not allow_zero and num_val == 0:
                return None

            # Return integer if whole number, otherwise float
            if num_val == int(num_val):
                return int(num_val)
            else:
                return round(num_val, 6)  # 6 decimal places for coordinates

        except (ValueError, TypeError, OverflowError):
            return None

    def format_field_value(self, field_key, value, monday_field):
        """ENHANCED: Handle the new numeric scoring fields"""
        try:
            if not self.is_valid_value(value, field_key):
                return None

            # Handle the new numeric scoring fields
            if monday_field in ['numeric_mktx3jgs', 'numbers66__1', 'numbers46__1',
                                'numeric_mknpptf4', 'numeric_mknphdv8', 'numeric_mknpp74r']:
                # These are all numeric fields for scoring
                result = self.safe_format_number(value, allow_zero=True)
                if result is not None:
                    logger.debug(f"✅ Scoring field formatted: {field_key} = {result}")
                return result

            # Existing field handling logic
            elif monday_field in ['numbers6', 'dup__of_score__0___3___1', 'numbers85__1',
                                'numeric_mktwrwry']:
                result = self.safe_format_number(value, allow_zero=True)
                if result is not None:
                    logger.debug(f"✅ Number formatted: {field_key} = {result}")
                return result

            # Coordinate fields
            elif monday_field in ['latitude__1', 'longitude__1']:
                result = self.safe_format_number(value, allow_zero=False)
                if result is not None:
                    coord_str = str(result)
                    logger.debug(f"✅ Coordinate formatted: {field_key} = {coord_str}")
                    return coord_str
                return None

            # County ID
            elif monday_field == 'county_id__1':
                try:
                    if isinstance(value, str):
                        clean_value = ''.join(filter(str.isdigit, value))
                        if len(clean_value) >= 2:
                            result = clean_value.zfill(5)
                            logger.debug(f"✅ County ID formatted: {result}")
                            return result
                    else:
                        num_val = self.safe_format_number(value, allow_zero=False)
                        if num_val is not None:
                            result = str(num_val).zfill(5)
                            logger.debug(f"✅ County ID formatted: {result}")
                            return result
                except:
                    pass
                return None

            # ZIP codes
            elif monday_field in ['text_mktw4254', 'mzip']:
                try:
                    str_val = self.safe_convert_to_string(value)
                    if str_val.isdigit():
                        zip_num = int(str_val)
                        if zip_num == 0:
                            return "00000"
                        else:
                            return str(zip_num).zfill(5)
                    elif len(str_val) > 0:
                        digits_only = ''.join(filter(str.isdigit, str_val))
                        if len(digits_only) >= 3:
                            return digits_only.zfill(5)
                except:
                    pass
                return None

            # Long text fields
            elif monday_field == 'long_text__1':
                str_val = self.safe_convert_to_string(value)
                if len(str_val) > 0 and str_val not in ['{}', 'nan', 'null']:
                    result = str_val[:5000]
                    logger.debug(f"✅ Long text formatted: {field_key} = {result[:50]}...")
                    return result
                return None

            # Regular text fields
            else:
                str_val = self.safe_convert_to_string(value)
                if len(str_val) > 0 and str_val.lower() not in ['nan', 'null', 'none']:
                    max_length = 2000 if monday_field == 'text_mktw1gns' else 255
                    result = str_val[:max_length]
                    logger.debug(f"✅ Text formatted: {field_key} = {result[:50]}...")
                    return result
                return None

        except Exception as e:
            logger.error(f"❌ Formatting error for {field_key} = {repr(value)}: {e}")
            return None

    def prepare_parcel_for_crm(self, parcel, project_type):
        """ENHANCED: Better scoring data extraction"""
        values = {}
        processing_stats = {'found': 0, 'missing': 0, 'formatted': 0, 'rejected': 0}

        parcel_id = parcel.get('parcel_id', parcel.get('id', 'Unknown'))
        logger.info(f"🏠 Processing parcel {parcel_id} with scoring data")

        # Process each field individually with error isolation
        for field_key, monday_field in self.crm_field_mapping.items():
            if field_key == 'owner':
                continue

            try:
                raw_value = None

                # SPECIAL HANDLING for scoring fields
                if field_key == 'avg_slope':
                    raw_value = self._extract_slope_score(parcel)
                elif field_key == 'transmission_distance':
                    raw_value = self._extract_transmission_distance(parcel)
                elif field_key == 'transmission_voltage':
                    raw_value = self._extract_transmission_voltage(parcel)
                elif field_key == 'ml_score':
                    raw_value = self._extract_ml_score(parcel)
                elif field_key == 'traditional_score':
                    raw_value = self._extract_traditional_score(parcel)
                elif field_key == 'combined_score':
                    raw_value = self._extract_combined_score(parcel)
                elif field_key == 'solar_score':
                    raw_value = self._calculate_solar_score(parcel, project_type)
                elif field_key == 'wind_score':
                    raw_value = self._calculate_wind_score(parcel, project_type)
                elif field_key == 'battery_score':
                    raw_value = self._calculate_battery_score(parcel, project_type)
                else:
                    # Use existing field finding logic
                    raw_value = self.find_field_value(parcel, field_key)

                if raw_value is not None:
                    processing_stats['found'] += 1
                    formatted_value = self.format_field_value(field_key, raw_value, monday_field)

                    if formatted_value is not None:
                        processing_stats['formatted'] += 1
                        values[monday_field] = formatted_value
                        logger.debug(f"✅ Mapped {field_key}: {formatted_value}")
                    else:
                        processing_stats['rejected'] += 1
                        logger.warning(f"⚠️ Rejected {field_key}: {repr(raw_value)}")
                else:
                    processing_stats['missing'] += 1
                    logger.debug(f"➖ Missing {field_key}")

            except Exception as e:
                processing_stats['rejected'] += 1
                logger.error(f"❌ Error processing {field_key}: {e}")
                continue

        logger.info(f"📊 Parcel {parcel_id}: {processing_stats}")
        return values

    def _extract_slope_score(self, parcel):
        """Extract slope value from parcel analysis"""
        # Try multiple locations for slope data
        slope = (parcel.get('slope_degrees') or
                 parcel.get('suitability_analysis', {}).get('slope_degrees') or
                 parcel.get('ml_analysis', {}).get('slope_degrees'))

        if slope and slope != 'Unknown':
            try:
                return float(slope)
            except:
                pass
        return None

    def _extract_transmission_distance(self, parcel):
        """Extract transmission distance from parcel analysis"""
        distance = (parcel.get('transmission_distance') or
                    parcel.get('suitability_analysis', {}).get('transmission_distance') or
                    parcel.get('ml_analysis', {}).get('transmission_distance'))

        if distance and distance != 'Unknown':
            try:
                return float(distance)
            except:
                pass
        return None

    def _extract_transmission_voltage(self, parcel):
        """Extract transmission voltage from parcel analysis"""
        voltage = (parcel.get('transmission_voltage') or
                   parcel.get('suitability_analysis', {}).get('transmission_voltage') or
                   parcel.get('ml_analysis', {}).get('transmission_voltage'))

        if voltage and voltage != 'Unknown':
            try:
                return float(voltage)
            except:
                pass
        return None

    def _extract_ml_score(self, parcel):
        """Extract ML score from parcel analysis"""
        ml_analysis = parcel.get('ml_analysis', {})
        suitability = parcel.get('suitability_analysis', {})
        return (ml_analysis.get('predicted_score') or
                suitability.get('ml_score') or
                parcel.get('ml_score'))

    def _extract_traditional_score(self, parcel):
        """Extract traditional score from parcel analysis"""
        suitability = parcel.get('suitability_analysis', {})
        return (suitability.get('traditional_score') or
                suitability.get('slope_score') or
                parcel.get('traditional_score'))

    def _extract_combined_score(self, parcel):
        """Extract combined score from parcel analysis"""
        suitability = parcel.get('suitability_analysis', {})
        return (suitability.get('overall_score') or
                suitability.get('combined_score') or
                parcel.get('combined_score'))

    def _calculate_solar_score(self, parcel, project_type):
        """Calculate solar-specific score"""
        if project_type.lower() != 'solar':
            return None

        suitability = parcel.get('suitability_analysis', {})
        base_score = suitability.get('overall_score', 50)

        # Boost for solar-friendly characteristics
        slope = suitability.get('slope_degrees', 10)
        if slope and slope <= 10:
            base_score += 10

        return min(100, base_score)

    def _calculate_wind_score(self, parcel, project_type):
        """Calculate wind-specific score with enhanced logic"""
        suitability = parcel.get('suitability_analysis', {})
        base_score = suitability.get('overall_score', 50)

        # Get slope and transmission data
        slope = suitability.get('slope_degrees', 15)
        transmission_dist = suitability.get('transmission_distance', 2)
        acreage = parcel.get('acreage_calc', parcel.get('acreage', 0))

        # Wind-specific adjustments
        wind_score = base_score

        # Wind needs more space
        try:
            acreage_val = float(acreage)
            if acreage_val > 200:  # Large parcels better for wind
                wind_score += 15
            elif acreage_val > 100:
                wind_score += 10
            elif acreage_val < 50:  # Too small for wind
                wind_score -= 20
        except:
            pass

        # Wind is more tolerant of slopes than solar
        try:
            slope_val = float(slope)
            if slope_val <= 20:  # Wind can handle more slope than solar
                wind_score += 5
        except:
            pass

        # Wind needs good transmission (same as solar)
        try:
            trans_val = float(transmission_dist)
            if trans_val <= 1.0:
                wind_score += 10
            elif trans_val > 2.0:
                wind_score -= 15
        except:
            pass

        return min(100, max(0, wind_score))


    def _calculate_battery_score(self, parcel, project_type):
        """Calculate battery storage score"""
        suitability = parcel.get('suitability_analysis', {})
        transmission_score = suitability.get('transmission_score', 50)

        # Battery storage depends heavily on transmission access
        return transmission_score

    def create_crm_item(self, group_id, item_name, column_values):
        """FIXED: Better error handling and no silent field dropping"""

        # Validate inputs
        if not group_id or not item_name:
            logger.error("❌ Missing group_id or item_name")
            return False

        if not column_values:
            logger.warning("⚠️ No column values provided, creating with name only")
            column_values = {}

        # Clean item name
        clean_item_name = str(item_name)[:255]  # Monday.com limit

        # FIXED: More careful JSON serialization without dropping fields
        clean_column_values = {}
        dropped_fields = []

        logger.info(f"🔧 Processing {len(column_values)} fields for Monday.com")

        for key, value in column_values.items():
            try:
                # Test JSON serialization
                json.dumps(value)
                clean_column_values[key] = value
                logger.debug(f"✅ Field {key}: {repr(value)}")
            except (TypeError, ValueError) as e:
                # Try to fix common serialization issues
                try:
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        clean_column_values[key] = value
                        logger.debug(f"✅ Field {key} (fixed): {repr(value)}")
                    else:
                        # Convert to string as last resort
                        str_value = str(value)
                        json.dumps(str_value)  # Test if string version works
                        clean_column_values[key] = str_value
                        logger.warning(f"⚠️ Field {key} converted to string: {str_value[:50]}...")
                except:
                    dropped_fields.append(f"{key}={repr(value)}")
                    logger.error(f"❌ Dropping non-serializable field {key}: {e}")

        if dropped_fields:
            logger.error(f"❌ DROPPED FIELDS: {dropped_fields}")

        logger.info(f"📊 Final payload: {len(clean_column_values)} fields (dropped {len(dropped_fields)})")

        # Log the complete payload for debugging
        logger.info(f"🔍 Complete field list being sent:")
        for key, value in clean_column_values.items():
            logger.info(f"   {key}: {repr(value)}")

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
                "itemName": clean_item_name,
                "columnValues": json.dumps(clean_column_values)
            }
        }

        try:
            logger.debug(f"📤 Creating item: {clean_item_name} with {len(clean_column_values)} fields")

            response = requests.post(self.api_url, json=mutation, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            # Check for Monday.com errors
            if 'errors' in result:
                logger.error(f"❌ Monday.com API errors: {result['errors']}")
                return False

            if 'data' in result and 'create_item' in result['data'] and result['data']['create_item']:
                logger.info(f"✅ Created CRM item: {clean_item_name}")
                return True
            else:
                logger.error(f"❌ Unexpected response: {result}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error creating item {clean_item_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error creating item {clean_item_name}: {e}")
            return False

    def proper_case_with_exceptions(self, name):
        """Convert text to proper case"""
        if not name:
            return "Unknown Owner"

        try:
            str_name = self.safe_convert_to_string(name)
            if len(str_name) == 0:
                return "Unknown Owner"

            # Handle special cases
            special_cases = ['LLC', 'INC', 'CORP', 'LTD', 'LLP', 'PA', 'OF', 'AND', 'THE']
            words = str_name.upper().split()
            formatted_words = []

            for word in words:
                if word in special_cases:
                    formatted_words.append(word)
                else:
                    formatted_words.append(word.capitalize())

            result = ' '.join(formatted_words)
            return result[:255]  # Limit length

        except Exception as e:
            logger.error(f"❌ Error formatting name {repr(name)}: {e}")
            return "Unknown Owner"

    def generate_group_name(self, location, project_type):
        """Generate group name"""
        try:
            clean_location = str(location)[:50]  # Limit length
            clean_project = str(project_type).title()[:20]
            date_str = datetime.now().strftime('%Y-%m-%d')
            return f"{clean_location} - {clean_project} - {date_str}"
        except:
            return f"Parcels - {datetime.now().strftime('%Y-%m-%d')}"

    def create_group_in_board(self, group_name):
        """Create group with better error handling"""
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
                "groupName": str(group_name)[:255]  # Limit length
            }
        }

        try:
            response = requests.post(self.api_url, json=mutation, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if 'errors' in result:
                logger.error(f"❌ Group creation errors: {result['errors']}")
                return None

            if 'data' in result and 'create_group' in result['data']:
                group_id = result['data']['create_group']['id']
                logger.info(f"✅ Created group: {group_name} (ID: {group_id})")
                return group_id
            else:
                logger.error(f"❌ Unexpected group creation response: {result}")
                return None

        except Exception as e:
            logger.error(f"❌ Error creating group: {e}")
            return None

    def export_parcels_to_crm(self, parcels, project_type, location):
        """FIXED: More robust export with better error handling"""
        try:
            if not parcels:
                return {'success': False, 'error': 'No parcels provided'}

            logger.info(f"🚀 Starting CRM export of {len(parcels)} parcels")

            # Create group
            group_name = self.generate_group_name(location, project_type)
            group_id = self.create_group_in_board(group_name)
            if not group_id:
                return {'success': False, 'error': 'Failed to create group in CRM'}

            successful_exports = 0
            failed_exports = 0
            export_details = []

            for i, parcel in enumerate(parcels):
                try:
                    parcel_id = parcel.get('parcel_id', parcel.get('id', f'Parcel_{i + 1}'))
                    logger.info(f"🏠 Processing {i + 1}/{len(parcels)}: {parcel_id}")

                    # Validate parcel data
                    if not isinstance(parcel, dict):
                        logger.error(f"❌ Invalid parcel data type for {parcel_id}")
                        failed_exports += 1
                        continue

                    # Process parcel
                    crm_values = self.prepare_parcel_for_crm(parcel, project_type)

                    # Get owner name
                    owner_name = self.proper_case_with_exceptions(
                        parcel.get('owner', parcel.get('owner_name', 'Unknown Owner'))
                    )

                    # Create CRM item
                    success = self.create_crm_item(group_id, owner_name, crm_values)

                    if success:
                        successful_exports += 1
                        export_details.append({
                            'parcel_id': parcel_id,
                            'owner': owner_name,
                            'status': 'success',
                            'fields_mapped': len(crm_values)
                        })
                    else:
                        failed_exports += 1
                        export_details.append({
                            'parcel_id': parcel_id,
                            'owner': owner_name,
                            'status': 'failed',
                            'error': 'CRM creation failed'
                        })

                    # Rate limiting
                    time.sleep(0.75)  # Slightly longer delay

                except Exception as e:
                    logger.error(f"❌ Error processing parcel {i + 1}: {e}")
                    failed_exports += 1
                    export_details.append({
                        'parcel_id': parcel.get('parcel_id', f'Parcel_{i + 1}'),
                        'owner': parcel.get('owner', 'Unknown'),
                        'status': 'failed',
                        'error': str(e)
                    })

            logger.info(f"🎯 Export complete: {successful_exports} success, {failed_exports} failed")

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
            logger.error(f"❌ Export error: {e}")
            return {'success': False, 'error': str(e)}

    def test_connection(self):
        """Test Monday.com connection"""
        query = {"query": "query { me { name email } }"}

        try:
            response = requests.post(self.api_url, json=query, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'me' in result['data']:
                return {'success': True, 'user': result['data']['me']}
            else:
                return {'success': False, 'error': f'Invalid response: {result}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def debug_field_mapping(self, parcel):
        """Debug field mapping"""
        debug_info = {
            'parcel_id': parcel.get('parcel_id', 'Unknown'),
            'available_fields': list(parcel.keys()),
            'field_analysis': {},
            'total_fields_available': len(parcel.keys()),
            'mappable_fields': 0,
            'unmappable_fields': []
        }

        for field_key, monday_field in self.crm_field_mapping.items():
            if field_key == 'owner':
                continue

            raw_value = self.find_field_value(parcel, field_key)
            formatted_value = None

            if raw_value is not None:
                formatted_value = self.format_field_value(field_key, raw_value, monday_field)
                if formatted_value is not None:
                    debug_info['mappable_fields'] += 1
                else:
                    debug_info['unmappable_fields'].append(field_key)
            else:
                debug_info['unmappable_fields'].append(field_key)

            debug_info['field_analysis'][field_key] = {
                'monday_field': monday_field,
                'raw_value': raw_value,
                'formatted_value': formatted_value,
                'mapping_success': formatted_value is not None
            }

        return debug_info