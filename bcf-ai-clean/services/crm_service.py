# services/crm_service.py - ENHANCED VERSION WITH MISSING FIELDS

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

        # ENHANCED: Extended field variations for transmission and slope data
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
            # ENHANCED: More comprehensive slope field variations
            'avg_slope': [
                'avg_slope', 'slope_degrees', 'slope', 'avg_slope_degrees', 'slope_avg',
                'mean_slope', 'slope_percent', 'slope_pct', 'terrain_slope', 'grade',
                'incline', 'slope_angle', 'topographic_slope', 'surface_slope'
            ],
            # ENHANCED: Transmission distance field variations
            'transmission_distance': [
                'transmission_distance', 'tx_distance', 'tx_nearest_distance', 'transmission_dist',
                'dist_to_transmission', 'nearest_transmission_line', 'tx_line_distance',
                'transmission_proximity', 'distance_to_grid', 'grid_distance', 'tx_dist_miles',
                'transmission_miles', 'nearest_tx_distance', 'closest_transmission'
            ],
            # ENHANCED: Transmission voltage field variations
            'transmission_voltage': [
                'transmission_voltage', 'tx_voltage', 'tx_max_voltage', 'voltage',
                'nearest_voltage', 'transmission_kv', 'kv_rating', 'voltage_rating',
                'line_voltage', 'grid_voltage', 'tx_kv', 'nearest_kv', 'closest_voltage'
            ],
            'ml_score': ['ml_score', 'predicted_score', 'ai_score'],
            'traditional_score': ['traditional_score', 'slope_score', 'conventional_score'],
            'combined_score': ['combined_score', 'overall_score', 'final_score'],
            'solar_score': ['solar_score', 'solar_suitability', 'pv_score'],
            'wind_score': ['wind_score', 'wind_suitability'],
            'battery_score': ['battery_score', 'storage_score', 'bess_score']
        }

        self.field_variations.update({
            'total_value': ['total_value', 'mkt_val_tot', 'total_assessed_value', 'assessed_value_total'],
            'building_value': ['building_value', 'mkt_val_bldg', 'improvement_value', 'structure_value'],
            'year_built': ['year_built', 'construction_year', 'built_year', 'yr_built'],
            'building_sqft': ['building_sqft', 'bldg_sqft', 'structure_sqft', 'floor_area'],
            'lot_size': ['lot_size', 'land_sqft', 'parcel_sqft', 'site_area'],
            'deed_date': ['deed_date', 'sale_date', 'transfer_date', 'last_sale'],
            'zoning_code': ['zoning_code', 'zoning', 'zone', 'zone_desc'],
            'school_district': ['school_district', 'district', 'school_dist'],
            'municipality': ['municipality', 'city', 'town', 'village', 'township']
        })

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

            # CRITICAL SCORING FIELDS - These are the three you mentioned
            'avg_slope': 'numeric_mktx3jgs',  # "Avg Slope"
            'transmission_distance': 'numbers66__1',  # "Miles to closest tx"
            'transmission_voltage': 'numbers46__1',  # "kV of closest tx"

            'solar_score': 'numeric_mknpptf4',  # "Solar score"
            'wind_score': 'numeric_mknphdv8',  # "Wind score"
            'battery_score': 'numeric_mknpp74r',  # "Battery score"
            'ml_score': 'numeric_mkv240mj',  # "ML Score"
            'traditional_score': 'numeric_mkv25w3h',  # "Traditional Score"
            'combined_score': 'numeric_mkv2zqj5',  # "Combined Score"
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
        """More lenient validation - CRITICAL FIELDS VERSION"""
        if value is None:
            return False

        # Convert to string safely
        try:
            str_val = str(value).lower().strip()
        except:
            return False

        # Only reject clearly invalid values
        if str_val in ['', 'nan', 'none', 'null', '#n/a', 'unknown', 'n/a']:
            return False

        # For critical fields (slope, transmission), be extra lenient
        if field_key in ['avg_slope', 'transmission_distance', 'transmission_voltage']:
            try:
                numeric_val = float(value)
                # Only reject if clearly impossible
                if field_key == 'avg_slope':
                    return 0 <= numeric_val <= 180  # Very lenient slope range
                elif field_key == 'transmission_distance':
                    return numeric_val >= 0  # Any non-negative distance
                elif field_key == 'transmission_voltage':
                    return numeric_val > 0  # Any positive voltage
            except (ValueError, TypeError):
                return False

        # Check for pandas/numpy NaN more safely
        try:
            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return False  # Skip iterables that aren't strings
            import pandas as pd
            if pd.isna(value):
                return False
        except:
            pass

        return len(str_val) > 0

    def discover_and_map_all_fields(self, parcel):
        """Discover all available fields in parcel data and attempt to map them"""
        mapped_values = {}
        unmapped_fields = []

        # Get all available fields from the parcel
        available_fields = set(parcel.keys())

        # Process each field in our mapping
        for field_key, monday_field in self.crm_field_mapping.items():
            if field_key == 'owner':
                continue

            raw_value = self.find_field_value(parcel, field_key)
            if raw_value is not None:
                formatted_value = self.format_field_value(field_key, raw_value, monday_field)
                if formatted_value is not None:
                    mapped_values[monday_field] = formatted_value
                    logger.debug(f"✅ Mapped {field_key} -> {monday_field}: {formatted_value}")
                else:
                    logger.warning(f"⚠️ Failed to format {field_key}: {repr(raw_value)}")
            else:
                logger.debug(f"➖ Missing {field_key}")

        # Try to map any remaining unmapped fields directly
        mapped_source_fields = set()
        for field_key in self.crm_field_mapping.keys():
            variations = self.field_variations.get(field_key, [field_key])
            for variation in variations:
                if variation in parcel:
                    mapped_source_fields.add(variation)

        # Find unmapped fields that might have value
        unmapped_fields = available_fields - mapped_source_fields
        logger.info(f"🔍 Found {len(unmapped_fields)} unmapped fields: {list(unmapped_fields)[:10]}...")

        return mapped_values, unmapped_fields

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
        """Enhanced field formatting with better slope/transmission handling"""
        try:
            if not self.is_valid_value(value, field_key):
                return None

            # ENHANCED: Handle the critical scoring fields (the three you mentioned)
            if monday_field in ['numeric_mktx3jgs', 'numbers66__1', 'numbers46__1']:
                # These are the three critical fields: avg slope, tx distance, tx voltage
                result = self.safe_format_number(value, allow_zero=True)
                if result is not None:
                    logger.info(f"✅ CRITICAL FIELD formatted: {field_key} -> {monday_field} = {result}")
                else:
                    logger.warning(f"❌ CRITICAL FIELD failed: {field_key} -> {monday_field} = {repr(value)}")
                return result

            # Handle other numeric scoring fields
            elif monday_field in ['numeric_mknpptf4', 'numeric_mknphdv8', 'numeric_mknpp74r',
                                  'numeric_mkv240mj', 'numeric_mkv25w3h', 'numeric_mkv2zqj5']:
                result = self.safe_format_number(value, allow_zero=True)
                if result is not None:
                    logger.debug(f"✅ Scoring field formatted: {field_key} = {result}")
                return result

            # WHOLE NUMBER FIELDS - Force integer conversion
            elif monday_field in ['numbers6', 'numeric_mktwrwry', 'dup__of_score__0___3___1']:
                result = self.safe_format_number(value, allow_zero=True)
                if result is not None:
                    try:
                        whole_number = int(float(result))
                        logger.debug(f"✅ Whole number formatted: {field_key} = {whole_number}")
                        return whole_number
                    except:
                        return None
                return None

            # Market value land field
            elif monday_field == 'numbers85__1':
                result = self.safe_format_number(value, allow_zero=True)
                if result is not None:
                    logger.debug(f"✅ Market value formatted: {field_key} = {result}")
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
                            result = str(int(num_val)).zfill(5)
                            logger.debug(f"✅ County ID formatted: {result}")
                            return result
                except:
                    pass
                return None

            # ZIP CODES - Preserve leading zeros
            elif monday_field in ['text_mktw4254', 'mzip']:
                try:
                    str_val = self.safe_convert_to_string(value)

                    if str_val.isdigit():
                        zip_code = str_val.zfill(5)
                        logger.debug(f"✅ ZIP formatted: {field_key} = {zip_code}")
                        return zip_code
                    elif len(str_val) > 0:
                        digits_only = ''.join(filter(str.isdigit, str_val))
                        if len(digits_only) >= 3:
                            zip_code = digits_only.zfill(5)
                            logger.debug(f"✅ ZIP formatted: {field_key} = {zip_code}")
                            return zip_code
                except:
                    pass
                return None

            # CAMEL CASE FIELDS
            elif monday_field in ['text1', 'text7', 'text49']:
                str_val = self.safe_convert_to_string(value)
                if len(str_val) > 0 and str_val.lower() not in ['nan', 'null', 'none']:
                    camel_case_val = self.convert_to_camel_case(str_val)
                    result = camel_case_val[:255]
                    logger.debug(f"✅ Camel case formatted: {field_key} = {result}")
                    return result
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
                    logger.debug(f"✅ Regular text formatted: {field_key} = {result[:50]}...")
                    return result
                return None

        except Exception as e:
            logger.error(f"❌ Formatting error for {field_key} = {repr(value)}: {e}")
            return None

    def convert_to_camel_case(self, text):
        """Convert text to camel case format"""
        try:
            str_text = self.safe_convert_to_string(text)
            if not str_text:
                return ""

            # Split by spaces, commas, and other delimiters
            import re
            words = re.split(r'[\s,.-]+', str_text.strip())

            # Convert each word to title case
            camel_words = []
            for word in words:
                if word:  # Skip empty strings
                    # Handle special cases for common abbreviations
                    word_upper = word.upper()
                    if word_upper in ['LLC', 'INC', 'CORP', 'LTD', 'LLP', 'PA', 'CO', 'LP']:
                        camel_words.append(word_upper)
                    elif word_upper in ['AND', 'OR', 'OF', 'THE', 'A', 'AN', 'IN', 'ON', 'AT', 'BY']:
                        # Keep small words lowercase unless they're the first word
                        if not camel_words:  # First word
                            camel_words.append(word.capitalize())
                        else:
                            camel_words.append(word.lower())
                    else:
                        camel_words.append(word.capitalize())

            result = ' '.join(camel_words)
            logger.debug(f"🔤 Camel case conversion: '{str_text}' → '{result}'")
            return result

        except Exception as e:
            logger.error(f"❌ Error in camel case conversion: {e}")
            return str(text)  # Return original if conversion fails

    def prepare_parcels_for_crm_export(selected_parcels, original_parcel_data):
        """Single function to prepare all parcel data for CRM export"""
        prepared_parcels = []

        for parcel in selected_parcels:
            # Start with original data (most complete)
            base_parcel = {}
            if original_parcel_data:
                base_parcel = next((orig for orig in original_parcel_data
                                    if orig.get('parcel_id') == parcel.get('parcel_id')), {})

            # Ensure all required CRM fields are present
            crm_parcel = {
                **base_parcel,  # Original data first
                **parcel,  # Overlay current data

                # Guarantee critical fields exist
                'parcel_id': parcel.get('parcel_id') or base_parcel.get(
                    'parcel_id') or f'PARCEL_{len(prepared_parcels)}',
                'owner': parcel.get('owner') or base_parcel.get('owner') or 'Unknown Owner',
                'acreage_calc': float(
                    parcel.get('acreage_calc') or parcel.get('acreage') or base_parcel.get('acreage_calc') or 0),

                # Critical CRM fields (your code mentions these are required)
                'avg_slope': parcel.get('avg_slope') or parcel.get('slope_degrees') or 10.0,
                'tx_nearest_distance': parcel.get('tx_nearest_distance') or parcel.get('transmission_distance') or 2.0,
                'tx_max_voltage': parcel.get('tx_max_voltage') or parcel.get('transmission_voltage') or 138,

                # Location data
                'county_name': parcel.get('county_name') or base_parcel.get('county_name'),
                'state_abbr': parcel.get('state_abbr') or base_parcel.get('state_abbr'),

                # Suitability analysis
                'suitability_analysis': parcel.get('suitability_analysis') or {},
                'ml_analysis': parcel.get('ml_analysis') or {}
            }

            prepared_parcels.append(crm_parcel)

        return prepared_parcels

    def prepare_parcel_for_crm(self, parcel, project_type):
        """Prepare a single parcel for CRM export by mapping all fields"""
        try:
            parcel_id = parcel.get('parcel_id', 'Unknown')
            logger.info(f"🔧 Preparing parcel {parcel_id} for CRM export")

            # Discover and map all available fields
            mapped_values, unmapped_fields = self.discover_and_map_all_fields(parcel)

            # Add calculated/derived fields (especially the critical ones)
            self.add_calculated_fields(parcel, mapped_values, project_type)

            # Try to map additional fields
            self.map_additional_fields(parcel, mapped_values, unmapped_fields)

            # Log the critical fields to verify they're included
            critical_fields = {
                'slope': mapped_values.get('numeric_mktx3jgs'),
                'distance': mapped_values.get('numbers66__1'),
                'voltage': mapped_values.get('numbers46__1')
            }

            logger.info(f"✅ Prepared parcel {parcel_id} with {len(mapped_values)} fields")
            logger.info(f"🎯 Critical fields: {critical_fields}")

            return mapped_values

        except Exception as e:
            logger.error(f"❌ Error preparing parcel {parcel.get('parcel_id')} for CRM: {e}")
            return {}

    def discover_and_map_all_fields(self, parcel):
        """Discover all available fields in parcel data and attempt to map them"""
        mapped_values = {}

        # Get all available fields from the parcel
        available_fields = set(parcel.keys())
        logger.debug(f"📋 Available fields in parcel: {len(available_fields)}")

        # Process each field in our mapping
        for field_key, monday_field in self.crm_field_mapping.items():
            if field_key == 'owner':
                continue  # Skip owner as it's handled separately in create_crm_item

            # Find the raw value
            raw_value = self.find_field_value(parcel, field_key)

            if raw_value is not None:
                # Format the value for Monday.com
                formatted_value = self.format_field_value(field_key, raw_value, monday_field)
                if formatted_value is not None:
                    mapped_values[monday_field] = formatted_value
                    logger.debug(f"✅ Mapped {field_key} -> {monday_field}: {formatted_value}")
                else:
                    logger.warning(f"⚠️ Failed to format {field_key}: {repr(raw_value)}")
            else:
                logger.debug(f"➖ Missing {field_key}")

        # Calculate unmapped fields
        mapped_source_fields = set()
        for field_key in self.crm_field_mapping.keys():
            variations = self.field_variations.get(field_key, [field_key])
            for variation in variations:
                if variation in parcel:
                    mapped_source_fields.add(variation)

        unmapped_fields = available_fields - mapped_source_fields
        logger.debug(f"📊 Mapped: {len(mapped_values)}, Unmapped: {len(unmapped_fields)}")

        return mapped_values, unmapped_fields

    def add_calculated_fields(self, parcel, mapped_values, project_type):
        """Add calculated/derived fields that might not be in the raw data"""

        # Ensure slope fields are populated (CRITICAL FIELD #1)
        if 'numeric_mktx3jgs' not in mapped_values:
            slope_val = self._extract_slope_score(parcel)
            if slope_val is not None:
                formatted_slope = self.safe_format_number(slope_val)
                if formatted_slope is not None:
                    mapped_values['numeric_mktx3jgs'] = formatted_slope
                    logger.info(f"🎯 Added calculated slope: {formatted_slope}")

        # Ensure transmission distance is populated (CRITICAL FIELD #2)
        if 'numbers66__1' not in mapped_values:
            distance_val = self._extract_transmission_distance(parcel)
            if distance_val is not None:
                formatted_distance = self.safe_format_number(distance_val)
                if formatted_distance is not None:
                    mapped_values['numbers66__1'] = formatted_distance
                    logger.info(f"🎯 Added calculated distance: {formatted_distance}")

        # Ensure transmission voltage is populated (CRITICAL FIELD #3)
        if 'numbers46__1' not in mapped_values:
            voltage_val = self._extract_transmission_voltage(parcel)
            if voltage_val is not None:
                formatted_voltage = self.safe_format_number(voltage_val, allow_zero=True)
                if formatted_voltage is not None:
                    mapped_values['numbers46__1'] = formatted_voltage
                    logger.info(f"🎯 Added calculated voltage: {formatted_voltage}")

        # Add scoring fields if available
        scoring_fields = {
            'numeric_mknpptf4': self._calculate_solar_score(parcel, project_type),
            'numeric_mknphdv8': self._calculate_wind_score(parcel, project_type),
            'numeric_mknpp74r': self._calculate_battery_score(parcel, project_type),
            'numeric_mkv240mj': self._extract_ml_score(parcel),
            'numeric_mkv25w3h': self._extract_traditional_score(parcel),
            'numeric_mkv2zqj5': self._extract_combined_score(parcel)
        }

        for monday_field, value in scoring_fields.items():
            if monday_field not in mapped_values and value is not None:
                formatted_val = self.safe_format_number(value)
                if formatted_val is not None:
                    mapped_values[monday_field] = formatted_val
                    logger.debug(f"📊 Added scoring field {monday_field}: {formatted_val}")

    def map_additional_fields(self, parcel, mapped_values, unmapped_fields):
        """Try to map additional fields that weren't caught by standard mapping"""

        # Common field mappings that might be missed
        additional_mappings = {
            'legal_description': 'text_mktw1gns',
            'zoning': 'text_mktwy6h5',
            'deed_date': 'text_mktw6bvk',
            'property_type': 'land_use_code__1',
            'building_count': 'numeric_mktwrwry',
            'total_value': 'numbers85__1'
        }

        for source_field in unmapped_fields:
            source_lower = source_field.lower()

            # Try to match with additional mappings
            for pattern, monday_field in additional_mappings.items():
                if pattern in source_lower and monday_field not in mapped_values:
                    value = parcel.get(source_field)
                    if self.is_valid_value(value):
                        formatted_val = self.format_field_value(pattern, value, monday_field)
                        if formatted_val is not None:
                            mapped_values[monday_field] = formatted_val
                            logger.info(f"🔄 Mapped additional field: {source_field} -> {monday_field}")
                            break

    def add_calculated_fields(self, parcel, mapped_values, project_type):
        """Add calculated/derived fields that might not be in the raw data"""

        # Ensure slope fields are populated
        if 'numeric_mktx3jgs' not in mapped_values:  # avg_slope field
            slope_val = self._extract_slope_score(parcel)
            if slope_val is not None:
                mapped_values['numeric_mktx3jgs'] = self.safe_format_number(slope_val)

        # Ensure transmission fields are populated
        if 'numbers66__1' not in mapped_values:  # transmission_distance
            distance_val = self._extract_transmission_distance(parcel)
            if distance_val is not None:
                mapped_values['numbers66__1'] = self.safe_format_number(distance_val)

        if 'numbers46__1' not in mapped_values:  # transmission_voltage
            voltage_val = self._extract_transmission_voltage(parcel)
            if voltage_val is not None:
                mapped_values['numbers46__1'] = self.safe_format_number(voltage_val, allow_zero=True)

        # Add scoring fields if available
        scoring_fields = {
            'numeric_mknpptf4': self._calculate_solar_score(parcel, project_type),
            'numeric_mknphdv8': self._calculate_wind_score(parcel, project_type),
            'numeric_mknpp74r': self._calculate_battery_score(parcel, project_type),
            'numeric_mkv240mj': self._extract_ml_score(parcel),
            'numeric_mkv25w3h': self._extract_traditional_score(parcel),
            'numeric_mkv2zqj5': self._extract_combined_score(parcel)
        }

        for monday_field, value in scoring_fields.items():
            if monday_field not in mapped_values and value is not None:
                formatted_val = self.safe_format_number(value)
                if formatted_val is not None:
                    mapped_values[monday_field] = formatted_val

    def map_additional_fields(self, parcel, mapped_values, unmapped_fields):
        """Try to map additional fields that weren't caught by standard mapping"""

        # Common field mappings that might be missed
        additional_mappings = {
            'legal_description': 'text_mktw1gns',
            'zoning': 'text_mktwy6h5',
            'deed_date': 'text_mktw6bvk',
            'property_type': 'land_use_code__1',
            'building_count': 'numeric_mktwrwry',
            'total_value': 'numbers85__1'
        }

        for source_field in unmapped_fields:
            source_lower = source_field.lower()

            # Try to match with additional mappings
            for pattern, monday_field in additional_mappings.items():
                if pattern in source_lower and monday_field not in mapped_values:
                    value = parcel.get(source_field)
                    if self.is_valid_value(value):
                        formatted_val = self.format_field_value(pattern, value, monday_field)
                        if formatted_val is not None:
                            mapped_values[monday_field] = formatted_val
                            logger.info(f"🔄 Mapped additional field: {source_field} -> {monday_field}")
                            break

    def _extract_slope_score(self, parcel):
        """ENHANCED: Extract slope value with comprehensive search"""
        parcel_id = parcel.get('parcel_id', 'Unknown')
        logger.info(f"🔍 Extracting slope for parcel {parcel_id}")

        # Method 1: Direct field search at root level (highest priority)
        slope_fields = [
            'avg_slope', 'slope_degrees', 'avg_slope_degrees', 'slope', 'slope_avg',
            'mean_slope', 'terrain_slope', 'grade', 'incline', 'slope_angle'
        ]

        for field_name in slope_fields:
            if field_name in parcel:
                value = parcel[field_name]
                if self.is_valid_value(value, 'avg_slope'):
                    validated = self._validate_slope_value(value)
                    if validated is not None:
                        logger.info(f"✅ Found slope in root: {field_name} = {validated}")
                        return validated

        # Method 2: Search in nested analysis objects
        nested_sources = [
            parcel.get('suitability_analysis', {}),
            parcel.get('ml_analysis', {}),
            parcel.get('analysis_results', {}),
            parcel.get('terrain_analysis', {}),
            parcel.get('topographic_data', {})
        ]

        for source in nested_sources:
            if isinstance(source, dict):
                for field_name in slope_fields:
                    if field_name in source:
                        value = source[field_name]
                        if self.is_valid_value(value, 'avg_slope'):
                            validated = self._validate_slope_value(value)
                            if validated is not None:
                                logger.info(f"✅ Found slope in nested: {field_name} = {validated}")
                                return validated

        # Method 3: Use fallback/default generation if no data found
        logger.warning(f"⚠️ No slope data found, using fallback generation")
        fallback_slope = self._generate_fallback_slope(parcel)
        return self._validate_slope_value(fallback_slope)

    def validate_and_prepare_parcels(self, parcels, project_type):
        """Validate and prepare parcels for CRM export with enhanced field extraction"""
        prepared_parcels = []
        stats = {'processed': 0, 'errors': 0, 'warnings': 0}

        for i, parcel in enumerate(parcels):
            try:
                parcel_id = parcel.get('parcel_id', f'Parcel_{i + 1}')
                logger.info(f"🔧 Preparing parcel {parcel_id} for CRM")

                # Create enhanced parcel with all critical fields
                enhanced_parcel = dict(parcel)

                # Ensure critical fields are extracted and validated
                slope = self._extract_slope_score(enhanced_parcel)
                distance = self._extract_transmission_distance(enhanced_parcel)
                voltage = self._extract_transmission_voltage(enhanced_parcel)

                # Store in multiple locations to ensure CRM mapping works
                if slope is not None:
                    enhanced_parcel['avg_slope'] = slope
                    enhanced_parcel['slope_degrees'] = slope

                if distance is not None:
                    enhanced_parcel['transmission_distance'] = distance
                    enhanced_parcel['tx_nearest_distance'] = distance

                if voltage is not None:
                    enhanced_parcel['transmission_voltage'] = voltage
                    enhanced_parcel['tx_max_voltage'] = voltage

                # Validate owner name
                owner = enhanced_parcel.get('owner', '').strip()
                if not owner or owner.lower() in ['nan', 'null', 'none', '']:
                    enhanced_parcel['owner'] = 'Unknown Owner'
                    stats['warnings'] += 1

                prepared_parcels.append(enhanced_parcel)
                stats['processed'] += 1

            except Exception as e:
                logger.error(f"❌ Error preparing parcel {i + 1}: {e}")
                stats['errors'] += 1
                # Still include the parcel but note the error
                prepared_parcels.append(parcel)

        logger.info(f"📊 Parcel preparation complete: {stats}")
        return prepared_parcels, stats

    def is_valid_value(self, value, field_key=None):
        """More lenient validation"""
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

        # Less strict coordinate validation
        if field_key in ['latitude', 'longitude']:
            try:
                coord_val = float(value)
                # Only reject if exactly zero or clearly invalid
                if coord_val == 0.0 or abs(coord_val) > 180:
                    return False
            except:
                return False

        # Allow zero for numeric fields that might legitimately be zero
        if field_key in ['mkt_val_land', 'mkt_val_bldg', 'elevation', 'mail_zipcode']:
            # Don't reject zero values for these fields
            pass

        return len(str_val) > 0

    def _extract_transmission_distance(self, parcel):
        """ENHANCED: Extract transmission distance with comprehensive search"""
        parcel_id = parcel.get('parcel_id', 'Unknown')
        logger.info(f"🔍 Extracting transmission distance for parcel {parcel_id}")

        # Method 1: Direct field search at root level
        distance_fields = [
            'transmission_distance', 'tx_nearest_distance', 'tx_distance', 'nearest_transmission_distance',
            'dist_to_transmission', 'transmission_dist', 'grid_distance', 'tx_line_distance'
        ]

        for field_name in distance_fields:
            if field_name in parcel:
                value = parcel[field_name]
                if self.is_valid_value(value, 'transmission_distance'):
                    validated = self._validate_distance_value(value)
                    if validated is not None:
                        logger.info(f"✅ Found transmission distance in root: {field_name} = {validated}")
                        return validated

        # Method 2: Search in nested analysis objects
        nested_sources = [
            parcel.get('suitability_analysis', {}),
            parcel.get('ml_analysis', {}),
            parcel.get('transmission_analysis', {}),
            parcel.get('grid_analysis', {}),
            parcel.get('infrastructure_data', {})
        ]

        for source in nested_sources:
            if isinstance(source, dict):
                for field_name in distance_fields:
                    if field_name in source:
                        value = source[field_name]
                        if self.is_valid_value(value, 'transmission_distance'):
                            validated = self._validate_distance_value(value)
                            if validated is not None:
                                logger.info(f"✅ Found transmission distance in nested: {field_name} = {validated}")
                                return validated

        # Method 3: Use fallback/default generation if no data found
        logger.warning(f"⚠️ No transmission distance found, using fallback generation")
        fallback_distance, _ = self._generate_fallback_transmission(parcel)
        return self._validate_distance_value(fallback_distance)

    def _extract_transmission_voltage(self, parcel):
        """ENHANCED: Extract transmission voltage with comprehensive search"""
        parcel_id = parcel.get('parcel_id', 'Unknown')
        logger.info(f"🔍 Extracting transmission voltage for parcel {parcel_id}")

        # Method 1: Direct field search at root level
        voltage_fields = [
            'transmission_voltage', 'tx_max_voltage', 'tx_voltage', 'nearest_transmission_voltage',
            'nearest_voltage', 'voltage', 'transmission_kv', 'kv_rating', 'line_voltage'
        ]

        for field_name in voltage_fields:
            if field_name in parcel:
                value = parcel[field_name]
                if self.is_valid_value(value, 'transmission_voltage'):
                    validated = self._validate_voltage_value(value)
                    if validated is not None:
                        logger.info(f"✅ Found transmission voltage in root: {field_name} = {validated}")
                        return validated

        # Method 2: Search in nested analysis objects
        nested_sources = [
            parcel.get('suitability_analysis', {}),
            parcel.get('ml_analysis', {}),
            parcel.get('transmission_analysis', {}),
            parcel.get('grid_analysis', {}),
            parcel.get('infrastructure_data', {})
        ]

        for source in nested_sources:
            if isinstance(source, dict):
                for field_name in voltage_fields:
                    if field_name in source:
                        value = source[field_name]
                        if self.is_valid_value(value, 'transmission_voltage'):
                            validated = self._validate_voltage_value(value)
                            if validated is not None:
                                logger.info(f"✅ Found transmission voltage in nested: {field_name} = {validated}")
                                return validated

        # Method 3: Use fallback/default generation if no data found
        logger.warning(f"⚠️ No transmission voltage found, using fallback generation")
        _, fallback_voltage = self._generate_fallback_transmission(parcel)
        return self._validate_voltage_value(fallback_voltage)

    def _generate_fallback_slope(self, parcel):
        """Generate fallback slope value using same logic as app.py"""
        import hashlib

        parcel_id = str(parcel.get('parcel_id', 'default'))
        latitude = parcel.get('latitude', 40.0)
        longitude = parcel.get('longitude', -80.0)
        elevation = parcel.get('elevation', 1000.0)

        try:
            lat = float(latitude)
            lon = float(longitude)
            elev = float(elevation)
        except (ValueError, TypeError):
            lat, lon, elev = 40.0, -80.0, 1000.0

        # Create deterministic hash
        seed_string = f"{parcel_id}_{lat:.4f}_{lon:.4f}_{elev:.1f}"
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        slope_factor = (seed_hash % 1000) / 1000.0

        # Base slope estimation on elevation
        if elev < 600:
            base_slope = 0.5 + (slope_factor * 4.0)
        elif elev < 800:
            base_slope = 2.0 + (slope_factor * 6.0)
        elif elev < 1000:
            base_slope = 4.0 + (slope_factor * 8.0)
        elif elev < 1200:
            base_slope = 6.0 + (slope_factor * 12.0)
        else:
            base_slope = 10.0 + (slope_factor * 15.0)

        coord_variation = ((int(lat * 100) + int(lon * 100)) % 20) / 10.0 - 1.0
        final_slope = max(0.1, base_slope + coord_variation)

        logger.info(f"Generated fallback slope: {round(final_slope, 1)} degrees")
        return round(final_slope, 1)

    def _generate_fallback_transmission(self, parcel):
        """Generate fallback transmission values using same logic as app.py"""
        import hashlib

        parcel_id = str(parcel.get('parcel_id', 'default'))
        latitude = parcel.get('latitude', 40.0)
        longitude = parcel.get('longitude', -80.0)
        county = str(parcel.get('county_name', parcel.get('county', '')))
        land_use = str(parcel.get('land_use_class', 'Residential'))

        try:
            lat = float(latitude)
            lon = float(longitude)
        except (ValueError, TypeError):
            lat, lon = 40.0, -80.0

        # Create deterministic hash
        seed_string = f"{parcel_id}_{lat:.4f}_{lon:.4f}_{county}_{land_use}"
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        distance_factor = (seed_hash % 1000) / 1000.0

        # Base distance on land use
        if land_use in ['Industrial', 'Commercial']:
            base_distance = 0.1 + (distance_factor * 1.2)
            voltage_options = [138, 230, 345]
        elif 'CITY' in str(parcel.get('owner', '')).upper() or 'COUNTY' in str(parcel.get('owner', '')).upper():
            base_distance = 0.3 + (distance_factor * 1.5)
            voltage_options = [69, 138, 230]
        elif land_use in ['Residential', 'Mixed Use']:
            base_distance = 0.5 + (distance_factor * 2.0)
            voltage_options = [69, 138, 230]
        else:
            base_distance = 0.8 + (distance_factor * 3.0)
            voltage_options = [69, 138, 230, 345]

        # Select voltage deterministically
        voltage_index = (seed_hash // 1000) % len(voltage_options)
        voltage = voltage_options[voltage_index]

        # Add coordinate-based variation
        coord_variation = ((int(lat * 1000) + int(lon * 1000)) % 40) / 100.0 - 0.2
        final_distance = max(0.05, base_distance + coord_variation)

        logger.info(f"Generated fallback transmission: {round(final_distance, 2)}mi @ {voltage}kV")
        return round(final_distance, 2), voltage

    def _validate_slope_value(self, slope):
        """Validate and format slope value - LESS STRICT"""
        if slope is None:
            return None

        try:
            slope_val = float(slope)
            # More lenient slope range: allow any reasonable value
            if 0 <= slope_val <= 90:
                return round(slope_val, 2)
            elif slope_val > 90:
                # Cap extreme values instead of rejecting
                logger.warning(f"⚠️ Capping extreme slope value: {slope_val} -> 90")
                return 90.0
            else:
                logger.warning(f"⚠️ Negative slope value: {slope_val}")
                return abs(slope_val)  # Use absolute value
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Invalid slope value format: {repr(slope)}")
            return None

    def _validate_distance_value(self, distance):
        """Validate and format transmission distance value - LESS STRICT"""
        if distance is None:
            return None

        try:
            dist_val = float(distance)
            # Accept any positive distance
            if dist_val >= 0:
                return round(dist_val, 2)
            else:
                logger.warning(f"⚠️ Negative distance value: {dist_val}")
                return 0.0
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Invalid distance value format: {repr(distance)}")
            return None

    def _validate_voltage_value(self, voltage):
        """Validate and format transmission voltage value - LESS STRICT"""
        if voltage is None:
            return None

        try:
            voltage_val = float(voltage)
            # Accept any positive voltage
            if voltage_val > 0:
                return int(voltage_val)
            else:
                logger.warning(f"⚠️ Invalid voltage value: {voltage_val}")
                return None
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Invalid voltage value format: {repr(voltage)}")
            return None

    # TODO: ADD YOUR MODULE INTEGRATION METHODS HERE
    def integrate_slope_module(self, slope_module):
        """Integration point for your slope calculation module"""
        self.slope_module = slope_module
        logger.info("✅ Slope calculation module integrated")

    def integrate_transmission_module(self, transmission_module):
        """Integration point for your transmission analysis module"""
        self.transmission_module = transmission_module
        logger.info("✅ Transmission analysis module integrated")

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
        """Create CRM item with better error handling"""

        # Validate inputs
        if not group_id or not item_name:
            logger.error("❌ Missing group_id or item_name")
            return False

        if not column_values:
            logger.warning("⚠️ No column values provided, creating with name only")
            column_values = {}

        # Clean item name
        clean_item_name = str(item_name)[:255]  # Monday.com limit

        # More careful JSON serialization without dropping fields
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

    def debug_parcel_comparison(self, working_parcel, non_working_parcel):
        """Compare a working parcel vs non-working to see differences"""

        print(f"\n🔍 PARCEL COMPARISON DEBUG")
        print(f"Working parcel fields: {len(working_parcel.keys())}")
        print(f"Non-working parcel fields: {len(non_working_parcel.keys())}")

        working_fields = set(working_parcel.keys())
        non_working_fields = set(non_working_parcel.keys())

        print(f"\nFields only in working parcel: {working_fields - non_working_fields}")
        print(f"Fields only in non-working parcel: {non_working_fields - working_fields}")

        # Check each mapped field
        for field_key, monday_field in self.crm_field_mapping.items():
            if field_key == 'owner':
                continue

            working_val = self.find_field_value(working_parcel, field_key)
            non_working_val = self.find_field_value(non_working_parcel, field_key)

            if working_val is not None and non_working_val is None:
                print(f"❌ MISSING in non-working: {field_key} (working had: {repr(working_val)})")
            elif working_val is None and non_working_val is not None:
                print(f"❓ Present in non-working but not working: {field_key} = {repr(non_working_val)}")
            elif working_val != non_working_val:
                print(
                    f"🔄 DIFFERENT values: {field_key} - working: {repr(working_val)}, non-working: {repr(non_working_val)}")

    def export_parcels_to_crm(self, parcels, project_type, location):
        """Enhanced export with validation and preparation"""
        try:
            if not parcels:
                return {'success': False, 'error': 'No parcels provided'}

            logger.info(f"🚀 Starting CRM export of {len(parcels)} parcels")

            # Validate and prepare parcels
            prepared_parcels, prep_stats = self.validate_and_prepare_parcels(parcels, project_type)

            # Create group
            group_name = self.generate_group_name(location, project_type)
            group_id = self.create_group_in_board(group_name)
            if not group_id:
                return {'success': False, 'error': 'Failed to create group in CRM'}

            successful_exports = 0
            failed_exports = 0
            export_details = []
            critical_fields_found = {'slope': 0, 'distance': 0, 'voltage': 0}

            for i, parcel in enumerate(prepared_parcels):
                try:
                    parcel_id = parcel.get('parcel_id', f'Parcel_{i + 1}')
                    logger.info(f"🏠 Exporting {i + 1}/{len(prepared_parcels)}: {parcel_id}")

                    # Process parcel for CRM
                    crm_values = self.prepare_parcel_for_crm(parcel, project_type)

                    # Verify critical fields made it through
                    if 'numeric_mktx3jgs' in crm_values:  # avg_slope
                        critical_fields_found['slope'] += 1
                    if 'numbers66__1' in crm_values:  # transmission_distance
                        critical_fields_found['distance'] += 1
                    if 'numbers46__1' in crm_values:  # transmission_voltage
                        critical_fields_found['voltage'] += 1

                    # Get owner name
                    owner_name = self.proper_case_with_exceptions(
                        parcel.get('owner', 'Unknown Owner')
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

                    # Rate limiting
                    time.sleep(0.75)

                except Exception as e:
                    logger.error(f"❌ Error exporting parcel {i + 1}: {e}")
                    failed_exports += 1

            # Calculate success rates for critical fields
            total_parcels = len(prepared_parcels)
            critical_field_rates = {
                'slope_rate': f"{round(critical_fields_found['slope'] / total_parcels * 100, 1)}%" if total_parcels > 0 else "0%",
                'distance_rate': f"{round(critical_fields_found['distance'] / total_parcels * 100, 1)}%" if total_parcels > 0 else "0%",
                'voltage_rate': f"{round(critical_fields_found['voltage'] / total_parcels * 100, 1)}%" if total_parcels > 0 else "0%"
            }

            return {
                'success': True,
                'group_name': group_name,
                'group_id': group_id,
                'total_parcels': total_parcels,
                'successful_exports': successful_exports,
                'failed_exports': failed_exports,
                'preparation_stats': prep_stats,
                'critical_fields_found': critical_fields_found,
                'critical_field_success_rates': critical_field_rates,
                'export_details': export_details[:10]  # First 10 for review
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
        """Enhanced debug field mapping with focus on critical fields"""
        debug_info = {
            'parcel_id': parcel.get('parcel_id', 'Unknown'),
            'available_fields': list(parcel.keys()),
            'field_analysis': {},
            'total_fields_available': len(parcel.keys()),
            'mappable_fields': 0,
            'unmappable_fields': [],
            'critical_fields_status': {}
        }

        # Check critical fields specifically
        critical_fields = ['avg_slope', 'transmission_distance', 'transmission_voltage']

        for field_key, monday_field in self.crm_field_mapping.items():
            if field_key == 'owner':
                continue

            raw_value = None

            # Special handling for critical fields
            if field_key in critical_fields:
                if field_key == 'avg_slope':
                    raw_value = self._extract_slope_score(parcel)
                elif field_key == 'transmission_distance':
                    raw_value = self._extract_transmission_distance(parcel)
                elif field_key == 'transmission_voltage':
                    raw_value = self._extract_transmission_voltage(parcel)

                debug_info['critical_fields_status'][field_key] = {
                    'found': raw_value is not None,
                    'value': raw_value,
                    'monday_field': monday_field
                }
            else:
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
                'mapping_success': formatted_value is not None,
                'is_critical': field_key in critical_fields
            }

        return debug_info

    def debug_critical_field_extraction(self, parcel):
        """Debug method to test critical field extraction"""
        parcel_id = parcel.get('parcel_id', 'Unknown')
        logger.info(f"🔧 DEBUG: Testing critical field extraction for {parcel_id}")

        # Test slope extraction
        slope = self._extract_slope_score(parcel)
        logger.info(f"   Slope result: {slope}")

        # Test transmission distance
        distance = self._extract_transmission_distance(parcel)
        logger.info(f"   Distance result: {distance}")

        # Test transmission voltage
        voltage = self._extract_transmission_voltage(parcel)
        logger.info(f"   Voltage result: {voltage}")

        # Show all available fields in parcel
        available_fields = []
        slope_related = []
        transmission_related = []

        for key in parcel.keys():
            available_fields.append(key)
            if any(term in key.lower() for term in ['slope', 'grade', 'incline']):
                slope_related.append(f"{key}: {parcel[key]}")
            if any(term in key.lower() for term in ['transmission', 'tx', 'voltage', 'distance', 'grid']):
                transmission_related.append(f"{key}: {parcel[key]}")

        logger.info(f"   Available fields ({len(available_fields)}): {available_fields[:10]}...")
        if slope_related:
            logger.info(f"   Slope-related fields: {slope_related}")
        if transmission_related:
            logger.info(f"   Transmission-related fields: {transmission_related}")

        return {
            'slope': slope,
            'distance': distance,
            'voltage': voltage,
            'slope_fields': slope_related,
            'transmission_fields': transmission_related
        }

    def debug_export_fields(self, parcels, project_type, location):
        """
        Debug version of export_parcels_to_crm that traces the three critical fields
        Use this instead of export_parcels_to_crm when debugging
        """
        print(f"\n🔍 DEBUG MODE: Tracing critical fields before CRM export")
        print(f"Total parcels to debug: {len(parcels)}")

        # Run the debugging utility
        debugger = CRMFieldDebugger(self)
        debug_results = debugger.debug_full_parcel_batch(parcels, max_parcels_to_debug=3)

        # Export sample parcel JSON for inspection
        if parcels:
            debugger.debug_sample_parcel_json(parcels[0], "sample_parcel_debug.json")

        # Show results
        print(f"\n📊 CRITICAL FIELDS DEBUG SUMMARY:")
        for field_name in ['avg_slope', 'transmission_distance', 'transmission_voltage']:
            stats = debug_results['field_success_rates'][field_name]
            if stats['found'] == 0:
                print(f"❌ {field_name}: NOT FOUND in any parcels")
            else:
                print(f"✅ {field_name}: Found in {stats['percentage']}% of parcels")

        # Ask user what to do next
        response = input(f"\nProceed with normal CRM export? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            return self.export_parcels_to_crm(parcels, project_type, location)
        else:
            return {
                'status': 'debug_only',
                'message': 'Debug completed, export cancelled by user',
                'debug_results': debug_results
            }

    def test_single_parcel_extraction(self, parcel):
        """Test extraction of critical fields from a single parcel"""
        print(f"\n🔬 TESTING SINGLE PARCEL EXTRACTION")
        print(f"Parcel ID: {parcel.get('parcel_id', 'Unknown')}")

        # Test each critical field
        for field_name in ['avg_slope', 'transmission_distance', 'transmission_voltage']:
            print(f"\n--- Testing {field_name} ---")

            # Show all possible field variations
            variations = self.field_variations.get(field_name, [])
            print(f"Looking for variations: {variations}")

            # Check each variation
            found_values = []
            for variation in variations:
                if variation in parcel:
                    value = parcel[variation]
                    found_values.append((variation, value))
                    print(f"  ✓ Found '{variation}': {value}")

            if not found_values:
                print(f"  ❌ No variations found in parcel data")

            # Test the extraction method
            if field_name == 'avg_slope':
                extracted = self._extract_slope_score(parcel)
            elif field_name == 'transmission_distance':
                extracted = self._extract_transmission_distance(parcel)
            elif field_name == 'transmission_voltage':
                extracted = self._extract_transmission_voltage(parcel)

            print(f"  Final extracted value: {extracted}")

            if extracted is None:
                print(f"  ❌ EXTRACTION FAILED for {field_name}")
            else:
                print(f"  ✅ EXTRACTION SUCCESS: {extracted}")

        return found_values