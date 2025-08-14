# config.py - Fixed version with proper config object interface

import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize empty mappings as fallback
COUNTY_FIPS_MAP = {}
STATE_COUNTIES_MAP = {}


class Config:
    """Configuration class that provides both dictionary-style and method access"""

    def __init__(self):
        self._config = {
            'RAUSA_CLIENT_KEY': os.getenv('RAUSA_CLIENT_KEY'),
            'RAUSA_API_VERSION': os.getenv('RAUSA_API_VERSION', '9'),
            'RAUSA_API_URL': os.getenv('RAUSA_API_URL', 'https://reportallusa.com/api/parcels'),
            'BUCKET_NAME': os.getenv('BUCKET_NAME'),
            'GOOGLE_APPLICATION_CREDENTIALS': os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        }

    def get(self, key, default=None):
        """Get configuration value"""
        return self._config.get(key, default)

    def is_cloud_environment(self):
        """Check if running in cloud environment"""
        return bool(os.getenv('GAE_ENV') or os.getenv('GOOGLE_CLOUD_PROJECT'))

    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self._config[key]

    def __contains__(self, key):
        """Allow 'in' operator"""
        return key in self._config


# Create the config instance
config = Config()


def load_county_fips_mapping():
    """Load county to FIPS mapping from JSON file with robust error handling"""
    try:
        # Look for the counties JSON file in multiple locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'counties-trimmed.json'),
            'counties-trimmed.json',
            os.path.join(os.getcwd(), 'counties-trimmed.json'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'counties-trimmed.json')
        ]

        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break

        if not json_path:
            print(f"ERROR: counties-trimmed.json not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            return {}, {}

        print(f"Loading county FIPS mapping from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            counties_data = json.load(f)

        if not counties_data or not isinstance(counties_data, list):
            print(f"ERROR: Invalid JSON data in {json_path}")
            return {}, {}

        # Create mapping from county name to FIPS
        county_fips_map = {}
        state_counties_map = {}

        for county_info in counties_data:
            if not isinstance(county_info, dict):
                continue

            county_name = county_info.get('county')
            state = county_info.get('state')
            fips = county_info.get('fips')

            # Validate required fields
            if not county_name or not state or not fips:
                print(f"WARNING: Skipping incomplete county record: {county_info}")
                continue

            # Map county name to FIPS
            county_fips_map[county_name] = fips

            # Group counties by state
            if state not in state_counties_map:
                state_counties_map[state] = []
            state_counties_map[state].append({
                'name': county_name,
                'fips': fips
            })

        print(f"Successfully loaded {len(county_fips_map)} counties across {len(state_counties_map)} states")
        return county_fips_map, state_counties_map

    except FileNotFoundError as e:
        print(f"ERROR: County FIPS file not found: {e}")
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in county FIPS file: {e}")
        return {}, {}
    except Exception as e:
        print(f"ERROR: Unexpected error loading county FIPS mapping: {e}")
        return {}, {}


# Load the mappings when the module is imported
try:
    COUNTY_FIPS_MAP, STATE_COUNTIES_MAP = load_county_fips_mapping()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load county mappings: {e}")
    COUNTY_FIPS_MAP = {}
    STATE_COUNTIES_MAP = {}


# LEGACY FUNCTION NAMES (for compatibility with existing app.py)
def get_county_id(county_name):
    """Legacy function name - returns FIPS code for a county name"""
    return get_county_fips(county_name)


def get_county_fips(county_name):
    """Get FIPS code for a county name with error handling"""
    if not county_name:
        return None

    if not COUNTY_FIPS_MAP:
        print("WARNING: County FIPS mapping not loaded")
        return None

    # Try exact match first
    fips = COUNTY_FIPS_MAP.get(county_name)
    if fips:
        return fips

    # Try case-insensitive search
    for county, fips_code in COUNTY_FIPS_MAP.items():
        if county.lower() == county_name.lower():
            return fips_code

    # Try partial match (for cases like "Laramie" vs "Laramie County")
    for county, fips_code in COUNTY_FIPS_MAP.items():
        if county_name.lower() in county.lower() or county.lower() in county_name.lower():
            return fips_code

    print(f"WARNING: No FIPS code found for county: '{county_name}'")
    print(f"Available counties: {list(COUNTY_FIPS_MAP.keys())[:10]}...")
    return None


def get_counties_for_state(state_abbr):
    """Get list of counties for a state with error handling"""
    if not state_abbr:
        return []

    if not STATE_COUNTIES_MAP:
        print("WARNING: State counties mapping not loaded")
        return []

    # Try exact match
    counties = STATE_COUNTIES_MAP.get(state_abbr, [])
    if counties:
        return counties

    # Try case-insensitive search
    for state, county_list in STATE_COUNTIES_MAP.items():
        if state.lower() == state_abbr.lower():
            return county_list

    print(f"WARNING: No counties found for state: '{state_abbr}'")
    print(f"Available states: {list(STATE_COUNTIES_MAP.keys())}")
    return []


def get_state_name_from_abbr(state_abbr):
    """Convert state abbreviation to full name"""
    state_mapping = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    return state_mapping.get(state_abbr, state_abbr)


# Additional standalone functions for compatibility
def is_cloud_environment():
    """Standalone function to check if running in cloud environment"""
    return config.is_cloud_environment()


# Legacy COUNTY_ID_MAPPING for compatibility
COUNTY_ID_MAPPING = COUNTY_FIPS_MAP


# Add debugging function
def debug_county_mapping():
    """Debug function to check county mapping status"""
    print(f"County FIPS mapping loaded: {len(COUNTY_FIPS_MAP)} counties")
    print(f"State counties mapping loaded: {len(STATE_COUNTIES_MAP)} states")

    if COUNTY_FIPS_MAP:
        print("Sample counties:")
        for i, (county, fips) in enumerate(list(COUNTY_FIPS_MAP.items())[:5]):
            print(f"  {county}: {fips}")

    if STATE_COUNTIES_MAP:
        print("Available states:")
        for state in sorted(STATE_COUNTIES_MAP.keys()):
            print(f"  {state}: {len(STATE_COUNTIES_MAP[state])} counties")


# Call debug function if running as main
if __name__ == "__main__":
    debug_county_mapping()