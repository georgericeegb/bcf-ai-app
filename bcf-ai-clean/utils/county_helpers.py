import json
import os
import logging

logger = logging.getLogger(__name__)

def load_counties_from_file(state_code):
    """Load counties from the counties file"""
    try:
        counties_file = os.path.join(os.path.dirname(__file__), '..', 'counties-trimmed.json')
        
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
                    'full_name': county.get('county', ''),
                    'state': county.get('state', ''),
                    'population': county.get('population', 0)
                })

        logger.info(f"Loaded {len(state_counties)} counties for {state_code}")
        return state_counties

    except Exception as e:
        logger.error(f"Error loading counties: {e}")
        return []

def get_state_name(state_code):
    """Convert state code to full state name"""
    state_names = {
        'AL': 'Alabama', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'FL': 'Florida', 'GA': 'Georgia', 'IL': 'Illinois',
        'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky',
        'LA': 'Louisiana', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'NE': 'Nebraska', 'NV': 'Nevada', 'NM': 'New Mexico',
        'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
        'OH': 'Ohio', 'OK': 'Oklahoma', 'PA': 'Pennsylvania', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VA': 'Virginia', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    return state_names.get(state_code, state_code)