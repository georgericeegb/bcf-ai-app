# census_api.py
import requests
import logging
from typing import List, Dict, Optional


class CensusCountyAPI:
    def __init__(self):
        self.base_url = "https://api.census.gov/data"
        self.logger = logging.getLogger(__name__)

        # State FIPS codes
        self.state_fips = {
            'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09',
            'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
            'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25',
            'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32',
            'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
            'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
            'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56'
        }

    def get_state_counties_with_population(self, state_code: str) -> List[Dict]:
        """
        Fetch all counties in a state with population and area data
        Returns list of counties with demographic data for scoring
        """
        try:
            state_fips = self.state_fips.get(state_code.upper())
            if not state_fips:
                raise ValueError(f"Invalid state code: {state_code}")

            # Get population estimates (most recent year available)
            population_url = f"{self.base_url}/2022/pep/population"
            population_params = {
                'get': 'NAME,POP_2022,DENSITY_2022',
                'for': 'county:*',
                'in': f'state:{state_fips}'
            }

            pop_response = requests.get(population_url, params=population_params, timeout=30)

            if pop_response.status_code != 200:
                # Fallback to older dataset if 2022 not available
                population_url = f"{self.base_url}/2021/pep/population"
                population_params['get'] = 'NAME,POP,DENSITY'
                pop_response = requests.get(population_url, params=population_params, timeout=30)

            pop_response.raise_for_status()
            pop_data = pop_response.json()

            # Get county characteristics (land area, etc.)
            char_url = f"{self.base_url}/2021/pep/charagegroups"
            char_params = {
                'get': 'NAME,GEONAME',
                'for': 'county:*',
                'in': f'state:{state_fips}'
            }

            counties = []

            # Parse population data (skip header row)
            for row in pop_data[1:]:
                try:
                    county_name = row[0].replace(' County', '').replace(f', {self.get_state_name(state_code)}', '')
                    population = int(row[1]) if row[1] and row[1] != 'null' else 0
                    density = float(row[2]) if len(row) > 2 and row[2] and row[2] != 'null' else 0
                    county_fips = f"{state_fips}{row[-1].zfill(3)}"  # Last element is county code

                    # Calculate land area from density and population
                    land_area_sq_miles = population / density if density > 0 else 0

                    # Calculate population density category for scoring
                    if density <= 25:
                        density_category = "Very Low"
                        density_score = 1.0  # Best for renewable projects
                    elif density <= 100:
                        density_category = "Low"
                        density_score = 0.8
                    elif density <= 300:
                        density_category = "Medium"
                        density_score = 0.6
                    elif density <= 1000:
                        density_category = "High"
                        density_score = 0.3
                    else:
                        density_category = "Very High"
                        density_score = 0.1  # Worst for renewable projects

                    county_data = {
                        'name': county_name,
                        'fips': county_fips,
                        'state_code': state_code.upper(),
                        'population': population,
                        'population_density': density,
                        'density_category': density_category,
                        'density_score': density_score,
                        'land_area_sq_miles': round(land_area_sq_miles, 2),
                        'rural_indicator': density <= 100,  # Rural counties generally better for renewables
                        'population_tier': self.classify_population_tier(population)
                    }

                    counties.append(county_data)

                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing county data for {row}: {e}")
                    continue

            self.logger.info(f"Retrieved {len(counties)} counties for {state_code}")
            return sorted(counties, key=lambda x: x['name'])

        except requests.RequestException as e:
            self.logger.error(f"Census API request failed: {e}")
            # Return fallback data if API fails
            return self.get_fallback_counties(state_code)

        except Exception as e:
            self.logger.error(f"Error fetching county data: {e}")
            return self.get_fallback_counties(state_code)

    def classify_population_tier(self, population: int) -> str:
        """Classify counties by population size"""
        if population >= 1000000:
            return "Major Metro"
        elif population >= 500000:
            return "Large Metro"
        elif population >= 100000:
            return "Metro"
        elif population >= 50000:
            return "Urban"
        elif population >= 10000:
            return "Small City"
        else:
            return "Rural"

    def get_state_name(self, state_code: str) -> str:
        """Convert state code to full name"""
        state_names = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
        return state_names.get(state_code.upper(), state_code)

    def get_fallback_counties(self, state_code: str) -> List[Dict]:
        """Fallback county data if Census API fails"""
        fallback_data = {
            'OH': [
                {'name': 'Adams', 'fips': '39001', 'population': 27750, 'population_density': 47.3},
                {'name': 'Allen', 'fips': '39003', 'population': 102351, 'population_density': 252.1},
                {'name': 'Cuyahoga', 'fips': '39035', 'population': 1264817, 'population_density': 2753.3},
                {'name': 'Franklin', 'fips': '39049', 'population': 1323807, 'population_density': 2462.1},
                {'name': 'Hamilton', 'fips': '39061', 'population': 830639, 'population_density': 2080.4}
            ]
        }

        counties = fallback_data.get(state_code.upper(), [])
        for county in counties:
            county['density_score'] = 1.0 if county['population_density'] <= 100 else 0.5
            county['rural_indicator'] = county['population_density'] <= 100
            county['population_tier'] = self.classify_population_tier(county['population'])

        return counties