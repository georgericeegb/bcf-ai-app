from google.cloud import bigquery
import logging
from typing import List, Dict, Optional
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class BigQueryCountiesManager:
    def __init__(self):
        try:
            self.client = bigquery.Client()
            self.dataset_id = "renewable_energy"
            self.table_id = "state_counties"
            self.table_ref = f"{self.client.project}.{self.dataset_id}.{self.table_id}"
            logger.info("BigQuery counties manager initialized")
        except Exception as e:
            logger.error(f"BigQuery initialization failed: {e}")
            self.client = None

    def get_state_counties(self, state_code: str) -> List[Dict]:
        """Get counties for a state from BigQuery"""
        if not self.client:
            return []

        query = f"""
        SELECT 
            county_name,
            county_fips,
            population,
            population_density,
            land_area_sq_miles,
            rural_indicator,
            population_tier
        FROM `{self.table_ref}`
        WHERE state_code = @state_code
        ORDER BY population DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper())
            ]
        )

        try:
            results = self.client.query(query, job_config=job_config)
            counties = []

            for row in results:
                counties.append({
                    'name': row.county_name,
                    'fips': row.county_fips,
                    'population': row.population,
                    'population_density': row.population_density,
                    'land_area_sq_miles': row.land_area_sq_miles,
                    'rural_indicator': row.rural_indicator,
                    'population_tier': row.population_tier,
                    'state_code': state_code.upper()
                })

            logger.info(f"Retrieved {len(counties)} counties for {state_code} from BigQuery")
            return counties

        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            return []

    def counties_exist_for_state(self, state_code: str) -> bool:
        """Check if counties already exist for a state"""
        if not self.client:
            return False

        query = f"""
        SELECT COUNT(*) as county_count
        FROM `{self.table_ref}`
        WHERE state_code = @state_code
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper())
            ]
        )

        try:
            results = self.client.query(query, job_config=job_config)
            count = list(results)[0].county_count
            return count > 0
        except Exception as e:
            logger.error(f"Error checking counties existence: {e}")
            return False

    def populate_counties_with_ai(self, state_code: str, api_key: str) -> bool:
        """Use AI to get county data and store in BigQuery"""
        if not self.client:
            return False

        try:
            state_names = {
                'NC': 'North Carolina', 'CA': 'California', 'TX': 'Texas', 'FL': 'Florida',
                'NY': 'New York', 'PA': 'Pennsylvania', 'OH': 'Ohio', 'GA': 'Georgia',
                'VA': 'Virginia', 'WA': 'Washington', 'CO': 'Colorado', 'AZ': 'Arizona'
            }

            state_name = state_names.get(state_code.upper(), state_code)

            prompt = f"""List ALL counties in {state_name} with their population data.

Return JSON format with all counties:
{{
    "counties": [
        {{"name": "Wake", "population": 1100000, "is_rural": false}},
        {{"name": "Johnston", "population": 200000, "is_rural": true}},
        {{"name": "Vance", "population": 45000, "is_rural": true}}
    ]
}}

Include ALL counties in {state_name} (should be 50+ counties for most states), with realistic 2024 population estimates and rural classification based on population density."""

            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            }

            payload = {
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 6000,  # Increased for all counties
                'messages': [{'role': 'user', 'content': prompt}]
            }

            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"AI API failed: {response.status_code}")
                return False

            result = response.json()
            analysis_text = result['content'][0]['text']

            import re, json
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in AI response")
                return False

            county_data = json.loads(json_match.group())
            counties = county_data.get('counties', [])

            if not counties:
                logger.error("No counties found in AI response")
                return False

            # Insert into BigQuery
            rows_to_insert = []
            state_fips = self._get_state_fips(state_code)

            for i, county in enumerate(counties):
                population = county.get('population', 50000)
                # Calculate realistic density
                area = population / 150 if population > 100000 else population / 50
                density = population / area if area > 0 else 50

                rows_to_insert.append({
                    'state_code': state_code.upper(),
                    'county_name': county['name'],
                    'county_fips': f"{state_fips}{i + 1:03d}",
                    'population': population,
                    'population_density': density,
                    'land_area_sq_miles': area,
                    'rural_indicator': county.get('is_rural', population < 100000),
                    'population_tier': self._classify_population_tier(population),
                    'last_updated': datetime.utcnow(),
                    'data_source': 'AI_Generated'
                })

            table = self.client.get_table(self.table_ref)
            errors = self.client.insert_rows_json(table, rows_to_insert)

            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return False

            logger.info(f"Successfully populated {len(counties)} counties for {state_code}")
            return True

        except Exception as e:
            logger.error(f"Error populating counties with AI: {e}")
            return False

    def _get_state_fips(self, state_code: str) -> str:
        """Get FIPS code for state"""
        fips_codes = {
            'NC': '37', 'CA': '06', 'TX': '48', 'FL': '12', 'NY': '36', 'PA': '42',
            'OH': '39', 'GA': '13', 'VA': '51', 'WA': '53', 'CO': '08', 'AZ': '04'
        }
        return fips_codes.get(state_code.upper(), '99')

    def _classify_population_tier(self, population: int) -> str:
        """Classify population tier"""
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