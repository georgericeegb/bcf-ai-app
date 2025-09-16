from google.cloud import bigquery
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)


class EnhancedCountyManager:
    def __init__(self):
        try:
            self.client = bigquery.Client()
            self.dataset_id = "renewable_energy"
            self.counties_table = "state_counties_with_analysis"
            self.counties_table_ref = f"{self.client.project}.{self.dataset_id}.{self.counties_table}"
            logger.info("Enhanced County Manager initialized")
            self._ensure_table_exists()
        except Exception as e:
            logger.error(f"BigQuery initialization failed: {e}")
            self.client = None

    def _ensure_table_exists(self):
        """Ensure the enhanced counties table exists"""
        if not self.client:
            return False

        try:
            # Try to get the table
            self.client.get_table(self.counties_table_ref)
            logger.info("Counties with analysis table already exists")
            return True
        except Exception:
            # Table doesn't exist, create it
            logger.info("Creating counties with analysis table...")
            return self._create_enhanced_table()

    def _create_enhanced_table(self):
        """Create the enhanced counties table with AI analysis fields"""
        try:
            schema = [
                bigquery.SchemaField("state_code", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("county_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("county_fips", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("population", "INTEGER"),
                bigquery.SchemaField("population_density", "FLOAT"),
                bigquery.SchemaField("land_area_sq_miles", "FLOAT"),
                bigquery.SchemaField("rural_indicator", "BOOLEAN"),
                bigquery.SchemaField("population_tier", "STRING"),

                # AI Analysis fields
                bigquery.SchemaField("solar_analysis_score", "FLOAT"),
                bigquery.SchemaField("solar_analysis_date", "TIMESTAMP"),
                bigquery.SchemaField("solar_analysis_data", "JSON"),
                bigquery.SchemaField("wind_analysis_score", "FLOAT"),
                bigquery.SchemaField("wind_analysis_date", "TIMESTAMP"),
                bigquery.SchemaField("wind_analysis_data", "JSON"),

                # Metadata
                bigquery.SchemaField("last_updated", "TIMESTAMP"),
                bigquery.SchemaField("data_source", "STRING"),
                bigquery.SchemaField("analysis_version", "STRING"),
            ]

            table = bigquery.Table(self.counties_table_ref, schema=schema)
            table = self.client.create_table(table)
            logger.info("✅ Created enhanced counties table")
            return True

        except Exception as e:
            logger.error(f"Failed to create enhanced table: {e}")
            return False

    def get_cached_analysis(self, state_code: str, project_type: str, max_age_days: int = 30) -> Optional[Dict]:
        """Get cached county analysis if it exists and is fresh"""
        if not self.client:
            return None

        try:
            # Check if we have recent analysis for this state/project type
            analysis_field = f"{project_type.lower()}_analysis_date"
            data_field = f"{project_type.lower()}_analysis_data"
            score_field = f"{project_type.lower()}_analysis_score"

            query = f"""
            SELECT 
                county_name,
                county_fips,
                population,
                population_density,
                {score_field} as analysis_score,
                {data_field} as analysis_data,
                {analysis_field} as analysis_date
            FROM `{self.counties_table_ref}`
            WHERE state_code = @state_code
            AND {analysis_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @max_age_days DAY)
            AND {analysis_field} IS NOT NULL
            ORDER BY population DESC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper()),
                    bigquery.ScalarQueryParameter("max_age_days", "INT64", max_age_days)
                ]
            )

            results = self.client.query(query, job_config=job_config)
            counties = list(results)

            if not counties:
                logger.info(f"No cached analysis found for {state_code} {project_type}")
                return None

            # Convert to the expected format
            county_data = []
            for row in counties:
                county_info = {
                    'name': row.county_name,
                    'fips': row.county_fips,
                    'population': row.population,
                    'population_density': row.population_density,
                    'score': float(row.analysis_score) if row.analysis_score else 50.0,
                    'analysis_date': row.analysis_date.isoformat() if row.analysis_date else None
                }

                # Parse stored analysis data
                if row.analysis_data:
                    try:
                        analysis_data = json.loads(row.analysis_data) if isinstance(row.analysis_data,
                                                                                    str) else row.analysis_data
                        county_info.update(analysis_data)
                    except Exception as e:
                        logger.warning(f"Could not parse analysis data for {row.county_name}: {e}")

                county_data.append(county_info)

            logger.info(f"✅ Found cached analysis for {len(county_data)} counties in {state_code}")

            return {
                'state': state_code,
                'project_type': project_type,
                'counties': county_data,
                'total_counties': len(county_data),
                'cache_hit': True,
                'analysis_date': counties[0].analysis_date.isoformat() if counties else None
            }

        except Exception as e:
            logger.error(f"Error getting cached analysis: {e}")
            return None

    def save_analysis_results(self, state_code: str, project_type: str, analysis_results: Dict) -> bool:
        """Save new analysis results to BigQuery"""
        if not self.client:
            return False

        try:
            counties = analysis_results.get('county_rankings', [])
            if not counties:
                logger.error("No counties in analysis results")
                return False

            # Prepare upsert operations
            analysis_field_date = f"{project_type.lower()}_analysis_date"
            analysis_field_data = f"{project_type.lower()}_analysis_data"
            analysis_field_score = f"{project_type.lower()}_analysis_score"

            success_count = 0

            for county in counties:
                try:
                    county_name = county.get('name', '').strip()
                    if not county_name:
                        continue

                    # Prepare analysis data (exclude large nested objects)
                    analysis_data = {
                        'strengths': county.get('strengths', []),
                        'challenges': county.get('challenges', []),
                        'resource_quality': county.get('resource_quality', 'Unknown'),
                        'policy_environment': county.get('policy_environment', 'Unknown'),
                        'development_potential': county.get('development_potential', 'Unknown'),
                        'investment_tier': county.get('investment_tier', 'Unknown'),
                        'estimated_capacity_mw': county.get('estimated_capacity_mw', 0),
                        'development_timeline': county.get('development_timeline', 'Unknown')
                    }

                    # First, try to update existing record
                    update_query = f"""
                    UPDATE `{self.counties_table_ref}`
                    SET 
                        {analysis_field_score} = @analysis_score,
                        {analysis_field_data} = @analysis_data,
                        {analysis_field_date} = @analysis_date,
                        last_updated = @last_updated,
                        analysis_version = @analysis_version
                    WHERE state_code = @state_code 
                    AND county_name = @county_name
                    """

                    job_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("state_code", "STRING", state_code.upper()),
                            bigquery.ScalarQueryParameter("county_name", "STRING", county_name),
                            bigquery.ScalarQueryParameter("analysis_score", "FLOAT", float(county.get('score', 50.0))),
                            bigquery.ScalarQueryParameter("analysis_data", "JSON", json.dumps(analysis_data)),
                            bigquery.ScalarQueryParameter("analysis_date", "TIMESTAMP", datetime.utcnow()),
                            bigquery.ScalarQueryParameter("last_updated", "TIMESTAMP", datetime.utcnow()),
                            bigquery.ScalarQueryParameter("analysis_version", "STRING", "2.0_ai_enhanced")
                        ]
                    )

                    job = self.client.query(update_query, job_config=job_config)
                    job.result()

                    if job.num_dml_affected_rows > 0:
                        success_count += 1
                    else:
                        # County doesn't exist, insert new record
                        self._insert_new_county_with_analysis(state_code, county, project_type, analysis_data)
                        success_count += 1

                except Exception as county_error:
                    logger.error(f"Error saving analysis for {county_name}: {county_error}")
                    continue

            logger.info(f"✅ Saved analysis for {success_count}/{len(counties)} counties")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return False

    def _insert_new_county_with_analysis(self, state_code: str, county: Dict, project_type: str, analysis_data: Dict):
        """Insert a new county record with analysis"""
        try:
            # Get state FIPS
            state_fips_map = {
                'NC': '37', 'CA': '06', 'TX': '48', 'FL': '12', 'NY': '36', 'PA': '42',
                'OH': '39', 'GA': '13', 'VA': '51', 'WA': '53', 'CO': '08', 'AZ': '04'
            }
            state_fips = state_fips_map.get(state_code.upper(), '99')

            county_fips = f"{state_fips}{county.get('rank', 1):03d}"
            population = county.get('population', county.get('estimated_population', 50000))

            # Fix: Convert datetime to string for JSON serialization
            current_time = datetime.utcnow()

            row_data = {
                'state_code': state_code.upper(),
                'county_name': county.get('name'),
                'county_fips': county_fips,
                'population': int(population),
                'population_density': float(county.get('population_density', population / 500)),
                'land_area_sq_miles': float(county.get('land_area_sq_miles', population / 100)),
                'rural_indicator': bool(county.get('rural_indicator', population < 100000)),
                'population_tier': county.get('population_tier', self._classify_population_tier(population)),
                'last_updated': current_time,  # BigQuery will handle this
                'data_source': 'AI_Enhanced_Analysis',
                'analysis_version': '2.0_ai_enhanced'
            }

            # Add project-specific analysis with proper datetime handling
            if project_type.lower() == 'solar':
                row_data.update({
                    'solar_analysis_score': float(county.get('score', 50.0)),
                    'solar_analysis_data': json.dumps(analysis_data),  # Already JSON string
                    'solar_analysis_date': current_time  # BigQuery timestamp
                })
            elif project_type.lower() == 'wind':
                row_data.update({
                    'wind_analysis_score': float(county.get('score', 50.0)),
                    'wind_analysis_data': json.dumps(analysis_data),  # Already JSON string
                    'wind_analysis_date': current_time  # BigQuery timestamp
                })

            table = self.client.get_table(self.counties_table_ref)
            errors = self.client.insert_rows_json(table, [row_data])

            if errors:
                logger.error(f"Insert errors for {county.get('name')}: {errors}")
            else:
                logger.info(f"Inserted new county: {county.get('name')}")

        except Exception as e:
            logger.error(f"Error inserting new county: {e}")


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