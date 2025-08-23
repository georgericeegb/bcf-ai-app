import os
import logging
from datetime import datetime
from google.cloud import bigquery
import json

logger = logging.getLogger(__name__)


class OutreachTracker:
    def __init__(self, project_id='bcfparcelsearchrepository'):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = 'ml_feedback'
        self.table_id = 'outreach_events'
        self.full_table_id = f"{project_id}.{self.dataset_id}.{self.table_id}"

        # Ensure table exists
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create the outreach tracking table if it doesn't exist"""
        try:
            # Check if dataset exists
            dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} exists")
            except Exception as e:
                # Create dataset
                logger.info(f"Creating dataset {self.dataset_id}")
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "ML feedback and tracking data"
                self.client.create_dataset(dataset)
                logger.info(f"Created dataset {self.dataset_id}")

            # Check if table exists
            try:
                self.client.get_table(self.full_table_id)
                logger.info(f"Outreach tracking table exists: {self.full_table_id}")
            except Exception as e:
                # Create table
                logger.info(f"Creating outreach tracking table: {self.full_table_id}")
                schema = [
                    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("parcel_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("owner_name", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("outreach_timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("project_type", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("ml_score", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("traditional_score", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("combined_score", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("user_action", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("crm_export_successful", "BOOLEAN", mode="NULLABLE"),
                    bigquery.SchemaField("parcel_characteristics", "STRING", mode="NULLABLE"),
                    # Changed from JSON to STRING
                    bigquery.SchemaField("location", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("session_id", "STRING", mode="NULLABLE")
                ]

                table = bigquery.Table(self.full_table_id, schema=schema)
                table.description = "Tracks user outreach actions for ML model feedback"
                self.client.create_table(table)
                logger.info(f"Created outreach tracking table: {self.full_table_id}")

        except Exception as e:
            logger.error(f"Error ensuring outreach table exists: {e}")
            # Don't raise exception, just log error

    def track_outreach(self, parcels, project_type, location, crm_success=True, session_id=None):
        """Track outreach events for multiple parcels"""
        try:
            import uuid

            if not session_id:
                session_id = str(uuid.uuid4())

            if not parcels:
                logger.warning("No parcels provided for tracking")
                return False

            rows_to_insert = []

            for parcel in parcels:
                try:
                    # Extract scores safely
                    ml_analysis = parcel.get('ml_analysis', {})
                    suitability_analysis = parcel.get('suitability_analysis', {})

                    ml_score = ml_analysis.get('predicted_score')
                    traditional_score = suitability_analysis.get('traditional_score') or suitability_analysis.get(
                        'overall_score')
                    combined_score = suitability_analysis.get('overall_score')

                    # Create characteristics JSON string
                    characteristics = {
                        'acreage': parcel.get('acreage_calc') or parcel.get('acreage'),
                        'land_value': parcel.get('mkt_val_land'),
                        'elevation': parcel.get('elevation'),
                        'land_use_class': parcel.get('land_use_class'),
                        'county': parcel.get('county_name') or parcel.get('county'),
                        'state': parcel.get('state_abbr') or parcel.get('state'),
                        'coordinates': {
                            'lat': parcel.get('latitude'),
                            'lon': parcel.get('longitude')
                        }
                    }

                    # Convert to string for BigQuery
                    characteristics_str = json.dumps(characteristics) if characteristics else '{}'

                    row = {
                        'event_id': str(uuid.uuid4()),
                        'parcel_id': str(parcel.get('parcel_id', '')),
                        'owner_name': str(parcel.get('owner', ''))[:255],
                        'outreach_timestamp': datetime.utcnow().isoformat(),
                        'project_type': str(project_type),
                        'ml_score': float(ml_score) if ml_score is not None else None,
                        'traditional_score': float(traditional_score) if traditional_score is not None else None,
                        'combined_score': float(combined_score) if combined_score is not None else None,
                        'user_action': 'crm_export',
                        'crm_export_successful': bool(crm_success),
                        'parcel_characteristics': characteristics_str,
                        'location': str(location)[:255],
                        'session_id': session_id
                    }

                    rows_to_insert.append(row)
                    logger.debug(f"Prepared tracking row for parcel {parcel.get('parcel_id')}")

                except Exception as e:
                    logger.error(f"Error preparing parcel {parcel.get('parcel_id')} for tracking: {e}")
                    continue

            if rows_to_insert:
                # Insert using streaming API (simpler)
                errors = self.client.insert_rows_json(
                    self.full_table_id,
                    rows_to_insert
                )

                if errors:
                    logger.error(f"BigQuery insert errors: {errors}")
                    return False
                else:
                    logger.info(f"✅ Tracked {len(rows_to_insert)} outreach events in BigQuery")
                    return True
            else:
                logger.warning("No valid rows to insert for outreach tracking")
                return False

        except Exception as e:
            logger.error(f"Error tracking outreach: {e}")
            import traceback
            logger.error(f"Tracking traceback: {traceback.format_exc()}")
            return False

    def track_parcel_exclusion(self, parcel_id, exclusion_reason, is_recommended,
                               parcel_data, project_type, location, session_id=None):
        """Track why a parcel was excluded from outreach"""
        try:
            import uuid

            if not session_id:
                session_id = str(uuid.uuid4())

            # Create exclusion record
            characteristics = {
                'exclusion_reason': exclusion_reason,
                'is_recommended': is_recommended,
                'analysis_override': exclusion_reason == 'analysis_override',
                'parcel_data': {
                    'acreage': parcel_data.get('acreage_calc') or parcel_data.get('acreage'),
                    'slope': parcel_data.get('suitability_analysis', {}).get('slope_degrees'),
                    'transmission_distance': parcel_data.get('suitability_analysis', {}).get('transmission_distance'),
                    'ml_score': parcel_data.get('ml_analysis', {}).get('predicted_score'),
                    'traditional_score': parcel_data.get('suitability_analysis', {}).get('overall_score'),
                    'land_use': parcel_data.get('land_use_class'),
                    'owner': parcel_data.get('owner')
                }
            }

            row = {
                'event_id': str(uuid.uuid4()),
                'parcel_id': str(parcel_id),
                'owner_name': str(parcel_data.get('owner', ''))[:255],
                'outreach_timestamp': datetime.utcnow().isoformat(),
                'project_type': str(project_type),
                'ml_score': float(parcel_data.get('ml_analysis', {}).get('predicted_score', 0)) if parcel_data.get(
                    'ml_analysis', {}).get('predicted_score') else None,
                'traditional_score': float(
                    parcel_data.get('suitability_analysis', {}).get('overall_score', 0)) if parcel_data.get(
                    'suitability_analysis', {}).get('overall_score') else None,
                'combined_score': None,
                'user_action': f'exclusion_{exclusion_reason}',
                'crm_export_successful': False,
                'parcel_characteristics': json.dumps(characteristics),
                'location': str(location)[:255],
                'session_id': session_id
            }

            errors = self.client.insert_rows_json(self.full_table_id, [row])

            if errors:
                logger.error(f"BigQuery insert errors for exclusion: {errors}")
                return False
            else:
                logger.info(f"✅ Tracked exclusion: {parcel_id} - {exclusion_reason}")
                return True

        except Exception as e:
            logger.error(f"Error tracking exclusion: {e}")
            return False

    def track_parcel_feedback(self, parcel_id, feedback_reason, custom_reason, parcel_data,
                              feedback_details, project_type, location, session_id=None):
        """Track individual parcel feedback for ML training"""
        try:
            import uuid

            if not session_id:
                session_id = str(uuid.uuid4())

            # Create comprehensive feedback record
            characteristics = {
                'feedback_reason': feedback_reason,
                'custom_reason': custom_reason,
                'feedback_details': feedback_details,
                'parcel_characteristics': {
                    'acreage': parcel_data.get('acreage_calc') or parcel_data.get('acreage'),
                    'land_value': parcel_data.get('mkt_val_land'),
                    'elevation': parcel_data.get('elevation'),
                    'land_use_class': parcel_data.get('land_use_class'),
                    'owner': parcel_data.get('owner'),
                    'coordinates': {
                        'lat': parcel_data.get('latitude'),
                        'lon': parcel_data.get('longitude')
                    }
                }
            }

            row = {
                'event_id': str(uuid.uuid4()),
                'parcel_id': str(parcel_id),
                'owner_name': str(parcel_data.get('owner', ''))[:255],
                'outreach_timestamp': datetime.utcnow().isoformat(),
                'project_type': str(project_type),
                'ml_score': float(feedback_details.get('ml_score')) if feedback_details.get(
                    'ml_score') is not None else None,
                'traditional_score': float(feedback_details.get('traditional_score')) if feedback_details.get(
                    'traditional_score') is not None else None,
                'combined_score': None,  # This is feedback, not selection
                'user_action': f'parcel_feedback_{feedback_reason}',
                'crm_export_successful': False,  # This is feedback
                'parcel_characteristics': json.dumps(characteristics),
                'location': str(location)[:255],
                'session_id': session_id
            }

            # Insert using streaming API
            errors = self.client.insert_rows_json(self.full_table_id, [row])

            if errors:
                logger.error(f"BigQuery insert errors for parcel feedback: {errors}")
                return False
            else:
                logger.info(f"✅ Tracked parcel feedback: {parcel_id} - {feedback_reason}")
                return True

        except Exception as e:
            logger.error(f"Error tracking parcel feedback: {e}")
            return False

    def track_pattern_feedback(self, feedback_type, reason, feedback_data, project_type, location, session_id=None):
        """Track pattern-based feedback for ML improvement"""
        try:
            import uuid

            if not session_id:
                session_id = str(uuid.uuid4())

            # Create summary of feedback data
            feedback_summary = {
                'feedback_type': feedback_type,
                'reason': reason,
                'not_recommended_selected_count': len(feedback_data.get('notRecommendedSelected', [])),
                'high_ml_not_selected_count': len(feedback_data.get('highMLNotSelected', [])),
                'total_recommended': feedback_data.get('totalRecommended', 0)
            }

            row = {
                'event_id': str(uuid.uuid4()),
                'parcel_id': 'PATTERN_FEEDBACK',
                'owner_name': 'Pattern Analysis',
                'outreach_timestamp': datetime.utcnow().isoformat(),
                'project_type': str(project_type),
                'ml_score': None,
                'traditional_score': None,
                'combined_score': None,
                'user_action': f'pattern_feedback_{feedback_type}',
                'crm_export_successful': False,
                'parcel_characteristics': json.dumps(feedback_summary),
                'location': str(location)[:255],
                'session_id': session_id
            }

            # Insert using streaming API
            errors = self.client.insert_rows_json(self.full_table_id, [row])

            if errors:
                logger.error(f"BigQuery insert errors for pattern feedback: {errors}")
                return False
            else:
                logger.info(f"✅ Tracked pattern feedback: {feedback_type} - {reason}")
                return True

        except Exception as e:
            logger.error(f"Error tracking pattern feedback: {e}")
            return False

    def track_rejection(self, parcel, rejection_reason, project_type, location, session_id=None):
        """Track parcel rejection for ML feedback"""
        try:
            import uuid

            if not session_id:
                session_id = str(uuid.uuid4())

            # Extract scores safely
            ml_analysis = parcel.get('ml_analysis', {})
            suitability_analysis = parcel.get('suitability_analysis', {})

            ml_score = ml_analysis.get('predicted_score')
            traditional_score = suitability_analysis.get('traditional_score') or suitability_analysis.get(
                'overall_score')
            combined_score = suitability_analysis.get('overall_score')

            # Create characteristics JSON string
            characteristics = {
                'acreage': parcel.get('acreage_calc') or parcel.get('acreage'),
                'land_value': parcel.get('mkt_val_land'),
                'elevation': parcel.get('elevation'),
                'land_use_class': parcel.get('land_use_class'),
                'county': parcel.get('county_name') or parcel.get('county'),
                'state': parcel.get('state_abbr') or parcel.get('state'),
                'slope_degrees': suitability_analysis.get('slope_degrees'),
                'transmission_distance': suitability_analysis.get('transmission_distance'),
                'transmission_voltage': suitability_analysis.get('transmission_voltage'),
                'rejection_reason': rejection_reason,
                'coordinates': {
                    'lat': parcel.get('latitude'),
                    'lon': parcel.get('longitude')
                }
            }

            characteristics_str = json.dumps(characteristics)

            row = {
                'event_id': str(uuid.uuid4()),
                'parcel_id': str(parcel.get('parcel_id', '')),
                'owner_name': str(parcel.get('owner', ''))[:255],
                'outreach_timestamp': datetime.utcnow().isoformat(),
                'project_type': str(project_type),
                'ml_score': float(ml_score) if ml_score is not None else None,
                'traditional_score': float(traditional_score) if traditional_score is not None else None,
                'combined_score': float(combined_score) if combined_score is not None else None,
                'user_action': f'rejection_{rejection_reason}',
                'crm_export_successful': False,  # This is a rejection
                'parcel_characteristics': characteristics_str,
                'location': str(location)[:255],
                'session_id': session_id
            }

            # Insert using streaming API
            errors = self.client.insert_rows_json(self.full_table_id, [row])

            if errors:
                logger.error(f"BigQuery insert errors for rejection: {errors}")
                return False
            else:
                logger.info(f"✅ Tracked rejection for parcel {parcel.get('parcel_id')}: {rejection_reason}")
                return True

        except Exception as e:
            logger.error(f"Error tracking rejection: {e}")
            return False

    def get_outreach_history(self, parcel_id=None, days=30):
        """Get outreach history for analysis"""
        try:
            where_clause = ""
            if parcel_id:
                where_clause = f"WHERE parcel_id = '{parcel_id}'"
            else:
                where_clause = f"WHERE outreach_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)"

            query = f"""
            SELECT 
                event_id,
                parcel_id,
                owner_name,
                outreach_timestamp,
                project_type,
                ml_score,
                traditional_score,
                combined_score,
                user_action,
                crm_export_successful,
                location
            FROM `{self.full_table_id}`
            {where_clause}
            ORDER BY outreach_timestamp DESC
            """

            results = self.client.query(query).result()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting outreach history: {e}")
            return []