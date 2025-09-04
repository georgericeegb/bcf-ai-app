import os
import json
import logging
from datetime import datetime, timedelta
from google.cloud import storage

logger = logging.getLogger(__name__)


class StateAnalysisCache:
    def __init__(self):
        self.bucket_name = os.getenv('CACHE_BUCKET_NAME', os.getenv('BUCKET_NAME'))
        self.client = storage.Client()
        self.bucket = None

        if self.bucket_name:
            try:
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"✅ Cache initialized with bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"❌ Cache bucket initialization error: {e}")
                self.bucket = None
        else:
            logger.warning("⚠️ No cache bucket configured - caching disabled")

    def get_cached_analysis(self, cache_key, max_age_days=30):
        """Get cached state analysis if it exists and is recent enough"""
        if not self.bucket:
            logger.debug("No cache bucket available")
            return None

        try:
            cache_path = f"state_analysis/{cache_key}.json"
            blob = self.bucket.blob(cache_path)

            if not blob.exists():
                logger.debug(f"No cached data found for: {cache_key}")
                return None

            # Download and parse
            data_str = blob.download_as_text()
            cached_data = json.loads(data_str)

            # Check age
            cached_timestamp = cached_data.get('analysis_timestamp')
            if cached_timestamp:
                try:
                    cache_date = datetime.fromisoformat(cached_timestamp.replace('Z', ''))
                    age = datetime.now() - cache_date

                    if age.days < max_age_days:
                        cached_data['cache_age_days'] = age.days
                        logger.info(f"✅ Using cached analysis: {cache_key} (age: {age.days} days)")
                        return cached_data
                    else:
                        logger.info(f"⏰ Cache expired for {cache_key} (age: {age.days} days)")
                        return None
                except Exception as date_error:
                    logger.error(f"Error parsing cache date: {date_error}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error retrieving cached analysis: {e}")
            return None

    def save_analysis(self, cache_key, analysis_data):
        """Save analysis to cache"""
        if not self.bucket:
            logger.debug("No cache bucket available - skipping cache save")
            return False

        try:
            cache_data = {
                **analysis_data,
                'analysis_timestamp': datetime.now().isoformat(),
                'cache_key': cache_key
            }

            cache_path = f"state_analysis/{cache_key}.json"
            blob = self.bucket.blob(cache_path)

            blob.upload_from_string(
                json.dumps(cache_data, indent=2),
                content_type='application/json'
            )

            logger.info(f"✅ Cached state analysis: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Error caching analysis: {e}")
            return False