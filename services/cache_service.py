# services/cache_service.py - AI Response Caching Service
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from google.cloud import storage
import os


class AIResponseCache:
    def __init__(self, bucket_name: str = "bcfparcelsearchrepository"):
        """Initialize cache service with Google Cloud Storage"""
        self.bucket_name = bucket_name

        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
        except Exception as e:
            print(f"Warning: Google Cloud Storage not available: {e}")
            self.client = None
            self.bucket = None

        # Cache expiration rules (in days)
        self.cache_expiration = {
            'state': 30,  # State-level data changes slowly
            'county': 14,  # County data moderate changes
            'local': 7  # Local data changes more frequently
        }

    def _generate_cache_key(self, analysis_type: str, project_type: str,
                            analysis_level: str, location: str,
                            criteria: list = None) -> str:
        """Generate unique cache key for the request"""

        # Create criteria hash to handle different criteria combinations
        criteria_str = '|'.join(sorted(criteria)) if criteria else 'none'
        criteria_hash = hashlib.md5(criteria_str.encode()).hexdigest()[:8]

        # Include date for expiration checking
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Clean location name for file path
        clean_location = location.replace(' ', '_').replace(',', '_').lower()

        cache_key = f"ai_cache/{analysis_type}/{analysis_level}/{project_type}/{clean_location}_{criteria_hash}_{date_str}.json"

        return cache_key

    def _is_cache_valid(self, cache_data: Dict[str, Any], analysis_level: str) -> bool:
        """Check if cached data is still valid based on expiration rules"""

        if 'timestamp' not in cache_data:
            return False

        try:
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            expiration_days = self.cache_expiration.get(analysis_level, 7)
            expiration_time = cache_time + timedelta(days=expiration_days)

            return datetime.now() < expiration_time
        except Exception as e:
            print(f"Cache validation error: {e}")
            return False

    def get_cached_response(self, analysis_type: str, project_type: str,
                            analysis_level: str, location: str,
                            criteria: list = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached AI response if available and valid"""

        if not self.bucket:
            return None

        try:
            cache_key = self._generate_cache_key(analysis_type, project_type,
                                                 analysis_level, location, criteria)

            # Try to get the blob from Google Cloud Storage
            blob = self.bucket.blob(cache_key)

            if not blob.exists():
                print(f"Cache miss: {cache_key}")
                return None

            # Download and parse cached data
            cache_data = json.loads(blob.download_as_text())

            # Check if cache is still valid
            if self._is_cache_valid(cache_data, analysis_level):
                print(f"Cache hit: {cache_key}")
                return cache_data.get('response')
            else:
                print(f"Cache expired: {cache_key}")
                # Optionally delete expired cache
                try:
                    blob.delete()
                except:
                    pass
                return None

        except Exception as e:
            print(f"Cache retrieval error: {e}")
            return None

    def store_response(self, analysis_type: str, project_type: str,
                       analysis_level: str, location: str,
                       response: Dict[str, Any], criteria: list = None) -> bool:
        """Store AI response in cache"""

        if not self.bucket:
            return False

        try:
            cache_key = self._generate_cache_key(analysis_type, project_type,
                                                 analysis_level, location, criteria)

            # Prepare cache data with metadata
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'project_type': project_type,
                'analysis_level': analysis_level,
                'location': location,
                'criteria': criteria,
                'response': response
            }

            # Upload to Google Cloud Storage
            blob = self.bucket.blob(cache_key)
            blob.upload_from_string(
                json.dumps(cache_data, indent=2),
                content_type='application/json'
            )

            print(f"Cached response: {cache_key}")
            return True

        except Exception as e:
            print(f"Cache storage error: {e}")
            return False

    def clear_location_cache(self, location: str, analysis_level: str = None) -> int:
        """Clear all cached responses for a specific location"""

        if not self.bucket:
            return 0

        try:
            clean_location = location.replace(' ', '_').replace(',', '_').lower()

            if analysis_level:
                prefix = f"ai_cache/{analysis_level}/"
            else:
                prefix = "ai_cache/"

            deleted_count = 0
            for blob in self.bucket.list_blobs(prefix=prefix):
                if clean_location in blob.name:
                    blob.delete()
                    deleted_count += 1

            print(f"Cleared {deleted_count} cached responses for {location}")
            return deleted_count

        except Exception as e:
            print(f"Cache clearing error: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""

        if not self.bucket:
            return {
                'error': 'Google Cloud Storage not available',
                'total_cached_responses': 0,
                'by_analysis_level': {'state': 0, 'county': 0, 'local': 0},
                'by_project_type': {'solar': 0, 'wind': 0}
            }

        try:
            stats = {
                'total_cached_responses': 0,
                'by_analysis_level': {'state': 0, 'county': 0, 'local': 0},
                'by_project_type': {'solar': 0, 'wind': 0},
                'oldest_cache': None,
                'newest_cache': None
            }

            cache_dates = []

            for blob in self.bucket.list_blobs(prefix="ai_cache/"):
                stats['total_cached_responses'] += 1

                # Parse blob name for statistics
                name_parts = blob.name.split('/')
                if len(name_parts) >= 4:
                    analysis_level = name_parts[2]
                    project_type = name_parts[3]

                    if analysis_level in stats['by_analysis_level']:
                        stats['by_analysis_level'][analysis_level] += 1

                    if project_type in stats['by_project_type']:
                        stats['by_project_type'][project_type] += 1

                # Track cache dates
                if blob.time_created:
                    cache_dates.append(blob.time_created)

            if cache_dates:
                stats['oldest_cache'] = min(cache_dates).isoformat()
                stats['newest_cache'] = max(cache_dates).isoformat()

            return stats

        except Exception as e:
            print(f"Cache stats error: {e}")
            return {'error': str(e)}