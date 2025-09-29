import logging
from typing import Optional
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from config.settings import config

logger = logging.getLogger(__name__)

class DatabaseClients:
    """Centralized database and cloud service clients"""
    
    def __init__(self):
        self._bigquery_client = None
        self._storage_client = None
        self._credentials = None
        
    @property
    def bigquery(self) -> Optional[bigquery.Client]:
        """Get BigQuery client (lazy initialization)"""
        if self._bigquery_client is None:
            try:
                if config.GOOGLE_APPLICATION_CREDENTIALS:
                    self._bigquery_client = bigquery.Client(project=config.GCP_PROJECT_ID)
                else:
                    # Use default credentials (for Cloud Run/App Engine)
                    self._bigquery_client = bigquery.Client()
                
                # Test connection
                query = "SELECT 1 as test"
                list(self._bigquery_client.query(query).result())
                logger.info("BigQuery client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize BigQuery client: {e}")
                self._bigquery_client = None
                
        return self._bigquery_client
    
    @property
    def storage(self) -> Optional[storage.Client]:
        """Get Cloud Storage client (lazy initialization)"""
        if self._storage_client is None:
            try:
                if config.GOOGLE_APPLICATION_CREDENTIALS:
                    self._storage_client = storage.Client(project=config.GCP_PROJECT_ID)
                else:
                    # Use default credentials
                    self._storage_client = storage.Client()
                
                # Test connection by checking if bucket exists
                bucket = self._storage_client.bucket(config.GCS_BUCKET_NAME)
                bucket.exists()  # This will raise an exception if we can't access it
                logger.info("Cloud Storage client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Cloud Storage client: {e}")
                self._storage_client = None
                
        return self._storage_client
    
    def test_connections(self) -> dict:
        """Test all database connections"""
        results = {
            'bigquery': False,
            'storage': False,
            'errors': []
        }
        
        # Test BigQuery
        try:
            if self.bigquery:
                query = "SELECT 1 as test"
                list(self.bigquery.query(query).result())
                results['bigquery'] = True
        except Exception as e:
            results['errors'].append(f"BigQuery: {str(e)}")
        
        # Test Cloud Storage
        try:
            if self.storage:
                bucket = self.storage.bucket(config.GCS_BUCKET_NAME)
                bucket.exists()
                results['storage'] = True
        except Exception as e:
            results['errors'].append(f"Storage: {str(e)}")
            
        return results

# Global database clients instance
db = DatabaseClients()