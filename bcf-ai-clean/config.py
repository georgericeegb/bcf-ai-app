import os
import logging

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded successfully")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Falling back to system environment variables")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        # ReportAll API configuration - CORRECTED URL
        self.RAUSA_CLIENT_KEY = os.getenv('RAUSA_CLIENT_KEY')
        self.RAUSA_API_VERSION = os.getenv('RAUSA_API_VERSION', '9')
        self.RAUSA_API_URL = os.getenv('RAUSA_API_URL', 'https://reportallusa.com/api/parcels')  # FIXED URL

        # GCS configuration
        self.BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'bcfparcelsearchrepository')

        # Debug logging
        logger.info(f"Config initialized:")
        logger.info(f"  - API Key set: {bool(self.RAUSA_CLIENT_KEY)}")
        logger.info(f"  - API URL: {self.RAUSA_API_URL}")
        logger.info(f"  - API Version: {self.RAUSA_API_VERSION}")
        logger.info(f"  - Bucket: {self.BUCKET_NAME}")

        # Additional validation
        if not self.RAUSA_CLIENT_KEY:
            logger.warning("❌ RAUSA_CLIENT_KEY not found in environment variables!")
            logger.warning("   Make sure your .env file contains: RAUSA_CLIENT_KEY=your-api-key")
        else:
            logger.info(f"✅ API Key found: ***{self.RAUSA_CLIENT_KEY[-4:]}")

    def get(self, key, default=None):
        """Get configuration value with optional default"""
        return getattr(self, key, default)

    def is_cloud_environment(self):
        """Check if running in cloud environment"""
        return bool(os.getenv('GOOGLE_CLOUD_PROJECT'))


# Create global config instance
config = Config()


def get_config():
    return config