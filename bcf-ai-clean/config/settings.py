import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY: str = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')
    DEBUG: bool = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 8080))
    
    # AI Service settings
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    
    # Google Cloud settings
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    GCP_PROJECT_ID: Optional[str] = os.getenv('GCP_PROJECT_ID')
    
    # Storage settings
    GCS_BUCKET_NAME: str = os.getenv('GCS_BUCKET_NAME', 'bcfparcelsearchrepository')
    
    # BigQuery settings
    BQ_DATASET_COUNTIES: str = os.getenv('BQ_DATASET_COUNTIES', 'county_data')
    BQ_DATASET_ANALYSIS: str = os.getenv('BQ_DATASET_ANALYSIS', 'renewable_energy')
    BQ_DATASET_TRANSMISSION: str = os.getenv('BQ_DATASET_TRANSMISSION', 'transmission_analysis')
    BQ_DATASET_SLOPE: str = os.getenv('BQ_DATASET_SLOPE', 'spatial_analysis')
    
    # Authentication
    ADMIN_PASSWORD: str = os.getenv('ADMIN_PASSWORD', 'admin123')
    
    # Rate limiting
    AI_REQUEST_TIMEOUT: int = int(os.getenv('AI_REQUEST_TIMEOUT', 60))
    MAX_PARCELS_PER_SEARCH: int = int(os.getenv('MAX_PARCELS_PER_SEARCH', 10000))
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required")
        elif not self.ANTHROPIC_API_KEY.startswith('sk-ant-'):
            errors.append("ANTHROPIC_API_KEY format is invalid")
            
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            errors.append("GOOGLE_APPLICATION_CREDENTIALS is required")
            
        if not self.GCP_PROJECT_ID:
            errors.append("GCP_PROJECT_ID is required")
            
        return errors
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('GAE_ENV') == 'standard' or os.getenv('ENV') == 'production'

# Global config instance
config = Config()

# Validate on import
config_errors = config.validate()
if config_errors:
    print("Configuration errors found:")
    for error in config_errors:
        print(f"  - {error}")
    if config.is_production:
        raise RuntimeError("Invalid configuration in production environment")