# BCF.ai - Renewable Energy Project Analyzer

## Overview

BCF.ai is an AI-powered web application designed to streamline renewable energy project development by providing comprehensive market analysis, land parcel identification, and development feasibility assessments. The platform combines artificial intelligence with geospatial data analysis to help developers identify optimal locations for solar, wind, and battery storage projects.

## Key Features

### 🗺️ State-Level Market Analysis
- **AI-Powered County Rankings**: Automated analysis of all counties within a target state
- **Comprehensive Scoring**: Counties ranked 0-100 based on renewable energy development potential
- **Multi-Factor Assessment**: Evaluation includes resource quality, policy environment, grid infrastructure, land availability, and regulatory factors
- **Interactive County Cards**: Visual county comparison with detailed metrics and activity indicators

### 🧠 AI Market Intelligence
- **County-Specific Analysis**: Detailed AI-generated market assessments for individual counties
- **Strategic Recommendations**: Actionable insights for market entry and development strategies
- **Risk Assessment**: Identification of challenges and mitigation strategies
- **Competitive Positioning**: Analysis of market opportunities and timing considerations

### 🔍 Advanced Parcel Search
- **Intelligent Land Discovery**: Search parcels by acreage, owner, location, and custom criteria
- **Cloud Storage Integration**: Automatic file storage and retrieval via Google Cloud Storage
- **Existing Data Management**: Access and analyze previously searched parcel datasets
- **Preview Capabilities**: Quick preview of search results before full analysis

### 📊 Technical Analysis Pipeline
- **Slope Analysis**: Automated terrain assessment using elevation data
- **Transmission Line Proximity**: Distance and voltage analysis for grid interconnection planning
- **Suitability Scoring**: ML-enhanced parcel scoring based on multiple development factors
- **BigQuery Integration**: Scalable data processing and storage for large datasets

### 🎯 Parcel Analysis & Selection
- **AI-Powered Individual Analysis**: Detailed feasibility assessment for specific parcels
- **Interactive Selection Tools**: Batch selection and filtering capabilities
- **Outreach Management**: CRM-ready parcel data for landowner engagement
- **Export Functionality**: CSV and GPKG file downloads for external analysis

## Technology Stack

### Backend
- **Flask**: Python web framework with RESTful API architecture
- **Anthropic Claude API**: AI-powered market analysis and insights
- **Google Cloud Platform**: 
  - BigQuery for data warehousing and analytics
  - Cloud Storage for file management and caching
- **GeoPandas**: Geospatial data processing and analysis
- **PostgreSQL**: Structured data storage and management

### Frontend
- **Responsive Design**: Mobile-first UI with Bootstrap 5
- **Interactive Workflows**: Step-by-step project development guidance
- **Real-Time Updates**: Live progress tracking for analysis operations
- **Advanced Data Visualization**: County rankings, parcel maps, and analytical dashboards

### Infrastructure
- **Cloud-Native Architecture**: Designed for scalable deployment
- **Caching System**: AI response caching for improved performance
- **Error Handling**: Comprehensive logging and user-friendly error management
- **Security**: Environment-based configuration and secure API key management

## Project Structure

```
bcf-ai/
├── app.py                          # Main Flask application
├── services/
│   ├── ai_service.py              # AI analysis service (Claude API)
│   ├── census_api.py              # Census data integration
│   └── enhanced_parcel_ai_service.py  # Individual parcel AI analysis
├── templates/
│   ├── index.html                 # Main application interface
│   └── login.html                 # Authentication interface
├── bigquery_transmission_storage.py   # BigQuery data management
├── transmission_analysis_bigquery.py  # Transmission line analysis
├── enhanced_parcel_search.py      # Advanced parcel search functionality
└── requirements.txt               # Python dependencies
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Google Cloud Platform account with BigQuery and Cloud Storage enabled
- Anthropic Claude API key

### Environment Variables
Create a `.env` file with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
FLASK_SECRET_KEY=your_secret_key
CACHE_BUCKET_NAME=your_gcs_bucket_name
```

### Installation
```bash
pip install -r requirements.txt
python app.py
```

## Usage Workflow

1. **State Selection**: Choose target state and project type (solar/wind/battery)
2. **County Analysis**: AI analyzes and ranks all counties in the selected state
3. **County Selection**: Choose high-potential counties for detailed investigation
4. **Market Research**: Run AI market analysis for specific counties
5. **Parcel Search**: Search for suitable land parcels using advanced criteria
6. **Technical Analysis**: Automated slope and transmission analysis
7. **Selection & Outreach**: Select optimal parcels and export for CRM integration

## Recent Updates

- Fixed AI market analysis modal display functionality
- Enhanced error handling and debugging capabilities
- Improved user interface responsiveness and visual feedback
- Added comprehensive parcel selection and management tools
- Integrated advanced geospatial analysis pipeline

## API Endpoints

- `/api/analyze-state-counties` - Generate AI-powered county rankings
- `/api/county-market-analysis` - Detailed county market assessment
- `/api/execute-parcel-search` - Advanced parcel search functionality
- `/api/analyze-existing-file-bq` - BigQuery-based parcel analysis
- `/api/individual-parcel-analysis` - AI analysis for specific parcels

## Contributing

This is a proprietary renewable energy development tool. For questions or contributions, contact the development team.

---

## Commit Message

```
🔧 Fix AI market analysis modal display and enhance debugging

BREAKING CHANGES:
- Fixed displayAIMarketAnalysisResults function to properly show analysis results to users
- Enhanced runAIMarketAnalysis method with comprehensive error handling and debugging
- Replaced Bootstrap modal dependency with custom modal implementation
- Added extensive logging and DOM verification for troubleshooting

IMPROVEMENTS:
- Custom modal styling with inline CSS for better compatibility
- Direct DOM insertion using insertAdjacentHTML for reliable display
- Enhanced debugging output for function existence and execution tracking
- Improved error messages and user feedback during analysis operations

TECHNICAL DETAILS:
- Removed dependency on Bootstrap modal system that was causing display failures
- Added return value tracking and DOM verification for modal elements
- Implemented timeout-based DOM checking for modal visibility confirmation
- Enhanced browser console debugging capabilities for development

TESTING:
- Verified AI API calls working correctly with proper response handling
- Confirmed modal display functionality across different browser environments
- Tested county selection and analysis workflow end-to-end
- Validated error handling and fallback mechanisms

This fix resolves the critical issue where AI market analysis was working in the 
backend but results were not being displayed to users. The solution provides a 
more robust and debuggable modal system for showing analysis results.

Files modified:
- index.html (displayAIMarketAnalysisResults, runAIMarketAnalysis methods)
- Enhanced error handling and logging throughout the workflow
```
