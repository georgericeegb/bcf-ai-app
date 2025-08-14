# 🌱 Renewable Energy Project Analyzer

A comprehensive web application for analyzing renewable energy project feasibility and identifying suitable land parcels with AI-powered insights and advanced suitability analysis.

## Features

### Core Analysis Capabilities
- **Multi-tier Analysis**: State, County, and Local level renewable energy feasibility analysis
- **AI-Powered Insights**: Focus area analysis for solar and wind projects using Anthropic Claude
- **Parcel Suitability Analysis**: Advanced slope and transmission line proximity analysis
- **Parcel Search**: Integration with ReportAll USA for nationwide parcel data retrieval

### Data Management
- **GCS Integration**: Secure cloud storage for search results and analysis outputs
- **Multi-format Export**: CSV and GeoPackage formats for different use cases
- **Preview & Download**: View data before downloading with multiple format support
- **Real-time Processing**: Live analysis with progress indicators

### User Experience
- **Interactive UI**: Modern, responsive web interface with Bootstrap 5
- **Workflow-driven**: Step-by-step guided process from analysis to parcel selection
- **Caching**: Intelligent response caching for improved performance
- **Error Handling**: Comprehensive error handling with user-friendly messages

## Tech Stack

- **Backend**: Python 3.8+, Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing**: Pandas, GeoPandas, Shapely
- **Spatial Analysis**: BigQuery integration for slope and transmission analysis
- **Cloud Storage**: Google Cloud Storage
- **External APIs**: ReportAll USA, Anthropic Claude
- **Styling**: Bootstrap 5, Font Awesome

## Setup

### Prerequisites
- Python 3.8+
- Google Cloud Storage account with appropriate permissions
- ReportAll USA API credentials
- Anthropic API key (optional, for AI analysis)
- BigQuery access (for suitability analysis)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/renewable-energy-analyzer.git
cd renewable-energy-analyzer
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Run the application:
```bash
python app.py
```

## Configuration

Create a `.env` file with the following variables:

```bash
# ReportAll API
RAUSA_CLIENT_KEY=your_reportall_key
RAUSA_API_URL=https://reportallusa.com/api/parcels
RAUSA_API_VERSION=9

# Google Cloud Storage
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# OR use JSON string directly:
# GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
CACHE_BUCKET_NAME=your-bucket-name

# Optional: Anthropic API for AI analysis
ANTHROPIC_API_KEY=sk-ant-your-key

# Flask Configuration
SECRET_KEY=your-secret-key
FLASK_ENV=development
PORT=8080
```

## Usage Workflow

### 1. Project Configuration
- **Select Project Type**: Choose between Solar or Wind projects
- **Choose Analysis Tier**: State, County, or Local level analysis
- **Select Criteria**: Pick focus areas for analysis (regulatory, technical, economic, etc.)

### 2. Location Selection
- **Choose Location**: Select your target area (state/county/local)
- **Generate Analysis**: Get AI-powered feasibility insights
- **Review Results**: Examine focus area analysis and recommendations

### 3. Parcel Identification
- **Search Parcels**: Find suitable land parcels using various criteria:
  - Minimum/maximum acreage
  - Owner name searches
  - Specific parcel ID lookup
- **Preview Results**: See estimated parcel counts before full search

### 4. Suitability Analysis
- **Analyze Parcels**: Run advanced suitability analysis including:
  - Slope analysis (terrain suitability)
  - Transmission line proximity
  - Combined scoring algorithms
- **Review Scored Results**: Filter and sort parcels by suitability scores
- **Export Data**: Download results in multiple formats

### 5. Data Export
- **Preview Files**: View data structure before download
- **Download Options**: 
  - CSV format for spreadsheet analysis
  - GeoPackage format for GIS applications
- **Generate Reports**: Create comprehensive feasibility reports

## File Structure

```
renewable-energy-analyzer/
├── app.py                          # Main Flask application
├── enhanced_parcel_search.py       # Parcel search logic with GCS integration
├── bigquery_slope_analysis.py      # Slope analysis using BigQuery
├── transmission_analysis_bigquery.py # Transmission line analysis
├── config.py                       # Configuration management
├── counties-trimmed.json           # County FIPS mappings
├── services/
│   ├── ai_service.py               # Anthropic Claude integration
│   └── cache_service.py            # Response caching
├── models/
│   └── project_config.py           # Project configuration models
├── templates/
│   ├── index.html                  # Main application interface
│   ├── file_preview.html           # File preview template
│   └── cache_dashboard.html        # Cache management dashboard
├── static/                         # Static assets
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── README.md                       # This file
```

## API Endpoints

### Analysis Endpoints
- `GET /` - Main application interface
- `GET /api/tier-info` - Get analysis tier information
- `POST /api/analyze-focus-areas` - Generate AI feasibility analysis
- `POST /api/analyze-next-step` - Get implementation guidance

### Parcel Search Endpoints
- `POST /api/parcel-search/preview` - Preview search results count
- `POST /api/parcel-search/execute` - Execute full parcel search
- `GET /api/counties/<state>` - Get counties for a state

### Suitability Analysis Endpoints
- `POST /api/parcel/analyze_suitability` - Run parcel suitability analysis

### File Management Endpoints
- `GET /preview/<path:file_path>` - Preview files in browser
- `GET /download/<path:file_path>` - Download files directly
- `GET /api/parcel-data-preview` - Get JSON preview of parcel data

### Utility Endpoints
- `GET /health` - Health check for Cloud Run deployment
- `GET /api/cloud-storage/status` - Check GCS configuration
- `GET /api/cache/stats` - Cache usage statistics

## Data Processing Pipeline

1. **Parcel Search**: ReportAll USA API → Raw parcel data
2. **Data Cleaning**: JSON serialization fixes, coordinate validation
3. **Spatial Processing**: Convert to GeoDataFrame, create geometries
4. **Cloud Storage**: Save to GCS in multiple formats
5. **Suitability Analysis**: 
   - BigQuery slope analysis
   - Transmission line proximity analysis
   - Combined scoring algorithms
6. **Results Export**: Filtered, scored, and formatted results

## Troubleshooting

### Common Issues

**JSON Parse Errors**
- Usually caused by NaN or infinity values in large datasets
- Fixed by data cleaning in `enhanced_parcel_search.py`

**GCS Authentication**
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account key
- Alternative: Use `GOOGLE_SERVICE_ACCOUNT_JSON` environment variable

**ReportAll API Issues**
- Verify `RAUSA_CLIENT_KEY` is valid
- Check county ID mappings in `config.py`

**Suitability Analysis Failures**
- Ensure BigQuery access is configured
- Verify parcel data has valid coordinates

### Performance Optimization

- **Caching**: AI responses are cached to reduce API costs
- **Batching**: Large parcel datasets are processed in batches
- **Lazy Loading**: Files are loaded on-demand from GCS

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Add logging for debugging
- Include error handling for external API calls
- Test with various dataset sizes
- Document any new environment variables

## Deployment

### Local Development
```bash
python app.py
```

### Cloud Run Deployment
```bash
gcloud run deploy renewable-energy-analyzer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ReportAll USA for parcel data
- Anthropic for AI analysis capabilities
- Google Cloud Platform for infrastructure
- Bootstrap and Font Awesome for UI components