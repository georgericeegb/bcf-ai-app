import os
import logging
import json
import re
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass
from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    """Standardized AI response format"""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class AIServiceError(Exception):
    """Base exception for AI service errors"""
    pass

class UnifiedAIService:
    """
    Clean, unified AI service that consolidates all AI functionality
    """
    
    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.base_url = 'https://api.anthropic.com/v1/messages'
        self.anthropic_client = None
        self.connection_method = None
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize AI connection using the most reliable method"""
        if not self.api_key or not self.api_key.startswith('sk-ant-'):
            logger.error("Invalid or missing ANTHROPIC_API_KEY")
            raise AIServiceError("ANTHROPIC_API_KEY not properly configured")
        
        # Try anthropic client first (preferred method)
        if self._try_anthropic_client():
            self.connection_method = 'anthropic_client'
            logger.info("AI Service initialized with Anthropic client")
            return
        
        # Fallback to direct API calls
        if self._try_direct_api():
            self.connection_method = 'direct_api'
            logger.info("AI Service initialized with direct API calls")
            return
        
        # If both fail, raise error
        raise AIServiceError("Failed to initialize AI service with any method")
    
    def _try_anthropic_client(self) -> bool:
        """Try to initialize Anthropic client"""
        try:
            # Clear proxy environment variables that can interfere
            original_proxies = self._clear_proxy_vars()
            
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
            
            # Test the connection
            test_response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            # Restore proxy variables
            self._restore_proxy_vars(original_proxies)
            
            logger.info("Anthropic client test successful")
            return True
            
        except Exception as e:
            logger.warning(f"Anthropic client initialization failed: {e}")
            # Restore proxy variables even on error
            if 'original_proxies' in locals():
                self._restore_proxy_vars(original_proxies)
            return False
    
    def _try_direct_api(self) -> bool:
        """Try direct API calls as fallback"""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    'x-api-key': self.api_key,
                    'content-type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'test'}]
                },
                timeout=30
            )
            
            return response.status_code == 200
                
        except Exception as e:
            logger.warning(f"Direct API test failed: {e}")
            return False
    
    def _clear_proxy_vars(self) -> Dict[str, str]:
        """Clear proxy environment variables and return original values"""
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        original_values = {}
        
        for var in proxy_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
        
        return original_values
    
    def _restore_proxy_vars(self, original_values: Dict[str, str]):
        """Restore original proxy environment variables"""
        for var, value in original_values.items():
            os.environ[var] = value
    
    def test_connection(self) -> AIResponse:
        """Test the AI service connection"""
        try:
            response = self._make_request(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            
            if response.success:
                return AIResponse(
                    success=True,
                    content="Connection working",
                    metadata={"method": self.connection_method}
                )
            else:
                return AIResponse(success=False, error=response.error)
                
        except Exception as e:
            return AIResponse(success=False, error=str(e))
    
    def _make_request(self, messages: List[Dict], max_tokens: int = 4000, model: str = "claude-3-haiku-20240307") -> AIResponse:
        """Make AI request using the initialized connection method"""
        try:
            if self.connection_method == 'anthropic_client':
                return self._anthropic_client_request(messages, max_tokens, model)
            elif self.connection_method == 'direct_api':
                return self._direct_api_request(messages, max_tokens, model)
            else:
                raise AIServiceError("No valid connection method available")
                
        except Exception as e:
            logger.error(f"AI request failed: {e}")
            return AIResponse(success=False, error=str(e))
    
    def _anthropic_client_request(self, messages: List[Dict], max_tokens: int, model: str) -> AIResponse:
        """Make request using Anthropic client"""
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages
        )
        
        return AIResponse(
            success=True,
            content=response.content[0].text,
            metadata={"method": "anthropic_client", "model": model}
        )
   
    def _direct_api_request(self, messages: List[Dict], max_tokens: int, model: str) -> AIResponse:
        """Make request using direct API calls"""
        response = requests.post(
            self.base_url,
            headers={
                'x-api-key': self.api_key,
                'content-type': 'application/json',
                'anthropic-version': '2023-06-01'
            },
            json={
                'model': model,
                'max_tokens': max_tokens,
                'messages': messages
            },
            timeout=config.AI_REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return AIResponse(
                success=True,
                content=result['content'][0]['text'],
                metadata={"method": "direct_api", "model": model}
            )
        else:
            raise AIServiceError(f"API request failed: {response.status_code} - {response.text}")
    
    def analyze_state_counties(self, state: str, project_type: str, counties: List[Dict]) -> Optional[Dict]:
        """Analyze counties for renewable energy development potential"""
        if not counties:
            logger.warning("No counties provided for analysis")
            return None
        
        county_names = [c.get('name', f'County_{i}') for i, c in enumerate(counties)]
        
        prompt = f"""You are a renewable energy market analyst. Analyze and rank ALL counties in {state} for {project_type} energy development potential.

Counties to analyze: {', '.join(county_names)}

For EACH county listed above, provide a score 0-100 and return as JSON:
{{
    "analysis_summary": "Brief overview of {state}'s {project_type} energy landscape",
    "county_rankings": [
        {{
            "name": "County Name",
            "score": 85,
            "rank": 1,
            "strengths": ["Strong infrastructure", "Good resource quality"],
            "challenges": ["Limited land availability"],
            "resource_quality": "Excellent",
            "policy_environment": "Supportive", 
            "development_activity": "High",
            "summary": "Brief county assessment"
        }}
    ]
}}

Provide realistic scores based on typical renewable energy development factors."""

        try:
            response = self._make_request(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000
            )
            
            if not response.success:
                logger.error(f"County analysis failed: {response.error}")
                return self._create_fallback_analysis(state, project_type, counties)
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())
                    
                    # Add missing FIPS codes
                    for ranking in analysis_data.get('county_rankings', []):
                        county_name = ranking.get('name', '')
                        for county in counties:
                            if county.get('name', '').lower() in county_name.lower():
                                ranking['fips'] = county.get('fips', f'{state}999')
                                break
                    
                    logger.info(f"Successfully analyzed {len(analysis_data.get('county_rankings', []))} counties")
                    return analysis_data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    return self._create_fallback_analysis(state, project_type, counties)
            
            logger.warning("No valid JSON found in AI response")
            return self._create_fallback_analysis(state, project_type, counties)
            
        except Exception as e:
            logger.error(f"County analysis error: {e}")
            return self._create_fallback_analysis(state, project_type, counties)
    
    def analyze_county_market(self, county_name: str, state: str, project_type: str, county_fips: str = None) -> AIResponse:
        """Detailed market analysis for a specific county"""
        prompt = f"""Provide a comprehensive market analysis for {county_name} County, {state} for {project_type} energy development.

Include:
1. Market conditions and development potential
2. Resource quality assessment  
3. Regulatory and policy environment
4. Infrastructure and grid access
5. Competition and market saturation
6. Strategic recommendations
7. Risk factors and mitigation strategies
8. Timeline and next steps

Format as a detailed strategic report suitable for development planning."""

        try:
            response = self._make_request(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000
            )
            
            if response.success:
                logger.info(f"County market analysis completed for {county_name}, {state}")
            
            return response
            
        except Exception as e:
            logger.error(f"County market analysis failed: {e}")
            return AIResponse(success=False, error=str(e))

    def analyze_individual_parcel(self, parcel_data: Dict, county_name: str, state: str, project_type: str) -> AIResponse:
        """Detailed AI analysis for individual parcel development potential"""
        
        # Extract parcel information with fallbacks
        owner = parcel_data.get('owner', parcel_data.get('mail_name', 'Unknown Owner'))
        acres = parcel_data.get('acreage_calc', parcel_data.get('acreage', 0))
        slope = parcel_data.get('avg_slope_degrees', 'Unknown')
        tx_distance = parcel_data.get('tx_distance_miles', 'Unknown')
        tx_voltage = parcel_data.get('tx_voltage_kv', 'Unknown')
        address = parcel_data.get('address', 'Address not available')
        parcel_id = parcel_data.get('parcel_id', 'Unknown')
        
        prompt = f"""You are a senior renewable energy development analyst. Provide a comprehensive development assessment for this specific property:

PROPERTY DETAILS:
- Owner: {owner}
- Location: {address}, {county_name} County, {state}
- Parcel ID: {parcel_id}
- Size: {acres} acres
- Average Slope: {slope}°
- Transmission Distance: {tx_distance} miles
- Transmission Voltage: {tx_voltage} kV
- Project Type: {project_type.upper()} Energy Development

Provide a detailed professional development assessment that covers:

1. EXECUTIVE SUMMARY (2-3 sentences on overall viability)

2. TECHNICAL ANALYSIS
   - Terrain suitability and grading requirements
   - Grid connectivity and interconnection costs
   - Project scale and capacity potential
   - Construction considerations

3. ECONOMIC EVALUATION
   - Development cost factors
   - Revenue potential
   - Risk assessment
   - Expected timeline

4. STRATEGIC RECOMMENDATIONS
   - Priority level for development
   - Key next steps
   - Potential challenges and mitigation

5. MARKET CONTEXT
   - How this property compares to regional opportunities
   - Competitive advantages or disadvantages

Format this as a professional development memo suitable for executive review. Be specific about technical factors and provide actionable insights based on the property characteristics provided.

Focus on practical development considerations for {project_type} energy projects."""

        try:
            response = self._make_request(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                model='claude-3-5-sonnet-20241022' # Use Sonnet for more detailed analysis
            )
            
            if response.success:
                logger.info(f"Individual parcel analysis completed for {parcel_id} in {county_name}, {state}")
                
                # Return the analysis in the expected format
                return AIResponse(
                    success=True,
                    content=response.content,
                    metadata={
                        "parcel_id": parcel_id,
                        "county": county_name,
                        "state": state,
                        "project_type": project_type,
                        "analysis_type": "individual_parcel"
                    }
                )
            else:
                logger.error(f"Individual parcel analysis failed: {response.error}")
                return response
                
        except Exception as e:
            logger.error(f"Individual parcel analysis error: {e}")
            return AIResponse(success=False, error=str(e))

    def _create_fallback_analysis(self, state: str, project_type: str, counties: List[Dict]) -> Dict:
        """Create fallback analysis when AI is unavailable"""
        logger.info(f"Creating fallback analysis for {state} {project_type}")
        
        analyzed_counties = []
        for i, county in enumerate(counties):
            base_score = max(35, min(95, 75 - (i * 2)))
            
            analyzed_counties.append({
                'name': county.get('name', f'County_{i}'),
                'fips': county.get('fips', f'{state}{i:03d}'),
                'score': base_score,
                'rank': i + 1,
                'strengths': [f'{project_type.title()} potential', 'Infrastructure access'],
                'challenges': ['Requires detailed analysis'],
                'resource_quality': 'Good' if base_score >= 70 else 'Moderate',
                'policy_environment': 'Supportive',
                'development_activity': 'Medium',
                'summary': f'Moderate {project_type} development potential'
            })
        
        return {
            'analysis_summary': f'{state} {project_type} development analysis (Fallback Mode)',
            'county_rankings': analyzed_counties
        }
    
    def get_service_status(self) -> Dict:
        """Get current service status and configuration"""
        return {
            'initialized': self.connection_method is not None,
            'connection_method': self.connection_method,
            'api_key_configured': bool(self.api_key and self.api_key.startswith('sk-ant-')),
            'anthropic_client_available': self.anthropic_client is not None
        }

# Global AI service instance
ai_service = UnifiedAIService()