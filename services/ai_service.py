# services/ai_service.py - AI Analysis Service with Caching
import os
import logging
import anthropic
import json
from typing import Dict, List, Any
import traceback

logger = logging.getLogger(__name__)

# Import cache service with error handling
try:
    from services.cache_service import AIResponseCache

    CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache service not available: {e}")
    CACHE_AVAILABLE = False

# Import project config with error handling
try:
    from models.project_config import ProjectConfig

    PROJECT_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ProjectConfig not available: {e}")
    PROJECT_CONFIG_AVAILABLE = False


    # Create a fallback ProjectConfig class
    class ProjectConfig:
        @staticmethod
        def get_tier_criteria(analysis_level):
            return ["Resource Quality", "Market Opportunity", "Technical Feasibility", "Regulatory Environment"]

        @staticmethod
        def get_tier_description(analysis_level):
            descriptions = {
                'state': 'State-level market entry and opportunity assessment',
                'county': 'County-level development feasibility analysis',
                'site': 'Site-specific technical and commercial evaluation'
            }
            return descriptions.get(analysis_level, f"{analysis_level.title()}-level renewable energy analysis")


class AIAnalysisService:
    def __init__(self, api_key=None):
        logger.info("Initializing AI Analysis Service...")

        try:
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            logger.info(f"API key length: {len(self.api_key) if self.api_key else 0}")

            # Initialize cache with error handling
            self.cache = None
            if CACHE_AVAILABLE:
                try:
                    self.cache = AIResponseCache()
                    logger.info("Cache service initialized")
                except Exception as e:
                    logger.warning(f"Cache service failed to initialize: {e}")
                    self.cache = None

            if not self.api_key:
                logger.error("No Anthropic API key provided")
                self.client = None
                raise ValueError("ANTHROPIC_API_KEY is required")

            if not self.api_key.startswith('sk-ant-'):
                logger.error("Invalid Anthropic API key format")
                self.client = None
                raise ValueError("Invalid Anthropic API key format")

            # Initialize the Anthropic client with minimal parameters
            logger.info("Creating Anthropic client...")

            # FIXED: Create client with only essential parameters
            self.client = anthropic.Anthropic(
                api_key=self.api_key
                # Do not pass any other parameters like proxies, timeout, etc.
            )

            # Test the client immediately with a simple call
            logger.info("Testing Anthropic client connection...")
            test_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )

            logger.info("Anthropic client initialized and tested successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.client = None
            # Don't re-raise the exception, just set client to None
            # This allows the app to continue with fallback analysis

    def test_connection(self):
        """Test the Anthropic client connection"""
        if not self.client:
            return {"success": False, "error": "Client not initialized"}

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=20,
                messages=[{"role": "user", "content": "Connection test"}]
            )
            return {"success": True, "message": "Client working properly",
                    "response_length": len(response.content[0].text)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_connection(self):
        """Test the Anthropic client connection"""
        if not self.client:
            return {"success": False, "error": "Client not initialized"}

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=20,
                messages=[{"role": "user", "content": "Test connection"}]
            )
            return {"success": True, "message": "Client working properly"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def recommend_project_variables(self, project_type: str, analysis_level: str, location: str,
                                    selected_criteria: List[str] = None) -> Dict[str, Any]:
        """Recommend critical variables for renewable energy project"""

        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get_cached_response(
                analysis_type='variables',
                project_type=project_type,
                analysis_level=analysis_level,
                location=location,
                criteria=selected_criteria
            )
            if cached_response:
                return cached_response

        # Get tier-specific criteria and description
        tier_criteria = ProjectConfig.get_tier_criteria(analysis_level)
        tier_description = ProjectConfig.get_tier_description(analysis_level)

        # Use selected criteria if provided, otherwise use all tier criteria
        focus_criteria = selected_criteria if selected_criteria else tier_criteria

        prompt = f"""
        You are a renewable energy project development expert helping developers with {tier_description}.

        Project Context:
        - Type: {project_type} renewable energy project
        - Analysis Tier: {analysis_level.title()} Level
        - Location: {location}

        PRIORITY FOCUS AREAS (selected by developer):
        {chr(10).join([f"• {criteria}" for criteria in focus_criteria])}

        Based on the developer's selected focus areas above, recommend 8-10 critical variables that directly support analysis of these priority areas. Weight your recommendations heavily toward variables that help evaluate the selected focus areas.

        For each variable provide:
        1. Variable name (use industry standard terminology)
        2. Developer category (Resource, Market, Technical, Financial, Regulatory, Risk)
        3. Importance level (Critical, High, Medium) 
        4. Why developers care (impact on go/no-go decision)
        5. Typical evaluation method or data source
        6. Decision threshold or benchmark range
        7. How this specifically supports the selected focus areas

        Return as JSON:
        {{
            "tier_focus": "{tier_description}",
            "selected_focus_areas": {focus_criteria},
            "variables": [
                {{
                    "name": "Industry standard variable name",
                    "category": "Resource|Market|Technical|Financial|Regulatory|Risk", 
                    "importance": "Critical|High|Medium",
                    "developer_impact": "Why this drives developer decisions",
                    "evaluation_method": "How developers typically assess this",
                    "benchmark_range": "Typical thresholds or acceptable ranges",
                    "focus_area_relevance": "How this specifically supports the selected focus areas"
                }}
            ],
            "focus_area_insights": "Specific insights about the selected focus areas for this location",
            "next_steps": "What developers typically do after {analysis_level} level analysis"
        }}
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Changed to working model
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text

            # Parse JSON response
            import json
            result = json.loads(response_text)

            # Cache the successful response if cache is available
            if self.cache:
                self.cache.store_response(
                    analysis_type='variables',
                    project_type=project_type,
                    analysis_level=analysis_level,
                    location=location,
                    response=result,
                    criteria=selected_criteria
                )

            return result

        except Exception as e:
            print(f"AI service error: {e}")
            # Fallback response
            return self._get_fallback_variables(project_type, analysis_level)

    def analyze_state_counties(self, state, project_type, counties):
        """Generate comprehensive county rankings for all counties in state"""
        if not self.client:
            logger.error("No AI client available")
            return None

        # Get state name for better context
        state_names = {
            'AL': 'Alabama', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'FL': 'Florida', 'GA': 'Georgia', 'IL': 'Illinois',
            'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky',
            'LA': 'Louisiana', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'NE': 'Nebraska', 'NV': 'Nevada', 'NM': 'New Mexico',
            'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
            'OH': 'Ohio', 'OK': 'Oklahoma', 'PA': 'Pennsylvania', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VA': 'Virginia', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
        }

        state_full_name = state_names.get(state, state)

        # Create a comprehensive prompt that ensures all counties get scored
        county_names = [c.get('name', f'County_{i}') for i, c in enumerate(counties)]

        prompt = f"""
        You are a renewable energy market analyst. Analyze and rank ALL counties in {state_full_name} for {project_type} energy development potential.

        Counties to analyze: {', '.join(county_names)}

        For EACH county listed above, provide a score 0-100 considering:
        1. Renewable energy policies and incentives
        2. Grid infrastructure and transmission access  
        3. {project_type} resource quality (solar irradiance/wind speeds)
        4. Land availability and terrain suitability
        5. Permitting and regulatory environment
        6. Economic development priorities
        7. Existing renewable energy projects
        8. Community acceptance and political support

        You MUST rank ALL {len(county_names)} counties. Provide exactly this JSON format:

        {{
            "analysis_summary": "Brief overview of {state_full_name}'s {project_type} energy landscape and development potential",
            "county_rankings": [
                {{
                    "name": "County Name",
                    "fips": "county_fips_code",
                    "score": 85,
                    "rank": 1,
                    "strengths": ["Strong grid infrastructure", "Favorable policies", "Good resource quality"],
                    "challenges": ["Limited land availability"],
                    "summary": "Brief 1-2 sentence explanation of why this county ranks here",
                    "resource_quality": "Excellent",
                    "policy_environment": "Very Supportive", 
                    "development_activity": "High"
                }}
            ]
        }}

        Requirements:
        - Include ALL {len(county_names)} counties in your rankings
        - Scores should range from 30-95 to show meaningful differences
        - Rank from 1 to {len(county_names)} (1 = best)
        - Include 2-4 specific strengths per county
        - Include 1-2 challenges where relevant
        - Resource quality: Excellent/Very Good/Good/Fair/Poor
        - Policy environment: Very Supportive/Supportive/Neutral/Restrictive
        - Development activity: High/Medium/Low/None

        Focus on actionable intelligence for {project_type} project developers.
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,  # CHANGED: Reduced from 6000 to 4000 (safe limit)
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            analysis_text = response.content[0].text
            logger.info(f"AI Response received: {len(analysis_text)} characters")

            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())

                    # Add FIPS codes from our county data and validate completeness
                    returned_counties = analysis_data.get('county_rankings', [])

                    # Match with our county list and add missing FIPS
                    for ranking in returned_counties:
                        county_name = ranking.get('name', '')
                        # Match with our county list
                        for county in counties:
                            if county.get('name',
                                          '').lower() in county_name.lower() or county_name.lower() in county.get(
                                    'name', '').lower():
                                ranking['fips'] = county.get('fips', f'{state}999')
                                break

                    # If AI didn't return all counties, add the missing ones with default scores
                    returned_names = {r.get('name', '').lower() for r in returned_counties}
                    missing_counties = []

                    for county in counties:
                        county_name = county.get('name', '')
                        if county_name.lower() not in returned_names and not any(
                                county_name.lower() in name for name in returned_names):
                            missing_counties.append({
                                'name': county_name,
                                'fips': county.get('fips', f'{state}999'),
                                'score': 50,  # Default middle score
                                'rank': len(returned_counties) + len(missing_counties) + 1,
                                'strengths': ['Standard grid access', 'Basic infrastructure'],
                                'challenges': ['Limited analysis available'],
                                'summary': f"County requires additional market analysis for {project_type} development potential",
                                'resource_quality': 'Good',
                                'policy_environment': 'Neutral',
                                'development_activity': 'Low'
                            })

                    # Combine and re-sort
                    all_counties = returned_counties + missing_counties
                    all_counties.sort(key=lambda x: x.get('score', 0), reverse=True)

                    # Update ranks
                    for i, county in enumerate(all_counties):
                        county['rank'] = i + 1

                    analysis_data['county_rankings'] = all_counties

                    logger.info(
                        f"Generated rankings for {len(all_counties)} counties ({len(returned_counties)} from AI, {len(missing_counties)} added)")
                    return analysis_data

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    logger.error(f"Raw response: {analysis_text}")
                    return None

            else:
                logger.error("Failed to extract JSON from AI response")
                logger.error(f"Raw response: {analysis_text}")
                return None

        except Exception as e:
            logger.error(f"AI county analysis error: {e}")
            return None

    def analyze_focus_areas(self, project_type: str, analysis_level: str, location: str,
                            selected_criteria: List[str]) -> Dict[str, Any]:
        """Analyze selected focus areas for the chosen tier and location"""

        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get_cached_response(
                analysis_type='focus_areas',
                project_type=project_type,
                analysis_level=analysis_level,
                location=location,
                criteria=selected_criteria
            )
            if cached_response:
                return cached_response

        tier_description = ProjectConfig.get_tier_description(analysis_level)

        prompt = f"""
        As a renewable energy development expert, provide a concise analysis of the selected focus areas for this project:

        Project Context:
        - Type: {project_type.title()} renewable energy project
        - Analysis Level: {analysis_level.title()} Level ({tier_description})
        - Location: {location}

        Selected Focus Areas to Analyze:
        {chr(10).join([f"• {criteria}" for criteria in selected_criteria])}

        For each selected focus area, provide:
        1. Current status/score for {location} (1-10 scale)
        2. Key strengths and challenges
        3. Specific opportunities for {project_type} development
        4. Risk factors and mitigation strategies
        5. Actionable next steps

        Provide a concise, data-driven analysis that helps developers make {analysis_level}-level decisions.

        Return as JSON:
        {{
            "location": "{location}",
            "analysis_level": "{analysis_level}",
            "project_type": "{project_type}",
            "overall_assessment": {{
                "viability_score": 8.5,
                "confidence_level": "High|Medium|Low",
                "key_recommendation": "Primary recommendation for this location"
            }},
            "focus_area_analysis": [
                {{
                    "focus_area": "Name of focus area",
                    "score": 8.0,
                    "status": "Strong|Moderate|Weak",
                    "strengths": ["List of key strengths"],
                    "challenges": ["List of key challenges"],
                    "opportunities": ["Specific opportunities for development"],
                    "risks": ["Key risk factors"],
                    "next_steps": ["Actionable next steps"]
                }}
            ],
            "location_insights": {{
                "competitive_advantages": ["Location-specific advantages"],
                "market_context": "Market situation in this location",
                "timing_considerations": "Timing factors for development"
            }},
            "developer_recommendations": {{
                "immediate_actions": ["Actions for next 30 days"],
                "investment_thesis": "Key investment rationale",
                "success_factors": ["Critical factors for success"]
            }}
        }}
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text

            import json
            result = json.loads(response_text)

            # Cache the successful response if cache is available
            if self.cache:
                self.cache.store_response(
                    analysis_type='focus_areas',
                    project_type=project_type,
                    analysis_level=analysis_level,
                    location=location,
                    response=result,
                    criteria=selected_criteria
                )

            return result

        except Exception as e:
            print(f"Focus area analysis error: {e}")
            return {
                "error": f"Focus area analysis failed: {str(e)}",
                "location": location,
                "analysis_level": analysis_level,
                "project_type": project_type
            }

    def analyze_variable_deeply(self, project_config: ProjectConfig, variable_name: str, criteria: List[str]) -> Dict[
        str, Any]:
        """Perform deep analysis on a specific variable"""

        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get_cached_response(
                analysis_type='deep_variable',
                project_type=project_config.project_type,
                analysis_level=project_config.analysis_level,
                location=f"{project_config.location}_{variable_name}",
                criteria=criteria
            )
            if cached_response:
                return cached_response

        tier_context = ProjectConfig.get_tier_description(project_config.analysis_level)

        prompt = f"""
        As a renewable energy development expert, analyze the "{variable_name}" variable for a {project_config.project_type} project.

        Developer Context:
        - Analysis Tier: {project_config.analysis_level.title()} Level ({tier_context})
        - Location: {project_config.location}
        - Current Variable Values: {project_config.variables}
        - Focus Criteria: {', '.join(criteria) if criteria else 'Standard developer due diligence'}

        Provide analysis that helps developers make informed decisions:

        1. **Current Assessment**: How does this variable look for this project?
        2. **Developer Benchmarks**: How does this compare to industry standards/successful projects?
        3. **Risk Assessment**: What could go wrong and probability/impact?
        4. **Optimization Strategies**: Concrete steps to improve this variable
        5. **Sensitivity Analysis**: How sensitive is project success to changes in this variable?
        6. **Due Diligence Recommendations**: Next steps for deeper investigation
        7. **Deal Impact**: How this affects project financing/investment attractiveness

        Format as JSON:
        {{
            "variable_name": "{variable_name}",
            "current_assessment": {{
                "status": "Strong|Moderate|Weak|Unknown",
                "key_findings": "Current state analysis",
                "data_quality": "Assessment of available data"
            }},
            "developer_benchmarks": {{
                "industry_standard": "What good projects typically show",
                "comparison": "How this project compares",
                "percentile_ranking": "Estimated performance percentile"
            }},
            "risk_assessment": [
                {{
                    "risk": "Specific risk description",
                    "probability": "High|Medium|Low", 
                    "impact": "Project impact if risk occurs",
                    "mitigation": "Specific mitigation strategy",
                    "cost_to_mitigate": "Estimated cost/effort"
                }}
            ],
            "optimization_strategies": [
                {{
                    "strategy": "Specific optimization approach",
                    "implementation": "How to implement",
                    "expected_improvement": "Quantified improvement expected",
                    "timeline": "Implementation timeline",
                    "cost": "Estimated cost"
                }}
            ],
            "sensitivity_analysis": {{
                "sensitivity_level": "High|Medium|Low",
                "impact_analysis": "How changes affect project economics/viability",
                "key_thresholds": "Critical breakpoints to monitor"
            }},
            "due_diligence_next_steps": [
                "Specific next investigation steps for developers"
            ],
            "deal_impact": {{
                "financing_impact": "How this affects project financing",
                "investor_concerns": "What investors will focus on",
                "timeline_impact": "How this affects development timeline"
            }}
        }}
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Changed to working model
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text

            import json
            result = json.loads(response_text)

            # Cache the successful response if cache is available
            if self.cache:
                self.cache.store_response(
                    analysis_type='deep_variable',
                    project_type=project_config.project_type,
                    analysis_level=project_config.analysis_level,
                    location=f"{project_config.location}_{variable_name}",
                    response=result,
                    criteria=criteria
                )

            return result

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def comprehensive_project_analysis(self, project_config: ProjectConfig) -> Dict[str, Any]:
        """Run comprehensive analysis of entire project"""

        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get_cached_response(
                analysis_type='comprehensive',
                project_type=project_config.project_type,
                analysis_level=project_config.analysis_level,
                location=project_config.location,
                criteria=list(project_config.variables.keys()) if project_config.variables else None
            )
            if cached_response:
                return cached_response

        tier_context = ProjectConfig.get_tier_description(project_config.analysis_level)

        prompt = f"""
        Conduct a comprehensive renewable energy project analysis from a developer perspective:

        Project Configuration:
        - Type: {project_config.project_type}
        - Location: {project_config.location}
        - Analysis Tier: {project_config.analysis_level.title()} Level ({tier_context})
        - Variables Analyzed: {project_config.variables}

        Provide analysis structured like a developer investment committee presentation:

        1. **Executive Summary**: Go/No-Go recommendation with key rationale
        2. **Project Scoring**: Quantified assessment across key developer criteria  
        3. **Investment Thesis**: Why this project makes sense (or doesn't)
        4. **Risk Profile**: Major risks and mitigation strategies
        5. **Financial Outlook**: Revenue potential, cost drivers, returns
        6. **Development Pathway**: Next steps and timeline to COD
        7. **Competitive Positioning**: How this compares to other opportunities

        Format as JSON:
        {{
            "executive_summary": {{
                "recommendation": "GO|CONDITIONAL_GO|NO_GO",
                "confidence_level": "High|Medium|Low",
                "key_rationale": "Primary reasons for recommendation",
                "critical_assumptions": "Key assumptions driving recommendation"
            }},
            "project_scoring": {{
                "overall_score": 8.5,
                "resource_quality": {{ "score": 8.0, "notes": "Resource assessment" }},
                "market_opportunity": {{ "score": 7.5, "notes": "Market assessment" }},
                "technical_feasibility": {{ "score": 9.0, "notes": "Technical assessment" }},
                "financial_returns": {{ "score": 8.0, "notes": "Financial assessment" }},
                "development_risk": {{ "score": 7.0, "notes": "Risk assessment" }},
                "regulatory_pathway": {{ "score": 8.5, "notes": "Regulatory assessment" }}
            }},
            "investment_thesis": {{
                "key_strengths": ["Primary project advantages"],
                "value_drivers": ["What makes this project valuable"],
                "competitive_advantages": ["Unique project benefits"]
            }},
            "risk_profile": {{
                "overall_risk": "Low|Medium|High",
                "primary_risks": [
                    {{
                        "category": "Resource|Market|Technical|Financial|Regulatory",
                        "risk": "Specific risk description", 
                        "impact": "High|Medium|Low",
                        "probability": "High|Medium|Low",
                        "mitigation": "Mitigation strategy",
                        "residual_risk": "Risk after mitigation"
                    }}
                ],
                "risk_mitigation_cost": "Estimated cost of risk mitigation"
            }},
            "financial_outlook": {{
                "revenue_drivers": ["Key revenue sources"],
                "cost_structure": ["Major cost components"],
                "return_expectations": "Expected return profile",
                "financing_considerations": "Key financing factors"
            }},
            "development_pathway": {{
                "next_phase_requirements": ["What's needed for next development phase"],
                "estimated_timeline": "Development timeline to COD",
                "key_milestones": ["Critical development milestones"],
                "resource_requirements": "Capital and human resources needed"
            }},
            "competitive_positioning": {{
                "market_position": "How this project ranks vs alternatives",
                "timing_advantages": "Timing-related benefits",
                "strategic_value": "Strategic importance beyond returns"
            }},
            "action_items": {{
                "immediate_next_steps": ["Priority actions for next 30 days"],
                "phase_gate_decisions": ["Key decision points coming up"],
                "success_metrics": ["KPIs to track project progress"]
            }}
        }}
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Changed to working model
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text

            import json
            result = json.loads(response_text)

            # Cache the successful response if cache is available
            if self.cache:
                self.cache.store_response(
                    analysis_type='comprehensive',
                    project_type=project_config.project_type,
                    analysis_level=project_config.analysis_level,
                    location=project_config.location,
                    response=result,
                    criteria=list(project_config.variables.keys()) if project_config.variables else None
                )

            return result

        except Exception as e:
            return {"error": f"Comprehensive analysis failed: {str(e)}"}

    def _get_fallback_variables(self, project_type: str, analysis_level: str) -> Dict[str, Any]:
        """Fallback variables if AI service fails"""

        tier_description = ProjectConfig.get_tier_description(analysis_level)

        if project_type.lower() == 'solar':
            if analysis_level == 'state':
                variables = [
                    {
                        "name": "Solar Resource (GHI)",
                        "category": "Resource",
                        "importance": "Critical",
                        "developer_impact": "Primary driver of energy yield and project economics",
                        "evaluation_method": "NREL solar resource maps and PVLIB analysis",
                        "benchmark_range": ">4.5 kWh/m²/day for utility scale",
                        "tier_relevance": "State-level resource screening for market entry"
                    },
                    {
                        "name": "Renewable Portfolio Standard",
                        "category": "Regulatory",
                        "importance": "Critical",
                        "developer_impact": "Creates guaranteed market demand and pricing support",
                        "evaluation_method": "State energy policy analysis",
                        "benchmark_range": ">20% renewable target with solar carve-outs preferred",
                        "tier_relevance": "Determines state-level market opportunity"
                    }
                ]
            else:
                variables = [
                    {
                        "name": "Land Availability",
                        "category": "Technical",
                        "importance": "Critical",
                        "developer_impact": "Determines maximum project capacity and development feasibility",
                        "evaluation_method": "GIS analysis and land use mapping",
                        "benchmark_range": "5-8 acres per MW for utility scale",
                        "tier_relevance": f"Essential for {analysis_level}-level site selection"
                    }
                ]
        else:  # wind
            variables = [
                {
                    "name": "Wind Resource (Class)",
                    "category": "Resource",
                    "importance": "Critical",
                    "developer_impact": "Determines capacity factor and project economics",
                    "evaluation_method": "Wind resource maps and met tower data",
                    "benchmark_range": "Class 4+ (>7.0 m/s) for utility scale",
                    "tier_relevance": f"Primary screening criterion at {analysis_level} level"
                }
            ]

        return {
            "tier_focus": tier_description,
            "variables": variables,
            "next_steps": f"Proceed to detailed {analysis_level}-level due diligence",
            "key_decisions": f"Market entry feasibility at {analysis_level} level"
        }

    # Cache management methods (only if cache is available)
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        if self.cache:
            return self.cache.get_cache_stats()
        else:
            return {"error": "Cache service not available"}

    def clear_location_cache(self, location: str, analysis_level: str = None) -> int:
        """Clear cached responses for a location"""
        if self.cache:
            return self.cache.clear_location_cache(location, analysis_level)
        else:
            return 0

    def analyze_next_step_implementation(self, next_step: str, project_type: str, analysis_level: str, location: str,
                                         category: str) -> Dict[str, Any]:
        """Provide detailed implementation guidance for a specific next step"""

        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get_cached_response(
                analysis_type='next_step',
                project_type=project_type,
                analysis_level=analysis_level,
                location=f"{location}_{next_step[:50]}",  # Truncate for cache key
                criteria=[category]
            )
            if cached_response:
                return cached_response

        tier_description = ProjectConfig.get_tier_description(analysis_level)

        prompt = f"""
        As a renewable energy development expert, provide detailed implementation guidance for this specific next step:

        Next Step: "{next_step}"

        Project Context:
        - Type: {project_type.title()} renewable energy project
        - Location: {location}
        - Analysis Level: {analysis_level.title()} Level ({tier_description})
        - Category: {category}

        Provide comprehensive, actionable guidance for implementing this next step in {location}. Your response should be practical and specific to this location and project type.

        Include the following information:

        1. **Step-by-step implementation process**
        2. **Key contacts and organizations** specific to {location}
        3. **Required documentation and permits**
        4. **Estimated timeline and costs**
        5. **Common challenges and how to avoid them**
        6. **Pro tips for success**

        Return as JSON:
        {{
            "next_step": "{next_step}",
            "location": "{location}",
            "project_type": "{project_type}",
            "implementation_steps": [
                {{
                    "action": "Specific action to take",
                    "details": "Detailed explanation of how to do this",
                    "timeline": "Expected timeframe",
                    "resources": "Resources needed or helpful websites/contacts"
                }}
            ],
            "key_contacts": [
                {{
                    "organization": "Organization name",
                    "role": "What they do/why contact them",
                    "website": "Website URL if available",
                    "phone": "Phone number if available",
                    "notes": "Additional notes or tips for contact"
                }}
            ],
            "requirements": [
                "List of specific requirements, permits, or documents needed"
            ],
            "estimated_timeline": "Overall timeline for completing this step",
            "estimated_cost": "Cost range or specific costs if known",
            "tips": [
                "Pro tips for success based on common practices"
            ],
            "potential_challenges": [
                "Common challenges and how to mitigate them"
            ],
            "location_specific_notes": "Notes specific to {location} that affect this process"
        }}
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=3500,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text

            import json
            result = json.loads(response_text)

            # Cache the successful response if cache is available
            if self.cache:
                self.cache.store_response(
                    analysis_type='next_step',
                    project_type=project_type,
                    analysis_level=analysis_level,
                    location=f"{location}_{next_step[:50]}",
                    response=result,
                    criteria=[category]
                )

            return result

        except Exception as e:
            print(f"Next step analysis error: {e}")
            return {
                "error": f"Implementation guidance failed: {str(e)}",
                "next_step": next_step,
                "location": location,
                "project_type": project_type
            }