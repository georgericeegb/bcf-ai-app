# services/enhanced_parcel_ai_service.py
import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import anthropic

logger = logging.getLogger(__name__)


class EnhancedParcelAIService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("No Anthropic API key provided")
            self.client = None
        else:
            try:
                # Updated initialization for newer anthropic library
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Enhanced Parcel AI Service initialized")
            except Exception as e:
                logger.error(f"Anthropic client initialization failed: {e}")
                self.client = None

    def analyze_parcel_suitability_batch(self, parcels_data: List[Dict],
                                         project_type: str = 'solar',
                                         location: str = 'Unknown') -> List[Dict]:
        """
        Analyze multiple parcels for renewable energy suitability using AI

        Args:
            parcels_data: List of parcel dictionaries with all available data
            project_type: Type of renewable energy project (solar, wind, battery)
            location: Location context for analysis

        Returns:
            List of parcels with enhanced AI analysis
        """
        if not self.client:
            logger.error("AI client not available, using fallback analysis")
            return self._fallback_batch_analysis(parcels_data, project_type)

        # Process in chunks to avoid token limits
        chunk_size = 15  # Analyze 15 parcels at a time for detailed analysis
        all_results = []

        for i in range(0, len(parcels_data), chunk_size):
            chunk = parcels_data[i:i + chunk_size]
            logger.info(f"Processing AI analysis chunk {i // chunk_size + 1} ({len(chunk)} parcels)")

            try:
                chunk_results = self._analyze_parcel_chunk(chunk, project_type, location)
                all_results.extend(chunk_results)
            except Exception as e:
                logger.error(f"AI analysis failed for chunk {i // chunk_size + 1}: {e}")
                # Use fallback for failed chunks
                fallback_results = self._fallback_batch_analysis(chunk, project_type)
                all_results.extend(fallback_results)

        logger.info(f"Completed AI analysis for {len(all_results)} parcels")
        return all_results

    def _analyze_parcel_chunk(self, parcels_chunk: List[Dict],
                              project_type: str, location: str) -> List[Dict]:
        """Analyze a chunk of parcels using AI"""

        # Prepare parcel summaries for AI analysis
        parcel_summaries = []
        for i, parcel in enumerate(parcels_chunk):
            summary = self._create_parcel_summary(parcel, i)
            parcel_summaries.append(summary)

        prompt = self._build_comprehensive_analysis_prompt(
            parcel_summaries, project_type, location
        )

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Parse AI response and merge with original data
            ai_analysis = self._parse_ai_response(response_text)

            # Merge AI analysis back with original parcel data
            enhanced_parcels = []
            for i, parcel in enumerate(parcels_chunk):
                enhanced_parcel = parcel.copy()

                # Add AI analysis if available
                if i < len(ai_analysis.get('parcel_analyses', [])):
                    ai_data = ai_analysis['parcel_analyses'][i]
                    enhanced_parcel.update(self._format_ai_analysis(ai_data))
                else:
                    # Fallback if AI didn't analyze this parcel
                    enhanced_parcel.update(self._fallback_single_analysis(parcel, project_type))

                enhanced_parcels.append(enhanced_parcel)

            return enhanced_parcels

        except Exception as e:
            logger.error(f"AI chunk analysis failed: {e}")
            return self._fallback_batch_analysis(parcels_chunk, project_type)

    def _create_parcel_summary(self, parcel: Dict, index: int) -> str:
        """Create a concise summary of parcel data for AI analysis"""

        # Extract key fields with defaults
        parcel_id = parcel.get('parcel_id', f'parcel_{index}')
        acreage = parcel.get('acreage_calc', parcel.get('acreage', 0))
        land_cover = parcel.get('land_cover', 'Unknown')
        land_value = parcel.get('mkt_val_land', parcel.get('mkt_land_val', 0))
        land_use_code = parcel.get('land_use_code', 'Unknown')
        land_use_class = parcel.get('land_use_class', 'Unknown')
        adjacent_acres = parcel.get('acreage_adjacent_with_sameowner', 0)
        buildings = parcel.get('buildings', 0)
        flood_zone = parcel.get('fld_zone', 'Unknown')
        zone_subtype = parcel.get('zone_subty', 'Unknown')

        # Transmission analysis results
        tx_distance = parcel.get('tx_nearest_distance', 'Unknown')
        tx_voltage = parcel.get('tx_primary_voltage', parcel.get('tx_max_voltage', 'Unknown'))
        tx_lines_count = parcel.get('tx_lines_count', 0)

        # Slope analysis results
        avg_slope = parcel.get('avg_slope_degrees', 'Unknown')
        slope_category = parcel.get('slope_category', 'Unknown')
        buildability = parcel.get('buildability_5_35', 'Unknown')

        # Owner information
        owner = parcel.get('owner', 'Unknown')

        summary = f"""
Parcel {index}: {parcel_id}
Size: {acreage} acres (+ {adjacent_acres} adjacent same-owner acres)
Land: {land_cover}, {land_use_class} ({land_use_code})
Value: ${land_value:,} land value
Buildings: {buildings}
Flood Zone: {flood_zone}, Zoning: {zone_subtype}
Owner: {owner}
Transmission: {tx_distance} miles to nearest line ({tx_voltage}kV), {tx_lines_count} lines nearby
Terrain: {avg_slope}° average slope, {slope_category}, {buildability} buildability
        """.strip()

        return summary

    def _build_comprehensive_analysis_prompt(self, parcel_summaries: List[str],
                                             project_type: str, location: str) -> str:
        """Build comprehensive AI analysis prompt"""

        prompt = f"""
You are a senior renewable energy development analyst specializing in {project_type} projects. Analyze these parcels in {location} for commercial-scale {project_type} development suitability.

PROJECT CONTEXT:
- Technology: {project_type.title()} renewable energy
- Location: {location}
- Analysis Level: Site-specific development feasibility
- Target: Commercial/utility-scale development

PARCELS TO ANALYZE:
{chr(10).join(parcel_summaries)}

ANALYSIS FRAMEWORK:
For each parcel, evaluate these critical factors:

1. **Size & Expansion Potential** (25% weight)
   - Primary parcel size adequacy for {project_type}
   - Adjacent same-owner land for expansion
   - Total developable area potential

2. **Land Characteristics** (20% weight)
   - Land use compatibility with {project_type} development
   - Land cover and existing improvements
   - Zoning and regulatory constraints

3. **Technical Feasibility** (25% weight)
   - Slope/terrain suitability for {project_type}
   - Flood risk and environmental constraints
   - Site accessibility and buildability

4. **Grid Infrastructure** (20% weight)
   - Distance to transmission lines
   - Voltage level and interconnection potential
   - Grid access complexity and cost

5. **Development Economics** (10% weight)
   - Land acquisition cost vs. market rates
   - Existing structures requiring removal
   - Owner type and acquisition complexity

SCORING CRITERIA:
- **EXCELLENT (85-100)**: Prime development site with minimal constraints
- **GOOD (70-84)**: Strong development potential with manageable challenges  
- **FAIR (55-69)**: Viable with significant mitigation required
- **POOR (0-54)**: Major constraints make development questionable

For each parcel, provide:
1. Overall suitability score (0-100)
2. Suitability category (EXCELLENT/GOOD/FAIR/POOR)
3. Top 3 strengths for {project_type} development
4. Top 3 challenges or risks
5. Development complexity (LOW/MEDIUM/HIGH)
6. Investment priority (HIGH/MEDIUM/LOW)
7. Key next steps for development
8. Estimated development timeline (FAST/STANDARD/SLOW/CHALLENGING)

Return as JSON:
{{
    "analysis_summary": "Overall assessment of this parcel group for {project_type} development",
    "location_context": "Key factors affecting {project_type} development in {location}",
    "parcel_analyses": [
        {{
            "parcel_index": 0,
            "suitability_score": 85,
            "suitability_category": "EXCELLENT",
            "strengths": ["Large contiguous area", "Excellent grid access", "Flat terrain"],
            "challenges": ["High land value", "Existing structures", "Flood zone mapping"],
            "development_complexity": "LOW",
            "investment_priority": "HIGH", 
            "next_steps": ["Conduct grid interconnection study", "Verify zoning compliance", "Initiate landowner discussions"],
            "development_timeline": "STANDARD",
            "technical_notes": "Specific technical considerations for this parcel",
            "economic_notes": "Key economic factors and cost drivers",
            "risk_assessment": "Primary risks and mitigation strategies"
        }}
    ],
    "comparative_insights": "How these parcels compare to each other and typical {project_type} sites",
    "portfolio_recommendations": "Strategic recommendations for this parcel portfolio"
}}

Focus on actionable intelligence that helps developers make investment and development decisions.
        """

        return prompt

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response with robust error handling"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.error("No JSON found in AI response")
                return {'parcel_analyses': []}
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return {'parcel_analyses': []}

    def _format_ai_analysis(self, ai_data: Dict) -> Dict[str, Any]:
        """Format AI analysis data for integration with parcel data"""

        return {
            'ai_suitability_score': ai_data.get('suitability_score', 50),
            'ai_suitability_category': ai_data.get('suitability_category', 'FAIR'),
            'ai_development_complexity': ai_data.get('development_complexity', 'MEDIUM'),
            'ai_investment_priority': ai_data.get('investment_priority', 'MEDIUM'),
            'ai_development_timeline': ai_data.get('development_timeline', 'STANDARD'),
            'ai_strengths': ai_data.get('strengths', []),
            'ai_challenges': ai_data.get('challenges', []),
            'ai_next_steps': ai_data.get('next_steps', []),
            'ai_technical_notes': ai_data.get('technical_notes', ''),
            'ai_economic_notes': ai_data.get('economic_notes', ''),
            'ai_risk_assessment': ai_data.get('risk_assessment', ''),
            'ai_analysis_timestamp': datetime.now().isoformat(),
            'ai_analysis_version': 'enhanced_v2.0'
        }

    def _fallback_batch_analysis(self, parcels_data: List[Dict], project_type: str) -> List[Dict]:
        """Fallback analysis when AI is unavailable"""
        logger.info(f"Using fallback analysis for {len(parcels_data)} parcels")

        results = []
        for parcel in parcels_data:
            enhanced_parcel = parcel.copy()
            enhanced_parcel.update(self._fallback_single_analysis(parcel, project_type))
            results.append(enhanced_parcel)

        return results

    def _fallback_single_analysis(self, parcel: Dict, project_type: str) -> Dict[str, Any]:
        """Deterministic fallback analysis for a single parcel"""

        score = 50  # Base score

        # Size scoring
        acreage = float(parcel.get('acreage_calc', parcel.get('acreage', 0)))
        adjacent_acres = float(parcel.get('acreage_adjacent_with_sameowner', 0))
        total_acres = acreage + adjacent_acres

        if total_acres >= 100:
            score += 20
        elif total_acres >= 50:
            score += 15
        elif total_acres >= 20:
            score += 10
        elif total_acres >= 10:
            score += 5
        else:
            score -= 10

        # Land use scoring
        land_use = str(parcel.get('land_use_class', '')).lower()
        if any(use in land_use for use in ['agricultural', 'vacant', 'forestry']):
            score += 15
        elif 'commercial' in land_use:
            score += 5
        elif 'residential' in land_use:
            score -= 15

        # Slope scoring
        avg_slope = parcel.get('avg_slope_degrees')
        if avg_slope is not None:
            slope_val = float(avg_slope)
            if project_type.lower() == 'solar':
                if slope_val <= 5:
                    score += 15
                elif slope_val <= 15:
                    score += 5
                elif slope_val > 25:
                    score -= 20
            elif project_type.lower() == 'wind':
                if slope_val <= 20:
                    score += 10
                elif slope_val > 35:
                    score -= 15

        # Transmission scoring
        tx_distance = parcel.get('tx_nearest_distance')
        if tx_distance is not None:
            dist_val = float(tx_distance)
            if dist_val <= 0.5:
                score += 20
            elif dist_val <= 1.0:
                score += 10
            elif dist_val <= 2.0:
                score += 5
            elif dist_val > 5.0:
                score -= 15

        # Voltage bonus
        tx_voltage = parcel.get('tx_primary_voltage', parcel.get('tx_max_voltage'))
        if tx_voltage and float(tx_voltage) >= 115:
            score += 10

        # Flood zone penalty
        flood_zone = str(parcel.get('fld_zone', '')).upper()
        if flood_zone.startswith('A'):
            score -= 15
        elif flood_zone in ['X', 'ZONE X']:
            score += 5

        # Buildings penalty
        buildings = parcel.get('buildings', 0)
        if buildings and float(buildings) > 0:
            score -= 10

        # Finalize score
        final_score = max(5, min(95, score))

        # Determine category
        if final_score >= 85:
            category = 'EXCELLENT'
            complexity = 'LOW'
            priority = 'HIGH'
            timeline = 'FAST'
        elif final_score >= 70:
            category = 'GOOD'
            complexity = 'MEDIUM'
            priority = 'HIGH'
            timeline = 'STANDARD'
        elif final_score >= 55:
            category = 'FAIR'
            complexity = 'MEDIUM'
            priority = 'MEDIUM'
            timeline = 'STANDARD'
        else:
            category = 'POOR'
            complexity = 'HIGH'
            priority = 'LOW'
            timeline = 'CHALLENGING'

        # Generate basic insights
        strengths = []
        challenges = []

        if total_acres >= 50:
            strengths.append('Adequate land area')
        if tx_distance and float(tx_distance) <= 1.0:
            strengths.append('Close to transmission')
        if avg_slope and float(avg_slope) <= 10:
            strengths.append('Suitable terrain')

        if buildings and float(buildings) > 0:
            challenges.append('Existing structures')
        if flood_zone.startswith('A'):
            challenges.append('Flood risk')
        if total_acres < 20:
            challenges.append('Limited size')

        # Fill in defaults if empty
        if not strengths:
            strengths = ['Standard development factors']
        if not challenges:
            challenges = ['Requires detailed site assessment']

        return {
            'ai_suitability_score': final_score,
            'ai_suitability_category': category,
            'ai_development_complexity': complexity,
            'ai_investment_priority': priority,
            'ai_development_timeline': timeline,
            'ai_strengths': strengths[:3],
            'ai_challenges': challenges[:3],
            'ai_next_steps': ['Site verification', 'Grid study', 'Permitting review'],
            'ai_technical_notes': f'Fallback analysis based on {project_type} development criteria',
            'ai_economic_notes': f'Score: {final_score}/100',
            'ai_risk_assessment': 'Requires comprehensive due diligence',
            'ai_analysis_timestamp': datetime.now().isoformat(),
            'ai_analysis_version': 'fallback_v1.0'
        }

    def analyze_single_parcel_detailed(self, parcel_data: Dict,
                                       project_type: str = 'solar',
                                       location: str = 'Unknown') -> Dict[str, Any]:
        """Detailed AI analysis for a single high-priority parcel"""

        if not self.client:
            return self._fallback_single_analysis(parcel_data, project_type)

        # Create detailed single-parcel prompt
        parcel_summary = self._create_detailed_parcel_summary(parcel_data)

        prompt = f"""
As a senior renewable energy development expert, provide comprehensive due diligence analysis for this {project_type} development site in {location}.

PARCEL DETAILS:
{parcel_summary}

COMPREHENSIVE ANALYSIS REQUIRED:

1. **Site Suitability Assessment**
   - Overall development viability (0-100 score)
   - Key technical advantages and constraints
   - Optimal project configuration recommendations

2. **Risk Analysis**
   - Technical risks and mitigation strategies
   - Regulatory/permitting risks
   - Economic/financial risks
   - Timeline risks

3. **Development Strategy**
   - Recommended development approach
   - Critical path milestones
   - Resource requirements
   - Success factors

4. **Comparative Market Position**
   - How this site compares to typical {project_type} projects
   - Competitive advantages
   - Market positioning

5. **Investment Analysis**
   - Development complexity assessment
   - Capital requirements estimate
   - Revenue potential factors
   - ROI considerations

Return detailed JSON analysis with actionable recommendations for this specific {project_type} development opportunity.
        """

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            ai_analysis = self._parse_ai_response(response_text)

            # Format for single parcel
            if ai_analysis and 'suitability_score' in ai_analysis:
                return self._format_ai_analysis(ai_analysis)
            else:
                return self._fallback_single_analysis(parcel_data, project_type)

        except Exception as e:
            logger.error(f"Detailed single parcel analysis failed: {e}")
            return self._fallback_single_analysis(parcel_data, project_type)

    def _create_detailed_parcel_summary(self, parcel: Dict) -> str:
        """Create detailed summary for single-parcel analysis"""

        summary_parts = []

        # Basic information
        parcel_id = parcel.get('parcel_id', 'Unknown')
        summary_parts.append(f"Parcel ID: {parcel_id}")

        # Size and expansion
        acreage = parcel.get('acreage_calc', parcel.get('acreage', 0))
        adjacent = parcel.get('acreage_adjacent_with_sameowner', 0)
        summary_parts.append(f"Size: {acreage} acres (+ {adjacent} adjacent same-owner)")

        # Land characteristics
        land_cover = parcel.get('land_cover', 'Unknown')
        land_use_class = parcel.get('land_use_class', 'Unknown')
        land_use_code = parcel.get('land_use_code', 'Unknown')
        summary_parts.append(f"Land Use: {land_use_class} ({land_use_code}), Cover: {land_cover}")

        # Economic factors
        land_value = parcel.get('mkt_val_land', parcel.get('mkt_land_val', 0))
        summary_parts.append(f"Land Value: ${land_value:,}")

        # Infrastructure
        buildings = parcel.get('buildings', 0)
        summary_parts.append(f"Buildings: {buildings}")

        # Environmental
        flood_zone = parcel.get('fld_zone', 'Unknown')
        zone_subtype = parcel.get('zone_subty', 'Unknown')
        summary_parts.append(f"Flood Zone: {flood_zone}, Zoning: {zone_subtype}")

        # Owner
        owner = parcel.get('owner', 'Unknown')
        summary_parts.append(f"Owner: {owner}")

        # Transmission analysis
        tx_distance = parcel.get('tx_nearest_distance', 'Unknown')
        tx_voltage = parcel.get('tx_primary_voltage', 'Unknown')
        tx_lines = parcel.get('tx_lines_count', 0)
        summary_parts.append(f"Grid Access: {tx_distance} miles to {tx_voltage}kV line, {tx_lines} lines nearby")

        # Slope analysis
        avg_slope = parcel.get('avg_slope_degrees', 'Unknown')
        slope_category = parcel.get('slope_category', 'Unknown')
        buildability = parcel.get('buildability_5_35', 'Unknown')
        summary_parts.append(f"Terrain: {avg_slope}° average slope, {slope_category}, {buildability} buildability")

        return '\n'.join(summary_parts)