# services/suitability_service.py
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SuitabilityAnalysisService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_parcels(self, parcels: List[Dict], project_type: str, location: str) -> Dict[str, Any]:
        """Analyze parcels for renewable energy suitability"""
        try:
            self.logger.info(f"Starting suitability analysis for {len(parcels)} parcels in {location}")

            # Process each parcel through suitability analysis
            enhanced_parcels = []
            excellent_count = 0
            good_count = 0
            fair_count = 0
            poor_count = 0

            for i, parcel in enumerate(parcels):
                try:
                    # Get ML score if available
                    ml_analysis = parcel.get('ml_analysis', {})
                    ml_score = ml_analysis.get('predicted_score', 50.0)

                    # Calculate traditional suitability factors
                    traditional_score = self._calculate_traditional_score(parcel, project_type)

                    # Combine ML and traditional scores (60% ML, 40% traditional)
                    final_score = (ml_score * 0.6) + (traditional_score * 0.4)

                    # Add suitability analysis to parcel
                    parcel['suitability_analysis'] = {
                        'final_score': round(final_score, 1),
                        'ml_score': round(ml_score, 1),
                        'traditional_score': round(traditional_score, 1),
                        'score_category': self._get_score_category(final_score),
                        'suitability_factors': self._get_suitability_factors(parcel, project_type),
                        'analysis_timestamp': datetime.now().isoformat()
                    }

                    # Count categories
                    if final_score >= 85:
                        excellent_count += 1
                    elif final_score >= 70:
                        good_count += 1
                    elif final_score >= 55:
                        fair_count += 1
                    else:
                        poor_count += 1

                    enhanced_parcels.append(parcel)

                except Exception as parcel_error:
                    self.logger.error(f"Error analyzing parcel {i}: {parcel_error}")
                    # Add default analysis on error
                    parcel['suitability_analysis'] = {
                        'final_score': 50.0,
                        'ml_score': 50.0,
                        'traditional_score': 50.0,
                        'score_category': 'fair',
                        'suitability_factors': {'error': 'Analysis failed'},
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    fair_count += 1
                    enhanced_parcels.append(parcel)

            # Sort by final score
            enhanced_parcels.sort(key=lambda x: x['suitability_analysis']['final_score'], reverse=True)

            # Create summary
            analysis_summary = {
                'total_parcels': len(enhanced_parcels),
                'excellent': excellent_count,
                'good': good_count,
                'fair': fair_count,
                'poor': poor_count,
                'location': location,
                'project_type': project_type,
                'analysis_date': datetime.now().isoformat(),
                'average_score': sum(p['suitability_analysis']['final_score'] for p in enhanced_parcels) / len(
                    enhanced_parcels) if enhanced_parcels else 0
            }

            self.logger.info(
                f"Suitability analysis completed: {excellent_count} excellent, {good_count} good, {fair_count} fair, {poor_count} poor")

            return {
                'enhanced_parcels': enhanced_parcels,
                'summary': analysis_summary,
                'analysis_metadata': {
                    'service': 'SuitabilityAnalysisService',
                    'version': '1.0',
                    'scoring_method': 'hybrid_ml_traditional'
                }
            }

        except Exception as e:
            self.logger.error(f"Suitability analysis failed: {e}")
            return {
                'enhanced_parcels': parcels,
                'summary': {
                    'total_parcels': len(parcels),
                    'excellent': 0,
                    'good': 0,
                    'fair': len(parcels),
                    'poor': 0,
                    'error': str(e)
                },
                'analysis_metadata': {
                    'service': 'SuitabilityAnalysisService',
                    'version': '1.0',
                    'error': str(e)
                }
            }

    def _calculate_traditional_score(self, parcel: Dict, project_type: str) -> float:
        """Calculate traditional suitability score based on parcel characteristics"""
        score = 50.0  # Base score

        try:
            # Acreage factor
            acreage = float(parcel.get('acreage_calc', parcel.get('acreage', 0)))
            if acreage >= 50:
                score += 20
            elif acreage >= 20:
                score += 15
            elif acreage >= 10:
                score += 10
            elif acreage >= 5:
                score += 5
            else:
                score -= 10

            # Land use factor
            land_use = str(parcel.get('land_use_class', '')).lower()
            if any(use in land_use for use in ['agricultural', 'vacant', 'forestry']):
                score += 15
            elif 'commercial' in land_use:
                score += 5
            elif 'residential' in land_use:
                score -= 10

            # Land value factor (lower is better for development)
            land_value = float(parcel.get('mkt_val_land', 0))
            if land_value == 0:
                score += 5  # Might be undeveloped
            elif land_value < 50000:
                score += 10
            elif land_value < 100000:
                score += 5
            elif land_value > 500000:
                score -= 15

            # Slope factor (if available)
            slope = parcel.get('slope_percent')
            if slope is not None:
                slope_val = float(slope)
                if project_type.lower() == 'solar':
                    if slope_val <= 5:
                        score += 15
                    elif slope_val <= 10:
                        score += 5
                    elif slope_val > 20:
                        score -= 20
                elif project_type.lower() == 'wind':
                    if slope_val <= 15:
                        score += 10
                    elif slope_val > 30:
                        score -= 15

            # Flood zone factor
            flood_zone = str(parcel.get('fld_zone', '')).upper()
            if flood_zone in ['X', 'ZONE X']:
                score += 10  # Low flood risk
            elif flood_zone.startswith('A'):
                score -= 20  # High flood risk

            return max(10, min(95, score))

        except Exception as e:
            self.logger.error(f"Error calculating traditional score: {e}")
            return 50.0

    def _get_score_category(self, score: float) -> str:
        """Get score category"""
        if score >= 85:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 55:
            return 'fair'
        else:
            return 'poor'

    def _get_suitability_factors(self, parcel: Dict, project_type: str) -> Dict[str, Any]:
        """Get detailed suitability factors"""
        factors = {}

        try:
            # Acreage assessment
            acreage = float(parcel.get('acreage_calc', parcel.get('acreage', 0)))
            if acreage >= 50:
                factors['acreage'] = 'Excellent - Large site'
            elif acreage >= 20:
                factors['acreage'] = 'Good - Adequate size'
            elif acreage >= 10:
                factors['acreage'] = 'Fair - Medium size'
            else:
                factors['acreage'] = 'Poor - Small site'

            # Land use assessment
            land_use = str(parcel.get('land_use_class', '')).lower()
            if any(use in land_use for use in ['agricultural', 'vacant', 'forestry']):
                factors['land_use'] = 'Excellent - Rural/undeveloped'
            elif 'commercial' in land_use:
                factors['land_use'] = 'Good - Commercial zoning'
            else:
                factors['land_use'] = 'Fair - Mixed use'

            # Add project-specific factors
            if project_type.lower() == 'solar':
                slope = parcel.get('slope_percent')
                if slope is not None:
                    slope_val = float(slope)
                    if slope_val <= 5:
                        factors['terrain'] = 'Excellent - Flat terrain'
                    elif slope_val <= 15:
                        factors['terrain'] = 'Good - Gentle slope'
                    else:
                        factors['terrain'] = 'Poor - Steep slope'

            # Flood risk
            flood_zone = str(parcel.get('fld_zone', '')).upper()
            if flood_zone in ['X', 'ZONE X']:
                factors['flood_risk'] = 'Low flood risk'
            elif flood_zone.startswith('A'):
                factors['flood_risk'] = 'High flood risk'
            else:
                factors['flood_risk'] = 'Unknown flood risk'

            return factors

        except Exception as e:
            self.logger.error(f"Error getting suitability factors: {e}")
            return {'error': 'Factor analysis failed'}