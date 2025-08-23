# services/ml_service.py
import logging
from datetime import datetime
import uuid
import random

logger = logging.getLogger(__name__)


class MLParcelService:
    def __init__(self):
        self.mock_mode = True  # Set to False when real model is ready

    def score_parcels(self, parcels, project_type):
        """Score parcels with ML model"""
        logger.info(f"🧠 ML scoring {len(parcels)} parcels for {project_type}")

        scored_parcels = []

        for i, parcel in enumerate(parcels):
            try:
                # Extract features for ML scoring
                features = self._extract_features(parcel)

                if self.mock_mode:
                    # Generate realistic mock scores based on parcel characteristics
                    ml_score = self._generate_realistic_score(features, project_type)
                    confidence = self._calculate_confidence(features)
                    rank = i + 1
                else:
                    # Use real ML model here
                    ml_score, confidence, rank = self._predict_with_model(features, project_type)

                # Add ML analysis to parcel
                parcel['ml_analysis'] = {
                    'predicted_score': round(float(ml_score), 1),
                    'confidence_score': round(float(confidence), 2),
                    'ml_rank': int(rank),
                    'model_version': 'v1.0_mock' if self.mock_mode else 'v1.0',
                    'features_used': list(features.keys()),
                    'prediction_timestamp': datetime.now().isoformat()
                }

                scored_parcels.append(parcel)

            except Exception as e:
                logger.error(f"Error scoring parcel {i}: {e}")
                # Add default ML analysis on error
                parcel['ml_analysis'] = {
                    'predicted_score': 50.0,
                    'confidence_score': 0.1,
                    'ml_rank': 999,
                    'model_version': 'error_fallback',
                    'features_used': [],
                    'prediction_timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                scored_parcels.append(parcel)

        # Sort by ML score
        scored_parcels.sort(key=lambda x: x['ml_analysis']['predicted_score'], reverse=True)

        # Update ranks
        for i, parcel in enumerate(scored_parcels):
            parcel['ml_analysis']['ml_rank'] = i + 1

        logger.info(f"✅ ML scoring completed. Top score: {scored_parcels[0]['ml_analysis']['predicted_score']}")
        return scored_parcels

    def _extract_features(self, parcel):
        """Extract features for ML model"""
        try:
            return {
                'acreage': float(parcel.get('acreage_calc', parcel.get('acreage', 0))),
                'land_value': float(parcel.get('mkt_val_land', 0)),
                'elevation': float(parcel.get('elevation', 1000)),
                'latitude': float(parcel.get('latitude', 40.0)),
                'longitude': float(parcel.get('longitude', -80.0)),
                'land_use_class': parcel.get('land_use_class', 'Residential'),
                'owner_type': self._classify_owner_type(parcel.get('owner', '')),
                'flood_zone': parcel.get('fld_zone', 'Unknown'),
                'adjacent_acres': float(parcel.get('acreage_adjacent_with_sameowner', 0))
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {'acreage': 0, 'land_value': 0, 'elevation': 1000}

    def _generate_realistic_score(self, features, project_type):
        """Generate realistic ML scores based on characteristics"""
        base_score = 50

        # Acreage scoring (larger is better, up to a point)
        acreage = features.get('acreage', 0)
        if acreage >= 100:
            base_score += 25
        elif acreage >= 50:
            base_score += 15
        elif acreage >= 20:
            base_score += 10
        elif acreage >= 10:
            base_score += 5
        else:
            base_score -= 10

        # Land value scoring (lower is better for development cost)
        land_value = features.get('land_value', 0)
        if land_value == 0:
            base_score += 10  # No assessed value might mean undeveloped
        elif land_value < 50000:
            base_score += 15
        elif land_value < 100000:
            base_score += 5
        else:
            base_score -= 5

        # Owner type scoring
        owner_type = features.get('owner_type', 'individual')
        if owner_type in ['government', 'utility']:
            base_score += 20  # Often easier to work with
        elif owner_type == 'corporate':
            base_score += 10

        # Adjacent acres bonus
        adjacent = features.get('adjacent_acres', 0)
        if adjacent > acreage * 1.5:
            base_score += 15  # Expansion potential

        # Random variation for realism
        import random
        random.seed(int(features.get('acreage', 1) * 100))  # Deterministic randomness
        variation = random.uniform(-10, 10)

        final_score = max(10, min(95, base_score + variation))
        return final_score

    def _classify_owner_type(self, owner_name):
        """Classify owner type from name"""
        if not owner_name:
            return 'unknown'

        owner_upper = str(owner_name).upper()

        if any(word in owner_upper for word in ['CITY', 'COUNTY', 'STATE', 'GOVERNMENT', 'TOWNSHIP']):
            return 'government'
        elif any(word in owner_upper for word in ['LLC', 'CORP', 'INC', 'COMPANY', 'TRUST']):
            return 'corporate'
        elif any(word in owner_upper for word in ['UTILITY', 'ELECTRIC', 'POWER', 'ENERGY']):
            return 'utility'
        else:
            return 'individual'

    def _calculate_confidence(self, features):
        """Calculate confidence score based on available data"""
        available_features = sum(1 for v in features.values() if v and v != 0)
        total_features = len(features)
        return min(0.95, available_features / total_features)