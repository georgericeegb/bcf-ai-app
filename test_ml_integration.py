#!/usr/bin/env python3
"""
Test script for ML integration debugging
Tests the entire parcel analysis pipeline step by step
"""

import os
import sys
import json
import logging
from datetime import datetime
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_traditional_analysis():
    """Test the existing slope + transmission analysis"""
    logger.info("🔍 STEP 1: Testing Traditional Analysis (Slope + Transmission)")

    try:
        # Import your existing modules
        import bigquery_slope_analysis as slope_module
        import transmission_analysis_bigquery as transmission_module

        # Test file path (adjust to your actual test file)
        test_file = "gs://bcfparcelsearchrepository/PA/Blair/Parcel_Files/BlairCoPA_052620250506.gpkg"

        logger.info(f"Testing with file: {test_file}")

        # Test slope analysis
        logger.info("Testing slope analysis...")
        slope_result = slope_module.run_headless_fixed(
            input_file_path=test_file,
            max_slope_degrees=15.0,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        logger.info(f"Slope analysis result: {slope_result['status']}")
        if slope_result['status'] == 'success':
            logger.info(f"  - Parcels processed: {slope_result.get('parcels_processed', 0)}")
            logger.info(f"  - Suitable parcels: {slope_result.get('parcels_suitable_slope', 0)}")

        # Test transmission analysis
        logger.info("Testing transmission analysis...")
        transmission_result = transmission_module.run_headless(
            input_file_path=test_file,
            buffer_distance_miles=1.0,
            output_bucket='bcfparcelsearchrepository',
            project_id='bcfparcelsearchrepository'
        )

        logger.info(f"Transmission analysis result: {transmission_result['status']}")
        if transmission_result['status'] == 'success':
            logger.info(f"  - Parcels processed: {transmission_result.get('parcels_processed', 0)}")
            logger.info(f"  - Near transmission: {transmission_result.get('parcels_near_transmission', 0)}")

        return {
            'status': 'success',
            'slope_result': slope_result,
            'transmission_result': transmission_result
        }

    except Exception as e:
        logger.error(f"Traditional analysis test failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def test_ml_service():
    """Test the ML service separately"""
    logger.info("🧠 STEP 2: Testing ML Service")

    try:
        from services.ml_service import MLParcelService

        # Create sample parcel data
        sample_parcels = [
            {
                'parcel_id': 'TEST_001',
                'acreage': 50.0,
                'owner': 'Test Owner 1',
                'avg_slope_degrees': 5.0,
                'distance_to_transmission_km': 0.5,
                'nearest_line_voltage_kv': 138.0,
                'transmission_lines_10km': 3,
                'elevation': 800.0,
                'mkt_val_land': 100000
            },
            {
                'parcel_id': 'TEST_002',
                'acreage': 25.0,
                'owner': 'Test Owner 2',
                'avg_slope_degrees': 12.0,
                'distance_to_transmission_km': 1.2,
                'nearest_line_voltage_kv': 230.0,
                'transmission_lines_10km': 1,
                'elevation': 1000.0,
                'mkt_val_land': 50000
            }
        ]

        ml_service = MLParcelService()
        logger.info(f"ML Service mock mode: {ml_service.mock_mode}")

        scored_parcels = ml_service.score_parcels(sample_parcels, 'solar')

        logger.info(f"ML scoring completed for {len(scored_parcels)} parcels")
        for parcel in scored_parcels:
            ml_data = parcel.get('ml_analysis', {})
            logger.info(f"  - {parcel['parcel_id']}: ML Score = {ml_data.get('predicted_score', 'N/A')}")

        return {
            'status': 'success',
            'scored_parcels': scored_parcels
        }

    except Exception as e:
        logger.error(f"ML service test failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def test_flask_endpoint():
    """Test the Flask ML endpoint directly"""
    logger.info("🌐 STEP 3: Testing Flask ML Endpoint")

    try:
        import requests

        # Sample data matching your parcel format
        test_data = {
            "parcels": [
                {
                    "parcel_id": "TEST_FLASK_001",
                    "acreage": 45.0,
                    "owner": "Flask Test Owner",
                    "county": "Test County",
                    "state_abbr": "PA"
                }
            ],
            "project_type": "solar",
            "analysis_type": "ml_enhanced"
        }

        # Test the endpoint
        url = "http://127.0.0.1:8080/api/parcel/analyze_suitability_with_ml"

        logger.info(f"POST to {url}")
        response = requests.post(url, json=test_data, timeout=30)

        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Flask endpoint success: {result.get('status', 'unknown')}")

            if result.get('status') == 'success':
                data = result.get('data', {})
                logger.info(f"  - Total parcels: {data.get('total_parcels', 0)}")
                logger.info(f"  - Suitable parcels: {data.get('suitable_parcels', 0)}")
                parcels = data.get('parcels', [])
                if parcels:
                    sample_parcel = parcels[0]
                    analysis = sample_parcel.get('suitability_analysis', {})
                    logger.info(f"  - Sample ML score: {analysis.get('ml_score', 'N/A')}")
                    logger.info(f"  - Sample overall score: {analysis.get('overall_score', 'N/A')}")

            return {
                'status': 'success',
                'response': result
            }
        else:
            error_text = response.text
            logger.error(f"Flask endpoint failed: {response.status_code} - {error_text}")
            return {
                'status': 'error',
                'status_code': response.status_code,
                'error': error_text
            }

    except Exception as e:
        logger.error(f"Flask endpoint test failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def test_data_compatibility():
    """Test data format compatibility between traditional and ML analysis"""
    logger.info("🔄 STEP 4: Testing Data Format Compatibility")

    try:
        # Simulate traditional analysis output
        traditional_parcels = [
            {
                'parcel_id': 'COMPAT_001',
                'acreage': 30.0,
                'owner': 'Compatibility Test',
                'suitability_analysis': {
                    'is_suitable': True,
                    'overall_score': 75.0,
                    'slope_degrees': 8.0,
                    'transmission_distance': 0.8,
                    'transmission_voltage': 138.0,
                    'analysis_notes': 'Traditional analysis result'
                }
            }
        ]

        # Test ML enhancement
        from services.ml_service import MLParcelService
        ml_service = MLParcelService()

        # Score with ML
        ml_scored = ml_service.score_parcels(traditional_parcels, 'solar')

        # Test combination logic (from app.py)
        final_parcels = []
        for parcel in ml_scored:
            traditional_score = parcel.get('suitability_analysis', {}).get('overall_score', 50)
            ml_score = parcel.get('ml_analysis', {}).get('predicted_score', 50)

            # Weighted combination: 60% ML, 40% traditional
            combined_score = (ml_score * 0.6) + (traditional_score * 0.4)

            parcel['final_analysis'] = {
                'combined_score': round(combined_score, 2),
                'ml_score': ml_score,
                'traditional_score': traditional_score,
                'ml_rank': parcel.get('ml_analysis', {}).get('ml_rank', 999),
                'recommended': combined_score > 65,
                'confidence': 'high' if abs(ml_score - traditional_score) < 20 else 'medium'
            }

            final_parcels.append(parcel)

        logger.info("Data compatibility test completed")
        for parcel in final_parcels:
            final = parcel['final_analysis']
            logger.info(
                f"  - {parcel['parcel_id']}: Combined={final['combined_score']}, ML={final['ml_score']}, Traditional={final['traditional_score']}")

        return {
            'status': 'success',
            'final_parcels': final_parcels
        }

    except Exception as e:
        logger.error(f"Data compatibility test failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def test_frontend_simulation():
    """Simulate the frontend request exactly as it would be sent"""
    logger.info("🖥️  STEP 5: Frontend Simulation Test")

    try:
        # Simulate parcel search results (what comes from your parcel search)
        mock_parcel_results = {
            'parcel_data': [
                {
                    'parcel_id': 'SIM_001',
                    'acreage_calc': 40.0,
                    'acreage': 40.0,  # Backup field
                    'owner': 'Frontend Simulation Owner',
                    'county_name': 'Test County',
                    'state_abbr': 'PA',
                    'mkt_val_land': 80000
                },
                {
                    'parcel_id': 'SIM_002',
                    'acreage_calc': 15.0,
                    'acreage': 15.0,
                    'owner': 'Another Test Owner',
                    'county_name': 'Test County',
                    'state_abbr': 'PA',
                    'mkt_val_land': 40000
                }
            ],
            'record_count': 2
        }

        mock_analysis = {
            'project_type': 'solar',
            'location': 'Test County, PA',
            'analysis_level': 'county'
        }

        # Import your app functions directly
        from app import calculate_parcel_suitability_scores, create_fallback_analysis_results
        from services.ml_service import MLParcelService

        # Step 1: Traditional analysis
        logger.info("Running traditional suitability analysis...")
        analysis_results = create_fallback_analysis_results()
        analyzed_parcels = calculate_parcel_suitability_scores(
            mock_parcel_results['parcel_data'],
            analysis_results,
            mock_analysis['project_type']
        )

        logger.info(f"Traditional analysis completed for {len(analyzed_parcels)} parcels")

        # Step 2: ML analysis
        logger.info("Running ML analysis...")
        ml_service = MLParcelService()
        ml_scored_parcels = ml_service.score_parcels(analyzed_parcels, mock_analysis['project_type'])

        # Step 3: Combined scoring
        logger.info("Combining scores...")
        final_parcels = []
        for parcel in ml_scored_parcels:
            traditional_score = parcel.get('suitability_analysis', {}).get('overall_score', 50)
            ml_score = parcel.get('ml_analysis', {}).get('predicted_score', 50)

            combined_score = (ml_score * 0.6) + (traditional_score * 0.4)

            # Create final analysis structure
            simple_parcel = {
                'parcel_id': str(parcel.get('parcel_id', '')),
                'owner': str(parcel.get('owner', 'Unknown')),
                'acreage': float(parcel.get('acreage_calc', parcel.get('acreage', 0))),
                'county': str(parcel.get('county_name', 'Unknown')),
                'state_abbr': str(parcel.get('state_abbr', 'Unknown')),
                'is_suitable': bool(combined_score > 65),
                'suitability_analysis': {
                    'is_suitable': bool(combined_score > 65),
                    'overall_score': float(combined_score),
                    'ml_score': float(ml_score),
                    'traditional_score': float(traditional_score),
                    'ml_rank': int(parcel.get('ml_analysis', {}).get('ml_rank', 999)),
                    'confidence_level': 'high' if abs(ml_score - traditional_score) < 20 else 'medium',
                    'slope_degrees': parcel.get('suitability_analysis', {}).get('slope_degrees', 'Unknown'),
                    'transmission_distance': parcel.get('suitability_analysis', {}).get('transmission_distance',
                                                                                        'Unknown'),
                    'transmission_voltage': parcel.get('suitability_analysis', {}).get('transmission_voltage',
                                                                                       'Unknown'),
                    'analysis_notes': f"ML Score: {ml_score:.1f}, Traditional: {traditional_score:.1f}, Combined: {combined_score:.1f}"
                }
            }
            final_parcels.append(simple_parcel)

        # Sort by combined score
        final_parcels.sort(key=lambda x: x['suitability_analysis']['overall_score'], reverse=True)

        suitable_count = sum(1 for p in final_parcels if p['is_suitable'])

        # Create the exact response format
        response_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_parcels': len(final_parcels),
            'suitable_parcels': suitable_count,
            'analysis_type': 'ml_enhanced',
            'scoring_method': '60% ML + 40% Traditional Analysis',
            'parcels': final_parcels
        }

        logger.info("Frontend simulation completed successfully!")
        logger.info(f"  - Total parcels: {response_data['total_parcels']}")
        logger.info(f"  - Suitable parcels: {response_data['suitable_parcels']}")
        logger.info(f"  - Scoring method: {response_data['scoring_method']}")

        # Save to file for frontend testing
        output_file = 'test_ml_response.json'
        with open(output_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        logger.info(f"  - Response saved to: {output_file}")

        return {
            'status': 'success',
            'response_data': response_data,
            'output_file': output_file
        }

    except Exception as e:
        logger.error(f"Frontend simulation failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    """Run all tests"""
    print("🚀 ML Integration Test Suite")
    print("=" * 50)

    results = {}

    # Test 1: Traditional Analysis
    results['traditional'] = test_traditional_analysis()

    # Test 2: ML Service
    results['ml_service'] = test_ml_service()

    # Test 3: Flask Endpoint (only if app is running)
    try:
        results['flask_endpoint'] = test_flask_endpoint()
    except:
        logger.warning("Skipping Flask endpoint test (app not running)")
        results['flask_endpoint'] = {'status': 'skipped', 'reason': 'App not running'}

    # Test 4: Data Compatibility
    results['compatibility'] = test_data_compatibility()

    # Test 5: Frontend Simulation
    results['frontend_simulation'] = test_frontend_simulation()

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        status = result.get('status', 'unknown')
        status_emoji = "✅" if status == 'success' else "❌" if status == 'error' else "⏭️"
        print(f"{status_emoji} {test_name}: {status.upper()}")

        if status == 'error':
            print(f"    Error: {result.get('error', 'Unknown error')}")

    # Check if frontend simulation created test data
    if results['frontend_simulation']['status'] == 'success':
        output_file = results['frontend_simulation']['output_file']
        print(f"\n🎯 Frontend test data created: {output_file}")
        print("   You can use this to test the frontend display manually")

    print("\n🏁 Test suite completed!")


if __name__ == "__main__":
    main()