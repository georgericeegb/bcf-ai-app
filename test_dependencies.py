# test_dependencies.py
# Run this script to test if all dependencies are working

import sys


def test_dependencies():
    """Test all required dependencies for the suitability analysis"""

    print("🔍 Testing Dependencies for Parcel Suitability Analysis")
    print("=" * 60)

    # Test Google Cloud BigQuery
    try:
        from google.cloud import bigquery
        print("✅ Google Cloud BigQuery: Available")
        print(f"   Version: {bigquery.__version__}")
    except ImportError as e:
        print(f"❌ Google Cloud BigQuery: Missing ({e})")
        print("   Fix: pip install google-cloud-bigquery")
        return False

    # Test Google Cloud Storage
    try:
        from google.cloud import storage
        print("✅ Google Cloud Storage: Available")
        print(f"   Version: {storage.__version__}")
    except ImportError as e:
        print(f"❌ Google Cloud Storage: Missing ({e})")
        print("   Fix: pip install google-cloud-storage")
        return False

    # Test GeoPandas
    try:
        import geopandas as gpd
        print("✅ GeoPandas: Available")
        print(f"   Version: {gpd.__version__}")
    except ImportError as e:
        print(f"❌ GeoPandas: Missing ({e})")
        print("   Fix: pip install geopandas")
        return False

    # Test Shapely
    try:
        import shapely
        print("✅ Shapely: Available")
        print(f"   Version: {shapely.__version__}")
    except ImportError as e:
        print(f"❌ Shapely: Missing ({e})")
        print("   Fix: pip install shapely")
        return False

    # Test Pandas
    try:
        import pandas as pd
        print("✅ Pandas: Available")
        print(f"   Version: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: Missing ({e})")
        print("   Fix: pip install pandas")
        return False

    print("\n🎯 Testing Analysis Modules")
    print("-" * 40)

    # Test slope analysis module
    try:
        from bigquery_slope_analysis import run_headless_fixed
        print("✅ Slope Analysis Module: Available")
    except ImportError as e:
        print(f"⚠️ Slope Analysis Module: Not Available ({e})")
        print("   This is expected if dependencies aren't installed yet")

    # Test transmission analysis module
    try:
        from transmission_analysis_bigquery import run_headless
        print("✅ Transmission Analysis Module: Available")
    except ImportError as e:
        print(f"⚠️ Transmission Analysis Module: Not Available ({e})")
        print("   This is expected if dependencies aren't installed yet")

    print("\n🧪 Testing Basic Functionality")
    print("-" * 40)

    # Test basic geospatial operations
    try:
        from shapely.geometry import Point
        import geopandas as gpd

        # Create test point
        point = Point(-78.4, 40.5)
        gdf = gpd.GeoDataFrame([{'id': 1, 'geometry': point}], crs='EPSG:4326')
        print("✅ Basic geospatial operations: Working")

    except Exception as e:
        print(f"❌ Basic geospatial operations: Failed ({e})")
        return False

    print("\n🚀 All Core Dependencies Ready!")
    print("You can now run your Flask application with suitability analysis.")
    return True


if __name__ == "__main__":
    success = test_dependencies()
    if not success:
        print("\n❌ Some dependencies are missing. Please install them and try again.")
        sys.exit(1)
    else:
        print("\n✅ All dependencies are ready!")
        sys.exit(0)