import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Waitress for production WSGI serving
from waitress import serve

# Import the Flask application and necessary initialization functions
from app import app, _load_cached_metrics

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  AI Memory Forecaster — PRODUCTION WSGI SERVER")
    print("  Powered by Waitress")
    print("  Open: http://localhost:5000")
    print("=" * 55 + "\n")
    
    # Load cached metrics for immediate dashboard display
    _load_cached_metrics()
    
    # Run the Flask app with the Waitress WSGI server on port 5000
    # Utilizing 6 threads to handle concurrent dashboard operations
    serve(app, host='0.0.0.0', port=5000, threads=6)
