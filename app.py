"""
Root entry point to launch the Smart Traffic AI Application.
Runs the Flask server on port 5000.
"""

from backend.app import app

if __name__ == "__main__":
    print("=" * 65)
    print("🚦 STARTING SMART TRAFFIC AI (Q-LEARNING FLASK APP)")
    print("🌐 Open Browser at: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
