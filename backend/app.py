"""
Smart Traffic Signal Optimization Using Reinforcement Learning
Flask REST API & Web Application Backend
"""

import os
import sys
from flask import Flask, jsonify, request, send_from_directory

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from backend.q_learning import QLearningAgent
from backend.traffic_simulation import TrafficSimulation
from backend.train_agent import train_q_learning_agent

# Initialize Flask app serving frontend static assets
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Initialize Q-Learning agent and simulation
q_table_file = os.path.join(os.path.dirname(__file__), "q_table.json")
agent = QLearningAgent(q_table_path=q_table_file)
simulation = TrafficSimulation(q_agent=agent)

# Start simulation by default on boot
simulation.start()


@app.route("/")
def serve_index():
    """Serves the main dashboard single-page application."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serves static CSS, JS, and asset files."""
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns current real-time simulation and AI controller state."""
    state = simulation.get_state_payload()
    return jsonify(state)


@app.route("/api/start", methods=["POST"])
def start_simulation():
    """Starts / Resumes simulation loop."""
    simulation.start()
    return jsonify({"success": True, "status": "running"})


@app.route("/api/pause", methods=["POST"])
def pause_simulation():
    """Pauses simulation loop."""
    simulation.pause()
    return jsonify({"success": True, "status": "paused"})


@app.route("/api/reset", methods=["POST"])
def reset_simulation():
    """Resets simulation time, vehicles, queues, and metrics."""
    simulation.reset()
    return jsonify({"success": True, "status": "reset"})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    """Switches control policy: 'ai' (Q-Learning) or 'fixed' (Fixed-Time 30s)."""
    data = request.get_json() or {}
    mode = data.get("mode", "ai")
    if mode in ["ai", "fixed"]:
        simulation.set_mode(mode)
        return jsonify({"success": True, "mode": mode})
    return jsonify({"success": False, "error": "Invalid mode. Use 'ai' or 'fixed'."}), 400


@app.route("/api/density", methods=["POST"])
def set_density():
    """Adjusts traffic generation arrival rates."""
    data = request.get_json() or {}
    density = data.get("density", "medium")
    if density in ["low", "medium", "high", "rush_hour"]:
        simulation.set_density(density)
        return jsonify({"success": True, "density": density})
    return jsonify({"success": False, "error": "Invalid density setting."}), 400


@app.route("/api/train", methods=["POST"])
def train_agent():
    """Triggers Q-Learning agent training episodes and reloads updated Q-table."""
    data = request.get_json() or {}
    episodes = int(data.get("episodes", 25))
    
    # Pause active simulation briefly during training
    was_running = simulation.is_running
    simulation.pause()

    results = train_q_learning_agent(episodes=episodes, q_table_path=q_table_file)
    
    # Reload trained weights into simulation agent
    agent.load(q_table_file)

    if was_running:
        simulation.start()

    return jsonify(results)


@app.route("/api/qtable", methods=["GET"])
def get_qtable():
    """Returns serialized Q-table data for explainability."""
    return jsonify({
        "q_table": agent.q_table,
        "total_states": len(agent.q_table),
        "epsilon": round(agent.epsilon, 3),
        "alpha": agent.alpha,
        "gamma": agent.gamma
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print(f"🚦 SMART TRAFFIC AI FLASK BACKEND RUNNING")
    print(f"🌐 Access Dashboard at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
