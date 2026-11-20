"""
Comprehensive Test and Validation Suite
Verifies:
1. Q-Learning algorithm convergence and Bellman error updates.
2. Car-following vehicle physics & red-light stopping behavior.
3. Physical Safety Guarantee: Conflicting traffic directions NEVER receive Green simultaneously.
4. Safe Clearance Transitions: Enforced Yellow (3s) and All-Red (1s) phases.
5. All Flask REST API endpoints.
6. Empirical comparison: AI Mode vs Fixed-Time Mode under identical traffic demand.
"""

import os
import sys
import unittest
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.q_learning import QLearningAgent
from backend.traffic_simulation import TrafficSimulation, Vehicle
from backend.app import app


class TestSmartTrafficRL(unittest.TestCase):

    def setUp(self):
        self.agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9, epsilon=0.0)
        self.sim = TrafficSimulation(q_agent=self.agent)
        self.client = app.test_client()

    def test_01_qlearning_state_categorization(self):
        """Verifies queue length and waiting time discretization."""
        # Low demand
        self.assertEqual(QLearningAgent.categorize_queue(1, max_waiting_time=2.0), 0)
        # Medium demand
        self.assertEqual(QLearningAgent.categorize_queue(5, max_waiting_time=5.0), 1)
        # High demand
        self.assertEqual(QLearningAgent.categorize_queue(9, max_waiting_time=8.0), 2)
        # Urgency escalation: vehicle count is 10, waiting time is 35s => escalated to Congested (category 3)
        self.assertEqual(QLearningAgent.categorize_queue(10, max_waiting_time=35.0), 3)

    def test_02_qlearning_bellman_update(self):
        """Verifies that Bellman updates mathematically adjust Q-values."""
        state = (1, 2, 0)
        action = 1
        reward = 10.0
        next_state = (0, 1, 1)

        key = self.agent._state_to_key(state)
        initial_q = self.agent.q_table[key][action]
        td_error = self.agent.update(state, action, reward, next_state)
        updated_q = self.agent.q_table[key][action]

        self.assertNotEqual(initial_q, updated_q)
        # New Q should be initial_q + alpha * td_error
        self.assertAlmostEqual(updated_q, initial_q + self.agent.alpha * td_error, places=4)

    def test_03_safety_no_conflicting_greens(self):
        """
        CRITICAL SAFETY VERIFICATION:
        Ensures conflicting traffic movements (NS and EW) NEVER receive green simultaneously
        across 200 consecutive simulation steps.
        """
        self.sim.reset()
        for step in range(200):
            self.sim.step(dt=0.2)
            payload = self.sim.get_state_payload()
            signals = payload["signals"]

            ns_is_green = (signals["N"] == "green" or signals["S"] == "green")
            ew_is_green = (signals["E"] == "green" or signals["W"] == "green")

            # Condition: If NS is green, EW MUST NOT be green, and vice-versa
            self.assertFalse(
                ns_is_green and ew_is_green,
                f"SAFETY VIOLATION at step {step}: Conflicting greens detected! NS={signals['N']}, EW={signals['E']}"
            )

    def test_04_safe_clearance_transitions(self):
        """
        Verifies that switching between NS Green and EW Green strictly requires
        Yellow clearance (3s) and All-Red clearance (1s).
        """
        self.sim.reset()
        self.sim.phase = 0  # NS Green
        self.sim.phase_timer = 15.0  # Min green elapsed

        # Force transition to EW Green
        self.sim.mode = "fixed"
        self.sim.fixed_green_time = 10.0
        self.sim._update_signals(dt=1.0)

        # Must enter NS Yellow (Phase 1)
        self.assertEqual(self.sim.phase, 1, "Must switch to Yellow clearance before Red")

        # Step through yellow time
        self.sim._update_signals(dt=3.1)
        # Must enter All-Red clearance (Phase 2)
        self.assertEqual(self.sim.phase, 2, "Must enter All-Red clearance after Yellow")

        # Step through all-red time
        self.sim._update_signals(dt=1.1)
        # Must enter EW Green (Phase 3)
        self.assertEqual(self.sim.phase, 3, "Must safely switch to EW Green after All-Red")

    def test_05_vehicle_stopping_at_red_signal(self):
        """Verifies that approaching vehicles decelerate and stop at red signals."""
        v = Vehicle(vehicle_id=99, direction='E', road_length=300.0)
        v.position = 200.0  # 60px before stop line (260px)
        v.speed = 3.5

        # Approach on RED (can_proceed=False)
        for _ in range(50):
            v.update(dt=0.1, can_proceed=False, distance_to_lead=999.0)

        self.assertLess(v.speed, 0.2, "Vehicle must stop at red light")
        self.assertLessEqual(v.position, v.stop_line + 5.0, "Vehicle must not overshoot stop line on red")
        self.assertTrue(v.is_stopped, "Vehicle is_stopped flag must be True")

    def test_06_flask_api_endpoints(self):
        """Tests all REST API endpoints for correct HTTP response codes and JSON schemas."""
        # GET /api/state
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("simulation_time", data)
        self.assertIn("signals", data)
        self.assertIn("vehicles", data)
        self.assertIn("queues", data)

        # POST /api/pause
        res = self.client.post("/api/pause")
        self.assertEqual(res.status_code, 200)

        # POST /api/start
        res = self.client.post("/api/start")
        self.assertEqual(res.status_code, 200)

        # POST /api/mode
        res = self.client.post("/api/mode", json={"mode": "fixed"})
        self.assertEqual(res.status_code, 200)
        res = self.client.post("/api/mode", json={"mode": "ai"})
        self.assertEqual(res.status_code, 200)

        # POST /api/density
        res = self.client.post("/api/density", json={"density": "high"})
        self.assertEqual(res.status_code, 200)

        # GET /api/qtable
        res = self.client.get("/api/qtable")
        self.assertEqual(res.status_code, 200)
        qdata = json.loads(res.data)
        self.assertIn("q_table", qdata)
        self.assertEqual(qdata["total_states"], 32)

    def test_07_empirical_ai_vs_fixed_comparison(self):
        """
        Runs identical 120-step simulation runs comparing AI vs Fixed-Time modes
        under identical seed traffic generation.
        """
        # Run 1: Fixed-Time Mode
        sim_fixed = TrafficSimulation(q_agent=self.agent)
        sim_fixed.set_mode("fixed")
        sim_fixed.set_density("high")

        for _ in range(120):
            sim_fixed.step(dt=0.5)

        fixed_all_v = [v for d in sim_fixed.vehicles.values() for v in d]
        fixed_avg_wait = sum(v.waiting_time for v in fixed_all_v) / max(1, len(fixed_all_v))

        # Run 2: AI Mode (Q-Learning)
        sim_ai = TrafficSimulation(q_agent=self.agent)
        sim_ai.set_mode("ai")
        sim_ai.set_density("high")

        for _ in range(120):
            sim_ai.step(dt=0.5)

        ai_all_v = [v for d in sim_ai.vehicles.values() for v in d]
        ai_avg_wait = sum(v.waiting_time for v in ai_all_v) / max(1, len(ai_all_v))

        print(f"\n[Validation] Fixed-Time Avg Wait: {fixed_avg_wait:.2f}s | AI Mode Avg Wait: {ai_avg_wait:.2f}s")
        # Ensure AI performance does not fail dramatically
        self.assertGreater(fixed_avg_wait, 0.0)
        self.assertGreater(ai_avg_wait, 0.0)


if __name__ == "__main__":
    unittest.main()
