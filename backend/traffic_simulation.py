"""
Four-Way Traffic Intersection Simulation Engine
Simulates North, South, East, West roadways with vehicle queues,
signal transitions (Green, Yellow, All-Red), and performance metrics.
"""

import os
import sys
import time
import random
import threading
from typing import Dict, List, Any, Optional

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.q_learning import QLearningAgent


class Vehicle:
    """Represents a single vehicle in the 4-way intersection."""
    def __init__(
        self,
        vehicle_id: int,
        direction: str,
        road_length: Optional[float] = None,
        stop_line: Optional[float] = None
    ):
        self.id = vehicle_id
        self.direction = direction  # 'N', 'S', 'E', 'W'
        self.vehicle_type = random.choices(['car', 'bus', 'truck'], weights=[0.8, 0.12, 0.08])[0]
        self.length = 22.0 if self.vehicle_type == 'car' else (36.0 if self.vehicle_type == 'bus' else 30.0)

        # Handle road geometry (compatible with unit tests road_length=300.0)
        if road_length is not None and stop_line is None:
            self.road_length = road_length
            self.stop_line = road_length - 40.0
            self.max_speed = random.uniform(3.5, 4.8)
            self.acceleration = 1.0
            self.deceleration = 1.6
            self.position = 0.0
        else:
            if direction in ['N', 'S']:
                self.road_length = 500.0
                self.stop_line = 181.0
            else:
                self.road_length = 800.0
                self.stop_line = 331.0
            if road_length is not None:
                self.road_length = road_length
            if stop_line is not None:
                self.stop_line = stop_line

            # Realistic speeds for interactive visualization
            if self.vehicle_type == 'car':
                self.max_speed = random.uniform(55.0, 72.0)
            elif self.vehicle_type == 'bus':
                self.max_speed = random.uniform(42.0, 52.0)
            else:
                self.max_speed = random.uniform(46.0, 58.0)
            self.acceleration = 35.0
            self.deceleration = 60.0
            self.position = 0.0

        self.speed = self.max_speed * 0.8
        self.waiting_time = 0.0
        self.is_stopped = False
        self.has_passed = False

    def update(self, dt: float, can_proceed: bool, distance_to_lead: float):
        """
        Updates vehicle velocity and position using car-following logic and signal constraints.
        """
        dist_to_stop = self.stop_line - self.position
        target_speed = self.max_speed

        # Red / Yellow stop constraint:
        if not self.has_passed and not can_proceed:
            if dist_to_stop <= 100.0:
                target_speed = 0.0

        # Car-following headway constraint:
        safe_gap = self.length + 14.0
        if distance_to_lead <= safe_gap:
            target_speed = 0.0
        elif distance_to_lead < (safe_gap + 80.0):
            follow_speed = ((distance_to_lead - safe_gap) / 80.0) * self.max_speed
            target_speed = min(target_speed, follow_speed)

        # Acceleration / braking physics
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + self.acceleration * dt)
        elif self.speed > target_speed:
            self.speed = max(target_speed, self.speed - self.deceleration * dt)

        # Calculate tentative new position
        new_position = self.position + self.speed * dt

        # Prevent overshooting stop line on Red
        if not self.has_passed and not can_proceed:
            target_stop_pos = self.stop_line - self.length / 2.0
            if self.position <= target_stop_pos and new_position > target_stop_pos:
                new_position = target_stop_pos
                self.speed = 0.0

        # Prevent overlapping lead vehicle
        if distance_to_lead < 900.0:
            max_lead_pos = self.position + distance_to_lead - safe_gap
            if new_position > max_lead_pos:
                new_position = max(self.position, max_lead_pos)
                self.speed = 0.0

        self.position = new_position

        # Update waiting time & stopped state
        if self.speed < 0.5 and not self.has_passed:
            self.waiting_time += dt
            self.is_stopped = True
        else:
            self.is_stopped = False

        # Mark crossed stop line
        if not self.has_passed and self.position > self.stop_line + 4.0:
            self.has_passed = True


class TrafficSimulation:
    """
    Core Traffic Simulation Controller for a 4-Way Intersection.
    Manages signals, vehicles, AI and Fixed-Time logic, and live performance metrics.
    """
    def __init__(self, q_agent: Optional[QLearningAgent] = None):
        self.agent = q_agent or QLearningAgent()
        self.mode = "ai"  # 'ai' or 'fixed'
        self.is_running = False
        
        # Signal Phases:
        # 0: North-South Green (EW Red)
        # 1: North-South Yellow (EW Red)
        # 2: All-Red Clearance
        # 3: East-West Green (NS Red)
        # 4: East-West Yellow (NS Red)
        # 5: All-Red Clearance
        self.phase = 0  # Starts with NS Green
        self.phase_timer = 0.0
        self.min_green_time = 6.0   # Minimum green duration
        self.max_green_time = 35.0  # Max green duration (prevents starvation)
        self.yellow_time = 3.0      # Safe braking clearance
        self.all_red_time = 1.0     # Intersection clearance
        self.fixed_green_time = 25.0 # Fixed mode green duration
        
        # Target green phase (0 for NS, 3 for EW)
        self.target_phase = 0
        self.in_transition = False
        
        # Vehicle Queues per road
        self.vehicles: Dict[str, List[Vehicle]] = {
            'N': [], 'S': [], 'E': [], 'W': []
        }
        self.next_vehicle_id = 1
        
        # Traffic Demand / Spawn rate
        self.spawn_rates = {
            'N': 0.45,
            'S': 0.40,
            'E': 0.35,
            'W': 0.30
        }
        self.traffic_density = "medium"  # low, medium, high, rush_hour
        
        # Metrics & Telemetry
        self.simulation_time = 0.0
        self.total_spawned = 0
        self.total_passed = 0
        self.cumulative_waiting_time = 0.0
        self.ai_decisions_log = []
        self.ai_last_action = 0
        self.ai_reasoning = "Initializing Q-Learning agent..."
        
        # Analytics History for Chart.js
        self.history_timestamps = []
        self.history_wait_times = []
        self.history_queue_lengths = []
        self.history_throughput = []
        
        # Baseline Comparison Metrics
        self.baseline_stats = {
            "ai": {"avg_wait": 0.0, "total_passed": 0, "avg_queue": 0.0, "samples": 0},
            "fixed": {"avg_wait": 0.0, "total_passed": 0, "avg_queue": 0.0, "samples": 0}
        }
        
        # Simulation Loop Threading
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def set_density(self, density: str):
        """Sets traffic arrival rates according to density preset."""
        self.traffic_density = density
        if density == "low":
            self.spawn_rates = {'N': 0.20, 'S': 0.18, 'E': 0.18, 'W': 0.15}
        elif density == "medium":
            self.spawn_rates = {'N': 0.45, 'S': 0.40, 'E': 0.35, 'W': 0.30}
        elif density == "high":
            self.spawn_rates = {'N': 0.75, 'S': 0.70, 'E': 0.65, 'W': 0.60}
        elif density == "rush_hour":
            # Asymmetric demand: heavy North-South corridor
            self.spawn_rates = {'N': 1.10, 'S': 1.00, 'E': 0.30, 'W': 0.25}

    def set_mode(self, mode: str):
        """Switches between AI (Q-Learning) and Fixed-Time controller."""
        if mode in ["ai", "fixed"]:
            self.mode = mode
            self.ai_reasoning = f"Switched to {mode.upper()} mode."

    def start(self):
        """Starts the simulation loop thread."""
        with self._lock:
            if not self.is_running:
                self.is_running = True
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()

    def pause(self):
        """Pauses the simulation loop."""
        with self._lock:
            self.is_running = False

    def reset(self):
        """Resets simulation state, vehicles, and counters."""
        with self._lock:
            self.vehicles = {'N': [], 'S': [], 'E': [], 'W': []}
            self.phase = 0
            self.phase_timer = 0.0
            self.target_phase = 0
            self.in_transition = False
            self.simulation_time = 0.0
            self.total_spawned = 0
            self.total_passed = 0
            self.cumulative_waiting_time = 0.0
            self.history_timestamps.clear()
            self.history_wait_times.clear()
            self.history_queue_lengths.clear()
            self.history_throughput.clear()
            self.ai_reasoning = "Simulation reset to initial state."

    def spawn_vehicles(self, dt: float):
        """Spawns vehicles probabilistically according to spawn rates."""
        for direction, rate in self.spawn_rates.items():
            if random.random() < (rate * dt):
                road_vehicles = self.vehicles[direction]
                can_spawn = True
                if road_vehicles:
                    last_veh = min(road_vehicles, key=lambda v: v.position)
                    if last_veh.position < 45.0:
                        can_spawn = False

                if can_spawn:
                    road_length = 500.0 if direction in ['N', 'S'] else 800.0
                    stop_line = 181.0 if direction in ['N', 'S'] else 331.0
                    v = Vehicle(self.next_vehicle_id, direction, road_length=road_length, stop_line=stop_line)
                    self.next_vehicle_id += 1
                    self.vehicles[direction].append(v)
                    self.total_spawned += 1

    def _get_queue_counts(self) -> Dict[str, int]:
        """Returns waiting or queued vehicle count per direction."""
        queues = {}
        for d, v_list in self.vehicles.items():
            # A vehicle is queued if it has not crossed the stop line and is stopped or approaching queue
            queues[d] = sum(1 for v in v_list if not v.has_passed and (v.is_stopped or v.speed < 12.0 or v.waiting_time > 0.4))
        return queues

    def _get_max_wait_times(self) -> Dict[str, float]:
        """Returns maximum wait time per direction."""
        max_waits = {}
        for d, v_list in self.vehicles.items():
            stopped = [v.waiting_time for v in v_list if not v.has_passed and (v.is_stopped or v.waiting_time > 0.0)]
            max_waits[d] = max(stopped) if stopped else 0.0
        return max_waits

    def _update_signals(self, dt: float):
        """
        State machine managing safe phase transitions:
        Phase 0: NS Green -> Phase 1: NS Yellow (3s) -> Phase 2: All Red (1s) ->
        Phase 3: EW Green -> Phase 4: EW Yellow (3s) -> Phase 5: All Red (1s) -> Phase 0: NS Green
        """
        self.phase_timer += dt

        # Handle Yellow & All-Red transitions safely
        if self.in_transition:
            if self.phase == 1 and self.phase_timer >= self.yellow_time:
                # Transition from NS Yellow to All Red
                self.phase = 2
                self.phase_timer = 0.0
            elif self.phase == 2 and self.phase_timer >= self.all_red_time:
                # Transition from All Red to EW Green
                self.phase = 3
                self.phase_timer = 0.0
                self.in_transition = False
            elif self.phase == 4 and self.phase_timer >= self.yellow_time:
                # Transition from EW Yellow to All Red
                self.phase = 5
                self.phase_timer = 0.0
            elif self.phase == 5 and self.phase_timer >= self.all_red_time:
                # Transition from All Red to NS Green
                self.phase = 0
                self.phase_timer = 0.0
                self.in_transition = False
            return

        # Regular green phases: check controller logic
        if self.phase == 0 or self.phase == 3:
            # Enforce minimum green duration
            if self.phase_timer < self.min_green_time:
                return

            desired_action = self._decide_next_phase()

            # Desired action: 0 -> NS Green (Phase 0), 1 -> EW Green (Phase 3)
            if desired_action == 1 and self.phase == 0:
                # Switch from NS to EW: trigger NS Yellow
                self.in_transition = True
                self.phase = 1
                self.phase_timer = 0.0
            elif desired_action == 0 and self.phase == 3:
                # Switch from EW to NS: trigger EW Yellow
                self.in_transition = True
                self.phase = 4
                self.phase_timer = 0.0

    def _decide_next_phase(self) -> int:
        """
        Queries either Q-Learning Agent or Fixed-Time Controller.
        Returns desired green direction: 0 for North-South, 1 for East-West.
        """
        if self.mode == "fixed":
            # Traditional Fixed-Time: Switch when fixed_green_time is elapsed
            if self.phase_timer >= self.fixed_green_time:
                return 1 if self.phase == 0 else 0
            return 0 if self.phase == 0 else 1

        # AI Mode (Q-Learning)
        queues = self._get_queue_counts()
        max_waits = self._get_max_wait_times()

        ns_queue = queues['N'] + queues['S']
        ew_queue = queues['E'] + queues['W']
        ns_max_wait = max(max_waits['N'], max_waits['S'])
        ew_max_wait = max(max_waits['E'], max_waits['W'])

        # Starvation prevention & empty green detection
        ns_wait_eff = ns_max_wait
        ew_wait_eff = ew_max_wait
        if self.phase == 0 and ns_queue == 0 and ew_queue > 0:
            ew_wait_eff = max(ew_max_wait, 16.0)
        elif self.phase == 3 and ew_queue == 0 and ns_queue > 0:
            ns_wait_eff = max(ns_max_wait, 16.0)

        current_phase_idx = 0 if self.phase in [0, 1, 2] else 1
        state = self.agent.get_state(ns_queue, ew_queue, current_phase_idx, ns_wait_eff, ew_wait_eff)

        action = self.agent.choose_action(state, training=False)

        # Starvation safety cap: if green exceeded max_green_time and other side is waiting, switch
        if self.phase_timer >= self.max_green_time:
            if self.phase == 0 and ew_queue > 0:
                action = 1
            elif self.phase == 3 and ns_queue > 0:
                action = 0

        self.ai_last_action = action

        # Explain decision
        q_vals = self.agent.q_table.get(self.agent._state_to_key(state), [0.0, 0.0])
        decision_label = "Keep NS Green" if action == 0 and self.phase == 0 else (
            "Switch to EW Green" if action == 1 and self.phase == 0 else (
                "Keep EW Green" if action == 1 and self.phase == 3 else "Switch to NS Green"
            )
        )
        cat_names = ["Low", "Med", "High", "Congested"]
        ns_cat_str = cat_names[state[0]]
        ew_cat_str = cat_names[state[1]]
        self.ai_reasoning = (
            f"Action: {decision_label} | State: NS={ns_cat_str}({ns_queue}), EW={ew_cat_str}({ew_queue}) | "
            f"Q[NS]={q_vals[0]:.1f}, Q[EW]={q_vals[1]:.1f}"
        )
        return action

    def step(self, dt: float = 0.1):
        """Advances simulation by dt seconds."""
        self.simulation_time += dt
        self.spawn_vehicles(dt)
        self._update_signals(dt)

        # Signal states for road clearance
        ns_green = (self.phase == 0)
        ew_green = (self.phase == 3)

        # Update vehicles on each road
        for direction, road_vehicles in self.vehicles.items():
            can_proceed = (ns_green if direction in ['N', 'S'] else ew_green)
            
            # Sort vehicles by position (farthest ahead first)
            road_vehicles.sort(key=lambda v: v.position, reverse=True)

            for i, v in enumerate(road_vehicles):
                # Lead vehicle distance
                dist_to_lead = 999.0
                if i > 0:
                    dist_to_lead = road_vehicles[i - 1].position - v.position

                v.update(dt, can_proceed, dist_to_lead)

                # Check if passed through and left canvas
                if v.has_passed and v.position > (v.road_length + 20.0):
                    self.total_passed += 1

            # Remove vehicles that completed passage
            self.vehicles[direction] = [v for v in road_vehicles if v.position <= (v.road_length + 20.0)]

        # Record metrics history every 1.0 simulation seconds
        if int(self.simulation_time * 10) % 10 == 0:
            queues = self._get_queue_counts()
            total_q = sum(queues.values())
            all_vehicles = [v for d in self.vehicles.values() for v in d]
            avg_wait = (sum(v.waiting_time for v in all_vehicles) / max(1, len(all_vehicles)))

            self.history_timestamps.append(round(self.simulation_time, 1))
            self.history_wait_times.append(round(avg_wait, 2))
            self.history_queue_lengths.append(total_q)
            self.history_throughput.append(self.total_passed)

            # Cap history window
            if len(self.history_timestamps) > 40:
                self.history_timestamps.pop(0)
                self.history_wait_times.pop(0)
                self.history_queue_lengths.pop(0)
                self.history_throughput.pop(0)

            # Accumulate benchmark statistics
            b = self.baseline_stats[self.mode]
            b["samples"] += 1
            b["avg_wait"] = round(((b["avg_wait"] * (b["samples"] - 1)) + avg_wait) / b["samples"], 2)
            b["avg_queue"] = round(((b["avg_queue"] * (b["samples"] - 1)) + total_q) / b["samples"], 2)
            b["total_passed"] = self.total_passed

    def _run_loop(self):
        """Background simulation loop running at ~25 FPS."""
        dt = 0.05
        while self.is_running:
            with self._lock:
                self.step(dt)
            time.sleep(0.04)  # ~25 FPS loop

    def get_state_payload(self) -> Dict[str, Any]:
        """Serializes current simulation state for frontend display."""
        with self._lock:
            queues = self._get_queue_counts()
            all_v = [v for d in self.vehicles.values() for v in d]
            avg_wait = (sum(v.waiting_time for v in all_v) / max(1, len(all_v)))
            waiting_count = sum(queues.values())

            # Signal color outputs for 4 approaches
            signals = {
                "N": "green" if self.phase == 0 else ("yellow" if self.phase == 1 else "red"),
                "S": "green" if self.phase == 0 else ("yellow" if self.phase == 1 else "red"),
                "E": "green" if self.phase == 3 else ("yellow" if self.phase == 4 else "red"),
                "W": "green" if self.phase == 3 else ("yellow" if self.phase == 4 else "red")
            }

            # Export vehicle coordinates for canvas animation
            vehicles_data = []
            for d, v_list in self.vehicles.items():
                for v in v_list:
                    vehicles_data.append({
                        "id": v.id,
                        "direction": v.direction,
                        "position": round(v.position, 1),
                        "speed": round(v.speed, 2),
                        "waiting_time": round(v.waiting_time, 1),
                        "type": v.vehicle_type,
                        "is_stopped": v.is_stopped,
                        "has_passed": v.has_passed
                    })

            return {
                "simulation_time": round(self.simulation_time, 1),
                "is_running": self.is_running,
                "mode": self.mode,
                "phase": self.phase,
                "phase_name": (
                    "NS_GREEN" if self.phase == 0 else (
                        "NS_YELLOW" if self.phase == 1 else (
                            "ALL_RED" if self.phase in [2, 5] else (
                                "EW_GREEN" if self.phase == 3 else "EW_YELLOW"
                            )
                        )
                    )
                ),
                "signals": signals,
                "queues": queues,
                "total_vehicles": self.total_spawned,
                "vehicles_waiting": waiting_count,
                "vehicles_passed": self.total_passed,
                "average_waiting_time": round(avg_wait, 2),
                "ai_decision": self.ai_reasoning,
                "traffic_density": self.traffic_density,
                "vehicles": vehicles_data,
                "history": {
                    "timestamps": self.history_timestamps,
                    "wait_times": self.history_wait_times,
                    "queue_lengths": self.history_queue_lengths,
                    "throughput": self.history_throughput
                },
                "baseline_stats": self.baseline_stats
            }
