"""
Q-Learning Agent for Smart Traffic Signal Optimization
Real Tabular Q-Learning Implementation
"""

import json
import os
import random
from typing import Tuple, Dict, Any


class QLearningAgent:
    """
    Tabular Q-Learning Agent for 4-way intersection traffic signal control.
    
    States:
        - NS queue length category: 0 (Low: 0-2), 1 (Medium: 3-6), 2 (High: 7-11), 3 (Congested: 12+)
        - EW queue length category: 0 (Low: 0-2), 1 (Medium: 3-6), 2 (High: 7-11), 3 (Congested: 12+)
        - Current green phase: 0 (North-South Green), 1 (East-West Green)
        State is represented as tuple (ns_cat, ew_cat, current_phase)
        
    Actions:
        - 0: Choose / Maintain North-South Green Phase (NS=Green, EW=Red)
        - 1: Choose / Maintain East-West Green Phase (EW=Green, NS=Red)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        q_table_path: str = "backend/q_table.json"
    ):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_path = q_table_path
        
        # Q-Table: Dict mapping state string -> list of action values [Q(s, a0), Q(s, a1)]
        self.q_table: Dict[str, list] = {}
        self.action_space = [0, 1]  # 0: NS Green, 1: EW Green
        
        # Initialize default Q-table for all possible states
        self._init_q_table()
        
        # Load existing table if present
        if os.path.exists(self.q_table_path):
            self.load()

    def _state_to_key(self, state: Tuple[int, int, int]) -> str:
        """Converts state tuple to string key for Q-table dictionary."""
        return f"{state[0]}_{state[1]}_{state[2]}"

    def _init_q_table(self):
        """Initializes all discrete states with default Q-values."""
        # NS cat (0-3), EW cat (0-3), phase (0-1) => 4 * 4 * 2 = 32 discrete states
        for ns in range(4):
            for ew in range(4):
                for phase in range(2):
                    key = self._state_to_key((ns, ew, phase))
                    if key not in self.q_table:
                        self.q_table[key] = [0.0, 0.0]

    @staticmethod
    def categorize_queue(queue_length: int, max_waiting_time: float = 0.0) -> int:
        """
        Categorizes vehicle demand by considering both queue length AND vehicle waiting time.
        Ensures the AI does not simply look at vehicle count, but considers starvation.
        """
        # Effective queue weighted by waiting time urgency
        urgency_bonus = 0
        if max_waiting_time > 30.0:
            urgency_bonus = 2
        elif max_waiting_time > 15.0:
            urgency_bonus = 1

        effective_metric = queue_length + urgency_bonus

        if effective_metric <= 2:
            return 0  # Low
        elif effective_metric <= 6:
            return 1  # Medium
        elif effective_metric <= 11:
            return 2  # High
        else:
            return 3  # Congested

    def get_state(self, ns_queue: int, ew_queue: int, current_phase: int, 
                  ns_max_wait: float = 0.0, ew_max_wait: float = 0.0) -> Tuple[int, int, int]:
        """Maps continuous simulation measurements to discrete RL state tuple."""
        ns_cat = self.categorize_queue(ns_queue, ns_max_wait)
        ew_cat = self.categorize_queue(ew_queue, ew_max_wait)
        return (ns_cat, ew_cat, current_phase)

    def choose_action(self, state: Tuple[int, int, int], training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        If training=False, exploits best Q-value deterministically.
        """
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0]

        # Exploration vs Exploitation
        if training and random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        # Exploitation: Argmax Q(s, a)
        q_values = self.q_table[key]
        if q_values[0] == q_values[1]:
            # Tie-breaker: prefer currently green phase to prevent unnecessary flickering
            return state[2]
        return 0 if q_values[0] > q_values[1] else 1

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int]
    ) -> float:
        """
        Bellman equation update:
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        curr_key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        if curr_key not in self.q_table:
            self.q_table[curr_key] = [0.0, 0.0]
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0, 0.0]

        current_q = self.q_table[curr_key][action]
        max_future_q = max(self.q_table[next_key])
        
        # TD Target and Error
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q

        # Q-value update
        self.q_table[curr_key][action] = current_q + self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        """Gradually decays exploration rate toward epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_reward(
        self,
        ns_queue: int,
        ew_queue: int,
        total_waiting_time: float,
        vehicles_passed: int,
        phase_changed: bool
    ) -> float:
        """
        Reward function formulation:
        - Penalizes total queued vehicles
        - Penalizes accumulated waiting time
        - Rewards vehicles successfully passing the intersection (throughput in current step)
        - Small penalty for rapid phase oscillation (driver comfort & safety)
        """
        total_queue = ns_queue + ew_queue
        
        # 1. Queue penalty (quadratic penalty to penalize heavy bottleneck accumulation)
        queue_penalty = -0.5 * (total_queue ** 1.3)
        
        # 2. Waiting time penalty
        wait_penalty = -0.15 * total_waiting_time
        
        # 3. Throughput reward (based on vehicles clearing in this step)
        throughput_reward = 12.0 * vehicles_passed
        
        # 4. Phase change penalty (prevents switching green lights every single second)
        switch_penalty = -2.5 if phase_changed else 0.5
        
        return float(queue_penalty + wait_penalty + throughput_reward + switch_penalty)

    def save(self, filepath: str = None):
        """Saves Q-table and agent state to JSON."""
        target_path = filepath or self.q_table_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
        with open(target_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str = None) -> bool:
        """Loads Q-table from JSON file."""
        target_path = filepath or self.q_table_path
        if not os.path.exists(target_path):
            return False
        try:
            with open(target_path, "r") as f:
                data = json.load(f)
            self.q_table = data.get("q_table", self.q_table)
            self.epsilon = data.get("epsilon", self.epsilon_min)
            return True
        except Exception as e:
            print(f"[!] Warning: Failed to load Q-table: {e}")
            return False
