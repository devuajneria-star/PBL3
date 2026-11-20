"""
Q-Learning Agent Offline & Online Training Pipeline
Simulates multi-episode traffic demand to train the tabular Q-table.
"""

import os
import sys
import time
from typing import Dict, List, Any

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.q_learning import QLearningAgent
from backend.traffic_simulation import TrafficSimulation


def train_q_learning_agent(
    episodes: int = 50,
    steps_per_episode: int = 200,
    q_table_path: str = "backend/q_table.json"
) -> Dict[str, Any]:
    """
    Trains the tabular Q-Learning agent on simulated traffic episodes.
    Saves the converged Q-table to disk.
    """
    print("=" * 65)
    print("🚦 INITIATING TABULAR Q-LEARNING TRAFFIC AGENT TRAINING")
    print(f"[*] Episodes: {episodes} | Steps/Episode: {steps_per_episode}")
    print(f"[*] Output Model: {q_table_path}")
    print("=" * 65)

    agent = QLearningAgent(
        learning_rate=0.15,
        discount_factor=0.90,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95,
        q_table_path=q_table_path
    )

    episode_rewards: List[float] = []
    episode_wait_times: List[float] = []
    episode_throughputs: List[int] = []

    start_total_time = time.time()

    for ep in range(1, episodes + 1):
        sim = TrafficSimulation(q_agent=agent)
        sim.mode = "ai"
        
        # Vary density across episodes to create generalizable policy
        densities = ["low", "medium", "high", "rush_hour"]
        sim.set_density(densities[ep % len(densities)])

        cum_reward = 0.0
        prev_phase = sim.phase

        for step in range(steps_per_episode):
            dt = 0.5  # Accelerated simulation time step
            sim.step(dt)

            # Observe state
            queues = sim._get_queue_counts()
            max_waits = sim._get_max_wait_times()

            ns_q = queues['N'] + queues['S']
            ew_q = queues['E'] + queues['W']
            ns_max_wait = max(max_waits['N'], max_waits['S'])
            ew_max_wait = max(max_waits['E'], max_waits['W'])

            curr_phase_idx = 0 if sim.phase in [0, 1, 2] else 1
            state = agent.get_state(ns_q, ew_q, curr_phase_idx, ns_max_wait, ew_max_wait)

            # Agent chooses action with epsilon-greedy exploration
            action = agent.choose_action(state, training=True)

            # Apply selected action to simulation transitions
            if not sim.in_transition and sim.phase_timer >= sim.min_green_time:
                if action == 1 and sim.phase == 0:
                    sim.in_transition = True
                    sim.phase = 1
                    sim.phase_timer = 0.0
                elif action == 0 and sim.phase == 3:
                    sim.in_transition = True
                    sim.phase = 4
                    sim.phase_timer = 0.0

            # Step forward
            prev_passed = sim.total_passed
            sim.step(dt)
            step_passed = sim.total_passed - prev_passed

            # Observe next state
            next_queues = sim._get_queue_counts()
            next_max_waits = sim._get_max_wait_times()
            next_ns_q = next_queues['N'] + next_queues['S']
            next_ew_q = next_queues['E'] + next_queues['W']
            next_phase_idx = 0 if sim.phase in [0, 1, 2] else 1

            next_state = agent.get_state(
                next_ns_q,
                next_ew_q,
                next_phase_idx,
                max(next_max_waits['N'], next_max_waits['S']),
                max(next_max_waits['E'], next_max_waits['W'])
            )

            # Compute reward
            all_v = [v for d in sim.vehicles.values() for v in d]
            tot_wait = sum(v.waiting_time for v in all_v)
            phase_changed = (sim.phase != prev_phase)
            prev_phase = sim.phase

            reward = agent.compute_reward(
                next_ns_q,
                next_ew_q,
                tot_wait,
                step_passed,
                phase_changed
            )
            cum_reward += reward

            # Update Q-table with Bellman error
            agent.update(state, action, reward, next_state)

        # Decay exploration epsilon
        agent.decay_epsilon()

        all_v = [v for d in sim.vehicles.values() for v in d]
        avg_wait = (sum(v.waiting_time for v in all_v) / max(1, len(all_v)))
        
        episode_rewards.append(round(cum_reward, 2))
        episode_wait_times.append(round(avg_wait, 2))
        episode_throughputs.append(sim.total_passed)

        if ep % 5 == 0 or ep == 1 or ep == episodes:
            print(
                f"Episode {ep:02d}/{episodes:02d} | "
                f"Reward: {cum_reward:9.1f} | "
                f"Avg Wait: {avg_wait:5.1f}s | "
                f"Passed: {sim.total_passed:3d} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Save trained Q-table
    agent.save(q_table_path)
    total_elapsed = round(time.time() - start_total_time, 2)

    print("=" * 65)
    print(f"[+] Training completed in {total_elapsed}s!")
    print(f"[+] Final Epsilon: {agent.epsilon:.3f}")
    print(f"[+] Q-Table saved successfully to: {q_table_path}")
    print("=" * 65)

    return {
        "success": True,
        "episodes": episodes,
        "elapsed_time": total_elapsed,
        "final_epsilon": round(agent.epsilon, 3),
        "episode_rewards": episode_rewards,
        "episode_wait_times": episode_wait_times,
        "episode_throughputs": episode_throughputs,
        "q_table_size": len(agent.q_table)
    }


if __name__ == "__main__":
    train_q_learning_agent(episodes=30, steps_per_episode=150)
