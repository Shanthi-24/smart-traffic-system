# smart-traffic-system
import random
import numpy as np
import time

class SmartTrafficController:
    def __init__(self, num_directions=4):
        self.num_directions = num_directions
        self.q_table = np.zeros((num_directions, 2))  # 2 actions: 0 - Red, 1 - Green
        self.vehicle_counts = np.zeros(num_directions, dtype=int)
        self.learning_rate = 0.1
        self.gamma = 0.9  # discount factor

    def detect_vehicles(self):
        """Simulate CNN vehicle detection (replace with real OpenCV/TensorFlow model)."""
        self.vehicle_counts = np.random.randint(0, 20, size=self.num_directions)
        print(f"Detected vehicles: {self.vehicle_counts}")

    def select_best_direction(self):
        """Select direction with most vehicles for green signal."""
        return np.argmax(self.vehicle_counts)

    def apply_traffic_signal(self, green_direction):
        """Apply green signal to one direction, red to others."""
        print(f"\n🚦 Green light for direction {green_direction}")
        for i in range(self.num_directions):
            light = "GREEN" if i == green_direction else "RED"
            print(f"Direction {i}: {light}")

    def update_q_table(self, direction, reward):
        """Update Q-values based on reward."""
        old_value = self.q_table[direction][1]
        self.q_table[direction][1] += self.learning_rate * (reward + self.gamma * np.max(self.q_table[direction]) - old_value)
        print(f"Updated Q-value for direction {direction}: {self.q_table[direction][1]:.2f}")

    def calculate_reward(self, direction):
        """Reward based on traffic reduction; fewer vehicles = better."""
        vehicles = self.vehicle_counts[direction]
        return 1 if vehicles < 10 else -1

    def run_simulation(self, cycles=10):
        for cycle in range(cycles):
            print(f"\n======= Cycle {cycle + 1} =======")
            self.detect_vehicles()
            best_direction = self.select_best_direction()
            self.apply_traffic_signal(best_direction)
            reward = self.calculate_reward(best_direction)
            self.update_q_table(best_direction, reward)
            time.sleep(1)

if __name__ == "__main__":
    controller = SmartTrafficController(num_directions=4)
    controller.run_simulation(cycles=5)
