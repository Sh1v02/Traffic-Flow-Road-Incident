import random
from typing import Dict, Text, Optional

import numpy as np

from HighwayEnv.highway_env import utils
from HighwayEnv.highway_env.envs.common.abstract import AbstractEnv
from HighwayEnv.highway_env.envs.common.action import Action
from HighwayEnv.highway_env.road.road import Road, RoadNetwork
from HighwayEnv.highway_env.utils import near_split
from HighwayEnv.highway_env.vehicle.behavior import BrokenDownVehicle
from HighwayEnv.highway_env.vehicle.controller import ControlledVehicle
from HighwayEnv.highway_env.vehicle.kinematics import Vehicle
from HighwayEnv.highway_env.vehicle.objects import Obstacle
from src.Utilities import graphics_settings, multi_agent_settings

Observation = np.ndarray


# TODO: Allow for multi-agent compatibility (check IntersectionEnvironment step function and look at the methods called)
class HighwayEnvWithObstructions(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    # "see_behind": True,
                    "normalize": False,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "absolute": False,
                    "order": "sorted",
                    # "observe_intentions": True
                },
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": graphics_settings.LANE_COUNT,
                "vehicles_count": 0,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 90,  # [s]
                "ego_spacing": 0.5,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "absolute": True,
                "offroad_terminal": True,
                "simulation_frequency": 15,  # [Hz]
                "policy_frequency": 1,  # [Hz]
                "obstruction_count": graphics_settings.OBSTRUCTION_COUNT,
                "obstruction_type": "BrokenDownVehicle"  # [Obstacle, BrokenDownVehicle]
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        # Create obstructions
        for i in range(self.config["obstruction_count"]):
            vehicle = other_vehicles_type.create_random(
                self.road, speed=0, spacing=1 / self.config["vehicles_density"]
            )
            position = [vehicle.position[0] + (50 * (i + 1)), vehicle.position[1]]
            obstruction = Obstacle(self.road, position) if self.config["obstruction_type"] == "Obstacle" else (
                BrokenDownVehicle(self.road, position))
            self.road.objects.append(obstruction)

    # TODO: Understand whether the average reward is used for each agents TD-error,
    #  or if its own individual reward is used
    def _reward(self, action: Action) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    # TODO: Create a MultiAgentWrapper that handles this to prevent duplication
    def _agent_reward(self, vehicle: Vehicle) -> float:
        """
        Per-agent reward signal.
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._agent_rewards(vehicle)
        if rewards["on_road_reward"] == 0.0:
            return -15.0
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        # reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        # reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"],
                 # self.config["lane_centering_reward"],
                 # self.config["arrived_reward]],
                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _agent_rewards(self, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)

        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        # TODO: For lane centering, look at racetrack_env.py _rewards()
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            # "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": float(vehicle.on_road),
            # "in_lane": 1 - (lateral / self.vehicle.lane.width) ** 2,
            # # TODO: Remove lane_center_reward/replace with this reward?
            # "lane_centering_reward": 1
            #                          / (1 + self.config["lane_centering_cost"] * lateral ** 2),
        }

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _is_terminated(self) -> bool:
        """The episode is over if any of the agents are in a terminal state (crashed, offroad if applicable)."""
        return (
            any(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        )

    # TODO: (High) End episode when all agents have passed the last obstruction
    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the vehicle is offroad."""
        return vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: Optional[Action] = None) -> dict:
        # info = super()._info(obs, action)
        info = {
            "agents_actions": action,
            "agents_rewards": tuple(self._agent_reward(vehicle) for vehicle in self.controlled_vehicles),
            "agents_dones": tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles),
            "agents_speeds": tuple(vehicle.speed for vehicle in self.controlled_vehicles)
        }
        return info
