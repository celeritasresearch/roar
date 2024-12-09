import gymnasium as gym
import numpy as np
from typing import Any, List, Optional, SupportsFloat, Tuple, Dict, Union
import roar_py_rl_carla
import roar_py_carla
import roar_py_interface
import carla

class SimplifyCarlaActionFilter(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_space = gym.spaces.Dict({
            "throttle": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
            "steer": gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        })
    
    def action(self, action: Dict[str, Union[SupportsFloat, float]]) -> Dict[str, Union[SupportsFloat, float]]:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        # action = {
        #     "throttle": [-1.0, 1.0],
        #     "steer": [-1.0, 1.0]
        # }
        # if isinstance(self.env.unwrapped, roar_py_rl_carla.RoarRLCarlaSimEnv):
        #     velocity = self.env.unwrapped.roar_py_actor.get_linear_3d_velocity()
        #     velocity = np.linalg.norm(velocity)
        
        real_action = {
            "throttle": np.clip(action["throttle"], 0.0, 1.0),
            "brake": np.clip(-action["throttle"], 0.0, 1.0),
            "steer": action["steer"],
            "hand_brake": 0.0,
            "reverse": 0.0
        }
        return real_action

async def initialize_roar_env(
    carla_host : str = "localhost", 
    carla_port : int = 2000, 
    control_timestep : float = 0.05, 
    physics_timestep : float = 0.01,
    waypoint_information_distances : list = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 80.0, 100.0],
    image_width : int = 400,
    image_height : int = 200
):
    carla_client = carla.Client(carla_host, carla_port)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(control_timestep, physics_timestep)
    world.set_asynchronous(False)
    await world.step()
    roar_py_instance.clean_actors_not_registered(["vehicle.*", "sensor.*"])

    spawn_point = world.spawn_points[0]

    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point[0] + np.array([0, 0, 2.0]),
        spawn_point[1],
        True,
        "vehicle"
    )
    assert vehicle is not None, "Failed to spawn vehicle"
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3),
        name="collision_sensor"
    )
    assert collision_sensor is not None, "Failed to attach collision sensor"
    
    #TODO: Attach next waypoint to observation
    velocimeter_sensor = vehicle.attach_velocimeter_sensor("velocimeter")
    local_velocimeter_sensor = vehicle.attach_local_velocimeter_sensor("local_velocimeter")

    location_sensor = vehicle.attach_location_in_world_sensor("location")
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor("roll_pitch_yaw")
    gyroscope_sensor = vehicle.attach_gyroscope_sensor("gyroscope")
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=image_width,
        image_height=image_height
    )
    # occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
    #     50,
    #     50,
    #     4.0,
    #     4.0,
    #     name="occupancy_map"
    # )

    await world.step()
    await vehicle.receive_observation()
    env = roar_py_rl_carla.RoarRLCarlaSimEnv(
        vehicle,
        world.maneuverable_waypoints,
        location_sensor,
        rpy_sensor,
        velocimeter_sensor,
        collision_sensor,
        waypoint_information_distances=set(waypoint_information_distances),
        world = world, 
        collision_threshold = 1.0
    )
    env = SimplifyCarlaActionFilter(env)
    env = gym.wrappers.FilterObservation(env, ["gyroscope", "waypoints_information", "local_velocimeter"])
    return env