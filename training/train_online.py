import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from stable_baselines3 import SAC 
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import asyncio
import nest_asyncio
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import torch as th
from typing import SupportsFloat
from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList, BaseCallback
import random

RUN_FPS = 25
SUBSTEPS_PER_STEP = 5
MODEL_SAVE_FREQ = 50_000
VIDEO_SAVE_FREQ = 10_000
TIME_LIMIT = RUN_FPS * 2 * 60

training_params = dict(
    learning_rate=1e-4,  # be smaller than 2.5e-4
    gamma=0.97,  # rec range .9 - .99 0.999997
    ent_coef=0.05,
    use_sde=True,
    sde_sample_freq=RUN_FPS,
    verbose=1,
    seed=1,
    device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=th.nn.ReLU
    ),
    buffer_size=1_000_000,
    gradient_steps=10,
    tau=0.005
)

def find_latest_model(root_path: Path) -> Optional[Path]:
    """
        Find the path of the latest model if exists.
    """
    logs_path = os.path.join(root_path, "logs")
    if not os.path.exists(logs_path):
        print(f"No previous record found in {logs_path}")
        return None
    print(f"logs_path: {logs_path}")
    files = os.listdir(logs_path)
    paths = sorted(files)
    paths_dict: Dict[int, Path] = {int(path.split("_")[2]): path for path in paths if path.split("_")[2].isdigit()}
    if not paths_dict:
        return None
    latest_model_file_path: Optional[Path] = Path(os.path.join(logs_path, paths_dict[max(paths_dict.keys())]))
    return latest_model_file_path

def get_env() -> Tuple[gym.Env, int]:
    code = random.randint(0, 1000)
    env = asyncio.run(initialize_roar_env(control_timestep=1.0/RUN_FPS, physics_timestep=1.0/(RUN_FPS*SUBSTEPS_PER_STEP)))
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=TIME_LIMIT)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f"videos/run_{code}", step_trigger=lambda x: x % VIDEO_SAVE_FREQ == 0)
    env = Monitor(env, f"logs/run_{code}", allow_early_resets=True)
    print(f"using run identifier {code}")
    return env, code

def main():
    env, code = get_env()

    model = SAC(
        "MlpPolicy",
        env,
        replay_buffer_kwargs={"handle_timeout_termination": True},
        **training_params
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=MODEL_SAVE_FREQ,
        verbose=2,
        save_path=f"logs/"
    )
    event_callback = EveryNTimesteps(
        n_steps=MODEL_SAVE_FREQ,
        callback=checkpoint_callback
    )

    callbacks = CallbackList([
        checkpoint_callback, 
        event_callback
    ])

    model.learn(
        total_timesteps=int(1e7),
        progress_bar=True,
        reset_num_timesteps=False,
        callback=callbacks  # Added callbacks to the learn method
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
