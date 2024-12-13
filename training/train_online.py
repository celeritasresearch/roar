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
from typing import Optional, Dict
import torch as th
from typing import Dict, SupportsFloat, Union, Tuple
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
    learning_rate = 1e-4,  # be smaller 2.5e-4
    #n_steps = 256 * RUN_FPS, #1024
    # n_epochs=10,
    gamma=0.97,  # rec range .9 - .99 0.999997
    ent_coef=0.05,
    # gae_lambda=0.95,
    # clip_range_vf=None,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq = RUN_FPS,
    # target_kl=None,
    # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
    # create_eval_env=False,
    # policy_kwargs=None,
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
    # _init_setup_model=True,
)

def find_latest_model(root_path: Path) -> Optional[Path]:
    """
        Find the path of latest model if exists.
    """
    logs_path = (os.path.join(root_path, "logs"))
    if os.path.exists(logs_path) is False:
        print(f"No previous record found in {logs_path}")
        return None
    print(f"logs_path: {logs_path}")
    files = os.listdir(logs_path)
    paths = sorted(files)
    paths_dict: Dict[int, Path] = {int(path.split("_")[2]): path for path in paths}
    if len(paths_dict) == 0:
        return None
    latest_model_file_path: Optional[Path] = Path(os.path.join(logs_path, paths_dict[max(paths_dict.keys())]))
    return latest_model_file_path

def get_env() -> Tuple[gym.Env, int]:
    code = random.randint(0, 1000)
    env = asyncio.run(initialize_roar_env(control_timestep=1.0/RUN_FPS, physics_timestep=1.0/(RUN_FPS*SUBSTEPS_PER_STEP)))
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = TIME_LIMIT)
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
        # optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": True},
        **training_params
    )

    checkpoint_callback = CheckpointCallback(
        save_freq = MODEL_SAVE_FREQ,
        verbose = 2,
        save_path = f"logs/"
    )
    event_callback = EveryNTimesteps(
        n_steps = MODEL_SAVE_FREQ,
        callback = checkpoint_callback
    )

    callbacks = CallbackList([
        checkpoint_callback, 
        event_callback
    ])

    model.learn(
        total_timesteps = 1e7,
        progress_bar = True,
        reset_num_timesteps = False,
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
