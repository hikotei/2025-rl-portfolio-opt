import torch
from dataclasses import dataclass


@dataclass
class DRLConfig:
    """Configuration class for DRL training parameters"""

    # Window configuration
    n_windows: int = 2
    agents_per_window: int = 2
    base_start_year: int = 2006
    seed_policy: bool = True

    # Environment parameters
    env_window_size: int = 60
    transaction_cost: float = 0.0
    initial_balance: float = 100_000
    reward_scaling: float = 1.0
    eta_dsr: float = 1 / 252

    # Training parameters
    n_envs: int = 10
    total_timesteps_per_round: int = 100_000
    n_steps_per_env: int = 252 * 3
    batch_size: int = 1260
    n_epochs: int = 16
    gamma: float = 0.9
    gae_lambda: float = 0.9
    clip_range: float = 0.25
    log_std_init: float = -1.0

    # Learning rate parameters
    initial_lr: float = 3e-4
    final_lr: float = 1e-5

    # Paths
    model_save_dir: str = "models"
    tensorboard_log_dir: str = "logs"
    data_dir: str = "data"

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=[64, 64],
        log_std_init=log_std_init,
    )