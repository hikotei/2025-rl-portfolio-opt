# GPU Utilization in DRL Agent Training

This document explains potential reasons for high CPU usage during DRL agent training and provides guidance on utilizing GPUs, particularly on macOS.

## Why is the training CPU-heavy?

Based on the analysis of `drl_agent_jules.py` and `drl_train_sliding_window_jules.ipynb`, several factors can contribute to high CPU usage:

1.  **Data Preprocessing:** Loading, slicing, and manipulating large financial datasets using `pandas` is primarily a CPU-bound task.
2.  **Environment Interactions:** The `PortfolioEnv` (especially if complex) runs in parallel across multiple processes (`SubprocVecEnv`). While parallelization helps, the core logic of stepping through the environment, calculating rewards, and managing state likely runs on the CPU.
3.  **Agent Training Overhead:** While `stable-baselines3` with PyTorch can offload neural network computations to the GPU, there's still overhead in managing the training loop, collecting experiences, and coordinating parallel environments. These parts often run on the CPU.
4.  **Python Global Interpreter Lock (GIL):** The GIL can limit the true parallelism of Python threads, potentially affecting CPU-bound tasks even with parallelization attempts.
5.  **Metric Calculation:** Post-training analysis and metric calculations using `numpy` and `pandas` are also CPU-intensive.

## Should RL Training Utilize GPU?

Yes, RL training, especially with deep neural networks (as used in PPO), can significantly benefit from GPU acceleration. GPUs excel at the parallel matrix multiplications and tensor operations that form the core of neural network forward and backward passes. Offloading these to the GPU can lead to:

*   **Faster training times:** Reducing the time taken for each training iteration.
*   **Ability to train larger models:** GPUs can handle more complex models with more parameters than CPUs.
*   **More experimentation:** Faster training allows for more iterations of model tuning and hyperparameter optimization.

However, it's important to note that not all parts of an RL pipeline are GPU-accelerated. Data loading, preprocessing, and environment simulation often remain on the CPU. The goal is to ensure the neural network computations, which are typically the bottleneck, are efficiently handled by the GPU.

## How to Enable More GPU Computation

The current codebase uses `stable-baselines3` which, in turn, uses PyTorch. PyTorch should automatically detect and use an available GPU, including Apple's Metal Performance Shaders (MPS) on compatible Macs.

Here's how to ensure and potentially improve GPU utilization:

1.  **Verify MPS Availability and PyTorch Setup (for macOS):**
    *   Ensure you have a Mac with an M1 chip or later and macOS 12.3+.
    *   Make sure your PyTorch installation was built with MPS enabled. You can check this and explicitly set the device to "mps":
        ```python
        import torch

        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")
        else:
            print("MPS is available!")
            mps_device = torch.device("mps")
            # Example: Move model to MPS
            # model.to(mps_device)
            # Example: Create tensor on MPS
            # x = torch.ones(5, device=mps_device)
        ```
    *   In `drl_agent_jules.py`, within the `DRLAgent` class, you can explicitly set the device for the `stable-baselines3` model:
        ```python
        # Inside DRLAgent.__init__ or a similar setup method
        import torch
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=self.tensorboard_log,
            verbose=0,
            device=self.device, # Explicitly pass the device
            n_steps=self.n_steps,
            ent_coef=self.ent_coef,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            n_epochs=self.n_epochs
        )
        ```
        *Note: The `DRLAgent` in the provided code already uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. You'll need to modify this to include "mps".*

2.  **Ensure `stable-baselines3` is Using the Correct Device:**
    *   As shown above, when initializing the PPO model (or other SB3 models), explicitly pass the `device` argument (e.g., `device="mps"` or `device="cuda"`). While SB3 often auto-detects, being explicit can prevent issues. The subtask analysis indicated that the code relies on default behavior, so making this explicit is a good step.

3.  **Monitor GPU Usage:**
    *   On macOS, use `Activity Monitor` (select Window > GPU History) to observe GPU utilization during training.
    *   For NVIDIA GPUs (if applicable on other systems), use `nvidia-smi`.
    *   If GPU utilization is low, it might indicate that CPU-bound operations are the bottleneck, or the model is too small for significant GPU speedup.

4.  **Optimize Data Loading and Preprocessing:**
    *   If data loading is a bottleneck, consider:
        *   Using more efficient file formats (e.g., Feather, Parquet, HDF5).
        *   Optimizing `pandas` operations.
        *   Performing preprocessing steps in advance and saving the results.
        *   Using PyTorch's `DataLoader` with `num_workers` if applicable, though this is more common in supervised learning.

5.  **Profile Your Code:**
    *   Use Python's built-in `cProfile` or other profiling tools (e.g., `line_profiler`, `py-spy`) to identify the exact bottlenecks in your code. This will show whether the CPU time is spent in data handling, environment simulation, or other parts of the RL loop.

6.  **Consider Model Complexity:**
    *   If the neural network in your PPO agent is very small, the overhead of transferring data to/from the GPU might negate the benefits. For deeper or wider networks, GPU acceleration becomes more pronounced.

7.  **Batch Size:**
    *   Ensure your `batch_size` for training is large enough to effectively utilize GPU parallelism. Small batch sizes might not saturate the GPU's computational capacity. The `DRLAgent` uses `batch_size=1024` which is a reasonable start.

By systematically checking these points, you should be able to identify why GPU utilization might be low and take steps to improve it.
