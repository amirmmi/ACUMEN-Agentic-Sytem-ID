# ACUMEN: Active Cross-Entropy Method with Uncertainty-driven Neural ODEs

## Project Overview
This repository contains the implementation of **ACUMEN** (Active Cross-Entropy Method with Uncertainty-driven Neural ODEs), a data-efficient framework for system identification in healthcare applications. ACUMEN addresses the challenge of learning accurate physiological models from limited and noisy time-series data by combining Neural Ordinary Differential Equations (Neural ODEs) with uncertainty-driven active exploration via Cross-Entropy Method Model Predictive Control (CEM-MPC).

The framework iteratively alternates between uncertainty-guided exploration and model retraining, progressively directing data collection toward underrepresented regions of the state space while avoiding redundant sampling. This approach is particularly valuable in healthcare settings where data collection is expensive, ethically constrained, or poses patient burden.


**Key features include:**
- **Neural ODE Ensemble**: Training an ensemble of Neural ODEs to model continuous-time physiological dynamics with epistemic uncertainty quantification
- **Uncertainty-Driven Exploration**: Using CEM-MPC to select actions that maximize epistemic uncertainty, with optimistic state progression and adaptive scaling
- **Statistical Validation**: Comprehensive bootstrap analysis and statistical testing demonstrating robust improvements
- **Healthcare Applications**: Evaluation in the Reinforcement Learning for Deep Brain Stimulation (RL-DBS) environment, demonstrating up to 24.2% improvement in mean squared prediction error over passive data collection
- **Reproducible Implementation**: Full training, evaluation, and visualization workflow provided in multiple Jupyter notebooks, with the main implementation in `NeuralODE_CEM_MPC_Method.ipynb`

**Research Impact:**
- Achieves statistically significant improvements (p = 0.017) with 15.7% ± 8.8% average MSE reduction
- Demonstrates consistent superiority across all tested sample sizes (Wilcoxon signed-rank test: p = 0.031)
- Provides both improved accuracy and better uncertainty calibration for healthcare applications

This work was developed as part of a research project inspired by Google Summer of Code (GSoC) guidelines, focusing on advancing uncertainty-aware active learning for time-series data in healthcare.

## Algorithm Overview

![ACUMEN Algorithm Overview](algorithm.svg)

*Figure: Overview of the ACUMEN framework showing the integration of Neural ODEs with CEM-MPC for uncertainty-driven active learning.*

## Methodology

### Problem Formulation
ACUMEN addresses system identification in healthcare as learning patient-specific dynamical models from irregular and noisy time-series. Given latent state $x(t) ∈ ℝⁿ$ (e.g., neural or cardiovascular activity) and external interventions $u(t) ∈ ℝᵈ$ (e.g., stimulation, drug dosage), observations are collected at irregular times with dynamics governed by $\dot{x}(t) = f_θ(x(t), u(t), t)$.

### Core Components
1. **Neural ODE Ensemble**: Multiple Neural ODE models parameterized by MLPs to capture continuous-time dynamics with epistemic uncertainty quantification through model disagreement
2. **CEM-MPC Planning**: Cross-Entropy Method for Model Predictive Control adapted to maximize epistemic uncertainty with optimistic state progression and action smoothness penalties
3. **Adaptive Scaling**: Normalization of uncertainty estimates across heterogeneous state variables to prevent bias toward high-variance dimensions
4. **Iterative Refinement**: Progressive model improvement through targeted data collection in uncertain regions

## Contributions
### Framework Implementation
- Developed Neural ODE-based surrogate models for continuous-time system identification.
- Implemented ensemble-based epistemic uncertainty estimation.
- Created the CEM-MPC planner for uncertainty-driven exploration, including optimistic state progression and novelty-based terms.
- Added adaptive scaling for handling heterogeneous state variables and incremental retraining.
- Integrated the full iterative ACUMEN algorithm.

### Environment and Evaluation Setup
- Integrated with the RL-DBS environment for simulating neuromodulation.
- Built data collection pipelines for active (CEM-MPC) and baseline (random) exploration.
- Implemented evaluation scripts to compute MSE learning curves, per-channel uncertainty, and trajectory visualizations.


### Code and Documentation
- Modular Python codebase with models, algorithms, and utilities.
- Multiple Jupyter notebooks for interactive analysis:
  - `NeuralODE_CEM_MPC_Method.ipynb`: Main implementation of the ACUMEN framework
  - `Baseline model.ipynb`: Baseline model implementation
  - `Baseline model Epistemic Loss1.ipynb`: Epistemic uncertainty analysis
  - `Baseline model ASID.ipynb`: Alternative system identification approaches
- Example code demonstrates use of both `stable-baselines3` (PyTorch) and `stable-baselines` (TensorFlow) RL libraries.

All code is original and reproducible.

No upstream merges are applicable as this is a standalone research implementation, but the code is designed for easy integration into larger projects.

## Current State
The project is fully functional and reproduces the results reported:
- Achieves 22.5% average MSE reduction (0.002046 vs. 0.002640) and tighter uncertainty bands in the RL-DBS environment.
- Supports custom environments, hyperparameters, and extensions to other healthcare domains.
- Tested on Python 3.8+ with PyTorch; runs on standard hardware (CPU/GPU).


The repository includes:
- Multiple Jupyter notebooks containing the full code implementation:
  - `NeuralODE_CEM_MPC_Method.ipynb`: Main ACUMEN framework implementation
  - `Baseline model.ipynb`: Baseline model training and evaluation  
  - `Baseline model Epistemic Loss1.ipynb`: Epistemic uncertainty analysis
  - `Baseline model ASID.ipynb`: Alternative system identification methods
- Custom environment and C++ extension: `gym_oscillator` and `oscillator_cpp` are required for the RL-DBS environment.
- `Results/`: Generated experimental results, figures, and model weights
- Sample data generation scripts (RL-DBS datasets can be generated on-the-fly).

The work is 100% complete but open for extensions. It is usable by others for research in active learning and system identification—simply clone, install dependencies, and run the main scripts.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+ (CUDA optional for GPU)
- TensorFlow 1.x/2.x (for stable-baselines, if using PPO2)
- Dependencies: NumPy, SciPy, Matplotlib, Gym (for RL-DBS), stable-baselines3, stable-baselines
- `gym_oscillator` and the custom C++ extension `oscillator_cpp` must be installed and available in the Python path.

### Setup
Clone the repo:
```bash
git clone https://github.com/amirmmi/ACUMEN-Agentic-Sytem-ID.git
cd acumen
```

Install requirements:
```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib jupyter
pip install gym stable-baselines3
# Optional: stable-baselines for TensorFlow-based experiments
pip install stable-baselines tensorflow
```

Install custom environment and extension:
```bash
# Install gym_oscillator (if not already installed)
pip install git+https://github.com/dylov/rl-dbs.git 
```


## Usage

The main workflow is provided in the notebook `NeuralODE_CEM_MPC_Method.ipynb`. To reproduce results:

1. Ensure all dependencies and custom environments are installed.
2. Open the notebook in Jupyter or VS Code.
3. Run all cells to train the model, evaluate, and generate plots.

Additional notebooks provide specific analyses:
- `Baseline model.ipynb`: For baseline model training and comparison
- `Baseline model Epistemic Loss1.ipynb`: For epistemic uncertainty analysis
- `Baseline model ASID.ipynb`: For alternative system identification methods

### Results
- **Learning Curves:** Lower MSE with active data, up to 24.2% improvement for >2000 samples (see notebook plots).
- **Trajectory Predictions:** Better tracking and narrower uncertainty bands.
- **Uncertainty Summary:** Per-channel reductions, 22.5% overall MSE improvement.

## Key Results

### Statistical Validation
- **Statistically significant improvements**: Paired t-test p = 0.017, demonstrating robust superiority over random data collection
- **Consistent performance**: CEM-MPC outperformed random sampling across all tested sample sizes (Wilcoxon signed-rank test: p = 0.031)
- **Quantified improvements**: 15.7% ± 8.8% average MSE reduction with up to 24.2% improvement at optimal sample sizes
- **Bootstrap validation**: 95% confidence interval [0.015, 0.045] for mean difference, excluding zero

### Performance Metrics
The following figure summarizes the comprehensive comparison between the neural ODE CEM-MPC approach and random exploration:

![Neural ODE CEM-MPC vs Random Combined](neural_ode_cem_mpc_vs_random_combined.png)

### Uncertainty Calibration
- **Improved model reliability**: CEM-MPC training yields better-calibrated uncertainty estimates
- **Reduced predictive uncertainty**: Tighter uncertainty bands with maintained accuracy
- **Enhanced exploration efficiency**: Targeted data collection in underrepresented state space regions

## Challenges and Learnings
### Challenges
- Balancing exploration and safety in CEM-MPC to avoid aggressive actions in healthcare simulations—addressed with penalties and constraints.
- Handling irregular, noisy, heterogeneous time-series—solved using Neural ODEs' continuous-time nature and adaptive scaling.
- Computational cost of ensemble rollouts—optimized with batched ODE solves and efficient sampling.
- Ensuring reproducibility in stochastic environments like RL-DBS—used fixed seeds and detailed logging.

### Learnings
- Ensembles provide robust, calibrated uncertainty without complex Bayesian methods.
- Active learning reduces sample complexity significantly in data-scarce domains.
- Optimism and novelty enhance exploration diversity and coverage.
- Importance of modular design for research code to facilitate extensions and collaborations.

This project enhanced understanding of Neural ODEs, MPC, and active learning in healthcare AI.

## Future Work
- **Real-world validation**: Extension to real physiological datasets (glucose-insulin dynamics, pharmacokinetics)
- **Closed-loop integration**: Combination with reinforcement learning for adaptive therapy systems
- **Safety enhancements**: Advanced constraint handling for clinical deployment
- **Multi-modal support**: Extension to heterogeneous data types and sensor modalities
- **Scalability improvements**: Optimization for larger state spaces and longer horizons


We welcome contributions, feedback, and applications to new healthcare domains!
