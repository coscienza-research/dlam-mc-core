# DLAM: Dreaming L-directional Associative Memory

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neunet.2026.109729-0077c2.svg)](https://doi.org/10.1016/j.neunet.2026.109729)
[![Journal](https://img.shields.io/badge/Journal-Neural%20Networks%20(2026)-ff6f00.svg)](https://www.sciencedirect.com/science/article/pii/S0893608026011846)
[![arXiv](https://img.shields.io/badge/arXiv-2605.13721-b31b1b.svg)](https://arxiv.org/abs/2605.13721)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](tests)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!IMPORTANT]
> ### Official Reference Implementation
> This repository provides the official implementation, Monte Carlo simulation engine, and symbolic theory routines for the research paper:
>
> **Do Hopfield networks dream of stored patterns? A statistical–mechanical theory of dreaming in multidirectional associative memories**  
> Adriano Barra, Fabrizio Durante, Andrea Ladiana, Michela Marra Solazzo  
> *Neural Networks*, Volume 206, Article 109729, 2026.  
> **Published Article:** [DOI: 10.1016/j.neunet.2026.109729](https://doi.org/10.1016/j.neunet.2026.109729) | **Preprint:** [arXiv:2605.13721](https://arxiv.org/abs/2605.13721)

---

## Overview

The **Dreaming $L$-directional Associative Memory (DLAM)** integrates off-line unlearning ("dreaming") and supervised hetero-associative coupling across layers within a unified energy-based model (EBM). 

This repository contains two main scientific components:

1. **Python Monte Carlo Simulation Core (`dlam_mc_core`)**: A clean, publication-grade, fully reproducible Monte Carlo engine implementing synchronous Glauber dynamics, dreaming kernels, and trajectory tracking for $L=3$ layers (including pattern retrieval and *pattern disentanglement*).
2. **Symbolic Theory Package (`LAM_symbolic_arbitraryL.wl`)**: A Mathematica / Wolfram Language package (`LAMsym```) that derives and validates the replica-symmetric (RS) free-energy landscape, kernels, determinants, gradients, and order parameters for an **arbitrary number of layers $L$**.

---

## Scientific Scope

The 3-layer DLAM simulator implements:
- Noisy supervised data generation per layer with dataset entropy $\rho_l = (1-r_l^2)/(M_l r_l^2)$,
- Regularized dreaming kernel $A_l(t_l) = (1+t_l)(I + t_l C_l)^{-1}$,
- Hetero-associative inter-layer couplings $g_{lm}$,
- Synchronous Glauber dynamics at inverse temperature $\beta$,
- Trajectory-level magnetization tracking across all layers.

### Key Features

- **Strict reproducibility**: Explicit random seed control and full per-run provenance metadata (UTC timestamp, Python/NumPy versions, parameter SHA-256 digest, optional git commit SHA).
- **Typed dataclass API**: Parameter schemas (`SimParams`), trajectory outputs (`SimulationResult`), and batch helpers (`run_multi_seed`).
- **Clean modular design**: Clear separation between generation, kernel solves, local field updates, and simulation orchestration.
- **Pure NumPy**: Fast, portable execution without required heavy binary dependencies.
- **CLI & Serialization**: Direct CLI entrypoint and portable `.npz` result export.
- **Tested & Linted**: High test coverage (>94%) with Ruff linting and formatting.

---

## Installation

Install in editable mode:

```bash
pip install -e .
```

For development, testing, and linting tools:

```bash
pip install -e .[dev]
```

---

## Quick Start (Python)

### Single Simulation Run

```python
from dlam_mc_core import SimParams, run_simulation

# Configure a 3-layer DLAM in disentanglement mode
params = SimParams(
    N1=300,
    N2=300,
    N3=300,
    K=30,
    steps=200,
    beta=1.5,
    seed=7,
    init_mode="disentangle",
)

result = run_simulation(params)
print(f"Final magnetizations: m1={result.m1[-1]:.4f}, m2={result.m2[-1]:.4f}, m3={result.m3[-1]:.4f}")
```

### Multi-Seed Batch Execution

```python
from dlam_mc_core import SimParams, run_multi_seed

base = SimParams(steps=100, init_mode="disentangle")
results = run_multi_seed(base, seeds=[0, 1, 2, 3], dtype="float32")
print([r.m1[-1] for r in results])
```

---

## Command Line Usage

Run directly from the terminal:

```bash
python -m dlam_mc_core --steps 200 --seed 7 --beta 1.5 --dtype float32
```

Export trajectories, parameters, and reproducibility metadata to an `.npz` archive:

```bash
python -m dlam_mc_core --steps 200 --seed 7 --output outputs/run_seed7.npz
```

---

## Symbolic Theory for Arbitrary $L$ (Mathematica)

The repository includes [`LAM_symbolic_arbitraryL.wl`](LAM_symbolic_arbitraryL.wl), a standalone Wolfram Language / Mathematica package (`LAMsym```) that implements the analytical replica-symmetric (RS) statistical-mechanical theory for any number of layers $L$.

### Available Routines

| Function | Description |
| :--- | :--- |
| `LAMKernel[gMat, Hvec, beta]` | Constructs the $L \times L$ RS LAM kernel matrix $C(H)$. |
| `LAMDet[gMat, Hvec, beta]` | Computes the symbolic determinant $\det C(H)$. |
| `LAMLogDet[gMat, Hvec, beta]` | Computes $\log \det C(H)$. |
| `LAMLogDetGrad[gMat, Hsym, beta]` | Calculates the analytical gradient $\partial_{H_l} \log \det C(H)$ using Jacobi's formula $\mathrm{Tr}(C^{-1} \partial C)$. |
| `LAMDquad[gMat, Hsym, beta]` | Computes the RS quadratic correction $D_{\text{LAM}}(H)$. |
| `LAMDquadNum[gMat, Hsym, beta]` | Returns the polynomial numerator of the quadratic correction. |
| `LAMPiBar[gMat, Hsym, beta]` | Derives the conjugate LAM overlaps $\bar{\Pi}_l$. |
| `LAMFreeEnergyBlock[gMat, Hsym, beta, alpha]` | Computes the RS LAM free-energy block $A_{\text{LAM}}$. |
| `LAMPositivityCondition[gMat, Hvec, beta]` | Evaluates the eigenvalues of $C(H)$ to verify Gaussian integration validity. |
| `LAMHomogeneousSymbol[L, g, beta, H]` | Builds the closed-form symbolic kernel for identical couplings $g_{lm} = g$ and overlaps $H_l = H$. |

The package includes automated symbolic checks verifying the analytical formulas of the paper for $L=3$ (both homogeneous and heterogeneous couplings) and arbitrary $L \ge 2$.

To run the validations using the Wolfram Engine / Mathematica kernel:

```bash
wolframscript -file LAM_symbolic_arbitraryL.wl
```

---

## Reproducibility & Provenance

Each `SimulationResult` object carries complete provenance metadata:
- UTC run timestamp,
- Python and NumPy environment versions,
- Platform and architecture descriptors,
- Compute dtype (`float32` or `float64`),
- Random seed,
- SHA-256 digest computed across all simulation parameters,
- Git commit SHA (`GITHUB_SHA` if available).

This metadata is embedded whenever saving with `save_result_npz(...)`.

---

## Project Structure

```text
dlam-mc-core/
├── LAM_symbolic_arbitraryL.wl  # Mathematica package for arbitrary-L RS theory
├── src/dlam_mc_core/           # Python Monte Carlo simulation engine
│   ├── __init__.py             # Public API exports
│   ├── __main__.py             # CLI entrypoint
│   ├── types.py                # Dataclasses (SimParams, SimulationResult, etc.)
│   ├── random.py               # Seed management and cue initializations
│   ├── generation.py           # Layer archetype and noisy example generation
│   ├── kernels.py              # Dreaming kernel linear solver
│   ├── dynamics.py             # Local fields and synchronous Glauber updates
│   ├── simulation.py           # Simulation loops & multi-seed batch runners
│   └── io.py                   # NPZ serialization & metadata helpers
├── tests/                      # Unit & regression test suite (>94% coverage)
├── docs/                       # Theoretical and algorithmic documentation
├── examples/                   # Executable scripts and usage examples
├── CITATION.cff                # Citation File Format (enables GitHub Cite button)
└── pyproject.toml              # Build metadata and tool configuration
```

---

## Verification & Testing

Run the test suite:

```bash
pytest
```

Run code formatting and lint checks:

```bash
ruff check .
ruff format --check .
```

---

## Citation

If you use this software, the DLAM architecture, or the theoretical derivations in your research, please cite our article:

```bibtex
@article{barra2026hopfield,
  title={Do Hopfield networks dream of stored patterns? A statistical-mechanical theory of dreaming in multidirectional associative memories},
  author={Barra, Adriano and Durante, Fabrizio and Ladiana, Andrea and Solazzo, Michela Marra},
  journal={Neural Networks},
  volume={206},
  pages={109729},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.neunet.2026.109729},
  url={https://doi.org/10.1016/j.neunet.2026.109729}
}
```

Preprint reference:

```bibtex
@article{barra2026dreaming_arxiv,
  title={Do Hopfield Networks Dream of Stored Patterns? A Statistical-Mechanical Theory of Dreaming in Multidirectional Associative Memories},
  author={Barra, Adriano and Durante, Fabrizio and Ladiana, Andrea and Solazzo, Michela Marra},
  journal={arXiv preprint arXiv:2605.13721},
  year={2026},
  url={https://arxiv.org/abs/2605.13721}
}
```

You can also use GitHub's **"Cite this repository"** button in the sidebar (powered by [`CITATION.cff`](CITATION.cff)).

---

## License

MIT License — Copyright (c) 2026 Andrea Ladiana, Scuderia "Cò'Scienza" (Sapienza University of Rome). See [`LICENSE`](LICENSE) for details.
