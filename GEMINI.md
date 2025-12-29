# T1DSim_AI

## Project Overview

**T1DSim_AI** is a physiologically-constrained Neural Network Digital Twin framework designed to replicate glucose dynamics in Type 1 Diabetes (T1D). It allows for the creation of individualized in-silico models for pre-clinical testing of new technologies and treatment strategies.

The framework uses a novel Neural Network state-space model architecture that ensures observability and interpretability, adhering to known glucose-insulin dynamics.

### Key Technologies

*   **Language:** Python (>= 3.9)
*   **Core Libraries:**
    *   `torch` (PyTorch) - Neural network modeling
    *   `numpy` & `pandas` - Data manipulation
    *   `scikit-learn` - Preprocessing
    *   `matplotlib` - Visualization
*   **Build System:** `hatchling` (configured via `pyproject.toml`)

## Building and Running

### Installation

To install the package for usage:

```bash
pip install t1dsim-ai
```

### Running Examples

The `example/` directory contains scripts to demonstrate the framework's capabilities.

**Note:** The example scripts may contain hardcoded paths or require specific working directories. Check the scripts before running.

1.  **Simulation:** Simulates glucose dynamics for a digital twin.
    ```bash
    python example/runDigitalTwin.py
    ```
    *Generates plots in `example/img/`.*

2.  **Training:** Trains a new digital twin model from data.
    ```bash
    python example/trainDigitalTwin.py
    ```
    *Uses data from `example/example_model/data_example.csv`.*

## Development Conventions

### Setup for Contributors

1.  **Clone and Install in Editable Mode:**
    ```bash
    git clone https://github.com/mosqueralopez/T1DSim_AI.git
    cd T1DSim_AI
    pip install -e ".[dev]"
    ```
    This installs the package in editable mode along with development dependencies (`pytest`, `mypy`, `pre-commit`, `ruff`).

2.  **Pre-commit Hooks:**
    Set up pre-commit hooks to ensure code quality (formatting, linting) before committing.
    ```bash
    pre-commit install
    ```
    Run manually:
    ```bash
    pre-commit run --all-files
    ```

### Testing

Run the test suite using `pytest`:

```bash
python -m pytest tests/
```

### Code Style

*   **Linting/Formatting:** The project uses `ruff` (implied by dev dependencies) and `pre-commit` to enforce style.
*   **Type Checking:** `mypy` is used for static type checking.

## Directory Structure

*   `src/t1dsim_ai/`: Main source code package.
    *   `individual_model.py`: Digital Twin implementation.
    *   `population_model.py`: Population-level model logic.
    *   `models/`: Pre-trained models and scalers.
    *   `utils/`: Utility functions for metrics and preprocessing.
*   `example/`: Example scripts (`runDigitalTwin.py`, `trainDigitalTwin.py`) and sample data.
*   `tests/`: Unit tests for the package.
*   `pyproject.toml`: Project configuration and dependencies.
