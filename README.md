# Deep xVA Solver - PyTorch Implementation

This is a PyTorch conversion of the [Deep xVA Solver](https://arxiv.org/abs/2005.02633) originally implemented in TensorFlow 2.0.

## Overview

This repository implements deep learning methods for solving Backward Stochastic Differential Equations (BSDEs) with applications to financial derivatives pricing and XVA (X-Value Adjustment) calculations.

## Key Features

- **Full PyTorch conversion** from TensorFlow 2.x
- Supports GPU acceleration (CUDA)
- Implements BSDE solvers for:
  - Forward contracts
  - European call options
  - Basket options (multi-dimensional)
  - FVA (Funding Valuation Adjustment)
  - BCVA (Bilateral Credit Valuation Adjustment)

## Installation

### Requirements

```bash
pip install torch numpy scipy matplotlib pandas munch tqdm openpyxl
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.12.0 (with CUDA support optional)
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Pandas >= 1.3.0
- Munch >= 2.5.0
- tqdm >= 4.62.0
- openpyxl >= 3.0.0 (for Excel output)

## Quick Start

### Example 1: Forward Contract Pricing

```python
python forward.py
```

### Example 2: European Call Option

```python
python callOption.py
```

### Example 3: Basket Call Option (100 underlyings)

```python
python basketCall.py
```

### Example 4: FVA on Forward Contract

```python
python fvaForward.py
```

### Example 5: BCVA on Basket Call

```python
python basketCallWithCVA.py
```

## Architecture

The codebase is organized as follows:

- **equation.py**: Base class for defining PDEs and terminal conditions
- **xvaEquation.py**: Financial equation implementations (forwards, options, XVA)
- **solver.py**: Main BSDE solver with deep neural networks
- **RecursiveEquation.py**: Recursive equations for FVA and BCVA
- **XvaSolver.py**: Specialized solver for XVA computations
- **Example scripts**: Complete examples demonstrating usage

## Key Differences from TensorFlow Version

### 1. Framework-Specific Changes

| TensorFlow | PyTorch |
|------------|---------|
| `tf.keras.Model` | `nn.Module` |
| `tf.keras.layers.Dense` | `nn.Linear` |
| `tf.keras.layers.BatchNormalization` | `nn.BatchNorm1d` |
| `tf.Variable` | `nn.Parameter` |
| `tf.function` | Not needed (eager by default) |
| `tf.GradientTape` | Automatic differentiation |
| `tf.keras.backend.set_floatx()` | `torch.set_default_dtype()` |

### 2. Method Naming

- `f_tf()` → `f_torch()` (generator function)
- `g_tf()` → `g_torch()` (terminal condition)
- Training mode is handled via `.train()` and `.eval()` methods

### 3. Data Handling

- PyTorch uses **channels-first** convention by default, but this code maintains NumPy's structure
- Explicit device management (CPU/GPU) with `.to(device)`
- Dtype management is more explicit

### 4. Training Loop

The PyTorch version implements a manual learning rate schedule instead of TensorFlow's `PiecewiseConstantDecay`.

## GPU Support

The code automatically detects and uses CUDA if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

To force CPU usage:

```python
device = torch.device('cpu')
```

## Model Configuration

Example configuration:

```python
config = {
    "eqn_config": {
        "eqn_name": "CallOption",
        "total_time": 1.0,
        "dim": 1,
        "num_time_interval": 200,
        "strike": 100,
        "r": 0.01,
        "sigma": 0.25,
        "x_init": 100
    },
    "net_config": {
        "y_init_range": [9, 11],
        "num_hiddens": [21, 21],
        "lr_values": [5e-2, 5e-3],
        "lr_boundaries": [2000],
        "num_iterations": 4000,
        "batch_size": 64,
        "valid_size": 1024,
        "logging_frequency": 100,
        "dtype": "float64",
        "verbose": True
    }
}
```

## Output

Each example produces:

1. **Training history**: Loss and Y0 values during training
2. **Visualization**: PDF plots comparing deep solver vs exact solutions
3. **Excel files**: Exposure paths for further analysis

## Performance Considerations

- **Batch size**: Adjust based on GPU memory (default: 64)
- **Network depth**: More layers can improve accuracy but slow training
- **Precision**: Use `float64` for financial applications requiring high accuracy
- **GPU acceleration**: Can provide 5-10x speedup for larger problems

## Validation

The implementation has been validated against:

- Black-Scholes analytical solutions for European options
- Monte Carlo estimations for XVA calculations
- Original TensorFlow implementation results

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{gnoatto2020deep,
  title={Deep xVA solver--A neural network based counterparty credit risk management framework},
  author={Gnoatto, Alessandro and Picarelli, Athena and Reisinger, Christoph},
  journal={arXiv preprint arXiv:2005.02633},
  year={2020}
}
```

## Acknowledgements

- Original TensorFlow implementation by Alessandro Gnoatto, Athena Picarelli, and Christoph Reisinger
- Chang Jiang for TensorFlow 1.x to 2.x conversion
- PyTorch conversion by [Your Name/Organization]

## License

[Same as original repository]

## Support

For issues related to:
- **Original algorithm**: See the [original paper](https://arxiv.org/abs/2005.02633)
- **PyTorch implementation**: Open an issue in this repository
- **PyTorch framework**: See [PyTorch documentation](https://pytorch.org/docs/)

## Future Enhancements

Potential improvements:

- [ ] Distributed training support
- [ ] Mixed precision training (FP16)
- [ ] TorchScript compilation for deployment
- [ ] Additional optimization algorithms
- [ ] Model checkpointing and recovery
- [ ] Tensorboard integration for monitoring
