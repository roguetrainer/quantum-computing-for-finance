# Quantum Computing for Finance: Practical Examples

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.33+-orange.svg)](https://pennylane.ai/)

Complete, executable examples demonstrating quantum computing applications in finance using PennyLane.

## Overview

This repository contains four comprehensive case studies showing how quantum computing can be applied to financial problems:

1. **Option Pricing** (Simulation) - Quantum Monte Carlo with Amplitude Estimation
2. **Portfolio Optimization** (Optimization) - QAOA for asset allocation
3. **Credit Classification** (Machine Learning) - Quantum Neural Networks
4. **Correlation Analysis** (Quantum-Inspired) - Tensor Networks

## Key Features

- Complete working code - All examples run out of the box
- Rich visualizations - Publication-quality plots
- Detailed explanations - Both in notebooks and comments
- Classical baselines - Compare quantum vs classical approaches
- Educational focus - Clear learning objectives for each example
- Scalable - Examples work on simulators and real quantum hardware

## Quick Start

### Installation

Clone the repository:
    git clone https://github.com/yourusername/quantum-finance-examples.git
    cd quantum-finance-examples

Create virtual environment (recommended):
    python -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate

Install dependencies:
    pip install -r requirements.txt

Or using conda:
    conda env create -f environment.yml
    conda activate quantum-finance

### Run Examples

Option 1: Python Scripts
    cd 1_option_pricing
    python option_pricing.py

Option 2: Jupyter Notebooks
    jupyter notebook
    # Navigate to desired example folder and open .ipynb file

Option 3: Google Colab
- Upload notebooks to Google Colab
- Install requirements in first cell
- Run all cells

## Repository Structure
```
    quantum-finance-examples/
    |
    +-- 1_option_pricing/
    |   +-- option_pricing.py
    |   +-- option_pricing.ipynb
    |   +-- README.md
    |
    +-- 2_portfolio_optimization/
    |   +-- portfolio_qaoa.py
    |   +-- portfolio_qaoa.ipynb
    |   +-- README.md
    |
    +-- 3_credit_classification/
    |   +-- credit_qml.py
    |   +-- credit_qml.ipynb
    |   +-- README.md
    |
    +-- 4_tensor_networks/
    |   +-- tensor_network_correlation.py
    |   +-- tensor_network_correlation.ipynb
    |   +-- README.md
    |
    +-- utils/
    |   +-- plotting.py
    |   +-- data_generation.py
    |
    +-- results/
    |
    +-- requirements.txt
    +-- environment.yml
    +-- README.md
```

## Examples Overview

### Example 1: Option Pricing with Quantum Monte Carlo

**Financial Problem**: Price European call options faster than classical Monte Carlo

**Quantum Advantage**: Quadratic speedup - O(1/√N) → O(1/N) convergence

**Key Concepts**:
- Quantum Amplitude Estimation (QAE)
- State preparation for probability distributions
- Payoff function implementation

**Files**:
- 1_option_pricing/option_pricing.ipynb - Interactive tutorial
- 1_option_pricing/option_pricing.py - Standalone script

**Runtime**: ~5 minutes (simulator)

**Prerequisites**: Basic options knowledge, Monte Carlo familiarity

### Example 2: Portfolio Optimization with QAOA

**Financial Problem**: Select optimal asset allocation under constraints

**Quantum Advantage**: Heuristic approach for NP-hard combinatorial optimization

**Key Concepts**:
- Quantum Approximate Optimization Algorithm (QAOA)
- Hamiltonian formulation of portfolio problem
- Variational quantum algorithms
- Classical-quantum hybrid optimization

**Files**:
- 2_portfolio_optimization/portfolio_qaoa.ipynb - Interactive tutorial
- 2_portfolio_optimization/portfolio_qaoa.py - Standalone script

**Runtime**: ~10 minutes (simulator)

**Prerequisites**: Basic portfolio theory, optimization concepts

### Example 3: Credit Classification with Quantum ML

**Financial Problem**: Classify loan applicants as creditworthy/risky

**Quantum Advantage**: Potential advantage through quantum kernels

**Key Concepts**:
- Variational Quantum Circuits (VQC)
- Quantum feature maps
- Quantum kernels
- Hybrid quantum-classical training

**Files**:
- 3_credit_classification/credit_qml.ipynb - Interactive tutorial
- 3_credit_classification/credit_qml.py - Standalone script

**Runtime**: ~15 minutes (simulator)

**Prerequisites**: Machine learning basics, classification concepts

### Example 4: Correlation Analysis with Tensor Networks

**Financial Problem**: Analyze 100+ asset correlations efficiently

**Quantum Advantage**: Linear vs cubic scaling (quantum-inspired, runs on classical hardware!)

**Key Concepts**:
- Matrix Product States (MPS)
- Tensor network decomposition
- DMRG algorithm
- Quantum-inspired computing

**Files**:
- 4_tensor_networks/tensor_network_correlation.ipynb - Interactive tutorial
- 4_tensor_networks/tensor_network_correlation.py - Standalone script

**Runtime**: ~8 minutes (classical hardware)

**Prerequisites**: Linear algebra, correlation/covariance matrices

## Learning Path

**Beginner** (New to quantum computing):
1. Start with Example 4 (Tensor Networks) - no quantum hardware needed
2. Then Example 1 (Option Pricing) - clearest quantum advantage
3. Example 3 (Credit Classification) - bridges to machine learning
4. Finally Example 2 (Portfolio Optimization) - most complex

**Intermediate** (Some quantum background):
1. Example 1 → Example 2 → Example 3 → Example 4

**Advanced** (Want to extend/modify):
- Each example has exercises at the end
- See CONTRIBUTING.md for guidelines

## Requirements

### Core Dependencies
- Python 3.8+
- PennyLane >= 0.33.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- pandas >= 1.3.0

### Optional (for specific examples)
- TensorNetwork (Example 4)
- scikit-learn (Example 3)
- qiskit (for IBM hardware)
- amazon-braket-sdk (for AWS hardware)

### Hardware
- **CPU**: Any modern processor (examples optimized for simulators)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional (can speed up Example 4)

## Running on Real Quantum Hardware

All examples can be run on real quantum computers via cloud platforms:

### IBM Quantum

    import pennylane as qml
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
    backend = service.backend("ibmq_manila")
    dev = qml.device('qiskit.ibmq', wires=4, backend=backend)

### Amazon Braket

    import pennylane as qml
    dev = qml.device('braket.aws.qubit', 
                     device_arn='arn:aws:braket:us-east-1::device/qpu/ionq/Harmony', 
                     wires=4)

### Azure Quantum

    import pennylane as qml
    dev = qml.device('azure.ionq', 
                     subscription_id="YOUR_ID", 
                     resource_group="YOUR_GROUP", 
                     wires=4)

**Note**: Real hardware execution requires account setup with quantum cloud provider, API tokens/credentials, and understanding of job queuing and costs. See individual example READMEs for hardware-specific instructions.

## Performance Benchmarks

| Example | Simulator Time | Hardware Time | Speedup | Accuracy |
|---------|---------------|---------------|---------|----------|
| Option Pricing | 2-5 min | 10-30 min | ~100x (theoretical) | 99%+ |
| Portfolio Opt | 5-10 min | 20-60 min | Heuristic | 95%+ |
| Credit Class | 10-15 min | 30-90 min | TBD | 90%+ |
| Tensor Network | 5-8 min | N/A (classical) | 10-50x | 99%+ |

Hardware times include queuing. Speedup is problem-size dependent.

## Educational Resources

### Background Reading
- PennyLane Documentation: https://pennylane.ai/qml/
- Quantum Computing for Finance Survey: https://arxiv.org/abs/2201.02773
- Nielsen & Chuang - Quantum Computation and Quantum Information

### Related Papers
1. **Option Pricing**: Stamatopoulos et al. (2020) - "Option Pricing using Quantum Computers"
2. **Portfolio Optimization**: Venturelli & Kondratyev (2019) - "Reverse Quantum Annealing"
3. **Quantum ML**: Havlíček et al. (2019) - "Supervised Learning with Quantum-Enhanced Feature Spaces"
4. **Tensor Networks**: Huggins et al. (2019) - "Towards Quantum Machine Learning with Tensor Networks"

### Video Tutorials
- PennyLane YouTube Channel: https://www.youtube.com/c/PennyLaneAI
- IBM Quantum Learning: https://learning.quantum.ibm.com/

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

**Ways to contribute**:
- Report bugs or issues
- Suggest new financial examples
- Improve documentation
- Add features or optimizations
- Create better visualizations

## Citation

If you use these examples in your research or teaching, please cite:

    @software{quantum_finance_examples,
      author = {Ian Buckley},
      title = {Quantum Computing for Finance: Practical Examples},
      year = {2025},
      url = {https://github.com/roguetrainer/quantum-finance-examples}
    }

## License

MIT License - see LICENSE file for details

## Acknowledgments

- PennyLane team for excellent quantum ML framework
- Xanadu for quantum computing tools
- Financial examples inspired by industry partnerships
- Survey paper: Herman et al. (2022) "A Survey of Quantum Computing for Finance"

## Contact

- **Issues**: Open a GitHub issue
- **Questions**: Start a GitHub discussion

## Related Projects

- PennyLane: https://github.com/PennyLaneAI/pennylane
- Qiskit Finance: https://github.com/Qiskit/qiskit-finance
- TensorNetwork: https://github.com/google/TensorNetwork

---

**If you find these examples helpful, please star the repository!**

---

Last updated: January 2025