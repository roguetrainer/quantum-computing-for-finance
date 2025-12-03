# Example 1: Option Pricing with Quantum Monte Carlo

## Overview

This example demonstrates how quantum amplitude estimation can provide a **quadratic speedup** for pricing financial derivatives using Monte Carlo methods.

## Financial Problem

**Objective**: Price a European call option

**Classical Approach**: Monte Carlo simulation with N samples
- Convergence rate: O(1/√N)
- For 1% accuracy: Need ~10,000 samples

**Quantum Approach**: Quantum Amplitude Estimation (QAE)
- Convergence rate: O(1/N)
- For 1% accuracy: Need ~100 quantum queries
- **100x fewer evaluations required!**

## Background

### Black-Scholes Model

Stock price follows geometric Brownian motion:

    dS = μS dt + σS dW

Where:
- S = stock price
- μ = drift (risk-free rate in risk-neutral measure)
- σ = volatility
- W = Wiener process

### European Call Option

**Payoff at maturity T**:

    C = max(S(T) - K, 0)

Where:
- S(T) = stock price at maturity
- K = strike price

**Fair price today**:

    V(0) = e^(-rT) * E[C]

### Classical Monte Carlo

Pseudocode:

    for i in range(N):
        Z_i = random_normal()
        ST_i = S0 * exp((r - 0.5*σ²)*T + σ*sqrt(T)*Z_i)
        payoffs[i] = max(ST_i - K, 0)
    
    option_price = exp(-r*T) * mean(payoffs)
    error = std(payoffs) / sqrt(N)

**Complexity**: O(N) samples for error ε = O(1/√N)

### Quantum Amplitude Estimation

**Key Idea**: Encode payoff probability in quantum amplitude

    |ψ⟩ = √a |good⟩ + √(1-a) |bad⟩

Where:
- a = normalized expected payoff
- |good⟩ = states with positive payoff
- |bad⟩ = states with zero payoff

**QAE estimates a using**:
- Quantum Phase Estimation (QPE)
- Grover-like amplitude amplification
- O(1/ε) quantum queries for error ε

**Complexity**: O(1/ε) queries for error ε

## Implementation Details

### Quantum Circuit Components

**1. State Preparation (Operator A):**
- Load log-normal distribution into quantum state
- Uses controlled rotations based on cumulative distribution
- Encodes possible stock prices S(T) in computational basis
- Implementation: Grover-Rudolph algorithm or variants

**2. Payoff Oracle (Operator O):**
- Compares S(T) with strike price K using quantum comparator
- Marks states where S(T) > K (in-the-money)
- Rotates ancilla qubit proportional to payoff value
- Angle: θ = arcsin(√[(S(T)-K)/S(T)])

**3. Amplitude Amplification:**
- Applies Grover operator: G = -A S₀ A† Sχ
- S₀ reflects about |0⟩ state
- Sχ reflects about marked states (payoff > 0)
- Repeated m times to amplify amplitude

**4. Phase Estimation:**
- Estimates phase θ where G|ψ⟩ = e^(iθ)|ψ⟩
- Uses controlled-G operations with evaluation qubits
- Inverse QFT extracts phase
- sin²(θ/2) gives probability a

### Circuit Depth

For n price qubits and precision ε:
- State preparation: O(2^n) gates (exponential bottleneck!)
- Payoff oracle: O(n) gates (quantum comparator)
- QAE iterations: O(1/ε)
- QPE: O(log(1/ε))
- **Total depth**: O(2^n + n/ε)

**For 4 qubits, 1% error**: ~500-1000 gates

### NISQ Considerations

**Challenges:**
- State preparation is exponentially expensive in general
- Need efficient methods for specific distributions
- Circuit depth too large for current hardware
- Noise accumulates over many gates

**Workarounds:**
- Use amplitude encoding only for small n (4-6 qubits)
- Approximate distributions with low-depth circuits
- Hybrid classical-quantum approaches
- Error mitigation techniques

## Running the Example

### Quick Start

    cd 1_option_pricing
    python option_pricing.py

### Jupyter Notebook

    jupyter notebook option_pricing.ipynb

### Command Line Options

Edit parameters at top of script:

    # Option parameters
    S0 = 100      # Initial stock price ($)
    K = 100       # Strike price ($)
    r = 0.05      # Risk-free rate (5% annual)
    sigma = 0.20  # Volatility (20% annual)
    T = 1.0       # Time to maturity (years)
    
    # Quantum parameters
    n_qubits = 4      # Price discretization qubits
    n_eval_qubits = 4 # QAE evaluation qubits
    
    # Classical comparison
    mc_samples = [100, 1000, 10000, 100000]

## Expected Output

### Console Output

    ========================================
    QUANTUM OPTION PRICING EXAMPLE
    ========================================
    
    Option Parameters:
      Initial Price (S0): $100.00
      Strike Price (K): $100.00
      Risk-free Rate (r): 5.00%
      Volatility (σ): 20.00%
      Time to Maturity (T): 1.0 years
    
    Black-Scholes Exact Price: $10.45
    
    ----------------------------------------
    Classical Monte Carlo Results
    ----------------------------------------
      Samples |  Price  |  Error  |  Time
         100  | $10.23  | ±$1.45 | 0.001s
       1,000  | $10.38  | ±$0.46 | 0.010s
      10,000  | $10.44  | ±$0.15 | 0.100s
     100,000  | $10.45  | ±$0.05 | 1.000s
    
    ----------------------------------------
    Quantum Amplitude Estimation Results
    ----------------------------------------
     Eval Qubits |  Price  |  Error  |  Time
          1      |  $9.87  | ±$0.58 | 0.150s
          2      | $10.21  | ±$0.24 | 0.300s
          3      | $10.42  | ±$0.03 | 0.600s
          4      | $10.45  | ±$0.00 | 1.200s
    
    ----------------------------------------
    Comparison Summary
    ----------------------------------------
    For 1% accuracy:
      Classical MC: 10,000 samples
      Quantum QAE:  100 queries (4 eval qubits)
      Speedup: ~100x in number of evaluations
    
    Note: Quantum advantage assumes:
      - Efficient state preparation (not yet available)
      - Fault-tolerant quantum computer
      - Current estimate is proof-of-concept

### Generated Plots

The script creates `quantum_option_pricing.png` with 4 subplots:

1. **Convergence Comparison**
   - X-axis: Number of samples/queries (log scale)
   - Y-axis: Absolute error from Black-Scholes
   - Blue line: Classical MC (slope -1/2)
   - Orange line: Quantum QAE (slope -1)
   - Shows quadratic speedup visually

2. **Price Estimates**
   - Bar chart comparing all methods
   - Horizontal line: Black-Scholes exact
   - Error bars for MC methods
   - Color coding: Classical vs Quantum

3. **Time vs Accuracy Trade-off**
   - X-axis: Computation time
   - Y-axis: Accuracy achieved
   - Shows practical performance
   - Classical vs Quantum comparison

4. **Quantum State Visualization**
   - Probability distribution of |ψ⟩
   - Shows encoded stock prices
   - Highlights in-the-money states
   - Amplitude corresponding to option value

## Exercises

### Beginner Exercises

**Exercise 1.1: Out-of-the-Money Option**
- Change strike price: K = 110
- Run both classical and quantum
- **Question**: How does this affect convergence? Why?
- **Expected**: Lower payoff probability, harder to estimate

**Exercise 1.2: Volatility Impact**
- Try σ = 0.10 (low volatility)
- Try σ = 0.40 (high volatility)
- **Question**: Which converges faster? Why?
- **Hint**: Think about payoff distribution

**Exercise 1.3: Time to Maturity**
- Test T = 0.5 years
- Test T = 2.0 years
- **Question**: How does T affect the option price?
- **Bonus**: Plot price vs T (option theta)

### Intermediate Exercises

**Exercise 2.1: European Put Option**
- Modify payoff: max(K - S(T), 0)
- Adjust quantum oracle accordingly
- **Deliverable**: Working put option pricer
- **Hint**: Compare put-call parity

**Exercise 2.2: Binary (Digital) Option**
- Payoff: $1 if S(T) > K, else $0
- Simpler oracle (no payoff calculation)
- **Question**: Does convergence improve? Why?
- **Insight**: Binary payoff → simpler amplitude

**Exercise 2.3: Basket Option**
- Add second asset with correlation ρ
- Payoff: max(w₁S₁ + w₂S₂ - K, 0)
- **Challenge**: Multi-dimensional state preparation
- **Required qubits**: 2n for two assets

### Advanced Exercises

**Exercise 3.1: Asian Option**
- Path-dependent: Payoff depends on average price
- Need to encode multiple time steps
- **Complexity**: How does circuit depth scale?
- **Bonus**: Compare with classical MC advantage

**Exercise 3.2: Barrier Option**
- Up-and-out: Payoff = 0 if S(t) > B at any time
- Requires path encoding
- **Implementation**: Check barrier at multiple time steps
- **Question**: When is quantum advantage larger?

**Exercise 3.3: Real Hardware**
- Run on IBM Quantum or AWS Braket
- Compare noisy results with simulator
- **Tasks**:
  - Implement error mitigation
  - Measure actual circuit fidelity
  - Estimate when real advantage occurs
- **Deliverable**: Hardware execution report

**Exercise 3.4: Custom Distribution**
- Replace log-normal with different distribution
- Examples: Student-t, mixed Gaussian
- **Challenge**: Efficient state preparation
- **Question**: Which distributions are QAE-friendly?

## Limitations & Considerations

### Theoretical Limitations

**1. State Preparation Bottleneck**
- Loading arbitrary distribution requires O(2^n) gates
- Only specific distributions (log-concave) have efficient methods
- This is the main barrier to quantum advantage
- **Solution**: Grover-Rudolph works for limited cases

**2. qRAM Requirement**
- Efficient classical data loading needs quantum RAM
- No physical qRAM exists yet
- Without qRAM, data loading negates speedup
- **Workaround**: Hybrid methods, small datasets

**3. Error Accumulation**
- Deep circuits on NISQ devices accumulate errors
- Need O(1/ε) circuit repetitions
- Each with O(n) depth
- **Reality**: Errors dominate on current hardware

### Practical Considerations

**When Quantum Helps:**
✅ High-dimensional problems (many assets)
✅ Path-dependent payoffs (Asian, lookback, barriers)
✅ Complex derivatives (multiple conditions)
✅ Cases where classical MC is slow

**When Classical is Better:**
❌ Simple options with closed-form solutions
❌ Low accuracy requirements (ε > 5%)
❌ Small problem sizes (n < 3 assets)
❌ When state preparation cost dominates

### Current Hardware Reality

**2025 Status:**
- Available qubits: 50-1000 (but noisy)
- Coherence time: ~100 μs
- Gate fidelity: 99% (single), 95% (two-qubit)
- Practical circuit depth: ~100 gates
- **Verdict**: Proof-of-concept only

**Path to Advantage:**
- Need: 1000+ logical qubits (error-corrected)
- Need: Efficient state preparation method
- Need: Fault-tolerant operations
- **Timeline**: 5-15 years for production use

## Performance Analysis

### Theoretical Complexity

| Method | Samples/Queries | Circuit Depth | Memory |
|--------|----------------|---------------|---------|
| Classical MC | O(1/ε²) | N/A | O(1) |
| Quantum QAE | O(1/ε) | O(n/ε) | O(n) qubits |

### Asymptotic Advantage

Quantum is better when:
- ε < 1% (high accuracy)
- N > 10,000 samples classically
- Payoff evaluation is expensive
- State preparation is efficient

### Practical Crossover

With ideal quantum computer:
- ε = 1%: Quantum wins (100x fewer queries)
- ε = 5%: Quantum wins (20x fewer queries)
- ε = 10%: Marginal benefit

With NISQ hardware:
- Errors dominate
- No advantage yet
- Useful for learning/research

### Resource Estimates (Fault-Tolerant)

For pricing to 1% accuracy:
- Logical qubits: 50-100
- T-gates: ~10,000
- Physical qubits (surface code): ~50,000
- Runtime: Minutes

**Comparison**: Classical MC on laptop: Seconds to minutes

## Troubleshooting

### Common Issues

**1. "Import Error: No module named pennylane"**
- Solution: Install requirements
- Run: pip install -r requirements.txt

**2. "Device not found"**
- Solution: Check device name
- Use: qml.device('default.qubit', wires=n)

**3. "Out of memory"**
- Problem: Too many qubits for classical simulator
- Solution: Reduce n_qubits to 4-6
- Alternative: Use lightning.qubit device

**4. "Results don't match Black-Scholes"**
- Check: Option parameters (S0, K, r, σ, T)
- Verify: Correct payoff function
- Note: Small n_qubits → discretization error

**5. "Very slow execution"**
- Reduce: n_eval_qubits
- Reduce: n_qubits
- Use: pennylane-lightning (faster backend)

### Performance Tips

- Start with n_qubits=3 or 4 (not 8!)
- Use pennylane-lightning for speed
- Vectorize classical MC code (NumPy)
- Run on GPU if available (for tensor networks)

## Further Reading

### Foundational Papers

**Quantum Monte Carlo:**
1. Montanaro (2015) - "Quantum speedup of Monte Carlo methods"
   - Original QAE speedup proof
   - arXiv:1504.06987

2. Brassard et al. (2002) - "Quantum Amplitude Amplification and Estimation"
   - QAE algorithm details
   - arXiv:quant-ph/0005055

**Finance Applications:**
3. Stamatopoulos et al. (2020) - "Option pricing using quantum computers"
   - Practical implementation
   - Quantum 4, 291

4. Rebentrost & Lloyd (2018) - "Quantum computational finance"
   - Overview of quantum finance
   - arXiv:1811.03975

### Tutorials & Resources

**PennyLane:**
- QML Demos: https://pennylane.ai/qml/demonstrations.html
- Quantum Finance: https://pennylane.ai/qml/demos/tutorial_finance.html

**Qiskit:**
- Qiskit Finance Tutorials: https://qiskit.org/ecosystem/finance/tutorials/

**Books:**
- "Quantum Computation and Quantum Information" (Nielsen & Chuang)
- "Options, Futures, and Other Derivatives" (Hull)

### Related Examples

- Qiskit Finance Tutorials
- PennyLane QML Demos
- Google Cirq Examples

## Citation

If you use this example in your research, please cite:

    @article{stamatopoulos2020option,
      title={Option pricing using quantum computers},
      author={Stamatopoulos, Nikitas and Egger, Daniel J and Sun, Yue and others},
      journal={Quantum},
      volume={4},
      pages={291},
      year={2020},
      publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den Quantenwissenschaften}
    }

And this repository:

    @software{quantum_finance_option_pricing,
      author = {Ian Buckley},
      title = {Quantum Option Pricing Example},
      year = {2025},
      url = {https://github.com/roguetrainer/quantum-finance-examples/tree/main/1_option_pricing}
    }

## Support

**Issues**: https://github.com/roguetrainer/quantum-finance-examples/issues
**Discussions**: https://github.com/roguetrainer/quantum-finance-examples/discussions


---

[← Back to Main README](../README.md) | [Next Example: Portfolio Optimization →](../2_portfolio_optimization/README.md)