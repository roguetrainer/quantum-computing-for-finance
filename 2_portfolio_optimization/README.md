# Example 2: Portfolio Optimization with QAOA

## Overview

This example demonstrates how the Quantum Approximate Optimization Algorithm (QAOA) can solve portfolio optimization problems - finding the optimal allocation of assets to maximize returns while minimizing risk.

## Financial Problem

**Objective**: Select assets from a portfolio to maximize risk-adjusted returns subject to constraints

**Classical Approach**: 
- Brute force: O(2^n) for n assets - intractable for n>20
- Heuristics: Genetic algorithms, simulated annealing
- Convex relaxation: May not satisfy integer constraints

**Quantum Approach**: QAOA
- Variational quantum algorithm
- Heuristic but potentially better than classical heuristics
- Natural fit for NISQ hardware

## Background

### Portfolio Theory

**Mean-Variance Optimization** (Markowitz, 1952):

Maximize Sharpe ratio:

    Sharpe = (Expected Return - Risk-Free Rate) / Portfolio Risk

For discrete allocation (binary variables x_i ∈ {0,1}):

    Maximize: Σ r_i * x_i - q * Σ Σ x_i * Σ_ij * x_j
    Subject to: Σ x_i = K (budget constraint)

Where:
- r_i = expected return of asset i
- Σ_ij = covariance between assets i and j
- q = risk aversion parameter
- K = number of assets to select

### QUBO Formulation

Quadratic Unconstrained Binary Optimization:

    minimize: x^T Q x + b^T x

For portfolio:
- Q_ij = q * Σ_ij (risk term)
- b_i = -r_i (return term)
- Budget constraint: Add penalty λ(Σ x_i - K)²

### Ising Hamiltonian

Convert QUBO to quantum problem using x_i = (1-z_i)/2:

    H = Σ h_i Z_i + Σ J_ij Z_i Z_j

Where:
- Z_i = Pauli-Z operator on qubit i
- h_i = linear coefficients (returns)
- J_ij = quadratic coefficients (covariances)

## QAOA Algorithm

### Overview

QAOA approximates the ground state of H using:
1. Parameterized quantum circuit
2. Classical optimizer
3. Measurement and feedback loop

### Circuit Structure

**1. Initial State**: Uniform superposition

    |s⟩ = H^⊗n |0⟩ = (1/√2^n) Σ |x⟩

**2. QAOA Layers** (repeat p times):

Problem Hamiltonian (cost layer):

    U(C, γ) = exp(-iγH)

Mixing Hamiltonian (exploration layer):

    U(B, β) = exp(-iβ Σ X_i)

**3. Final State**:

    |γ, β⟩ = U(B,β_p) U(C,γ_p) ... U(B,β_1) U(C,γ_1) |s⟩

**4. Measurement**: Sample bitstrings, compute cost

### Parameter Optimization

Minimize expectation of H:

    min F(γ, β) = ⟨γ,β| H |γ,β⟩

Using classical optimizer:
- Gradient descent
- COBYLA (gradient-free)
- Simultaneous Perturbation Stochastic Approximation (SPSA)

### Performance

- p=1: Fast but low quality
- p→∞: Approaches optimal (adiabatic limit)
- Typical: p=2-5 for NISQ hardware

## Implementation Details

### Hamiltonian Construction

**Returns Term**:

    H_return = -Σ r_i Z_i

Maps return r_i to Pauli-Z on qubit i

**Risk Term**:

    H_risk = q * Σ_ij Σ_ij Z_i Z_j

Maps covariance to ZZ interactions

**Budget Constraint**:

    H_budget = λ (Σ (I - Z_i)/2 - K)²

Penalty for violating Σ x_i = K

### Circuit Implementation

For n=4 assets, p=2 layers:

**Layer structure**:
1. Initialize: H gates on all qubits
2. Cost layer 1: RZ(γ₁) and RZZ(γ₁) gates
3. Mixing layer 1: RX(β₁) gates
4. Cost layer 2: RZ(γ₂) and RZZ(γ₂) gates
5. Mixing layer 2: RX(β₂) gates
6. Measure all qubits

**Total gates**:
- Single-qubit: ~4n per layer → 16 for n=4, p=2
- Two-qubit: ~n² per layer → 32 for n=4, p=2
- **Total depth**: ~50-100 gates

### Optimization Strategy

**1. Initialize Parameters**:
- Random: γ, β ~ Uniform(0, 2π)
- Or use heuristic (e.g., linear ramp)

**2. Classical Loop**:

    for iteration in range(max_iter):
        # Run quantum circuit
        cost = quantum_expectation(γ, β)
        
        # Update parameters
        γ, β = optimizer.step(cost, γ, β)
        
        if converged:
            break

**3. Sample Solutions**:
- Run final circuit 1000+ times
- Get bitstring distribution
- Select best or most probable

### Constraint Handling

**Soft Constraints** (penalty method):
- Add penalty term to Hamiltonian
- Tune penalty weight λ
- **Pros**: Simple, works with QAOA
- **Cons**: May violate constraints slightly

**Hard Constraints** (custom mixer):
- Design mixer that preserves constraints
- Example: XY mixer for budget constraint
- **Pros**: Guarantees feasibility
- **Cons**: More complex circuit

## Running the Example

### Quick Start

    cd 2_portfolio_optimization
    python portfolio_qaoa.py

### Jupyter Notebook

    jupyter notebook portfolio_qaoa.ipynb

### Configuration

Edit parameters at top of script:

    # Portfolio parameters
    n_assets = 4
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    returns = np.array([0.12, 0.15, 0.10, 0.18])
    covariances = np.array([[...]])  # 4x4 matrix
    risk_aversion = 0.5
    budget = 2  # Select 2 assets
    
    # QAOA parameters
    p_layers = 2
    max_iterations = 100
    optimizer = 'COBYLA'

## Expected Output

### Console Output

    ========================================
    PORTFOLIO OPTIMIZATION WITH QAOA
    ========================================
    
    Portfolio Parameters:
      Assets: 4 (AAPL, GOOGL, MSFT, AMZN)
      Budget: Select 2 assets
      Risk Aversion: 0.5
    
    Expected Returns:
      AAPL: 12.0%
      GOOGL: 15.0%
      MSFT: 10.0%
      AMZN: 18.0%
    
    ----------------------------------------
    Classical Brute Force
    ----------------------------------------
    Enumerating all 6 combinations...
    
    Optimal Portfolio: [GOOGL, AMZN]
    Expected Return: 16.5%
    Portfolio Risk: 0.0625
    Objective Value: 0.1338
    Time: 0.003s
    
    ----------------------------------------
    QAOA Optimization (p=2)
    ----------------------------------------
    Iteration   Cost      Parameters
        0     -0.0523    γ=[0.85, 1.23], β=[0.34, 0.91]
       10     -0.0892    γ=[1.12, 1.45], β=[0.67, 1.03]
       20     -0.1156    γ=[1.34, 1.67], β=[0.89, 1.21]
       30     -0.1289    γ=[1.45, 1.78], β=[0.98, 1.34]
       40     -0.1325    γ=[1.51, 1.83], β=[1.03, 1.39]
       50     -0.1337    γ=[1.53, 1.85], β=[1.05, 1.42]
    
    Converged after 52 iterations
    Optimization Time: 8.45s
    
    ----------------------------------------
    QAOA Sampling (1000 shots)
    ----------------------------------------
    Bitstring | Portfolio | Objective | Frequency
     0110     | [GOOGL, MSFT] | 0.1156 |  34.2%
     1001     | [GOOGL, AMZN] | 0.1338 |  31.5%
     1010     | [GOOGL, MSFT] | 0.1156 |  12.8%
     0101     | [AAPL, AMZN]  | 0.1289 |  11.3%
     Others   |  ...          |  ...   |  10.2%
    
    Best QAOA Solution: [GOOGL, AMZN]
    Matches Classical Optimal: YES ✓
    
    ----------------------------------------
    Summary
    ----------------------------------------
    Classical: 0.003s, Optimal found
    QAOA: 8.45s, Optimal found (31.5% probability)
    
    Note: For n=4, classical is faster
          Quantum advantage expected for n>20

### Generated Plots

The script creates `portfolio_qaoa_results.png` with 4 subplots:

**1. QAOA Convergence**
- X-axis: Iteration
- Y-axis: Cost (negative objective)
- Shows optimization trajectory
- Marks convergence point

**2. Solution Distribution**
- Bar chart of top 10 bitstrings
- Height = sampling frequency
- Color = objective value
- Highlights optimal solution

**3. Parameter Evolution**
- X-axis: Iteration  
- Y-axis: Parameter values
- Two lines: γ and β evolution
- Shows how parameters change

**4. Efficient Frontier**
- X-axis: Portfolio risk (volatility)
- Y-axis: Expected return
- Scatter: Random portfolios (gray)
- Star: QAOA solution (red)
- Shows risk-return trade-off

## Exercises

### Beginner Exercises

**Exercise 1.1: More Assets**
- Increase to n=6 assets
- Add: TSLA (20% return), META (14% return)
- Budget K=3
- **Question**: Does QAOA still find optimum?

**Exercise 1.2: Risk Aversion**
- Try q=0.1 (low risk aversion)
- Try q=2.0 (high risk aversion)
- **Question**: How does optimal portfolio change?
- **Plot**: Efficient frontier for different q

**Exercise 1.3: QAOA Depth**
- Test p=1, 2, 3, 4
- **Question**: How does solution quality improve?
- **Trade-off**: Quality vs circuit depth

### Intermediate Exercises

**Exercise 2.1: Sector Constraints**
- Group assets by sector
- Constraint: Max 1 asset per sector
- **Implementation**: Modify Hamiltonian
- **Challenge**: Encode complex constraints

**Exercise 2.2: Long-Short Portfolio**
- Allow short positions: x_i ∈ {-1, 0, +1}
- Need 2 qubits per asset (encoding)
- **Question**: How does circuit size scale?

**Exercise 2.3: Alternative Mixer**
- Implement XY mixer (preserves budget)
- Compare with standard X mixer
- **Hypothesis**: XY mixer converges faster
- **Test**: Run 10 random instances

### Advanced Exercises

**Exercise 3.1: Warm Start**
- Use classical solution as initial state
- Modify initialization (not uniform superposition)
- **Implementation**: State preparation circuit
- **Question**: Does it reduce iterations?

**Exercise 3.2: Parameter Initialization**
- Try interpolating: γ from 0→1, β from 1→0
- Compare with random initialization
- **Reference**: QAOA parameter concentration
- **Analysis**: Success probability vs initialization

**Exercise 3.3: Real Hardware**
- Run on IBM Quantum (5-qubit processor)
- Implement: Error mitigation, readout correction
- **Compare**: Noisy results vs simulator
- **Deliverable**: Hardware execution report

**Exercise 3.4: Hybrid Decomposition**
- Decompose large portfolio (n=20) into sub-problems
- Solve each on quantum (n=5 per sub-problem)
- Classical master problem
- **Question**: When is hybrid beneficial?

## Limitations & Considerations

### Theoretical Limitations

**1. QAOA is Heuristic**
- No guarantee of finding optimal solution
- Approximation ratio depends on problem structure
- May get stuck in local minima

**2. NP-Hard Problem**
- Portfolio optimization is NP-hard
- QAOA doesn't change computational complexity class
- Advantage is constant-factor speedup (if any)

**3. Parameter Optimization**
- Finding good γ, β is hard (NP-hard!)
- Classical optimizer can fail
- Need many circuit evaluations

### Practical Considerations

**When QAOA Helps:**
✅ Medium-sized problems (n=20-100)
✅ When classical heuristics struggle
✅ Problems with specific structure (e.g., sparse)
✅ When approximate solution sufficient

**When Classical is Better:**
❌ Small problems (n<10) - brute force works
❌ Very large problems (n>100) - circuit too deep
❌ When exact solution required
❌ When convex relaxation is good enough

### Current Hardware Reality

**2025 Capabilities:**
- Max assets: ~20 (limited qubits)
- Circuit depth: ~100 gates (noise)
- Solution quality: 70-90% of optimal
- Time: Minutes to hours (queue + execution)

**Classical Comparison:**
- Gurobi solver: Exact, fast for small n
- Genetic algorithms: Better for large n
- Simulated annealing: Similar performance

**Verdict**: QAOA is research tool, not production-ready

## Performance Analysis

### Theoretical Complexity

| Method | Time | Solution Quality | Scalability |
|--------|------|-----------------|-------------|
| Brute Force | O(2^n) | Optimal | n<20 |
| Classical Heuristic | O(poly(n)) | 90-95% | n<1000 |
| QAOA | O(poly(n)) | 70-95% | n<50 (NISQ) |

### Approximation Ratio

For MaxCut (similar to portfolio):
- p=1: ~0.6924 approximation
- p→∞: Approaches optimal
- Empirical: p=10-20 needed for >0.95

### Parameter Count

Parameters to optimize:
- p=1: 2 parameters (γ, β)
- p=10: 20 parameters
- Classical optimization cost: O(iterations * parameters)

### Resource Requirements

For n assets, p layers:
- Qubits: n
- Gates: O(n² * p)
- Circuit depth: O(n * p)
- Classical optimization: 100-1000 iterations

**Example (n=20, p=5)**:
- Qubits: 20
- Gates: ~2000
- Depth: ~100
- Optimization: ~500 iterations
- Total time: 30-60 minutes

## Troubleshooting

### Common Issues

**1. "QAOA doesn't converge"**
- Increase max_iterations
- Try different optimizer (COBYLA → Adam)
- Reduce p (simpler landscape)
- Check Hamiltonian construction

**2. "Solution violates constraints"**
- Increase penalty weight λ
- Try custom mixer (XY for budget)
- Use projection after optimization

**3. "Classical is faster"**
- Expected for small n!
- QAOA advantage is for n>20
- Current example is educational

**4. "All solutions have same low probability"**
- Problem: Optimization stuck
- Solutions:
  - Better parameter initialization
  - Increase p
  - Try recursive QAOA

**5. "Circuit too deep for hardware"**
- Reduce n (number of assets)
- Reduce p (QAOA layers)
- Use variational compilation
- Try circuit cutting techniques

### Performance Tips

- Start with p=1, increase gradually
- Use gradient-free optimizers (COBYLA, SPSA)
- Parallelize: Evaluate multiple parameters simultaneously
- Cache results: Avoid recomputing same parameters
- Use parameter transfer: Similar problems have similar optima

## Further Reading

### Foundational Papers

**QAOA:**
1. Farhi, Goldstone, Gutmann (2014) - "A Quantum Approximate Optimization Algorithm"
   - Original QAOA paper
   - arXiv:1411.4028

2. Hadfield et al. (2019) - "From QAOA to Quantum Alternating Operator Ansatz"
   - Generalizations of QAOA
   - arXiv:1709.03489

**Portfolio Optimization:**
3. Venturelli & Kondratyev (2019) - "Reverse Quantum Annealing for Portfolio Optimization"
   - Practical application
   - arXiv:1810.08584

4. Hodson et al. (2019) - "Portfolio Rebalancing with QAOA"
   - Alternative formulation
   - arXiv:1911.05296

### Tutorials & Resources

**PennyLane:**
- QAOA Tutorial: https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html
- Max-Cut Example: https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html

**Qiskit:**
- QAOA for Finance: https://qiskit.org/ecosystem/finance/tutorials/

**Reviews:**
- Zhou et al. (2020) - "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation"

## Citation

    @article{farhi2014quantum,
      title={A quantum approximate optimization algorithm},
      author={Farhi, Edward and Goldstone, Jeffrey and Gutmann, Sam},
      journal={arXiv preprint arXiv:1411.4028},
      year={2014}
    }

    @software{quantum_finance_portfolio_qaoa,
      author = {Ian Buckley},
      title = {QAOA Portfolio Optimization Example},
      year = {2025},
      url = {https://github.com/roguetrainer/quantum-finance-examples/tree/main/2_portfolio_optimization}
    }

## Support

**Issues**: https://github.com/roguetrainer/quantum-finance-examples/issues

---

[← Previous: Option Pricing](../1_option_pricing/README.md) | [Back to Main](../README.md) | [Next: Credit Classification →](../3_credit_classification/README.md)