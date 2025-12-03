"""
Portfolio Optimization with QAOA

This script demonstrates the Quantum Approximate Optimization Algorithm (QAOA)
for solving portfolio optimization problems.

Author: Ian Buckley
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

import pennylane as qml
from pennylane import numpy as pnp


# ============================================================================
# PARAMETERS
# ============================================================================

# Portfolio parameters
n_assets = 4                # Number of assets
budget = 2                  # Number of assets to select
risk_aversion = 0.5         # Risk aversion parameter (λ)

# QAOA parameters
n_qubits = n_assets
n_layers = 2                # Number of QAOA layers (p)
n_iterations = 100          # Optimization iterations

# Random seed
np.random.seed(42)


# ============================================================================
# GENERATE PORTFOLIO DATA
# ============================================================================

def generate_portfolio_data(n_assets, correlation=0.3):
    """
    Generate synthetic portfolio data.
    
    Returns:
        expected_returns: Expected return for each asset
        cov_matrix: Covariance matrix
    """
    # Expected returns (annual)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Generate covariance matrix
    # Start with random correlation matrix
    volatilities = np.random.uniform(0.1, 0.3, n_assets)
    
    # Create correlation matrix with specified average correlation
    corr_matrix = np.ones((n_assets, n_assets)) * correlation
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Add some randomness
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr_matrix[i, j] = corr_matrix[j, i] = correlation + np.random.uniform(-0.2, 0.2)
            # Ensure correlations in [-1, 1]
            corr_matrix[i, j] = np.clip(corr_matrix[i, j], -0.95, 0.95)
            corr_matrix[j, i] = corr_matrix[i, j]
    
    # Convert to covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    return expected_returns, cov_matrix


# ============================================================================
# CLASSICAL OPTIMIZATION (BASELINE)
# ============================================================================

def classical_portfolio_optimization(returns, cov_matrix, budget, risk_aversion):
    """
    Solve portfolio optimization classically using brute force for small problems.
    
    Objective: Maximize returns - risk_aversion * variance
    Constraint: Select exactly 'budget' assets
    """
    from itertools import combinations
    
    n = len(returns)
    best_objective = -np.inf
    best_portfolio = None
    
    # Try all combinations
    for combo in combinations(range(n), budget):
        # Create binary portfolio vector
        x = np.zeros(n)
        x[list(combo)] = 1
        
        # Calculate objective
        portfolio_return = np.dot(returns, x)
        portfolio_variance = x.T @ cov_matrix @ x
        objective = portfolio_return - risk_aversion * portfolio_variance
        
        if objective > best_objective:
            best_objective = objective
            best_portfolio = x
    
    return best_portfolio, best_objective


# ============================================================================
# QAOA IMPLEMENTATION
# ============================================================================

def create_cost_hamiltonian(returns, cov_matrix, risk_aversion, budget):
    """
    Create cost Hamiltonian for portfolio optimization.
    
    H = -Σᵢ rᵢ Zᵢ + λ Σᵢⱼ σᵢⱼ ZᵢZⱼ + μ(Σᵢ Zᵢ - 2B + n)²
    
    Where:
        - First term: negative returns (we minimize)
        - Second term: risk (variance)
        - Third term: budget constraint penalty
    """
    n = len(returns)
    
    # Pauli Z basis: |0⟩ -> +1, |1⟩ -> -1
    # Map to portfolio: xᵢ = (1 - Zᵢ)/2
    # So Zᵢ = 1 - 2xᵢ
    
    # Convert to QUBO form
    # Objective: maximize  Σᵢ rᵢ xᵢ - λ Σᵢⱼ σᵢⱼ xᵢxⱼ
    #            subject to Σᵢ xᵢ = B
    
    # For Hamiltonian, we minimize -objective + penalty
    
    # Coefficients for linear terms (Zᵢ)
    coeffs_linear = -returns / 2  # Factor of 1/2 from mapping
    
    # Coefficients for quadratic terms (ZᵢZⱼ)
    coeffs_quadratic = risk_aversion * cov_matrix / 4  # Factor of 1/4 from mapping
    
    # Budget constraint penalty (simplified)
    penalty_strength = 10.0  # Large penalty for constraint violation
    
    return coeffs_linear, coeffs_quadratic, penalty_strength


def qaoa_circuit(gammas, betas, returns, cov_matrix, risk_aversion, budget):
    """
    QAOA circuit for portfolio optimization.
    
    Args:
        gammas: Cost Hamiltonian angles (length p)
        betas: Mixer Hamiltonian angles (length p)
    """
    p = len(gammas)
    
    # Initial state: uniform superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Get Hamiltonian coefficients
    c_linear, c_quad, penalty = create_cost_hamiltonian(
        returns, cov_matrix, risk_aversion, budget
    )
    
    # QAOA layers
    for layer in range(p):
        # Cost Hamiltonian: U(C, γ) = exp(-iγC)
        
        # Linear terms: exp(-iγ cᵢ Zᵢ) = RZ(2γcᵢ)
        for i in range(n_qubits):
            qml.RZ(2 * gammas[layer] * c_linear[i], wires=i)
        
        # Quadratic terms: exp(-iγ cᵢⱼ ZᵢZⱼ)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gammas[layer] * c_quad[i,j], wires=j)
                qml.CNOT(wires=[i, j])
        
        # Budget constraint penalty (simplified)
        # This is approximate - full implementation would be more complex
        for i in range(n_qubits):
            qml.RZ(2 * gammas[layer] * penalty * (1 - 2*budget/n_qubits), wires=i)
        
        # Mixer Hamiltonian: U(B, β) = exp(-iβB) where B = Σᵢ Xᵢ
        for i in range(n_qubits):
            qml.RX(2 * betas[layer], wires=i)
    
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def cost_function(params, returns, cov_matrix, risk_aversion, budget, dev):
    """
    Cost function to minimize.
    
    Extracts portfolio from quantum state and evaluates objective.
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    
    @qml.qnode(dev)
    def circuit():
        return qaoa_circuit(gammas, betas, returns, cov_matrix, risk_aversion, budget)
    
    # Get expectation values
    expectation_vals = circuit()
    
    # Convert to portfolio weights (approximately)
    # Z expectation: +1 -> not selected, -1 -> selected
    # x = (1 - Z)/2
    portfolio = np.array([(1 - z) / 2 for z in expectation_vals])
    
    # Evaluate objective (we minimize, so negate)
    portfolio_return = np.dot(returns, portfolio)
    portfolio_variance = portfolio.T @ cov_matrix @ portfolio
    
    objective = portfolio_return - risk_aversion * portfolio_variance
    
    # Add penalty for budget violation
    budget_violation = abs(np.sum(portfolio) - budget)
    penalty = 10.0 * budget_violation
    
    return -(objective - penalty)  # Minimize negative objective


def run_qaoa(returns, cov_matrix, risk_aversion, budget, n_layers, n_iterations):
    """Run QAOA optimization."""
    
    print("-" * 80)
    print("QAOA OPTIMIZATION")
    print("-" * 80)
    
    # Create device
    dev = qml.device('default.qubit', wires=n_qubits)
    
    # Initialize parameters (small random values)
    np.random.seed(42)
    initial_gammas = np.random.uniform(0, 0.1, n_layers)
    initial_betas = np.random.uniform(0, 0.1, n_layers)
    initial_params = np.concatenate([initial_gammas, initial_betas])
    
    print(f"Parameters: {2*n_layers} ({n_layers} gammas + {n_layers} betas)")
    print(f"Optimization method: COBYLA")
    print(f"Maximum iterations: {n_iterations}")
    print()
    
    # Track optimization progress
    costs_history = []
    
    def callback(params):
        cost = cost_function(params, returns, cov_matrix, risk_aversion, budget, dev)
        costs_history.append(cost)
        if len(costs_history) % 10 == 0:
            print(f"Iteration {len(costs_history):3d}: Cost = {cost:.6f}")
    
    # Optimize
    print("Starting optimization...")
    start_time = time.time()
    
    result = minimize(
        cost_function,
        initial_params,
        args=(returns, cov_matrix, risk_aversion, budget, dev),
        method='COBYLA',
        options={'maxiter': n_iterations},
        callback=callback
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nOptimization complete in {elapsed:.2f}s")
    print(f"Final cost: {result.fun:.6f}")
    print(f"Function evaluations: {result.nfev}")
    print()
    
    # Extract final portfolio
    optimal_params = result.x
    p = len(optimal_params) // 2
    gammas = optimal_params[:p]
    betas = optimal_params[p:]
    
    @qml.qnode(dev)
    def final_circuit():
        return qaoa_circuit(gammas, betas, returns, cov_matrix, risk_aversion, budget)
    
    expectation_vals = final_circuit()
    portfolio = np.array([(1 - z) / 2 for z in expectation_vals])
    
    # Round to binary (0 or 1)
    portfolio_binary = np.round(portfolio)
    
    # Adjust to satisfy budget constraint
    if np.sum(portfolio_binary) != budget:
        # Select top 'budget' assets by weight
        indices = np.argsort(portfolio)[::-1][:budget]
        portfolio_binary = np.zeros(n_assets)
        portfolio_binary[indices] = 1
    
    # Evaluate final objective
    final_return = np.dot(returns, portfolio_binary)
    final_variance = portfolio_binary.T @ cov_matrix @ portfolio_binary
    final_objective = final_return - risk_aversion * final_variance
    
    return portfolio_binary, final_objective, costs_history, elapsed


# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def analyze_results(returns, cov_matrix, classical_portfolio, classical_objective,
                   qaoa_portfolio, qaoa_objective, costs_history):
    """Analyze and compare results."""
    
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    print("Classical Portfolio (Optimal):")
    print(f"  Selected assets: {np.where(classical_portfolio > 0)[0]}")
    print(f"  Weights: {classical_portfolio}")
    print(f"  Expected return: {np.dot(returns, classical_portfolio):.4f}")
    print(f"  Portfolio variance: {classical_portfolio.T @ cov_matrix @ classical_portfolio:.4f}")
    print(f"  Objective value: {classical_objective:.4f}")
    print()
    
    print("QAOA Portfolio:")
    print(f"  Selected assets: {np.where(qaoa_portfolio > 0)[0]}")
    print(f"  Weights: {qaoa_portfolio}")
    print(f"  Expected return: {np.dot(returns, qaoa_portfolio):.4f}")
    print(f"  Portfolio variance: {qaoa_portfolio.T @ cov_matrix @ qaoa_portfolio:.4f}")
    print(f"  Objective value: {qaoa_objective:.4f}")
    print()
    
    # Approximation ratio
    if classical_objective != 0:
        approx_ratio = qaoa_objective / classical_objective
        print(f"Approximation Ratio: {approx_ratio:.4f}")
        print(f"Optimality Gap: {(1-approx_ratio)*100:.2f}%")
    else:
        print("Approximation Ratio: N/A (classical objective is zero)")
    print()


def plot_results(returns, cov_matrix, classical_portfolio, classical_objective,
                qaoa_portfolio, qaoa_objective, costs_history):
    """Create visualizations."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. QAOA Convergence
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(costs_history, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (negative objective)')
    ax1.set_title('QAOA Optimization Convergence')
    ax1.grid(True, alpha=0.3)
    
    # 2. Portfolio Comparison
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(n_assets)
    width = 0.35
    ax2.bar(x - width/2, classical_portfolio, width, label='Classical', alpha=0.8)
    ax2.bar(x + width/2, qaoa_portfolio, width, label='QAOA', alpha=0.8)
    ax2.set_xlabel('Asset')
    ax2.set_ylabel('Weight')
    ax2.set_title('Portfolio Allocation Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Asset {i}' for i in range(n_assets)], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Risk-Return Plot
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate metrics for all individual assets
    asset_returns = returns
    asset_risks = np.sqrt(np.diag(cov_matrix))
    
    ax3.scatter(asset_risks, asset_returns, s=100, alpha=0.6, label='Individual Assets')
    
    # Classical portfolio
    classical_return = np.dot(returns, classical_portfolio)
    classical_risk = np.sqrt(classical_portfolio.T @ cov_matrix @ classical_portfolio)
    ax3.scatter(classical_risk, classical_return, s=200, marker='*', 
               color='red', label='Classical Portfolio', zorder=5)
    
    # QAOA portfolio
    qaoa_return = np.dot(returns, qaoa_portfolio)
    qaoa_risk = np.sqrt(qaoa_portfolio.T @ cov_matrix @ qaoa_portfolio)
    ax3.scatter(qaoa_risk, qaoa_return, s=200, marker='s',
               color='green', label='QAOA Portfolio', zorder=5)
    
    ax3.set_xlabel('Risk (Volatility)')
    ax3.set_ylabel('Expected Return')
    ax3.set_title('Risk-Return Space')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Label assets
    for i in range(n_assets):
        ax3.annotate(f'{i}', (asset_risks[i], asset_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Covariance Matrix
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(cov