## Related Repositories

These repositories provide additional depth on quantum-inspired methods and related mathematical frameworks, particularly relevant to **Example 4: Tensor Networks**.

### 🔗 [tensor-scalpel](https://github.com/roguetrainer/tensor-scalpel)

**Tensor Network Decomposition Toolkit**

A lightweight Python library for tensor network decomposition and analysis, with focus on financial applications. Provides optimized implementations of Matrix Product States (MPS), DMRG, and tensor contraction algorithms.

**Relation to Example 4:**
- Provides production-grade implementations of the MPS decomposition used in Example 4
- Includes additional decomposition methods (Tensor Train, Tucker decomposition)
- Optimized for large-scale portfolio correlation analysis (1000+ assets)
- Includes GPU acceleration for tensor operations
- Offers more sophisticated bond dimension optimization strategies

**Use Case:** If you want to move beyond the educational implementation in Example 4 to production-scale tensor network analysis, `tensor-scalpel` provides the tools you need.

**Key Features:**
- Efficient SVD-based and DMRG-based MPS construction
- Automatic bond dimension selection
- Memory-efficient tensor contractions
- Integration with financial data pipelines

---

### 🔗 [tensor_networks_finance](https://github.com/roguetrainer/tensor_networks_finance)

**Financial Applications of Tensor Networks**

Comprehensive collection of tensor network applications in quantitative finance, including correlation analysis, risk modeling, option pricing, and portfolio optimization.

**Relation to Example 4:**
- Extends Example 4's correlation analysis to additional use cases:
  - **Dynamic correlation modeling** (time-varying correlation matrices)
  - **Multi-asset option pricing** using tensor network states
  - **High-dimensional risk factor models** with tensor decomposition
  - **Portfolio optimization** with tensor network representations
- Includes real market data examples and backtests
- Demonstrates production workflows and data pipelines

**Use Case:** After understanding the basics in Example 4, explore how tensor networks solve real-world problems across multiple domains of quantitative finance.

**Key Applications:**
- Correlation structure analysis across asset classes
- Factor models for 1000+ securities
- Real-time risk monitoring systems
- Stress testing and scenario analysis

**Notable Results:**
- 50-100x speedup for correlation eigendecomposition (n>500)
- <0.5% error rates with appropriate bond dimensions
- Successfully deployed in production risk systems

---

### 🔗 [zx-calculus-demo](https://github.com/roguetrainer/zx-calculus-demo)

**ZX-Calculus for Quantum Circuit Optimization**

Interactive demonstrations of ZX-calculus, a graphical language for reasoning about quantum computations. While more theoretical, ZX-calculus provides insights into tensor network structure and optimization.

**Relation to Example 4:**
- **Mathematical foundation:** ZX-calculus and tensor networks share the same categorical/diagrammatic language
- **Circuit optimization:** Shows how quantum circuits (which are tensor networks) can be simplified
- **Visualization:** Provides intuitive graphical representations of tensor contractions
- **Theoretical connection:** Demonstrates the quantum mechanics roots of tensor network methods

**Use Case:** For those interested in the deeper mathematical foundations underlying tensor networks and why they work for classical optimization problems.

**Key Concepts:**
- Graphical representation of tensor operations
- Rewrite rules for tensor network simplification
- Connection between quantum circuits and classical tensor networks
- Why quantum-inspired methods work on classical hardware

**Educational Value:**
- Understand why tensor networks compress information efficiently
- See the quantum mechanics principles that inspire classical algorithms
- Learn systematic approaches to tensor network optimization
- Bridge between quantum computing theory and practical applications

---

### 🎯 How These Repos Fit Together
```
quantum-finance-examples (this repo)
    └── Example 4: Tensor Networks
            ↓
    Educational implementation showing:
    • Basic MPS decomposition
    • Correlation matrix compression
    • Proof of concept (~100 assets)
            ↓
    ┌───────────────────┬────────────────────┬──────────────────┐
    ↓                   ↓                    ↓                  ↓
tensor-scalpel   tensor_networks_    zx-calculus-demo   Continue to
                      finance                            Examples 1-3
Production-grade   Real applications   Theoretical         (Quantum)
implementations    & backtests        foundations
                                                      
- Optimized        • Portfolio mgmt    • Mathematical    • Option pricing
  algorithms       • Risk systems        framework       • QAOA
- GPU support      • Market data       • Circuit         • Quantum ML
- 1000+ assets     • Production          optimization    
                     workflows         • Visualization
```

**Learning Path:**

1. **Start here (Example 4):** Understand basic tensor network concepts
2. **Explore `tensor_networks_finance`:** See diverse financial applications
3. **Use `tensor-scalpel`:** Implement production systems
4. **Study `zx-calculus-demo`:** Understand theoretical foundations
5. **Return to Examples 1-3:** See how full quantum computing will enhance these methods

---

### 💡 Why Quantum-Inspired Methods Matter

These repositories demonstrate a crucial point: **you don't need quantum hardware to benefit from quantum thinking.**

- **Tensor Networks** (Example 4 + related repos): Available TODAY, proven speedups
- **Quantum Computing** (Examples 1-3): Coming in 5-10 years, transformative potential

The quantum-inspired approach lets you:
- ✅ Deploy solutions in production now
- ✅ Build expertise in quantum-adjacent methods
- ✅ Position for quantum advantage when hardware arrives
- ✅ Bridge classical and quantum computational paradigms

**Companies using quantum-inspired methods today:**
- **Nomura Securities** (portfolio optimization with Fujitsu)
- **SoftBank** (Vision Fund analysis with tensor networks)
- **Mizuho Bank** (FX arbitrage with Toshiba SBM)
- **Goldman Sachs** (risk analytics with quantum-inspired algorithms)

These firms aren't waiting for quantum computers - they're getting value from quantum mathematics on classical hardware **right now**.

---