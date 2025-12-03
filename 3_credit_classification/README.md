# Example 3: Credit Classification with Quantum Machine Learning

## Overview

This example demonstrates how quantum machine learning can be applied to credit risk classification - determining whether a loan applicant is creditworthy based on their financial features.

## Financial Problem

**Objective**: Classify loan applicants as "creditworthy" (approve) or "risky" (deny)

**Classical Approach**: 
- Support Vector Machines (SVM)
- Random Forests
- Neural Networks
- Typical accuracy: 85-95%

**Quantum Approach**: Variational Quantum Classifier (VQC)
- Quantum feature maps encode data
- Parameterized quantum circuits
- Hybrid quantum-classical training
- Potential advantage through quantum kernels

## Background

### Credit Scoring

**Features commonly used:**
- Income level
- Debt-to-income ratio
- Credit history length
- Employment stability
- Number of open accounts
- Payment history

**Challenge**: Nonlinear decision boundaries in high-dimensional space

### Classical Machine Learning

**Support Vector Machine (SVM):**

Find hyperplane that maximizes margin:

    w·x + b = 0

With kernel K(x, x'):
- Linear: K(x,x') = x·x'
- RBF: K(x,x') = exp(-γ||x-x'||²)
- Polynomial: K(x,x') = (x·x' + c)^d

Training complexity: O(n²) to O(n³)

### Quantum Machine Learning

**Two Approaches:**

**1. Variational Quantum Classifier (VQC)**
- Encode features in quantum state
- Apply parameterized circuit
- Measure expectation value
- Use as classifier output

**2. Quantum Kernel Method**
- Compute quantum kernel: K(x,x') = |⟨φ(x)|φ(x')⟩|²
- Use with classical SVM
- Potentially richer feature space

### Quantum Feature Map

Maps classical data x to quantum state |φ(x)⟩

**Angle Encoding:**

    |φ(x)⟩ = ⊗ᵢ RY(xᵢ) RZ(xᵢ) |0⟩

**Amplitude Encoding:**

    |φ(x)⟩ = Σᵢ xᵢ/||x|| |i⟩

**ZZ Feature Map:**

    U(x) = ⊗ᵢ H ⊗ᵢ RZ(xᵢ) ⊗ᵢⱼ RZZ(xᵢxⱼ)

Entangles features for richer representation

### Variational Circuit

Parameterized ansatz with trainable parameters θ:

    |ψ(x,θ)⟩ = V(θ) U(x) |0⟩

Where:
- U(x) = feature map (fixed)
- V(θ) = variational circuit (trainable)

**Measurement:**

    y(x,θ) = ⟨ψ(x,θ)| M |ψ(x,θ)⟩

Where M is observable (e.g., Z₀)

### Training

Minimize loss function:

    L(θ) = Σᵢ (yᵢ - f(xᵢ,θ))²

Using gradient descent:

    θ ← θ - η ∇L(θ)

Gradients computed via:
- Parameter-shift rule (exact)
- Finite differences
- Simultaneous perturbation

## Implementation Details

### Circuit Architecture

**4-qubit example (4 features):**

**1. Feature Map Layer:**

    for i in range(n_qubits):
        RY(x[i], wires=i)
        RZ(x[i], wires=i)
    
    # Entangle features
    for i in range(n_qubits-1):
        CNOT(wires=[i, i+1])
    CNOT(wires=[n_qubits-1, 0])

**2. Variational Layers (repeat L times):**

    for layer in range(L):
        # Rotation layer
        for i in range(n_qubits):
            RY(θ[layer,i,0], wires=i)
            RZ(θ[layer,i,1], wires=i)
        
        # Entangling layer
        for i in range(n_qubits-1):
            CNOT(wires=[i, i+1])

**3. Measurement:**

    return expectation(PauliZ(0))

### Parameter Count

For n qubits, L layers:
- Rotations: 2 * n * L parameters
- Total: O(n*L) parameters

Example (n=4, L=2): 16 parameters

### Training Loop

    # Initialize
    θ = random_initialization()
    optimizer = Adam(learning_rate=0.01)
    
    for epoch in range(n_epochs):
        # Batch training
        for batch in data_loader:
            X_batch, y_batch = batch
            
            # Forward pass
            predictions = [quantum_circuit(x, θ) for x in X_batch]
            loss = mean_squared_error(y_batch, predictions)
            
            # Backward pass
            gradients = compute_gradients(loss, θ)
            θ = optimizer.update(θ, gradients)
        
        # Validation
        val_accuracy = evaluate(X_val, y_val, θ)

### Quantum Kernel Approach

Alternative: Use quantum feature map to define kernel

**Kernel Computation:**

    K(x, x') = |⟨0| U†(x) U(x') |0⟩|²

**Implementation:**

    def quantum_kernel(x1, x2):
        # Apply feature map for x1
        apply_feature_map(x1)
        
        # Apply inverse feature map for x2
        apply_feature_map_adjoint(x2)
        
        # Measure overlap
        return probability(all_zeros)

**Use with SVM:**

    # Compute kernel matrix
    K_train = [[quantum_kernel(x1, x2) for x2 in X_train] 
               for x1 in X_train]
    
    # Train SVM with precomputed kernel
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)

## Running the Example

### Quick Start

    cd 3_credit_classification
    python credit_qml.py

### Jupyter Notebook

    jupyter notebook credit_qml.ipynb

### Configuration

Edit parameters at top of script:

    # Data parameters
    n_samples = 200
    n_features = 4
    test_size = 0.3
    random_state = 42
    
    # Quantum parameters
    n_qubits = 4
    n_layers = 2
    
    # Training parameters
    n_epochs = 50
    batch_size = 25
    learning_rate = 0.01
    optimizer = 'Adam'  # or 'SGD', 'RMSprop'

## Expected Output

### Console Output

    ========================================
    CREDIT CLASSIFICATION WITH QUANTUM ML
    ========================================
    
    Dataset Information:
      Total samples: 200
      Training samples: 140
      Test samples: 60
      Features: 4 (income, debt_ratio, credit_history, employment_years)
      Classes: 2 (creditworthy=1, risky=-1)
    
    Feature Statistics:
                    mean    std     min     max
      income        5.23    2.15    0.82    10.45
      debt_ratio    0.42    0.18    0.05    0.85
      credit_hist   7.34    3.21    1.00    15.00
      employment    4.12    2.87    0.00    12.00
    
    ----------------------------------------
    Classical SVM Baseline
    ----------------------------------------
    Kernel: RBF
    Training time: 0.23s
    
    Training accuracy: 89.3%
    Test accuracy: 86.7%
    
    Classification Report:
                  precision  recall  f1-score  support
    Creditworthy     0.88      0.85     0.86       30
    Risky            0.85      0.88     0.87       30
    
    Confusion Matrix:
                 Predicted
                 Credit  Risky
    Actual Credit   26      4
           Risky      4     26
    
    ----------------------------------------
    Quantum VQC Training
    ----------------------------------------
    Architecture:
      Qubits: 4
      Layers: 2
      Parameters: 16
      Feature map: ZZ entangling
    
    Epoch  Train Loss  Train Acc  Test Acc  Time
       1     0.8234      0.557     0.550    2.3s
       5     0.6123      0.664     0.633    2.1s
      10     0.4156      0.757     0.733    2.2s
      20     0.2234      0.864     0.833    2.1s
      30     0.1523      0.907     0.867    2.2s
      40     0.1234      0.921     0.883    2.1s
      50     0.1089      0.929     0.883    2.2s
    
    Final Test Accuracy: 88.3%
    Total Training Time: 107.5s
    
    Classification Report:
                  precision  recall  f1-score  support
    Creditworthy     0.90      0.87     0.88       30
    Risky            0.87      0.90     0.88       30
    
    Confusion Matrix:
                 Predicted
                 Credit  Risky
    Actual Credit   26      4
           Risky      3     27
    
    ----------------------------------------
    Quantum Kernel SVM
    ----------------------------------------
    Computing quantum kernel matrix...
      Training kernel: 100%|████████| 140x140
      Test kernel: 100%|████████| 60x140
    
    Kernel computation time: 45.2s
    SVM training time: 0.8s
    
    Test Accuracy: 85.0%
    
    Classification Report:
                  precision  recall  f1-score  support
    Creditworthy     0.83      0.87     0.85       30
    Risky            0.87      0.83     0.85       30
    
    ----------------------------------------
    Comparison Summary
    ----------------------------------------
    Method              Accuracy  Training Time
    Classical SVM       86.7%     0.23s
    Quantum VQC         88.3%     107.5s
    Quantum Kernel SVM  85.0%     46.0s
    
    Notes:
    - Quantum VQC: Slightly better accuracy, much slower
    - Quantum Kernel: Similar accuracy, slow kernel computation
    - Classical: Fast and competitive
    - Dataset small - quantum advantage not expected

### Generated Plots

The script creates `credit_classification_results.png` with 6 subplots:

**1. Training Curves**
- X-axis: Epoch
- Y-axis: Loss
- Line: Training loss over time
- Shows convergence

**2. Accuracy Comparison**
- Bar chart: Train vs Test accuracy
- Three methods side by side
- Error bars if multiple runs

**3. Confusion Matrices** (3 subplots)
- Classical SVM
- Quantum VQC
- Quantum Kernel SVM
- Heatmap format

**4. Feature Space Visualization**
- 2D PCA projection
- Points colored by true label
- Background colored by VQC predictions
- Shows decision boundary

**5. Kernel Matrix Heatmap**
- Quantum kernel values
- Shows similarity structure
- Blocks indicate class separation

**6. Training Time Comparison**
- Bar chart: Training time by method
- Log scale
- Highlights quantum overhead

## Exercises

### Beginner Exercises

**Exercise 1.1: More Features**
- Add 2 more features (6 total)
- Requires 6 qubits
- **Question**: How does accuracy change?
- **Observation**: Training time impact

**Exercise 1.2: Different Feature Map**
- Try amplitude encoding instead of angle encoding
- **Implementation**: Normalize features, load as amplitudes
- **Question**: Which works better for this data?

**Exercise 1.3: Circuit Depth**
- Test L=1, 2, 3, 4 layers
- **Plot**: Accuracy vs number of layers
- **Question**: Diminishing returns after how many layers?

### Intermediate Exercises

**Exercise 2.1: Imbalanced Dataset**
- Create dataset with 80% creditworthy, 20% risky
- **Challenge**: Model predicts majority class
- **Solutions**:
  - Class weights in loss function
  - Oversampling minority class
  - Different evaluation metrics (F1, AUC-ROC)

**Exercise 2.2: Cross-Validation**
- Implement 5-fold cross-validation
- **Report**: Mean ± std accuracy
- **Question**: Is quantum advantage statistically significant?

**Exercise 2.3: Hyperparameter Tuning**
- Grid search over:
  - Learning rate: [0.001, 0.01, 0.1]
  - Batch size: [10, 25, 50]
  - Number of layers: [1, 2, 3]
- **Deliverable**: Best configuration for quantum VQC

### Advanced Exercises

**Exercise 3.1: Feature Importance**
- Which features most important for quantum classifier?
- **Method**: Permutation importance
- **Implementation**:
  - Shuffle one feature at a time
  - Measure accuracy drop
  - Rank features
- **Compare**: Classical vs quantum feature ranking

**Exercise 3.2: Quantum Advantage Analysis**
- When does quantum approach win?
- **Hypothesis**: Advantage in high dimensions with entanglement
- **Test**: Vary number of features (2, 4, 6, 8)
- **Analysis**: Accuracy gap vs classical
- **Theory**: Quantum feature space expressivity

**Exercise 3.3: Hardware Execution**
- Run on real quantum hardware (IBM/IonQ)
- **Challenges**:
  - Limited connectivity
  - Noise in measurements
  - Queue times
- **Error Mitigation**:
  - Zero-noise extrapolation
  - Readout error mitigation
  - Probabilistic error cancellation
- **Deliverable**: Noisy vs noiseless comparison

**Exercise 3.4: Explainability**
- Quantum ML is "black box"
- **Goal**: Explain predictions
- **Approaches**:
  - Visualize quantum states (Bloch sphere)
  - Analyze measurement statistics
  - Sensitivity analysis
- **Question**: Why did model classify applicant as risky?

## Limitations & Considerations

### Theoretical Limitations

**1. No Proven Quantum Advantage**
- For classification, quantum advantage is debated
- "Dequantization" results (Tang et al.) show classical algorithms matching quantum
- Advantage likely problem-specific

**2. Barren Plateaus**
- Random VQC initialization can lead to flat loss landscape
- Gradients vanish exponentially with depth
- **Solutions**: Smart initialization, specific ansatz design

**3. Training Complexity**
- Training still requires many iterations
- Each iteration: quantum + classical computation
- No speedup in training time proven

### Practical Considerations

**When Quantum ML Helps:**
✅ High-dimensional data (>10 features)
✅ Complex feature interactions
✅ When quantum kernels express patterns classical kernels miss
✅ As research/learning tool

**When Classical is Better:**
❌ Small datasets (n<1000)
❌ Low-dimensional problems
❌ When interpretability is critical
❌ Production systems (reliability, speed)

### Current Hardware Reality

**2025 Status:**
- Available qubits: 50-1000
- Practical for QML: 4-10 qubits
- Accuracy competitive but not better
- Training time: 10-100x slower than classical
- **Verdict**: Research stage, not production

**Classical State-of-the-Art:**
- Random Forests: Fast, interpretable, accurate
- Gradient Boosting (XGBoost): Best accuracy
- Deep Learning: Best for very large data
- **Reality**: Hard to beat classical ML

## Performance Analysis

### Theoretical Complexity

| Method | Training Time | Prediction Time | Memory |
|--------|--------------|----------------|---------|
| Classical SVM | O(n²) - O(n³) | O(n_sv) | O(n²) |
| Quantum VQC | O(n·p·iter) | O(1) | O(log n) qubits |
| Quantum Kernel | O(n²·t_qc) | O(n·t_qc) | O(n²) |

Where:
- n = training samples
- p = circuit parameters
- iter = optimization iterations
- n_sv = support vectors
- t_qc = quantum circuit time

### Practical Performance

For this example (n=200, 4 features):
- Classical SVM: 0.2s training, 86-88% accuracy
- Quantum VQC: 100s training, 85-90% accuracy
- Quantum Kernel: 50s kernel + 0.5s SVM, 83-87% accuracy

**Speedup**: None (yet!)
**Accuracy**: Comparable

### Scaling

As n increases:
- Classical: Slows down significantly (quadratic/cubic)
- Quantum VQC: Training iterations increase, but circuit time constant
- Quantum Kernel: Kernel computation is bottleneck

**Hypothesis**: Quantum advantage for n>10,000 (unproven)

## Troubleshooting

### Common Issues

**1. "Accuracy stuck at 50% (random guessing)"**
- Problem: Barren plateau or bad initialization
- Solutions:
  - Reduce circuit depth (fewer layers)
  - Use layer-wise training
  - Initialize parameters near 0
  - Try different ansatz

**2. "Quantum kernel SVM very slow"**
- Expected: Computing n² kernel entries with quantum circuits
- Solutions:
  - Reduce training set size
  - Use subset of training data
  - Parallelize kernel computation
  - Use QSVM with approximate kernels

**3. "Results not reproducible"**
- Problem: Random initialization + stochastic gradients
- Solutions:
  - Set random seed (np.random.seed, torch.manual_seed)
  - Use deterministic algorithms
  - Average over multiple runs

**4. "Classical SVM beats quantum always"**
- This is expected for small datasets!
- Quantum advantage (if exists) is for large-scale problems
- Current example is educational

**5. "Training loss increases"**
- Problem: Learning rate too high or gradient explosion
- Solutions:
  - Reduce learning rate (try 0.001)
  - Use gradient clipping
  - Try different optimizer (Adam → RMSprop)
  - Check for bugs in circuit

### Performance Tips

**For Faster Training:**
- Use pennylane-lightning (GPU if available)
- Reduce batch size for faster iteration
- Reduce n_layers (1-2 sufficient for small data)
- Use classical pre-training (initialize near classical solution)

**For Better Accuracy:**
- Increase n_layers (but watch for barren plateaus)
- Try different feature maps
- Ensemble methods (train multiple VQCs, vote)
- Hybrid: Quantum features → Classical classifier

## Further Reading

### Foundational Papers

**Quantum ML Theory:**
1. Havlíček et al. (2019) - "Supervised learning with quantum-enhanced feature spaces"
   - Quantum kernel methods
   - Nature 567, 209-212

2. Schuld & Killoran (2019) - "Quantum machine learning in feature Hilbert spaces"
   - Theoretical framework
   - Physical Review Letters 122, 040504

3. Huang et al. (2021) - "Power of data in quantum machine learning"
   - When quantum helps
   - Nature Communications 12, 2631

**Barren Plateaus:**
4. McClean et al. (2018) - "Barren plateaus in quantum neural network training landscapes"
   - arXiv:1803.11173

**Dequantization:**
5. Tang (2019) - "A quantum-inspired classical algorithm for recommendation systems"
   - Challenges quantum advantage claims
   - STOC 2019

### Tutorials & Resources

**PennyLane:**
- QML Demos: https://pennylane.ai/qml/demonstrations.html
- VQC Tutorial: https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
- Quantum Kernels: https://pennylane.ai/qml/demos/tutorial_kernels_module.html

**Qiskit:**
- Qiskit Machine Learning: https://qiskit.org/ecosystem/machine-learning/

**Books:**
- "Supervised Learning with Quantum Computers" (Schuld & Petruccione)
- "Machine Learning with Quantum Computers" (Schuld & Petruccione)

## Citation

    @article{havlicek2019supervised,
      title={Supervised learning with quantum-enhanced feature spaces},
      author={Havl{\'\i}{\v{c}}ek, Vojt{\v{e}}ch and C{\'o}rcoles, Antonio D and others},
      journal={Nature},
      volume={567},
      number={7747},
      pages={209--212},
      year={2019},
      publisher={Nature Publishing Group}
    }

    @software{quantum_finance_credit_qml,
      author = {Ian Buckley},
      title = {Quantum Credit Classification Example},
      year = {2025},
      url = {https://github.com/roguetrainer/quantum-finance-examples/tree/main/3_credit_classification}
    }

## Support

**Issues**: https://github.com/roguetrainer/quantum-finance-examples/issues

---

[← Previous: Portfolio Optimization](../2_portfolio_optimization/README.md) | [Back to Main](../README.md) | [Next: Tensor Networks →](../4_tensor_networks/README.md)