# Issue 322: Comprehensive Matrix Decomposition Methods - Junior Developer Implementation Guide

## Table of Contents
1. [Understanding Matrix Decompositions](#understanding-matrix-decompositions)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation Guide](#implementation-guide)
4. [Testing Strategy](#testing-strategy)
5. [Performance Considerations](#performance-considerations)

---

## Understanding Matrix Decompositions

### What are Matrix Decompositions?

**For Beginners:** Matrix decomposition is like factoring a number, but for matrices. Just as 12 = 3 × 4, we can break down matrices into simpler component matrices that are easier to work with.

**Why decompose matrices?**
1. **Solving Linear Systems:** Instead of directly solving Ax = b, decompose A and solve easier subproblems
2. **Matrix Inversion:** Computing A⁻¹ directly is slow and unstable; decompositions are faster and more accurate
3. **Eigenvalues/Eigenvectors:** Find the fundamental characteristics of a matrix
4. **Dimensionality Reduction:** Compress data while preserving important information (PCA uses SVD)
5. **Numerical Stability:** Many decompositions are more numerically stable than direct methods

**Analogy:** Think of a complex machine:
- **Direct approach:** Try to understand the whole machine at once (hard!)
- **Decomposition approach:** Break it into subsystems (engine, transmission, wheels), understand each part separately, then see how they work together

### Types of Decompositions Covered in This Guide

| Decomposition | Purpose | When to Use |
|---------------|---------|-------------|
| **SVD** | Find principal components | Dimensionality reduction, pseudoinverse, rank |
| **NMF** | Non-negative factorization | Topic modeling, image processing, recommender systems |
| **LU** | Solve linear systems | Ax=b with square A, multiple right-hand sides |
| **QR** | Orthogonalization | Least squares, eigenvalue algorithms |
| **Cholesky** | Fast for symmetric positive definite | Optimization, Gaussian processes, covariance matrices |
| **Eigen** | Find eigenvalues/vectors | Stability analysis, PCA, spectral clustering |
| **ICA** | Independent component separation | Signal processing, blind source separation |

---

## Mathematical Foundations

### 1. Singular Value Decomposition (SVD)

**Note:** AiDotNet already has `SvdDecomposition.cs`. This section explains the math for understanding and potential enhancements.

**Factorization:**
```
A = U × Σ × Vᵀ
```

Where:
- **U** (m × m): Left singular vectors (orthonormal columns) - patterns in rows
- **Σ** (m × n): Diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0)
- **Vᵀ** (n × n): Right singular vectors (orthonormal rows) - patterns in columns

**Key Properties:**
1. **Orthonormality:** UᵀU = I, VᵀV = I
2. **Singular values:** σᵢ² are eigenvalues of AᵀA (or AAᵀ)
3. **Rank:** Number of non-zero singular values = rank(A)
4. **Best approximation:** Truncated SVD gives best low-rank approximation

**Computational Complexity:** O(min(m²n, mn²)) for dense matrices

**Applications:**
1. **Pseudoinverse:** A⁺ = V × Σ⁺ × Uᵀ (where Σ⁺ inverts non-zero σᵢ)
2. **Dimensionality Reduction:** Keep only top k singular values
3. **Data Compression:** Approximate A ≈ Uₖ × Σₖ × Vₖᵀ
4. **Recommender Systems:** Matrix completion with SVD

**For Beginners:** SVD finds the "principal directions" in your data. Imagine a cloud of points in 3D space - SVD finds the main axis the cloud is stretched along (first singular vector), then the second-most-important direction (perpendicular to the first), and so on.

### 2. Non-Negative Matrix Factorization (NMF)

**Factorization:**
```
A ≈ W × H
```

Where:
- **A** (m × n): Original non-negative matrix (all entries ≥ 0)
- **W** (m × k): Basis matrix (all entries ≥ 0)
- **H** (k × n): Coefficient matrix (all entries ≥ 0)
- **k**: Number of components (user-specified, typically k << min(m, n))

**Constraint:** All matrices must be non-negative (no negative entries).

**Optimization Problem:**
```
minimize ||A - WH||²_F
subject to W ≥ 0, H ≥ 0
```

**Algorithms:**
1. **Multiplicative Update Rules (Lee & Seung 2001):**
   ```
   W ← W ⊙ [(A × Hᵀ) / (W × H × Hᵀ)]
   H ← H ⊙ [(Wᵀ × A) / (Wᵀ × W × H)]
   where ⊙ is element-wise multiplication
   ```

2. **Alternating Least Squares (ALS):**
   - Fix H, solve for W (non-negative least squares)
   - Fix W, solve for H (non-negative least squares)
   - Repeat until convergence

**Applications:**
1. **Topic Modeling:** Documents × words → topics × word distributions
2. **Image Processing:** Face images → basis faces (like eigenfaces but parts-based)
3. **Recommender Systems:** Users × items → latent factors (interpretable)
4. **Audio Source Separation:** Spectrogram → source spectrograms

**For Beginners:** NMF finds "parts" that combine to make the whole. For face images, it might find components like "nose", "eyes", "mouth" (all non-negative) that add up to create faces. Unlike PCA/SVD which can have negative components (less interpretable), NMF gives only positive "parts" that you can intuitively understand.

### 3. LU Decomposition

**Factorization:**
```
PA = LU
```

Where:
- **P** (m × m): Permutation matrix (row swaps for numerical stability)
- **L** (m × m): Lower triangular matrix (diagonal elements = 1)
- **U** (m × m): Upper triangular matrix

**Without Pivoting (unstable):**
```
A = LU
```

**With Partial Pivoting (recommended):**
```
PA = LU
```

**Algorithm (Gaussian Elimination with Pivoting):**
```
For k = 1 to n-1:
  1. Find largest element in column k below diagonal (pivoting)
  2. Swap rows if needed (record in P)
  3. For i = k+1 to n:
       L[i,k] = A[i,k] / A[k,k]
       For j = k+1 to n:
           A[i,j] -= L[i,k] × A[k,j]
  4. U[k,:] = A[k,:] (copy kth row to U)
```

**Solving Ax = b using LU:**
```
Step 1: Permute b: b' = Pb
Step 2: Solve Ly = b' (forward substitution)
Step 3: Solve Ux = y (backward substitution)
```

**Computational Complexity:** O(n³) for decomposition, O(n²) per solve

**When to Use:**
- Multiple right-hand sides (decompose once, solve many times)
- Square matrices
- Need both L and U factors explicitly

**For Beginners:** LU decomposition turns a general matrix into two simpler triangular matrices. Solving systems with triangular matrices is much easier (like solving y = 2x + 3 step by step: first find x, then find y, rather than solving both at once).

### 4. QR Decomposition

**Factorization:**
```
A = QR
```

Where:
- **Q** (m × m): Orthogonal matrix (QᵀQ = I)
- **R** (m × n): Upper triangular matrix

**Thin QR (more common):**
```
A = Q̂R̂
where Q̂ is m × n, R̂ is n × n
```

**Algorithms:**

1. **Gram-Schmidt (simple but unstable):**
   ```
   For each column aⱼ of A:
     vⱼ = aⱼ - Σᵢ₌₁ʲ⁻¹ (qᵢᵀaⱼ)qᵢ  (orthogonalize)
     qⱼ = vⱼ / ||vⱼ||             (normalize)
     rᵢⱼ = qᵢᵀaⱼ for i ≤ j       (fill R)
   ```

2. **Modified Gram-Schmidt (more stable):**
   - Same idea but reorthogonalize at each step

3. **Householder Reflections (most stable):**
   - Use reflections to zero out below-diagonal elements
   - Most commonly used in practice

**Applications:**
1. **Least Squares:** Solve Ax = b when A is tall (more equations than unknowns)
   ```
   Minimize ||Ax - b||²
   Solution: x = R⁻¹(Qᵀb)
   ```

2. **Eigenvalue Algorithms:** QR iteration for finding eigenvalues

3. **Orthonormalization:** Extract orthonormal basis from columns of A

**Computational Complexity:** O(mn²) for Householder method

**For Beginners:** QR decomposition finds an orthonormal basis (Q) that spans the same space as your original matrix columns, plus the coordinates (R) to reconstruct the original. It's like finding perpendicular coordinate axes for your data.

### 5. Cholesky Decomposition

**Factorization (for symmetric positive definite A):**
```
A = LLᵀ
```

Where:
- **L** (n × n): Lower triangular matrix with positive diagonal entries

**Requirement:** A must be symmetric positive definite (all eigenvalues > 0)

**Algorithm:**
```
For j = 1 to n:
  L[j,j] = sqrt(A[j,j] - Σₖ₌₁ʲ⁻¹ L[j,k]²)
  For i = j+1 to n:
    L[i,j] = (A[i,j] - Σₖ₌₁ʲ⁻¹ L[i,k]×L[j,k]) / L[j,j]
```

**Solving Ax = b using Cholesky:**
```
Step 1: Solve Ly = b (forward substitution)
Step 2: Solve Lᵀx = y (backward substitution)
```

**Computational Complexity:** O(n³/3) (twice as fast as LU!)

**When to Use:**
- Covariance matrices (always symmetric positive definite)
- Normal equations in least squares: AᵀAx = Aᵀb
- Gaussian Process regression: K + σ²I
- Quadratic optimization: ½xᵀQx + cᵀx

**For Beginners:** Cholesky is like taking the square root of a matrix. Just as 9 = 3 × 3, Cholesky finds L such that A = L × Lᵀ. It only works for special "nice" matrices (symmetric positive definite), but when it works, it's the fastest decomposition.

### 6. Eigen Decomposition

**Factorization (for square A):**
```
A = VΛVᵀ  (if A is symmetric)
A = VΛV⁻¹ (general case)
```

Where:
- **V** (n × n): Matrix of eigenvectors (columns are eigenvectors)
- **Λ** (n × n): Diagonal matrix of eigenvalues

**Eigenvalue Equation:**
```
Av = λv
where v is an eigenvector, λ is the corresponding eigenvalue
```

**Properties:**
1. **Trace:** tr(A) = Σᵢ λᵢ
2. **Determinant:** det(A) = Πᵢ λᵢ
3. **Powers:** Aⁿ = VΛⁿVᵀ (fast matrix exponentiation)
4. **Invertibility:** A is invertible iff all λᵢ ≠ 0

**Algorithms:**
1. **Power Iteration:** Find largest eigenvalue
2. **QR Iteration:** Find all eigenvalues iteratively
3. **Jacobi Method:** For symmetric matrices (uses Givens rotations)
4. **Divide and Conquer:** Fast for symmetric tridiagonal matrices

**Applications:**
1. **Principal Component Analysis (PCA):** Eigendecomposition of covariance matrix
2. **Stability Analysis:** Eigenvalues determine system stability (control theory)
3. **Google PageRank:** Dominant eigenvector of web link matrix
4. **Quantum Mechanics:** Eigenvalues are observable values, eigenvectors are states

**For Beginners:** Eigendecomposition finds the "natural directions" of a transformation. If A represents a transformation (like stretching/rotating), eigenvectors are directions that only get scaled (not rotated), and eigenvalues tell you the scaling factor.

### 7. Independent Component Analysis (ICA)

**Problem:** Given mixed signals, separate them into independent sources.

**Model:**
```
X = AS
where:
  X (n × m): Observed mixed signals (n sensors, m time points)
  A (n × d): Mixing matrix (unknown)
  S (d × m): Independent source signals (unknown, to be estimated)
```

**Goal:** Find unmixing matrix W such that S ≈ WX

**Key Assumption:** Source signals are statistically independent (not just uncorrelated like PCA)

**Algorithm (FastICA - most common):**
```
1. Center data: X ← X - mean(X)
2. Whiten data: X ← whitening_matrix × X
   (makes X have identity covariance)
3. For each component:
   a. Initialize random weight vector w
   b. Update: w ← E[X × g(wᵀX)] - E[g'(wᵀX)]×w
      where g is a non-linearity (e.g., g(u) = tanh(u))
   c. Normalize: w ← w / ||w||
   d. Orthogonalize against previously found components
   e. Repeat until convergence
```

**Non-linearity Functions:**
1. **Logistic:** g(u) = tanh(u), g'(u) = 1 - tanh²(u)
2. **Exponential:** g(u) = u×exp(-u²/2), g'(u) = (1 - u²)×exp(-u²/2)
3. **Cubic:** g(u) = u³, g'(u) = 3u²

**Applications:**
1. **Cocktail Party Problem:** Separate multiple speakers from microphone recordings
2. **EEG/MEG Analysis:** Separate brain signals from scalp recordings
3. **Financial Data:** Find independent market factors
4. **Image Processing:** Separate mixed images (e.g., MRI scans with different contrasts)

**For Beginners:** ICA is like unmixing audio recordings at a party. If you have 3 microphones recording 3 people talking simultaneously, ICA can separate out each person's voice into 3 clean audio tracks. The key is that the sources (people talking) are independent.

**Contrast with PCA:**
- PCA finds uncorrelated components (second-order statistics)
- ICA finds independent components (higher-order statistics)
- PCA cares about variance, ICA cares about non-Gaussianity

---

## Implementation Guide

### Existing Implementations in AiDotNet

**Already Implemented:**
- ✅ `SvdDecomposition.cs` - Singular Value Decomposition
- ✅ `LuDecomposition.cs` - LU Decomposition
- ✅ `QrDecomposition.cs` - QR Decomposition
- ✅ `CholeskyDecomposition.cs` - Cholesky Decomposition
- ✅ `EigenDecomposition.cs` - Eigen Decomposition

**To Be Implemented:**
- ❌ `NMF.cs` - Non-Negative Matrix Factorization
- ❌ `ICA.cs` - Independent Component Analysis

Let me verify what exists and what needs to be implemented.

### Phase 1: Verify and Enhance Existing Decompositions

#### Task 1.1: Review SVD Implementation

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\DecompositionMethods\MatrixDecomposition\SvdDecomposition.cs`

**What to Check:**
1. **Multiple Algorithms:** Does it support different SVD algorithms?
   - ✅ Existing: GolubReinsch, Jacobi, Randomized, PowerIteration, TruncatedSVD, DividedAndConquer
   - Good! No changes needed.

2. **Numerical Stability:**
   - Check for jitter/regularization to handle near-singular matrices
   - Verify Cholesky decomposition is used (not direct inversion)

3. **Tests:** Ensure comprehensive test coverage
   - Test with ill-conditioned matrices
   - Test with various sizes (small, medium, large)
   - Test all algorithm variants

**Potential Enhancements:**
```csharp
// Add method for truncated SVD with specified rank
public (Matrix<T> U, Vector<T> S, Matrix<T> Vt) ComputeTruncatedSVD(Matrix<T> matrix, int rank)
{
    // Only compute top 'rank' singular values/vectors
    // More efficient than full SVD for large matrices
}

// Add method to compute pseudoinverse
public Matrix<T> ComputePseudoinverse(T tolerance = default)
{
    // A+ = V × Σ+ × U^T
    // where Σ+ inverts non-zero singular values (> tolerance)
}
```

#### Task 1.2: Review Other Existing Decompositions

Verify that `LuDecomposition.cs`, `QrDecomposition.cs`, `CholeskyDecomposition.cs`, and `EigenDecomposition.cs` follow NumOps patterns and have adequate tests.

**Checklist for Each:**
- [ ] Uses `INumericOperations<T>` throughout (no hardcoded doubles)
- [ ] Proper error handling (e.g., Cholesky checks for positive definiteness)
- [ ] Comprehensive unit tests (multiple numeric types, edge cases)
- [ ] XML documentation with "For Beginners" sections
- [ ] Implements `IMatrixDecomposition<T>` interface

### Phase 2: Implement Non-Negative Matrix Factorization (NMF)

#### Step 2.1: Create NMF Interface

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\INonNegativeMatrixFactorization.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for Non-Negative Matrix Factorization (NMF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> NMF decomposes a non-negative matrix into two non-negative factor matrices.
///
/// Unlike SVD which can have negative components, NMF constraints all values to be non-negative.
/// This makes the decomposition more interpretable - think of it as finding "parts" that add up
/// to make the whole, rather than "parts" that can add or subtract.
///
/// Example applications:
/// - Topic modeling: Find topics (word distributions) in documents
/// - Image processing: Find parts-based representations of faces
/// - Recommender systems: Find latent factors (user preferences, item features)
///
/// Mathematical formulation:
///   A ≈ W × H
///   where A, W, H ≥ 0 (all non-negative)
/// </remarks>
public interface INonNegativeMatrixFactorization<T>
{
    /// <summary>
    /// Gets the basis matrix W (m × k).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The W matrix contains the basis vectors (or "parts").
    ///
    /// For documents: Each column of W is a topic (distribution over words)
    /// For images: Each column of W is a basis image (e.g., a facial feature)
    /// For recommender systems: Each column represents a latent factor
    ///
    /// Dimensions:
    /// - Rows = number of features (e.g., vocabulary size, pixels)
    /// - Columns = number of components (user-specified k)
    /// </remarks>
    Matrix<T> W { get; }

    /// <summary>
    /// Gets the coefficient matrix H (k × n).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The H matrix contains the weights/coefficients.
    ///
    /// For documents: Each column of H is a document's distribution over topics
    /// For images: Each column specifies how to combine basis images
    /// For recommender systems: Each column represents an item's latent features
    ///
    /// Dimensions:
    /// - Rows = number of components (same k as W)
    /// - Columns = number of samples (e.g., documents, images, items)
    /// </remarks>
    Matrix<T> H { get; }

    /// <summary>
    /// Gets the reconstruction error after factorization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This measures how well W×H approximates the original matrix A.
    ///
    /// Computed as: ||A - W×H||²_F (Frobenius norm)
    ///
    /// Lower values indicate better approximation.
    /// If error is too high, try increasing the number of components (k).
    /// </remarks>
    T ReconstructionError { get; }

    /// <summary>
    /// Factorizes a non-negative matrix into W and H matrices.
    /// </summary>
    /// <param name="matrix">The non-negative matrix to factorize (m × n).</param>
    /// <param name="nComponents">The number of components (k) to extract.</param>
    /// <param name="maxIterations">Maximum number of iterations for the algorithm.</param>
    /// <param name="tolerance">Convergence tolerance (stop if change in error < tolerance).</param>
    /// <returns>Tuple containing W (m × k) and H (k × n) matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method performs the actual factorization.
    ///
    /// Parameters:
    /// - matrix: Your data (all values must be ≥ 0)
    /// - nComponents: How many "parts" to extract (typically much smaller than matrix dimensions)
    /// - maxIterations: How long to run the optimization (default: 200)
    /// - tolerance: When to stop (if improvement < tolerance, stop early)
    ///
    /// Returns:
    /// - W: Basis matrix (the "parts")
    /// - H: Coefficient matrix (how to combine the parts)
    ///
    /// After calling this method, you can reconstruct the original:
    ///   A_reconstructed = W × H
    ///
    /// Example:
    /// ```csharp
    /// var nmf = new NMF<double>();
    /// var (W, H) = nmf.Decompose(documentTermMatrix, nComponents: 10);
    /// // W now contains 10 topics, H contains document-topic weights
    /// ```
    /// </remarks>
    (Matrix<T> W, Matrix<T> H) Decompose(
        Matrix<T> matrix,
        int nComponents,
        int maxIterations = 200,
        T? tolerance = default);
}
```

#### Step 2.2: Implement NMF Class

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\DecompositionMethods\MatrixDecomposition\NMF.cs`

```csharp
namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Non-Negative Matrix Factorization using multiplicative update rules.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// NMF finds an approximate factorization A ≈ W × H where all matrices are non-negative.
/// This implementation uses the multiplicative update rules from Lee & Seung (2001).
/// </para>
/// <para>
/// <b>For Beginners:</b> NMF is like finding LEGO blocks (W) and instructions (H) that
/// build your original structure (A).
///
/// Imagine you have 1000 document descriptions:
/// - Each document is a bag of words (term frequencies)
/// - You want to find 10 topics (e.g., "sports", "politics", "technology")
/// - W will contain the word distributions for each topic
/// - H will contain each document's mixture of topics
///
/// The algorithm:
/// 1. Initialize W and H randomly (all positive)
/// 2. Iteratively update W and H to minimize ||A - W×H||²
/// 3. Stop when the error stops decreasing significantly
///
/// Default values (based on scikit-learn):
/// - Initialization: Random uniform [0, √(A.mean()/k)]
/// - Max iterations: 200
/// - Tolerance: 1e-4
/// - Beta loss: Frobenius norm (equivalent to squared error)
/// </para>
/// </remarks>
public class NMF<T> : INonNegativeMatrixFactorization<T>
{
    private readonly INumericOperations<T> _numOps;

    public Matrix<T> W { get; private set; } = new Matrix<T>(0, 0);
    public Matrix<T> H { get; private set; } = new Matrix<T>(0, 0);
    public T ReconstructionError { get; private set; }

    /// <summary>
    /// Initializes a new instance of NMF.
    /// </summary>
    public NMF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        ReconstructionError = _numOps.Zero;
    }

    public (Matrix<T> W, Matrix<T> H) Decompose(
        Matrix<T> matrix,
        int nComponents,
        int maxIterations = 200,
        T? tolerance = default)
    {
        T tol = tolerance ?? _numOps.FromDouble(1e-4);

        // Validate input
        ValidateInput(matrix, nComponents);

        int m = matrix.Rows;    // Number of features
        int n = matrix.Columns; // Number of samples
        int k = nComponents;    // Number of components

        // Initialize W and H with non-negative random values
        W = InitializeMatrix(m, k, matrix);
        H = InitializeMatrix(k, n, matrix);

        T previousError = _numOps.FromDouble(double.MaxValue);

        // Multiplicative update iterations
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Update H: H ← H ⊙ [(W^T × A) / (W^T × W × H)]
            UpdateH(matrix);

            // Update W: W ← W ⊙ [(A × H^T) / (W × H × H^T)]
            UpdateW(matrix);

            // Compute reconstruction error: ||A - W×H||²_F
            T currentError = ComputeReconstructionError(matrix);

            // Check convergence
            T errorChange = _numOps.Abs(_numOps.Subtract(previousError, currentError));
            if (_numOps.LessThan(errorChange, tol))
            {
                // Converged
                break;
            }

            previousError = currentError;
        }

        ReconstructionError = ComputeReconstructionError(matrix);

        return (W, H);
    }

    /// <summary>
    /// Validates that the input matrix is non-negative and dimensions are valid.
    /// </summary>
    private void ValidateInput(Matrix<T> matrix, int nComponents)
    {
        // Check for negative values
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                if (_numOps.LessThan(matrix[i, j], _numOps.Zero))
                {
                    throw new ArgumentException(
                        $"NMF requires all matrix entries to be non-negative. " +
                        $"Found negative value at ({i}, {j}): {matrix[i, j]}");
                }
            }
        }

        // Check dimensions
        if (nComponents <= 0 || nComponents > Math.Min(matrix.Rows, matrix.Columns))
        {
            throw new ArgumentException(
                $"Number of components must be positive and at most min(m, n). " +
                $"Got nComponents={nComponents} for matrix of size ({matrix.Rows}, {matrix.Columns})");
        }
    }

    /// <summary>
    /// Initializes a matrix with non-negative random values.
    /// </summary>
    /// <remarks>
    /// Uses initialization from scikit-learn:
    /// Values drawn from uniform distribution [0, √(A.mean() / k)]
    /// This tends to give good convergence properties.
    /// </remarks>
    private Matrix<T> InitializeMatrix(int rows, int cols, Matrix<T> original)
    {
        // Compute mean of original matrix
        T sum = _numOps.Zero;
        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                sum = _numOps.Add(sum, original[i, j]);
            }
        }
        T mean = _numOps.Divide(sum, _numOps.FromDouble(original.Rows * original.Columns));

        // Compute scale: sqrt(mean / k)
        T scale = _numOps.Sqrt(_numOps.Divide(mean, _numOps.FromDouble(cols)));

        // Initialize with random values
        var random = new Random();
        var result = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Random value in [0, scale]
                double randomValue = random.NextDouble() * Convert.ToDouble(scale);
                result[i, j] = _numOps.FromDouble(randomValue);
            }
        }

        return result;
    }

    /// <summary>
    /// Updates matrix H using multiplicative update rule.
    /// </summary>
    /// <remarks>
    /// Update rule: H ← H ⊙ [(W^T × A) / (W^T × W × H + ε)]
    /// where ⊙ is element-wise multiplication and ε prevents division by zero
    /// </remarks>
    private void UpdateH(Matrix<T> A)
    {
        T epsilon = _numOps.FromDouble(1e-10); // Numerical stability

        // Compute numerator: W^T × A
        Matrix<T> WT = W.Transpose();
        Matrix<T> numerator = WT.Multiply(A);

        // Compute denominator: W^T × W × H
        Matrix<T> WTW = WT.Multiply(W);
        Matrix<T> denominator = WTW.Multiply(H);

        // Update H element-wise: H[i,j] *= numerator[i,j] / (denominator[i,j] + ε)
        for (int i = 0; i < H.Rows; i++)
        {
            for (int j = 0; j < H.Columns; j++)
            {
                T ratio = _numOps.Divide(
                    numerator[i, j],
                    _numOps.Add(denominator[i, j], epsilon));

                H[i, j] = _numOps.Multiply(H[i, j], ratio);

                // Ensure non-negativity (numerical errors might make values slightly negative)
                if (_numOps.LessThan(H[i, j], _numOps.Zero))
                {
                    H[i, j] = _numOps.Zero;
                }
            }
        }
    }

    /// <summary>
    /// Updates matrix W using multiplicative update rule.
    /// </summary>
    /// <remarks>
    /// Update rule: W ← W ⊙ [(A × H^T) / (W × H × H^T + ε)]
    /// </remarks>
    private void UpdateW(Matrix<T> A)
    {
        T epsilon = _numOps.FromDouble(1e-10);

        // Compute numerator: A × H^T
        Matrix<T> HT = H.Transpose();
        Matrix<T> numerator = A.Multiply(HT);

        // Compute denominator: W × H × H^T
        Matrix<T> WH = W.Multiply(H);
        Matrix<T> denominator = WH.Multiply(HT);

        // Update W element-wise
        for (int i = 0; i < W.Rows; i++)
        {
            for (int j = 0; j < W.Columns; j++)
            {
                T ratio = _numOps.Divide(
                    numerator[i, j],
                    _numOps.Add(denominator[i, j], epsilon));

                W[i, j] = _numOps.Multiply(W[i, j], ratio);

                // Ensure non-negativity
                if (_numOps.LessThan(W[i, j], _numOps.Zero))
                {
                    W[i, j] = _numOps.Zero;
                }
            }
        }
    }

    /// <summary>
    /// Computes the Frobenius norm of (A - W×H).
    /// </summary>
    private T ComputeReconstructionError(Matrix<T> A)
    {
        // Compute W × H
        Matrix<T> WH = W.Multiply(H);

        // Compute ||A - W×H||²_F = Σᵢⱼ (Aᵢⱼ - (W×H)ᵢⱼ)²
        T error = _numOps.Zero;
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = 0; j < A.Columns; j++)
            {
                T diff = _numOps.Subtract(A[i, j], WH[i, j]);
                error = _numOps.Add(error, _numOps.Multiply(diff, diff));
            }
        }

        return _numOps.Sqrt(error); // Return Frobenius norm (not squared)
    }

    /// <summary>
    /// Reconstructs the original matrix from W and H.
    /// </summary>
    /// <returns>Reconstructed matrix A_approx = W × H</returns>
    public Matrix<T> Reconstruct()
    {
        return W.Multiply(H);
    }

    /// <summary>
    /// Transforms new data using the learned H matrix basis.
    /// </summary>
    /// <param name="newData">New data matrix (same number of features as training data).</param>
    /// <returns>Transformed representation in the NMF space (k × newData.Columns).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Once you've learned the basis W, you can project new data
    /// into the same space by finding the best H that reconstructs it.
    ///
    /// This is useful for:
    /// - Applying learned topics to new documents
    /// - Encoding new images using learned basis images
    /// - Making predictions on new data using the learned factors
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> newData)
    {
        // Find H_new such that newData ≈ W × H_new
        // Use non-negative least squares or multiplicative updates
        // (Simplified implementation: just fit H with fixed W)

        int k = W.Columns;
        int n_new = newData.Columns;
        Matrix<T> H_new = InitializeMatrix(k, n_new, newData);

        // Run a few iterations of H updates only
        for (int iter = 0; iter < 100; iter++)
        {
            // Same update rule as UpdateH, but with newData
            T epsilon = _numOps.FromDouble(1e-10);
            Matrix<T> WT = W.Transpose();
            Matrix<T> numerator = WT.Multiply(newData);
            Matrix<T> WTW = WT.Multiply(W);
            Matrix<T> denominator = WTW.Multiply(H_new);

            for (int i = 0; i < H_new.Rows; i++)
            {
                for (int j = 0; j < H_new.Columns; j++)
                {
                    T ratio = _numOps.Divide(
                        numerator[i, j],
                        _numOps.Add(denominator[i, j], epsilon));
                    H_new[i, j] = _numOps.Multiply(H_new[i, j], ratio);

                    if (_numOps.LessThan(H_new[i, j], _numOps.Zero))
                    {
                        H_new[i, j] = _numOps.Zero;
                    }
                }
            }
        }

        return H_new;
    }
}
```

### Phase 3: Implement Independent Component Analysis (ICA)

#### Step 3.1: Create ICA Interface

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IIndependentComponentAnalysis.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for Independent Component Analysis (ICA).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> ICA separates mixed signals into independent source signals.
///
/// Classic example - Cocktail Party Problem:
/// - You have 3 microphones recording 3 people talking simultaneously
/// - Each microphone records a mixture of all 3 voices
/// - ICA separates the recordings back into 3 clean audio tracks (one per person)
///
/// Key difference from PCA:
/// - PCA finds uncorrelated components (second-order independence)
/// - ICA finds statistically independent components (higher-order independence)
/// - PCA is linear and orthogonal; ICA is linear but not orthogonal
///
/// Applications:
/// - Audio source separation (cocktail party problem)
/// - EEG/MEG signal processing (separate brain sources from scalp recordings)
/// - Financial data analysis (find independent market factors)
/// - Image separation (e.g., separate mixed photos)
/// </remarks>
public interface IIndependentComponentAnalysis<T>
{
    /// <summary>
    /// Gets the unmixing matrix W.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The unmixing matrix W transforms mixed signals into sources.
    ///
    /// If X are the mixed observations and S are the independent sources:
    ///   S = W × X
    ///
    /// W is learned by the ICA algorithm to maximize independence of S.
    /// </remarks>
    Matrix<T> UnmixingMatrix { get; }

    /// <summary>
    /// Gets the mixing matrix A (inverse of W, if available).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The mixing matrix A shows how sources combine to form observations.
    ///
    /// If S are the sources and X are the observations:
    ///   X = A × S
    ///
    /// This is the inverse transformation of W:
    ///   A = W⁻¹
    /// </remarks>
    Matrix<T> MixingMatrix { get; }

    /// <summary>
    /// Gets the independent components (sources) extracted from the data.
    /// </summary>
    Matrix<T> Sources { get; }

    /// <summary>
    /// Performs ICA to separate mixed signals into independent components.
    /// </summary>
    /// <param name="mixedSignals">
    /// Mixed signal matrix (d × m) where:
    ///   - d = number of signals/sensors
    ///   - m = number of time points/samples
    /// </param>
    /// <param name="nComponents">
    /// Number of independent components to extract.
    /// Typically equals the number of signals (d), but can be less for dimensionality reduction.
    /// </param>
    /// <param name="maxIterations">Maximum iterations for FastICA algorithm (default: 200).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-4).</param>
    /// <returns>Matrix of independent components (nComponents × m).</returns>
    Matrix<T> Decompose(
        Matrix<T> mixedSignals,
        int nComponents,
        int maxIterations = 200,
        T? tolerance = default);

    /// <summary>
    /// Transforms new mixed signals into independent components using the learned unmixing matrix.
    /// </summary>
    /// <param name="newMixedSignals">New mixed signals (d × m_new).</param>
    /// <returns>Independent components (nComponents × m_new).</returns>
    Matrix<T> Transform(Matrix<T> newMixedSignals);
}
```

#### Step 3.2: Implement ICA Class

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\DecompositionMethods\MatrixDecomposition\ICA.cs`

```csharp
namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Independent Component Analysis using the FastICA algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FastICA is an efficient algorithm for ICA based on maximizing non-Gaussianity.
/// It uses a fixed-point iteration scheme with a non-linear function to find independent components.
/// </para>
/// <para>
/// <b>For Beginners:</b> FastICA works in three main steps:
///
/// 1. **Centering**: Subtract the mean from each signal (zero mean)
/// 2. **Whitening**: Transform data so it has identity covariance (uncorrelated, unit variance)
/// 3. **Fixed-point iteration**: Find directions that maximize non-Gaussianity
///
/// The key insight: Independent signals are maximally non-Gaussian. By finding directions
/// of maximum non-Gaussianity, we recover the independent sources.
///
/// Default parameters (based on scikit-learn):
/// - Algorithm: parallel (extract all components simultaneously)
/// - Non-linearity: logcosh (good balance of robustness and speed)
/// - Max iterations: 200
/// - Tolerance: 1e-4
/// </para>
/// </remarks>
public class ICA<T> : IIndependentComponentAnalysis<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly NonLinearityFunction _nonLinearity;

    public Matrix<T> UnmixingMatrix { get; private set; } = new Matrix<T>(0, 0);
    public Matrix<T> MixingMatrix { get; private set; } = new Matrix<T>(0, 0);
    public Matrix<T> Sources { get; private set; } = new Matrix<T>(0, 0);

    private Vector<T> _mean = new Vector<T>(0);
    private Matrix<T> _whiteningMatrix = new Matrix<T>(0, 0);

    /// <summary>
    /// Enum for selecting the non-linearity function in FastICA.
    /// </summary>
    public enum NonLinearityFunction
    {
        /// <summary>
        /// Logistic/hyperbolic tangent: g(u) = tanh(u)
        /// Good for general purposes, robust.
        /// </summary>
        Logcosh,

        /// <summary>
        /// Exponential: g(u) = u × exp(-u²/2)
        /// Good for super-Gaussian (heavy-tailed) sources.
        /// </summary>
        Exponential,

        /// <summary>
        /// Cubic: g(u) = u³
        /// Good for sub-Gaussian (light-tailed) sources, but less robust.
        /// </summary>
        Cubic
    }

    /// <summary>
    /// Initializes a new instance of ICA.
    /// </summary>
    /// <param name="nonLinearity">
    /// The non-linearity function to use (default: Logcosh).
    /// Logcosh is a good default for most applications.
    /// </param>
    public ICA(NonLinearityFunction nonLinearity = NonLinearityFunction.Logcosh)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _nonLinearity = nonLinearity;
    }

    public Matrix<T> Decompose(
        Matrix<T> mixedSignals,
        int nComponents,
        int maxIterations = 200,
        T? tolerance = default)
    {
        T tol = tolerance ?? _numOps.FromDouble(1e-4);

        int d = mixedSignals.Rows;    // Number of signals
        int m = mixedSignals.Columns; // Number of time points

        if (nComponents <= 0 || nComponents > d)
        {
            throw new ArgumentException(
                $"Number of components must be between 1 and {d}. Got: {nComponents}");
        }

        // Step 1: Center the data (subtract mean)
        Matrix<T> X_centered = CenterData(mixedSignals);

        // Step 2: Whiten the data
        Matrix<T> X_whitened = WhitenData(X_centered);

        // Step 3: Run FastICA to find unmixing matrix
        Matrix<T> W = RunFastICA(X_whitened, nComponents, maxIterations, tol);

        // Step 4: Compute sources: S = W × X_whitened
        Sources = W.Multiply(X_whitened);

        // Step 5: Compute full unmixing matrix: W_full = W × whitening_matrix
        UnmixingMatrix = W.Multiply(_whiteningMatrix);

        // Step 6: Compute mixing matrix: A = W⁻¹
        MixingMatrix = UnmixingMatrix.Invert();

        return Sources;
    }

    public Matrix<T> Transform(Matrix<T> newMixedSignals)
    {
        // Apply the same preprocessing and unmixing
        // 1. Center using stored mean
        Matrix<T> centered = new Matrix<T>(newMixedSignals.Rows, newMixedSignals.Columns);
        for (int i = 0; i < newMixedSignals.Rows; i++)
        {
            for (int j = 0; j < newMixedSignals.Columns; j++)
            {
                centered[i, j] = _numOps.Subtract(newMixedSignals[i, j], _mean[i]);
            }
        }

        // 2. Whiten using stored whitening matrix
        Matrix<T> whitened = _whiteningMatrix.Multiply(centered);

        // 3. Apply unmixing
        // Since UnmixingMatrix = W × WhiteningMatrix, we need to extract W
        // For simplicity, directly apply unmixing to centered data
        return UnmixingMatrix.Multiply(centered);
    }

    /// <summary>
    /// Centers the data by subtracting the mean of each signal.
    /// </summary>
    private Matrix<T> CenterData(Matrix<T> X)
    {
        int d = X.Rows;
        int m = X.Columns;

        // Compute mean for each signal
        _mean = new Vector<T>(d);
        for (int i = 0; i < d; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < m; j++)
            {
                sum = _numOps.Add(sum, X[i, j]);
            }
            _mean[i] = _numOps.Divide(sum, _numOps.FromDouble(m));
        }

        // Subtract mean
        Matrix<T> X_centered = new Matrix<T>(d, m);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < m; j++)
            {
                X_centered[i, j] = _numOps.Subtract(X[i, j], _mean[i]);
            }
        }

        return X_centered;
    }

    /// <summary>
    /// Whitens the data so it has identity covariance matrix.
    /// </summary>
    private Matrix<T> WhitenData(Matrix<T> X_centered)
    {
        int d = X_centered.Rows;
        int m = X_centered.Columns;

        // Compute covariance matrix: C = (1/m) × X × Xᵀ
        Matrix<T> Xt = X_centered.Transpose();
        Matrix<T> XXt = X_centered.Multiply(Xt);

        Matrix<T> covariance = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                covariance[i, j] = _numOps.Divide(XXt[i, j], _numOps.FromDouble(m));
            }
        }

        // Eigen decomposition: C = V × Λ × Vᵀ
        var eigen = new EigenDecomposition<T>(covariance);
        Matrix<T> V = eigen.Eigenvectors;
        Vector<T> eigenvalues = eigen.Eigenvalues;

        // Compute whitening matrix: W = Λ^(-1/2) × Vᵀ
        // This transforms data to have identity covariance
        Matrix<T> Lambda_inv_sqrt = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            T sqrt_lambda = _numOps.Sqrt(eigenvalues[i]);
            T inv_sqrt_lambda = _numOps.Divide(_numOps.One, sqrt_lambda);
            Lambda_inv_sqrt[i, i] = inv_sqrt_lambda;
        }

        Matrix<T> Vt = V.Transpose();
        _whiteningMatrix = Lambda_inv_sqrt.Multiply(Vt);

        // Apply whitening: X_whitened = W × X_centered
        return _whiteningMatrix.Multiply(X_centered);
    }

    /// <summary>
    /// Runs the FastICA algorithm to find the unmixing matrix W.
    /// </summary>
    private Matrix<T> RunFastICA(
        Matrix<T> X_whitened,
        int nComponents,
        int maxIterations,
        T tolerance)
    {
        int d = X_whitened.Rows;
        int m = X_whitened.Columns;

        Matrix<T> W = new Matrix<T>(nComponents, d);

        // Extract each component sequentially
        for (int comp = 0; comp < nComponents; comp++)
        {
            // Initialize weight vector randomly
            Vector<T> w = InitializeRandomVector(d);

            // Orthogonalize against previously found components
            if (comp > 0)
            {
                w = Orthogonalize(w, W, comp);
            }

            // Normalize
            w = Normalize(w);

            // Fixed-point iteration
            for (int iter = 0; iter < maxIterations; iter++)
            {
                Vector<T> w_old = w.Clone();

                // Compute w_new = E[X × g(wᵀX)] - E[g'(wᵀX)] × w
                w = FixedPointUpdate(w, X_whitened);

                // Orthogonalize against previously found components
                if (comp > 0)
                {
                    w = Orthogonalize(w, W, comp);
                }

                // Normalize
                w = Normalize(w);

                // Check convergence: |w · w_old| ≈ 1
                T dot_product = w.DotProduct(w_old);
                T diff = _numOps.Abs(_numOps.Subtract(_numOps.Abs(dot_product), _numOps.One));

                if (_numOps.LessThan(diff, tolerance))
                {
                    break; // Converged
                }
            }

            // Store converged component
            for (int i = 0; i < d; i++)
            {
                W[comp, i] = w[i];
            }
        }

        return W;
    }

    /// <summary>
    /// Initializes a random unit vector.
    /// </summary>
    private Vector<T> InitializeRandomVector(int dimension)
    {
        var random = new Random();
        Vector<T> v = new Vector<T>(dimension);

        for (int i = 0; i < dimension; i++)
        {
            v[i] = _numOps.FromDouble(random.NextDouble() - 0.5);
        }

        return Normalize(v);
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// </summary>
    private Vector<T> Normalize(Vector<T> v)
    {
        T norm = _numOps.Sqrt(v.DotProduct(v));
        Vector<T> normalized = new Vector<T>(v.Length);

        for (int i = 0; i < v.Length; i++)
        {
            normalized[i] = _numOps.Divide(v[i], norm);
        }

        return normalized;
    }

    /// <summary>
    /// Orthogonalizes vector w against previously found components.
    /// </summary>
    private Vector<T> Orthogonalize(Vector<T> w, Matrix<T> W, int currentComponent)
    {
        Vector<T> w_orth = w.Clone();

        // Gram-Schmidt: w_orth = w - Σⱼ (wⱼᵀ w) wⱼ
        for (int j = 0; j < currentComponent; j++)
        {
            Vector<T> wj = W.GetRow(j);
            T projection = wj.DotProduct(w);

            for (int i = 0; i < w.Length; i++)
            {
                w_orth[i] = _numOps.Subtract(w_orth[i], _numOps.Multiply(projection, wj[i]));
            }
        }

        return w_orth;
    }

    /// <summary>
    /// Performs one fixed-point update iteration.
    /// </summary>
    private Vector<T> FixedPointUpdate(Vector<T> w, Matrix<T> X)
    {
        int m = X.Columns; // Number of samples
        int d = X.Rows;    // Dimension

        // Compute wᵀX for all samples
        Vector<T> wtX = new Vector<T>(m);
        for (int j = 0; j < m; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < d; i++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(w[i], X[i, j]));
            }
            wtX[j] = sum;
        }

        // Compute g(wᵀX) and g'(wᵀX) based on non-linearity
        Vector<T> g_wtX = new Vector<T>(m);
        Vector<T> g_prime_wtX = new Vector<T>(m);

        for (int j = 0; j < m; j++)
        {
            (g_wtX[j], g_prime_wtX[j]) = ApplyNonLinearity(wtX[j]);
        }

        // Compute E[X × g(wᵀX)]
        Vector<T> term1 = new Vector<T>(d);
        for (int i = 0; i < d; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < m; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(X[i, j], g_wtX[j]));
            }
            term1[i] = _numOps.Divide(sum, _numOps.FromDouble(m));
        }

        // Compute E[g'(wᵀX)]
        T mean_g_prime = _numOps.Zero;
        for (int j = 0; j < m; j++)
        {
            mean_g_prime = _numOps.Add(mean_g_prime, g_prime_wtX[j]);
        }
        mean_g_prime = _numOps.Divide(mean_g_prime, _numOps.FromDouble(m));

        // Compute w_new = E[X × g(wᵀX)] - E[g'(wᵀX)] × w
        Vector<T> w_new = new Vector<T>(d);
        for (int i = 0; i < d; i++)
        {
            T term2_i = _numOps.Multiply(mean_g_prime, w[i]);
            w_new[i] = _numOps.Subtract(term1[i], term2_i);
        }

        return w_new;
    }

    /// <summary>
    /// Applies the selected non-linearity function.
    /// </summary>
    /// <returns>Tuple of (g(u), g'(u))</returns>
    private (T g, T g_prime) ApplyNonLinearity(T u)
    {
        double u_double = Convert.ToDouble(u);

        switch (_nonLinearity)
        {
            case NonLinearityFunction.Logcosh:
                // g(u) = tanh(u)
                // g'(u) = 1 - tanh²(u)
                double tanh_u = Math.Tanh(u_double);
                return (
                    _numOps.FromDouble(tanh_u),
                    _numOps.FromDouble(1.0 - tanh_u * tanh_u)
                );

            case NonLinearityFunction.Exponential:
                // g(u) = u × exp(-u²/2)
                // g'(u) = (1 - u²) × exp(-u²/2)
                double exp_term = Math.Exp(-u_double * u_double / 2.0);
                return (
                    _numOps.FromDouble(u_double * exp_term),
                    _numOps.FromDouble((1.0 - u_double * u_double) * exp_term)
                );

            case NonLinearityFunction.Cubic:
                // g(u) = u³
                // g'(u) = 3u²
                return (
                    _numOps.FromDouble(u_double * u_double * u_double),
                    _numOps.FromDouble(3.0 * u_double * u_double)
                );

            default:
                throw new InvalidOperationException($"Unknown non-linearity: {_nonLinearity}");
        }
    }
}
```

---

## Testing Strategy

### Unit Tests for NMF

**File:** `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\DecompositionMethods\MatrixDecompositionTests.cs`

```csharp
[TestMethod]
public void TestNMF_ReconstructionQuality()
{
    // Create a simple non-negative matrix
    var A = new Matrix<double>(new double[,]
    {
        { 1, 2, 3, 4 },
        { 2, 4, 6, 8 },
        { 3, 6, 9, 12 }
    });

    var nmf = new NMF<double>();
    var (W, H) = nmf.Decompose(A, nComponents: 2, maxIterations: 500);

    // Reconstruct
    Matrix<double> A_reconstructed = W.Multiply(H);

    // Assert: Reconstruction error is small
    double error = 0;
    for (int i = 0; i < A.Rows; i++)
    {
        for (int j = 0; j < A.Columns; j++)
        {
            double diff = A[i, j] - A_reconstructed[i, j];
            error += diff * diff;
        }
    }
    error = Math.Sqrt(error);

    Assert.IsTrue(error < 1.0, $"Reconstruction error too large: {error}");
}

[TestMethod]
public void TestNMF_NonNegativityConstraint()
{
    var A = new Matrix<double>(100, 50);
    var random = new Random();
    for (int i = 0; i < A.Rows; i++)
    {
        for (int j = 0; j < A.Columns; j++)
        {
            A[i, j] = random.NextDouble() * 10;
        }
    }

    var nmf = new NMF<double>();
    var (W, H) = nmf.Decompose(A, nComponents: 10);

    // Assert: All entries in W and H are non-negative
    for (int i = 0; i < W.Rows; i++)
    {
        for (int j = 0; j < W.Columns; j++)
        {
            Assert.IsTrue(W[i, j] >= 0, $"W[{i},{j}] is negative: {W[i, j]}");
        }
    }

    for (int i = 0; i < H.Rows; i++)
    {
        for (int j = 0; j < H.Columns; j++)
        {
            Assert.IsTrue(H[i, j] >= 0, $"H[{i},{j}] is negative: {H[i, j]}");
        }
    }
}

[TestMethod]
[ExpectedException(typeof(ArgumentException))]
public void TestNMF_RejectsNegativeValues()
{
    var A = new Matrix<double>(10, 10);
    A[5, 5] = -1.0; // Insert negative value

    var nmf = new NMF<double>();
    nmf.Decompose(A, nComponents: 3); // Should throw
}

[TestMethod]
public void TestNMF_TopicModeling()
{
    // Simulate document-term matrix
    // 20 documents, 50 words, 3 underlying topics
    var A = SimulateTopicModel(numDocs: 20, vocabSize: 50, numTopics: 3);

    var nmf = new NMF<double>();
    var (W, H) = nmf.Decompose(A, nComponents: 3);

    // Assert: W has shape (50, 3) - word distributions per topic
    Assert.AreEqual(50, W.Rows);
    Assert.AreEqual(3, W.Columns);

    // Assert: H has shape (3, 20) - topic distributions per document
    Assert.AreEqual(3, H.Rows);
    Assert.AreEqual(20, H.Columns);

    // Assert: Reconstruction error is reasonable
    Assert.IsTrue(nmf.ReconstructionError < 10.0);
}
```

### Unit Tests for ICA

```csharp
[TestMethod]
public void TestICA_CocktailPartyProblem()
{
    // Simulate 3 independent sources (e.g., sin waves)
    int n_samples = 1000;
    Matrix<double> S = GenerateIndependentSources(3, n_samples);

    // Create random mixing matrix
    Matrix<double> A = GenerateRandomMixingMatrix(3, 3);

    // Mix sources: X = A × S
    Matrix<double> X = A.Multiply(S);

    // Run ICA to recover sources
    var ica = new ICA<double>();
    Matrix<double> S_recovered = ica.Decompose(X, nComponents: 3);

    // Assert: Recovered sources are correlated with original sources
    // (up to permutation and scaling)
    double avgCorrelation = ComputeAverageCorrelation(S, S_recovered);
    Assert.IsTrue(avgCorrelation > 0.9, $"Correlation too low: {avgCorrelation}");
}

[TestMethod]
public void TestICA_OrthogonalComponents()
{
    var X = GenerateMixedSignals(5, 500);

    var ica = new ICA<double>();
    ica.Decompose(X, nComponents: 5);

    // Assert: Unmixing matrix rows are orthonormal (after whitening)
    Matrix<double> W = ica.UnmixingMatrix;
    Matrix<double> WWT = W.Multiply(W.Transpose());

    // Check that W × Wᵀ ≈ I
    for (int i = 0; i < W.Rows; i++)
    {
        for (int j = 0; j < W.Rows; j++)
        {
            double expected = (i == j) ? 1.0 : 0.0;
            Assert.AreEqual(expected, WWT[i, j], 0.1);
        }
    }
}

[TestMethod]
public void TestICA_InversionProperty()
{
    var X = GenerateMixedSignals(3, 500);

    var ica = new ICA<double>();
    ica.Decompose(X, nComponents: 3);

    // Assert: W^(-1) = A (unmixing is inverse of mixing)
    Matrix<double> W = ica.UnmixingMatrix;
    Matrix<double> A = ica.MixingMatrix;
    Matrix<double> WA = W.Multiply(A);

    // Check that W × A ≈ I
    for (int i = 0; i < W.Rows; i++)
    {
        for (int j = 0; j < W.Rows; j++)
        {
            double expected = (i == j) ? 1.0 : 0.0;
            Assert.AreEqual(expected, WA[i, j], 0.1);
        }
    }
}

[TestMethod]
public void TestICA_Transform()
{
    // Train ICA on one dataset
    var X_train = GenerateMixedSignals(3, 500);

    var ica = new ICA<double>();
    ica.Decompose(X_train, nComponents: 3);

    // Apply to new data
    var X_test = GenerateMixedSignals(3, 100);
    Matrix<double> S_test = ica.Transform(X_test);

    // Assert: Transformed data has correct dimensions
    Assert.AreEqual(3, S_test.Rows);
    Assert.AreEqual(100, S_test.Columns);
}
```

---

## Performance Considerations

### Computational Complexity

| Decomposition | Time Complexity | Space Complexity | Numerical Stability |
|---------------|-----------------|------------------|---------------------|
| SVD (Golub-Reinsch) | O(min(m²n, mn²)) | O(mn) | Excellent |
| NMF | O(kmn × iter) | O(mn + kn + km) | Good |
| LU | O(n³) | O(n²) | Good (with pivoting) |
| QR (Householder) | O(mn²) | O(mn) | Excellent |
| Cholesky | O(n³/3) | O(n²) | Excellent (for SPD) |
| Eigen (QR iteration) | O(n³) | O(n²) | Good |
| ICA (FastICA) | O(n³ + kmn×iter) | O(mn + kn) | Good |

Where:
- m, n = matrix dimensions
- k = number of components (NMF, ICA)
- iter = number of iterations

### Optimization Tips

1. **NMF Convergence:**
   - Use good initialization (not completely random)
   - Monitor reconstruction error every 10 iterations
   - Early stopping if error plateaus
   - Adjust tolerance based on application needs

2. **ICA Whitening:**
   - Whitening is crucial for FastICA performance
   - Use eigen decomposition (more stable than Cholesky for ill-conditioned covariance)
   - Regularize eigenvalues: λ_i = max(λ_i, ε) to avoid division by zero

3. **Memory Management:**
   - For large matrices, compute in blocks
   - Don't store full intermediate products if not needed
   - For NMF, can use sparse matrix operations if A is sparse

4. **Parallelization:**
   - NMF updates can be parallelized (W and H updates are independent within iteration)
   - ICA component extraction can be parallelized (parallel FastICA variant)

---

## Complexity Estimates

### Issue 322 Breakdown

| Task | Story Points | Estimated Hours | Complexity |
|------|--------------|-----------------|------------|
| Review Existing SVD | 3 | 3-4 | Low |
| Review Other Decompositions | 5 | 5-6 | Low-Medium |
| NMF Implementation | 13 | 14-16 | Medium |
| NMF Unit Tests | 5 | 5-6 | Low-Medium |
| ICA Implementation | 13 | 14-16 | Medium-High |
| ICA Unit Tests | 5 | 5-6 | Low-Medium |
| Enhance SVD (pseudoinverse) | 3 | 3-4 | Low |
| Enhance Cholesky (regularization) | 3 | 3-4 | Low |
| Integration Tests (all decompositions) | 10 | 10-12 | Medium |
| **Total** | **60** | **62-74** | **Medium** |

### Implementation Order (Recommended)

1. **Week 1:** Review existing implementations (8 points, 8-10 hours)
   - Verify NumOps usage
   - Check test coverage
   - Identify enhancement opportunities

2. **Week 2:** NMF Implementation (18 points, 19-22 hours)
   - Create interface
   - Implement multiplicative updates
   - Write unit tests
   - Test with real-world data (topic modeling)

3. **Week 3:** ICA Implementation (18 points, 19-22 hours)
   - Create interface
   - Implement whitening and FastICA
   - Write unit tests
   - Test cocktail party problem

4. **Week 4:** Enhancements and Integration (16 points, 18-22 hours)
   - Add pseudoinverse to SVD
   - Enhance other decompositions
   - Write comprehensive integration tests
   - Performance benchmarking

**Total Estimated Time:** 4-5 weeks for a junior developer

---

## Additional Resources

### Academic Papers
1. Lee & Seung (2001): "Algorithms for Non-negative Matrix Factorization"
2. Hyvärinen & Oja (2000): "Independent Component Analysis: Algorithms and Applications"
3. Golub & Reinsch (1970): "Singular Value Decomposition and Least Squares Solutions"
4. Cholesky (1924): "Sur la résolution numérique des systèmes d'équations linéaires"

### Online Resources
1. NumPy decompositions documentation: https://numpy.org/doc/stable/reference/routines.linalg.html
2. scikit-learn NMF: https://scikit-learn.org/stable/modules/decomposition.html#nmf
3. scikit-learn FastICA: https://scikit-learn.org/stable/modules/decomposition.html#ica
4. LAPACK (reference implementations): https://www.netlib.org/lapack/

### Code References
- NumPy linalg: https://github.com/numpy/numpy/tree/main/numpy/linalg
- scikit-learn decomposition: https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/decomposition
- Eigen library (C++): https://eigen.tuxfamily.org/

---

## Summary

This guide has covered:

1. **Understanding**: What matrix decompositions are and when to use each one
2. **Math**: Detailed algorithms for SVD, NMF, LU, QR, Cholesky, Eigen, ICA
3. **Implementation**: Complete code with NumOps for NMF and ICA (others exist)
4. **Testing**: Comprehensive test strategies for each decomposition
5. **Performance**: Complexity analysis and optimization techniques

**Key Takeaways:**

- SVD is the most versatile (works for any matrix)
- LU/QR/Cholesky are for solving linear systems (Cholesky fastest for SPD matrices)
- Eigen decomposition finds fundamental characteristics
- NMF is for non-negative data (interpretable parts)
- ICA separates mixed independent signals

**Architecture Notes:**
- All decompositions should implement `IMatrixDecomposition<T>` where applicable
- Use NumOps throughout for type genericity
- Provide comprehensive XML documentation with "For Beginners" sections
- Write tests for multiple numeric types (double, float)

Good luck with your implementation!
