# Junior Developer Implementation Guide: Issue #384

## Overview
**Issue**: Support Vector Machines (SVM)
**Goal**: Implement Support Vector Machine classifiers with multiple kernel functions
**Difficulty**: Advanced
**Estimated Time**: 12-16 hours

## What You'll Be Building

You'll implement **Support Vector Machine (SVM) classifiers** with multiple kernel functions:

1. **ISVMClassifier Interface** - Defines SVM-specific methods
2. **SVMClassifierBase** - Base class with shared SVM logic
3. **LinearSVMClassifier** - Linear kernel SVM
4. **RBFSVMClassifier** - Radial Basis Function (Gaussian) kernel SVM
5. **PolynomialSVMClassifier** - Polynomial kernel SVM
6. **Kernel Functions** - Linear, RBF, Polynomial, Sigmoid kernels
7. **Comprehensive Unit Tests** - 80%+ coverage

## Understanding Support Vector Machines

### What is a Support Vector Machine?

**Support Vector Machine (SVM)** is a powerful classification algorithm that finds the optimal hyperplane to separate classes in feature space.

**Key Concepts:**

1. **Hyperplane**: A decision boundary that separates different classes
   - In 2D: A line
   - In 3D: A plane
   - In higher dimensions: A hyperplane

2. **Margin**: The distance between the hyperplane and the nearest data points from each class
   - SVM finds the hyperplane with the **maximum margin**
   - Larger margin = better generalization

3. **Support Vectors**: The data points closest to the hyperplane
   - These points "support" or define the hyperplane
   - Only support vectors matter; other points can be removed without changing the model

4. **Kernel Trick**: Transform data into higher dimensions to make it linearly separable
   - Linear kernel: No transformation (data already separable)
   - RBF kernel: Maps to infinite-dimensional space
   - Polynomial kernel: Maps to polynomial feature space

**Real-World Analogy:**

Imagine you're separating apples from oranges on a table:
- **Hyperplane**: A stick you place on the table to separate them
- **Margin**: The width of the gap between the stick and the nearest fruits
- **Support Vectors**: The fruits closest to the stick (touching the margin)
- **Kernel Trick**: If fruits are mixed, you can lift the table into 3D space where they become separable

### Mathematical Formulas

**Primal Problem (Linear SVM):**
```
Minimize: (1/2) * ||w||^2 + C * sum(ξi)
Subject to: yi * (w · xi + b) >= 1 - ξi
            ξi >= 0

where:
    w = weight vector (defines hyperplane orientation)
    b = bias term (defines hyperplane position)
    C = regularization parameter (trade-off between margin and misclassification)
    ξi = slack variables (allow some misclassification)
    yi = class label (+1 or -1)
    xi = feature vector for sample i
```

**Dual Problem (Kernel SVM):**
```
Maximize: sum(αi) - (1/2) * sum(αi * αj * yi * yj * K(xi, xj))
Subject to: 0 <= αi <= C
            sum(αi * yi) = 0

where:
    αi = Lagrange multipliers (dual variables)
    K(xi, xj) = kernel function
```

**Decision Function:**
```
f(x) = sign(sum(αi * yi * K(xi, x)) + b)

Prediction:
    if f(x) > 0: class +1
    if f(x) < 0: class -1
```

**Kernel Functions:**

1. **Linear Kernel:**
   ```
   K(x, y) = x · y

   Use when: Data is already linearly separable
   ```

2. **RBF (Gaussian) Kernel:**
   ```
   K(x, y) = exp(-γ * ||x - y||^2)

   where γ = 1 / (2 * σ^2)

   Use when: Non-linear decision boundary needed, no prior knowledge of data structure
   ```

3. **Polynomial Kernel:**
   ```
   K(x, y) = (γ * (x · y) + r)^d

   where:
       d = degree of polynomial
       γ = scaling parameter
       r = coefficient (bias term)

   Use when: Data has polynomial relationships
   ```

4. **Sigmoid Kernel:**
   ```
   K(x, y) = tanh(γ * (x · y) + r)

   Use when: Data resembles neural network activation patterns
   ```

**Support Vectors:**
```
Support vectors are points where: 0 < αi < C (on margin) or αi = C (misclassified/inside margin)

Only these points contribute to predictions:
    f(x) = sum over support vectors only (αi * yi * K(xi, x)) + b
```

## Understanding the Codebase

### Key Files to Review

**Existing Regression Implementation:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Regression\SupportVectorRegression.cs
```

**Existing Kernel Functions (if any):**
```
C:\Users\cheat\source\repos\AiDotNet\src\KernelFunctions\*
```

**Interfaces:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IKernelFunction.cs
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\INumericOperations.cs
```

**Loss Functions:**
```
C:\Users\cheat\source\repos\AiDotNet\src\LossFunctions\HingeLoss.cs
```

### Three-Tier Architecture Pattern

```
ISVMClassifier<T>           (Interface - defines contract)
    ↓
SVMClassifierBase<T>        (Base class - shared logic, SMO algorithm)
    ↓
┌────────────┬─────────────┬──────────────┐
LinearSVMClassifier  RBFSVMClassifier  PolynomialSVMClassifier
(Concrete implementations - specific kernels)
```

## Step-by-Step Implementation Guide

### Step 1: Create ISVMClassifier Interface

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\ISVMClassifier.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for Support Vector Machine (SVM) classification models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// Support Vector Machines (SVMs) are supervised learning models used for classification
/// and regression analysis. SVMs are effective in high-dimensional spaces and are particularly
/// well-suited for complex classification problems.
/// </para>
/// <para><b>For Beginners:</b> An SVM is like drawing the best possible line (or curve) to separate
/// different groups of data points.
///
/// Think of it like organizing colored marbles on a table:
/// - You have red marbles and blue marbles mixed together
/// - An SVM finds the best way to draw a line separating red from blue
/// - It tries to make the gap between the line and nearest marbles as wide as possible
/// - This "maximum margin" helps the model work better on new, unseen marbles
///
/// SVMs are powerful because:
/// - They work well even with complex, non-linear patterns (using kernel tricks)
/// - They focus on the most important data points (support vectors)
/// - They're resistant to overfitting when configured properly
/// - They work well in high-dimensional spaces
///
/// Real-world uses:
/// - Text classification (spam detection, sentiment analysis)
/// - Image recognition (face detection, handwriting recognition)
/// - Bioinformatics (protein classification, gene expression)
/// - Financial forecasting (credit scoring, fraud detection)
/// </para>
/// </remarks>
public interface ISVMClassifier<T>
{
    /// <summary>
    /// Trains the SVM classifier on the provided data.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The class labels for each sample (typically +1 and -1 for binary classification).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method teaches the SVM to recognize patterns.
    ///
    /// During training, the SVM:
    /// 1. Looks at all the data points and their labels
    /// 2. Finds the hyperplane that separates the classes with the maximum margin
    /// 3. Identifies which points are "support vectors" (closest to the decision boundary)
    /// 4. Calculates weights (alpha values) for each support vector
    ///
    /// After training, the SVM can predict labels for new, unseen data.
    /// </para>
    /// </remarks>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    /// <param name="X">The feature matrix to predict labels for.</param>
    /// <returns>A vector of predicted class labels (+1 or -1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method makes predictions on new data.
    ///
    /// For each new sample:
    /// 1. The SVM calculates the decision function using support vectors
    /// 2. If the result is positive, it predicts class +1
    /// 3. If the result is negative, it predicts class -1
    ///
    /// The magnitude of the result indicates confidence (farther from zero = more confident).
    /// </para>
    /// </remarks>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Computes the decision function values for the input data.
    /// </summary>
    /// <param name="X">The feature matrix to compute decision values for.</param>
    /// <returns>A vector of decision function values (distance from hyperplane).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The decision function tells you how far each point is from the boundary.
    ///
    /// - Positive values: Point is on the +1 side of the hyperplane
    /// - Negative values: Point is on the -1 side of the hyperplane
    /// - Values near zero: Point is close to the decision boundary (uncertain)
    /// - Large absolute values: Point is far from boundary (high confidence)
    ///
    /// This is useful when you need a confidence measure for predictions.
    /// </para>
    /// </remarks>
    Vector<T> DecisionFunction(Matrix<T> X);

    /// <summary>
    /// Gets the support vectors found during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Support vectors are the most important training points.
    ///
    /// These are the data points that:
    /// - Lie closest to the decision boundary
    /// - Actually define where the boundary is
    /// - Are the only points needed for making predictions
    ///
    /// Often, only a small fraction of training points become support vectors,
    /// making SVM predictions very efficient.
    /// </para>
    /// </remarks>
    Matrix<T>? SupportVectors { get; }

    /// <summary>
    /// Gets the dual coefficients (alpha * y) for each support vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These coefficients determine how much each support vector
    /// influences predictions.
    ///
    /// - Larger absolute values: Support vector has more influence
    /// - Positive values: Support vector from +1 class
    /// - Negative values: Support vector from -1 class
    /// </para>
    /// </remarks>
    Vector<T>? DualCoefficients { get; }

    /// <summary>
    /// Gets the bias term (intercept) of the decision function.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The bias shifts the decision boundary.
    ///
    /// Think of it like adjusting the position of a dividing line on a table.
    /// The weights control the angle, and the bias controls where it's placed.
    /// </para>
    /// </remarks>
    T Bias { get; }
}
```

### Step 2: Create Kernel Function Interface and Implementations

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IKernelFunction.cs` (if it doesn't exist)

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a kernel function for transforming data into higher-dimensional spaces.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A kernel function is a mathematical trick that helps SVM
/// work with non-linear patterns.
///
/// Imagine you have red and blue points mixed together on a table (2D).
/// If you can't separate them with a straight line, a kernel function is like:
/// 1. Lifting some points up into 3D space
/// 2. Finding a plane in 3D that separates them
/// 3. Projecting that plane back down to 2D (which becomes a curve)
///
/// The amazing part: Kernels calculate this without actually transforming the data,
/// making it very efficient even for infinite-dimensional spaces (like RBF kernel).
/// </para>
/// </remarks>
public interface IKernelFunction<T>
{
    /// <summary>
    /// Computes the kernel function between two vectors.
    /// </summary>
    /// <param name="x">The first vector.</param>
    /// <param name="y">The second vector.</param>
    /// <returns>The kernel value (similarity measure).</returns>
    T Compute(Vector<T> x, Vector<T> y);

    /// <summary>
    /// Computes the kernel matrix (Gram matrix) for a set of vectors.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <returns>A symmetric matrix where K[i,j] = kernel(X[i], X[j]).</returns>
    Matrix<T> ComputeMatrix(Matrix<T> X);

    /// <summary>
    /// Computes kernel values between training data and test data.
    /// </summary>
    /// <param name="X">Training data matrix.</param>
    /// <param name="Y">Test data matrix.</param>
    /// <returns>A matrix where K[i,j] = kernel(X[i], Y[j]).</returns>
    Matrix<T> ComputeMatrix(Matrix<T> X, Matrix<T> Y);
}
```

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\KernelFunctions\LinearKernel.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.KernelFunctions;

/// <summary>
/// Implements the linear kernel function: K(x, y) = x · y
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The linear kernel is the simplest kernel - it's just the dot product.
///
/// Use linear kernel when:
/// - Your data is already linearly separable (can be separated with a straight line)
/// - You want the fastest training and prediction
/// - You want the most interpretable model
/// - You have high-dimensional data (text classification, etc.)
///
/// Formula: K(x, y) = x · y = sum(xi * yi)
///
/// Example: K([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// </para>
/// </remarks>
public class LinearKernel<T> : IKernelFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the linear kernel between two vectors (dot product).
    /// </summary>
    public T Compute(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        T result = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(x[i], y[i]));
        }
        return result;
    }

    /// <summary>
    /// Computes the linear kernel matrix (Gram matrix) for a set of vectors.
    /// </summary>
    public Matrix<T> ComputeMatrix(Matrix<T> X)
    {
        // K = X * X^T
        return MatrixHelper<T>.Multiply(X, MatrixHelper<T>.Transpose(X));
    }

    /// <summary>
    /// Computes kernel values between training data and test data.
    /// </summary>
    public Matrix<T> ComputeMatrix(Matrix<T> X, Matrix<T> Y)
    {
        // K = X * Y^T
        return MatrixHelper<T>.Multiply(X, MatrixHelper<T>.Transpose(Y));
    }
}
```

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\KernelFunctions\RBFKernel.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.KernelFunctions;

/// <summary>
/// Implements the Radial Basis Function (RBF) / Gaussian kernel:
/// K(x, y) = exp(-gamma * ||x - y||^2)
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The RBF kernel is the most popular kernel for SVM.
///
/// Think of it as measuring similarity based on distance:
/// - Points close together: High kernel value (near 1)
/// - Points far apart: Low kernel value (near 0)
///
/// Use RBF kernel when:
/// - You don't know the shape of the decision boundary
/// - Your data has complex, non-linear patterns
/// - You want a flexible model that can learn curves and complex shapes
///
/// Formula: K(x, y) = exp(-gamma * ||x - y||^2)
///
/// The gamma parameter controls the "reach" of each training point:
/// - Small gamma: Wide reach, smooth decision boundary (may underfit)
/// - Large gamma: Narrow reach, complex decision boundary (may overfit)
///
/// Default gamma: 1 / (n_features * variance(X))
/// </para>
/// </remarks>
public class RBFKernel<T> : IKernelFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly T _gamma;

    /// <summary>
    /// Creates an RBF kernel with the specified gamma parameter.
    /// </summary>
    /// <param name="gamma">
    /// The gamma parameter that controls the width of the Gaussian function.
    /// If not specified, it will be calculated as 1 / n_features during training.
    /// </param>
    public RBFKernel(T? gamma = null)
    {
        _gamma = gamma ?? NumOps.One; // Will be updated during training if null
    }

    /// <summary>
    /// Computes the RBF kernel between two vectors.
    /// </summary>
    public T Compute(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        // Calculate ||x - y||^2
        T squaredDistance = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T diff = NumOps.Subtract(x[i], y[i]);
            squaredDistance = NumOps.Add(squaredDistance, NumOps.Multiply(diff, diff));
        }

        // K(x, y) = exp(-gamma * ||x - y||^2)
        T exponent = NumOps.Negate(NumOps.Multiply(_gamma, squaredDistance));
        return NumOps.Exp(exponent);
    }

    /// <summary>
    /// Computes the RBF kernel matrix (Gram matrix) for a set of vectors.
    /// </summary>
    public Matrix<T> ComputeMatrix(Matrix<T> X)
    {
        int n = X.Rows;
        var K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = Compute(X.GetRow(i), X.GetRow(j));
                K[i, j] = value;
                K[j, i] = value; // Symmetric
            }
        }

        return K;
    }

    /// <summary>
    /// Computes kernel values between training data and test data.
    /// </summary>
    public Matrix<T> ComputeMatrix(Matrix<T> X, Matrix<T> Y)
    {
        int nX = X.Rows;
        int nY = Y.Rows;
        var K = new Matrix<T>(nX, nY);

        for (int i = 0; i < nX; i++)
        {
            for (int j = 0; j < nY; j++)
            {
                K[i, j] = Compute(X.GetRow(i), Y.GetRow(j));
            }
        }

        return K;
    }

    /// <summary>
    /// Gets the gamma parameter.
    /// </summary>
    public T Gamma => _gamma;
}
```

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\KernelFunctions\PolynomialKernel.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.KernelFunctions;

/// <summary>
/// Implements the Polynomial kernel:
/// K(x, y) = (gamma * (x · y) + coef0)^degree
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The polynomial kernel creates curved decision boundaries.
///
/// Use polynomial kernel when:
/// - You know your data has polynomial relationships
/// - You want interactions between features
/// - Linear kernel is too simple but RBF is too complex
///
/// Formula: K(x, y) = (gamma * (x · y) + coef0)^degree
///
/// Parameters:
/// - degree: The polynomial degree (2 = quadratic, 3 = cubic, etc.)
///   - degree = 1: Equivalent to linear kernel (with coef0 = 0)
///   - degree = 2: Quadratic decision boundary
///   - degree = 3+: Higher-order polynomial boundaries
///
/// - gamma: Scaling factor for the dot product
///   - Controls the influence of each feature
///
/// - coef0: Independent term (bias)
///   - Controls how much the higher-degree terms matter
///
/// Example with degree=2, gamma=1, coef0=0:
///   K([1,2], [3,4]) = ((1*3 + 2*4) + 0)^2 = 11^2 = 121
/// </para>
/// </remarks>
public class PolynomialKernel<T> : IKernelFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _degree;
    private readonly T _gamma;
    private readonly T _coef0;

    /// <summary>
    /// Creates a polynomial kernel with the specified parameters.
    /// </summary>
    /// <param name="degree">The degree of the polynomial (default: 3).</param>
    /// <param name="gamma">The gamma parameter (default: 1).</param>
    /// <param name="coef0">The independent term (default: 0).</param>
    public PolynomialKernel(int degree = 3, T? gamma = null, T? coef0 = null)
    {
        if (degree < 1)
            throw new ArgumentException("Degree must be >= 1", nameof(degree));

        _degree = degree;
        _gamma = gamma ?? NumOps.One;
        _coef0 = coef0 ?? NumOps.Zero;
    }

    /// <summary>
    /// Computes the polynomial kernel between two vectors.
    /// </summary>
    public T Compute(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        // Calculate dot product: x · y
        T dotProduct = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(x[i], y[i]));
        }

        // K(x, y) = (gamma * (x · y) + coef0)^degree
        T scaled = NumOps.Add(NumOps.Multiply(_gamma, dotProduct), _coef0);

        // Compute scaled^degree
        T result = NumOps.One;
        for (int i = 0; i < _degree; i++)
        {
            result = NumOps.Multiply(result, scaled);
        }

        return result;
    }

    /// <summary>
    /// Computes the polynomial kernel matrix (Gram matrix) for a set of vectors.
    /// </summary>
    public Matrix<T> ComputeMatrix(Matrix<T> X)
    {
        int n = X.Rows;
        var K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = Compute(X.GetRow(i), X.GetRow(j));
                K[i, j] = value;
                K[j, i] = value; // Symmetric
            }
        }

        return K;
    }

    /// <summary>
    /// Computes kernel values between training data and test data.
    /// </summary>
    public Matrix<T> ComputeMatrix(Matrix<T> X, Matrix<T> Y)
    {
        int nX = X.Rows;
        int nY = Y.Rows;
        var K = new Matrix<T>(nX, nY);

        for (int i = 0; i < nX; i++)
        {
            for (int j = 0; j < nY; j++)
            {
                K[i, j] = Compute(X.GetRow(i), Y.GetRow(j));
            }
        }

        return K;
    }

    /// <summary>
    /// Gets the polynomial degree.
    /// </summary>
    public int Degree => _degree;

    /// <summary>
    /// Gets the gamma parameter.
    /// </summary>
    public T Gamma => _gamma;

    /// <summary>
    /// Gets the independent term (coef0).
    /// </summary>
    public T Coef0 => _coef0;
}
```

### Step 3: Create SVMClassifierBase with SMO Algorithm

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\SVMClassifierBase.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Base class for Support Vector Machine (SVM) classifiers implementing the Sequential Minimal Optimization (SMO) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This base class implements the core SVM training algorithm using Sequential Minimal Optimization (SMO),
/// which efficiently solves the quadratic programming problem that arises in SVM training.
/// </para>
/// <para><b>For Beginners:</b> This base class contains the "brain" of the SVM.
///
/// The SMO algorithm is like a negotiation process:
/// 1. Start with random guesses for how important each training point is (alpha values)
/// 2. Pick two training points at a time
/// 3. Adjust their importance to improve the overall separation
/// 4. Repeat until you can't improve anymore
///
/// This is much faster than trying to optimize all points at once,
/// making SVM training practical even for large datasets.
///
/// The key insight: By only updating two points at a time, we can guarantee
/// that the constraints are always satisfied, turning a complex optimization
/// problem into a series of simple analytical solutions.
/// </para>
/// </remarks>
public abstract class SVMClassifierBase<T> : ISVMClassifier<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The regularization parameter that controls the trade-off between margin maximization and training error.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> C controls how strict the SVM is about classification errors.
    ///
    /// - Small C (e.g., 0.1): Allows more errors, prefers wider margin (may underfit)
    ///   - Use when: You have noisy data, want simpler model, more generalization
    ///
    /// - Large C (e.g., 100): Tries to classify all points correctly, narrower margin (may overfit)
    ///   - Use when: You have clean data, want precise fit, data is well-separated
    ///
    /// Typical values: 0.1, 1, 10, 100
    /// Start with C=1 and adjust based on validation performance.
    /// </para>
    /// </remarks>
    protected T C { get; set; }

    /// <summary>
    /// The convergence tolerance for the SMO algorithm.
    /// </summary>
    protected T Tolerance { get; set; }

    /// <summary>
    /// The maximum number of iterations for the SMO algorithm.
    /// </summary>
    protected int MaxIterations { get; set; }

    /// <summary>
    /// The kernel function used to transform the data.
    /// </summary>
    protected IKernelFunction<T> Kernel { get; set; }

    /// <summary>
    /// The training data (support vectors after training).
    /// </summary>
    protected Matrix<T>? TrainingData { get; set; }

    /// <summary>
    /// The training labels.
    /// </summary>
    protected Vector<T>? TrainingLabels { get; set; }

    /// <summary>
    /// The Lagrange multipliers (alpha values) for each training point.
    /// </summary>
    protected Vector<T>? Alphas { get; set; }

    /// <summary>
    /// The bias term of the decision function.
    /// </summary>
    protected T _bias;

    /// <summary>
    /// The error cache for SMO optimization.
    /// </summary>
    protected Vector<T>? ErrorCache { get; set; }

    /// <summary>
    /// Creates a new SVM classifier base with the specified parameters.
    /// </summary>
    protected SVMClassifierBase(IKernelFunction<T> kernel, T? c = null, T? tolerance = null, int maxIterations = 1000)
    {
        Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        C = c ?? NumOps.FromInt(1);
        Tolerance = tolerance ?? NumOps.FromDouble(0.001);
        MaxIterations = maxIterations;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Trains the SVM classifier using the SMO algorithm.
    /// </summary>
    public virtual void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of samples in X must match length of y");

        TrainingData = X;
        TrainingLabels = y;

        int n = X.Rows;
        Alphas = new Vector<T>(n); // Initialize to zeros
        _bias = NumOps.Zero;
        ErrorCache = new Vector<T>(n);

        // Initialize error cache
        for (int i = 0; i < n; i++)
        {
            ErrorCache[i] = NumOps.Subtract(ComputeDecisionFunction(i), y[i]);
        }

        // Run SMO algorithm
        RunSMO();

        // Extract support vectors (points where alpha > 0)
        ExtractSupportVectors();
    }

    /// <summary>
    /// Runs the Sequential Minimal Optimization (SMO) algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The SMO algorithm is the training process.
    ///
    /// It works in iterations:
    /// 1. Look for pairs of points that violate the optimality conditions
    /// 2. Update their alpha values to improve the solution
    /// 3. Update the bias term
    /// 4. Repeat until no more improvements can be made or max iterations reached
    ///
    /// The algorithm prioritizes points that violate the KKT conditions most,
    /// ensuring rapid convergence to the optimal solution.
    /// </para>
    /// </remarks>
    protected virtual void RunSMO()
    {
        int n = TrainingData!.Rows;
        int numChanged = 0;
        bool examineAll = true;
        int iterations = 0;

        while ((numChanged > 0 || examineAll) && iterations < MaxIterations)
        {
            numChanged = 0;

            if (examineAll)
            {
                // Examine all samples
                for (int i = 0; i < n; i++)
                {
                    numChanged += ExamineExample(i);
                }
            }
            else
            {
                // Examine non-bound samples (0 < alpha < C)
                for (int i = 0; i < n; i++)
                {
                    T alpha = Alphas![i];
                    if (NumOps.GreaterThan(alpha, NumOps.Zero) && NumOps.LessThan(alpha, C))
                    {
                        numChanged += ExamineExample(i);
                    }
                }
            }

            if (examineAll)
            {
                examineAll = false;
            }
            else if (numChanged == 0)
            {
                examineAll = true;
            }

            iterations++;
        }
    }

    /// <summary>
    /// Examines a training example and attempts to optimize it.
    /// </summary>
    protected virtual int ExamineExample(int i2)
    {
        T y2 = TrainingLabels![i2];
        T alpha2 = Alphas![i2];
        T E2 = ErrorCache![i2];
        T r2 = NumOps.Multiply(E2, y2);

        // Check KKT conditions
        if ((NumOps.LessThan(r2, NumOps.Negate(Tolerance)) && NumOps.LessThan(alpha2, C)) ||
            (NumOps.GreaterThan(r2, Tolerance) && NumOps.GreaterThan(alpha2, NumOps.Zero)))
        {
            // Find i1 to optimize with i2
            int i1 = SelectSecondAlpha(i2, E2);

            if (TakeStep(i1, i2))
            {
                return 1;
            }
        }

        return 0;
    }

    /// <summary>
    /// Selects the second alpha to optimize using heuristics.
    /// </summary>
    protected virtual int SelectSecondAlpha(int i2, T E2)
    {
        int n = TrainingData!.Rows;
        int i1 = 0;
        T maxDelta = NumOps.Zero;

        // Heuristic: Choose i1 to maximize |E1 - E2|
        for (int i = 0; i < n; i++)
        {
            if (i == i2) continue;

            T E1 = ErrorCache![i];
            T delta = NumOps.Abs(NumOps.Subtract(E1, E2));

            if (NumOps.GreaterThan(delta, maxDelta))
            {
                maxDelta = delta;
                i1 = i;
            }
        }

        return i1;
    }

    /// <summary>
    /// Attempts to optimize the pair (i1, i2).
    /// </summary>
    protected virtual bool TakeStep(int i1, int i2)
    {
        if (i1 == i2) return false;

        T alpha1 = Alphas![i1];
        T alpha2 = Alphas![i2];
        T y1 = TrainingLabels![i1];
        T y2 = TrainingLabels![i2];
        T E1 = ErrorCache![i1];
        T E2 = ErrorCache![i2];

        T s = NumOps.Multiply(y1, y2);

        // Compute L and H (bounds on alpha2)
        T L, H;
        if (NumOps.GreaterThan(s, NumOps.Zero))
        {
            // Same sign: L = max(0, alpha1 + alpha2 - C), H = min(C, alpha1 + alpha2)
            T sum = NumOps.Add(alpha1, alpha2);
            L = NumOps.Max(NumOps.Zero, NumOps.Subtract(sum, C));
            H = NumOps.Min(C, sum);
        }
        else
        {
            // Different sign: L = max(0, alpha2 - alpha1), H = min(C, C + alpha2 - alpha1)
            T diff = NumOps.Subtract(alpha2, alpha1);
            L = NumOps.Max(NumOps.Zero, diff);
            H = NumOps.Min(C, NumOps.Add(C, diff));
        }

        if (NumOps.GreaterThanOrEqual(L, H))
        {
            return false;
        }

        // Compute eta (second derivative of objective function)
        T k11 = Kernel.Compute(TrainingData!.GetRow(i1), TrainingData.GetRow(i1));
        T k12 = Kernel.Compute(TrainingData.GetRow(i1), TrainingData.GetRow(i2));
        T k22 = Kernel.Compute(TrainingData.GetRow(i2), TrainingData.GetRow(i2));

        T eta = NumOps.Add(NumOps.Add(k11, k22), NumOps.Multiply(NumOps.FromInt(-2), k12));

        T alpha2New;
        if (NumOps.GreaterThan(eta, NumOps.Zero))
        {
            // Compute new alpha2
            alpha2New = NumOps.Add(alpha2, NumOps.Divide(NumOps.Multiply(y2, NumOps.Subtract(E1, E2)), eta));

            // Clip alpha2 to [L, H]
            alpha2New = NumOps.Min(H, NumOps.Max(L, alpha2New));
        }
        else
        {
            // eta <= 0 is unusual; use boundary values
            alpha2New = L;
        }

        // Check for sufficient change
        if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(alpha2New, alpha2)),
            NumOps.Multiply(Tolerance, NumOps.Add(alpha2, NumOps.Add(alpha2New, Tolerance)))))
        {
            return false;
        }

        // Compute new alpha1
        T alpha1New = NumOps.Add(alpha1,
            NumOps.Multiply(s, NumOps.Subtract(alpha2, alpha2New)));

        // Update alphas
        Alphas[i1] = alpha1New;
        Alphas[i2] = alpha2New;

        // Update bias
        UpdateBias(i1, i2, alpha1, alpha2, alpha1New, alpha2New, E1, E2, k11, k12, k22);

        // Update error cache
        UpdateErrorCache(i1, i2, alpha1, alpha2, alpha1New, alpha2New);

        return true;
    }

    /// <summary>
    /// Updates the bias term after alpha optimization.
    /// </summary>
    protected virtual void UpdateBias(int i1, int i2, T alpha1Old, T alpha2Old,
        T alpha1New, T alpha2New, T E1, T E2, T k11, T k12, T k22)
    {
        T y1 = TrainingLabels![i1];
        T y2 = TrainingLabels![i2];

        // Compute b1 and b2
        T b1 = NumOps.Subtract(_bias, NumOps.Add(E1,
            NumOps.Add(
                NumOps.Multiply(NumOps.Multiply(y1, NumOps.Subtract(alpha1New, alpha1Old)), k11),
                NumOps.Multiply(NumOps.Multiply(y2, NumOps.Subtract(alpha2New, alpha2Old)), k12)
            )));

        T b2 = NumOps.Subtract(_bias, NumOps.Add(E2,
            NumOps.Add(
                NumOps.Multiply(NumOps.Multiply(y1, NumOps.Subtract(alpha1New, alpha1Old)), k12),
                NumOps.Multiply(NumOps.Multiply(y2, NumOps.Subtract(alpha2New, alpha2Old)), k22)
            )));

        // Update bias
        if (NumOps.GreaterThan(alpha1New, NumOps.Zero) && NumOps.LessThan(alpha1New, C))
        {
            _bias = b1;
        }
        else if (NumOps.GreaterThan(alpha2New, NumOps.Zero) && NumOps.LessThan(alpha2New, C))
        {
            _bias = b2;
        }
        else
        {
            _bias = NumOps.Divide(NumOps.Add(b1, b2), NumOps.FromInt(2));
        }
    }

    /// <summary>
    /// Updates the error cache after alpha optimization.
    /// </summary>
    protected virtual void UpdateErrorCache(int i1, int i2, T alpha1Old, T alpha2Old,
        T alpha1New, T alpha2New)
    {
        int n = TrainingData!.Rows;

        for (int i = 0; i < n; i++)
        {
            if (i == i1 || i == i2) continue;

            T k1i = Kernel.Compute(TrainingData.GetRow(i1), TrainingData.GetRow(i));
            T k2i = Kernel.Compute(TrainingData.GetRow(i2), TrainingData.GetRow(i));

            ErrorCache![i] = NumOps.Add(ErrorCache[i],
                NumOps.Add(
                    NumOps.Multiply(NumOps.Multiply(TrainingLabels![i1], NumOps.Subtract(alpha1New, alpha1Old)), k1i),
                    NumOps.Multiply(NumOps.Multiply(TrainingLabels[i2], NumOps.Subtract(alpha2New, alpha2Old)), k2i)
                ));
        }

        ErrorCache[i1] = NumOps.Zero;
        ErrorCache[i2] = NumOps.Zero;
    }

    /// <summary>
    /// Computes the decision function for a training example (using index).
    /// </summary>
    protected virtual T ComputeDecisionFunction(int index)
    {
        int n = TrainingData!.Rows;
        T result = NumOps.Negate(_bias);

        for (int i = 0; i < n; i++)
        {
            T alpha = Alphas![i];
            if (NumOps.GreaterThan(alpha, NumOps.Zero))
            {
                T kernelValue = Kernel.Compute(TrainingData.GetRow(i), TrainingData.GetRow(index));
                result = NumOps.Add(result,
                    NumOps.Multiply(NumOps.Multiply(alpha, TrainingLabels![i]), kernelValue));
            }
        }

        return result;
    }

    /// <summary>
    /// Extracts support vectors and their coefficients after training.
    /// </summary>
    protected virtual void ExtractSupportVectors()
    {
        // Support vectors are points where alpha > 0
        var supportIndices = new List<int>();

        for (int i = 0; i < Alphas!.Length; i++)
        {
            if (NumOps.GreaterThan(Alphas[i], NumOps.Zero))
            {
                supportIndices.Add(i);
            }
        }

        int numSupport = supportIndices.Count;
        if (numSupport == 0)
        {
            SupportVectors = null;
            DualCoefficients = null;
            return;
        }

        // Create support vector matrix
        SupportVectors = new Matrix<T>(numSupport, TrainingData!.Columns);
        DualCoefficients = new Vector<T>(numSupport);

        for (int i = 0; i < numSupport; i++)
        {
            int idx = supportIndices[i];
            for (int j = 0; j < TrainingData.Columns; j++)
            {
                SupportVectors[i, j] = TrainingData[idx, j];
            }
            // Dual coefficient = alpha * y
            DualCoefficients[i] = NumOps.Multiply(Alphas[idx], TrainingLabels![idx]);
        }
    }

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    public virtual Vector<T> Predict(Matrix<T> X)
    {
        var decisions = DecisionFunction(X);
        var predictions = new Vector<T>(decisions.Length);

        for (int i = 0; i < decisions.Length; i++)
        {
            predictions[i] = NumOps.GreaterThanOrEqual(decisions[i], NumOps.Zero)
                ? NumOps.One
                : NumOps.Negate(NumOps.One);
        }

        return predictions;
    }

    /// <summary>
    /// Computes the decision function values for new data.
    /// </summary>
    public virtual Vector<T> DecisionFunction(Matrix<T> X)
    {
        if (SupportVectors == null || DualCoefficients == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        var result = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T value = NumOps.Negate(_bias);

            for (int j = 0; j < SupportVectors.Rows; j++)
            {
                T kernelValue = Kernel.Compute(SupportVectors.GetRow(j), X.GetRow(i));
                value = NumOps.Add(value, NumOps.Multiply(DualCoefficients[j], kernelValue));
            }

            result[i] = value;
        }

        return result;
    }

    /// <summary>
    /// Gets the support vectors.
    /// </summary>
    public Matrix<T>? SupportVectors { get; protected set; }

    /// <summary>
    /// Gets the dual coefficients (alpha * y).
    /// </summary>
    public Vector<T>? DualCoefficients { get; protected set; }

    /// <summary>
    /// Gets the bias term.
    /// </summary>
    public T Bias => _bias;
}
```

### Step 4: Create Concrete SVM Classifier Implementations

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\LinearSVMClassifier.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.KernelFunctions;

namespace AiDotNet.Classification;

/// <summary>
/// Implements a linear Support Vector Machine classifier using a linear kernel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Linear SVM is the simplest and fastest SVM variant.
///
/// Use Linear SVM when:
/// - Your data is linearly separable (or nearly so)
/// - You have high-dimensional data (e.g., text classification)
/// - You want fast training and prediction
/// - You need an interpretable model (can extract feature weights)
///
/// Linear SVM finds a straight line (or hyperplane in higher dimensions) that
/// best separates your classes with the maximum margin.
///
/// Examples where linear SVM excels:
/// - Text classification (spam detection, sentiment analysis)
/// - High-dimensional genomics data
/// - Image classification with pre-extracted features
/// </para>
/// </remarks>
public class LinearSVMClassifier<T> : SVMClassifierBase<T>
{
    /// <summary>
    /// Creates a new linear SVM classifier.
    /// </summary>
    /// <param name="c">The regularization parameter (default: 1.0).</param>
    /// <param name="tolerance">The convergence tolerance (default: 0.001).</param>
    /// <param name="maxIterations">The maximum number of iterations (default: 1000).</param>
    public LinearSVMClassifier(T? c = null, T? tolerance = null, int maxIterations = 1000)
        : base(new LinearKernel<T>(), c, tolerance, maxIterations)
    {
    }
}
```

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\RBFSVMClassifier.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.KernelFunctions;

namespace AiDotNet.Classification;

/// <summary>
/// Implements a Support Vector Machine classifier using the Radial Basis Function (RBF) kernel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> RBF SVM is the most popular SVM variant for complex problems.
///
/// Use RBF SVM when:
/// - You don't know the shape of the decision boundary
/// - Your data has non-linear patterns
/// - You want a flexible model that can learn complex shapes
/// - You have enough training data to avoid overfitting
///
/// The RBF kernel measures similarity based on distance in feature space.
/// It can learn decision boundaries of any shape, from simple curves to
/// very complex regions.
///
/// Key parameters to tune:
/// - C: How strictly to fit the training data
/// - gamma: How far the influence of each training point reaches
///
/// Examples where RBF SVM excels:
/// - Image recognition
/// - Handwriting recognition
/// - Bioinformatics (protein classification)
/// - Complex pattern recognition tasks
/// </para>
/// </remarks>
public class RBFSVMClassifier<T> : SVMClassifierBase<T>
{
    /// <summary>
    /// Creates a new RBF SVM classifier.
    /// </summary>
    /// <param name="gamma">
    /// The gamma parameter for the RBF kernel.
    /// If not specified, it will be set to 1 / (n_features * variance(X)) during training.
    /// </param>
    /// <param name="c">The regularization parameter (default: 1.0).</param>
    /// <param name="tolerance">The convergence tolerance (default: 0.001).</param>
    /// <param name="maxIterations">The maximum number of iterations (default: 1000).</param>
    public RBFSVMClassifier(T? gamma = null, T? c = null, T? tolerance = null, int maxIterations = 1000)
        : base(new RBFKernel<T>(gamma), c, tolerance, maxIterations)
    {
    }
}
```

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\PolynomialSVMClassifier.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.KernelFunctions;

namespace AiDotNet.Classification;

/// <summary>
/// Implements a Support Vector Machine classifier using a polynomial kernel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Polynomial SVM creates curved decision boundaries.
///
/// Use Polynomial SVM when:
/// - You know your data has polynomial relationships
/// - Linear is too simple but RBF is too complex
/// - You want to control the degree of non-linearity
/// - You need interactions between features
///
/// The polynomial kernel implicitly computes all polynomial combinations
/// of your features up to the specified degree, without actually creating
/// those features (which would be computationally expensive).
///
/// Common degree values:
/// - degree = 2: Quadratic decision boundary (good starting point)
/// - degree = 3: Cubic decision boundary
/// - degree = 4+: Higher-order boundaries (risk of overfitting)
///
/// Examples where polynomial SVM is useful:
/// - Problems with known polynomial structure
/// - Image processing (certain types of edges and curves)
/// - Physics simulations with polynomial relationships
/// </para>
/// </remarks>
public class PolynomialSVMClassifier<T> : SVMClassifierBase<T>
{
    /// <summary>
    /// Creates a new polynomial SVM classifier.
    /// </summary>
    /// <param name="degree">The degree of the polynomial (default: 3).</param>
    /// <param name="gamma">The gamma parameter (default: 1).</param>
    /// <param name="coef0">The independent term (default: 0).</param>
    /// <param name="c">The regularization parameter (default: 1.0).</param>
    /// <param name="tolerance">The convergence tolerance (default: 0.001).</param>
    /// <param name="maxIterations">The maximum number of iterations (default: 1000).</param>
    public PolynomialSVMClassifier(int degree = 3, T? gamma = null, T? coef0 = null,
        T? c = null, T? tolerance = null, int maxIterations = 1000)
        : base(new PolynomialKernel<T>(degree, gamma, coef0), c, tolerance, maxIterations)
    {
    }
}
```

### Step 5: Create Comprehensive Unit Tests

Create file: `C:\Users\cheat\source\repos\AiDotNet\tests\Classification\SVMClassifierTests.cs`

```csharp
using System;
using AiDotNet.Classification;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Classification
{
    public class SVMClassifierTests
    {
        private static void AssertClose(double actual, double expected, double tolerance = 1e-6)
        {
            Assert.True(Math.Abs(actual - expected) <= tolerance,
                $"Expected {expected}, but got {actual}. Difference: {Math.Abs(actual - expected)}");
        }

        [Fact]
        public void LinearSVM_SimpleLinearlySepar