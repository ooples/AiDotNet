# Junior Developer Implementation Guide: Issue #386

## Overview
**Issue**: Ridge and Lasso Regression
**Goal**: Implement regularized linear regression models (L2 and L1 regularization)
**Difficulty**: Intermediate-Advanced
**Estimated Time**: 10-14 hours

## What You'll Be Building

You'll implement **regularized linear regression** models:

1. **IRidgeRegression Interface** - Specific to Ridge regression
2. **ILassoRegression Interface** - Specific to Lasso regression
3. **RidgeRegression** - L2 regularization (closed-form solution)
4. **LassoRegression** - L1 regularization (coordinate descent algorithm)
5. **ElasticNetRegression** - Combines L1 and L2 (bonus)
6. **Comprehensive Unit Tests** - 80%+ coverage

## Understanding Regularized Regression

### What is Regularization?

**Regularization** adds a penalty term to prevent overfitting by keeping model coefficients small.

**Problem with Ordinary Least Squares (OLS):**
- Can overfit on training data
- Large coefficients for correlated features
- Unstable when features > samples
- Poor generalization to new data

**Solution: Add penalty for large coefficients**

### Ridge Regression (L2 Regularization)

**Ridge** adds the sum of squared coefficients as penalty.

**Mathematical Formula:**
```
Minimize: ||y - Xw||² + α * ||w||²

where:
    y = target values
    X = feature matrix
    w = coefficient weights
    α = regularization parameter (alpha)
    ||w||² = sum of squared coefficients (L2 norm)

Expanded:
    Loss = ∑(yi - ŷi)² + α * ∑(wj²)

    First term: Prediction error (MSE)
    Second term: Penalty for large coefficients
```

**Closed-Form Solution:**
```
w = (X^T X + α I)^(-1) X^T y

where:
    I = identity matrix
    Adding α I makes matrix invertible even when X^T X is singular
```

**Properties:**
- Shrinks coefficients toward zero but never exactly to zero
- All features retained (no feature selection)
- Handles multicollinearity well
- Computationally efficient (closed-form solution)

**When to Use:**
- Many correlated features
- Preventing overfitting
- Regularization without feature selection
- When you want all features in the model

**Real-World Example:**
Predicting house prices with 100 correlated features (size, rooms, location metrics).
Ridge keeps all features but prevents any single feature from dominating.

### Lasso Regression (L1 Regularization)

**Lasso** (Least Absolute Shrinkage and Selection Operator) adds the sum of absolute coefficients as penalty.

**Mathematical Formula:**
```
Minimize: ||y - Xw||² + α * ||w||₁

where:
    ||w||₁ = sum of absolute values of coefficients (L1 norm)

Expanded:
    Loss = ∑(yi - ŷi)² + α * ∑|wj|
```

**Properties:**
- Shrinks coefficients toward zero
- **Sets some coefficients exactly to zero** (feature selection!)
- Produces sparse models
- Requires iterative optimization (no closed-form solution)
- Common algorithm: Coordinate Descent

**When to Use:**
- Feature selection needed (sparse models)
- Many irrelevant features
- Interpretable models desired
- You suspect only few features are truly important

**Real-World Example:**
Gene expression analysis with 10,000 genes but only ~100 are relevant.
Lasso automatically identifies and keeps only the important genes.

### ElasticNet (L1 + L2)

**ElasticNet** combines Ridge and Lasso penalties.

**Mathematical Formula:**
```
Minimize: ||y - Xw||² + α * (ρ * ||w||₁ + (1-ρ) * ||w||²)

where:
    ρ = mixing parameter (0 to 1)
    ρ = 0: Ridge regression
    ρ = 1: Lasso regression
    0 < ρ < 1: Elastic Net
```

**When to Use:**
- Features are correlated AND you want feature selection
- Lasso is too aggressive (eliminates too many features)
- Best of both worlds: feature selection + handling multicollinearity

### Comparison Table

| Aspect | Ridge (L2) | Lasso (L1) | ElasticNet |
|--------|-----------|-----------|------------|
| Penalty | Sum of squares | Sum of absolute | Both |
| Feature Selection | No (keeps all) | Yes (sets to 0) | Yes |
| Solution | Closed-form | Iterative | Iterative |
| Speed | Fast | Slower | Slower |
| Multicollinearity | Excellent | Poor | Good |
| Sparsity | No | Yes | Yes |

## Understanding the Codebase

### Architecture Pattern

```
IRegression<T>                  (Base regression interface)
    ↓
RegressionBase<T>               (Common regression logic - may exist)
    ↓
┌──────────────┬──────────────┬────────────────┐
RidgeRegression  LassoRegression  ElasticNetRegression
(L2 - closed)   (L1 - iterative)  (L1+L2 - iterative)
```

## Step-by-Step Implementation Guide

### Step 1: Create RidgeRegression

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Regression\RidgeRegression.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Ridge Regression with L2 regularization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Ridge Regression adds an L2 penalty (sum of squared coefficients) to the ordinary
/// least squares objective function. This regularization prevents overfitting by
/// shrinking coefficient estimates toward zero.
/// </para>
/// <para><b>For Beginners:</b> Ridge Regression is like ordinary linear regression
/// but with a constraint that prevents coefficients from getting too large.
///
/// Think of it like this:
/// - Ordinary regression: Find the line that fits data best (no matter how)
/// - Ridge regression: Find the line that fits data well AND keeps coefficients reasonable
///
/// Why is this useful?
/// - Prevents overfitting (model too complex for training data)
/// - Handles correlated features better
/// - More stable predictions on new data
/// - Works even when you have more features than samples
///
/// The alpha parameter controls the trade-off:
/// - alpha = 0: Ordinary least squares (no regularization)
/// - Small alpha (0.01): Light regularization
/// - Large alpha (100): Heavy regularization, coefficients shrink toward zero
///
/// Real-world example: Predicting house prices with many correlated features
/// (square footage, number of rooms, lot size). Ridge prevents any single
/// feature from dominating the prediction.
/// </para>
/// </remarks>
public class RidgeRegression<T> : IRegression<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The regularization parameter (alpha).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alpha controls how much regularization to apply.
    ///
    /// - alpha = 0: No regularization (standard linear regression)
    /// - Small alpha (0.01 - 1.0): Light regularization
    /// - Medium alpha (1.0 - 10.0): Moderate regularization
    /// - Large alpha (10.0 - 100.0): Heavy regularization
    ///
    /// Higher alpha = smaller coefficients = simpler model = less overfitting
    /// But too high alpha can cause underfitting (model too simple).
    ///
    /// Typical approach: Try values [0.01, 0.1, 1, 10, 100] and use cross-validation
    /// to find the best alpha for your data.
    /// </para>
    /// </remarks>
    public T Alpha { get; set; }

    /// <summary>
    /// Whether to fit an intercept term.
    /// </summary>
    public bool FitIntercept { get; set; }

    /// <summary>
    /// The learned coefficient weights.
    /// </summary>
    public Vector<T>? Coefficients { get; private set; }

    /// <summary>
    /// The intercept term (bias).
    /// </summary>
    public T Intercept { get; private set; }

    /// <summary>
    /// Creates a new Ridge Regression model.
    /// </summary>
    /// <param name="alpha">The regularization parameter (default: 1.0).</param>
    /// <param name="fitIntercept">Whether to fit an intercept (default: true).</param>
    public RidgeRegression(T? alpha = null, bool fitIntercept = true)
    {
        Alpha = alpha ?? NumOps.One;
        FitIntercept = fitIntercept;
        Intercept = NumOps.Zero;
    }

    /// <summary>
    /// Trains the Ridge Regression model using the closed-form solution.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="y">The target values.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training finds the best coefficients using a mathematical formula.
    ///
    /// The algorithm:
    /// 1. If fit_intercept is true, center the data (subtract means)
    /// 2. Compute: w = (X^T X + α I)^(-1) X^T y
    /// 3. If fit_intercept is true, compute intercept from means
    ///
    /// The key insight: Adding α I to X^T X ensures the matrix is invertible,
    /// even when X has more columns than rows or has perfectly correlated features.
    ///
    /// This is a "closed-form" solution, meaning we can compute it directly
    /// without iterative optimization. This makes Ridge very fast to train.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of samples in X must match length of y");

        int n = X.Rows;
        int p = X.Columns;

        Matrix<T> XCentered = X;
        Vector<T> yCentered = y;
        Vector<T>? xMeans = null;
        T yMean = NumOps.Zero;

        // Center data if fitting intercept
        if (FitIntercept)
        {
            xMeans = new Vector<T>(p);
            XCentered = new Matrix<T>(n, p);

            // Calculate feature means
            for (int j = 0; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    sum = NumOps.Add(sum, X[i, j]);
                }
                xMeans[j] = NumOps.Divide(sum, NumOps.FromInt(n));
            }

            // Calculate target mean
            T sumY = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sumY = NumOps.Add(sumY, y[i]);
            }
            yMean = NumOps.Divide(sumY, NumOps.FromInt(n));

            // Center X and y
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    XCentered[i, j] = NumOps.Subtract(X[i, j], xMeans[j]);
                }
            }

            yCentered = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                yCentered[i] = NumOps.Subtract(y[i], yMean);
            }
        }

        // Compute X^T X
        Matrix<T> XtX = MatrixHelper<T>.Multiply(
            MatrixHelper<T>.Transpose(XCentered),
            XCentered
        );

        // Add α I to diagonal (regularization)
        for (int i = 0; i < p; i++)
        {
            XtX[i, i] = NumOps.Add(XtX[i, i], Alpha);
        }

        // Compute X^T y
        Vector<T> Xty = new Vector<T>(p);
        for (int j = 0; j < p; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(XCentered[i, j], yCentered[i]));
            }
            Xty[j] = sum;
        }

        // Solve (X^T X + α I) w = X^T y
        // Using matrix inversion (can be replaced with more efficient solver)
        Matrix<T> XtXInv = MatrixHelper<T>.Invert(XtX);
        Coefficients = MatrixHelper<T>.MultiplyVector(XtXInv, Xty);

        // Compute intercept if needed
        if (FitIntercept)
        {
            // intercept = y_mean - w^T x_means
            T intercept = yMean;
            for (int j = 0; j < p; j++)
            {
                intercept = NumOps.Subtract(intercept,
                    NumOps.Multiply(Coefficients[j], xMeans![j]));
            }
            Intercept = intercept;
        }
        else
        {
            Intercept = NumOps.Zero;
        }
    }

    /// <summary>
    /// Predicts target values for new data.
    /// </summary>
    /// <param name="X">The feature matrix to predict for.</param>
    /// <returns>A vector of predicted values.</returns>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (Coefficients == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        if (X.Columns != Coefficients.Length)
            throw new ArgumentException($"Expected {Coefficients.Length} features, got {X.Columns}");

        var predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T pred = Intercept;
            for (int j = 0; j < X.Columns; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(Coefficients[j], X[i, j]));
            }
            predictions[i] = pred;
        }

        return predictions;
    }

    /// <summary>
    /// Calculates the R² score (coefficient of determination).
    /// </summary>
    public T Score(Matrix<T> X, Vector<T> y)
    {
        var predictions = Predict(X);

        // Calculate mean of y
        T yMean = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            yMean = NumOps.Add(yMean, y[i]);
        }
        yMean = NumOps.Divide(yMean, NumOps.FromInt(y.Length));

        // Calculate SS_tot and SS_res
        T ssTot = NumOps.Zero;
        T ssRes = NumOps.Zero;

        for (int i = 0; i < y.Length; i++)
        {
            T residual = NumOps.Subtract(y[i], predictions[i]);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(residual, residual));

            T deviation = NumOps.Subtract(y[i], yMean);
            ssTot = NumOps.Add(ssTot, NumOps.Multiply(deviation, deviation));
        }

        // R² = 1 - (SS_res / SS_tot)
        return NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));
    }
}
```

### Step 2: Create LassoRegression with Coordinate Descent

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Regression\LassoRegression.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Lasso Regression with L1 regularization using Coordinate Descent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Lasso Regression adds an L1 penalty (sum of absolute coefficient values) to the
/// ordinary least squares objective. This penalty encourages sparsity, setting many
/// coefficients exactly to zero, thus performing automatic feature selection.
/// </para>
/// <para><b>For Beginners:</b> Lasso is like Ridge but with a key difference:
/// it can set coefficients exactly to zero, effectively removing features.
///
/// Think of it like cleaning out a closet:
/// - Ridge: Makes all items smaller (shrinks everything toward zero)
/// - Lasso: Throws away items you don't need (sets some coefficients to zero)
///
/// Why is this useful?
/// - Automatic feature selection (identifies important features)
/// - Creates simple, interpretable models
/// - Works well when many features are irrelevant
/// - Produces sparse models (many zero coefficients)
///
/// Example: Predicting disease from 10,000 genes
/// - Most genes are probably irrelevant
/// - Lasso automatically identifies the ~100 important genes
/// - Sets coefficients of irrelevant genes to exactly zero
///
/// The alpha parameter controls sparsity:
/// - Small alpha: Keeps more features (less sparse)
/// - Large alpha: Removes more features (more sparse)
/// </para>
/// </remarks>
public class LassoRegression<T> : IRegression<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The regularization parameter (alpha).
    /// </summary>
    public T Alpha { get; set; }

    /// <summary>
    /// Whether to fit an intercept term.
    /// </summary>
    public bool FitIntercept { get; set; }

    /// <summary>
    /// Maximum number of iterations for coordinate descent.
    /// </summary>
    public int MaxIterations { get; set; }

    /// <summary>
    /// Convergence tolerance for coordinate descent.
    /// </summary>
    public T Tolerance { get; set; }

    /// <summary>
    /// The learned coefficient weights.
    /// </summary>
    public Vector<T>? Coefficients { get; private set; }

    /// <summary>
    /// The intercept term (bias).
    /// </summary>
    public T Intercept { get; private set; }

    /// <summary>
    /// Creates a new Lasso Regression model.
    /// </summary>
    /// <param name="alpha">The regularization parameter (default: 1.0).</param>
    /// <param name="fitIntercept">Whether to fit an intercept (default: true).</param>
    /// <param name="maxIterations">Maximum iterations (default: 1000).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-4).</param>
    public LassoRegression(T? alpha = null, bool fitIntercept = true,
        int maxIterations = 1000, T? tolerance = null)
    {
        Alpha = alpha ?? NumOps.One;
        FitIntercept = fitIntercept;
        MaxIterations = maxIterations;
        Tolerance = tolerance ?? NumOps.FromDouble(1e-4);
        Intercept = NumOps.Zero;
    }

    /// <summary>
    /// Trains the Lasso Regression model using Coordinate Descent.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Coordinate Descent is an iterative optimization algorithm.
    ///
    /// Unlike Ridge (which has a closed-form solution), Lasso requires iteration because
    /// the L1 penalty makes the optimization problem non-differentiable at zero.
    ///
    /// How Coordinate Descent works:
    /// 1. Initialize all coefficients to zero (or random values)
    /// 2. Repeat until convergence:
    ///    a. For each coefficient:
    ///       - Fix all other coefficients
    ///       - Update this coefficient using soft-thresholding
    ///    b. Check if changes are smaller than tolerance
    /// 3. Stop when converged or max iterations reached
    ///
    /// Soft-thresholding is the key operation:
    /// - If |update| < α: Set coefficient to 0 (feature removed)
    /// - If update > α: Set coefficient to (update - α)
    /// - If update < -α: Set coefficient to (update + α)
    ///
    /// This is what creates sparsity (zero coefficients).
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of samples in X must match length of y");

        int n = X.Rows;
        int p = X.Columns;

        Matrix<T> XCentered = X;
        Vector<T> yCentered = y;
        Vector<T>? xMeans = null;
        T yMean = NumOps.Zero;

        // Center and normalize data if fitting intercept
        if (FitIntercept)
        {
            xMeans = new Vector<T>(p);
            XCentered = new Matrix<T>(n, p);

            // Calculate feature means
            for (int j = 0; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    sum = NumOps.Add(sum, X[i, j]);
                }
                xMeans[j] = NumOps.Divide(sum, NumOps.FromInt(n));
            }

            // Calculate target mean
            T sumY = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sumY = NumOps.Add(sumY, y[i]);
            }
            yMean = NumOps.Divide(sumY, NumOps.FromInt(n));

            // Center X and y
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    XCentered[i, j] = NumOps.Subtract(X[i, j], xMeans[j]);
                }
            }

            yCentered = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                yCentered[i] = NumOps.Subtract(y[i], yMean);
            }
        }

        // Initialize coefficients to zero
        Coefficients = new Vector<T>(p);

        // Precompute X column norms squared
        var xNormsSquared = new Vector<T>(p);
        for (int j = 0; j < p; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(XCentered[i, j], XCentered[i, j]));
            }
            xNormsSquared[j] = sum;
        }

        // Coordinate Descent
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            T maxChange = NumOps.Zero;

            for (int j = 0; j < p; j++)
            {
                T oldCoef = Coefficients[j];

                // Compute residual excluding feature j
                // r = y - X_(-j) * w_(-j)
                var residual = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    T pred = NumOps.Zero;
                    for (int k = 0; k < p; k++)
                    {
                        if (k != j)
                        {
                            pred = NumOps.Add(pred,
                                NumOps.Multiply(XCentered[i, k], Coefficients[k]));
                        }
                    }
                    residual[i] = NumOps.Subtract(yCentered[i], pred);
                }

                // Compute correlation: X_j^T * residual
                T correlation = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    correlation = NumOps.Add(correlation,
                        NumOps.Multiply(XCentered[i, j], residual[i]));
                }

                // Soft-thresholding update
                Coefficients[j] = SoftThreshold(correlation, Alpha);

                // Normalize by X_j^T X_j
                if (NumOps.GreaterThan(xNormsSquared[j], NumOps.Zero))
                {
                    Coefficients[j] = NumOps.Divide(Coefficients[j], xNormsSquared[j]);
                }

                // Track maximum change for convergence
                T change = NumOps.Abs(NumOps.Subtract(Coefficients[j], oldCoef));
                if (NumOps.GreaterThan(change, maxChange))
                {
                    maxChange = change;
                }
            }

            // Check convergence
            if (NumOps.LessThan(maxChange, Tolerance))
            {
                break;
            }
        }

        // Compute intercept if needed
        if (FitIntercept)
        {
            T intercept = yMean;
            for (int j = 0; j < p; j++)
            {
                intercept = NumOps.Subtract(intercept,
                    NumOps.Multiply(Coefficients[j], xMeans![j]));
            }
            Intercept = intercept;
        }
        else
        {
            Intercept = NumOps.Zero;
        }
    }

    /// <summary>
    /// Applies soft-thresholding operator for L1 penalty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Soft-thresholding is the magic behind Lasso's sparsity.
    ///
    /// Formula:
    /// - If x > α: return (x - α)
    /// - If x < -α: return (x + α)
    /// - If |x| <= α: return 0
    ///
    /// This creates a "dead zone" around zero where coefficients are set to exactly zero.
    /// This is how Lasso performs feature selection - coefficients below the threshold
    /// become zero and those features are removed from the model.
    /// </para>
    /// </remarks>
    private T SoftThreshold(T x, T threshold)
    {
        if (NumOps.GreaterThan(x, threshold))
        {
            return NumOps.Subtract(x, threshold);
        }
        else if (NumOps.LessThan(x, NumOps.Negate(threshold)))
        {
            return NumOps.Add(x, threshold);
        }
        else
        {
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Predicts target values for new data.
    /// </summary>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (Coefficients == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        if (X.Columns != Coefficients.Length)
            throw new ArgumentException($"Expected {Coefficients.Length} features, got {X.Columns}");

        var predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T pred = Intercept;
            for (int j = 0; j < X.Columns; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(Coefficients[j], X[i, j]));
            }
            predictions[i] = pred;
        }

        return predictions;
    }

    /// <summary>
    /// Calculates the R² score.
    /// </summary>
    public T Score(Matrix<T> X, Vector<T> y)
    {
        var predictions = Predict(X);

        T yMean = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            yMean = NumOps.Add(yMean, y[i]);
        }
        yMean = NumOps.Divide(yMean, NumOps.FromInt(y.Length));

        T ssTot = NumOps.Zero;
        T ssRes = NumOps.Zero;

        for (int i = 0; i < y.Length; i++)
        {
            T residual = NumOps.Subtract(y[i], predictions[i]);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(residual, residual));

            T deviation = NumOps.Subtract(y[i], yMean);
            ssTot = NumOps.Add(ssTot, NumOps.Multiply(deviation, deviation));
        }

        return NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));
    }

    /// <summary>
    /// Gets the number of non-zero coefficients (sparsity indicator).
    /// </summary>
    public int NonZeroCoefficients
    {
        get
        {
            if (Coefficients == null) return 0;

            int count = 0;
            for (int i = 0; i < Coefficients.Length; i++)
            {
                if (!NumOps.Equals(Coefficients[i], NumOps.Zero))
                {
                    count++;
                }
            }
            return count;
        }
    }
}
```

## Test Coverage Checklist

**Ridge Regression:**
- [ ] Perfect fit on simple linear data
- [ ] Coefficient shrinkage (compare to OLS)
- [ ] Handles multicollinearity (correlated features)
- [ ] Works with more features than samples
- [ ] Alpha = 0 gives OLS solution
- [ ] Large alpha shrinks coefficients toward zero
- [ ] R² score calculation

**Lasso Regression:**
- [ ] Sets some coefficients exactly to zero
- [ ] Feature selection (sparsity)
- [ ] Convergence of coordinate descent
- [ ] Soft-thresholding behavior
- [ ] Handles irrelevant features
- [ ] Compare sparsity at different alpha values
- [ ] NonZeroCoefficients property

## Common Mistakes to Avoid

1. **Not centering data**: Can cause intercept issues
2. **Forgetting to normalize features**: Lasso is scale-sensitive
3. **Wrong alpha range**: Try logarithmic scale (0.001, 0.01, 0.1, 1, 10, 100)
4. **Not checking convergence**: Coordinate descent needs sufficient iterations
5. **Comparing coefficients between Ridge and Lasso directly**: Different scales
6. **Using Lasso when features are highly correlated**: Ridge is better

## Learning Resources

- **Ridge Regression**: https://en.wikipedia.org/wiki/Ridge_regression
- **Lasso Regression**: https://en.wikipedia.org/wiki/Lasso_(statistics)
- **Coordinate Descent**: https://en.wikipedia.org/wiki/Coordinate_descent
- **Regularization**: https://scikit-learn.org/stable/modules/linear_model.html

## Validation Criteria

1. Ridge has closed-form solution (fast)
2. Lasso uses coordinate descent (iterative)
3. Lasso produces sparse solutions (zero coefficients)
4. Both handle regularization parameter alpha correctly
5. Test coverage 80%+
6. Proper intercept handling

---

**Good luck!** Ridge and Lasso are fundamental regularization techniques used in almost every ML pipeline.
