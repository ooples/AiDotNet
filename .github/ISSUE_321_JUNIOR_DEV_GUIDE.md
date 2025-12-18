# Issue 321: Optimized Gradient Boosting, GLMs, and Regression Module Refactor - Junior Developer Implementation Guide

## Table of Contents
1. [Understanding Regression Models](#understanding-regression-models)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation Guide](#implementation-guide)
4. [Testing Strategy](#testing-strategy)
5. [Performance Considerations](#performance-considerations)

---

## Understanding Regression Models

### What is Regression?

**For Beginners:** Regression is about predicting continuous numerical values (like house prices, temperatures, or stock values) based on input features.

Think of regression as:
- **Input:** Features/attributes (square footage, number of bedrooms, location)
- **Output:** A continuous number (house price: $350,000)
- **Goal:** Find the relationship between inputs and outputs

**Contrast with Classification:**
- Classification: Predicts categories (spam/not spam, cat/dog)
- Regression: Predicts numbers (price, temperature, distance)

### Types of Regression Models

1. **Linear Models:** Assume a straight-line (or hyperplane) relationship
   - Simple Linear Regression: y = mx + b
   - Multiple Regression: y = β₀ + β₁x₁ + β₂x₂ + ...

2. **Generalized Linear Models (GLMs):** Extend linear models for different distributions
   - Poisson Regression: For count data (number of events)
   - Gamma Regression: For positive, skewed data (insurance claims, rainfall)
   - Tweedie Regression: For data with zeros and positive values (insurance, rainfall with dry days)

3. **Tree-Based Models:** Use decision trees
   - Gradient Boosting: Sequentially combine many trees
   - XGBoost, LightGBM, CatBoost: Optimized gradient boosting frameworks

4. **Ordinal Regression:** For ordered categories
   - Examples: Survey ratings (1-5 stars), education level (high school < bachelor < master < PhD)

---

## Mathematical Foundations

### 1. Generalized Linear Models (GLMs)

GLMs extend linear regression by:
1. Allowing the response variable to follow different distributions (not just Gaussian)
2. Using a link function to connect the linear predictor to the mean of the response

**Structure:**
```
Linear predictor: η = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
Link function: g(μ) = η
Mean: μ = g⁻¹(η)
```

#### Gamma Regression

**Purpose:** Model positive, right-skewed continuous data (e.g., insurance claims, income, rainfall amounts).

**Distribution:** Response variable Y follows Gamma(α, β) distribution
```
PDF: f(y; α, β) = (β^α / Γ(α)) × y^(α-1) × exp(-βy)
Mean: E[Y] = α/β
Variance: Var[Y] = α/β²
```

**Link Function (Log Link - recommended):**
```
log(μ) = η = X × β
μ = exp(X × β)
```

**When to Use:**
- Positive continuous data (cannot be zero or negative)
- Right-skewed distribution (long tail to the right)
- Variance increases with the mean
- Examples: Insurance claim amounts, survival times, income

**Training (Maximum Likelihood Estimation):**
```
Log-likelihood: ℓ(β) = Σᵢ [αᵢ × log(βᵢ) - log(Γ(αᵢ)) + (αᵢ-1)×log(yᵢ) - βᵢ×yᵢ]
where: μᵢ = exp(xᵢᵀ × β)
       βᵢ = α/μᵢ (shape/scale parameterization)

Optimize: β* = argmax ℓ(β)
```

Use iteratively reweighted least squares (IRLS) or gradient-based optimization.

#### Tweedie Regression

**Purpose:** Model data with a mixture of zeros and positive values (e.g., insurance claims where many people have zero claims).

**Distribution:** Tweedie distribution is a family parameterized by power parameter p:
- p = 0: Normal distribution
- p = 1: Poisson distribution
- 1 < p < 2: Compound Poisson-Gamma (most common use case)
- p = 2: Gamma distribution
- p = 3: Inverse Gaussian

**Key Property:** Can model exact zeros with positive mass at zero, unlike Gamma.

**Link Function (Log Link):**
```
log(μ) = η = X × β
μ = exp(X × β)
```

**When to Use:**
- Data has many zeros (e.g., 80% of customers have zero claims)
- Positive values are right-skewed
- Examples: Insurance claims (many people have no claims), precipitation (many dry days), customer purchases (many non-buyers)

**Training:**
```
Log-likelihood: ℓ(β; p, φ) = Σᵢ [yᵢθᵢ - b(θᵢ)] / φ - c(yᵢ, φ)
where: θᵢ is the canonical parameter
       b(θᵢ) is the cumulant function
       φ is the dispersion parameter

Power parameter p ∈ (1, 2) typically set to 1.5 or estimated from data
```

#### Ordinal Regression

**Purpose:** Predict ordered categories (e.g., "poor" < "fair" < "good" < "excellent").

**Key Difference from Multi-Class Classification:**
- Multi-class: Categories are unordered (cat, dog, bird)
- Ordinal: Categories have a natural order (cold < warm < hot)

**Proportional Odds Model (Most Common):**
```
P(Y ≤ j | X) = logit⁻¹(θⱼ - X × β)
where:
  θⱼ are cutoff parameters (threshold between categories)
  β are feature coefficients (same for all categories - "proportional odds")
```

**Interpretation:**
- θⱼ: Thresholds that separate categories
- β: How features affect the odds of being in higher categories

**Example:**
Survey rating (1-5 stars):
- θ₁ = -2.0: Threshold between 1 and 2 stars
- θ₂ = -0.5: Threshold between 2 and 3 stars
- θ₃ = 1.0: Threshold between 3 and 4 stars
- θ₄ = 2.5: Threshold between 4 and 5 stars
- β = [0.5, -0.3]: Feature coefficients (e.g., price increases rating, wait time decreases)

**Training:**
```
Maximize log-likelihood:
ℓ(θ, β) = Σᵢ log[P(Yᵢ = yᵢ | Xᵢ; θ, β)]

where: P(Y = j | X) = P(Y ≤ j | X) - P(Y ≤ j-1 | X)
```

### 2. Optimized Gradient Boosting

The existing `GradientBoostingRegression` in AiDotNet is a basic implementation. XGBoost, LightGBM, and CatBoost are highly optimized frameworks with advanced features.

#### XGBoost (Extreme Gradient Boosting)

**Key Innovations:**
1. **Regularization:** Adds L1 (Lasso) and L2 (Ridge) penalties to prevent overfitting
2. **Tree Pruning:** Uses max_depth and prunes trees backward
3. **Parallel Processing:** Parallelizes tree construction
4. **Hardware Optimization:** Efficient memory usage and cache-aware algorithms
5. **Sparsity Awareness:** Handles missing values automatically

**Objective Function:**
```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

where:
  L(yᵢ, ŷᵢ) = loss function (e.g., MSE)
  Ω(fₖ) = γT + (λ/2)||w||² (regularization)
  T = number of leaves
  w = leaf weights
```

**Why Use XGBoost:**
- State-of-the-art performance on structured/tabular data
- Handles missing values automatically
- Built-in regularization prevents overfitting
- Very fast training and prediction
- Extensive hyperparameter control

#### LightGBM (Light Gradient Boosting Machine)

**Key Innovations:**
1. **Leaf-wise Growth:** Grows trees leaf-by-leaf (not level-by-level like XGBoost)
2. **Histogram-based Learning:** Bins continuous features into discrete bins
3. **GOSS (Gradient-based One-Side Sampling):** Keeps instances with large gradients
4. **EFB (Exclusive Feature Bundling):** Bundles mutually exclusive features

**Advantages:**
- Faster training than XGBoost (especially on large datasets)
- Lower memory usage
- Better accuracy on some datasets
- Supports categorical features natively

**Trade-offs:**
- Leaf-wise growth can overfit on small datasets
- Requires careful tuning of max_depth

**Why Use LightGBM:**
- Fastest among the three for large datasets (N > 10,000)
- Best for datasets with many features (D > 100)
- Native categorical feature support

#### CatBoost (Categorical Boosting)

**Key Innovations:**
1. **Ordered Boosting:** Prevents prediction shift (target leakage)
2. **Categorical Features:** Handles categorical variables automatically with novel encoding
3. **Symmetric Trees:** Builds balanced trees (oblivious trees)
4. **No Hyperparameter Tuning:** Good defaults that work out-of-the-box

**Categorical Feature Handling:**
- Target-based encoding with smoothing
- Prevents overfitting on high-cardinality categories
- No need for manual encoding (one-hot, label encoding)

**Why Use CatBoost:**
- Best for datasets with many categorical features
- Robust to hyperparameter choices (less tuning needed)
- Often achieves good results with default parameters
- Better on smaller datasets than LightGBM

### 3. Comparison: XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed (large data) | Fast | Fastest | Fast |
| Memory Usage | Moderate | Low | Moderate |
| Categorical Features | Manual encoding | Good | Best |
| Small Datasets | Best | Risk of overfit | Good |
| Large Datasets | Good | Best | Good |
| Hyperparameter Tuning | Needed | Needed | Less needed |
| Default Performance | Good | Good | Best |

**General Recommendations:**
- **XGBoost:** When you need reliability and have time for tuning
- **LightGBM:** When speed is critical and data is large
- **CatBoost:** When you have categorical features or want good defaults

---

## Implementation Guide

### Phase 1: Refactor Classification Models

#### Step 1.1: Create Classification Module

**Directory:** `C:\Users\cheat\source\repos\AiDotNet\src\Classification\`

This is straightforward - just move files and update namespaces.

**Files to Move:**
```
src/Regression/LogisticRegression.cs → src/Classification/LogisticRegression.cs
src/Regression/MultinomialLogisticRegression.cs → src/Classification/MultinomialLogisticRegression.cs
```

**Namespace Update:**
```csharp
// Old namespace
namespace AiDotNet.Regression;

// New namespace
namespace AiDotNet.Classification;
```

**Update References:**
Search the codebase for imports:
```
using AiDotNet.Regression;
```

Replace with:
```
using AiDotNet.Classification;
```

### Phase 2: Implement Optimized Gradient Boosting Frameworks

#### Integration Strategy

**Important:** XGBoost, LightGBM, and CatBoost are C++ libraries with .NET wrappers. Integration options:

**Option 1: P/Invoke (Native Interop)**
```csharp
[DllImport("xgboost.dll", CallingConvention = CallingConvention.Cdecl)]
private static extern IntPtr XGBoosterCreate(IntPtr[] dmats, ulong len);

// Pros: Maximum performance, direct control
// Cons: Complex, platform-specific, requires native DLLs
```

**Option 2: Use Existing .NET Bindings**
```csharp
// XGBoost: Microsoft.ML.FastTree or XGBoost.NET
// LightGBM: Microsoft.ML.LightGBM or LightGBM.NET
// CatBoost: CatBoostNet

// Pros: Easier integration, maintained by community
// Cons: Additional dependency, may not expose all features
```

**Recommended Approach:** Create wrapper classes that use existing .NET bindings while conforming to AiDotNet architecture.

#### Step 2.1: XGBoost Wrapper

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Regression\XGBoostRegressor.cs`

**Interface:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IXGBoostRegressor.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for XGBoost regression models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> XGBoost is an optimized gradient boosting framework.
///
/// Key features:
/// - Regularization: Prevents overfitting with L1/L2 penalties
/// - Parallel processing: Fast training using multiple CPU cores
/// - Handling missing values: Automatically learns best direction for missing values
/// - Tree pruning: Removes branches that don't improve the model
///
/// XGBoost is often the first choice for tabular data competitions and real-world applications.
/// </remarks>
public interface IXGBoostRegressor<T> : IRegression<T>
{
    /// <summary>
    /// Gets or sets the maximum depth of trees.
    /// </summary>
    /// <remarks>
    /// Default: 6 (based on XGBoost defaults)
    /// Range: [1, ∞), typically 3-10
    /// Higher values: More complex model, risk of overfitting
    /// Lower values: Simpler model, may underfit
    /// </remarks>
    int MaxDepth { get; set; }

    /// <summary>
    /// Gets or sets the learning rate (eta).
    /// </summary>
    /// <remarks>
    /// Default: 0.3 (based on XGBoost defaults)
    /// Range: [0, 1]
    /// Lower values: More robust, requires more trees
    /// Higher values: Faster training, risk of overfitting
    /// Typical production values: 0.01-0.1
    /// </remarks>
    double LearningRate { get; set; }

    /// <summary>
    /// Gets or sets the number of boosting rounds (trees).
    /// </summary>
    /// <remarks>
    /// Default: 100
    /// Range: [1, ∞), typically 50-1000
    /// More trees: Better fit, longer training, risk of overfitting
    /// Use early stopping to find optimal number
    /// </remarks>
    int NumBoostRound { get; set; }

    /// <summary>
    /// Gets or sets the L1 regularization term (alpha).
    /// </summary>
    /// <remarks>
    /// Default: 0 (no L1 regularization)
    /// Range: [0, ∞)
    /// Higher values: More regularization, sparser models
    /// </remarks>
    double Alpha { get; set; }

    /// <summary>
    /// Gets or sets the L2 regularization term (lambda).
    /// </summary>
    /// <remarks>
    /// Default: 1.0 (based on XGBoost defaults)
    /// Range: [0, ∞)
    /// Higher values: More regularization, smoother models
    /// </remarks>
    double Lambda { get; set; }
}
```

**Implementation Template:**

```csharp
namespace AiDotNet.Regression;

/// <summary>
/// XGBoost regression implementation using native XGBoost library.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This class wraps the XGBoost C++ library for use in AiDotNet.
///
/// XGBoost uses gradient boosting to build an ensemble of decision trees:
/// 1. Start with a simple prediction (mean of target values)
/// 2. Build a tree to predict the residuals (errors)
/// 3. Add the tree's predictions (scaled by learning rate) to the model
/// 4. Repeat for num_boost_round iterations
///
/// The key difference from basic gradient boosting:
/// - Adds regularization to prevent overfitting
/// - Uses second-order gradients (Newton method) for better optimization
/// - Prunes trees to remove unnecessary splits
/// - Handles missing values automatically
///
/// Default values are based on XGBoost documentation and common practices.
/// </remarks>
public class XGBoostRegressor<T> : RegressionBase<T>, IXGBoostRegressor<T>
{
    private readonly INumericOperations<T> _numOps;

    // XGBoost hyperparameters with sensible defaults
    public int MaxDepth { get; set; } = 6;
    public double LearningRate { get; set; } = 0.3;
    public int NumBoostRound { get; set; } = 100;
    public double Alpha { get; set; } = 0.0;
    public double Lambda { get; set; } = 1.0;

    // Native XGBoost handles (platform-dependent)
    private IntPtr _boosterHandle = IntPtr.Zero;
    private Matrix<T> _xTrain = new Matrix<T>(0, 0);
    private Vector<T> _yTrain = new Vector<T>(0);

    public XGBoostRegressor()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        _xTrain = x;
        _yTrain = y;

        // Convert data to XGBoost DMatrix format
        // Create booster with specified hyperparameters
        // Train for num_boost_round iterations
        // Store the trained booster handle
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Convert input to DMatrix format
        // Call XGBoost predict API
        // Convert results back to Vector<T>
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.XGBoost,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", MaxDepth },
                { "LearningRate", LearningRate },
                { "NumBoostRound", NumBoostRound },
                { "Alpha", Alpha },
                { "Lambda", Lambda }
            }
        };
    }

    // Implement Serialize/Deserialize using XGBoost's save/load model APIs
}
```

**Important Notes:**

1. **Native Library Handling:**
   - Include xgboost.dll (Windows), libxgboost.so (Linux), libxgboost.dylib (Mac)
   - Set up proper DLL loading based on platform
   - Handle IntPtr memory management carefully (dispose properly)

2. **Data Conversion:**
   - XGBoost expects float32 data (convert T to float if needed)
   - Handle sparse matrices efficiently (XGBoost supports CSR format)

3. **Error Handling:**
   - Wrap native calls with try-catch
   - Check return codes from XGBoost API
   - Provide meaningful error messages

#### Step 2.2: LightGBM Wrapper

Similar structure to XGBoost, but with LightGBM-specific parameters:

**Key Differences:**
- `num_leaves` instead of `max_depth` (default: 31)
- `min_data_in_leaf` for regularization (default: 20)
- `bagging_fraction` for row subsampling (default: 1.0)
- `feature_fraction` for column subsampling (default: 1.0)

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Regression\LightGBMRegressor.cs`

#### Step 2.3: CatBoost Wrapper

**Key Differences:**
- `iterations` instead of `num_boost_round` (default: 1000)
- `depth` instead of `max_depth` (default: 6)
- `l2_leaf_reg` for regularization (default: 3.0)
- Native categorical feature support (pass categorical column indices)

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Regression\CatBoostRegressor.cs`

### Phase 3: Implement Generalized Linear Models

#### Step 3.1: Gamma Regression

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Regression\GammaRegression.cs`

**Interface:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IGammaRegression.cs`

```csharp
namespace AiDotNet.Regression;

/// <summary>
/// Implements Gamma regression for modeling positive, right-skewed continuous data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Gamma regression is used when your target variable is:
/// - Always positive (greater than zero)
/// - Right-skewed (most values are small, few are very large)
/// - Has variance that increases with the mean
///
/// Real-world examples:
/// - Insurance claim amounts (most claims are small, few are very large)
/// - Rainfall amounts (excluding dry days)
/// - Income (most people earn modest amounts, few earn millions)
/// - Survival times in medical studies
///
/// The model uses:
/// - Gamma distribution to model the response
/// - Log link function to ensure predictions are always positive
/// - Maximum likelihood estimation to find the best parameters
///
/// Default values:
/// - Link function: Log (recommended for Gamma regression)
/// - Max iterations: 100 (for IRLS algorithm)
/// - Convergence tolerance: 1e-6
/// </remarks>
public class GammaRegression<T> : RegressionBase<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _maxIterations;
    private readonly T _convergenceTolerance;

    private Vector<T> _coefficients = new Vector<T>(0);
    private T _shapeParameter; // Gamma distribution shape parameter (α)

    /// <summary>
    /// Initializes a new instance of Gamma regression.
    /// </summary>
    /// <param name="maxIterations">
    /// Maximum number of iterations for IRLS algorithm.
    /// Default: 100 (based on statsmodels and R's glm defaults).
    /// </param>
    /// <param name="convergenceTolerance">
    /// Convergence criterion for coefficient changes.
    /// Default: 1e-6 (standard for numerical optimization).
    /// </param>
    public GammaRegression(
        int maxIterations = 100,
        T? convergenceTolerance = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _maxIterations = maxIterations;
        _convergenceTolerance = convergenceTolerance ?? _numOps.FromDouble(1e-6);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Validate: all y values must be positive
        for (int i = 0; i < y.Length; i++)
        {
            if (_numOps.LessThanOrEquals(y[i], _numOps.Zero))
            {
                throw new ArgumentException(
                    "Gamma regression requires all target values to be positive (> 0). " +
                    $"Found non-positive value at index {i}: {y[i]}");
            }
        }

        // Add intercept column
        Matrix<T> X = AddInterceptColumn(x);
        int n = X.Rows;
        int p = X.Columns;

        // Initialize coefficients (use log(mean(y)) for intercept)
        _coefficients = new Vector<T>(p);
        T meanY = y.Sum();
        meanY = _numOps.Divide(meanY, _numOps.FromDouble(n));
        _coefficients[0] = _numOps.Log(meanY); // Log link

        // Iteratively Reweighted Least Squares (IRLS)
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Step 1: Compute linear predictor η = X × β
            Vector<T> eta = X.Multiply(_coefficients);

            // Step 2: Compute mean μ = exp(η) (inverse log link)
            Vector<T> mu = eta.Select(_numOps.Exp);

            // Step 3: Compute working response z = η + (y - μ) / μ
            Vector<T> z = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                T residual = _numOps.Subtract(y[i], mu[i]);
                T adjustment = _numOps.Divide(residual, mu[i]);
                z[i] = _numOps.Add(eta[i], adjustment);
            }

            // Step 4: Compute weights W = diag(μ²) (for Gamma with log link)
            Vector<T> weights = mu.Select(m => _numOps.Multiply(m, m));

            // Step 5: Solve weighted least squares: β_new = (X^T W X)^(-1) X^T W z
            Matrix<T> XtW = X.Transpose().MultiplyDiagonal(weights);
            Matrix<T> XtWX = XtW.Multiply(X);
            Vector<T> XtWz = XtW.Multiply(z);

            // Solve system using Cholesky decomposition (XtWX is positive definite)
            var chol = new CholeskyDecomposition<T>(XtWX);
            Vector<T> coefficientsNew = chol.Solve(XtWz);

            // Step 6: Check convergence
            T maxChange = _numOps.Zero;
            for (int i = 0; i < p; i++)
            {
                T change = _numOps.Abs(_numOps.Subtract(coefficientsNew[i], _coefficients[i]));
                if (_numOps.GreaterThan(change, maxChange))
                {
                    maxChange = change;
                }
            }

            _coefficients = coefficientsNew;

            if (_numOps.LessThan(maxChange, _convergenceTolerance))
            {
                break; // Converged
            }
        }

        // Estimate shape parameter α using method of moments
        // α = mean² / variance
        Vector<T> predictions = Predict(x);
        T meanSquared = _numOps.Zero;
        T variance = _numOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T residual = _numOps.Subtract(y[i], predictions[i]);
            meanSquared = _numOps.Add(meanSquared, _numOps.Multiply(predictions[i], predictions[i]));
            variance = _numOps.Add(variance, _numOps.Multiply(residual, residual));
        }

        meanSquared = _numOps.Divide(meanSquared, _numOps.FromDouble(n));
        variance = _numOps.Divide(variance, _numOps.FromDouble(n - p));
        _shapeParameter = _numOps.Divide(meanSquared, variance);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Add intercept column
        Matrix<T> X = AddInterceptColumn(input);

        // Compute η = X × β
        Vector<T> eta = X.Multiply(_coefficients);

        // Apply inverse link: μ = exp(η)
        return eta.Select(_numOps.Exp);
    }

    private Matrix<T> AddInterceptColumn(Matrix<T> x)
    {
        var result = new Matrix<T>(x.Rows, x.Columns + 1);
        for (int i = 0; i < x.Rows; i++)
        {
            result[i, 0] = _numOps.One; // Intercept
            for (int j = 0; j < x.Columns; j++)
            {
                result[i, j + 1] = x[i, j];
            }
        }
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GammaRegression,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Coefficients", _coefficients.ToArray() },
                { "ShapeParameter", _shapeParameter },
                { "MaxIterations", _maxIterations },
                { "ConvergenceTolerance", _convergenceTolerance }
            }
        };
    }
}
```

#### Step 3.2: Tweedie Regression

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Regression\TweedieRegression.cs`

**Key Implementation Details:**

```csharp
/// <summary>
/// Implements Tweedie regression for modeling data with zeros and positive values.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Tweedie regression is perfect for data that has:
/// - Many exact zeros (e.g., 80% of customers make zero purchases)
/// - Positive values that are right-skewed
///
/// Examples:
/// - Insurance claims (many people have no claims, some have large claims)
/// - Customer purchases (many people buy nothing, some buy a lot)
/// - Precipitation (many days have zero rain, some have heavy rainfall)
///
/// The power parameter p controls the distribution:
/// - p = 1.5: Common choice for insurance (recommended default)
/// - p closer to 1: More weight on zeros
/// - p closer to 2: Behaves more like Gamma (less weight on zeros)
///
/// Default values:
/// - Power parameter p: 1.5 (based on scikit-learn and insurance literature)
/// - Link function: Log
/// - Max iterations: 100
/// </remarks>
public class TweedieRegression<T> : RegressionBase<T>
{
    private readonly double _powerParameter; // p ∈ (1, 2)
    private readonly int _maxIterations;
    private readonly T _convergenceTolerance;

    public TweedieRegression(
        double powerParameter = 1.5,
        int maxIterations = 100,
        T? convergenceTolerance = default)
    {
        if (powerParameter <= 1.0 || powerParameter >= 2.0)
        {
            throw new ArgumentException(
                $"Power parameter must be in (1, 2) for compound Poisson-Gamma. Got: {powerParameter}. " +
                "Use p=1 for Poisson, p=2 for Gamma, or 1<p<2 for Tweedie.");
        }

        _powerParameter = powerParameter;
        _maxIterations = maxIterations;
        _convergenceTolerance = convergenceTolerance ?? _numOps.FromDouble(1e-6);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Similar IRLS algorithm to Gamma regression
        // Key difference: Use Tweedie variance function V(μ) = μ^p
        // Weights: W_i = μ_i^(2-p) (instead of μ² for Gamma)
    }
}
```

### Phase 4: Implement Ordinal Regression

**File:** `C:\Users\cheat\source\repos\AiDotNet\src\Regression\OrdinalRegression.cs`

**Interface:** Could be `IOrdinalClassifier<T>` (new interface) or extend `IRegression<T>`

```csharp
/// <summary>
/// Implements ordinal regression using the proportional odds model.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Ordinal regression is for predicting ordered categories.
///
/// Examples:
/// - Survey ratings: "strongly disagree" < "disagree" < "neutral" < "agree" < "strongly agree"
/// - Education level: "high school" < "bachelor" < "master" < "PhD"
/// - Movie ratings: 1 star < 2 stars < 3 stars < 4 stars < 5 stars
///
/// The key difference from regular classification:
/// - Regular classification: Treats categories as unrelated (cat vs dog vs bird)
/// - Ordinal regression: Respects the order (poor < fair < good)
///
/// The proportional odds model learns:
/// - Thresholds between categories (θ₁, θ₂, ..., θ_{K-1})
/// - Feature coefficients (β) that apply to all categories
///
/// Default values:
/// - Max iterations: 100 (for numerical optimization)
/// - Optimizer: L-BFGS (standard for logistic models)
/// </remarks>
public class OrdinalRegression<T> : RegressionBase<T>
{
    private Vector<T> _thresholds = new Vector<T>(0); // θ₁, θ₂, ..., θ_{K-1}
    private Vector<T> _coefficients = new Vector<T>(0); // β

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Determine number of categories K
        int K = (int)y.Max() + 1; // Assuming y ∈ {0, 1, ..., K-1}

        // Step 2: Initialize thresholds and coefficients
        // Thresholds: Use quantiles of the logit-transformed target
        // Coefficients: Start with zeros

        // Step 3: Maximize log-likelihood using optimizer
        // Constraints: θ₁ < θ₂ < ... < θ_{K-1} (enforce monotonicity)

        // Step 4: Store optimized parameters
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Return predicted category (argmax of probabilities)
        Matrix<T> probs = PredictProbabilities(input);
        // For each row, find category with highest probability
    }

    public Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        // For each sample and category:
        // P(Y = j | X) = P(Y ≤ j | X) - P(Y ≤ j-1 | X)
        // where P(Y ≤ j | X) = sigmoid(θⱼ - X×β)
    }
}
```

---

## Testing Strategy

### Unit Testing Checklist

#### Gamma Regression Tests

```csharp
[TestClass]
public class GammaRegressionTests
{
    [TestMethod]
    public void TestGammaRegression_SimulatedData()
    {
        // Generate data from known Gamma distribution
        // True parameters: β = [0.5, 1.0], α = 2.0
        // Generate X ~ N(0, 1)
        // Generate Y ~ Gamma(α, β₀ + β₁×X)

        var gamma = new GammaRegression<double>();
        gamma.Train(X, Y);

        // Assert: Recovered coefficients close to true values
        // Assert: Predictions have low MSE
    }

    [TestMethod]
    public void TestGammaRegression_RejectsNonPositiveTargets()
    {
        var X = new Matrix<double>(100, 2);
        var Y = new Vector<double>(100);
        Y[50] = 0.0; // Insert zero value

        var gamma = new GammaRegression<double>();

        // Assert: Throws ArgumentException with helpful message
        Assert.ThrowsException<ArgumentException>(() => gamma.Train(X, Y));
    }

    [TestMethod]
    public void TestGammaRegression_InsuranceClaims()
    {
        // Use real or realistic insurance claims data
        // Features: age, driving history, car type
        // Target: claim amount (positive values only)

        // Assert: R² > 0.5 (reasonable fit)
        // Assert: Predictions are always positive
    }
}
```

#### Tweedie Regression Tests

```csharp
[TestClass]
public class TweedieRegressionTests
{
    [TestMethod]
    public void TestTweedieRegression_WithZeros()
    {
        // Generate data with 70% zeros and 30% positive values
        var X = new Matrix<double>(1000, 3);
        var Y = new Vector<double>(1000);

        // Fill 70% with zeros
        for (int i = 0; i < 700; i++) Y[i] = 0.0;

        // Fill 30% with Gamma-distributed positive values
        for (int i = 700; i < 1000; i++)
        {
            Y[i] = GenerateGammaRandom(2.0, 3.0);
        }

        var tweedie = new TweedieRegression<double>(powerParameter: 1.5);
        tweedie.Train(X, Y);

        // Assert: Model handles zeros correctly
        // Assert: Predictions for "zero-like" inputs are close to zero
        // Assert: Predictions for "positive-like" inputs are positive
    }

    [TestMethod]
    public void TestTweedieRegression_PowerParameterValidation()
    {
        // Assert: p = 0.5 throws exception (outside valid range)
        Assert.ThrowsException<ArgumentException>(
            () => new TweedieRegression<double>(powerParameter: 0.5));

        // Assert: p = 2.5 throws exception (use Gamma instead)
        Assert.ThrowsException<ArgumentException>(
            () => new TweedieRegression<double>(powerParameter: 2.5));
    }
}
```

#### Ordinal Regression Tests

```csharp
[TestClass]
public class OrdinalRegressionTests
{
    [TestMethod]
    public void TestOrdinalRegression_SurveyRatings()
    {
        // Simulate survey data: ratings 1-5
        // Features: service quality, wait time, price
        // Target: rating (1 = worst, 5 = best)

        var ordinal = new OrdinalRegression<double>();
        ordinal.Train(X, Y);

        // Assert: Thresholds are monotonically increasing
        // Assert: Predictions respect ordering
        // Assert: Mean absolute error < 1.0 (on 1-5 scale)
    }

    [TestMethod]
    public void TestOrdinalRegression_EducationLevel()
    {
        // Predict education level based on demographics
        // 0 = high school, 1 = bachelor, 2 = master, 3 = PhD

        // Assert: Coefficients make sense (e.g., age increases education)
        // Assert: Predicted probabilities sum to 1 for each sample
    }

    [TestMethod]
    public void TestOrdinalRegression_ThresholdMonotonicity()
    {
        // After training, check that θ₁ < θ₂ < θ₃ < ...
        var thresholds = ordinal.GetThresholds();
        for (int i = 0; i < thresholds.Length - 1; i++)
        {
            Assert.IsTrue(thresholds[i] < thresholds[i + 1],
                $"Thresholds must be increasing: θ[{i}]={thresholds[i]} >= θ[{i+1}]={thresholds[i+1]}");
        }
    }
}
```

#### XGBoost/LightGBM/CatBoost Tests

```csharp
[TestClass]
public class OptimizedGradientBoostingTests
{
    [TestMethod]
    public void TestXGBoost_BasicRegression()
    {
        // Use Boston housing or similar dataset
        var xgb = new XGBoostRegressor<double>();
        xgb.Train(X_train, Y_train);

        var predictions = xgb.Predict(X_test);

        // Assert: R² > 0.7
        // Assert: Predictions are reasonable (not NaN, not extreme)
    }

    [TestMethod]
    public void TestXGBoost_SerializationDeserialization()
    {
        var xgb = new XGBoostRegressor<double>();
        xgb.Train(X, Y);

        byte[] modelBytes = xgb.Serialize();
        var xgbLoaded = new XGBoostRegressor<double>();
        xgbLoaded.Deserialize(modelBytes);

        // Assert: Predictions from original and loaded model are identical
    }

    [TestMethod]
    public void TestLightGBM_LargeDataset()
    {
        // Generate large dataset (N=100,000, D=50)
        var lgbm = new LightGBMRegressor<double>();

        // Measure training time
        var stopwatch = Stopwatch.StartNew();
        lgbm.Train(X, Y);
        stopwatch.Stop();

        // Assert: Training completes in reasonable time (< 60 seconds)
        // Assert: Model achieves good performance
    }

    [TestMethod]
    public void TestCatBoost_CategoricalFeatures()
    {
        // Create dataset with categorical features
        // e.g., ["red", "blue", "green"], ["small", "medium", "large"]

        var catboost = new CatBoostRegressor<double>();
        catboost.CategoricalFeatureIndices = new[] { 0, 2 }; // Columns 0 and 2 are categorical

        catboost.Train(X, Y);

        // Assert: Model handles categorical features without manual encoding
        // Assert: Performance is good
    }

    [TestMethod]
    public void TestComparison_XGBoostVsLightGBMVsCatBoost()
    {
        // Train all three on the same dataset
        // Compare:
        // - Training time
        // - Prediction accuracy (R²)
        // - Memory usage

        // Assert: All three achieve similar accuracy (within 5%)
        // Assert: LightGBM is fastest on large data
    }
}
```

---

## Performance Considerations

### Computational Complexity

| Model | Training Time | Prediction Time | Memory |
|-------|---------------|-----------------|--------|
| Gamma Regression | O(p² × n × iter) | O(p × n) | O(p²) |
| Tweedie Regression | O(p² × n × iter) | O(p × n) | O(p²) |
| Ordinal Regression | O(p × n × K × iter) | O(p × n × K) | O(p × K) |
| XGBoost | O(n × d × T × log(n)) | O(n × T × log(T)) | O(n × d) |
| LightGBM | O(n × d × T) | O(n × T) | O(n × d/8) |
| CatBoost | O(n × d × T × log(n)) | O(n × T) | O(n × d) |

Where:
- n = number of samples
- p = number of features
- K = number of categories (ordinal)
- T = number of trees
- d = tree depth
- iter = optimization iterations

### Optimization Tips

#### For GLMs (Gamma, Tweedie)

1. **Feature Scaling:** Not strictly necessary, but helps convergence
   ```csharp
   // Standardize features: (x - mean) / std
   var scaler = new StandardScaler<T>();
   Matrix<T> X_scaled = scaler.FitTransform(X);
   ```

2. **Starting Values:** Use good initial coefficient estimates
   ```csharp
   // For Gamma with log link: start with log(mean(y))
   _coefficients[0] = _numOps.Log(y.Mean());
   ```

3. **Convergence:** Monitor log-likelihood changes, not just coefficient changes
   ```csharp
   T logLikOld = ComputeLogLikelihood(coefficients, X, y);
   // ... update coefficients ...
   T logLikNew = ComputeLogLikelihood(coefficientsNew, X, y);
   if (_numOps.Abs(_numOps.Subtract(logLikNew, logLikOld)) < tolerance)
       break;
   ```

#### For Gradient Boosting

1. **Early Stopping:** Stop training when validation performance plateaus
   ```csharp
   // Split data into train/validation
   // Monitor validation RMSE each iteration
   // Stop if no improvement for N rounds
   ```

2. **Learning Rate Scheduling:** Use smaller learning rates with more trees
   ```
   Best practice: learning_rate × num_trees ≈ 30
   Example: lr=0.1, trees=300 or lr=0.01, trees=3000
   ```

3. **Memory Management:** For large datasets, use streaming or chunking
   ```csharp
   // Process data in batches if memory is limited
   for (int batch = 0; batch < numBatches; batch++)
   {
       var XBatch = LoadBatch(batch);
       // Train on batch
   }
   ```

---

## Complexity Estimates

### Issue 321 Breakdown

| Task | Story Points | Estimated Hours | Complexity |
|------|--------------|-----------------|------------|
| Move Logistic Regression Models | 5 | 2-3 | Low |
| XGBoostRegressor Wrapper | 18 | 20-24 | High (native interop) |
| LightGBMRegressor Wrapper | 18 | 20-24 | High (native interop) |
| CatBoostRegressor Wrapper | 18 | 20-24 | High (native interop) |
| Unit Tests for Gradient Boosting | 10 | 10-12 | Medium |
| GammaRegression Implementation | 13 | 14-16 | Medium (IRLS) |
| TweedieRegression Implementation | 13 | 14-16 | Medium (IRLS) |
| Unit Tests for GLMs | 8 | 8-10 | Medium |
| OrdinalRegression Implementation | 13 | 14-16 | Medium (constrained opt) |
| Unit Tests for Ordinal | 8 | 8-10 | Medium |
| **Total** | **124** | **130-155** | **High** |

### Implementation Order (Recommended)

1. **Week 1:** Refactor classification models (5 points, 2-3 hours)
   - Move files
   - Update namespaces and references
   - Verify tests still pass

2. **Week 2-3:** Gamma and Tweedie Regression (34 points, 36-42 hours)
   - Implement IRLS algorithm (reusable for both)
   - Implement Gamma regression
   - Implement Tweedie regression
   - Write comprehensive tests

3. **Week 4-5:** Ordinal Regression (21 points, 22-26 hours)
   - Implement proportional odds model
   - Handle threshold constraints
   - Write tests

4. **Week 6-9:** Optimized Gradient Boosting (64 points, 70-84 hours)
   - Research native interop for each library
   - Implement XGBoost wrapper
   - Implement LightGBM wrapper
   - Implement CatBoost wrapper
   - Write integration tests
   - Performance benchmarking

**Total Estimated Time:** 9-10 weeks for a junior developer

### Alternative Approach (If Native Interop is Complex)

If native library integration proves too difficult:

1. **Use Existing .NET Bindings:**
   - XGBoost.NET (NuGet package)
   - LightGBM.NET (NuGet package)
   - CatBoost.NET (NuGet package)

2. **Create Thin Wrappers:**
   - Focus on API consistency with AiDotNet
   - Delegate heavy lifting to existing bindings

This approach reduces complexity from High to Medium and saves ~30-40 hours.

---

## Additional Resources

### Academic Papers
1. McCullagh & Nelder (1989): "Generalized Linear Models" [GLM Bible]
2. Dunn & Smyth (2005): "Series Evaluation of Tweedie Exponential Dispersion Models"
3. McCullagh (1980): "Regression Models for Ordinal Data"
4. Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"

### Online Resources
1. XGBoost documentation: https://xgboost.readthedocs.io/
2. LightGBM documentation: https://lightgbm.readthedocs.io/
3. CatBoost documentation: https://catboost.ai/docs/
4. Statsmodels GLM guide: https://www.statsmodels.org/stable/glm.html

### Code References
- scikit-learn GLMs: https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/linear_model
- statsmodels Tweedie: https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Tweedie.html
- mord (ordinal regression): https://github.com/fabianp/mord

---

## Summary

This guide has covered:

1. **Understanding**: Different types of regression models and when to use them
2. **Math**: GLMs, gradient boosting, and ordinal regression equations
3. **Implementation**: IRLS for GLMs, native interop for gradient boosting, constrained optimization for ordinal
4. **Testing**: Comprehensive test suites for each model type
5. **Performance**: Complexity analysis and optimization strategies

**Key Takeaways:**

- Gamma regression for positive continuous data, Tweedie for data with zeros
- Ordinal regression for ordered categories (not regular classification)
- XGBoost/LightGBM/CatBoost require native library integration
- IRLS is the standard algorithm for GLM training
- Always validate model assumptions (e.g., positive targets for Gamma)
- Use NumOps throughout for type-generic implementations
- Native interop requires careful memory management

Good luck with your implementation!
