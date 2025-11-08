# Issue #319: Junior Developer Implementation Guide
## Implement Foundational Traditional Machine Learning Models

---

## Table of Contents
1. [Understanding Traditional ML Models](#understanding-traditional-ml-models)
2. [What EXISTS in the Codebase](#what-exists-in-the-codebase)
3. [What's MISSING - What You Need to Build](#whats-missing---what-you-need-to-build)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Common Pitfalls](#common-pitfalls)
6. [Testing Strategy](#testing-strategy)

---

## Understanding Traditional ML Models

### For Beginners: What are Traditional Machine Learning Models?

Think of machine learning models as different types of tools in a toolbox. Just like you wouldn't use a hammer for every job, you wouldn't use the same model for every data problem.

**Classification vs Regression:**
- **Classification**: Sorting things into categories (like "spam" or "not spam", "cat" or "dog")
- **Regression**: Predicting a number (like house prices, temperature, stock prices)
- **Clustering**: Finding natural groups in data without labels (like customer segments)
- **Dimensionality Reduction**: Simplifying complex data while keeping important information

**Real-World Analogy:**
Imagine you're a doctor:
- **Logistic Regression** is like using a checklist of symptoms to diagnose if someone has a disease (yes/no)
- **Ridge Regression** is like predicting how tall a child will be based on parents' heights, nutrition, etc.
- **Decision Tree** is like following a flowchart of yes/no questions
- **Random Forest** is like asking multiple doctors and voting on the diagnosis
- **SVM** is like drawing the best possible line to separate healthy from sick patients
- **K-Nearest Neighbors** is like diagnosing based on the 5 most similar previous patients
- **Naive Bayes** is like calculating probabilities based on symptom frequencies
- **K-Means** is like grouping patients by similar characteristics without knowing their diagnoses
- **PCA** is like summarizing a 100-page medical report into 2 key insights

---

## What EXISTS in the Codebase

### Existing Infrastructure You Can Leverage:

1. **Interfaces** (already exist):
   - `IFeatureSelector<T, TInput>` - for feature selection
   - `IOutlierRemoval<T, TInput, TOutput>` - for outlier handling
   - `IDataPreprocessor<T, TInput, TOutput>` - for data preprocessing
   - `INormalizer<T, TInput, TOutput>` - for normalization
   - `IRegression<T>` - for regression models
   - `INumericOperations<T>` - for type-generic math operations

2. **Helper Classes** (src/Helpers/):
   - `MathHelper` - provides numeric operations for generic types
   - `StatisticsHelper<T>` - calculate mean, variance, standard deviation
   - `MatrixHelper<T>` - matrix operations
   - `VectorHelper<T>` - vector operations
   - `InputHelper<T, TInput>` - extract batch size and input dimensions
   - `ModelHelper` - model-related utilities

3. **Existing Linear Algebra**:
   - `Matrix<T>` - 2D matrix operations
   - `Vector<T>` - 1D vector operations
   - `Tensor<T>` - multi-dimensional arrays

4. **Existing Regression Models** (src/Regression/):
   - `LogisticRegression<T>` - ALREADY EXISTS for classification
   - `BayesianRegression<T>` - probabilistic regression
   - `KNearestNeighborsRegression<T>` - KNN for regression
   - `DecisionTreeRegression<T>` - tree-based regression
   - `GradientBoostingRegression<T>` - ensemble regression
   - Many others!

5. **Existing Pattern**:
   - Classes directly implement interfaces (no base classes in this case)
   - Use `INumericOperations<T>` for all math operations
   - XML documentation with `<b>For Beginners:</b>` sections

---

## What's MISSING - What You Need to Build

### CRITICAL DISCOVERY: Most Models Already Exist!

After analyzing the codebase, here's what's actually missing:

### Phase 1: Linear Models
- **AC 1.1: LogisticRegression** - ✅ **ALREADY EXISTS** in `src/Regression/LogisticRegression.cs`
- **AC 1.2: RidgeRegression** - ❌ **MISSING** - needs implementation
- **AC 1.3: Unit Tests** - ❌ **MISSING** - need comprehensive tests

### Phase 2: Tree-Based Models
- **AC 2.1: DecisionTreeClassifier** - ⚠️ **EXISTS as DecisionTreeRegression** - may need classifier variant
- **AC 2.2: RandomForestClassifier** - ❌ **MISSING** - needs implementation
- **AC 2.3: Unit Tests** - ❌ **MISSING** - need comprehensive tests

### Phase 3: Support Vector Machines
- **AC 3.1: SVC (Support Vector Classifier)** - ❌ **MISSING** - needs implementation
- **AC 3.2: Unit Tests** - ❌ **MISSING** - need comprehensive tests

### Phase 4: KNN and Naive Bayes
- **AC 4.1: KNeighborsClassifier** - ⚠️ **EXISTS as KNearestNeighborsRegression** - may need classifier variant
- **AC 4.2: GaussianNaiveBayes** - ❌ **MISSING** - needs implementation
- **AC 4.3: Unit Tests** - ❌ **MISSING** - need comprehensive tests

### Phase 5: Clustering
- **AC 5.1: KMeans** - ❌ **MISSING** - needs implementation
- **AC 5.2: Unit Tests** - ❌ **MISSING** - need comprehensive tests

### Phase 6: Dimensionality Reduction
- **AC 6.1: PCA (Principal Component Analysis)** - ❌ **MISSING** - needs implementation
- **AC 6.2: Unit Tests** - ❌ **MISSING** - need comprehensive tests

### NEW Interfaces Needed:
- `IClassifier<T>` - for classification models
- `IRegressor<T>` - for regression models (may already exist as IRegression)
- `IClusterer<T>` - for clustering algorithms
- `IDimensionalityReducer<T>` - for dimensionality reduction

---

## Step-by-Step Implementation

### STEP 0: Create Missing Interfaces

Before implementing models, create the interfaces they'll implement.

#### 0.1: Create IClassifier<T> Interface

```csharp
// File: src/Interfaces/IClassifier.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for classification models that predict discrete class labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A classifier is a model that sorts data into categories.
///
/// Real-world examples:
/// - Email spam detection (spam vs not spam)
/// - Medical diagnosis (disease vs healthy)
/// - Image recognition (cat, dog, bird, etc.)
/// - Sentiment analysis (positive, negative, neutral)
///
/// Unlike regression (which predicts numbers), classification predicts categories.
/// </remarks>
public interface IClassifier<T>
{
    /// <summary>
    /// Trains the classifier on the provided data.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The class labels for each sample (typically 0, 1, 2, etc. for different classes).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method teaches the classifier to recognize patterns.
    ///
    /// Think of it like teaching a child to recognize animals:
    /// - X contains descriptions (has fur, number of legs, size, etc.)
    /// - y contains the correct answers (cat, dog, bird)
    /// - The classifier learns the patterns that distinguish each category
    /// </remarks>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    /// <param name="X">The feature matrix to predict labels for.</param>
    /// <returns>A vector of predicted class labels.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method makes predictions on new, unseen data.
    ///
    /// After training, you can give the classifier new data and it will
    /// predict which category each sample belongs to.
    /// </remarks>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Predicts class probabilities for each sample.
    /// </summary>
    /// <param name="X">The feature matrix to predict probabilities for.</param>
    /// <returns>A matrix where each row contains probabilities for all classes.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of just saying "this is a cat", this tells you
    /// "I'm 80% confident it's a cat, 15% confident it's a dog, 5% confident it's a bird".
    ///
    /// This is useful when you need to know how certain the prediction is.
    /// </remarks>
    Matrix<T> PredictProbabilities(Matrix<T> X);
}
```

#### 0.2: Create IRegressor<T> Interface

```csharp
// File: src/Interfaces/IRegressor.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for regression models that predict continuous numeric values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A regressor is a model that predicts numbers, not categories.
///
/// Real-world examples:
/// - House price prediction (predicting dollars)
/// - Temperature forecasting (predicting degrees)
/// - Stock price prediction (predicting price)
/// - Age estimation (predicting years)
///
/// Unlike classification (which predicts categories), regression predicts continuous values.
/// </remarks>
public interface IRegressor<T>
{
    /// <summary>
    /// Trains the regressor on the provided data.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target values for each sample.</param>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts numeric values for new data.
    /// </summary>
    /// <param name="X">The feature matrix to predict values for.</param>
    /// <returns>A vector of predicted values.</returns>
    Vector<T> Predict(Matrix<T> X);
}
```

#### 0.3: Create IClusterer<T> Interface

```csharp
// File: src/Interfaces/IClusterer.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for clustering algorithms that group similar data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A clusterer finds natural groups in your data without being told what to look for.
///
/// Real-world examples:
/// - Customer segmentation (finding types of customers)
/// - Image compression (grouping similar colors)
/// - Document organization (grouping similar documents)
/// - Anomaly detection (finding unusual patterns)
///
/// Unlike classification (where you know the categories), clustering discovers
/// the categories by finding patterns in the data.
/// </remarks>
public interface IClusterer<T>
{
    /// <summary>
    /// Learns cluster centers from the data.
    /// </summary>
    /// <param name="X">The feature matrix to cluster.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds the natural groupings in your data.
    ///
    /// It's like organizing a messy room:
    /// - You don't have pre-defined categories
    /// - You look at the items and group similar things together
    /// - The algorithm finds the best way to organize the data
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Assigns cluster labels to new data points.
    /// </summary>
    /// <param name="X">The feature matrix to assign clusters to.</param>
    /// <returns>A vector of cluster labels (0, 1, 2, etc.).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> After learning the groups, this assigns new data to the closest group.
    /// </remarks>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Gets the cluster centers (centroids).
    /// </summary>
    /// <returns>A matrix where each row is a cluster center.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This returns the "typical example" for each group.
    ///
    /// For example, in customer segmentation:
    /// - Cluster 1 center might represent "young, tech-savvy, high income"
    /// - Cluster 2 center might represent "middle-aged, budget-conscious, families"
    /// </remarks>
    Matrix<T> ClusterCenters { get; }
}
```

#### 0.4: Create IDimensionalityReducer<T> Interface

```csharp
// File: src/Interfaces/IDimensionalityReducer.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for dimensionality reduction algorithms that reduce the number of features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Dimensionality reduction simplifies complex data while keeping the important information.
///
/// Imagine you have 100 measurements about houses (size, bedrooms, bathrooms, neighborhood crime rate,
/// distance to schools, etc.). Many of these are probably related (e.g., bigger houses tend to have
/// more bedrooms). Dimensionality reduction finds 2-3 "summary features" that capture most of the
/// important information.
///
/// Why reduce dimensions?
/// - Makes data easier to visualize (2D or 3D plots)
/// - Speeds up machine learning algorithms
/// - Removes redundant information
/// - Reduces memory usage
///
/// Common technique: PCA (Principal Component Analysis)
/// </remarks>
public interface IDimensionalityReducer<T>
{
    /// <summary>
    /// Learns the dimensionality reduction transformation from the data.
    /// </summary>
    /// <param name="X">The feature matrix with many features.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes your data to find the best way to simplify it.
    ///
    /// It's like creating a summary:
    /// - You read a long document (high-dimensional data)
    /// - You identify the key points that matter most
    /// - You learn how to create a short summary that captures the essence
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Transforms high-dimensional data to lower dimensions.
    /// </summary>
    /// <param name="X">The high-dimensional feature matrix.</param>
    /// <returns>The transformed low-dimensional matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts complex data into simplified form.
    ///
    /// Example:
    /// - Input: 100 features per house
    /// - Output: 3 "principal components" that capture most of the variation
    /// - These 3 components might represent: "overall quality", "location desirability", "size"
    /// </remarks>
    Matrix<T> Transform(Matrix<T> X);

    /// <summary>
    /// Transforms low-dimensional data back to the original space (approximate reconstruction).
    /// </summary>
    /// <param name="X">The low-dimensional transformed data.</param>
    /// <returns>The reconstructed high-dimensional matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method reverses the simplification to get back approximate original data.
    ///
    /// Like expanding a summary back into a full document:
    /// - You won't get exactly the original (some detail is lost)
    /// - But you get a good approximation
    /// - Useful for understanding what information was preserved vs. lost
    /// </remarks>
    Matrix<T> InverseTransform(Matrix<T> X);

    /// <summary>
    /// Gets the number of components (reduced dimensions) to keep.
    /// </summary>
    int Components { get; }
}
```

---

### STEP 1: Implement Ridge Regression (AC 1.2)

Ridge Regression is Linear Regression with L2 regularization to prevent overfitting.

#### Mathematical Background:
```
Regular Linear Regression: minimize ||y - Xw||^2
Ridge Regression: minimize ||y - Xw||^2 + alpha * ||w||^2

Where:
- y = target values
- X = feature matrix
- w = weights/coefficients
- alpha = regularization strength (prevents overfitting)
```

**Solution**: w = (X^T * X + alpha * I)^(-1) * X^T * y

#### Full Implementation:

```csharp
// File: src/Models/Linear/RidgeRegression.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Linear;

/// <summary>
/// Implements Ridge Regression (L2 regularized linear regression).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Ridge Regression is like regular linear regression but with a penalty for complexity.
///
/// Imagine you're fitting a line to data:
/// - Regular regression finds the best line, but might get too complicated and overfit
/// - Ridge regression finds the best line while keeping coefficients (weights) small
/// - This prevents the model from relying too heavily on any single feature
///
/// The "alpha" parameter controls the penalty:
/// - alpha = 0: Same as regular linear regression
/// - alpha = small (0.1-1.0): Light regularization
/// - alpha = large (10+): Strong regularization, simpler model
///
/// Default alpha=1.0 is based on sklearn's default and works well for most problems.
///
/// Reference: scikit-learn Ridge Regression
/// https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
/// </remarks>
public class RidgeRegression<T> : IRegressor<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _alpha;
    private Vector<T>? _coefficients;
    private T _intercept;

    /// <summary>
    /// Initializes a new instance of the RidgeRegression class.
    /// </summary>
    /// <param name="alpha">Regularization strength. Must be positive. Default is 1.0.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a Ridge Regression model.
    ///
    /// The alpha parameter (default 1.0) comes from sklearn's default:
    /// - Chosen through extensive empirical testing
    /// - Balances fitting the data vs. keeping model simple
    /// - Works well across many real-world datasets
    /// - Increase if your model overfits (too complex)
    /// - Decrease if your model underfits (too simple)
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when alpha is not positive.</exception>
    public RidgeRegression(double alpha = 1.0)
    {
        if (alpha <= 0)
        {
            throw new ArgumentException("Alpha must be positive", nameof(alpha));
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _alpha = _numOps.FromDouble(alpha);
        _intercept = _numOps.Zero;
    }

    /// <summary>
    /// Gets the learned coefficients (weights) for each feature.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After training, these numbers tell you how much each feature matters.
    ///
    /// For house price prediction:
    /// - Coefficient for square footage might be +200 (bigger = more expensive)
    /// - Coefficient for age might be -5000 (older = less expensive)
    /// - Coefficient for distance from school might be -1000 (farther = less desirable)
    /// </remarks>
    public Vector<T> Coefficients => _coefficients ?? new Vector<T>(0);

    /// <summary>
    /// Gets the intercept (bias) term.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the baseline prediction when all features are zero.
    ///
    /// For house prices, it might be the average house price in the area.
    /// </remarks>
    public T Intercept => _intercept;

    /// <summary>
    /// Trains the Ridge Regression model on the provided data.
    /// </summary>
    /// <param name="X">Feature matrix (rows = samples, columns = features).</param>
    /// <param name="y">Target values.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method learns the relationship between features and target.
    ///
    /// Algorithm:
    /// 1. Add intercept column of 1s to X
    /// 2. Compute X^T * X + alpha * I (regularized Gram matrix)
    /// 3. Compute X^T * y
    /// 4. Solve (X^T * X + alpha * I) * w = X^T * y
    /// 5. Extract coefficients and intercept
    ///
    /// The regularization (alpha * I) prevents overfitting by penalizing large coefficients.
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X and y must match");
        }

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Add intercept column (column of 1s)
        var XWithIntercept = new Matrix<T>(nSamples, nFeatures + 1);
        for (int i = 0; i < nSamples; i++)
        {
            XWithIntercept[i, 0] = _numOps.One; // Intercept column
            for (int j = 0; j < nFeatures; j++)
            {
                XWithIntercept[i, j + 1] = X[i, j];
            }
        }

        // Compute X^T * X
        var XTX = MatrixHelper<T>.Transpose(XWithIntercept).Multiply(XWithIntercept);

        // Add regularization: X^T * X + alpha * I
        for (int i = 0; i < nFeatures + 1; i++)
        {
            // Don't regularize the intercept (first element)
            if (i > 0)
            {
                XTX[i, i] = _numOps.Add(XTX[i, i], _alpha);
            }
        }

        // Compute X^T * y
        var XTy = MatrixHelper<T>.Transpose(XWithIntercept).MultiplyVector(y);

        // Solve (X^T * X + alpha * I) * w = X^T * y
        var coefficientsWithIntercept = MatrixSolutionHelper<T>.Solve(XTX, XTy);

        // Extract intercept and coefficients
        _intercept = coefficientsWithIntercept[0];
        _coefficients = new Vector<T>(nFeatures);
        for (int i = 0; i < nFeatures; i++)
        {
            _coefficients[i] = coefficientsWithIntercept[i + 1];
        }
    }

    /// <summary>
    /// Predicts target values for new data.
    /// </summary>
    /// <param name="X">Feature matrix to predict for.</param>
    /// <returns>Predicted values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Uses the learned coefficients to make predictions.
    ///
    /// Formula: y = X * coefficients + intercept
    ///
    /// For each sample, multiply features by their coefficients, add them up, then add intercept.
    /// </remarks>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (_coefficients == null)
        {
            throw new InvalidOperationException("Model must be fitted before making predictions");
        }

        if (X.Columns != _coefficients.Length)
        {
            throw new ArgumentException($"Expected {_coefficients.Length} features, got {X.Columns}");
        }

        // y = X * coefficients + intercept
        var predictions = X.MultiplyVector(_coefficients);

        for (int i = 0; i < predictions.Length; i++)
        {
            predictions[i] = _numOps.Add(predictions[i], _intercept);
        }

        return predictions;
    }
}
```

---

### STEP 2: Implement K-Means Clustering (AC 5.1)

K-Means is one of the most popular clustering algorithms.

#### Mathematical Background:
```
Algorithm:
1. Randomly initialize K cluster centers
2. Repeat until convergence:
   a. Assign each point to nearest center
   b. Update centers to mean of assigned points
3. Return final cluster assignments

Distance metric: Euclidean distance = sqrt(sum((x - y)^2))
```

#### Full Implementation:

```csharp
// File: src/Models/Clustering/KMeans.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Clustering;

/// <summary>
/// Implements K-Means clustering algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> K-Means finds K groups (clusters) in your data.
///
/// Real-world analogy:
/// Imagine you have customers and want to group them by behavior:
/// - You decide to find 3 groups (K=3)
/// - The algorithm finds the 3 "typical customers" (cluster centers)
/// - Then assigns each customer to the most similar group
///
/// How it works:
/// 1. Start with K random cluster centers
/// 2. Assign each point to the nearest center
/// 3. Move each center to the average of its assigned points
/// 4. Repeat steps 2-3 until centers stop moving
///
/// Parameters:
/// - nClusters: How many groups to find (default 8 from sklearn)
/// - maxIterations: Maximum times to repeat (default 300 from sklearn)
/// - tolerance: When to stop (default 1e-4 from sklearn)
/// - randomState: Seed for reproducibility (default null = random)
///
/// Defaults from scikit-learn KMeans:
/// https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
/// </remarks>
public class KMeans<T> : IClusterer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _nClusters;
    private readonly int _maxIterations;
    private readonly T _tolerance;
    private readonly int? _randomState;
    private Matrix<T>? _clusterCenters;
    private Random _random;

    /// <summary>
    /// Initializes a new instance of the KMeans class.
    /// </summary>
    /// <param name="nClusters">Number of clusters to find. Default is 8.</param>
    /// <param name="maxIterations">Maximum number of iterations. Default is 300.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-4.</param>
    /// <param name="randomState">Random seed for reproducibility. Default is null (random).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a K-Means clusterer.
    ///
    /// Why these defaults?
    /// - nClusters=8: sklearn default, good middle ground for exploration
    /// - maxIterations=300: sklearn default, enough for most datasets to converge
    /// - tolerance=1e-4: sklearn default, stops when centers move less than this
    /// - randomState=null: Different results each run, set a number for reproducibility
    ///
    /// Choosing nClusters:
    /// - Too few: Misses important groups
    /// - Too many: Creates artificial subdivisions
    /// - Use "elbow method" or domain knowledge to choose
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public KMeans(
        int nClusters = 8,
        int maxIterations = 300,
        double tolerance = 1e-4,
        int? randomState = null)
    {
        if (nClusters <= 0)
        {
            throw new ArgumentException("nClusters must be positive", nameof(nClusters));
        }
        if (maxIterations <= 0)
        {
            throw new ArgumentException("maxIterations must be positive", nameof(maxIterations));
        }
        if (tolerance <= 0)
        {
            throw new ArgumentException("tolerance must be positive", nameof(tolerance));
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _nClusters = nClusters;
        _maxIterations = maxIterations;
        _tolerance = _numOps.FromDouble(tolerance);
        _randomState = randomState;
        _random = randomState.HasValue ? new Random(randomState.Value) : new Random();
    }

    /// <summary>
    /// Gets the cluster centers (centroids) after fitting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each row is the "typical example" for that cluster.
    ///
    /// For customer segmentation:
    /// - Cluster 0 center: [age=25, income=40k, purchases=5] = "Young budget shoppers"
    /// - Cluster 1 center: [age=45, income=80k, purchases=20] = "Middle-age affluent buyers"
    /// - Cluster 2 center: [age=65, income=60k, purchases=2] = "Retired occasional shoppers"
    /// </remarks>
    public Matrix<T> ClusterCenters => _clusterCenters ?? new Matrix<T>(0, 0);

    /// <summary>
    /// Learns cluster centers from the data.
    /// </summary>
    /// <param name="X">Feature matrix (rows = samples, columns = features).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Finds the K groups in your data.
    ///
    /// Steps:
    /// 1. Initialize K centers randomly from the data points
    /// 2. Repeat:
    ///    a. Assign each point to nearest center (Euclidean distance)
    ///    b. Update each center to mean of assigned points
    ///    c. Check if centers moved significantly
    /// 3. Stop when centers stabilize or max iterations reached
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        if (X.Rows < _nClusters)
        {
            throw new ArgumentException($"Number of samples ({X.Rows}) must be >= nClusters ({_nClusters})");
        }

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Step 1: Initialize centers by randomly selecting K points
        _clusterCenters = InitializeCenters(X);

        // Track cluster assignments
        var labels = new int[nSamples];

        // Step 2: Iterate until convergence
        for (int iteration = 0; iteration < _maxIterations; iteration++)
        {
            // Step 2a: Assign each point to nearest center
            var oldLabels = (int[])labels.Clone();
            for (int i = 0; i < nSamples; i++)
            {
                labels[i] = FindNearestCenter(X.GetRow(i), _clusterCenters);
            }

            // Step 2b: Update centers to mean of assigned points
            var newCenters = UpdateCenters(X, labels, nFeatures);

            // Step 2c: Check convergence
            var maxShift = CalculateMaxCenterShift(_clusterCenters, newCenters);
            _clusterCenters = newCenters;

            if (_numOps.LessThan(maxShift, _tolerance))
            {
                // Converged!
                break;
            }
        }
    }

    /// <summary>
    /// Assigns cluster labels to new data points.
    /// </summary>
    /// <param name="X">Feature matrix to assign clusters to.</param>
    /// <returns>Vector of cluster labels (0 to K-1).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> For each point, finds the nearest cluster center.
    ///
    /// Returns 0, 1, 2, ..., K-1 indicating which cluster each point belongs to.
    /// </remarks>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (_clusterCenters == null)
        {
            throw new InvalidOperationException("Model must be fitted before making predictions");
        }

        if (X.Columns != _clusterCenters.Columns)
        {
            throw new ArgumentException($"Expected {_clusterCenters.Columns} features, got {X.Columns}");
        }

        var labels = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            var nearestCluster = FindNearestCenter(X.GetRow(i), _clusterCenters);
            labels[i] = _numOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    /// <summary>
    /// Initializes cluster centers by randomly selecting K samples.
    /// </summary>
    private Matrix<T> InitializeCenters(Matrix<T> X)
    {
        int nFeatures = X.Columns;
        var centers = new Matrix<T>(_nClusters, nFeatures);

        // Random initialization: pick K random samples as initial centers
        var selectedIndices = new HashSet<int>();
        for (int k = 0; k < _nClusters; k++)
        {
            int randomIndex;
            do
            {
                randomIndex = _random.Next(X.Rows);
            } while (selectedIndices.Contains(randomIndex));

            selectedIndices.Add(randomIndex);

            for (int j = 0; j < nFeatures; j++)
            {
                centers[k, j] = X[randomIndex, j];
            }
        }

        return centers;
    }

    /// <summary>
    /// Finds the index of the nearest cluster center for a given point.
    /// </summary>
    private int FindNearestCenter(Vector<T> point, Matrix<T> centers)
    {
        int nearestCluster = 0;
        T minDistance = _numOps.FromDouble(double.MaxValue);

        for (int k = 0; k < centers.Rows; k++)
        {
            var center = centers.GetRow(k);
            var distance = EuclideanDistance(point, center);

            if (_numOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                nearestCluster = k;
            }
        }

        return nearestCluster;
    }

    /// <summary>
    /// Calculates Euclidean distance between two points.
    /// </summary>
    private T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = _numOps.Subtract(a[i], b[i]);
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(diff, diff));
        }
        return _numOps.Sqrt(sumSquares);
    }

    /// <summary>
    /// Updates cluster centers to the mean of their assigned points.
    /// </summary>
    private Matrix<T> UpdateCenters(Matrix<T> X, int[] labels, int nFeatures)
    {
        var newCenters = new Matrix<T>(_nClusters, nFeatures);
        var counts = new int[_nClusters];

        // Sum up points in each cluster
        for (int i = 0; i < X.Rows; i++)
        {
            int cluster = labels[i];
            counts[cluster]++;

            for (int j = 0; j < nFeatures; j++)
            {
                newCenters[cluster, j] = _numOps.Add(newCenters[cluster, j], X[i, j]);
            }
        }

        // Divide by count to get mean
        for (int k = 0; k < _nClusters; k++)
        {
            if (counts[k] > 0)
            {
                var count = _numOps.FromDouble(counts[k]);
                for (int j = 0; j < nFeatures; j++)
                {
                    newCenters[k, j] = _numOps.Divide(newCenters[k, j], count);
                }
            }
            // If a cluster has no points, keep the old center
        }

        return newCenters;
    }

    /// <summary>
    /// Calculates the maximum shift of cluster centers.
    /// </summary>
    private T CalculateMaxCenterShift(Matrix<T> oldCenters, Matrix<T> newCenters)
    {
        T maxShift = _numOps.Zero;

        for (int k = 0; k < _nClusters; k++)
        {
            var shift = EuclideanDistance(oldCenters.GetRow(k), newCenters.GetRow(k));
            if (_numOps.GreaterThan(shift, maxShift))
            {
                maxShift = shift;
            }
        }

        return maxShift;
    }
}
```

---

### STEP 3: Implement PCA (AC 6.1)

Principal Component Analysis reduces dimensions by finding directions of maximum variance.

#### Mathematical Background:
```
Algorithm:
1. Center the data (subtract mean)
2. Compute covariance matrix: C = X^T * X / (n-1)
3. Compute eigenvectors and eigenvalues of C
4. Sort eigenvectors by eigenvalues (descending)
5. Keep top K eigenvectors as principal components
6. Transform: X_new = X * components
```

#### Full Implementation:

```csharp
// File: src/Models/DimensionalityReduction/PCA.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.DimensionalityReduction;

/// <summary>
/// Implements Principal Component Analysis (PCA) for dimensionality reduction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> PCA simplifies complex data while keeping the most important information.
///
/// Real-world analogy:
/// You have 100 measurements about houses. PCA finds 2-3 "summary features" that capture
/// most of the variation:
/// - Component 1 might represent "overall quality" (combining size, age, condition)
/// - Component 2 might represent "location" (combining neighborhood, schools, crime)
/// - Component 3 might represent "luxury features" (pool, garage, finishes)
///
/// How it works:
/// 1. Centers the data (subtracts mean from each feature)
/// 2. Finds directions of maximum variance in the data
/// 3. Projects data onto these directions (principal components)
/// 4. Keeps only the top K components
///
/// Parameters:
/// - nComponents: How many dimensions to reduce to (default 2 for visualization)
///
/// Reference: scikit-learn PCA
/// https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
/// </remarks>
public class PCA<T> : IDimensionalityReducer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _nComponents;
    private Matrix<T>? _components;
    private Vector<T>? _mean;
    private Vector<T>? _explainedVariance;

    /// <summary>
    /// Initializes a new instance of the PCA class.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. Default is 2.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a PCA transformer.
    ///
    /// Why default 2 components?
    /// - Good for visualization (2D scatter plots)
    /// - Often captures most variation in real data
    /// - Based on common practice in exploratory data analysis
    ///
    /// Choosing nComponents:
    /// - 2-3: For visualization
    /// - Based on explained variance: Keep components explaining 95% of variance
    /// - Based on domain knowledge: How many meaningful factors exist?
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when nComponents is not positive.</exception>
    public PCA(int nComponents = 2)
    {
        if (nComponents <= 0)
        {
            throw new ArgumentException("nComponents must be positive", nameof(nComponents));
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _nComponents = nComponents;
    }

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int Components => _nComponents;

    /// <summary>
    /// Gets the principal components (eigenvectors).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each row is a principal component (a direction in the original space).
    ///
    /// These define the new coordinate system where data is most spread out.
    /// </remarks>
    public Matrix<T> ComponentVectors => _components ?? new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the amount of variance explained by each component.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tells you how much information each component captures.
    ///
    /// Example:
    /// - Component 1: explains 60% of variance
    /// - Component 2: explains 25% of variance
    /// - Component 3: explains 10% of variance
    /// - Total: 95% of original information preserved
    /// </remarks>
    public Vector<T> ExplainedVariance => _explainedVariance ?? new Vector<T>(0);

    /// <summary>
    /// Learns the principal components from the data.
    /// </summary>
    /// <param name="X">Feature matrix (rows = samples, columns = features).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Analyzes the data to find the best way to simplify it.
    ///
    /// Steps:
    /// 1. Calculate mean of each feature
    /// 2. Center the data (subtract mean)
    /// 3. Compute covariance matrix
    /// 4. Find eigenvectors (directions of variance)
    /// 5. Sort by eigenvalues (amount of variance)
    /// 6. Keep top nComponents eigenvectors
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        if (X.Rows < _nComponents)
        {
            throw new ArgumentException($"Number of samples ({X.Rows}) must be >= nComponents ({_nComponents})");
        }

        if (X.Columns < _nComponents)
        {
            throw new ArgumentException($"Number of features ({X.Columns}) must be >= nComponents ({_nComponents})");
        }

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Step 1: Calculate mean of each feature
        _mean = new Vector<T>(nFeatures);
        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            _mean[j] = StatisticsHelper<T>.CalculateMean(column);
        }

        // Step 2: Center the data
        var XCentered = new Matrix<T>(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                XCentered[i, j] = _numOps.Subtract(X[i, j], _mean[j]);
            }
        }

        // Step 3: Compute covariance matrix: C = X^T * X / (n-1)
        var XTX = MatrixHelper<T>.Transpose(XCentered).Multiply(XCentered);
        var nMinus1 = _numOps.FromDouble(nSamples - 1);

        var covarianceMatrix = new Matrix<T>(nFeatures, nFeatures);
        for (int i = 0; i < nFeatures; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                covarianceMatrix[i, j] = _numOps.Divide(XTX[i, j], nMinus1);
            }
        }

        // Step 4-6: Compute eigenvectors and eigenvalues, sort, keep top K
        var (eigenvectors, eigenvalues) = ComputeEigenvectors(covarianceMatrix);

        // Store top nComponents eigenvectors
        _components = new Matrix<T>(_nComponents, nFeatures);
        _explainedVariance = new Vector<T>(_nComponents);

        for (int i = 0; i < _nComponents; i++)
        {
            _explainedVariance[i] = eigenvalues[i];
            for (int j = 0; j < nFeatures; j++)
            {
                _components[i, j] = eigenvectors[i][j];
            }
        }
    }

    /// <summary>
    /// Transforms high-dimensional data to lower dimensions.
    /// </summary>
    /// <param name="X">High-dimensional feature matrix.</param>
    /// <returns>Low-dimensional transformed matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Projects your data onto the principal components.
    ///
    /// Formula: X_new = (X - mean) * components^T
    ///
    /// This converts 100 features into 2 principal components, for example.
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> X)
    {
        if (_components == null || _mean == null)
        {
            throw new InvalidOperationException("Model must be fitted before transforming");
        }

        if (X.Columns != _mean.Length)
        {
            throw new ArgumentException($"Expected {_mean.Length} features, got {X.Columns}");
        }

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Center the data
        var XCentered = new Matrix<T>(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                XCentered[i, j] = _numOps.Subtract(X[i, j], _mean[j]);
            }
        }

        // Project onto components: X_new = X_centered * components^T
        var XTransformed = XCentered.Multiply(MatrixHelper<T>.Transpose(_components));

        return XTransformed;
    }

    /// <summary>
    /// Transforms low-dimensional data back to original space (approximate reconstruction).
    /// </summary>
    /// <param name="X">Low-dimensional transformed data.</param>
    /// <returns>Reconstructed high-dimensional matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Converts simplified data back to original dimensions.
    ///
    /// Formula: X_reconstructed = X_transformed * components + mean
    ///
    /// This is approximate because we lost some information during reduction.
    /// </remarks>
    public Matrix<T> InverseTransform(Matrix<T> X)
    {
        if (_components == null || _mean == null)
        {
            throw new InvalidOperationException("Model must be fitted before inverse transforming");
        }

        if (X.Columns != _nComponents)
        {
            throw new ArgumentException($"Expected {_nComponents} components, got {X.Columns}");
        }

        // X_reconstructed = X * components
        var XReconstructed = X.Multiply(_components);

        // Add back the mean
        for (int i = 0; i < XReconstructed.Rows; i++)
        {
            for (int j = 0; j < XReconstructed.Columns; j++)
            {
                XReconstructed[i, j] = _numOps.Add(XReconstructed[i, j], _mean[j]);
            }
        }

        return XReconstructed;
    }

    /// <summary>
    /// Computes eigenvectors and eigenvalues of a symmetric matrix.
    /// Uses power iteration method for simplicity.
    /// </summary>
    private (Vector<T>[], Vector<T>) ComputeEigenvectors(Matrix<T> matrix)
    {
        // NOTE: In production, use a proper eigendecomposition library
        // This is a simplified implementation using power iteration

        int n = matrix.Rows;
        var eigenvectors = new Vector<T>[_nComponents];
        var eigenvalues = new Vector<T>(_nComponents);

        var workingMatrix = matrix.Clone();

        for (int k = 0; k < _nComponents; k++)
        {
            // Power iteration to find dominant eigenvector
            var v = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                v[i] = _numOps.FromDouble(1.0 / Math.Sqrt(n)); // Initialize
            }

            for (int iter = 0; iter < 100; iter++)
            {
                // v = matrix * v
                var vNew = workingMatrix.MultiplyVector(v);

                // Normalize
                var norm = _numOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    norm = _numOps.Add(norm, _numOps.Multiply(vNew[i], vNew[i]));
                }
                norm = _numOps.Sqrt(norm);

                for (int i = 0; i < n; i++)
                {
                    vNew[i] = _numOps.Divide(vNew[i], norm);
                }

                v = vNew;
            }

            eigenvectors[k] = v;

            // Compute eigenvalue: lambda = v^T * matrix * v
            var Av = workingMatrix.MultiplyVector(v);
            var lambda = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                lambda = _numOps.Add(lambda, _numOps.Multiply(v[i], Av[i]));
            }
            eigenvalues[k] = lambda;

            // Deflate matrix: remove this eigenvalue's contribution
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var deflation = _numOps.Multiply(
                        _numOps.Multiply(lambda, v[i]),
                        v[j]
                    );
                    workingMatrix[i, j] = _numOps.Subtract(workingMatrix[i, j], deflation);
                }
            }
        }

        return (eigenvectors, eigenvalues);
    }
}
```

---

## Common Pitfalls

### 1. NEVER Use `default(T)` or `default!`
```csharp
// ❌ WRONG
T value = default(T);
Matrix<T> matrix = default!;

// ✅ CORRECT
T value = _numOps.Zero;
Matrix<T> matrix = new Matrix<T>(0, 0);
```

### 2. NEVER Hardcode Numeric Types
```csharp
// ❌ WRONG
double threshold = 0.5;
if (value > threshold) { }

// ✅ CORRECT
T threshold = _numOps.FromDouble(0.5);
if (_numOps.GreaterThan(value, threshold)) { }
```

### 3. ALWAYS Use INumericOperations<T>
```csharp
// ❌ WRONG
var sum = a + b;
var product = a * b;

// ✅ CORRECT
var sum = _numOps.Add(a, b);
var product = _numOps.Multiply(a, b);
```

### 4. ALWAYS Validate Parameters
```csharp
// ✅ CORRECT
public RidgeRegression(double alpha = 1.0)
{
    if (alpha <= 0)
    {
        throw new ArgumentException("Alpha must be positive", nameof(alpha));
    }
    _alpha = _numOps.FromDouble(alpha);
}
```

### 5. ALWAYS Check if Model is Fitted
```csharp
// ✅ CORRECT
public Vector<T> Predict(Matrix<T> X)
{
    if (_coefficients == null)
    {
        throw new InvalidOperationException("Model must be fitted before making predictions");
    }
    // ... rest of prediction logic
}
```

### 6. ALWAYS Initialize Properties Properly
```csharp
// ❌ WRONG
private Vector<T>? _coefficients = default!;

// ✅ CORRECT
private Vector<T>? _coefficients;  // Nullable, will be set in Fit()
private T _intercept = _numOps.Zero;  // Non-nullable, initialize
```

### 7. NEVER Forget XML Documentation
```csharp
// ✅ CORRECT - Include beginner-friendly sections
/// <summary>
/// Brief description.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Detailed explanation with real-world analogies.
/// </remarks>
```

---

## Testing Strategy

### Unit Test Structure

```csharp
// File: tests/UnitTests/Models/RidgeRegressionTests.cs
using Xunit;
using AiDotNet.Models.Linear;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Models;

public class RidgeRegressionTests
{
    [Fact]
    public void Fit_SimpleLinearData_LearnsCorrectCoefficients()
    {
        // Arrange: Create simple linear relationship y = 2x + 1
        var X = new Matrix<double>(new double[,]
        {
            { 1 },
            { 2 },
            { 3 },
            { 4 },
            { 5 }
        });
        var y = new Vector<double>(new double[] { 3, 5, 7, 9, 11 });

        var model = new RidgeRegression<double>(alpha: 0.01); // Small alpha

        // Act
        model.Fit(X, y);

        // Assert
        Assert.NotNull(model.Coefficients);
        Assert.Single(model.Coefficients);

        // Coefficient should be close to 2.0
        Assert.InRange(model.Coefficients[0], 1.9, 2.1);

        // Intercept should be close to 1.0
        Assert.InRange(model.Intercept, 0.9, 1.1);
    }

    [Fact]
    public void Predict_AfterFitting_ReturnsCorrectPredictions()
    {
        // Arrange
        var X = new Matrix<double>(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = new Vector<double>(new double[] { 3, 5, 7 });

        var model = new RidgeRegression<double>();
        model.Fit(X, y);

        var XTest = new Matrix<double>(new double[,] { { 4 }, { 5 } });

        // Act
        var predictions = model.Predict(XTest);

        // Assert
        Assert.Equal(2, predictions.Length);

        // Predictions should be close to 9 and 11
        Assert.InRange(predictions[0], 8.5, 9.5);
        Assert.InRange(predictions[1], 10.5, 11.5);
    }

    [Fact]
    public void Constructor_NegativeAlpha_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new RidgeRegression<double>(alpha: -1.0));
    }

    [Fact]
    public void Predict_BeforeFitting_ThrowsInvalidOperationException()
    {
        // Arrange
        var model = new RidgeRegression<double>();
        var X = new Matrix<double>(new double[,] { { 1 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => model.Predict(X));
    }

    [Fact]
    public void Fit_WithRegularization_ProducesSmallerCoefficients()
    {
        // Arrange: Data with potential overfitting
        var X = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = new Vector<double>(new double[] { 3, 5, 7 });

        var modelNoReg = new RidgeRegression<double>(alpha: 0.001);
        var modelWithReg = new RidgeRegression<double>(alpha: 10.0);

        // Act
        modelNoReg.Fit(X, y);
        modelWithReg.Fit(X, y);

        // Assert: Regularized model should have smaller coefficient magnitudes
        double sumNoReg = 0, sumWithReg = 0;
        for (int i = 0; i < modelNoReg.Coefficients.Length; i++)
        {
            sumNoReg += Math.Abs(modelNoReg.Coefficients[i]);
            sumWithReg += Math.Abs(modelWithReg.Coefficients[i]);
        }

        Assert.True(sumWithReg < sumNoReg, "Regularized model should have smaller coefficients");
    }

    [Theory]
    [InlineData(typeof(double))]
    [InlineData(typeof(float))]
    public void Fit_WorksWithDifferentNumericTypes(Type numericType)
    {
        // Test that model works with both double and float
        if (numericType == typeof(double))
        {
            var model = new RidgeRegression<double>();
            var X = new Matrix<double>(new double[,] { { 1 }, { 2 } });
            var y = new Vector<double>(new double[] { 2, 4 });
            model.Fit(X, y);
            Assert.NotNull(model.Coefficients);
        }
        else if (numericType == typeof(float))
        {
            var model = new RidgeRegression<float>();
            var X = new Matrix<float>(new float[,] { { 1 }, { 2 } });
            var y = new Vector<float>(new float[] { 2, 4 });
            model.Fit(X, y);
            Assert.NotNull(model.Coefficients);
        }
    }
}
```

### Test Coverage Requirements

1. **Basic Functionality** (60% of tests):
   - Fit() learns correct coefficients
   - Predict() returns correct predictions
   - Properties are set correctly

2. **Edge Cases** (20% of tests):
   - Invalid parameters (negative values, zero, etc.)
   - Mismatched dimensions
   - Empty data
   - Single sample
   - High-dimensional data

3. **Error Handling** (10% of tests):
   - Predict before Fit throws exception
   - Invalid parameters throw ArgumentException
   - Dimension mismatches throw ArgumentException

4. **Type Compatibility** (10% of tests):
   - Works with double
   - Works with float
   - Uses INumericOperations correctly

### Minimum 80% Code Coverage

Run coverage with:
```bash
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover
```

Ensure:
- All public methods are tested
- All branches (if/else) are covered
- All exception paths are tested

---

## Summary

### What You Built:
1. ✅ 4 new interfaces for ML models
2. ✅ Ridge Regression (linear regression with L2 regularization)
3. ✅ K-Means Clustering (unsupervised learning)
4. ✅ PCA (dimensionality reduction)
5. ✅ Comprehensive unit tests for all implementations

### Key Learnings:
- Use `INumericOperations<T>` for type-generic math
- NEVER use `default(T)` or hardcode numeric types
- Validate all parameters in constructors
- Check if model is fitted before predictions
- Document WHY defaults were chosen
- Include beginner-friendly `<b>For Beginners:</b>` sections
- Test with multiple numeric types (double, float)
- Achieve minimum 80% code coverage

### Next Steps:
1. Implement remaining models (SVM, RandomForest, etc.)
2. Add more comprehensive integration tests
3. Test with real-world datasets
4. Optimize performance with parallel processing
5. Add model serialization/deserialization

### Resources:
- scikit-learn documentation for algorithm references
- AiDotNet existing implementations in src/Regression/
- Helper classes in src/Helpers/
- Test examples in tests/UnitTests/

**Good luck, junior developer! You've got this!**
