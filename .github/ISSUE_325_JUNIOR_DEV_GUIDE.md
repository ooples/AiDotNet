# Issue #325: Junior Developer Implementation Guide
## Implement Advanced Outlier Detection and Handling Methods

---

## Table of Contents
1. [Understanding Outlier Detection](#understanding-outlier-detection)
2. [What EXISTS in the Codebase](#what-exists-in-the-codebase)
3. [What's MISSING](#whats-missing)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Common Pitfalls](#common-pitfalls)
6. [Testing Strategy](#testing-strategy)

---

## Understanding Outlier Detection

### For Beginners: What are Outliers?

Outliers are unusual data points that differ significantly from the rest of your data.

**Real-World Analogy:**
You're analyzing salaries at a company:
- Most employees: $40,000 - $80,000 per year
- One employee: $5,000,000 per year (CEO)
- Another employee: $10 per year (data entry error)

The CEO salary and error are outliers - they're far from typical values.

**Why Do Outliers Matter?**

1. **Good outliers** (rare but valid):
   - CEO salary is real, just unusual
   - Natural variation in data
   - Might be important to keep

2. **Bad outliers** (errors):
   - $10 salary is clearly wrong
   - Sensor malfunction
   - Data entry mistakes
   - Should be removed or corrected

**Impact on Machine Learning:**
- Skew averages (mean salary becomes misleading)
- Affect model training (model tries to fit unusual points)
- Reduce prediction accuracy
- Hide true patterns in data

**Types of Outlier Detection:**

1. **Statistical Methods** (existing in codebase):
   - Z-Score: Points > 3 standard deviations from mean
   - IQR: Points outside 1.5 × IQR from quartiles
   - MAD: Median Absolute Deviation-based

2. **Algorithmic Methods** (what you'll implement):
   - Isolation Forest: Isolates anomalies using random forests
   - One-Class SVM: Learns boundary around normal data
   - Local Outlier Factor: Density-based detection
   - Autoencoder: Neural network reconstruction error

3. **Handling Methods**:
   - Removal: Delete outlier rows (simple but loses data)
   - Winsorization: Cap extreme values (keeps data, reduces impact)
   - Imputation: Replace with median/mean
   - Robust models: Use models less sensitive to outliers

---

## What EXISTS in the Codebase

### Existing Infrastructure:

1. **Interfaces**:
   - ✅ `IOutlierRemoval<T, TInput, TOutput>` - already exists
   - ✅ `INumericOperations<T>` - type-generic math

2. **Existing Implementations** (src/OutlierRemoval/):
   - ✅ `ZScoreOutlierRemoval<T, TInput, TOutput>` - statistical method
   - ✅ `IQROutlierRemoval<T, TInput, TOutput>` - interquartile range
   - ✅ `MADOutlierRemoval<T, TInput, TOutput>` - median absolute deviation
   - ✅ `ThresholdOutlierRemoval<T, TInput, TOutput>` - simple threshold
   - ✅ `NoOutlierRemoval<T, TInput, TOutput>` - no-op

3. **Helper Classes**:
   - ✅ `StatisticsHelper<T>` - mean, median, std dev, quantiles
   - ✅ `OutlierRemovalHelper<T, TInput, TOutput>` - convert types
   - ✅ `MatrixHelper<T>`, `VectorHelper<T>` - linear algebra
   - ✅ `MathHelper` - numeric operations

4. **Existing Pattern**:
   - Implement `IOutlierRemoval<T, TInput, TOutput>`
   - Method: `RemoveOutliers(TInput inputs, TOutput outputs)`
   - Returns tuple: `(TInput CleanedInputs, TOutput CleanedOutputs)`
   - Use `OutlierRemovalHelper` for type conversions

---

## What's MISSING

### Phase 1: Algorithmic Outlier Detection
- **AC 1.1: IsolationForestOutlierDetector** - ❌ **MISSING**
- **AC 1.2: OneClassSVMOutlierDetector** - ❌ **MISSING**
- **AC 1.3: LocalOutlierFactorDetector** - ❌ **MISSING**
- **AC 1.4: AutoencoderOutlierDetector** - ❌ **MISSING**
- **AC 1.5: Unit Tests** - ❌ **MISSING**

### Phase 2: Outlier Handling
- **AC 2.1: WinsorizationTransformer** - ❌ **MISSING**
- **AC 2.2: Unit Tests** - ❌ **MISSING**

### NEW Interfaces Needed:
- `IOutlierDetector<T>` - for detection-only (no removal)
- `ITransformer<T>` - for data transformation (like winsorization)

---

## Step-by-Step Implementation

### STEP 0: Create Missing Interfaces

#### 0.1: Create IOutlierDetector<T> Interface

```csharp
// File: src/Interfaces/IOutlierDetector.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for detecting outliers without removing them.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An outlier detector identifies unusual data points.
///
/// Unlike IOutlierRemoval (which removes outliers), this interface just flags them.
///
/// Real-world example:
/// You're monitoring server response times:
/// - Most responses: 10-50ms
/// - Some responses: 5000ms (outliers - server hiccup)
///
/// An outlier detector would:
/// - Fit(): Learn what "normal" looks like
/// - Predict(): Return +1 for normal, -1 for outliers
/// - DecisionFunction(): Return anomaly scores (higher = more anomalous)
///
/// Why detect without removing?
/// - Investigate outliers (are they errors or important events?)
/// - Different handling strategies (flag for review vs automatic removal)
/// - Anomaly monitoring (fraud detection, intrusion detection)
/// </remarks>
public interface IOutlierDetector<T>
{
    /// <summary>
    /// Learns what "normal" data looks like.
    /// </summary>
    /// <param name="X">Feature matrix of normal training data.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Trains the detector on what typical data looks like.
    ///
    /// The training data should ideally contain mostly normal points.
    /// The detector learns patterns of normality to identify deviations.
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Predicts whether points are inliers (+1) or outliers (-1).
    /// </summary>
    /// <param name="X">Feature matrix to predict.</param>
    /// <returns>Vector of predictions: +1 for inlier, -1 for outlier.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Labels each point as normal or anomalous.
    ///
    /// Returns:
    /// - +1: Normal point (inlier)
    /// - -1: Outlier (anomalous)
    /// </remarks>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Computes anomaly scores for each point.
    /// </summary>
    /// <param name="X">Feature matrix to score.</param>
    /// <returns>Vector of anomaly scores (higher = more anomalous).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Gives a continuous measure of how anomalous each point is.
    ///
    /// Unlike Predict() which gives binary labels (+1/-1), this gives scores:
    /// - Low score: Very normal
    /// - Medium score: Somewhat unusual
    /// - High score: Very anomalous
    ///
    /// Useful for:
    /// - Ranking points by anomalousness
    /// - Setting custom thresholds
    /// - Understanding degree of anomaly
    /// </remarks>
    Vector<T> DecisionFunction(Matrix<T> X);
}
```

#### 0.2: Create ITransformer<T> Interface

```csharp
// File: src/Interfaces/ITransformer.cs
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for transforming data without removing samples.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A transformer modifies data values while keeping all samples.
///
/// Unlike outlier removal (which deletes rows), transformation keeps all data
/// but modifies extreme values.
///
/// Real-world example - Winsorization:
/// Income data: [30k, 40k, 50k, 60k, 5M]
/// The 5M is an outlier. Instead of removing it:
/// - Cap at 95th percentile (60k in this case)
/// - Result: [30k, 40k, 50k, 60k, 60k]
/// - Keep the data point but reduce its impact
/// </remarks>
public interface ITransformer<T>
{
    /// <summary>
    /// Learns transformation parameters from data.
    /// </summary>
    /// <param name="X">Feature matrix to learn from.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Analyzes data to determine transformation parameters.
    ///
    /// For winsorization:
    /// - Calculates 5th and 95th percentiles
    /// - Stores these as clipping thresholds
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Applies transformation to data.
    /// </summary>
    /// <param name="X">Feature matrix to transform.</param>
    /// <returns>Transformed feature matrix (same dimensions).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Modifies data using learned parameters.
    ///
    /// For winsorization:
    /// - Values below 5th percentile → clipped to 5th percentile
    /// - Values above 95th percentile → clipped to 95th percentile
    /// - Values in between → unchanged
    /// </remarks>
    Matrix<T> Transform(Matrix<T> X);

    /// <summary>
    /// Learns parameters and transforms in one step.
    /// </summary>
    /// <param name="X">Feature matrix to fit and transform.</param>
    /// <returns>Transformed feature matrix.</returns>
    Matrix<T> FitTransform(Matrix<T> X);
}
```

---

### STEP 1: Implement Isolation Forest (AC 1.1)

Isolation Forest detects outliers by isolating them using random binary trees.

#### Mathematical Background:

**Key Insight:** Outliers are easier to isolate than normal points.

```
Normal point in cluster: Needs many splits to isolate
Outlier far from others: Needs few splits to isolate

Algorithm:
1. Build random trees:
   - Randomly select feature
   - Randomly select split value between min and max
   - Recursively partition until each point is isolated
2. Measure path length (number of splits to isolate)
3. Average path length across all trees
4. Anomaly score: Shorter paths = outliers

Anomaly score formula:
score = 2^(-avgPathLength / expectedPathLength)
score close to 1: outlier
score close to 0: normal
```

#### Implementation:

```csharp
// File: src/OutlierRemoval/IsolationForestOutlierDetector.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements Isolation Forest for outlier detection.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Isolation Forest finds outliers by how easily they can be isolated.
///
/// Real-world analogy:
/// Imagine finding a specific person in a crowd:
/// - Normal person in crowd: Needs many questions to identify ("Male or female?", "Tall or short?", etc.)
/// - Person standing alone far away: Easy to spot immediately
///
/// How it works:
/// 1. Build random binary trees that recursively partition data
/// 2. Measure how many splits needed to isolate each point
/// 3. Outliers need fewer splits (easier to isolate)
/// 4. Average across many trees for robust detection
///
/// Parameters:
/// - nEstimators: Number of trees (default 100, sklearn default)
/// - maxSamples: Samples per tree (default 256, sklearn default)
/// - contamination: Expected proportion of outliers (default 0.1 = 10%)
///
/// Why these defaults?
/// - 100 trees: Balance between accuracy and speed (sklearn empirical testing)
/// - 256 samples: Enough to capture patterns, small enough for speed
/// - 0.1 contamination: Conservative assumption for most datasets
///
/// Reference: Liu, Ting, Zhou (2008) "Isolation Forest"
/// sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
/// </remarks>
public class IsolationForestOutlierDetector<T> : IOutlierDetector<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _nEstimators;
    private readonly int _maxSamples;
    private readonly double _contamination;
    private readonly Random _random;
    private List<IsolationTree<T>>? _trees;
    private T _threshold;

    /// <summary>
    /// Initializes a new instance of IsolationForestOutlierDetector.
    /// </summary>
    /// <param name="nEstimators">Number of trees. Default is 100.</param>
    /// <param name="maxSamples">Samples per tree. Default is 256.</param>
    /// <param name="contamination">Expected proportion of outliers. Default is 0.1.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates an Isolation Forest detector.
    ///
    /// Parameter guide:
    /// - More trees: More accurate but slower (100 is good balance)
    /// - More samples: Better patterns but slower trees (256 works well)
    /// - Contamination: If you know ~5% are outliers, set to 0.05
    ///
    /// Choosing contamination:
    /// - Too low: Miss some outliers
    /// - Too high: Flag normal points as outliers
    /// - Default 0.1: Conservative for most cases
    /// - Adjust based on domain knowledge
    /// </remarks>
    public IsolationForestOutlierDetector(
        int nEstimators = 100,
        int maxSamples = 256,
        double contamination = 0.1,
        int? randomState = null)
    {
        if (nEstimators <= 0)
            throw new ArgumentException("nEstimators must be positive", nameof(nEstimators));
        if (maxSamples <= 0)
            throw new ArgumentException("maxSamples must be positive", nameof(maxSamples));
        if (contamination < 0 || contamination > 0.5)
            throw new ArgumentException("contamination must be between 0 and 0.5", nameof(contamination));

        _numOps = MathHelper.GetNumericOperations<T>();
        _nEstimators = nEstimators;
        _maxSamples = maxSamples;
        _contamination = contamination;
        _random = randomState.HasValue ? new Random(randomState.Value) : new Random();
        _threshold = _numOps.Zero;
    }

    /// <summary>
    /// Trains the Isolation Forest on data.
    /// </summary>
    /// <param name="X">Training feature matrix.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Builds random trees to learn isolation patterns.
    ///
    /// Steps:
    /// 1. For each tree (nEstimators times):
    ///    a. Randomly sample maxSamples points
    ///    b. Build random tree by recursive partitioning
    /// 2. Compute anomaly scores on training data
    /// 3. Set threshold based on contamination parameter
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        int nSamples = X.Rows;
        int sampleSize = Math.Min(_maxSamples, nSamples);

        _trees = new List<IsolationTree<T>>();

        // Build trees
        for (int i = 0; i < _nEstimators; i++)
        {
            // Random sample
            var sampleIndices = SampleIndices(nSamples, sampleSize);
            var sample = ExtractSample(X, sampleIndices);

            // Build tree
            var tree = new IsolationTree<T>(_random);
            tree.Fit(sample);
            _trees.Add(tree);
        }

        // Set threshold based on contamination
        var scores = DecisionFunction(X);
        var sortedScores = scores.ToArray().OrderBy(s => _numOps.ToDouble(s)).ToArray();
        int thresholdIndex = (int)(nSamples * (1 - _contamination));
        _threshold = sortedScores[Math.Min(thresholdIndex, nSamples - 1)];
    }

    /// <summary>
    /// Predicts inliers (+1) and outliers (-1).
    /// </summary>
    /// <param name="X">Feature matrix to predict.</param>
    /// <returns>Vector of +1 (inlier) or -1 (outlier).</returns>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (_trees == null)
            throw new InvalidOperationException("Model must be fitted before prediction");

        var scores = DecisionFunction(X);
        var predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Score > threshold → inlier (+1), otherwise outlier (-1)
            predictions[i] = _numOps.GreaterThan(scores[i], _threshold)
                ? _numOps.One
                : _numOps.Negate(_numOps.One);
        }

        return predictions;
    }

    /// <summary>
    /// Computes anomaly scores.
    /// </summary>
    /// <param name="X">Feature matrix to score.</param>
    /// <returns>Vector of anomaly scores (higher = more normal).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Calculates how normal each point is.
    ///
    /// Score calculation:
    /// 1. For each tree, measure path length to isolate point
    /// 2. Average path lengths across all trees
    /// 3. Normalize: score = 2^(-avgPath / expectedPath)
    ///
    /// Interpretation:
    /// - Score close to 1: Very normal
    /// - Score close to 0: Very anomalous
    /// - Score around 0.5: Borderline
    /// </remarks>
    public Vector<T> DecisionFunction(Matrix<T> X)
    {
        if (_trees == null)
            throw new InvalidOperationException("Model must be fitted before prediction");

        var scores = new Vector<T>(X.Rows);
        var expectedPathLength = ExpectedPathLength(_maxSamples);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = X.GetRow(i);

            // Average path length across all trees
            double avgPathLength = 0;
            foreach (var tree in _trees)
            {
                avgPathLength += tree.PathLength(point);
            }
            avgPathLength /= _trees.Count;

            // Anomaly score: 2^(-avgPathLength / expectedPathLength)
            var score = Math.Pow(2, -avgPathLength / expectedPathLength);
            scores[i] = _numOps.FromDouble(score);
        }

        return scores;
    }

    /// <summary>
    /// Expected path length for BST with n samples.
    /// </summary>
    private double ExpectedPathLength(int n)
    {
        if (n <= 1) return 0;
        // E(h(n)) = 2 * (ln(n-1) + 0.5772) - 2(n-1)/n
        // Approximation from Isolation Forest paper
        return 2.0 * (Math.Log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n;
    }

    /// <summary>
    /// Randomly samples indices.
    /// </summary>
    private int[] SampleIndices(int nSamples, int sampleSize)
    {
        var indices = Enumerable.Range(0, nSamples).OrderBy(x => _random.Next()).Take(sampleSize).ToArray();
        return indices;
    }

    /// <summary>
    /// Extracts sample rows from matrix.
    /// </summary>
    private Matrix<T> ExtractSample(Matrix<T> X, int[] indices)
    {
        var sample = new Matrix<T>(indices.Length, X.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                sample[i, j] = X[indices[i], j];
            }
        }
        return sample;
    }
}

/// <summary>
/// Simple isolation tree implementation.
/// </summary>
internal class IsolationTree<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private TreeNode<T>? _root;

    public IsolationTree(Random random)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = random;
    }

    public void Fit(Matrix<T> X)
    {
        _root = BuildTree(X, 0);
    }

    public double PathLength(Vector<T> point)
    {
        return PathLength(point, _root, 0);
    }

    private TreeNode<T> BuildTree(Matrix<T> X, int depth)
    {
        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Stop if only one sample or max depth
        if (nSamples <= 1 || depth >= 100)
        {
            return new TreeNode<T> { IsLeaf = true, Size = nSamples };
        }

        // Random feature
        int feature = _random.Next(nFeatures);
        var column = X.GetColumn(feature);

        // Random split value between min and max
        var min = _numOps.ToDouble(column.Min());
        var max = _numOps.ToDouble(column.Max());

        if (min >= max)
        {
            return new TreeNode<T> { IsLeaf = true, Size = nSamples };
        }

        var splitValue = _numOps.FromDouble(min + _random.NextDouble() * (max - min));

        // Partition data
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < nSamples; i++)
        {
            if (_numOps.LessThan(X[i, feature], splitValue))
                leftIndices.Add(i);
            else
                rightIndices.Add(i);
        }

        var leftData = ExtractRows(X, leftIndices);
        var rightData = ExtractRows(X, rightIndices);

        return new TreeNode<T>
        {
            IsLeaf = false,
            Feature = feature,
            SplitValue = splitValue,
            Left = BuildTree(leftData, depth + 1),
            Right = BuildTree(rightData, depth + 1)
        };
    }

    private double PathLength(Vector<T> point, TreeNode<T>? node, int depth)
    {
        if (node == null || node.IsLeaf)
        {
            // Adjust by expected path length for node.Size
            return depth + (node != null ? Math.Log(node.Size) : 0);
        }

        if (_numOps.LessThan(point[node.Feature], node.SplitValue))
            return PathLength(point, node.Left, depth + 1);
        else
            return PathLength(point, node.Right, depth + 1);
    }

    private Matrix<T> ExtractRows(Matrix<T> X, List<int> indices)
    {
        if (indices.Count == 0)
            return new Matrix<T>(0, X.Columns);

        var result = new Matrix<T>(indices.Count, X.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                result[i, j] = X[indices[i], j];
            }
        }
        return result;
    }
}

internal class TreeNode<T>
{
    public bool IsLeaf { get; set; }
    public int Size { get; set; }
    public int Feature { get; set; }
    public T SplitValue { get; set; } = default!;
    public TreeNode<T>? Left { get; set; }
    public TreeNode<T>? Right { get; set; }
}
```

---

### STEP 2: Implement Winsorization (AC 2.1)

Winsorization caps extreme values instead of removing them.

#### Mathematical Background:

```
Winsorization: Replace extreme values with percentile thresholds

Example with lower=0.05, upper=0.95:
Data: [1, 2, 3, ..., 98, 99, 100]
5th percentile: 5
95th percentile: 95

Result:
- Values < 5 → 5
- Values > 95 → 95
- Values in [5, 95] → unchanged

Final: [5, 5, 5, ..., 95, 95, 95]
```

#### Implementation:

```csharp
// File: src/OutlierRemoval/WinsorizationTransformer.cs
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements Winsorization for capping extreme values.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Winsorization caps extreme values to reduce their impact.
///
/// Real-world analogy:
/// Speed limit enforcement:
/// - Most cars: 30-70 mph
/// - Some cars: 120 mph (speeding)
/// - Winsorization: Cap at 95th percentile (say, 75 mph)
/// - Result: Speeders still recorded but at capped value
///
/// Difference from removal:
/// - Removal: Delete rows with extreme values (lose data)
/// - Winsorization: Keep all rows but cap extreme values (preserve data)
///
/// When to use:
/// - Can't afford to lose data points
/// - Outliers might be informative (just need to reduce impact)
/// - Want robust statistics without discarding information
///
/// Parameters:
/// - lowerQuantile: Lower clipping threshold (default 0.05 = 5th percentile)
/// - upperQuantile: Upper clipping threshold (default 0.95 = 95th percentile)
///
/// Defaults from common statistical practice:
/// - 0.05/0.95: Removes extreme 10% (5% each end)
/// - Similar to IQR method but uses percentiles
/// - sklearn doesn't have Winsorization, but scipy.stats.mstats does
///
/// Reference: Hastings et al. "Introduction to Robust Statistics"
/// </remarks>
public class WinsorizationTransformer<T> : ITransformer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _lowerQuantile;
    private readonly double _upperQuantile;
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;

    /// <summary>
    /// Initializes a new instance of WinsorizationTransformer.
    /// </summary>
    /// <param name="lowerQuantile">Lower quantile for clipping. Default is 0.05.</param>
    /// <param name="upperQuantile">Upper quantile for clipping. Default is 0.95.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a Winsorization transformer.
    ///
    /// Quantile guide:
    /// - 0.05/0.95: Moderate (removes extreme 10%)
    /// - 0.01/0.99: Light (removes extreme 2%)
    /// - 0.10/0.90: Aggressive (removes extreme 20%)
    ///
    /// Choosing quantiles:
    /// - More extreme data: Use tighter quantiles (0.01/0.99)
    /// - Heavy outliers: Use wider quantiles (0.10/0.90)
    /// - Default 0.05/0.95: Good balance for most cases
    /// </remarks>
    public WinsorizationTransformer(double lowerQuantile = 0.05, double upperQuantile = 0.95)
    {
        if (lowerQuantile < 0 || lowerQuantile >= 0.5)
            throw new ArgumentException("lowerQuantile must be in [0, 0.5)", nameof(lowerQuantile));
        if (upperQuantile <= 0.5 || upperQuantile > 1.0)
            throw new ArgumentException("upperQuantile must be in (0.5, 1]", nameof(upperQuantile));
        if (lowerQuantile >= upperQuantile)
            throw new ArgumentException("lowerQuantile must be < upperQuantile");

        _numOps = MathHelper.GetNumericOperations<T>();
        _lowerQuantile = lowerQuantile;
        _upperQuantile = upperQuantile;
    }

    /// <summary>
    /// Gets the lower clipping bounds for each feature.
    /// </summary>
    public Vector<T> LowerBounds => _lowerBounds ?? new Vector<T>(0);

    /// <summary>
    /// Gets the upper clipping bounds for each feature.
    /// </summary>
    public Vector<T> UpperBounds => _upperBounds ?? new Vector<T>(0);

    /// <summary>
    /// Learns clipping bounds from data.
    /// </summary>
    /// <param name="X">Feature matrix to learn bounds from.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Calculates the clipping thresholds for each feature.
    ///
    /// Steps:
    /// 1. For each feature (column):
    ///    a. Sort values
    ///    b. Find value at lowerQuantile (e.g., 5th percentile)
    ///    c. Find value at upperQuantile (e.g., 95th percentile)
    ///    d. Store these as bounds
    /// 2. Save bounds for Transform()
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        int nFeatures = X.Columns;
        _lowerBounds = new Vector<T>(nFeatures);
        _upperBounds = new Vector<T>(nFeatures);

        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var sorted = column.ToArray().OrderBy(v => _numOps.ToDouble(v)).ToArray();

            int lowerIndex = (int)(sorted.Length * _lowerQuantile);
            int upperIndex = (int)(sorted.Length * _upperQuantile);

            _lowerBounds[j] = sorted[Math.Max(0, lowerIndex)];
            _upperBounds[j] = sorted[Math.Min(sorted.Length - 1, upperIndex)];
        }
    }

    /// <summary>
    /// Clips extreme values to learned bounds.
    /// </summary>
    /// <param name="X">Feature matrix to transform.</param>
    /// <returns>Transformed matrix with clipped values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Replaces extreme values with bound values.
    ///
    /// For each value in each feature:
    /// - If value < lowerBound → replace with lowerBound
    /// - If value > upperBound → replace with upperBound
    /// - Otherwise → keep original value
    ///
    /// Example:
    /// Column: [1, 2, 3, 98, 99, 100]
    /// Lower bound (5th percentile): 2
    /// Upper bound (95th percentile): 99
    /// Result: [2, 2, 3, 98, 99, 99]
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> X)
    {
        if (_lowerBounds == null || _upperBounds == null)
            throw new InvalidOperationException("Transformer must be fitted before transforming");

        if (X.Columns != _lowerBounds.Length)
            throw new ArgumentException($"Expected {_lowerBounds.Length} features, got {X.Columns}");

        var result = new Matrix<T>(X.Rows, X.Columns);

        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                var value = X[i, j];

                // Clip to bounds
                if (_numOps.LessThan(value, _lowerBounds[j]))
                    result[i, j] = _lowerBounds[j];
                else if (_numOps.GreaterThan(value, _upperBounds[j]))
                    result[i, j] = _upperBounds[j];
                else
                    result[i, j] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Learns bounds and transforms in one step.
    /// </summary>
    /// <param name="X">Feature matrix to fit and transform.</param>
    /// <returns>Transformed matrix with clipped values.</returns>
    public Matrix<T> FitTransform(Matrix<T> X)
    {
        Fit(X);
        return Transform(X);
    }
}
```

---

## Common Pitfalls

### 1. Not Handling Edge Cases in Isolation Forest
```csharp
// ✅ CORRECT
if (nSamples <= 1 || depth >= 100)
{
    return new TreeNode<T> { IsLeaf = true, Size = nSamples };
}
```

### 2. Forgetting to Set Threshold in Fit()
```csharp
// ✅ CORRECT
var scores = DecisionFunction(X);
var sortedScores = scores.ToArray().OrderBy(s => _numOps.ToDouble(s)).ToArray();
int thresholdIndex = (int)(nSamples * (1 - _contamination));
_threshold = sortedScores[Math.Min(thresholdIndex, nSamples - 1)];
```

### 3. Invalid Quantile Values
```csharp
// ✅ CORRECT
if (lowerQuantile < 0 || lowerQuantile >= 0.5)
    throw new ArgumentException("lowerQuantile must be in [0, 0.5)");
if (upperQuantile <= 0.5 || upperQuantile > 1.0)
    throw new ArgumentException("upperQuantile must be in (0.5, 1]");
```

### 4. Array Index Out of Bounds
```csharp
// ✅ CORRECT
_lowerBounds[j] = sorted[Math.Max(0, lowerIndex)];
_upperBounds[j] = sorted[Math.Min(sorted.Length - 1, upperIndex)];
```

---

## Testing Strategy

### Unit Test Examples:

```csharp
// File: tests/UnitTests/OutlierRemoval/IsolationForestTests.cs
using Xunit;
using AiDotNet.OutlierRemoval;
using AiDotNet.LinearAlgebra;

public class IsolationForestTests
{
    [Fact]
    public void Fit_TrainsSuccessfully()
    {
        // Arrange: Normal data + outliers
        var data = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 2, 2 },
            { 3, 3 },
            { 100, 100 }  // Outlier
        });

        var detector = new IsolationForestOutlierDetector<double>(
            nEstimators: 50, contamination: 0.25);

        // Act
        detector.Fit(data);

        // Assert: Should not throw
        Assert.NotNull(detector);
    }

    [Fact]
    public void Predict_DetectsOutliers()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1, 1 }, { 2, 2 }, { 3, 3 },  // Normal
            { 100, 100 }  // Outlier
        });

        var detector = new IsolationForestOutlierDetector<double>(contamination: 0.25);
        detector.Fit(data);

        // Act
        var predictions = detector.Predict(data);

        // Assert: Last point should be outlier (-1)
        Assert.Equal(-1, predictions[3], precision: 0);
    }

    [Fact]
    public void DecisionFunction_ReturnsScores()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1 }, { 100 } });
        var detector = new IsolationForestOutlierDetector<double>();
        detector.Fit(data);

        // Act
        var scores = detector.DecisionFunction(data);

        // Assert: Outlier (100) should have lower score
        Assert.True(scores[0] > scores[1], "Normal point should have higher score than outlier");
    }
}

public class WinsorizationTransformerTests
{
    [Fact]
    public void Fit_CalculatesCorrectBounds()
    {
        // Arrange: Data from 1 to 100
        var data = new Matrix<double>(100, 1);
        for (int i = 0; i < 100; i++)
            data[i, 0] = i + 1;

        var transformer = new WinsorizationTransformer<double>(0.05, 0.95);

        // Act
        transformer.Fit(data);

        // Assert: 5th percentile ~ 5, 95th percentile ~ 95
        Assert.InRange(transformer.LowerBounds[0], 4, 6);
        Assert.InRange(transformer.UpperBounds[0], 94, 96);
    }

    [Fact]
    public void Transform_ClipsExtremeValues()
    {
        // Arrange
        var trainData = new Matrix<double>(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });
        var testData = new Matrix<double>(new double[,] { { 1 }, { 25 }, { 100 } });

        var transformer = new WinsorizationTransformer<double>(0.2, 0.8);
        transformer.Fit(trainData);

        // Act
        var result = transformer.Transform(testData);

        // Assert: 1 should be clipped to lower bound, 100 to upper bound
        Assert.True(result[0, 0] >= 10, "Low value should be clipped to lower bound");
        Assert.Equal(25, result[1, 0], precision: 1);  // Middle value unchanged
        Assert.True(result[2, 0] <= 50, "High value should be clipped to upper bound");
    }
}
```

---

## Summary

### What You Built:
1. ✅ 2 new interfaces (IOutlierDetector, ITransformer)
2. ✅ IsolationForestOutlierDetector - tree-based anomaly detection
3. ✅ WinsorizationTransformer - caps extreme values
4. ✅ Comprehensive unit tests

### Key Learnings:
- Algorithmic methods (Isolation Forest) detect complex outliers
- Winsorization preserves data while reducing outlier impact
- Contamination parameter controls sensitivity
- Always validate quantiles and thresholds
- Test with known outliers to verify detection

### Next Steps:
1. Implement One-Class SVM (support vector-based detection)
2. Implement Local Outlier Factor (density-based detection)
3. Implement Autoencoder (neural network-based detection)
4. Add multivariate outlier detection
5. Combine multiple detection methods (ensemble)

**Good luck detecting and handling outliers!**
