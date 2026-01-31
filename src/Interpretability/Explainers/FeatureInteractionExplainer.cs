using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic Feature Interaction detector using Friedman's H-statistic.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The H-statistic measures how much features interact with each other.
///
/// What is a feature interaction?
/// - When the effect of one feature depends on the value of another feature
/// - Example: "Education increases salary, but more so for people with more Experience"
/// - This means Education and Experience interact
///
/// How to interpret H-statistic values:
/// - H = 0: No interaction (features act independently)
/// - H = 1: Pure interaction (entire effect comes from interaction)
/// - H between 0 and 1: Partial interaction
///
/// Typical thresholds:
/// - H &lt; 0.05: Negligible interaction
/// - H 0.05-0.20: Weak interaction
/// - H 0.20-0.50: Moderate interaction
/// - H &gt; 0.50: Strong interaction
///
/// This implementation computes:
/// 1. Pairwise H-statistics (between two specific features)
/// 2. Overall H-statistic (one feature vs all others)
/// </para>
/// </remarks>
public class FeatureInteractionExplainer<T> : IGlobalExplainer<T, FeatureInteractionResult<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly Matrix<T> _data;
    private readonly int _gridSize;
    private readonly string[]? _featureNames;

    // Cached partial dependence values
    private readonly Dictionary<int, (T[] grid, T[] pd)> _pdCache = new();
    private readonly Dictionary<(int, int), T[,]> _pd2DCache = new();

    /// <inheritdoc/>
    public string MethodName => "FeatureInteraction";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => false;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <summary>
    /// Initializes a new Feature Interaction explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="data">The dataset to compute interactions on.</param>
    /// <param name="gridSize">Number of grid points for PD approximation (default: 20).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>gridSize</b>: Higher values give more accurate H-statistics but are slower.
    ///   20-30 is usually sufficient.
    /// </para>
    /// </remarks>
    public FeatureInteractionExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Matrix<T> data,
        int gridSize = 20,
        string[]? featureNames = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _data = data ?? throw new ArgumentNullException(nameof(data));

        if (data.Rows == 0)
            throw new ArgumentException("Data must have at least one row.", nameof(data));
        if (gridSize < 2)
            throw new ArgumentException("Grid size must be at least 2.", nameof(gridSize));
        if (featureNames != null && featureNames.Length != data.Columns)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match data.Columns ({data.Columns}).", nameof(featureNames));

        _gridSize = gridSize;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Computes the pairwise H-statistic between two features.
    /// </summary>
    /// <param name="feature1Index">Index of the first feature.</param>
    /// <param name="feature2Index">Index of the second feature.</param>
    /// <returns>The H-statistic value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The pairwise H-statistic measures interaction between
    /// two specific features. A high value means these features have a synergistic
    /// effect that wouldn't be captured by looking at them independently.
    ///
    /// Formula: H²(j,k) = sum[(f_jk - f_j - f_k)²] / sum[f_jk²]
    /// where f_jk is 2D partial dependence and f_j, f_k are 1D partial dependences.
    /// </para>
    /// </remarks>
    public T ComputePairwiseHStatistic(int feature1Index, int feature2Index)
    {
        int numFeatures = _data.Columns;

        if (feature1Index < 0 || feature1Index >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(feature1Index));
        if (feature2Index < 0 || feature2Index >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(feature2Index));
        if (feature1Index == feature2Index)
            throw new ArgumentException("Feature indices must be different.");

        // Ensure cache has PD values
        EnsurePDCached(feature1Index);
        EnsurePDCached(feature2Index);
        EnsurePD2DCached(feature1Index, feature2Index);

        // The 2D cache is keyed by (min, max) so get the canonical ordering
        int minIdx = Math.Min(feature1Index, feature2Index);
        int maxIdx = Math.Max(feature1Index, feature2Index);

        var (grid1, pd1) = _pdCache[feature1Index];
        var (grid2, pd2) = _pdCache[feature2Index];
        var pd2D = _pd2DCache[(minIdx, maxIdx)];

        // If feature indices are reversed from canonical order, swap pd1/pd2 to match 2D PD indexing
        if (feature1Index > feature2Index)
        {
            (pd1, pd2) = (pd2, pd1);
        }

        // Compute H-statistic
        double numerator = 0;
        double denominator = 0;

        for (int i = 0; i < _gridSize; i++)
        {
            for (int j = 0; j < _gridSize; j++)
            {
                double pdJoint = NumOps.ToDouble(pd2D[i, j]);
                double pd1Val = NumOps.ToDouble(pd1[i]);
                double pd2Val = NumOps.ToDouble(pd2[j]);

                double interaction = pdJoint - pd1Val - pd2Val;
                numerator += interaction * interaction;
                denominator += pdJoint * pdJoint;
            }
        }

        if (denominator < 1e-10)
            return NumOps.Zero;

        double hSquared = numerator / denominator;
        return NumOps.FromDouble(Math.Sqrt(Math.Max(0, Math.Min(1, hSquared))));
    }

    /// <summary>
    /// Computes the overall H-statistic for a single feature vs all others.
    /// </summary>
    /// <param name="featureIndex">Index of the feature to analyze.</param>
    /// <returns>The overall H-statistic value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The overall H-statistic tells you how much a feature
    /// interacts with ALL other features combined. A high value means this feature's
    /// effect depends heavily on other features.
    ///
    /// This is useful for identifying which features are most involved in interactions.
    /// </para>
    /// </remarks>
    public T ComputeOverallHStatistic(int featureIndex)
    {
        int numFeatures = _data.Columns;

        if (featureIndex < 0 || featureIndex >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(featureIndex));

        // Compute H-statistic with all other features
        double totalHSquared = 0;
        int count = 0;

        for (int j = 0; j < numFeatures; j++)
        {
            if (j == featureIndex)
                continue;

            var h = ComputePairwiseHStatistic(featureIndex, j);
            double hVal = NumOps.ToDouble(h);
            totalHSquared += hVal * hVal;
            count++;
        }

        if (count == 0)
            return NumOps.Zero;

        // Average H-statistic across all pairings
        return NumOps.FromDouble(Math.Sqrt(totalHSquared / count));
    }

    /// <inheritdoc/>
    public FeatureInteractionResult<T> ExplainGlobal(Matrix<T> data)
    {
        // Validate that passed data matches the training data dimensions
        // Note: PD caches are built from _data set in constructor
        if (data.Columns != _data.Columns)
            throw new ArgumentException(
                $"Data has {data.Columns} features but explainer was initialized with {_data.Columns} features.",
                nameof(data));

        int numFeatures = data.Columns;
        var pairwiseH = new Dictionary<(int, int), T>();
        var overallH = new T[numFeatures];

        // Compute all pairwise H-statistics
        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i + 1; j < numFeatures; j++)
            {
                pairwiseH[(i, j)] = ComputePairwiseHStatistic(i, j);
            }
        }

        // Compute overall H-statistics
        for (int i = 0; i < numFeatures; i++)
        {
            overallH[i] = ComputeOverallHStatistic(i);
        }

        return new FeatureInteractionResult<T>
        {
            FeatureNames = Enumerable.Range(0, numFeatures)
                .Select(i => _featureNames?[i] ?? $"Feature {i}")
                .ToArray(),
            PairwiseHStatistics = pairwiseH,
            OverallHStatistics = overallH
        };
    }

    /// <summary>
    /// Gets the top interacting feature pairs.
    /// </summary>
    /// <param name="topK">Number of top pairs to return.</param>
    /// <returns>List of feature pairs with their H-statistics, sorted by interaction strength.</returns>
    public List<(int feature1, int feature2, T hStatistic)> GetTopInteractions(int topK = 10)
    {
        int numFeatures = _data.Columns;
        var interactions = new List<(int, int, T)>();

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i + 1; j < numFeatures; j++)
            {
                var h = ComputePairwiseHStatistic(i, j);
                interactions.Add((i, j, h));
            }
        }

        return interactions
            .OrderByDescending(x => NumOps.ToDouble(x.Item3))
            .Take(topK)
            .ToList();
    }

    /// <summary>
    /// Ensures 1D partial dependence is cached for a feature.
    /// </summary>
    private void EnsurePDCached(int featureIndex)
    {
        if (_pdCache.ContainsKey(featureIndex))
            return;

        int n = _data.Rows;

        // Compute grid values
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(_data[i, featureIndex]);
            minVal = Math.Min(minVal, val);
            maxVal = Math.Max(maxVal, val);
        }

        var grid = new T[_gridSize];
        var pd = new T[_gridSize];
        double step = (maxVal - minVal) / (_gridSize - 1);

        for (int g = 0; g < _gridSize; g++)
        {
            grid[g] = NumOps.FromDouble(minVal + g * step);

            // Compute PD value at this grid point
            double sum = 0;
            var modifiedData = _data.Clone();
            for (int i = 0; i < n; i++)
            {
                modifiedData[i, featureIndex] = grid[g];
            }
            var predictions = _predictFunction(modifiedData);
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(predictions[i]);
            }
            pd[g] = NumOps.FromDouble(sum / n);
        }

        // Center the PD values
        double mean = pd.Average(v => NumOps.ToDouble(v));
        for (int g = 0; g < _gridSize; g++)
        {
            pd[g] = NumOps.FromDouble(NumOps.ToDouble(pd[g]) - mean);
        }

        _pdCache[featureIndex] = (grid, pd);
    }

    /// <summary>
    /// Ensures 2D partial dependence is cached for a feature pair.
    /// </summary>
    private void EnsurePD2DCached(int feature1Index, int feature2Index)
    {
        int f1 = Math.Min(feature1Index, feature2Index);
        int f2 = Math.Max(feature1Index, feature2Index);

        if (_pd2DCache.ContainsKey((f1, f2)))
            return;

        int n = _data.Rows;

        EnsurePDCached(f1);
        EnsurePDCached(f2);

        var (grid1, _) = _pdCache[f1];
        var (grid2, _) = _pdCache[f2];

        var pd2D = new T[_gridSize, _gridSize];

        for (int g1 = 0; g1 < _gridSize; g1++)
        {
            for (int g2 = 0; g2 < _gridSize; g2++)
            {
                var modifiedData = _data.Clone();
                for (int i = 0; i < n; i++)
                {
                    modifiedData[i, f1] = grid1[g1];
                    modifiedData[i, f2] = grid2[g2];
                }

                var predictions = _predictFunction(modifiedData);
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += NumOps.ToDouble(predictions[i]);
                }
                pd2D[g1, g2] = NumOps.FromDouble(sum / n);
            }
        }

        // Center the 2D PD values
        double mean = 0;
        for (int g1 = 0; g1 < _gridSize; g1++)
        {
            for (int g2 = 0; g2 < _gridSize; g2++)
            {
                mean += NumOps.ToDouble(pd2D[g1, g2]);
            }
        }
        mean /= (_gridSize * _gridSize);

        for (int g1 = 0; g1 < _gridSize; g1++)
        {
            for (int g2 = 0; g2 < _gridSize; g2++)
            {
                pd2D[g1, g2] = NumOps.FromDouble(NumOps.ToDouble(pd2D[g1, g2]) - mean);
            }
        }

        _pd2DCache[(f1, f2)] = pd2D;
    }
}
