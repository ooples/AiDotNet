using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic Accumulated Local Effects (ALE) explainer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ALE plots are an improved version of Partial Dependence Plots (PDP).
/// They show how a feature affects predictions, but handle correlated features much better.
///
/// The key difference from PDP:
/// - PDP: Averages over ALL data points (can create unrealistic combinations)
/// - ALE: Only looks at ACTUAL data points in each interval (realistic combinations)
///
/// Example: If you have Age and Years of Experience (highly correlated),
/// PDP might evaluate "Age=25 with 40 years experience" (impossible!).
/// ALE avoids this by only looking at real data within each age range.
///
/// How ALE works:
/// 1. Divide the feature into intervals (bins)
/// 2. For each interval, compute the effect as the average difference in predictions
///    when moving from the left edge to the right edge of the interval
/// 3. Accumulate these effects to get the final ALE curve
/// 4. Center the curve so the average effect is zero
///
/// When to use ALE vs PDP:
/// - Use ALE when features are correlated (most real-world cases)
/// - Use PDP when features are independent
/// </para>
/// </remarks>
public class AccumulatedLocalEffectsExplainer<T> : IGlobalExplainer<T, ALEResult<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly Matrix<T> _data;
    private readonly int _numIntervals;
    private readonly string[]? _featureNames;

    /// <inheritdoc/>
    public string MethodName => "AccumulatedLocalEffects";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => false;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <summary>
    /// Initializes a new Accumulated Local Effects explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="data">The dataset to compute ALE on.</param>
    /// <param name="numIntervals">Number of intervals for binning (default: 20).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>numIntervals</b>: How many bins to divide each feature into. More bins = smoother curve but needs more data.
    /// - <b>data</b>: Your actual dataset - ALE only evaluates realistic combinations from this data.
    /// </para>
    /// </remarks>
    public AccumulatedLocalEffectsExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Matrix<T> data,
        int numIntervals = 20,
        string[]? featureNames = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _data = data ?? throw new ArgumentNullException(nameof(data));

        if (data.Rows == 0)
            throw new ArgumentException("Data must have at least one row.", nameof(data));
        if (numIntervals < 2)
            throw new ArgumentException("Number of intervals must be at least 2.", nameof(numIntervals));

        _numIntervals = numIntervals;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Computes ALE for a single feature.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to analyze.</param>
    /// <returns>ALE result for the feature.</returns>
    public ALEResult<T> ComputeForFeature(int featureIndex)
    {
        return ComputeForFeatures(new[] { featureIndex });
    }

    /// <summary>
    /// Computes ALE for multiple features.
    /// </summary>
    /// <param name="featureIndices">The indices of features to analyze.</param>
    /// <returns>ALE result for all specified features.</returns>
    public ALEResult<T> ComputeForFeatures(int[] featureIndices)
    {
        if (featureIndices == null || featureIndices.Length == 0)
            throw new ArgumentException("At least one feature index is required.", nameof(featureIndices));

        int numFeatures = _data.Columns;
        foreach (var idx in featureIndices)
        {
            if (idx < 0 || idx >= numFeatures)
                throw new ArgumentOutOfRangeException(nameof(featureIndices), $"Feature index {idx} is out of range [0, {numFeatures}).");
        }

        var result = new ALEResult<T>
        {
            FeatureIndices = featureIndices,
            FeatureNames = featureIndices.Select(i => _featureNames?[i] ?? $"Feature {i}").ToArray(),
            IntervalBounds = new Dictionary<int, T[]>(),
            ALEValues = new Dictionary<int, T[]>(),
            IntervalCounts = new Dictionary<int, int[]>()
        };

        foreach (var featureIndex in featureIndices)
        {
            ComputeSingleFeatureALE(featureIndex, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public ALEResult<T> ExplainGlobal(Matrix<T> data)
    {
        // Compute ALE for all features
        var allFeatureIndices = Enumerable.Range(0, data.Columns).ToArray();
        return ComputeForFeatures(allFeatureIndices);
    }

    /// <summary>
    /// Computes ALE for a single feature.
    /// </summary>
    private void ComputeSingleFeatureALE(int featureIndex, ALEResult<T> result)
    {
        int n = _data.Rows;

        // Get feature values and sort them
        var featureValues = new double[n];
        var sortedIndices = new int[n];

        for (int i = 0; i < n; i++)
        {
            featureValues[i] = NumOps.ToDouble(_data[i, featureIndex]);
            sortedIndices[i] = i;
        }

        // Sort indices by feature value
        Array.Sort(sortedIndices, (a, b) => featureValues[a].CompareTo(featureValues[b]));

        // Compute quantile-based interval bounds
        var intervalBounds = new T[_numIntervals + 1];
        for (int i = 0; i <= _numIntervals; i++)
        {
            int idx = (int)Math.Round((double)i / _numIntervals * (n - 1));
            intervalBounds[i] = NumOps.FromDouble(featureValues[sortedIndices[idx]]);
        }

        // Ensure unique bounds (remove duplicates by adjusting slightly)
        for (int i = 1; i <= _numIntervals; i++)
        {
            double current = NumOps.ToDouble(intervalBounds[i]);
            double prev = NumOps.ToDouble(intervalBounds[i - 1]);
            if (current <= prev)
            {
                intervalBounds[i] = NumOps.FromDouble(prev + 1e-10);
            }
        }

        result.IntervalBounds[featureIndex] = intervalBounds;

        // Compute local effects for each interval
        var localEffects = new double[_numIntervals];
        var intervalCounts = new int[_numIntervals];

        for (int i = 0; i < n; i++)
        {
            double val = featureValues[i];

            // Find which interval this point belongs to
            int interval = FindInterval(val, intervalBounds);
            if (interval < 0 || interval >= _numIntervals)
                continue;

            intervalCounts[interval]++;

            // Compute local effect: f(x_right) - f(x_left)
            double leftBound = NumOps.ToDouble(intervalBounds[interval]);
            double rightBound = NumOps.ToDouble(intervalBounds[interval + 1]);

            // Create copies of the data point with feature at interval bounds
            var leftData = CreateSingleRowMatrix(i, featureIndex, leftBound);
            var rightData = CreateSingleRowMatrix(i, featureIndex, rightBound);

            var leftPred = _predictFunction(leftData)[0];
            var rightPred = _predictFunction(rightData)[0];

            double effect = NumOps.ToDouble(rightPred) - NumOps.ToDouble(leftPred);
            localEffects[interval] += effect;
        }

        // Average local effects within each interval
        for (int i = 0; i < _numIntervals; i++)
        {
            if (intervalCounts[i] > 0)
            {
                localEffects[i] /= intervalCounts[i];
            }
        }

        // Accumulate effects
        var aleValues = new T[_numIntervals + 1];
        aleValues[0] = NumOps.Zero;
        double cumSum = 0;

        for (int i = 0; i < _numIntervals; i++)
        {
            cumSum += localEffects[i];
            aleValues[i + 1] = NumOps.FromDouble(cumSum);
        }

        // Center the ALE values (mean should be zero)
        double mean = 0;
        for (int i = 0; i <= _numIntervals; i++)
        {
            mean += NumOps.ToDouble(aleValues[i]);
        }
        mean /= (_numIntervals + 1);

        for (int i = 0; i <= _numIntervals; i++)
        {
            aleValues[i] = NumOps.FromDouble(NumOps.ToDouble(aleValues[i]) - mean);
        }

        result.ALEValues[featureIndex] = aleValues;
        result.IntervalCounts[featureIndex] = intervalCounts;
    }

    /// <summary>
    /// Finds the interval index for a given value.
    /// </summary>
    private int FindInterval(double value, T[] bounds)
    {
        for (int i = 0; i < bounds.Length - 1; i++)
        {
            double left = NumOps.ToDouble(bounds[i]);
            double right = NumOps.ToDouble(bounds[i + 1]);
            if (value >= left && value < right)
                return i;
        }
        // Last interval includes the right bound
        return bounds.Length - 2;
    }

    /// <summary>
    /// Creates a single-row matrix with a modified feature value.
    /// </summary>
    private Matrix<T> CreateSingleRowMatrix(int rowIndex, int featureIndex, double newValue)
    {
        var matrix = new Matrix<T>(1, _data.Columns);
        for (int j = 0; j < _data.Columns; j++)
        {
            matrix[0, j] = j == featureIndex
                ? NumOps.FromDouble(newValue)
                : _data[rowIndex, j];
        }
        return matrix;
    }

    /// <summary>
    /// Computes 2D ALE for feature interactions.
    /// </summary>
    /// <param name="feature1Index">Index of the first feature.</param>
    /// <param name="feature2Index">Index of the second feature.</param>
    /// <returns>2D ALE result showing interaction effect.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> 2D ALE shows how two features interact.
    /// The result is the "pure" interaction effect - what's left after removing
    /// the individual effects of each feature.
    ///
    /// If the 2D ALE is all zeros, the features don't interact.
    /// Non-zero values show where the combined effect differs from
    /// the sum of individual effects.
    /// </para>
    /// </remarks>
    public ALE2DResult<T> ComputeInteraction(int feature1Index, int feature2Index)
    {
        int n = _data.Rows;
        int numFeatures = _data.Columns;

        if (feature1Index < 0 || feature1Index >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(feature1Index));
        if (feature2Index < 0 || feature2Index >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(feature2Index));

        // Use fewer intervals for 2D (computational cost is O(intervals^2))
        int intervals2D = Math.Min(_numIntervals, 10);

        // Get quantile-based bounds for both features
        var bounds1 = ComputeQuantileBounds(feature1Index, intervals2D);
        var bounds2 = ComputeQuantileBounds(feature2Index, intervals2D);

        // Compute 2D local effects
        var aleValues = new T[intervals2D, intervals2D];
        var counts = new int[intervals2D, intervals2D];

        for (int i = 0; i < n; i++)
        {
            double val1 = NumOps.ToDouble(_data[i, feature1Index]);
            double val2 = NumOps.ToDouble(_data[i, feature2Index]);

            int int1 = FindInterval(val1, bounds1);
            int int2 = FindInterval(val2, bounds2);

            if (int1 < 0 || int1 >= intervals2D || int2 < 0 || int2 >= intervals2D)
                continue;

            counts[int1, int2]++;

            // Compute 2D local effect using second-order differences
            double left1 = NumOps.ToDouble(bounds1[int1]);
            double right1 = NumOps.ToDouble(bounds1[int1 + 1]);
            double left2 = NumOps.ToDouble(bounds2[int2]);
            double right2 = NumOps.ToDouble(bounds2[int2 + 1]);

            // f(right1, right2) - f(left1, right2) - f(right1, left2) + f(left1, left2)
            var pred_rr = PredictWithModifiedFeatures(i, feature1Index, right1, feature2Index, right2);
            var pred_lr = PredictWithModifiedFeatures(i, feature1Index, left1, feature2Index, right2);
            var pred_rl = PredictWithModifiedFeatures(i, feature1Index, right1, feature2Index, left2);
            var pred_ll = PredictWithModifiedFeatures(i, feature1Index, left1, feature2Index, left2);

            double effect = pred_rr - pred_lr - pred_rl + pred_ll;
            aleValues[int1, int2] = NumOps.FromDouble(NumOps.ToDouble(aleValues[int1, int2]) + effect);
        }

        // Average and accumulate
        for (int i1 = 0; i1 < intervals2D; i1++)
        {
            for (int i2 = 0; i2 < intervals2D; i2++)
            {
                if (counts[i1, i2] > 0)
                {
                    aleValues[i1, i2] = NumOps.FromDouble(NumOps.ToDouble(aleValues[i1, i2]) / counts[i1, i2]);
                }
            }
        }

        // Accumulate in both dimensions
        var accumulated = Accumulate2D(aleValues, intervals2D);

        // Center
        double mean2D = 0;
        int totalCells = intervals2D * intervals2D;
        for (int i1 = 0; i1 < intervals2D; i1++)
        {
            for (int i2 = 0; i2 < intervals2D; i2++)
            {
                mean2D += NumOps.ToDouble(accumulated[i1, i2]);
            }
        }
        mean2D /= totalCells;

        for (int i1 = 0; i1 < intervals2D; i1++)
        {
            for (int i2 = 0; i2 < intervals2D; i2++)
            {
                accumulated[i1, i2] = NumOps.FromDouble(NumOps.ToDouble(accumulated[i1, i2]) - mean2D);
            }
        }

        return new ALE2DResult<T>
        {
            Feature1Index = feature1Index,
            Feature2Index = feature2Index,
            Feature1Name = _featureNames?[feature1Index] ?? $"Feature {feature1Index}",
            Feature2Name = _featureNames?[feature2Index] ?? $"Feature {feature2Index}",
            Bounds1 = bounds1,
            Bounds2 = bounds2,
            ALEValues2D = accumulated
        };
    }

    /// <summary>
    /// Computes quantile-based bounds for a feature.
    /// </summary>
    private T[] ComputeQuantileBounds(int featureIndex, int numIntervals)
    {
        int n = _data.Rows;
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(_data[i, featureIndex]);
        }
        Array.Sort(values);

        var bounds = new T[numIntervals + 1];
        for (int i = 0; i <= numIntervals; i++)
        {
            int idx = (int)Math.Round((double)i / numIntervals * (n - 1));
            bounds[i] = NumOps.FromDouble(values[idx]);
        }

        // Ensure unique bounds
        for (int i = 1; i <= numIntervals; i++)
        {
            double current = NumOps.ToDouble(bounds[i]);
            double prev = NumOps.ToDouble(bounds[i - 1]);
            if (current <= prev)
            {
                bounds[i] = NumOps.FromDouble(prev + 1e-10);
            }
        }

        return bounds;
    }

    /// <summary>
    /// Makes a prediction with two modified feature values.
    /// </summary>
    private double PredictWithModifiedFeatures(int rowIndex, int feat1, double val1, int feat2, double val2)
    {
        var matrix = new Matrix<T>(1, _data.Columns);
        for (int j = 0; j < _data.Columns; j++)
        {
            if (j == feat1)
                matrix[0, j] = NumOps.FromDouble(val1);
            else if (j == feat2)
                matrix[0, j] = NumOps.FromDouble(val2);
            else
                matrix[0, j] = _data[rowIndex, j];
        }
        return NumOps.ToDouble(_predictFunction(matrix)[0]);
    }

    /// <summary>
    /// Accumulates 2D local effects.
    /// </summary>
    private T[,] Accumulate2D(T[,] localEffects, int size)
    {
        var result = new T[size, size];

        // First accumulate along first dimension
        for (int i2 = 0; i2 < size; i2++)
        {
            double cumSum = 0;
            for (int i1 = 0; i1 < size; i1++)
            {
                cumSum += NumOps.ToDouble(localEffects[i1, i2]);
                result[i1, i2] = NumOps.FromDouble(cumSum);
            }
        }

        // Then accumulate along second dimension
        for (int i1 = 0; i1 < size; i1++)
        {
            double cumSum = 0;
            for (int i2 = 0; i2 < size; i2++)
            {
                cumSum += NumOps.ToDouble(result[i1, i2]);
                result[i1, i2] = NumOps.FromDouble(cumSum);
            }
        }

        return result;
    }
}

/// <summary>
/// Represents the result of an ALE (Accumulated Local Effects) analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ALEResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the feature indices analyzed.
    /// </summary>
    public int[] FeatureIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the interval bounds for each feature (feature index -> bounds).
    /// </summary>
    public Dictionary<int, T[]> IntervalBounds { get; set; } = new();

    /// <summary>
    /// Gets or sets the ALE values for each feature (feature index -> ALE values at interval boundaries).
    /// </summary>
    public Dictionary<int, T[]> ALEValues { get; set; } = new();

    /// <summary>
    /// Gets or sets the count of data points in each interval (feature index -> counts).
    /// </summary>
    public Dictionary<int, int[]> IntervalCounts { get; set; } = new();

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string> { "Accumulated Local Effects (ALE) Analysis:" };

        foreach (var featureIndex in FeatureIndices)
        {
            var aleValues = ALEValues[featureIndex];
            double minAle = aleValues.Min(v => NumOps.ToDouble(v));
            double maxAle = aleValues.Max(v => NumOps.ToDouble(v));
            double range = maxAle - minAle;

            string featureName = FeatureNames[Array.IndexOf(FeatureIndices, featureIndex)];
            lines.Add($"  {featureName}: ALE range = {range:F4} (min={minAle:F4}, max={maxAle:F4})");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Represents the result of a 2D ALE analysis (feature interaction).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ALE2DResult<T>
{
    /// <summary>
    /// Gets or sets the first feature index.
    /// </summary>
    public int Feature1Index { get; set; }

    /// <summary>
    /// Gets or sets the second feature index.
    /// </summary>
    public int Feature2Index { get; set; }

    /// <summary>
    /// Gets or sets the first feature name.
    /// </summary>
    public string Feature1Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the second feature name.
    /// </summary>
    public string Feature2Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the interval bounds for the first feature.
    /// </summary>
    public T[] Bounds1 { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the interval bounds for the second feature.
    /// </summary>
    public T[] Bounds2 { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the 2D ALE values [feature1 interval, feature2 interval].
    /// </summary>
    public T[,] ALEValues2D { get; set; } = new T[0, 0];
}
