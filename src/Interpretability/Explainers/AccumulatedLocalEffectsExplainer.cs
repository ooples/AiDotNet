using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        Guard.NotNull(data);
        _data = data;

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
    /// <remarks>
    /// Note: This method computes ALE using the data provided in the constructor, not the passed data parameter.
    /// The data parameter is used only to determine the number of features to analyze.
    /// To compute ALE on different data, create a new explainer instance with that data.
    /// </remarks>
    public ALEResult<T> ExplainGlobal(Matrix<T> data)
    {
        if (data.Columns != _data.Columns)
            throw new ArgumentException(
                $"Data has {data.Columns} features but explainer was initialized with {_data.Columns} features. " +
                "ALE computation uses the data provided in the constructor.",
                nameof(data));

        // Compute ALE for all features using the constructor-provided data
        var allFeatureIndices = Enumerable.Range(0, _data.Columns).ToArray();
        return ComputeForFeatures(allFeatureIndices);
    }

    /// <summary>
    /// Computes ALE for a single feature.
    /// </summary>
    private void ComputeSingleFeatureALE(int featureIndex, ALEResult<T> result)
    {
        var engine = AiDotNetEngine.Current;
        var featureValues = _data.GetColumn(featureIndex);

        // Compute quantile-based interval bounds
        var intervalBounds = QuantileBounds(featureValues, _numIntervals);
        result.IntervalBounds[featureIndex] = intervalBounds.ToArray();

        // Find which interval each point belongs to
        var intervals = FindIntervals(featureValues, intervalBounds);
        var intervalIndex = new Tensor<int>(intervals, new[] { intervals.Length });
        var intervalCounts = new int[_numIntervals];
        foreach (int interval in intervals)
        {
            intervalCounts[interval]++;
        }

        // Compute local effect: f(x_right) - f(x_left), with the feature moved to the interval bounds
        var boundsTensor = intervalBounds.AsTensor();
        var leftBounds = engine.TensorIndexSelect(boundsTensor, intervalIndex, 0).ToVector();
        var rightBounds = engine.TensorIndexSelect(boundsTensor,
            engine.TensorAddScalar(intervalIndex, 1), 0).ToVector();
        var leftPred = _predictFunction(WithColumn(_data, featureIndex, leftBounds));
        var rightPred = _predictFunction(WithColumn(_data, featureIndex, rightBounds));
        var effects = engine.Subtract(rightPred.GetSubVector(0, intervals.Length), leftPred.GetSubVector(0, intervals.Length));

        // Average local effects within each interval
        var localEffects = AverageByGroup(effects, intervalIndex, intervalCounts);

        // Accumulate effects: [0, cumsum(localEffects)]
        var cumulative = engine.TensorCumSum(localEffects, 0);
        var aleValues = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1 }), cumulative }, 0);

        // Center as in Apley and Zhu (2020), eq. (16), and their ALEPlot package: subtract the mean of the
        // uncentred effect over the data, i.e. the interval midpoint values (f(z_{k-1}) + f(z_k)) / 2 weighted by the
        // interval counts. (An unweighted mean over the grid points ignores where the data lies.)
        var midpoints = engine.TensorMultiplyScalar(engine.TensorAdd(
            engine.TensorSlice(aleValues, new[] { 0 }, new[] { _numIntervals }),
            engine.TensorSlice(aleValues, new[] { 1 }, new[] { _numIntervals })), NumOps.FromDouble(0.5));
        T dataMean = engine.TensorSum(engine.TensorMultiply(midpoints, CountWeights(intervalCounts, intervals.Length)));
        aleValues = engine.TensorSubtractScalar(aleValues, dataMean);

        result.ALEValues[featureIndex] = aleValues.ToArray();
        result.IntervalCounts[featureIndex] = intervalCounts;
    }

    /// <summary>
    /// The share count / total of every count, as T (the counts themselves can exceed what T holds exactly).
    /// </summary>
    private static Tensor<T> CountWeights(int[] counts, int total)
    {
        return new Tensor<double>(counts.Select(count => (double)count / total).ToArray(), new[] { counts.Length })
            .Cast<T>();
    }

    /// <summary>
    /// The mean of <paramref name="values"/> within each group (0 for an empty group).
    /// </summary>
    private static Tensor<T> AverageByGroup(Vector<T> values, Tensor<int> groupIndex, int[] groupCounts)
    {
        var engine = AiDotNetEngine.Current;
        int groups = groupCounts.Length;
        var sums = engine.TensorIndexAdd(new Tensor<T>(new[] { groups }), 0, groupIndex, values.AsTensor());
        var counts = new Tensor<int>(groupCounts, new[] { groups }).Cast<T>();
        return engine.TensorDivide(sums, engine.TensorMax(counts, NumOps.One));
    }

    /// <summary>
    /// A copy of <paramref name="data"/> with column <paramref name="column"/> replaced by <paramref name="values"/>.
    /// </summary>
    private static Matrix<T> WithColumn(Matrix<T> data, int column, Vector<T> values)
    {
        var modified = data.Clone();
        AiDotNetEngine.Current.SetColumn(modified, column, values);
        return modified;
    }

    /// <summary>
    /// Finds the interval index for a given value.
    /// </summary>
    private static int[] FindIntervals(Vector<T> values, Vector<T> bounds)
    {
        // Apley and Zhu (2020) and ALEPlot use right-closed intervals: the interval i with
        // bounds[i] < value <= bounds[i + 1], the first one also holding bounds[0] (cut(include.lowest = TRUE)).
        // Values outside the bounds fall in the nearest end interval.
        int last = bounds.Length - 2;
        var boundsBelow = AiDotNetEngine.Current.TensorSearchSorted(bounds.AsTensor(), values.AsTensor(), false).ToArray();
        return boundsBelow.Select(count => Math.Min(Math.Max(count - 1, 0), last)).ToArray();
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
        var engine = AiDotNetEngine.Current;

        // Get quantile-based bounds for both features
        var values1 = _data.GetColumn(feature1Index);
        var values2 = _data.GetColumn(feature2Index);
        var bounds1 = QuantileBounds(values1, intervals2D);
        var bounds2 = QuantileBounds(values2, intervals2D);

        // Interval cell of every point
        var int1 = FindIntervals(values1, bounds1);
        var int2 = FindIntervals(values2, bounds2);
        var index1 = new Tensor<int>(int1, new[] { n });
        var index2 = new Tensor<int>(int2, new[] { n });
        var cells = int1.Zip(int2, (a, b) => a * intervals2D + b).ToArray();
        var counts = new int[intervals2D * intervals2D];
        foreach (int cell in cells)
        {
            counts[cell]++;
        }

        // Compute 2D local effect using second-order differences:
        // f(right1, right2) - f(left1, right2) - f(right1, left2) + f(left1, left2)
        var left1 = engine.TensorIndexSelect(bounds1.AsTensor(), index1, 0).ToVector();
        var right1 = engine.TensorIndexSelect(bounds1.AsTensor(), engine.TensorAddScalar(index1, 1), 0).ToVector();
        var left2 = engine.TensorIndexSelect(bounds2.AsTensor(), index2, 0).ToVector();
        var right2 = engine.TensorIndexSelect(bounds2.AsTensor(), engine.TensorAddScalar(index2, 1), 0).ToVector();
        var predRR = PredictWithModifiedFeatures(feature1Index, right1, feature2Index, right2);
        var predLR = PredictWithModifiedFeatures(feature1Index, left1, feature2Index, right2);
        var predRL = PredictWithModifiedFeatures(feature1Index, right1, feature2Index, left2);
        var predLL = PredictWithModifiedFeatures(feature1Index, left1, feature2Index, left2);
        var effects = engine.Add(engine.Subtract(engine.Subtract(predRR, predLR), predRL), predLL);

        // Average per cell; an empty cell takes the effect of the nearest non-empty cell (cell centres scaled by
        // the feature ranges), as in Apley and Zhu's ALEPlot
        var cellIndex = new Tensor<int>(cells, new[] { n });
        var cellAverages = AverageByGroup(effects, cellIndex, counts);
        var source = NearestNonEmptyCells(counts, bounds1, bounds2, intervals2D);
        var cellEffects = engine.TensorIndexSelect(cellAverages, new Tensor<int>(source, new[] { source.Length }), 0)
            .Reshape(new[] { intervals2D, intervals2D });

        var accumulated = SecondOrderAle(cellEffects, CountWeights(counts, n).Reshape(new[] { intervals2D, intervals2D }));

        return new ALE2DResult<T>
        {
            Feature1Index = feature1Index,
            Feature2Index = feature2Index,
            Feature1Name = _featureNames?[feature1Index] ?? $"Feature {feature1Index}",
            Feature2Name = _featureNames?[feature2Index] ?? $"Feature {feature2Index}",
            Bounds1 = bounds1.ToArray(),
            Bounds2 = bounds2.ToArray(),
            ALEValues2D = accumulated.ToMatrix()
        };
    }

    /// <summary>
    /// The second-order ALE of Apley and Zhu (2020), eqs. (13)-(15), as computed by their ALEPlot package, on the
    /// (K + 1) x (K + 1) grid of interval bounds: the cell effects accumulated over both features from a zero row
    /// and column, minus the two first-order effects they contain, centred to a zero data mean.
    /// </summary>
    /// <param name="cellEffects">[K, K] average second-order difference per cell.</param>
    /// <param name="weights">[K, K] share of the data in each cell.</param>
    private static Tensor<T> SecondOrderAle(Tensor<T> cellEffects, Tensor<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        int k = cellEffects.Shape[0];
        var half = NumOps.FromDouble(0.5);

        // f[i, j] = sum over cells below i and j, with f[0, :] = f[:, 0] = 0
        var inner = engine.TensorCumSum(engine.TensorCumSum(cellEffects, 0), 1);
        var withColumn = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { k, 1 }), inner }, 1);
        var f = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, k + 1 }), withColumn }, 0);

        // First-order effect of feature 1: row steps averaged over each cell's two edges, weighted by the cells
        var rowSteps = engine.TensorSubtract(Block(f, 1, 0, k, k + 1), Block(f, 0, 0, k, k + 1));
        var rowStepPerCell = engine.TensorMultiplyScalar(
            engine.TensorAdd(Block(rowSteps, 0, 0, k, k), Block(rowSteps, 0, 1, k, k)), half);
        var f1 = Accumulate(WeightedAverage(rowStepPerCell, weights, 1));

        // First-order effect of feature 2: column steps averaged over each cell's two edges
        var columnSteps = engine.TensorSubtract(Block(f, 0, 1, k + 1, k), Block(f, 0, 0, k + 1, k));
        var columnStepPerCell = engine.TensorMultiplyScalar(
            engine.TensorAdd(Block(columnSteps, 0, 0, k, k), Block(columnSteps, 1, 0, k, k)), half);
        var f2 = Accumulate(WeightedAverage(columnStepPerCell, weights, 0));

        var shape = new[] { k + 1, k + 1 };
        var pure = engine.TensorSubtract(engine.TensorSubtract(f,
            engine.TensorBroadcastTo(f1.Reshape(new[] { k + 1, 1 }), shape)),
            engine.TensorBroadcastTo(f2.Reshape(new[] { 1, k + 1 }), shape));

        // Centre: the data mean of the cell averages of the four corners
        var corners = engine.TensorAdd(
            engine.TensorAdd(Block(pure, 0, 0, k, k), Block(pure, 0, 1, k, k)),
            engine.TensorAdd(Block(pure, 1, 0, k, k), Block(pure, 1, 1, k, k)));
        T dataMean = NumOps.Divide(
            engine.TensorSum(engine.TensorMultiply(engine.TensorMultiplyScalar(corners, NumOps.FromDouble(0.25)), weights)),
            engine.TensorSum(weights));
        return engine.TensorSubtractScalar(pure, dataMean);
    }

    /// <summary>
    /// The [rows, columns] block of a matrix tensor starting at (<paramref name="row"/>, <paramref name="column"/>).
    /// </summary>
    private static Tensor<T> Block(Tensor<T> matrix, int row, int column, int rows, int columns)
    {
        return AiDotNetEngine.Current.TensorSlice(matrix, new[] { row, column }, new[] { rows, columns });
    }

    /// <summary>
    /// The weighted average of <paramref name="values"/> along <paramref name="axis"/> (0 where the weights sum to 0).
    /// </summary>
    private static Tensor<T> WeightedAverage(Tensor<T> values, Tensor<T> weights, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var sums = engine.ReduceSum(engine.TensorMultiply(values, weights), new[] { axis }, false);
        var totals = engine.ReduceSum(weights, new[] { axis }, false);
        var occupied = engine.TensorGreaterThan(totals, NumOps.Zero);
        var ones = engine.TensorOnesLike(totals);
        return engine.TensorMultiply(engine.TensorDivide(sums, engine.TensorWhere(occupied, totals, ones)), occupied);
    }

    /// <summary>
    /// [0, cumsum(steps)].
    /// </summary>
    private static Tensor<T> Accumulate(Tensor<T> steps)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1 }), engine.TensorCumSum(steps, 0) }, 0);
    }

    /// <summary>
    /// For every cell of the [k, k] grid (row-major) the cell whose average it takes: itself when it holds data,
    /// otherwise the nearest non-empty cell by the distance between cell centres, each feature scaled by its range
    /// (the first such cell on ties).
    /// </summary>
    private static int[] NearestNonEmptyCells(int[] counts, Vector<T> bounds1, Vector<T> bounds2, int k)
    {
        var z1 = Tensor<T>.FromVector(bounds1).Cast<double>().ToArray();
        var z2 = Tensor<T>.FromVector(bounds2).Cast<double>().ToArray();
        double range1 = z1[k] - z1[0] > 0 ? z1[k] - z1[0] : 1.0;
        double range2 = z2[k] - z2[0] > 0 ? z2[k] - z2[0] : 1.0;
        var centres1 = Enumerable.Range(0, k).Select(i => (z1[i] + z1[i + 1]) / 2 / range1).ToArray();
        var centres2 = Enumerable.Range(0, k).Select(j => (z2[j] + z2[j + 1]) / 2 / range2).ToArray();
        var occupied = Enumerable.Range(0, counts.Length).Where(cell => counts[cell] > 0).ToArray();

        return Enumerable.Range(0, counts.Length).Select(cell => counts[cell] > 0 || occupied.Length == 0
            ? cell
            : occupied.OrderBy(other =>
                Math.Pow(centres1[cell / k] - centres1[other / k], 2) + Math.Pow(centres2[cell % k] - centres2[other % k], 2))
                .First()).ToArray();
    }

    /// <summary>
    /// Computes quantile-based bounds for a feature.
    /// </summary>
    private static Vector<T> QuantileBounds(Vector<T> featureValues, int numIntervals)
    {
        var engine = AiDotNetEngine.Current;
        int n = featureValues.Length;

        // Sorted values at the quantile positions round(i / numIntervals * (n - 1))
        var (sorted, _) = engine.TensorSort(featureValues.AsTensor());
        var positions = Enumerable.Range(0, numIntervals + 1)
            .Select(i => (int)Math.Round((double)i / numIntervals * (n - 1)))
            .ToArray();
        var bounds = engine.TensorIndexSelect(sorted, new Tensor<int>(positions, new[] { positions.Length }), 0);

        // Ensure unique bounds: a bound not above its predecessor moves 1e-10 past it, i.e.
        // bounds_i = offset_i + cummax(bounds - offset)_i with offset_i = i * 1e-10. (The scan differs from
        // a one-at-a-time nudge only for raw gaps already below 1e-10, by less than 1e-10.)
        var offsets = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, numIntervals + 1, 1), ExplainerTensorOps<T>.Tolerance(1e-10));
        var lifted = engine.TensorCumMax(engine.TensorSubtract(bounds, offsets), 0);
        return engine.TensorAdd(lifted, offsets).ToVector();
    }

    /// <summary>
    /// Makes a prediction with two modified feature values.
    /// </summary>
    private Vector<T> PredictWithModifiedFeatures(int feat1, Vector<T> values1, int feat2, Vector<T> values2)
    {
        var modified = WithColumn(_data, feat1, values1);
        AiDotNetEngine.Current.SetColumn(modified, feat2, values2);
        return _predictFunction(modified).GetSubVector(0, _data.Rows);
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
            var aleValues = new Vector<T>(ALEValues[featureIndex]).AsTensor();
            var engine = AiDotNetEngine.Current;
            T minAle = engine.TensorMinValue(aleValues);
            T maxAle = engine.TensorMaxValue(aleValues);
            T range = NumOps.Subtract(maxAle, minAle);

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
    /// Gets or sets the 2D ALE values on the grid of interval bounds [Bounds1 index, Bounds2 index], i.e.
    /// (K + 1) x (K + 1) for K intervals per feature, as in ALEPlot.
    /// </summary>
    public Matrix<T> ALEValues2D { get; set; } = new Matrix<T>(0, 0);
}
