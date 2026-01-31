using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic Partial Dependence Plot (PDP) explainer with Individual Conditional Expectation (ICE) curves.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Partial Dependence Plots (PDPs) show how a feature affects predictions
/// on average, while holding all other features constant.
///
/// Imagine you want to know "How does Age affect loan approval probability?"
/// - PDP: Shows the average effect of Age across all applicants
/// - ICE: Shows individual curves for each applicant (revealing if the effect varies)
///
/// Key insights:
/// - Upward slope = feature increases predictions
/// - Downward slope = feature decreases predictions
/// - Flat line = feature has little effect
/// - If ICE curves are parallel = consistent effect for everyone
/// - If ICE curves cross = the effect depends on other features (interaction)
///
/// PDPs are great for understanding global feature effects, but can be misleading
/// when features are correlated. Use ALE (Accumulated Local Effects) for correlated features.
/// </para>
/// </remarks>
public class PartialDependenceExplainer<T> : IGlobalExplainer<T, PartialDependenceResult<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly Matrix<T> _backgroundData;
    private readonly int _gridResolution;
    private readonly bool _computeIce;
    private readonly string[]? _featureNames;

    /// <inheritdoc/>
    public string MethodName => "PartialDependence";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => false;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <summary>
    /// Initializes a new Partial Dependence explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="backgroundData">Representative data used for computing marginal effects.</param>
    /// <param name="gridResolution">Number of points in the feature grid (default: 20).</param>
    /// <param name="computeIce">Whether to compute Individual Conditional Expectation curves (default: true).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>backgroundData</b>: A sample of your data used to estimate average effects
    /// - <b>gridResolution</b>: How many points to evaluate (more = smoother curves but slower)
    /// - <b>computeIce</b>: Set to true to see individual variation, false for faster computation
    /// </para>
    /// </remarks>
    public PartialDependenceExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Matrix<T> backgroundData,
        int gridResolution = 20,
        bool computeIce = true,
        string[]? featureNames = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _backgroundData = backgroundData ?? throw new ArgumentNullException(nameof(backgroundData));

        if (backgroundData.Rows == 0)
            throw new ArgumentException("Background data must have at least one row.", nameof(backgroundData));
        if (gridResolution < 2)
            throw new ArgumentException("Grid resolution must be at least 2.", nameof(gridResolution));

        _gridResolution = gridResolution;
        _computeIce = computeIce;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Computes partial dependence for a single feature.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to analyze.</param>
    /// <returns>Partial dependence result for the feature.</returns>
    public PartialDependenceResult<T> ComputeForFeature(int featureIndex)
    {
        return ComputeForFeatures(new[] { featureIndex });
    }

    /// <summary>
    /// Computes partial dependence for multiple features.
    /// </summary>
    /// <param name="featureIndices">The indices of features to analyze.</param>
    /// <returns>Partial dependence result for all specified features.</returns>
    public PartialDependenceResult<T> ComputeForFeatures(int[] featureIndices)
    {
        if (featureIndices == null || featureIndices.Length == 0)
            throw new ArgumentException("At least one feature index is required.", nameof(featureIndices));

        int numFeatures = _backgroundData.Columns;
        foreach (var idx in featureIndices)
        {
            if (idx < 0 || idx >= numFeatures)
                throw new ArgumentOutOfRangeException(nameof(featureIndices), $"Feature index {idx} is out of range [0, {numFeatures}).");
        }

        var result = new PartialDependenceResult<T>
        {
            FeatureIndices = featureIndices,
            FeatureNames = featureIndices.Select(i => _featureNames?[i] ?? $"Feature {i}").ToArray(),
            GridValues = new Dictionary<int, T[]>(),
            PartialDependence = new Dictionary<int, T[]>(),
            IceCurves = _computeIce ? new Dictionary<int, T[,]>() : null
        };

        foreach (var featureIndex in featureIndices)
        {
            ComputeSingleFeaturePD(featureIndex, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public PartialDependenceResult<T> ExplainGlobal(Matrix<T> data)
    {
        // Compute PDP for all features
        var allFeatureIndices = Enumerable.Range(0, data.Columns).ToArray();
        return ComputeForFeatures(allFeatureIndices);
    }

    /// <summary>
    /// Computes partial dependence for a single feature.
    /// </summary>
    private void ComputeSingleFeaturePD(int featureIndex, PartialDependenceResult<T> result)
    {
        int n = _backgroundData.Rows;

        // Compute grid values (range of the feature)
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(_backgroundData[i, featureIndex]);
            minVal = Math.Min(minVal, val);
            maxVal = Math.Max(maxVal, val);
        }

        // Create grid
        var gridValues = new T[_gridResolution];
        double step = (maxVal - minVal) / (_gridResolution - 1);

        for (int g = 0; g < _gridResolution; g++)
        {
            gridValues[g] = NumOps.FromDouble(minVal + g * step);
        }

        result.GridValues[featureIndex] = gridValues;

        // Compute PDP values
        var pdValues = new T[_gridResolution];
        T[,]? iceValues = _computeIce ? new T[n, _gridResolution] : null;

        for (int g = 0; g < _gridResolution; g++)
        {
            // Create modified data with feature set to grid value
            var modifiedData = _backgroundData.Clone();
            for (int i = 0; i < n; i++)
            {
                modifiedData[i, featureIndex] = gridValues[g];
            }

            // Get predictions
            var predictions = _predictFunction(modifiedData);

            // Store ICE values and compute mean
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double pred = NumOps.ToDouble(predictions[i]);
                sum += pred;

                if (iceValues != null)
                    iceValues[i, g] = predictions[i];
            }

            pdValues[g] = NumOps.FromDouble(sum / n);
        }

        result.PartialDependence[featureIndex] = pdValues;

        if (iceValues != null)
            result.IceCurves![featureIndex] = iceValues;
    }

    /// <summary>
    /// Computes 2D partial dependence for feature interactions.
    /// </summary>
    /// <param name="feature1Index">Index of the first feature.</param>
    /// <param name="feature2Index">Index of the second feature.</param>
    /// <returns>2D partial dependence result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> 2D PDP shows how two features interact.
    /// The result is a heatmap where each cell shows the average prediction
    /// when both features are set to specific values.
    ///
    /// If the heatmap shows simple gradients, features act independently.
    /// If you see complex patterns, the features interact (their combined effect
    /// is different from their individual effects).
    /// </para>
    /// </remarks>
    public PartialDependence2DResult<T> ComputeInteraction(int feature1Index, int feature2Index)
    {
        int n = _backgroundData.Rows;
        int numFeatures = _backgroundData.Columns;

        if (feature1Index < 0 || feature1Index >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(feature1Index));
        if (feature2Index < 0 || feature2Index >= numFeatures)
            throw new ArgumentOutOfRangeException(nameof(feature2Index));

        // Compute grid for feature 1
        double min1 = double.MaxValue, max1 = double.MinValue;
        double min2 = double.MaxValue, max2 = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            double v1 = NumOps.ToDouble(_backgroundData[i, feature1Index]);
            double v2 = NumOps.ToDouble(_backgroundData[i, feature2Index]);
            min1 = Math.Min(min1, v1);
            max1 = Math.Max(max1, v1);
            min2 = Math.Min(min2, v2);
            max2 = Math.Max(max2, v2);
        }

        var grid1 = new T[_gridResolution];
        var grid2 = new T[_gridResolution];
        double step1 = (max1 - min1) / (_gridResolution - 1);
        double step2 = (max2 - min2) / (_gridResolution - 1);

        for (int g = 0; g < _gridResolution; g++)
        {
            grid1[g] = NumOps.FromDouble(min1 + g * step1);
            grid2[g] = NumOps.FromDouble(min2 + g * step2);
        }

        // Compute 2D PDP
        var pdValues = new T[_gridResolution, _gridResolution];

        for (int g1 = 0; g1 < _gridResolution; g1++)
        {
            for (int g2 = 0; g2 < _gridResolution; g2++)
            {
                var modifiedData = _backgroundData.Clone();
                for (int i = 0; i < n; i++)
                {
                    modifiedData[i, feature1Index] = grid1[g1];
                    modifiedData[i, feature2Index] = grid2[g2];
                }

                var predictions = _predictFunction(modifiedData);
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += NumOps.ToDouble(predictions[i]);

                pdValues[g1, g2] = NumOps.FromDouble(sum / n);
            }
        }

        return new PartialDependence2DResult<T>
        {
            Feature1Index = feature1Index,
            Feature2Index = feature2Index,
            Feature1Name = _featureNames?[feature1Index] ?? $"Feature {feature1Index}",
            Feature2Name = _featureNames?[feature2Index] ?? $"Feature {feature2Index}",
            Grid1Values = grid1,
            Grid2Values = grid2,
            PartialDependence2D = pdValues
        };
    }
}

/// <summary>
/// Represents the result of a partial dependence analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PartialDependenceResult<T>
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
    /// Gets or sets the grid values for each feature (feature index -> grid values).
    /// </summary>
    public Dictionary<int, T[]> GridValues { get; set; } = new();

    /// <summary>
    /// Gets or sets the partial dependence values for each feature (feature index -> PD values).
    /// </summary>
    public Dictionary<int, T[]> PartialDependence { get; set; } = new();

    /// <summary>
    /// Gets or sets the ICE curves for each feature (feature index -> [sample, grid point]).
    /// </summary>
    public Dictionary<int, T[,]>? IceCurves { get; set; }

    /// <summary>
    /// Gets the grid resolution.
    /// </summary>
    public int GridResolution => GridValues.Count > 0 ? GridValues.Values.First().Length : 0;

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string> { "Partial Dependence Analysis:" };

        foreach (var featureIndex in FeatureIndices)
        {
            var pdValues = PartialDependence[featureIndex];
            double minPd = pdValues.Min(v => NumOps.ToDouble(v));
            double maxPd = pdValues.Max(v => NumOps.ToDouble(v));
            double range = maxPd - minPd;

            string featureName = FeatureNames[Array.IndexOf(FeatureIndices, featureIndex)];
            lines.Add($"  {featureName}: PD range = {range:F4} (min={minPd:F4}, max={maxPd:F4})");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Represents the result of a 2D partial dependence analysis (feature interaction).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PartialDependence2DResult<T>
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
    /// Gets or sets the grid values for the first feature.
    /// </summary>
    public T[] Grid1Values { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the grid values for the second feature.
    /// </summary>
    public T[] Grid2Values { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the 2D partial dependence values [grid1, grid2].
    /// </summary>
    public T[,] PartialDependence2D { get; set; } = new T[0, 0];
}
