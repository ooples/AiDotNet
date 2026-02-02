using AiDotNet.Helpers;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Represents a SHAP explanation for a single prediction.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A SHAP explanation tells you exactly how much each feature
/// contributed to a specific prediction.
///
/// Key concepts:
/// - <b>Baseline Value</b>: The average prediction (what the model predicts "by default")
/// - <b>SHAP Values</b>: How much each feature pushed the prediction up or down from the baseline
/// - <b>Prediction</b>: Should equal Baseline + Sum(SHAP Values)
///
/// Example: If predicting house prices:
/// - Baseline: $300,000 (average house price)
/// - Bedrooms: +$50,000 (having 4 bedrooms adds value)
/// - Location: +$100,000 (good neighborhood)
/// - Age: -$30,000 (older house reduces value)
/// - Prediction: $420,000
/// </para>
/// </remarks>
public class SHAPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the SHAP values for each feature.
    /// </summary>
    public Vector<T> ShapValues { get; }

    /// <summary>
    /// Gets the baseline (expected) prediction value.
    /// </summary>
    public T BaselineValue { get; }

    /// <summary>
    /// Gets the actual prediction for this instance.
    /// </summary>
    public T Prediction { get; }

    /// <summary>
    /// Gets the feature names, if available.
    /// </summary>
    public string[]? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures => ShapValues.Length;

    /// <summary>
    /// Initializes a new SHAP explanation.
    /// </summary>
    public SHAPExplanation(
        Vector<T> shapValues,
        T baselineValue,
        T prediction,
        string[]? featureNames = null)
    {
        ShapValues = shapValues ?? throw new ArgumentNullException(nameof(shapValues));
        BaselineValue = baselineValue;
        Prediction = prediction;
        FeatureNames = featureNames;
    }

    /// <summary>
    /// Gets the SHAP value for a specific feature by index.
    /// </summary>
    public T GetShapValue(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
            throw new ArgumentOutOfRangeException(nameof(featureIndex));
        return ShapValues[featureIndex];
    }

    /// <summary>
    /// Gets the SHAP value for a specific feature by name.
    /// </summary>
    public T GetShapValue(string featureName)
    {
        if (FeatureNames is null)
            throw new InvalidOperationException("Feature names not available.");

        int index = Array.IndexOf(FeatureNames, featureName);
        if (index < 0)
            throw new ArgumentException($"Feature '{featureName}' not found.");

        return ShapValues[index];
    }

    /// <summary>
    /// Gets features sorted by absolute SHAP value (most important first).
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Value)> GetSortedFeatures()
    {
        return Enumerable.Range(0, NumFeatures)
            .Select(i => (
                Index: i,
                Name: FeatureNames?[i],
                Value: ShapValues[i]))
            .OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Value)));
    }

    /// <summary>
    /// Gets the top N most important features by absolute SHAP value.
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Value)> GetTopFeatures(int n)
    {
        return GetSortedFeatures().Take(n);
    }

    /// <summary>
    /// Verifies that SHAP values sum to the difference between prediction and baseline.
    /// </summary>
    /// <param name="tolerance">Acceptable deviation (default: 1e-6).</param>
    /// <returns>True if the SHAP values are consistent.</returns>
    public bool VerifyConsistency(double tolerance = 1e-6)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < NumFeatures; i++)
        {
            sum = NumOps.Add(sum, ShapValues[i]);
        }

        var expected = NumOps.Subtract(Prediction, BaselineValue);
        var diff = Math.Abs(NumOps.ToDouble(NumOps.Subtract(sum, expected)));
        return diff <= tolerance;
    }

    /// <summary>
    /// Returns a human-readable summary of the explanation.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopFeatures(5).ToList();
        var lines = new List<string>
        {
            $"SHAP Explanation:",
            $"  Baseline: {BaselineValue}",
            $"  Prediction: {Prediction}",
            $"  Top contributing features:"
        };

        foreach (var (index, name, value) in top)
        {
            var featureLabel = name ?? $"Feature {index}";
            var direction = NumOps.ToDouble(value) >= 0 ? "+" : "";
            lines.Add($"    {featureLabel}: {direction}{value}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Represents global SHAP explanations aggregated across multiple instances.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> While local SHAP explains one prediction, global SHAP
/// helps you understand which features are important across ALL predictions.
///
/// It aggregates individual explanations to show:
/// - Which features have the biggest impact on average
/// - How consistent that impact is across different predictions
/// </para>
/// </remarks>
public class GlobalSHAPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the individual explanations.
    /// </summary>
    public SHAPExplanation<T>[] LocalExplanations { get; }

    /// <summary>
    /// Gets the feature names, if available.
    /// </summary>
    public string[]? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures { get; }

    /// <summary>
    /// Initializes a global SHAP explanation from local explanations.
    /// </summary>
    public GlobalSHAPExplanation(SHAPExplanation<T>[] localExplanations, string[]? featureNames = null)
    {
        LocalExplanations = localExplanations ?? throw new ArgumentNullException(nameof(localExplanations));
        if (localExplanations.Length == 0)
            throw new ArgumentException("At least one local explanation is required.", nameof(localExplanations));

        NumFeatures = localExplanations[0].NumFeatures;
        FeatureNames = featureNames ?? localExplanations[0].FeatureNames;
    }

    /// <summary>
    /// Gets the mean absolute SHAP value for each feature (global feature importance).
    /// </summary>
    public Vector<T> GetMeanAbsoluteShapValues()
    {
        var result = new T[NumFeatures];

        for (int j = 0; j < NumFeatures; j++)
        {
            double sum = 0;
            for (int i = 0; i < LocalExplanations.Length; i++)
            {
                sum += Math.Abs(NumOps.ToDouble(LocalExplanations[i].ShapValues[j]));
            }
            result[j] = NumOps.FromDouble(sum / LocalExplanations.Length);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Gets feature importance ranking based on mean absolute SHAP values.
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Importance)> GetFeatureImportance()
    {
        var meanAbsShap = GetMeanAbsoluteShapValues();
        return Enumerable.Range(0, NumFeatures)
            .Select(i => (
                Index: i,
                Name: FeatureNames?[i],
                Importance: meanAbsShap[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Importance));
    }

    /// <summary>
    /// Gets the mean SHAP value for each feature (can be positive or negative).
    /// </summary>
    public Vector<T> GetMeanShapValues()
    {
        var result = new T[NumFeatures];

        for (int j = 0; j < NumFeatures; j++)
        {
            var sum = NumOps.Zero;
            for (int i = 0; i < LocalExplanations.Length; i++)
            {
                sum = NumOps.Add(sum, LocalExplanations[i].ShapValues[j]);
            }
            result[j] = NumOps.Divide(sum, NumOps.FromDouble(LocalExplanations.Length));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Gets the standard deviation of SHAP values for each feature.
    /// </summary>
    public Vector<T> GetShapValueStdDev()
    {
        var mean = GetMeanShapValues();
        var result = new T[NumFeatures];

        for (int j = 0; j < NumFeatures; j++)
        {
            double sumSq = 0;
            double meanVal = NumOps.ToDouble(mean[j]);
            for (int i = 0; i < LocalExplanations.Length; i++)
            {
                double diff = NumOps.ToDouble(LocalExplanations[i].ShapValues[j]) - meanVal;
                sumSq += diff * diff;
            }
            result[j] = NumOps.FromDouble(Math.Sqrt(sumSq / LocalExplanations.Length));
        }

        return new Vector<T>(result);
    }
}
