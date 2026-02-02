using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Options;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for analyzing fairness and bias in machine learning models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Fairness analysis checks if your model treats different groups equally:
/// <list type="bullet">
/// <item>Does the model perform equally well for all demographic groups?</item>
/// <item>Are error rates similar across protected groups?</item>
/// <item>Are predictions independent of sensitive attributes?</item>
/// </list>
/// </para>
/// <para><b>Key fairness concepts:</b>
/// <list type="bullet">
/// <item><b>Demographic Parity:</b> Predictions should be independent of group membership</item>
/// <item><b>Equalized Odds:</b> TPR and FPR should be equal across groups</item>
/// <item><b>Equal Opportunity:</b> TPR should be equal across groups</item>
/// <item><b>Calibration:</b> Predicted probabilities should mean the same across groups</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FairnessEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly FairnessOptions? _options;

    /// <summary>
    /// Initializes the fairness engine.
    /// </summary>
    public FairnessEngine(FairnessOptions? options = null)
    {
        _options = options;
    }

    /// <summary>
    /// Analyzes fairness of model predictions across groups.
    /// </summary>
    /// <param name="predictions">Model predictions (binary or probability).</param>
    /// <param name="actuals">Actual labels (binary).</param>
    /// <param name="sensitiveAttribute">Group membership for each sample.</param>
    /// <returns>Fairness analysis results.</returns>
    public FairnessResult<T> Analyze(T[] predictions, T[] actuals, int[] sensitiveAttribute)
    {
        if (predictions.Length != actuals.Length || predictions.Length != sensitiveAttribute.Length)
            throw new ArgumentException("All arrays must have the same length.");

        var groups = sensitiveAttribute.Distinct().OrderBy(g => g).ToArray();
        var result = new FairnessResult<T>
        {
            Groups = groups,
            NumSamples = predictions.Length
        };

        // Compute per-group metrics
        foreach (var group in groups)
        {
            var groupIndices = Enumerable.Range(0, predictions.Length)
                .Where(i => sensitiveAttribute[i] == group)
                .ToArray();

            var groupPreds = groupIndices.Select(i => predictions[i]).ToArray();
            var groupActuals = groupIndices.Select(i => actuals[i]).ToArray();

            var metrics = ComputeGroupMetrics(groupPreds, groupActuals);
            result.GroupMetrics[group] = metrics;
        }

        // Compute fairness measures
        result.DemographicParityDifference = ComputeDemographicParityDifference(result.GroupMetrics);
        result.EqualizedOddsDifference = ComputeEqualizedOddsDifference(result.GroupMetrics);
        result.EqualOpportunityDifference = ComputeEqualOpportunityDifference(result.GroupMetrics);
        result.DisparateImpactRatio = ComputeDisparateImpactRatio(result.GroupMetrics);
        result.AverageOddsDifference = ComputeAverageOddsDifference(result.GroupMetrics);
        result.TheilIndex = ComputeTheilIndex(predictions, actuals, sensitiveAttribute);

        // Determine overall fairness
        double threshold = _options?.DisparityThreshold ?? 0.1;
        result.IsFair = Math.Abs(result.DemographicParityDifference) < threshold &&
                       Math.Abs(result.EqualizedOddsDifference) < threshold;

        return result;
    }

    private GroupFairnessMetrics ComputeGroupMetrics(T[] predictions, T[] actuals)
    {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        int positive = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            bool pred = NumOps.ToDouble(predictions[i]) >= 0.5;
            bool actual = NumOps.ToDouble(actuals[i]) >= 0.5;

            if (pred) positive++;
            if (pred && actual) tp++;
            else if (!pred && !actual) tn++;
            else if (pred && !actual) fp++;
            else fn++;
        }

        return new GroupFairnessMetrics
        {
            Size = predictions.Length,
            PositiveRate = predictions.Length > 0 ? (double)positive / predictions.Length : 0,
            TruePositiveRate = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0,
            FalsePositiveRate = (tn + fp) > 0 ? (double)fp / (tn + fp) : 0,
            TrueNegativeRate = (tn + fp) > 0 ? (double)tn / (tn + fp) : 0,
            FalseNegativeRate = (tp + fn) > 0 ? (double)fn / (tp + fn) : 0,
            Accuracy = predictions.Length > 0 ? (double)(tp + tn) / predictions.Length : 0,
            Precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0,
            Recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0
        };
    }

    private double ComputeDemographicParityDifference(Dictionary<int, GroupFairnessMetrics> groupMetrics)
    {
        if (groupMetrics.Count < 2) return 0;
        var rates = groupMetrics.Values.Select(m => m.PositiveRate).ToList();
        return rates.Max() - rates.Min();
    }

    private double ComputeEqualizedOddsDifference(Dictionary<int, GroupFairnessMetrics> groupMetrics)
    {
        if (groupMetrics.Count < 2) return 0;
        var tprDiff = groupMetrics.Values.Select(m => m.TruePositiveRate).Max() -
                      groupMetrics.Values.Select(m => m.TruePositiveRate).Min();
        var fprDiff = groupMetrics.Values.Select(m => m.FalsePositiveRate).Max() -
                      groupMetrics.Values.Select(m => m.FalsePositiveRate).Min();
        return Math.Max(tprDiff, fprDiff);
    }

    private double ComputeEqualOpportunityDifference(Dictionary<int, GroupFairnessMetrics> groupMetrics)
    {
        if (groupMetrics.Count < 2) return 0;
        var tprs = groupMetrics.Values.Select(m => m.TruePositiveRate).ToList();
        return tprs.Max() - tprs.Min();
    }

    private double ComputeDisparateImpactRatio(Dictionary<int, GroupFairnessMetrics> groupMetrics)
    {
        if (groupMetrics.Count < 2) return 1;
        var rates = groupMetrics.Values.Select(m => m.PositiveRate).ToList();
        double maxRate = rates.Max();
        double minRate = rates.Min();
        if (maxRate < 1e-10) return 1;
        return minRate / maxRate;
    }

    private double ComputeAverageOddsDifference(Dictionary<int, GroupFairnessMetrics> groupMetrics)
    {
        if (groupMetrics.Count < 2) return 0;
        var tprDiff = groupMetrics.Values.Select(m => m.TruePositiveRate).Max() -
                      groupMetrics.Values.Select(m => m.TruePositiveRate).Min();
        var fprDiff = groupMetrics.Values.Select(m => m.FalsePositiveRate).Max() -
                      groupMetrics.Values.Select(m => m.FalsePositiveRate).Min();
        return (tprDiff + fprDiff) / 2;
    }

    private double ComputeTheilIndex(T[] predictions, T[] actuals, int[] groups)
    {
        // Theil index measures inequality in benefit distribution
        int n = predictions.Length;
        if (n == 0) return 0;

        // Compute benefit (1 for correct prediction, 0 for incorrect)
        var benefits = new double[n];
        for (int i = 0; i < n; i++)
        {
            bool pred = NumOps.ToDouble(predictions[i]) >= 0.5;
            bool actual = NumOps.ToDouble(actuals[i]) >= 0.5;
            benefits[i] = pred == actual ? 1 : 0;
        }

        double meanBenefit = benefits.Average();
        if (meanBenefit < 1e-10) return 0;

        double theil = 0;
        for (int i = 0; i < n; i++)
        {
            if (benefits[i] > 0)
            {
                double ratio = benefits[i] / meanBenefit;
                theil += ratio * Math.Log(ratio);
            }
        }

        return theil / n;
    }
}

/// <summary>
/// Results from fairness analysis.
/// </summary>
public class FairnessResult<T>
{
    /// <summary>
    /// Array of group identifiers.
    /// </summary>
    public int[] Groups { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Total number of samples analyzed.
    /// </summary>
    public int NumSamples { get; init; }

    /// <summary>
    /// Per-group fairness metrics.
    /// </summary>
    public Dictionary<int, GroupFairnessMetrics> GroupMetrics { get; init; } = new();

    /// <summary>
    /// Demographic Parity Difference: max - min positive prediction rate across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Measures whether all groups receive positive predictions at similar rates.</para>
    /// <para>Ideal value: 0 (all groups have same positive rate)</para>
    /// </remarks>
    public double DemographicParityDifference { get; set; }

    /// <summary>
    /// Equalized Odds Difference: max of TPR and FPR differences across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Measures whether the model's accuracy is similar across groups.</para>
    /// <para>Ideal value: 0 (same TPR and FPR for all groups)</para>
    /// </remarks>
    public double EqualizedOddsDifference { get; set; }

    /// <summary>
    /// Equal Opportunity Difference: max - min TPR across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Measures whether qualified individuals are equally likely to be selected.</para>
    /// <para>Ideal value: 0 (same TPR for all groups)</para>
    /// </remarks>
    public double EqualOpportunityDifference { get; set; }

    /// <summary>
    /// Disparate Impact Ratio: min positive rate / max positive rate.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Common legal standard for fairness (80% rule).</para>
    /// <para>Ideal value: 1.0 (equal rates). Values below 0.8 often indicate discrimination.</para>
    /// </remarks>
    public double DisparateImpactRatio { get; set; }

    /// <summary>
    /// Average Odds Difference: mean of TPR and FPR differences.
    /// </summary>
    public double AverageOddsDifference { get; set; }

    /// <summary>
    /// Theil Index: measures inequality in model benefit distribution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Borrowed from economics, measures overall inequality.</para>
    /// <para>Ideal value: 0 (perfect equality)</para>
    /// </remarks>
    public double TheilIndex { get; set; }

    /// <summary>
    /// Whether the model is considered fair based on configured thresholds.
    /// </summary>
    public bool IsFair { get; set; }
}

/// <summary>
/// Fairness metrics for a specific group.
/// </summary>
public class GroupFairnessMetrics
{
    /// <summary>Number of samples in this group.</summary>
    public int Size { get; init; }
    /// <summary>Rate of positive predictions.</summary>
    public double PositiveRate { get; init; }
    /// <summary>True Positive Rate (Sensitivity).</summary>
    public double TruePositiveRate { get; init; }
    /// <summary>False Positive Rate.</summary>
    public double FalsePositiveRate { get; init; }
    /// <summary>True Negative Rate (Specificity).</summary>
    public double TrueNegativeRate { get; init; }
    /// <summary>False Negative Rate.</summary>
    public double FalseNegativeRate { get; init; }
    /// <summary>Accuracy for this group.</summary>
    public double Accuracy { get; init; }
    /// <summary>Precision for this group.</summary>
    public double Precision { get; init; }
    /// <summary>Recall for this group.</summary>
    public double Recall { get; init; }
}
