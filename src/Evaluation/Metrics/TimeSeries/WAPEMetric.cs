using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.TimeSeries;

/// <summary>
/// Computes Weighted Absolute Percentage Error (WAPE): total absolute error as percentage of total actuals.
/// </summary>
/// <remarks>
/// <para>WAPE = 100 * Σ|y - ŷ| / Σ|y|</para>
/// <para><b>For Beginners:</b> WAPE gives a single percentage measuring overall forecast accuracy:
/// <list type="bullet">
/// <item>More robust than MAPE when dealing with intermittent demand (zeros)</item>
/// <item>Weights errors by the magnitude of actuals</item>
/// <item>WAPE = 10% means your total errors are 10% of total actual values</item>
/// </list>
/// Also known as MAD/Mean ratio in some contexts.</para>
/// </remarks>
public class WAPEMetric<T> : ITimeSeriesMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "WAPE";
    public string Category => "TimeSeries";
    public string Description => "Weighted Absolute Percentage Error.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double totalAbsError = 0;
        double totalAbsActual = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double actual = NumOps.ToDouble(actuals[i]);
            double pred = NumOps.ToDouble(predictions[i]);
            totalAbsError += Math.Abs(actual - pred);
            totalAbsActual += Math.Abs(actual);
        }

        if (totalAbsActual < 1e-10) return NumOps.Zero;
        return NumOps.FromDouble(100.0 * totalAbsError / totalAbsActual);
    }
}
