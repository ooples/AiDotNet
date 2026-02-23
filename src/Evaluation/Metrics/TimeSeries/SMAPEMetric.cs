using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.TimeSeries;

/// <summary>
/// Computes Symmetric Mean Absolute Percentage Error (SMAPE): a bounded percentage error metric.
/// </summary>
/// <remarks>
/// <para>SMAPE = (100/N) * Σ |y - ŷ| / ((|y| + |ŷ|) / 2)</para>
/// <para><b>For Beginners:</b> SMAPE fixes some issues with MAPE:
/// <list type="bullet">
/// <item>Bounded between 0% and 200% (unlike MAPE which can be infinite)</item>
/// <item>Treats over-predictions and under-predictions more symmetrically</item>
/// <item>Still handles zero values better than MAPE</item>
/// </list>
/// Common in forecasting competitions (e.g., M-competitions).</para>
/// </remarks>
public class SMAPEMetric<T> : ITimeSeriesMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "SMAPE";
    public string Category => "TimeSeries";
    public string Description => "Symmetric Mean Absolute Percentage Error.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.FromDouble(200.0);
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double sum = 0;
        int validCount = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double actual = NumOps.ToDouble(actuals[i]);
            double pred = NumOps.ToDouble(predictions[i]);
            double denominator = (Math.Abs(actual) + Math.Abs(pred)) / 2.0;

            if (denominator > 1e-10)
            {
                sum += Math.Abs(actual - pred) / denominator;
                validCount++;
            }
        }

        return validCount > 0 ? NumOps.FromDouble(100.0 * sum / validCount) : NumOps.Zero;
    }
}
