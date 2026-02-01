using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.TimeSeries;

/// <summary>
/// Computes Theil's U Statistic: measures forecast accuracy relative to a naive no-change forecast.
/// </summary>
/// <remarks>
/// <para>Theil's U = √[Σ(ŷ_t - y_t)² / Σ(y_t - y_{t-1})²]</para>
/// <para><b>For Beginners:</b> Theil's U compares your model to the simplest possible forecast (no change):
/// <list type="bullet">
/// <item>U = 0: Perfect predictions</item>
/// <item>U = 1: Model is as accurate as naive "no change" forecast</item>
/// <item>U &lt; 1: Model outperforms naive forecast</item>
/// <item>U &gt; 1: Model is worse than simply predicting no change</item>
/// </list>
/// Particularly useful for economic forecasting.</para>
/// </remarks>
public class TheilUMetric<T> : ITimeSeriesMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "TheilU";
    public string Category => "TimeSeries";
    public string Description => "Theil's U statistic comparing to naive forecast.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length < 2) return NumOps.Zero;

        int n = predictions.Length;

        // Numerator: Sum of squared forecast errors
        double sumSquaredError = 0;
        for (int i = 0; i < n; i++)
        {
            double error = NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(actuals[i]);
            sumSquaredError += error * error;
        }

        // Denominator: Sum of squared actual changes (naive no-change forecast error)
        double sumSquaredChange = 0;
        for (int i = 1; i < n; i++)
        {
            double change = NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(actuals[i - 1]);
            sumSquaredChange += change * change;
        }

        if (sumSquaredChange < 1e-10)
            return sumSquaredError < 1e-10 ? NumOps.Zero : NumOps.FromDouble(double.MaxValue);

        return NumOps.FromDouble(Math.Sqrt(sumSquaredError / sumSquaredChange));
    }
}
