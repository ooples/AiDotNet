using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.TimeSeries;

/// <summary>
/// Computes Seasonal MASE: Mean Absolute Scaled Error with explicit seasonal comparison.
/// </summary>
/// <remarks>
/// <para>Seasonal MASE = MAE / MAE_seasonal_naive where seasonal naive predicts y[t-m] (m = seasonal period)</para>
/// <para><b>For Beginners:</b> This is MASE specifically designed for seasonal data:
/// <list type="bullet">
/// <item>Compares your model to a naive seasonal forecast (same period last year/month/week)</item>
/// <item>MASE &lt; 1: Model beats seasonal baseline</item>
/// <item>Seasonal period could be 7 (daily with weekly pattern), 12 (monthly with yearly pattern), etc.</item>
/// </list>
/// Essential for evaluating retail, energy, or any data with seasonal patterns.</para>
/// </remarks>
public class MASESeasonalMetric<T> : ITimeSeriesMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _defaultSeasonalPeriod;

    public string Name => "MASESeasonal";
    public string Category => "TimeSeries";
    public string Description => "MASE compared to seasonal naive forecast.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Initializes the Seasonal MASE metric.
    /// </summary>
    /// <param name="defaultSeasonalPeriod">Default seasonal period if not specified at compute time. Default is 12 (monthly data).</param>
    public MASESeasonalMetric(int defaultSeasonalPeriod = 12)
    {
        _defaultSeasonalPeriod = Math.Max(1, defaultSeasonalPeriod);
    }

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int period = seasonalPeriod ?? _defaultSeasonalPeriod;
        int n = predictions.Length;

        if (n <= period) return NumOps.Zero; // Not enough data for seasonal comparison

        // Calculate MAE of predictions
        double mae = 0;
        for (int i = 0; i < n; i++)
            mae += Math.Abs(NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(predictions[i]));
        mae /= n;

        // Calculate MAE of seasonal naive forecast (y[t-period])
        double maeSeasonalNaive = 0;
        int naiveCount = 0;
        for (int i = period; i < n; i++)
        {
            maeSeasonalNaive += Math.Abs(NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(actuals[i - period]));
            naiveCount++;
        }

        if (naiveCount == 0 || maeSeasonalNaive < 1e-10)
            return mae < 1e-10 ? NumOps.Zero : NumOps.FromDouble(double.MaxValue);

        maeSeasonalNaive /= naiveCount;
        return NumOps.FromDouble(mae / maeSeasonalNaive);
    }
}
