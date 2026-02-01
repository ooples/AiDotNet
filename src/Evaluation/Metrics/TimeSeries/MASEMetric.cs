using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.TimeSeries;

/// <summary>
/// Computes Mean Absolute Scaled Error (MASE): scale-independent measure for time series.
/// </summary>
/// <remarks>
/// <para>MASE = MAE / MAE_naive where MAE_naive is the MAE of a naive seasonal forecast.</para>
/// <para><b>For Beginners:</b> MASE compares your model to a simple baseline (naive forecast).
/// <list type="bullet">
/// <item>MASE &lt; 1: Model beats the naive baseline</item>
/// <item>MASE = 1: Model is as good as naive</item>
/// <item>MASE &gt; 1: Model is worse than naive</item>
/// </list>
/// </para>
/// </remarks>
public class MASEMetric<T> : ITimeSeriesMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "MASE";
    public string Category => "TimeSeries";
    public string Description => "Mean Absolute Scaled Error - comparison to naive forecast.";
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

        int period = seasonalPeriod ?? 1;
        int n = predictions.Length;

        // Calculate MAE of predictions
        double mae = 0;
        for (int i = 0; i < n; i++)
            mae += Math.Abs(NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(predictions[i]));
        mae /= n;

        // Calculate MAE of naive forecast (y[t-period])
        double maeNaive = 0;
        int naiveCount = 0;
        for (int i = period; i < n; i++)
        {
            maeNaive += Math.Abs(NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(actuals[i - period]));
            naiveCount++;
        }

        if (naiveCount == 0 || maeNaive < 1e-10)
            return mae < 1e-10 ? NumOps.Zero : NumOps.FromDouble(double.MaxValue);

        maeNaive /= naiveCount;
        return NumOps.FromDouble(mae / maeNaive);
    }
}
