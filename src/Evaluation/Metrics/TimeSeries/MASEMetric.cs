using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "MASE";
    /// <inheritdoc />
    public string Category => "TimeSeries";
    /// <inheritdoc />
    public string Description => "Mean Absolute Scaled Error - comparison to naive forecast.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => default;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => false;

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int period = seasonalPeriod ?? 1;
        if (period < 1)
            throw new ArgumentException("Seasonal period must be at least 1.", nameof(seasonalPeriod));
        int n = predictions.Length;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions).Cast<double>();
        var actual = EvaluationTensorOps<T>.ToTensor(actuals).Cast<double>();
        double mae = engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(actual, pred)));
        int naiveCount = Math.Max(0, n - period);
        double maeNaive = 0;
        if (naiveCount > 0)
        {
            var current = engine.TensorSlice(actual, new[] { period }, new[] { naiveCount });
            var lagged = engine.TensorSlice(actual, new[] { 0 }, new[] { naiveCount });
            maeNaive = engine.TensorSum(engine.TensorAbs(engine.TensorSubtract(current, lagged)));
        }
        if (naiveCount == 0 || maeNaive < 1e-10)
            return mae < 1e-10 ? NumOps.Zero : NumOps.MaxValue;
        return NumOps.FromDouble(mae / (maeNaive / naiveCount));
    }
}
