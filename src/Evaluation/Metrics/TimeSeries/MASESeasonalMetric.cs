using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "MASESeasonal";
    /// <inheritdoc />
    public string Category => "TimeSeries";
    /// <inheritdoc />
    public string Description => "MASE compared to seasonal naive forecast.";
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

    /// <summary>
    /// Initializes the Seasonal MASE metric.
    /// </summary>
    /// <param name="defaultSeasonalPeriod">Default seasonal period if not specified at compute time. Default is 12 (monthly data).</param>
    public MASESeasonalMetric(int defaultSeasonalPeriod = 12)
    {
        _defaultSeasonalPeriod = Math.Max(1, defaultSeasonalPeriod);
    }

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int period = seasonalPeriod ?? _defaultSeasonalPeriod;
        int n = predictions.Length;

        if (n <= period) return NumOps.Zero; // Not enough data for seasonal comparison

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var tolerance = NumOps.FromDouble(1e-10);

        // Calculate MAE of predictions
        T mae = engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(actual, pred)));

        // Calculate MAE of the naive forecast y[t - period] over t = period .. n - 1
        int naiveCount = Math.Max(0, n - period);
        T maeSeasonalNaive = NumOps.Zero;
        if (naiveCount > 0)
        {
            var current = engine.TensorSlice(actual, new[] { period }, new[] { naiveCount });
            var lagged = engine.TensorSlice(actual, new[] { 0 }, new[] { naiveCount });
            maeSeasonalNaive = engine.TensorSum(engine.TensorAbs(engine.TensorSubtract(current, lagged)));
        }

        if (naiveCount == 0 || NumOps.LessThan(maeSeasonalNaive, tolerance))
            return NumOps.LessThan(mae, tolerance) ? NumOps.Zero : NumOps.MaxValue;

        return NumOps.Divide(mae, NumOps.Divide(maeSeasonalNaive, NumOps.FromDouble(naiveCount)));
    }
}
