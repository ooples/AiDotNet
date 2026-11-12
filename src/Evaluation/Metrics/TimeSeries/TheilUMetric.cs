using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "TheilU";
    /// <inheritdoc />
    public string Category => "TimeSeries";
    /// <inheritdoc />
    public string Description => "Theil's U statistic comparing to naive forecast.";
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
        if (predictions.Length < 2) return NumOps.Zero;

        int n = predictions.Length;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var tolerance = NumOps.FromDouble(1e-10);

        // Numerator: Sum of squared forecast errors
        T sumSquaredError = engine.TensorSumOfSquares(engine.TensorSubtract(pred, actual));

        // Denominator: Sum of squared actual changes (naive no-change forecast error)
        T sumSquaredChange = engine.TensorSumOfSquares(engine.TensorDiff(actual, 1, 0));

        if (NumOps.LessThan(sumSquaredChange, tolerance))
            return NumOps.LessThan(sumSquaredError, tolerance) ? NumOps.Zero : NumOps.MaxValue;

        return NumOps.Sqrt(NumOps.Divide(sumSquaredError, sumSquaredChange));
    }
}
