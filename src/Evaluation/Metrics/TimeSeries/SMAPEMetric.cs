using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "SMAPE";
    /// <inheritdoc />
    public string Category => "TimeSeries";
    /// <inheritdoc />
    public string Description => "Symmetric Mean Absolute Percentage Error.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => NumOps.FromDouble(200.0);
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

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var tolerance = NumOps.FromDouble(1e-10);

        // |actual - pred| / ((|actual| + |pred|) / 2) over the samples whose denominator exceeds 1e-10
        var denominator = engine.TensorMultiplyScalar(
            engine.TensorAdd(engine.TensorAbs(actual), engine.TensorAbs(pred)), NumOps.FromDouble(0.5));
        var valid = engine.TensorGreaterThan(denominator, tolerance);
        int validCount = EvaluationTensorOps<T>.Count(valid);
        if (validCount == 0) return NumOps.Zero;

        var relative = engine.TensorDivide(engine.TensorAbs(engine.TensorSubtract(actual, pred)), denominator);
        T sum = engine.TensorSum(engine.TensorWhere(valid, relative, engine.TensorZerosLike(relative)));
        return NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(100.0), sum), NumOps.FromDouble(validCount));
    }
}
