using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.TimeSeries;

/// <summary>
/// Computes Weighted Absolute Percentage Error (WAPE): total absolute error as percentage of total actuals.
/// </summary>
/// <remarks>
/// <para>WAPE = Σ|y - ŷ| / Σ|y|</para>
/// <para><b>For Beginners:</b> WAPE gives a single number measuring overall forecast accuracy:
/// <list type="bullet">
/// <item>More robust than MAPE when dealing with intermittent demand (zeros)</item>
/// <item>Weights errors by the magnitude of actuals</item>
/// <item>WAPE = 0.1 means your total errors are 10% of total actual values</item>
/// </list>
/// Also known as MAD/Mean ratio in some contexts.</para>
/// </remarks>
public class WAPEMetric<T> : ITimeSeriesMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "WAPE";
    /// <inheritdoc />
    public string Category => "TimeSeries";
    /// <inheritdoc />
    public string Description => "Weighted Absolute Percentage Error.";
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

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var tolerance = NumOps.FromDouble(1e-10);

        T totalAbsError = engine.TensorSum(engine.TensorAbs(engine.TensorSubtract(actual, pred)));
        T totalAbsActual = engine.TensorSum(engine.TensorAbs(actual));

        if (NumOps.LessThan(totalAbsActual, tolerance)) return NumOps.Zero;
        return NumOps.Divide(totalAbsError, totalAbsActual);
    }
}
