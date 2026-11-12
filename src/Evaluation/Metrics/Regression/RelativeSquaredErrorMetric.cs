using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Relative Squared Error (RSE): sum of squared errors relative to baseline model.
/// </summary>
/// <remarks>
/// <para>RSE = Σ(y - ŷ)² / Σ(y - ȳ)²</para>
/// <para><b>For Beginners:</b> RSE compares your model to a baseline (mean predictor):
/// <list type="bullet">
/// <item>RSE &lt; 1: Your model is better than the mean baseline</item>
/// <item>RSE = 1: Your model is equivalent to predicting the mean</item>
/// <item>RSE &gt; 1: Your model is worse than the mean baseline</item>
/// <item>RSE = 1 - R² (when calculated correctly)</item>
/// </list>
/// </para>
/// </remarks>
public class RelativeSquaredErrorMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "RSE";
    /// <inheritdoc />
    public string Category => "Regression";
    /// <inheritdoc />
    public string Description => "Relative Squared Error (compared to mean predictor).";
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
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions).Cast<double>();
        var actual = EvaluationTensorOps<T>.ToTensor(actuals).Cast<double>();
        double mean = engine.TensorMean(actual);
        double error = engine.TensorSumOfSquares(engine.TensorSubtract(actual, pred));
        double total = engine.TensorSumOfSquares(engine.TensorSubtractScalar(actual, mean));
        return total < 1e-10 ? NumOps.Zero : NumOps.FromDouble(error / total);
    }

    /// <inheritdoc />
    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual, int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.Zero);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var values = new List<T>(samples);
        var predArr = pred.ToArray(); var actArr = actual.ToArray();
        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = predArr[idx]; sa[i] = actArr[idx]; }
            values.Add(Compute(sp, sa));
        }
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return EvaluationTensorOps<T>.OrderStatistics(values, lo, hi);
    }
}
