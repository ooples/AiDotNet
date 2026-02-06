using AiDotNet.Distributions;
using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Scoring;

namespace AiDotNet.Evaluation.Metrics.Probabilistic;

/// <summary>
/// Logarithmic scoring metric for probabilistic predictions (Negative Log Likelihood).
/// </summary>
/// <remarks>
/// <para>
/// Computes the mean negative log likelihood across all predictions.
/// This is the most common metric for evaluating probabilistic forecasts.
/// </para>
/// <para>
/// <b>For Beginners:</b> This metric measures how well your predicted probability
/// distributions match reality. A lower score means better predictions.
/// It's particularly useful for models that output full distributions rather
/// than just point predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LogScoreMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "LogScore";

    /// <inheritdoc/>
    public string Category => "Probabilistic";

    /// <inheritdoc/>
    public string Description => "Mean Negative Log Likelihood - measures probabilistic prediction quality.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.LowerIsBetter;

    /// <inheritdoc/>
    public T? MinValue => default;

    /// <inheritdoc/>
    public T? MaxValue => default;

    /// <inheritdoc/>
    public bool RequiresProbabilities => true;

    /// <inheritdoc/>
    public bool SupportsMultiClass => true;

    /// <summary>
    /// Computes the mean log score for probabilistic predictions.
    /// </summary>
    /// <param name="distributions">The predicted distributions.</param>
    /// <param name="observations">The observed values.</param>
    /// <returns>The mean negative log likelihood.</returns>
    public T ComputeFromDistributions(IParametricDistribution<T>[] distributions, T[] observations)
    {
        if (distributions.Length != observations.Length)
            throw new ArgumentException("Distributions and observations must have the same length.");

        var scorer = new LogScore<T>();
        return scorer.MeanScore(distributions, observations);
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        // For point predictions, compute Gaussian negative log-likelihood:
        // NLL = 0.5 * log(2π * σ²) + (y - ŷ)² / (2σ²)
        // First estimate σ² from residuals, then compute mean NLL
        double sumSqErr = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double diff = NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(actuals[i]);
            sumSqErr += diff * diff;
        }

        double variance = sumSqErr / predictions.Length;
        if (variance < 1e-15) variance = 1e-15;

        // Mean NLL under Gaussian with estimated variance
        double meanNll = 0.5 * Math.Log(2 * Math.PI * variance) + 0.5;

        return NumOps.FromDouble(meanNll);
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
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
        var values = new double[samples];
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();

        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n];
            var sa = new T[n];
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sp[i] = predArr[idx];
                sa[i] = actArr[idx];
            }
            values[b] = NumOps.ToDouble(Compute(sp, sa));
        }

        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
