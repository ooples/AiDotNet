using AiDotNet.Distributions;
using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Scoring;

namespace AiDotNet.Evaluation.Metrics.Probabilistic;

/// <summary>
/// Continuous Ranked Probability Score metric for probabilistic predictions.
/// </summary>
/// <remarks>
/// <para>
/// CRPS is a proper scoring rule that measures the quality of probabilistic predictions.
/// It generalizes mean absolute error to full probability distributions.
/// </para>
/// <para>
/// <b>For Beginners:</b> CRPS tells you how good your probability forecasts are.
/// Unlike log score, CRPS:
/// - Has the same units as your predicted variable
/// - For point predictions, equals mean absolute error
/// - Rewards both accuracy and appropriate uncertainty
///
/// Lower CRPS is better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CRPSMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "CRPS";

    /// <inheritdoc/>
    public string Category => "Probabilistic";

    /// <inheritdoc/>
    public string Description => "Continuous Ranked Probability Score - measures probabilistic forecast quality.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.LowerIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => default;

    /// <inheritdoc/>
    public bool RequiresProbabilities => true;

    /// <inheritdoc/>
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Computes the mean CRPS for probabilistic predictions.
    /// </summary>
    /// <param name="distributions">The predicted distributions.</param>
    /// <param name="observations">The observed values.</param>
    /// <returns>The mean CRPS.</returns>
    public T ComputeFromDistributions(IParametricDistribution<T>[] distributions, T[] observations)
    {
        if (distributions.Length != observations.Length)
            throw new ArgumentException("Distributions and observations must have the same length.");

        var scorer = new CRPSScore<T>();
        return scorer.MeanScore(distributions, observations);
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        // For point predictions, CRPS equals MAE
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double sum = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double diff = NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(actuals[i]);
            sum += Math.Abs(diff);
        }

        return NumOps.FromDouble(sum / predictions.Length);
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
        if (ciMethod != ConfidenceIntervalMethod.PercentileBootstrap &&
            ciMethod != ConfidenceIntervalMethod.BasicBootstrap &&
            ciMethod != ConfidenceIntervalMethod.BCaBootstrap &&
            ciMethod != ConfidenceIntervalMethod.StudentizedBootstrap)
        {
            throw new NotSupportedException(
                $"CRPSMetric supports bootstrap CI methods (Percentile, Basic, BCa, Studentized). '{ciMethod}' is not supported.");
        }
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
        var sp = new T[n];
        var sa = new T[n];

        for (int b = 0; b < samples; b++)
        {
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
