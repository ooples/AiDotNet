using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;

namespace AiDotNet.Evaluation.Metrics.Probabilistic;

/// <summary>
/// Energy Score metric for multivariate probabilistic predictions.
/// </summary>
/// <remarks>
/// <para>
/// The Energy Score is a multivariate generalization of CRPS. It measures the
/// quality of ensemble forecasts or probabilistic predictions in multiple dimensions.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you're predicting multiple related variables at once
/// (like temperature, humidity, and wind speed), Energy Score tells you how good
/// your combined prediction is. It considers both the accuracy and the correlations
/// between variables.
/// </para>
/// <para>
/// ES = E||X - y|| - 0.5 * E||X - X'||
/// where X, X' are independent samples from the predicted distribution
/// and y is the observation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EnergyScoreMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "EnergyScore";

    /// <inheritdoc/>
    public string Category => "Probabilistic";

    /// <inheritdoc/>
    public string Description => "Energy Score - multivariate probabilistic forecast metric.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.LowerIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => default;

    /// <inheritdoc/>
    public bool RequiresProbabilities => true;

    /// <inheritdoc/>
    public bool SupportsMultiClass => true;

    /// <summary>
    /// Computes the Energy Score for ensemble predictions.
    /// </summary>
    /// <param name="ensembleSamples">Array of ensemble members (shape: [numSamples, numDimensions]).</param>
    /// <param name="observation">The observed values (shape: [numDimensions]).</param>
    /// <returns>The Energy Score.</returns>
    public T ComputeFromEnsemble(T[][] ensembleSamples, T[] observation)
    {
        int m = ensembleSamples.Length;
        int d = observation.Length;

        if (m == 0)
            throw new ArgumentException("At least one ensemble member required.");

        // Validate dimensions
        foreach (var sample in ensembleSamples)
        {
            if (sample.Length != d)
                throw new ArgumentException("All ensemble samples must have the same dimension as observation.");
        }

        // E||X - y||
        double term1 = 0;
        for (int i = 0; i < m; i++)
        {
            term1 += EuclideanNorm(ensembleSamples[i], observation);
        }
        term1 /= m;

        // 0.5 * E||X - X'||
        double term2 = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                term2 += EuclideanNorm(ensembleSamples[i], ensembleSamples[j]);
            }
        }
        term2 = 0.5 * term2 / (m * m);

        return NumOps.FromDouble(term1 - term2);
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        // For univariate case, Energy Score reduces to MAE
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
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private double EuclideanNorm(T[] a, T[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]);
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
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
