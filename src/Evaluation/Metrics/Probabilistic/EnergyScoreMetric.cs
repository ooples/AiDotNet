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
internal class EnergyScoreMetric<T> : IRegressionMetric<T>
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

        // 0.5 * E||X - X'|| — only compute j > i and multiply by 2
        double term2 = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = i + 1; j < m; j++)
            {
                term2 += 2 * EuclideanNorm(ensembleSamples[i], ensembleSamples[j]);
            }
        }
        term2 = 0.5 * term2 / (m * m);

        return NumOps.FromDouble(term1 - term2);
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        // Fallback for point predictions: computes MAE. The true univariate
        // Energy Score reduces to CRPS and requires ensemble samples.
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
                $"EnergyScoreMetric supports bootstrap CI methods (Percentile, Basic, BCa, Studentized). '{ciMethod}' is not supported.");
        }
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, ciMethod, bootstrapSamples, confidenceLevel, randomSeed);
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

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual,
        ConfidenceIntervalMethod ciMethod, int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.Zero);

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();
        double thetaHat = NumOps.ToDouble(Compute(predArr, actArr));

        // Generate bootstrap samples
        var bootstrapValues = new double[samples];
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
            bootstrapValues[b] = NumOps.ToDouble(Compute(sp, sa));
        }

        double alpha = 1 - conf;

        return ciMethod switch
        {
            ConfidenceIntervalMethod.PercentileBootstrap => PercentileInterval(bootstrapValues, alpha),
            ConfidenceIntervalMethod.BasicBootstrap => BasicInterval(bootstrapValues, thetaHat, alpha),
            ConfidenceIntervalMethod.BCaBootstrap => BCaInterval(bootstrapValues, thetaHat, predArr, actArr, alpha),
            ConfidenceIntervalMethod.StudentizedBootstrap => StudentizedInterval(bootstrapValues, thetaHat, predArr, actArr, random, alpha),
            _ => PercentileInterval(bootstrapValues, alpha),
        };
    }

    private static (T, T) PercentileInterval(double[] bootstrapValues, double alpha)
    {
        Array.Sort(bootstrapValues);
        int lo = Math.Max(0, (int)(alpha / 2 * bootstrapValues.Length));
        int hi = Math.Min(bootstrapValues.Length - 1, (int)((1 - alpha / 2) * bootstrapValues.Length) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, bootstrapValues.Length - 1);
        return (NumOps.FromDouble(bootstrapValues[lo]), NumOps.FromDouble(bootstrapValues[hi]));
    }

    private static (T, T) BasicInterval(double[] bootstrapValues, double thetaHat, double alpha)
    {
        Array.Sort(bootstrapValues);
        int lo = Math.Max(0, (int)(alpha / 2 * bootstrapValues.Length));
        int hi = Math.Min(bootstrapValues.Length - 1, (int)((1 - alpha / 2) * bootstrapValues.Length) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, bootstrapValues.Length - 1);
        double lower = 2 * thetaHat - bootstrapValues[hi];
        double upper = 2 * thetaHat - bootstrapValues[lo];
        return (NumOps.FromDouble(Math.Max(0, lower)), NumOps.FromDouble(upper));
    }

    private (T, T) BCaInterval(double[] bootstrapValues, double thetaHat,
        T[] predArr, T[] actArr, double alpha)
    {
        int B = bootstrapValues.Length;
        int n = predArr.Length;

        // Bias correction z0: proportion of bootstrap values < thetaHat
        int countBelow = 0;
        for (int i = 0; i < B; i++)
        {
            if (bootstrapValues[i] < thetaHat) countBelow++;
        }
        double z0 = NormalQuantile(Math.Max(1e-10, Math.Min(1 - 1e-10, (double)countBelow / B)));

        // Acceleration via jackknife
        var jackValues = new double[n];
        for (int i = 0; i < n; i++)
        {
            var jp = new T[n - 1];
            var ja = new T[n - 1];
            int idx = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                jp[idx] = predArr[j];
                ja[idx] = actArr[j];
                idx++;
            }
            jackValues[i] = NumOps.ToDouble(Compute(jp, ja));
        }

        double jackMean = 0;
        for (int i = 0; i < n; i++) jackMean += jackValues[i];
        jackMean /= n;

        double numSum = 0, denSum = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = jackMean - jackValues[i];
            numSum += diff * diff * diff;
            denSum += diff * diff;
        }

        double acc = denSum > 1e-30 ? numSum / (6 * Math.Pow(denSum, 1.5)) : 0;

        // Adjusted percentiles
        double zAlphaLo = NormalQuantile(alpha / 2);
        double zAlphaHi = NormalQuantile(1 - alpha / 2);

        double adjLo = z0 + (z0 + zAlphaLo) / (1 - acc * (z0 + zAlphaLo));
        double adjHi = z0 + (z0 + zAlphaHi) / (1 - acc * (z0 + zAlphaHi));

        double pLo = NormalCdf(adjLo);
        double pHi = NormalCdf(adjHi);

        Array.Sort(bootstrapValues);
        int idxLo = Math.Max(0, Math.Min(B - 1, (int)(pLo * B)));
        int idxHi = Math.Max(0, Math.Min(B - 1, (int)(pHi * B)));
        idxHi = Math.Max(idxHi, idxLo + 1);
        idxHi = Math.Min(idxHi, B - 1);

        return (NumOps.FromDouble(bootstrapValues[idxLo]), NumOps.FromDouble(bootstrapValues[idxHi]));
    }

    private (T, T) StudentizedInterval(double[] bootstrapValues, double thetaHat,
        T[] predArr, T[] actArr, Random random, double alpha)
    {
        int B = bootstrapValues.Length;
        int n = predArr.Length;

        // Estimate standard error of thetaHat via jackknife
        var jackValues = new double[n];
        for (int i = 0; i < n; i++)
        {
            var jp = new T[n - 1];
            var ja = new T[n - 1];
            int idx = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                jp[idx] = predArr[j];
                ja[idx] = actArr[j];
                idx++;
            }
            jackValues[i] = NumOps.ToDouble(Compute(jp, ja));
        }

        double jackMean = 0;
        for (int i = 0; i < n; i++) jackMean += jackValues[i];
        jackMean /= n;

        double jackVar = 0;
        for (int i = 0; i < n; i++)
        {
            double d = jackValues[i] - jackMean;
            jackVar += d * d;
        }
        double seHat = Math.Sqrt((n - 1.0) / n * jackVar);
        if (seHat < 1e-30) seHat = 1e-30;

        // For each bootstrap sample, compute t-statistic using inner bootstrap SE
        int innerB = Math.Max(25, B / 10);
        var tStats = new double[B];
        for (int b = 0; b < B; b++)
        {
            var sp = new T[n];
            var sa = new T[n];
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sp[i] = predArr[idx];
                sa[i] = actArr[idx];
            }
            double thetaStar = NumOps.ToDouble(Compute(sp, sa));

            // Estimate SE via inner bootstrap
            double innerSum = 0;
            double innerSumSq = 0;
            for (int ib = 0; ib < innerB; ib++)
            {
                var ip = new T[n];
                var ia = new T[n];
                for (int i = 0; i < n; i++)
                {
                    int idx = random.Next(n);
                    ip[i] = sp[idx];
                    ia[i] = sa[idx];
                }
                double innerVal = NumOps.ToDouble(Compute(ip, ia));
                innerSum += innerVal;
                innerSumSq += innerVal * innerVal;
            }

            double innerMean = innerSum / innerB;
            double innerVar = innerSumSq / innerB - innerMean * innerMean;
            double seStar = Math.Sqrt(Math.Max(0, innerVar));
            if (seStar < 1e-30) seStar = 1e-30;

            tStats[b] = (thetaStar - thetaHat) / seStar;
        }

        Array.Sort(tStats);
        int lo = Math.Max(0, (int)(alpha / 2 * B));
        int hi = Math.Min(B - 1, (int)((1 - alpha / 2) * B) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, B - 1);

        double lower = thetaHat - tStats[hi] * seHat;
        double upper = thetaHat - tStats[lo] * seHat;
        return (NumOps.FromDouble(Math.Max(0, lower)), NumOps.FromDouble(upper));
    }

    /// <summary>
    /// Standard normal quantile (inverse CDF) using rational approximation (Abramowitz and Stegun).
    /// </summary>
    private static double NormalQuantile(double p)
    {
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;
        if (Math.Abs(p - 0.5) < 1e-15) return 0.0;

        double t = p < 0.5 ? Math.Sqrt(-2.0 * Math.Log(p)) : Math.Sqrt(-2.0 * Math.Log(1.0 - p));
        double z = t - (2.515517 + t * (0.802853 + t * 0.010328))
                       / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
        return p < 0.5 ? -z : z;
    }

    /// <summary>
    /// Standard normal CDF approximation using Abramowitz and Stegun formula 7.1.26.
    /// </summary>
    private static double NormalCdf(double x)
    {
        if (x < -8) return 0;
        if (x > 8) return 1;

        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        double t = 1.0 / (1.0 + p * Math.Abs(x));
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x / 2);

        return x < 0 ? 1 - y : y;
    }
}
