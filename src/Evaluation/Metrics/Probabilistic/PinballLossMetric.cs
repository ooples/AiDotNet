using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;

namespace AiDotNet.Evaluation.Metrics.Probabilistic;

/// <summary>
/// Pinball loss (quantile loss) metric for probabilistic predictions.
/// </summary>
/// <remarks>
/// <para>
/// Pinball loss is the standard loss function for quantile regression.
/// It asymmetrically penalizes over and under-predictions based on the quantile level.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you're predicting quantiles (like the median, or 90th percentile),
/// pinball loss tells you how good your predictions are. It penalizes under-predictions and
/// over-predictions differently based on which quantile you're targeting.
///
/// For the median (τ=0.5), under and over-predictions are penalized equally.
/// For the 90th percentile (τ=0.9), under-predictions are penalized more heavily.
/// </para>
/// <para>
/// Loss = τ * max(y - ŷ, 0) + (1 - τ) * max(ŷ - y, 0)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class PinballLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _tau;

    /// <summary>
    /// Initializes a new Pinball Loss metric for the specified quantile.
    /// </summary>
    /// <param name="tau">The quantile level (must be in (0, 1)). Default is 0.5 (median).</param>
    public PinballLossMetric(double tau = 0.5)
    {
        if (tau <= 0 || tau >= 1)
            throw new ArgumentOutOfRangeException(nameof(tau), "Quantile must be in (0, 1).");

        _tau = tau;
    }

    /// <inheritdoc/>
    public string Name => $"PinballLoss_{_tau:F2}";

    /// <inheritdoc/>
    public string Category => "Probabilistic";

    /// <inheritdoc/>
    public string Description => $"Quantile loss for τ={_tau:F2} - measures accuracy of quantile predictions.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.LowerIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => default;

    /// <inheritdoc/>
    public bool RequiresProbabilities => false;

    /// <inheritdoc/>
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Gets the quantile level.
    /// </summary>
    public double Tau => _tau;

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double sum = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = NumOps.ToDouble(predictions[i]);
            double actual = NumOps.ToDouble(actuals[i]);
            double diff = actual - pred;

            double loss = diff >= 0
                ? _tau * diff
                : (1 - _tau) * (-diff);

            sum += loss;
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
                $"PinballLossMetric supports bootstrap CI methods (Percentile, Basic, BCa, Studentized). '{ciMethod}' is not supported.");
        }
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, ciMethod, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
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

        // For each bootstrap sample, compute t-statistic using jackknife SE
        int innerB = Math.Max(25, B / 10);
        var tStats = new double[B];
        for (int b = 0; b < B; b++)
        {
            // Resample for this bootstrap replicate
            var sp = new T[n];
            var sa = new T[n];
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sp[i] = predArr[idx];
                sa[i] = actArr[idx];
            }
            double thetaStar = NumOps.ToDouble(Compute(sp, sa));

            // Estimate SE of this bootstrap sample via inner bootstrap
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
