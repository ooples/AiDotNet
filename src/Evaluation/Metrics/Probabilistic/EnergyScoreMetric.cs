using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

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

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var members = new Tensor<T>[m];
        for (int i = 0; i < m; i++)
        {
            members[i] = new Tensor<T>(ensembleSamples[i], new[] { d });
        }

        var ensemble = engine.TensorStack(members, 0);
        var obs = new Tensor<T>(observation, new[] { 1, d });

        // E||X - y||
        var toObservation = engine.TensorSubtract(ensemble, engine.TensorBroadcastTo(obs, new[] { m, d }));
        T term1 = engine.TensorMean(engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(toObservation), new[] { 1 }, false)));

        // 0.5 * E||X - X'|| over all ordered pairs (the distance matrix is symmetric with a zero diagonal)
        var shape = new[] { m, m, d };
        var left = engine.TensorBroadcastTo(engine.Reshape(ensemble, new[] { m, 1, d }), shape);
        var right = engine.TensorBroadcastTo(engine.Reshape(ensemble, new[] { 1, m, d }), shape);
        var pairDistances = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(left, right)), new[] { 2 }, false));
        T term2 = NumOps.Divide(
            NumOps.Multiply(NumOps.FromDouble(0.5), engine.TensorSum(pairDistances)),
            NumOps.FromDouble((double)m * m));

        return NumOps.Subtract(term1, term2);
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        // Fallback for point predictions: computes MAE. The true univariate
        // Energy Score reduces to CRPS and requires ensemble samples.
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        return engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(pred, actual)));
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

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual,
        ConfidenceIntervalMethod ciMethod, int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.Zero);

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();
        T thetaHat = Compute(predArr, actArr);

        // Generate bootstrap samples
        var bootstrapValues = new List<T>(samples);
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
            bootstrapValues.Add(Compute(sp, sa));
        }

        double alpha = 1 - conf;
        var bootstrapTensor = new Tensor<T>(bootstrapValues.ToArray(), new[] { samples });

        return ciMethod switch
        {
            ConfidenceIntervalMethod.PercentileBootstrap => PercentileInterval(bootstrapTensor, alpha),
            ConfidenceIntervalMethod.BasicBootstrap => BasicInterval(bootstrapTensor, thetaHat, alpha),
            ConfidenceIntervalMethod.BCaBootstrap => BCaInterval(bootstrapTensor, thetaHat, predArr, actArr, alpha),
            ConfidenceIntervalMethod.StudentizedBootstrap => StudentizedInterval(bootstrapTensor, thetaHat, predArr, actArr, random, alpha),
            _ => PercentileInterval(bootstrapTensor, alpha),
        };
    }

    private static (T, T) PercentileInterval(Tensor<T> bootstrapValues, double alpha)
    {
        int count = bootstrapValues.Length;
        int lo = Math.Max(0, (int)(alpha / 2 * count));
        int hi = Math.Min(count - 1, (int)((1 - alpha / 2) * count) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, count - 1);
        return EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, lo, hi);
    }

    private static (T, T) BasicInterval(Tensor<T> bootstrapValues, T thetaHat, double alpha)
    {
        int count = bootstrapValues.Length;
        int lo = Math.Max(0, (int)(alpha / 2 * count));
        int hi = Math.Min(count - 1, (int)((1 - alpha / 2) * count) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, count - 1);
        var (sortedLo, sortedHi) = EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, lo, hi);
        T twoTheta = NumOps.Multiply(NumOps.FromDouble(2.0), thetaHat);
        T lower = NumOps.Subtract(twoTheta, sortedHi);
        T upper = NumOps.Subtract(twoTheta, sortedLo);
        return (MaxWithZero(lower), upper);
    }

    // Math.Max(0, x): NaN stays NaN.
    private static T MaxWithZero(T value)
        => NumOps.IsNaN(value) || NumOps.GreaterThan(value, NumOps.Zero) ? value : NumOps.Zero;

    private (T, T) BCaInterval(Tensor<T> bootstrapValues, T thetaHat,
        T[] predArr, T[] actArr, double alpha)
    {
        int B = bootstrapValues.Length;
        int n = predArr.Length;
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();

        // Bias correction z0: proportion of bootstrap values < thetaHat
        int countBelow = EvaluationTensorOps<T>.Count(engine.TensorLessThan(bootstrapValues, thetaHat));
        double z0 = NormalQuantile(Math.Max(1e-10, Math.Min(1 - 1e-10, (double)countBelow / B)));

        // Acceleration via jackknife
        var jackValues = new List<T>(n);
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
            jackValues.Add(Compute(jp, ja));
        }

        var jack = new Tensor<T>(jackValues.ToArray(), new[] { n });
        var diff = engine.TensorNegate(engine.TensorSubtractScalar(jack, engine.TensorMean(jack)));
        T numSum = engine.TensorSum(engine.TensorMultiply(engine.TensorSquare(diff), diff));
        T denSum = engine.TensorSumOfSquares(diff);

        double acc = NumOps.GreaterThan(denSum, NumOps.FromDouble(1e-30))
            ? NumOps.ToDouble(NumOps.Divide(numSum, NumOps.Multiply(NumOps.FromDouble(6.0), NumOps.Power(denSum, NumOps.FromDouble(1.5)))))
            : 0;

        // Adjusted percentiles
        double zAlphaLo = NormalQuantile(alpha / 2);
        double zAlphaHi = NormalQuantile(1 - alpha / 2);

        double adjLo = z0 + (z0 + zAlphaLo) / (1 - acc * (z0 + zAlphaLo));
        double adjHi = z0 + (z0 + zAlphaHi) / (1 - acc * (z0 + zAlphaHi));

        double pLo = NormalCdf(adjLo);
        double pHi = NormalCdf(adjHi);

        int idxLo = Math.Max(0, Math.Min(B - 1, (int)(pLo * B)));
        int idxHi = Math.Max(0, Math.Min(B - 1, (int)(pHi * B)));
        idxHi = Math.Max(idxHi, idxLo + 1);
        idxHi = Math.Min(idxHi, B - 1);

        return EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, idxLo, idxHi);
    }

    private (T, T) StudentizedInterval(Tensor<T> bootstrapValues, T thetaHat,
        T[] predArr, T[] actArr, Random random, double alpha)
    {
        int B = bootstrapValues.Length;
        int n = predArr.Length;
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();

        // Estimate standard error of thetaHat via jackknife
        var jackValues = new List<T>(n);
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
            jackValues.Add(Compute(jp, ja));
        }

        var jack = new Tensor<T>(jackValues.ToArray(), new[] { n });
        T jackVar = engine.TensorSumOfSquares(engine.TensorSubtractScalar(jack, engine.TensorMean(jack)));
        T seHat = NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble((n - 1.0) / n), jackVar));
        T seFloor = NumOps.FromDouble(1e-30);
        if (NumOps.LessThan(seHat, seFloor)) seHat = seFloor;

        // For each bootstrap sample, compute t-statistic using inner bootstrap SE
        // The resamples draw in the same order as before; the statistics are reduced after the loops.
        int innerB = Math.Max(25, B / 10);
        var thetaStars = new List<T>(B);
        var innerValues = new List<T>(B * innerB);
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
            thetaStars.Add(Compute(sp, sa));


            // Estimate SE via inner bootstrap
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
                innerValues.Add(Compute(ip, ia));
            }
        }

        // seStar = sqrt(max(0, E[x^2] - E[x]^2)) over each replicate's inner values, floored at 1e-30
        var inner = engine.Reshape(new Tensor<T>(innerValues.ToArray(), new[] { B * innerB }), new[] { B, innerB });
        var innerMean = engine.ReduceMean(inner, new[] { 1 }, false);
        var innerMeanSquare = engine.ReduceMean(engine.TensorSquare(inner), new[] { 1 }, false);
        var innerVar = engine.TensorSubtract(innerMeanSquare, engine.TensorSquare(innerMean));
        var seStar = EvaluationTensorOps<T>.MaxPreservingNaN(
            engine.TensorSqrt(EvaluationTensorOps<T>.MaxPreservingNaN(innerVar, NumOps.Zero)), seFloor);

        var stars = new Tensor<T>(thetaStars.ToArray(), new[] { B });
        var tStats = engine.TensorDivide(engine.TensorSubtractScalar(stars, thetaHat), seStar);

        int lo = Math.Max(0, (int)(alpha / 2 * B));
        int hi = Math.Min(B - 1, (int)((1 - alpha / 2) * B) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, B - 1);
        var (tLo, tHi) = EvaluationTensorOps<T>.OrderStatistics(tStats, lo, hi);

        T lower = NumOps.Subtract(thetaHat, NumOps.Multiply(tHi, seHat));
        T upper = NumOps.Subtract(thetaHat, NumOps.Multiply(tLo, seHat));
        return (MaxWithZero(lower), upper);
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
