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

        if (d == 0) return NumOps.Zero;
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var members = new Tensor<double>[m];
        for (int i = 0; i < m; i++)
            members[i] = new Tensor<T>(ensembleSamples[i], [d]).Cast<double>();
        var ensemble = engine.TensorStack(members, 0);
        var obs = new Tensor<T>(observation, [1, d]).Cast<double>();

        // Bound temporary feature grids to 2 MiB, or one row when the feature vector is wider.
        int blockSize = Math.Max(1, 262144 / d);
        double observationDistances = 0, pairDistances = 0;
        for (int start = 0; start < m;)
        {
            int count = Math.Min(blockSize, m - start);
            var block = engine.TensorSlice(ensemble, [start, 0], [count, d]);
            var difference = engine.TensorSubtract(block, engine.TensorBroadcastTo(obs, [count, d]));
            observationDistances += engine.TensorSum(engine.TensorSqrt(
                engine.ReduceSum(engine.TensorSquare(difference), [1], false)));
            start += count;
        }
        for (int i = 0; i < m; i++)
        {
            var left = engine.TensorSlice(ensemble, [i, 0], [1, d]);
            for (int start = i + 1; start < m;)
            {
                int count = Math.Min(blockSize, m - start);
                var right = engine.TensorSlice(ensemble, [start, 0], [count, d]);
                var difference = engine.TensorSubtract(right, engine.TensorBroadcastTo(left, [count, d]));
                pairDistances += engine.TensorSum(engine.TensorSqrt(
                    engine.ReduceSum(engine.TensorSquare(difference), [1], false)));
                start += count;
            }
        }
        // Each unordered pair represents both ordered pairs in 0.5 E||X-X'||.
        return NumOps.FromDouble(observationDistances / m - pairDistances / ((double)m * m));
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
        var pred = EvaluationTensorOps<T>.ToTensor(predictions).Cast<double>();
        var actual = EvaluationTensorOps<T>.ToTensor(actuals).Cast<double>();
        return NumOps.FromDouble(engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(pred, actual))));
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
        if (!(confidenceLevel > 0 && confidenceLevel < 1))
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, ciMethod, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual,
        ConfidenceIntervalMethod ciMethod, int samples, double conf, int? seed)
    {
        if (pred.Length == 0) return (NumOps.Zero, NumOps.Zero);
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        using var noDoubleGrad = new NoGradScope<double>();
        var errors = engine.TensorAbs(engine.TensorSubtract(
            EvaluationTensorOps<T>.ToTensor(pred).Cast<double>(), EvaluationTensorOps<T>.ToTensor(actual).Cast<double>()));
        double theta = engine.TensorMean(errors);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var bootstrapValues = new double[samples];
        for (int b = 0; b < samples; b++)
            bootstrapValues[b] = engine.TensorMean(Resample(errors, random));
        var bootstrap = new Tensor<double>(bootstrapValues, [samples]);
        double alpha = 1 - conf;
        var (lower, upper) = ciMethod switch
        {
            ConfidenceIntervalMethod.PercentileBootstrap => PercentileInterval(bootstrap, alpha),
            ConfidenceIntervalMethod.BasicBootstrap => BasicInterval(bootstrap, theta, alpha),
            ConfidenceIntervalMethod.BCaBootstrap => BCaInterval(bootstrap, theta, errors, alpha),
            ConfidenceIntervalMethod.StudentizedBootstrap => StudentizedInterval(bootstrap, theta, errors, random, alpha),
            _ => throw new NotSupportedException()
        };
        return (NumOps.FromDouble(lower), NumOps.FromDouble(upper));
    }

    private static Tensor<double> Resample(Tensor<double> values, Random random)
    {
        var indices = new int[values.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = random.Next(values.Length);
        return AiDotNetEngine.Current.TensorIndexSelect(values, new Tensor<int>(indices, [indices.Length]), 0);
    }

    private static (double, double) PercentileInterval(Tensor<double> values, double alpha)
    {
        int lo = Math.Max(0, (int)(alpha / 2 * values.Length));
        int hi = Math.Min(values.Length - 1, (int)((1 - alpha / 2) * values.Length) - 1);
        hi = Math.Min(Math.Max(hi, lo + 1), values.Length - 1);
        return EvaluationTensorOps<double>.OrderStatistics(values, lo, hi);
    }

    private static (double, double) BasicInterval(Tensor<double> values, double theta, double alpha)
    {
        var (lo, hi) = PercentileInterval(values, alpha);
        return (Math.Max(0, 2 * theta - hi), Math.Max(0, 2 * theta - lo));
    }

    private static Tensor<double> Jackknife(Tensor<double> errors)
    {
        int n = errors.Length;
        if (n < 2) return new Tensor<double>([n]);
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivideScalar(engine.TensorSubtract(
            Tensor<double>.CreateDefault([n], engine.TensorSum(errors)), errors), n - 1.0);
    }

    private static (double, double) BCaInterval(Tensor<double> values, double theta, Tensor<double> errors, double alpha)
    {
        var engine = AiDotNetEngine.Current;
        int countBelow = EvaluationTensorOps<double>.Count(engine.TensorLessThan(values, theta));
        double z0 = NormalQuantile(Math.Max(1e-10, Math.Min(1 - 1e-10, (double)countBelow / values.Length)));
        var jack = Jackknife(errors);
        var difference = engine.TensorNegate(engine.TensorSubtractScalar(jack, engine.TensorMean(jack)));
        double scale = engine.TensorMaxValue(engine.TensorAbs(difference));
        double acceleration = 0;
        if (scale > 0 && !double.IsInfinity(scale))
        {
            var normalized = engine.TensorDivideScalar(difference, scale);
            double numerator = engine.TensorSum(engine.TensorMultiply(engine.TensorSquare(normalized), normalized));
            double denominator = engine.TensorSumOfSquares(normalized);
            acceleration = numerator / (6 * Math.Pow(denominator, 1.5));
        }
        double zLo = NormalQuantile(alpha / 2), zHi = NormalQuantile(1 - alpha / 2);
        double pLo = NormalCdf(z0 + (z0 + zLo) / (1 - acceleration * (z0 + zLo)));
        double pHi = NormalCdf(z0 + (z0 + zHi) / (1 - acceleration * (z0 + zHi)));
        int lo = Math.Max(0, Math.Min(values.Length - 1, (int)(pLo * values.Length)));
        int hi = Math.Max(0, Math.Min(values.Length - 1, (int)(pHi * values.Length)));
        hi = Math.Min(Math.Max(hi, lo + 1), values.Length - 1);
        return EvaluationTensorOps<double>.OrderStatistics(values, lo, hi);
    }

    private static (double, double) StudentizedInterval(Tensor<double> bootstrapValues, double theta,
        Tensor<double> errors, Random random, double alpha)
    {
        var engine = AiDotNetEngine.Current;
        int count = bootstrapValues.Length, n = errors.Length;
        var jack = Jackknife(errors);
        double jackVariance = engine.TensorSumOfSquares(engine.TensorSubtractScalar(jack, engine.TensorMean(jack)));
        double seHat = Math.Max(1e-30, Math.Sqrt((n - 1.0) / n * jackVariance));
        int innerCount = Math.Max(25, count / 10);
        var tStatistics = new double[count];
        for (int b = 0; b < count; b++)
        {
            var outer = Resample(errors, random);
            double star = engine.TensorMean(outer);
            var innerValues = new double[innerCount];
            for (int ib = 0; ib < innerCount; ib++)
                innerValues[ib] = engine.TensorMean(Resample(outer, random));
            var inner = new Tensor<double>(innerValues, [innerCount]);
            var centered = engine.TensorSubtractScalar(inner, engine.TensorMean(inner));
            double seStar = Math.Max(1e-30, Math.Sqrt(engine.TensorMean(engine.TensorSquare(centered))));
            tStatistics[b] = (star - theta) / seStar;
        }
        var (lo, hi) = PercentileInterval(new Tensor<double>(tStatistics, [count]), alpha);
        return (Math.Max(0, theta - hi * seHat), Math.Max(0, theta - lo * seHat));
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
