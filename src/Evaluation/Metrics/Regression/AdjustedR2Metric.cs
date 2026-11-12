using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Adjusted R² Score: R² adjusted for the number of predictors in the model.
/// </summary>
/// <remarks>
/// <para>Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)</para>
/// <para><b>For Beginners:</b> Adjusted R² penalizes adding unnecessary predictors to a model.
/// Unlike regular R² which always increases (or stays the same) when adding features,
/// adjusted R² will decrease if the new feature doesn't improve the model enough.
/// <list type="bullet">
/// <item>Use when comparing models with different numbers of features</item>
/// <item>Lower than R² when predictors don't contribute meaningfully</item>
/// <item>Can be negative if the model performs very poorly</item>
/// </list></para>
/// </remarks>
public class AdjustedR2Metric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numPredictors;

    /// <inheritdoc />
    public string Name => "AdjustedR2";
    /// <inheritdoc />
    public string Category => "Regression";
    /// <inheritdoc />
    public string Description => "R² adjusted for number of predictors.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => default; // Can be negative
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Initializes the Adjusted R² metric.
    /// </summary>
    /// <param name="numPredictors">Number of predictors (features) in the model. Default is 1.</param>
    public AdjustedR2Metric(int numPredictors = 1)
    {
        _numPredictors = Math.Max(1, numPredictors);
    }

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int n = predictions.Length;
        if (n <= _numPredictors + 1) return NumOps.Zero; // Not enough samples

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var tolerance = NumOps.FromDouble(1e-10);

        // Calculate R² from SS_res and SS_tot around the mean of actuals
        T mean = engine.TensorMean(actual);
        T ssRes = engine.TensorSumOfSquares(engine.TensorSubtract(actual, pred));
        T ssTot = engine.TensorSumOfSquares(engine.TensorSubtractScalar(actual, mean));

        // If actuals are constant (zero variance), R² is only 1 if predictions are also perfect
        if (NumOps.LessThan(NumOps.Abs(ssTot), tolerance))
            return NumOps.LessThan(NumOps.Abs(ssRes), tolerance) ? NumOps.One : NumOps.Zero;

        T r2 = NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));

        // Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
        T scale = NumOps.FromDouble((n - 1.0) / (n - _numPredictors - 1.0));
        return NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.Subtract(NumOps.One, r2), scale));
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
        if (n == 0) return (NumOps.Zero, NumOps.One);
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
