using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Area Under the ROC Curve (AUC-ROC): measures discrimination ability.
/// </summary>
/// <remarks>
/// <para>
/// AUC-ROC measures how well the model ranks positive examples higher than negative examples.
/// </para>
/// <para>
/// <b>For Beginners:</b> AUC-ROC answers: "If I pick a random positive and random negative,
/// what's the probability the model scores the positive higher?"
/// <list type="bullet">
/// <item>AUC = 1.0: Perfect ranking</item>
/// <item>AUC = 0.5: Random guessing (no discrimination)</item>
/// <item>AUC &lt; 0.5: Worse than random (model is inverted)</item>
/// </list>
/// </para>
/// <para>
/// <b>Advantages:</b> Threshold-independent, works well with imbalanced data, widely used.
/// </para>
/// </remarks>
public class AUCROCMetric<T> : IProbabilisticClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "AUCROC";
    /// <inheritdoc />
    public string Category => "Classification";
    /// <inheritdoc />
    public string Description => "Area Under the ROC Curve, measuring discrimination ability.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => true;
    /// <inheritdoc />
    public bool SupportsMultiClass => true;

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses = 2)
    {
        if (numClasses < 2)
            throw new ArgumentException("Number of classes must be at least 2.", nameof(numClasses));
        if (probabilities.Length == 0 || actuals.Length == 0) return NumOps.FromDouble(0.5);

        if (numClasses == 2)
        {
            if (probabilities.Length != actuals.Length)
                throw new ArgumentException("For binary, probabilities and actuals must have same length.");
            return ComputeBinaryAUC(probabilities, actuals);
        }
        else
        {
            // Multi-class: One-vs-Rest macro average
            // Validate probability array shape: should be n * numClasses where n = actuals.Length
            int expectedProbsLength = actuals.Length * numClasses;
            if (probabilities.Length != expectedProbsLength)
                throw new ArgumentException(
                    $"For {numClasses}-class problem, probabilities length ({probabilities.Length}) must equal " +
                    $"actuals.Length * numClasses ({actuals.Length} * {numClasses} = {expectedProbsLength}).",
                    nameof(probabilities));

            return ComputeMultiClassAUC(probabilities, actuals, numClasses);
        }
    }

    private T ComputeBinaryAUC(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals)
        => ComputeBinaryAUC(EvaluationTensorOps<T>.ToTensor(probs), EvaluationTensorOps<T>.ToTensor(actuals));

    private static T ComputeBinaryAUC(Tensor<T> probs, Tensor<T> actuals)
    {
        // Mann-Whitney U statistic: the fraction of (positive, negative) pairs ranked correctly, ties counting
        // one half, which is the trapezoidal area under the ROC curve over its distinct thresholds.
        Tensor<T> positiveMask;
        using (new NoGradScope<T>())
        {
            positiveMask = AiDotNetEngine.Current.TensorGreaterThan(actuals, NumOps.FromDouble(0.5));
        }

        int positives = EvaluationTensorOps<T>.Count(positiveMask);
        int negatives = probs.Length - positives;

        if (positives == 0 || negatives == 0) return NumOps.FromDouble(0.5);

        return EvaluationTensorOps<T>.RocAuc(probs, positiveMask);
    }

    private T ComputeMultiClassAUC(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals, int numClasses)
    {
        int n = actuals.Length;
        var engine = AiDotNetEngine.Current;
        var classAucs = new List<T>(numClasses);
        using var noGrad = new NoGradScope<T>();
        var probMatrix = engine.Reshape(EvaluationTensorOps<T>.ToTensor(probs), new[] { n, numClasses });
        var actualTensor = EvaluationTensorOps<T>.ToTensor(actuals);

        int c = 0;
        foreach (T classValue in engine.TensorArange<T>(0, numClasses, 1))
        {
            // Extract probabilities for class c and binary (one-vs-rest) labels
            var classProbs = engine.TensorSelect(probMatrix, 1, c);
            var classLabels = EvaluationTensorOps<T>.EqualMask(actualTensor, classValue);
            c++;

            // Check if we have both classes
            int positives = EvaluationTensorOps<T>.Count(classLabels);
            if (positives > 0 && positives < n)
            {
                classAucs.Add(EvaluationTensorOps<T>.RocAuc(classProbs, classLabels));
            }
        }

        return classAucs.Count > 0
            ? engine.TensorMean(new Tensor<T>(classAucs.ToArray(), new[] { classAucs.Count }))
            : NumOps.FromDouble(0.5);
    }

    /// <inheritdoc />
    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals,
        int numClasses = 2, ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(probabilities, actuals, numClasses);
        var (lower, upper) = BootstrapCI(probabilities, actuals, numClasses, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals, int numClasses,
        int samples, double conf, int? seed)
    {
        int n = actuals.Length;
        if (n == 0) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var values = new List<T>(samples);
        var probArr = probs.ToArray();
        var actArr = actuals.ToArray();
        int probsPerSample = numClasses == 2 ? 1 : numClasses;

        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n * probsPerSample];
            var sa = new T[n];
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sa[i] = actArr[idx];
                for (int c = 0; c < probsPerSample; c++)
                    sp[i * probsPerSample + c] = probArr[idx * probsPerSample + c];
            }
            values.Add(Compute(sp, sa, numClasses));
        }
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return EvaluationTensorOps<T>.OrderStatistics(values, lo, hi);
    }
}
