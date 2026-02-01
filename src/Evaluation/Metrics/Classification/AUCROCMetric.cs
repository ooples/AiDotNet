using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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

    public string Name => "AUCROC";
    public string Category => "Classification";
    public string Description => "Area Under the ROC Curve, measuring discrimination ability.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => true;
    public bool SupportsMultiClass => true;

    public T Compute(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses = 2)
    {
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
            return ComputeMultiClassAUC(probabilities, actuals, numClasses);
        }
    }

    private T ComputeBinaryAUC(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals)
    {
        // Using trapezoidal rule / Mann-Whitney U statistic
        int n = probs.Length;
        var pairs = new (double prob, double label)[n];
        for (int i = 0; i < n; i++)
            pairs[i] = (NumOps.ToDouble(probs[i]), NumOps.ToDouble(actuals[i]));

        // Sort by probability descending
        Array.Sort(pairs, (a, b) => b.prob.CompareTo(a.prob));

        int positives = 0, negatives = 0;
        for (int i = 0; i < n; i++)
        {
            if (pairs[i].label > 0.5) positives++;
            else negatives++;
        }

        if (positives == 0 || negatives == 0) return NumOps.FromDouble(0.5);

        // Calculate AUC using trapezoidal rule
        double auc = 0;
        int tp = 0, fp = 0;
        double prevTpr = 0, prevFpr = 0;

        for (int i = 0; i < n; i++)
        {
            if (pairs[i].label > 0.5) tp++;
            else fp++;

            double tpr = (double)tp / positives;
            double fpr = (double)fp / negatives;

            // Trapezoidal area
            auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
            prevTpr = tpr;
            prevFpr = fpr;
        }

        return NumOps.FromDouble(auc);
    }

    private T ComputeMultiClassAUC(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals, int numClasses)
    {
        int n = actuals.Length;
        double sumAuc = 0;
        int validClasses = 0;

        for (int c = 0; c < numClasses; c++)
        {
            // Extract probabilities for class c and binary labels
            var classProbs = new T[n];
            var classLabels = new T[n];

            for (int i = 0; i < n; i++)
            {
                classProbs[i] = probs[i * numClasses + c];
                classLabels[i] = Math.Abs(NumOps.ToDouble(actuals[i]) - c) < 0.5 ? NumOps.One : NumOps.Zero;
            }

            // Check if we have both classes
            bool hasPos = false, hasNeg = false;
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(classLabels[i]) > 0.5) hasPos = true;
                else hasNeg = true;
            }

            if (hasPos && hasNeg)
            {
                sumAuc += NumOps.ToDouble(ComputeBinaryAUC(classProbs, classLabels));
                validClasses++;
            }
        }

        return validClasses > 0 ? NumOps.FromDouble(sumAuc / validClasses) : NumOps.FromDouble(0.5);
    }

    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals,
        int numClasses = 2, ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.BCaBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        var value = Compute(probabilities, actuals, numClasses);
        var (lower, upper) = BootstrapCI(probabilities, actuals, numClasses, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals, int numClasses,
        int samples, double conf, int? seed)
    {
        int n = actuals.Length;
        if (n == 0) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var values = new double[samples];
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
            values[b] = NumOps.ToDouble(Compute(sp, sa, numClasses));
        }
        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
