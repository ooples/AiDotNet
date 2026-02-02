using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Hinge Loss: loss function used by Support Vector Machines (SVM).
/// </summary>
/// <remarks>
/// <para>Hinge Loss = (1/N) * Σmax(0, 1 - y_i * ŷ_i) where y ∈ {-1, +1}</para>
/// <para><b>For Beginners:</b> Hinge loss penalizes predictions that are on the wrong side
/// of the decision boundary OR are correct but not confident enough.
/// <list type="bullet">
/// <item>Loss = 0: All predictions are correct with margin ≥ 1</item>
/// <item>Loss increases when predictions are wrong or uncertain</item>
/// </list>
/// Used in SVM training and for evaluating margin-based classifiers.</para>
/// <para><b>Important:</b> Predictions should be raw decision function values (continuous,
/// typically unbounded) from SVM, not class labels. Actuals should be class labels (0/1 or -1/+1).
/// If predictions are already binary class labels, use 0-1 loss instead.</para>
/// </remarks>
public class HingeLossMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "HingeLoss";
    public string Category => "Classification";
    public string Description => "SVM margin-based loss function.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double totalLoss = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            // Predictions should be raw decision function values (continuous scores)
            // Actuals are class labels (0/1 or -1/+1)
            double pred = NumOps.ToDouble(predictions[i]);
            double actual = NumOps.ToDouble(actuals[i]);

            // If actual is 0/1 encoded, convert to -1/+1
            double y = actual <= 0.5 ? -1.0 : 1.0;

            // Use raw prediction score directly - this is the proper hinge loss
            // The prediction should be a decision function output (e.g., signed distance from hyperplane)
            // If predictions are already in [-1, 1], they work directly
            // If predictions are in [0, 1] probability space, convert to [-1, 1]
            double yHat = pred;
            if (pred >= 0.0 && pred <= 1.0)
            {
                // Likely probability output - convert to signed margin space
                yHat = 2.0 * pred - 1.0;
            }

            // Hinge loss: max(0, 1 - y * yHat)
            totalLoss += Math.Max(0.0, 1.0 - y * yHat);
        }

        return NumOps.FromDouble(totalLoss / predictions.Length);
    }

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
        if (n == 0) return (NumOps.Zero, NumOps.Zero);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var values = new double[samples];
        var predArr = pred.ToArray(); var actArr = actual.ToArray();
        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = predArr[idx]; sa[i] = actArr[idx]; }
            values[b] = NumOps.ToDouble(Compute(sp, sa));
        }
        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
