using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.CertifiedRobustness;

/// <summary>
/// Engine-op helpers shared by the certified defenses: class decisions and certified-accuracy summaries.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
internal static class CertificationMath<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Index of the largest entry (first one on ties), or 0 for an empty vector.
    /// </summary>
    public static int ArgMax(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return 0;
        }

        return AiDotNetEngine.Current.TensorArgMax(values.AsTensor(), 0).ToArray()[0];
    }

    /// <summary>
    /// Builds the certified-accuracy metrics from per-input counts and the radii of certified inputs.
    /// </summary>
    /// <param name="cleanCorrect">Inputs the base model classifies correctly.</param>
    /// <param name="certifiedCorrect">Inputs certified correct at the evaluation radius.</param>
    /// <param name="radius">The evaluation radius.</param>
    /// <param name="certifiedCount">Inputs that were certified at all (not abstained).</param>
    /// <param name="radii">The certified radius of every input, zero where it was not certified.</param>
    /// <remarks>
    /// The average and median run over all examples, as <see cref="CertifiedAccuracyMetrics{T}"/> documents;
    /// the certification rate is the fraction of examples that were certified.
    /// </remarks>
    public static CertifiedAccuracyMetrics<T> Summarize(
        int cleanCorrect,
        int certifiedCorrect,
        int certifiedCount,
        T radius,
        List<T> radii)
    {
        var metrics = new CertifiedAccuracyMetrics<T>
        {
            CleanAccuracy = radii.Count > 0 ? (double)cleanCorrect / radii.Count : 0.0,
            CertifiedAccuracy = radii.Count > 0 ? (double)certifiedCorrect / radii.Count : 0.0,
            CertificationRadius = radius,
            CertificationRate = radii.Count > 0 ? (double)certifiedCount / radii.Count : 0.0,
            AverageCertifiedRadius = NumOps.Zero,
            MedianCertifiedRadius = NumOps.Zero
        };

        if (radii.Count > 0)
        {
            var engine = AiDotNetEngine.Current;
            var values = new Vector<T>(radii);
            metrics.AverageCertifiedRadius = engine.Mean<T>(values);

            // Median: the middle order statistic, or the mean of the two middle ones for an even count.
            var order = engine.TensorArgsort(values.AsTensor(), 0, false);
            var sorted = engine.TensorGather(values.AsTensor(), order, 0).ToVector();
            int count = sorted.Length;
            int mid = count / 2;
            var span = sorted.AsSpan();
            metrics.MedianCertifiedRadius = count % 2 == 0
                ? NumOps.Divide(NumOps.Add(span[mid - 1], span[mid]), NumOps.FromDouble(2.0))
                : span[mid];
        }

        return metrics;
    }
}
