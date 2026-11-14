using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability;

/// <summary>
/// Helpers for aggregating per-instance feature attributions into a single global attribution vector.
/// </summary>
internal static class AttributionAggregation
{
    /// <summary>
    /// Global attribution = the mean absolute per-feature attribution across a set of local explanations.
    /// </summary>
    public static Vector<T> MeanAbsolute<T>(System.Collections.Generic.IEnumerable<IFeatureAttribution<T>> explanations)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<double>? sums = null;
        int count = 0;

        // The running sums are kept in double: a sum over many explanations in T can stall (Half stops adding 1
        // at 2048) or overflow, which would shrink the mean.
        foreach (var explanation in explanations)
        {
            var attributions = explanation.GetFeatureAttributions();
            sums ??= new Tensor<double>(new[] { attributions.Length });

            // Features missing from a shorter explanation add 0, so the tail keeps its running sum.
            var absolute = engine.Abs(AiDotNet.Interpretability.Helpers.ExplainerTensorOps<T>.PadOrTruncate(attributions, sums.Length));
            sums = engine.TensorAdd(sums, absolute.AsTensor().Cast<double>());
            count++;
        }

        if (sums is null || count == 0) return new Vector<T>(0);

        return engine.TensorDivideScalar(sums, (double)count).Cast<T>().ToVector();
    }
}
