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
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        Vector<T>? sums = null;
        int count = 0;

        foreach (var explanation in explanations)
        {
            var attributions = explanation.GetFeatureAttributions();
            sums ??= new Vector<T>(attributions.Length);
            int d = Math.Min(sums.Length, attributions.Length);
            var absolute = engine.Abs(d == attributions.Length ? attributions : attributions.GetSubVector(0, d));
            if (d == sums.Length)
            {
                sums = engine.Add(sums, absolute);
            }
            else
            {
                // Only the leading d features are present in this explanation; the tail keeps its running sum.
                engine.Add(sums.GetSubVector(0, d), absolute).AsSpan().CopyTo(sums.AsWritableSpan());
            }
            count++;
        }

        if (sums is null || count == 0) return new Vector<T>(0);

        return engine.Divide(sums, numOps.FromDouble(count));
    }
}
