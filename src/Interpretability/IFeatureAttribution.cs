using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability;

/// <summary>
/// A common accessor for the per-feature attribution vector an explanation carries, regardless of the
/// method that produced it (SHAP values, integrated gradients, DeepLIFT contributions, ablation deltas,
/// saliency, …).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Each explanation type names its attributions differently (<c>ShapValues</c>, <c>Attributions</c>,
/// <c>Saliency</c>, …). Implementing this lets downstream logic — most importantly the faithfulness
/// audit — read attributions uniformly without knowing the concrete explanation type.
/// </para>
/// </remarks>
public interface IFeatureAttribution<T>
{
    /// <summary>Returns the per-feature attribution vector (one score per input feature).</summary>
    Vector<T> GetFeatureAttributions();
}

/// <summary>
/// An opt-in capability for explainers that can produce a single global per-feature attribution vector
/// over a dataset (typically the mean absolute local attribution).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The explainer implements this using its own (concrete) explanation type, so callers get a uniform
/// attribution vector without dealing with the open explanation-type generic on the local/global
/// explainer interfaces.
/// </remarks>
public interface IGlobalAttributionExplainer<T>
{
    /// <summary>Computes a global per-feature attribution vector over the supplied data.</summary>
    Vector<T> ComputeGlobalAttributions(Matrix<T> data);
}
