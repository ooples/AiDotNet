namespace AiDotNet.Interpretability;

/// <summary>
/// The auto-audit of how faithful a model's explanations are — whether the features they highlight
/// actually drive the model's output. Surfaced on the built result when a model explainer is configured.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mainstream explainability libraries produce attributions but do not tell you whether to trust them.
/// This runs deletion/insertion and comprehensiveness/sufficiency perturbation tests against the trained
/// model, so a low faithfulness score is a warning that the explanation does not reflect the model.
/// </para>
/// </remarks>
public sealed class ExplanationFaithfulnessReport<T>
{
    /// <summary>
    /// Faithfulness of the configured explainer's global attributions, or <c>null</c> when the explainer
    /// cannot produce a global attribution vector (does not implement <see cref="IGlobalAttributionExplainer{T}"/>).
    /// </summary>
    public FaithfulnessReport<T>? ExplainerFaithfulness { get; init; }

    /// <summary>The configured explainer's type name, for reference.</summary>
    public string? ExplainerName { get; init; }

    /// <summary>
    /// Faithfulness of the model's own built-in feature importances, or <c>null</c> when they could not be
    /// aligned to the input features.
    /// </summary>
    public FaithfulnessReport<T>? ModelImportanceFaithfulness { get; init; }
}
