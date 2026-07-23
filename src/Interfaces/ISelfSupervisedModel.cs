namespace AiDotNet.Interfaces;

/// <summary>
/// Marker interface for models that train themselves from <b>unlabeled</b> data — the
/// model owns its training objective rather than fitting a supervised target.
/// </summary>
/// <remarks>
/// <para>
/// Diffusion models are the canonical example: their objective is noise prediction, not
/// <c>Predict(X) ≈ Y</c>. The supervised optimizer path (clone-evaluate-select against a
/// target) cannot express that. When a configured model implements this interface,
/// <see cref="AiModelBuilder{T, TInput, TOutput}"/> routes it to a self-supervised training
/// loop that runs epochs over the data calling the model's own <c>Train</c> per sample.
/// </para>
/// <para><b>For Beginners:</b> "Self-supervised" means the model makes its own learning
/// signal out of the raw data (for diffusion, by adding noise and learning to remove it) —
/// there are no separate labels. Implement this on a generative model so the facade trains
/// it correctly through <c>ConfigureModel(...).BuildAsync()</c>.</para>
/// </remarks>
public interface ISelfSupervisedModel
{
}
