namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the method used to remove an entity's influence from a trained VFL model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Under GDPR and similar regulations, individuals have the "right to
/// be forgotten". If a patient asks to be removed from a model, the model must be updated so that
/// it no longer contains any information learned from that patient's data. These methods vary
/// in how thoroughly they remove the influence and how computationally expensive they are.</para>
/// </remarks>
public enum VflUnlearningMethod
{
    /// <summary>
    /// Retrain the model from scratch without the removed entities.
    /// Most thorough but most expensive (requires full retraining).
    /// </summary>
    Retraining,

    /// <summary>
    /// Apply gradient ascent on the removed entities to approximately reverse their influence.
    /// Fast but may not fully remove all learned information.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Think of this as "anti-training" - the model is trained in
    /// reverse on the data to forget, which approximately cancels out what it learned.</para>
    /// </remarks>
    GradientAscent,

    /// <summary>
    /// Uses a primal-dual optimization framework for both sample and label unlearning in VFL.
    /// Provides stronger guarantees than gradient ascent with moderate computational cost.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Based on "Vertical Federated Unlearning via Primal-Dual Method" (2025).</para>
    /// </remarks>
    PrimalDual,

    /// <summary>
    /// Certified unlearning with mathematical guarantees that the unlearned model is statistically
    /// indistinguishable from a model trained without the removed data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides the strongest privacy guarantee: a mathematical proof
    /// that the model truly "forgot" the removed data. However, it may slightly degrade model
    /// accuracy compared to simpler methods.</para>
    /// </remarks>
    Certified
}
