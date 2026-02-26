namespace AiDotNet.Enums;

/// <summary>
/// Types of guidance methods for diffusion model inference.
/// </summary>
/// <remarks>
/// <para>
/// Guidance methods control how the diffusion model balances quality, diversity,
/// and adherence to conditioning signals during generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Guidance is like giving the AI artist instructions on how
/// closely to follow your prompt. Different methods trade off between creativity
/// and accuracy in different ways.
/// </para>
/// </remarks>
public enum GuidanceType
{
    /// <summary>No guidance — unconditional generation.</summary>
    None,

    /// <summary>Standard Classifier-Free Guidance (CFG) using conditional/unconditional interpolation.</summary>
    ClassifierFree,

    /// <summary>Perturbed Attention Guidance (PAG) — uses attention perturbation instead of negative prompt.</summary>
    PerturbedAttention,

    /// <summary>Self-Attention Guidance (SAG) — leverages self-attention maps for adaptive guidance.</summary>
    SelfAttention,

    /// <summary>Dynamic CFG — adjusts guidance scale per timestep for better quality.</summary>
    DynamicCFG,

    /// <summary>Rescaled CFG — rescales the guided prediction to prevent over-saturation.</summary>
    RescaledCFG,

    /// <summary>Adaptive Projected Guidance (APG) — projects guidance to reduce artifacts.</summary>
    AdaptiveProjected,

    /// <summary>ELLA (Efficient Large Language Model Adapter) guidance for enhanced text understanding.</summary>
    ELLA
}
