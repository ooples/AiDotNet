namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Detailed result from multimodal safety evaluation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MultimodalSafetyResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class MultimodalSafetyResult
{
    /// <summary>Whether the multimodal content is safe overall.</summary>
    public bool IsSafe { get; init; }

    /// <summary>Cross-modal alignment score (0.0 = misaligned, 1.0 = aligned).</summary>
    public double AlignmentScore { get; init; }

    /// <summary>Whether a cross-modal attack was detected.</summary>
    public bool CrossModalAttackDetected { get; init; }

    /// <summary>Modalities involved in the evaluation.</summary>
    public IReadOnlyList<string> Modalities { get; init; } = Array.Empty<string>();
}
