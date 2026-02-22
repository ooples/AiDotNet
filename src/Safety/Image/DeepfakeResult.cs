namespace AiDotNet.Safety.Image;

/// <summary>
/// Detailed result from deepfake and AI-generated image detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DeepfakeResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class DeepfakeResult
{
    /// <summary>Overall deepfake probability score (0.0 = authentic, 1.0 = fake).</summary>
    public double DeepfakeScore { get; init; }

    /// <summary>Whether the image is likely a deepfake or AI-generated.</summary>
    public bool IsDeepfake { get; init; }

    /// <summary>Frequency analysis score (artifacts in frequency domain).</summary>
    public double FrequencyScore { get; init; }

    /// <summary>Consistency analysis score (facial/spatial inconsistencies).</summary>
    public double ConsistencyScore { get; init; }

    /// <summary>Provenance analysis score (metadata and watermark clues).</summary>
    public double ProvenanceScore { get; init; }
}
