namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Detailed result from watermark detection across any modality.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> WatermarkResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class WatermarkResult
{
    /// <summary>Whether a watermark was detected.</summary>
    public bool WatermarkDetected { get; init; }

    /// <summary>Detection confidence (0.0 = no watermark, 1.0 = certain).</summary>
    public double Confidence { get; init; }

    /// <summary>The modality of the content checked (text, image, audio).</summary>
    public string Modality { get; init; } = string.Empty;

    /// <summary>The detected watermark type, if identifiable.</summary>
    public string WatermarkType { get; init; } = string.Empty;
}
