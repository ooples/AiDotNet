using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Interface for text watermarking modules that embed and detect watermarks in text.
/// </summary>
/// <remarks>
/// <para>
/// Text watermarkers modify the token distribution or lexical/syntactic structure of text
/// to embed an imperceptible watermark that can later be detected to prove AI origin.
/// Approaches include sampling distribution modification (SynthID-style), synonym
/// substitution, and structural rearrangement.
/// </para>
/// <para>
/// <b>For Beginners:</b> A text watermarker adds an invisible signature to AI-generated text.
/// Humans can't see the watermark, but a detector can find it later to prove the text was
/// AI-generated. This helps with transparency and regulatory compliance.
/// </para>
/// <para>
/// <b>References:</b>
/// - SynthID-Text: Production text watermarking at scale (Google DeepMind, Nature 2024)
/// - SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ITextWatermarker<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Detects the watermark confidence score in the given text (0.0 = no watermark, 1.0 = certain).
    /// </summary>
    /// <param name="text">The text to check for a watermark.</param>
    /// <returns>A watermark detection confidence score.</returns>
    double DetectWatermark(string text);
}
