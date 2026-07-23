using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Interface for audio watermarking modules that embed and detect watermarks in audio.
/// </summary>
/// <remarks>
/// <para>
/// Audio watermarkers embed imperceptible watermarks in audio using spread-spectrum,
/// frequency domain, or AudioSeal-style localized techniques. The watermark survives
/// common transformations like compression, resampling, and noise addition.
/// </para>
/// <para>
/// <b>For Beginners:</b> An audio watermarker adds an invisible signature to audio content.
/// Even after the audio is compressed or slightly modified, the watermark can still be
/// detected to prove the audio was AI-generated.
/// </para>
/// <para>
/// <b>References:</b>
/// - AudioSeal: Localized watermarking (Meta AI, 2024, arxiv:2401.17264)
/// - Only 38% of AI generators implement adequate watermarking (2025, arxiv:2503.18156)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IAudioWatermarker<T> : IAudioSafetyModule<T>
{
    /// <summary>
    /// Detects the watermark confidence score in the given audio (0.0 = no watermark, 1.0 = certain).
    /// </summary>
    /// <param name="audioSamples">The audio samples to check.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>A watermark detection confidence score.</returns>
    double DetectWatermark(Vector<T> audioSamples, int sampleRate);
}
