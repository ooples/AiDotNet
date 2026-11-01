using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Interface for keyed audio watermarkers that embed an inaudible watermark in audio and detect it later.
/// </summary>
/// <remarks>
/// <para>
/// A watermarker derives a pseudo-random pattern from its secret key, adds it to a spectral transform of the audio at
/// a level that follows the audio's own loudness, and detects it by correlating the same transform against the same
/// pattern. Only a watermarker constructed with the same key detects the mark; the mark survives added noise, gain
/// changes and re-quantization to 16 bits.
/// </para>
/// <para>
/// <b>For Beginners:</b> An audio watermarker adds an inaudible, password-protected signature to audio your system
/// produces, so the same watermarker can later tell whether a recording carries it.
/// </para>
/// <para>
/// <b>References:</b>
/// - Kirovski and Malvar, Spread-Spectrum Watermarking of Audio Signals (IEEE TSP, 2003)
/// - Only 38% of AI generators implement adequate watermarking (2025, arxiv:2503.18156)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IAudioWatermarker<T> : IAudioSafetyModule<T>
{
    /// <summary>
    /// Returns a copy of <paramref name="audioSamples"/> carrying this watermarker's keyed watermark.
    /// </summary>
    /// <param name="audioSamples">The audio samples to mark.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>The watermarked audio, with the same length. Audio within [-1, 1] stays within it.</returns>
    Vector<T> EmbedWatermark(Vector<T> audioSamples, int sampleRate);

    /// <summary>
    /// Detects the watermark confidence score in the given audio (0.0 = no watermark, 1.0 = certain).
    /// </summary>
    /// <param name="audioSamples">The audio samples to check.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>A watermark detection confidence score.</returns>
    double DetectWatermark(Vector<T> audioSamples, int sampleRate);

    /// <summary>
    /// Detects the watermark and reports the test behind the confidence.
    /// </summary>
    /// <param name="audioSamples">The audio samples to check.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>The detection result, including the z-score and p-value.</returns>
    WatermarkResult Detect(Vector<T> audioSamples, int sampleRate);
}
