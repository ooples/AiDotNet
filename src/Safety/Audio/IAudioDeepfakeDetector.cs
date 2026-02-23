using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Interface for audio deepfake and voice cloning detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Audio deepfake detectors analyze audio waveforms for signs of AI-generated speech,
/// voice cloning, or voice conversion. Approaches include spectral analysis of
/// mel spectrograms, speaker embedding verification, and watermark detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> An audio deepfake detector checks if a voice recording is
/// real or AI-generated. It can detect cloned voices, synthesized speech, and voice
/// conversion attacks by analyzing subtle patterns in the audio signal.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
/// - VoiceRadar: Voice deepfake detection framework (NDSS 2025)
/// - AudioSeal: Localized watermarking for voice cloning detection (Meta AI, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IAudioDeepfakeDetector<T> : IAudioSafetyModule<T>
{
    /// <summary>
    /// Gets the deepfake probability score for the given audio (0.0 = authentic, 1.0 = fake).
    /// </summary>
    /// <param name="audioSamples">The audio samples to evaluate.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>A deepfake probability score between 0.0 and 1.0.</returns>
    double GetDeepfakeScore(Vector<T> audioSamples, int sampleRate);
}
