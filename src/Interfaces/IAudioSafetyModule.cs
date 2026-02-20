using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for safety modules that operate on audio content.
/// </summary>
/// <remarks>
/// <para>
/// Audio safety modules analyze audio waveforms for safety risks such as deepfake voices,
/// toxic speech, and AI-generated audio. They can also detect embedded watermarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio safety modules check sound content for problems like
/// fake voices (deepfakes), hateful speech, and cloned voices. They help protect against
/// voice impersonation and audio-based fraud.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
/// - AudioSeal: Localized watermarking for voice cloning detection (Meta AI, 2024)
/// - VoiceRadar: Voice deepfake detection (NDSS 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IAudioSafetyModule<T> : ISafetyModule<T>
{
    /// <summary>
    /// Evaluates the given audio waveform for safety and returns any findings.
    /// </summary>
    /// <param name="audioSamples">
    /// The audio waveform as a vector of samples. Assumed mono, normalized to [-1, 1].
    /// </param>
    /// <param name="sampleRate">The sample rate in Hz (e.g., 16000, 44100).</param>
    /// <returns>
    /// A list of safety findings. An empty list means no safety issues were detected.
    /// </returns>
    IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate);
}
