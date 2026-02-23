using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Interface for audio toxicity detection modules that identify harmful speech content.
/// </summary>
/// <remarks>
/// <para>
/// Audio toxicity detectors analyze speech for harmful content through either
/// transcription-then-text-analysis (ASR pipeline) or direct acoustic feature analysis
/// of tone, prosody, and vocal characteristics associated with aggression or hostility.
/// </para>
/// <para>
/// <b>For Beginners:</b> An audio toxicity detector checks if speech contains harmful
/// content like hate speech, threats, or harassment. It can work by converting speech
/// to text and analyzing it, or by directly analyzing the sound patterns of the voice.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IAudioToxicityDetector<T> : IAudioSafetyModule<T>
{
    /// <summary>
    /// Gets the toxicity score for the given audio (0.0 = safe, 1.0 = maximally toxic).
    /// </summary>
    /// <param name="audioSamples">The audio samples to evaluate.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>A toxicity score between 0.0 and 1.0.</returns>
    double GetToxicityScore(Vector<T> audioSamples, int sampleRate);
}
