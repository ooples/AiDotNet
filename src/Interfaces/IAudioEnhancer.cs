using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for audio enhancement models that improve audio quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio enhancement encompasses various techniques to improve audio quality:
/// <list type="bullet">
/// <item><description>Noise Reduction: Remove background noise while preserving speech/music</description></item>
/// <item><description>Speech Enhancement: Improve speech intelligibility and quality</description></item>
/// <item><description>Dereverberation: Remove room echo and reverb artifacts</description></item>
/// <item><description>Echo Cancellation: Remove acoustic echo in communication systems</description></item>
/// <item><description>Bandwidth Extension: Extend frequency range of narrowband audio</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Audio enhancement is like photo editing for sound!
///
/// Common use cases:
/// - Cleaning up podcast recordings (removing AC hum, keyboard clicks)
/// - Improving phone call quality (reducing background noise)
/// - Restoring old recordings (removing tape hiss, crackle)
/// - Video conferencing (echo cancellation, noise suppression)
/// - Hearing aids (speech enhancement in noisy environments)
///
/// How it works (simplified):
/// 1. Analyze the audio to identify "noise" vs "signal"
/// 2. Create a filter that reduces noise while keeping the signal
/// 3. Apply the filter to produce cleaner audio
///
/// Modern approaches use neural networks that learn what clean audio
/// should sound like, producing much better results than traditional methods.
/// </para>
/// </remarks>
public interface IAudioEnhancer<T>
{
    /// <summary>
    /// Gets the sample rate this enhancer operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the number of audio channels supported.
    /// </summary>
    int NumChannels { get; }

    /// <summary>
    /// Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum).
    /// </summary>
    /// <remarks>
    /// Higher values provide more noise reduction but may introduce artifacts.
    /// Start with 0.5-0.7 for natural-sounding results.
    /// </remarks>
    double EnhancementStrength { get; set; }

    /// <summary>
    /// Enhances audio quality by reducing noise and artifacts.
    /// </summary>
    /// <param name="audio">Input audio tensor with shape [channels, samples] or [samples].</param>
    /// <returns>Enhanced audio tensor with the same shape as input.</returns>
    Tensor<T> Enhance(Tensor<T> audio);

    /// <summary>
    /// Enhances audio with a reference signal for echo cancellation.
    /// </summary>
    /// <param name="audio">Input audio (microphone signal).</param>
    /// <param name="reference">Reference audio (speaker playback signal).</param>
    /// <returns>Enhanced audio with echo removed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is for video calls!
    ///
    /// The problem: Your microphone picks up sound from your speakers,
    /// creating an echo for the other person.
    ///
    /// Solution: We know what's playing from the speakers (reference),
    /// so we can subtract it from what the microphone picks up.
    /// </para>
    /// </remarks>
    Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference);

    /// <summary>
    /// Processes audio in real-time streaming mode.
    /// </summary>
    /// <param name="audioChunk">A small chunk of audio for real-time processing.</param>
    /// <returns>Enhanced audio chunk (may have latency).</returns>
    /// <remarks>
    /// For real-time applications like video calls. The enhancer maintains
    /// internal state between calls for continuity.
    /// </remarks>
    Tensor<T> ProcessChunk(Tensor<T> audioChunk);

    /// <summary>
    /// Resets internal state for streaming mode.
    /// </summary>
    void ResetState();

    /// <summary>
    /// Gets the processing latency in samples.
    /// </summary>
    /// <remarks>
    /// Important for real-time applications. Lower latency means faster
    /// response but potentially lower quality enhancement.
    /// </remarks>
    int LatencySamples { get; }

    /// <summary>
    /// Estimates the noise profile from a segment of audio.
    /// </summary>
    /// <param name="noiseOnlyAudio">Audio containing only noise (no signal).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some enhancers work better if you tell them
    /// what the noise sounds like. Record a few seconds of "silence" (just the
    /// background noise) and pass it here.
    /// </para>
    /// </remarks>
    void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio);
}
