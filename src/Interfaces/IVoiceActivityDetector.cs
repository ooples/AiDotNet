using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for Voice Activity Detection (VAD) models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Voice Activity Detection determines when speech is present in an audio signal.
/// This is a fundamental building block for many speech processing systems.
/// </para>
/// <para><b>For Beginners:</b> VAD answers the question "Is someone speaking right now?"
///
/// Why VAD is important:
/// - Speech Recognition: Only process audio when speech is present (saves compute)
/// - Voice Assistants: Detect when user starts/stops talking
/// - VoIP/Video Calls: Only transmit audio when speaking (saves bandwidth)
/// - Transcription: Find speech segments in long recordings
/// - Speaker Diarization: First step to identify who spoke when
///
/// How it works:
/// 1. Traditional: Look at energy levels, zero-crossing rate, spectral features
/// 2. Modern (Neural): Train a model to classify frames as speech/non-speech
///
/// Key metrics:
/// - Accuracy: How often it's correct
/// - False Positive Rate: Saying "speech" when it's noise (annoying in voice assistants)
/// - False Negative Rate: Missing actual speech (drops words in transcription)
/// - Latency: How quickly it detects speech onset
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("VoiceActivityDetector")]
public interface IVoiceActivityDetector<T>
{
    /// <summary>
    /// Gets the sample rate this VAD operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the frame size in samples used for detection.
    /// </summary>
    int FrameSize { get; }

    /// <summary>
    /// Gets or sets the detection threshold (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// Higher threshold = fewer false positives but may miss quiet speech.
    /// Lower threshold = catches more speech but may trigger on noise.
    /// Default is typically 0.5.
    /// </remarks>
    double Threshold { get; set; }

    /// <summary>
    /// Detects whether speech is present in an audio frame.
    /// </summary>
    /// <param name="audioFrame">Audio frame with shape [samples] or [channels, samples].</param>
    /// <returns>True if speech is detected, false otherwise.</returns>
    bool DetectSpeech(Tensor<T> audioFrame);

    /// <summary>
    /// Gets the speech probability for an audio frame.
    /// </summary>
    /// <param name="audioFrame">Audio frame to analyze.</param>
    /// <returns>Probability of speech (0.0 = definitely not speech, 1.0 = definitely speech).</returns>
    T GetSpeechProbability(Tensor<T> audioFrame);

    /// <summary>
    /// Detects speech segments in a longer audio recording.
    /// </summary>
    /// <param name="audio">Full audio recording.</param>
    /// <returns>List of (startSample, endSample) tuples for each speech segment.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds all the parts where someone is talking.
    ///
    /// Example result for a 10-second recording:
    /// [(0.5s, 2.3s), (4.1s, 6.8s), (8.0s, 9.5s)]
    /// Meaning: Speech from 0.5-2.3s, silence, speech from 4.1-6.8s, etc.
    /// </para>
    /// </remarks>
    IReadOnlyList<(int StartSample, int EndSample)> DetectSpeechSegments(Tensor<T> audio);

    /// <summary>
    /// Gets frame-by-frame speech probabilities for the entire audio.
    /// </summary>
    /// <param name="audio">Full audio recording.</param>
    /// <returns>Array of speech probabilities, one per frame.</returns>
    T[] GetFrameProbabilities(Tensor<T> audio);

    /// <summary>
    /// Processes audio in streaming mode, maintaining state between calls.
    /// </summary>
    /// <param name="audioChunk">A chunk of audio for real-time processing.</param>
    /// <returns>Speech detection result with probability.</returns>
    (bool IsSpeech, T Probability) ProcessChunk(Tensor<T> audioChunk);

    /// <summary>
    /// Resets internal state for streaming mode.
    /// </summary>
    void ResetState();

    /// <summary>
    /// Gets or sets the minimum speech duration in milliseconds.
    /// </summary>
    /// <remarks>
    /// Speech segments shorter than this are ignored (reduces false triggers).
    /// </remarks>
    int MinSpeechDurationMs { get; set; }

    /// <summary>
    /// Gets or sets the minimum silence duration in milliseconds.
    /// </summary>
    /// <remarks>
    /// Silence gaps shorter than this don't split speech segments.
    /// </remarks>
    int MinSilenceDurationMs { get; set; }
}
