namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Interface for TTS models that support zero-shot or few-shot voice cloning from reference audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Voice cloning models can replicate a target speaker's voice from a short reference audio sample:
/// <list type="bullet">
/// <item>Zero-shot: VALL-E, CosyVoice (3-10 seconds of reference audio)</item>
/// <item>Few-shot: GPT-SoVITS, XTTS-v2 (minutes of reference audio)</item>
/// <item>Instant: OpenVoice (separate tone color converter)</item>
/// </list>
/// </para>
/// </remarks>
public interface IVoiceCloner<T> : ITtsModel<T>
{
    /// <summary>
    /// Synthesizes speech in the voice of a reference speaker.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="referenceAudio">Reference audio tensor from the target speaker.</param>
    /// <returns>Synthesized audio waveform in the target speaker's voice.</returns>
    Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio);

    /// <summary>
    /// Extracts a speaker embedding from reference audio.
    /// </summary>
    /// <param name="referenceAudio">Reference audio tensor.</param>
    /// <returns>Speaker embedding tensor.</returns>
    Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio);

    /// <summary>
    /// Gets the minimum reference audio duration in seconds required for cloning.
    /// </summary>
    double MinReferenceDuration { get; }

    /// <summary>
    /// Gets the speaker embedding dimensionality.
    /// </summary>
    int SpeakerEmbeddingDim { get; }
}
