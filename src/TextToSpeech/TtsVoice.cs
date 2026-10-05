namespace AiDotNet.TextToSpeech;

/// <summary>
/// The voice a multi-speaker or adaptive text-to-speech model synthesizes in: a speaker and, for models that read
/// one, a reference recording of that speaker.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Some papers define inference only relative to a voice. AdaSpeech (Chen et al. 2021, §3) synthesizes with the
/// speaker's embedding and "the utterance-level acoustic conditions ... extracted from another reference speech of
/// the speaker"; text alone does not determine its output. Such a model declares what its voice must carry
/// (<see cref="TtsModelBase{T}.SynthesisVoiceRequirement"/>) and refuses to synthesize until
/// <see cref="TtsModelBase{T}.Voice"/> is set.
/// </para>
/// <para><b>For Beginners:</b> Pick the speaker (its index in the speakers the model was trained on) and give a short
/// recording of them, as a mel spectrogram, so the model can match how they sound.</para>
/// </remarks>
public sealed class TtsVoice<T>
{
    /// <summary>Index of the speaker in the model's speaker table.</summary>
    public int SpeakerId { get; init; }

    /// <summary>A reference recording of the speaker as a mel spectrogram, <c>[frames, melChannels]</c>.</summary>
    public Tensor<T>? Reference { get; init; }

    /// <summary>
    /// The transcript of <see cref="Reference"/> as the model's text tokens, for models that continue a prompt from its
    /// text (E2 TTS, F5-TTS prefix it to the text to synthesize and take the speaking rate from the pair).
    /// </summary>
    public Tensor<T>? ReferenceTokens { get; init; }

    /// <summary>
    /// A reference recording of the speaker as a waveform, <c>[samples]</c> at the model's sample rate, for models whose
    /// speaker encoder reads audio (YourTTS's H/ASP speaker encoder).
    /// </summary>
    public Tensor<T>? ReferenceAudio { get; init; }

    /// <summary>Index of the language in a multilingual model's language table.</summary>
    public int? LanguageId { get; init; }
}
