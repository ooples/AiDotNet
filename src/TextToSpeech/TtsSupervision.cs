namespace AiDotNet.TextToSpeech;

/// <summary>
/// Training supervision a text-to-speech paper takes from outside the recording, which a token/mel pair does not
/// carry and the model cannot derive itself.
/// </summary>
[Flags]
public enum TtsSupervision
{
    /// <summary>The model trains on text and its target spectrogram alone.</summary>
    None = 0,

    /// <summary>Per-token durations from a forced aligner (e.g. FastSpeech 2's Montreal Forced Aligner).</summary>
    Durations = 1,

    /// <summary>A reference recording of the target speaker or style.</summary>
    SpeakerReference = 2,

    /// <summary>The speaker's index in a multi-speaker model's speaker table.</summary>
    SpeakerId = 4,

    /// <summary>
    /// The recording itself (or its linear spectrogram), beyond the mel spectrogram: Tacotron's post-processing
    /// network predicts the linear-frequency spectrogram (Wang et al. 2017, §3.4).
    /// </summary>
    Recording = 8,

    /// <summary>The language's index in a multilingual model's language table (YourTTS, Casanova et al. 2022).</summary>
    LanguageId = 16,

    /// <summary>
    /// A reference recording as a waveform, for models whose speaker encoder reads audio rather than the model's own
    /// spectrogram (YourTTS's H/ASP speaker encoder).
    /// </summary>
    ReferenceRecording = 32,

    /// <summary>
    /// The recording's discrete codec tokens, <c>[frames, codebooks]</c> (or the recording, which the model encodes with
    /// its codec), for models that generate codec tokens rather than a spectrogram (Pheme's SpeechTokenizer codes).
    /// </summary>
    CodecTokens = 64,

    /// <summary>
    /// The codec tokens of another utterance of the same speaker, <c>[frames, codebooks]</c>, which the model reads as an
    /// acoustic prompt in training (VALL-E X's NAR model reads the previous sentence of the same speaker, Zhang et al.
    /// 2023 Eq. 2).
    /// </summary>
    PromptCodecTokens = 128,
}
