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
}
