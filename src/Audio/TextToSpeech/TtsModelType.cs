namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Available TTS model types.
/// </summary>
public enum TtsModelType
{
    /// <summary>FastSpeech2 acoustic model.</summary>
    FastSpeech2,

    /// <summary>Tacotron2 acoustic model.</summary>
    Tacotron2,

    /// <summary>VITS end-to-end TTS model.</summary>
    VITS
}
