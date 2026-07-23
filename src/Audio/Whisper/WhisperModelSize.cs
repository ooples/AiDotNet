namespace AiDotNet.Audio.Whisper;

/// <summary>
/// Available Whisper model sizes.
/// </summary>
public enum WhisperModelSize
{
    /// <summary>Tiny model (~39M parameters) - fastest, least accurate.</summary>
    Tiny,

    /// <summary>Base model (~74M parameters) - good balance of speed and accuracy.</summary>
    Base,

    /// <summary>Small model (~244M parameters) - good accuracy.</summary>
    Small,

    /// <summary>Medium model (~769M parameters) - high accuracy.</summary>
    Medium,

    /// <summary>Large model (~1.5B parameters) - highest accuracy, slowest.</summary>
    Large,

    /// <summary>Large-v2 model - improved large model.</summary>
    LargeV2,

    /// <summary>Large-v3 model - latest large model with best accuracy.</summary>
    LargeV3
}
