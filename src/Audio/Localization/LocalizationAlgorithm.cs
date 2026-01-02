namespace AiDotNet.Audio.Localization;

/// <summary>
/// Sound source localization algorithms.
/// </summary>
public enum LocalizationAlgorithm
{
    /// <summary>Generalized Cross-Correlation with Phase Transform.</summary>
    GCCPHAT,

    /// <summary>Multiple Signal Classification.</summary>
    MUSIC,

    /// <summary>Steered Response Power with Phase Transform.</summary>
    SRPPHAT
}
