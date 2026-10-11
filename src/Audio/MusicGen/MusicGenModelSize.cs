namespace AiDotNet.Audio.MusicGen;

/// <summary>MusicGen's model sizes (Copet et al. 2023, §3.1; widths and depths from audiocraft's model scales).</summary>
public enum MusicGenModelSize
{
    /// <summary>300M parameters: D = 1024, 16 heads, 24 layers; trained with D-Adaptation.</summary>
    Small = 0,

    /// <summary>1.5B parameters: D = 1536, 24 heads, 48 layers.</summary>
    Medium = 1,

    /// <summary>3.3B parameters: D = 2048, 32 heads, 48 layers.</summary>
    Large = 2,
}
