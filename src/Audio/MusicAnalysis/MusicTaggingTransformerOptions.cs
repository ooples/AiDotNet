using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the Music Tagging Transformer model.
/// </summary>
/// <remarks>
/// <para>
/// The Music Tagging Transformer (Won et al., 2021) uses a Transformer encoder on mel spectrogram
/// features to predict music tags (genre, mood, instrument, era). It achieves state-of-the-art
/// results on the MagnaTagATune and Million Song Dataset benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model listens to music and automatically tags it with descriptive
/// labels—like "rock", "upbeat", "guitar", "1980s", or "relaxing". It's the technology behind
/// automatic music categorization in streaming services like Spotify's genre detection.
/// </para>
/// </remarks>
public class MusicTaggingTransformerOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 512;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the Transformer hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of Transformer layers.</summary>
    public int NumLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of output tags.</summary>
    public int NumTags { get; set; } = 50;

    /// <summary>Gets or sets the tag label names.</summary>
    public string[] TagLabels { get; set; } = [
        "rock", "pop", "electronic", "classical", "jazz", "hip-hop", "country", "metal", "folk", "r&b",
        "guitar", "piano", "drums", "bass", "synth", "strings", "vocals", "beats", "ambient", "acoustic",
        "happy", "sad", "energetic", "relaxing", "aggressive", "romantic", "dark", "upbeat", "melancholic", "chill",
        "fast", "slow", "medium-tempo", "loud", "quiet", "male-vocal", "female-vocal", "instrumental", "live", "studio",
        "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "indie", "dance", "experimental"
    ];

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion
}
