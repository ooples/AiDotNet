using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for the Conformer-based audio fingerprinting model.
/// </summary>
/// <remarks>
/// <para>
/// ConformerFP applies the Conformer architecture (convolution-augmented Transformer) to audio
/// fingerprinting. It combines self-attention for global context with convolutions for local
/// feature extraction, producing highly robust fingerprints for large-scale audio retrieval.
/// </para>
/// <para>
/// <b>For Beginners:</b> ConformerFP uses a powerful AI architecture called "Conformer" that
/// combines two different ways of understanding audio: one that looks at the big picture (like
/// reading a whole sentence) and one that looks at local details (like reading letter by letter).
/// This combination makes it very good at creating fingerprints that can identify songs even
/// from noisy or distorted recordings.
/// </para>
/// </remarks>
public class ConformerFPOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 8000;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 256;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 256;

    /// <summary>Gets or sets the segment duration in seconds.</summary>
    public double SegmentDurationSec { get; set; } = 1.0;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the fingerprint embedding dimension.</summary>
    public int EmbeddingDim { get; set; } = 128;

    /// <summary>Gets or sets the Conformer hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of Conformer layers.</summary>
    public int NumLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the convolution kernel size for the Conformer conv module.</summary>
    public int ConvKernelSize { get; set; } = 31;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Matching

    /// <summary>
    /// Gets or sets the cosine similarity threshold for considering a fingerprint match.
    /// </summary>
    public double MatchThreshold { get; set; } = 0.7;

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

    /// <summary>Gets or sets the contrastive loss temperature.</summary>
    public double Temperature { get; set; } = 0.05;

    #endregion
}
