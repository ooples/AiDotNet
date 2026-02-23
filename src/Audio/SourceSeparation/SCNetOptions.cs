using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Configuration options for the SCNet (Sparse Compression Network) source separation model.
/// </summary>
/// <remarks>
/// <para>
/// SCNet (Chen et al., 2024) uses a sparse compression approach that compresses frequency features
/// into compact representations before processing with attention layers. This reduces computation
/// while maintaining separation quality. It achieves competitive results on MUSDB18-HQ with
/// significantly fewer parameters than Transformer-based models.
/// </para>
/// <para>
/// <b>For Beginners:</b> SCNet is like a fast note-taker who summarizes a complex speech into
/// key points, processes just those points, and then expands them back to the full detail.
/// By compressing audio information before processing, it runs faster than methods that
/// process every frequency individually, while still producing high-quality separations.
/// </para>
/// </remarks>
public class SCNetOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Gets or sets the number of frequency bins.</summary>
    public int NumFreqBins { get; set; } = 1025;

    #endregion

    #region Sparse Compression Configuration

    /// <summary>Gets or sets the number of compression clusters.</summary>
    /// <remarks>
    /// <para>
    /// Frequency bins are grouped into a smaller number of clusters for efficient processing.
    /// More clusters preserve detail but increase computation.
    /// </para>
    /// </remarks>
    public int NumClusters { get; set; } = 64;

    /// <summary>Gets or sets the compression embedding dimension.</summary>
    public int CompressionDim { get; set; } = 128;

    /// <summary>Gets or sets the number of encoder blocks.</summary>
    public int NumEncoderBlocks { get; set; } = 6;

    /// <summary>Gets or sets the number of decoder blocks.</summary>
    public int NumDecoderBlocks { get; set; } = 6;

    #endregion

    #region Attention Architecture

    /// <summary>Gets or sets the hidden dimension for attention layers.</summary>
    public int AttentionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the feed-forward expansion dimension.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion

    #region Separation

    /// <summary>Gets or sets the source names to separate.</summary>
    public string[] Sources { get; set; } = ["vocals", "drums", "bass", "other"];

    /// <summary>Gets or sets the number of stems/sources.</summary>
    public int NumStems { get; set; } = 4;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 1e-2;

    #endregion
}
