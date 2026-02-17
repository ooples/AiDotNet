using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for the GraFPrint graph-based audio fingerprinting model.
/// </summary>
/// <remarks>
/// <para>
/// GraFPrint uses graph neural networks to model spectro-temporal relationships in audio for
/// robust fingerprinting. It constructs a graph from spectrogram features where nodes represent
/// time-frequency points and edges capture local relationships, then applies GNN layers to
/// produce compact fingerprint embeddings.
/// </para>
/// <para>
/// <b>For Beginners:</b> GraFPrint treats a song's spectrogram as a network (graph) of connected
/// sound points, then uses a special AI called a graph neural network to turn that network into
/// a fingerprint. This approach captures how different parts of the sound relate to each other,
/// making it very robust to distortions like noise or tempo changes.
/// </para>
/// </remarks>
public class GraFPrintOptions : ModelOptions
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

    /// <summary>Gets or sets the GNN hidden dimension.</summary>
    public int GnnHiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of GNN layers.</summary>
    public int NumGnnLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of graph attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the k-nearest neighbors for graph construction.</summary>
    public int KNeighbors { get; set; } = 8;

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
