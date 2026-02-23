using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Configuration options for the BS-RoFormer (Band-Split Rotary Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// BS-RoFormer (Lu et al., 2023) applies band-split processing with rotary position embeddings
/// to achieve state-of-the-art music source separation. It splits the spectrogram into frequency
/// bands, processes each with Transformers, and fuses results, achieving 12.8 dB SDR on vocals.
/// </para>
/// <para>
/// <b>For Beginners:</b> BS-RoFormer separates music into individual instruments by dividing
/// audio into frequency bands (like bass, mid, treble on an equalizer), processing each band
/// with AI, then combining the results. The "Rotary" part helps it understand where sounds
/// occur in time, even for long songs.
/// </para>
/// </remarks>
public class BSRoFormerOptions : ModelOptions
{
    #region Audio Preprocessing

    public int SampleRate { get; set; } = 44100;
    public int FftSize { get; set; } = 2048;
    public int HopLength { get; set; } = 512;
    public int NumFreqBins { get; set; } = 1025;

    #endregion

    #region Band-Split Configuration

    /// <summary>
    /// Gets or sets the number of frequency bands to split into.
    /// </summary>
    public int NumBands { get; set; } = 24;

    /// <summary>
    /// Gets or sets the band embedding dimension.
    /// </summary>
    public int BandEmbeddingDim { get; set; } = 128;

    #endregion

    #region Transformer Architecture

    public int NumTransformerLayers { get; set; } = 12;
    public int TransformerDim { get; set; } = 384;
    public int NumAttentionHeads { get; set; } = 8;
    public int FeedForwardDim { get; set; } = 1536;
    public bool UseRotaryEmbedding { get; set; } = true;
    public double DropoutRate { get; set; } = 0.0;

    #endregion

    #region Separation

    public string[] Sources { get; set; } = ["vocals", "drums", "bass", "other"];
    public int NumStems { get; set; } = 4;

    #endregion

    #region Model Loading

    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    public double LearningRate { get; set; } = 5e-5;
    public double WeightDecay { get; set; } = 1e-2;

    #endregion
}
