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

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 44100;
    /// <summary>Gets or sets the STFT window length in samples; the model rounds it up to a power of two.</summary>
    public int FftSize { get; set; } = 2048;
    /// <summary>Gets or sets the number of samples between successive STFT frames.</summary>
    public int HopLength { get; set; } = 512;
    /// <summary>Gets or sets the number of frequency bins per STFT frame (FftSize / 2 + 1).</summary>
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

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumTransformerLayers { get; set; } = 12;
    /// <summary>Gets or sets the hidden width of the transformer layers.</summary>
    public int TransformerDim { get; set; } = 384;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>Gets or sets the inner width of each feed-forward block.</summary>
    public int FeedForwardDim { get; set; } = 1536;
    /// <summary>Gets or sets whether the paper's rotary position embedding is used. The native layers use plain multi-head attention and do not read it.</summary>
    public bool UseRotaryEmbedding { get; set; } = true;
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion

    #region Separation

    /// <summary>Gets or sets the names of the separated stems, in output order.</summary>
    public string[] Sources { get; set; } = ["vocals", "drums", "bass", "other"];
    /// <summary>Gets or sets the number of stems the model separates.</summary>
    public int NumStems { get; set; } = 4;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate of the default optimizer.</summary>
    public double LearningRate { get; set; } = 5e-5;
    /// <summary>Gets or sets the weight decay of the default optimizer.</summary>
    public double WeightDecay { get; set; } = 1e-2;

    #endregion
}
