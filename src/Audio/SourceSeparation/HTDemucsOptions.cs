using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Configuration options for the HTDemucs (Hybrid Transformer Demucs) model.
/// </summary>
/// <remarks>
/// <para>
/// HTDemucs (Rouard et al., ICASSP 2023) is Meta's hybrid architecture combining a temporal
/// convolutional encoder with cross-domain Transformer attention. It operates on both waveform
/// and spectrogram simultaneously, achieving 9.0 dB SDR on MUSDB18-HQ.
/// </para>
/// <para>
/// <b>For Beginners:</b> HTDemucs is Meta's best music separator. It works in two ways at once:
/// one path processes the raw audio waves, another processes the frequency picture (spectrogram).
/// A Transformer helps both paths share information, giving the best of both worlds.
/// </para>
/// </remarks>
public class HTDemucsOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 44100;
    /// <summary>Gets or sets the STFT window length in samples; the model rounds it up to a power of two.</summary>
    public int FftSize { get; set; } = 4096;
    /// <summary>Gets or sets the number of samples between successive STFT frames.</summary>
    public int HopLength { get; set; } = 1024;
    /// <summary>Gets or sets the number of frequency bins per STFT frame (FftSize / 2 + 1).</summary>
    public int NumFreqBins { get; set; } = 2049;
    /// <summary>Gets or sets the number of audio channels of the reference configuration (2 for stereo). The native layers work on one spectrogram and do not read it.</summary>
    public int NumChannels { get; set; } = 2;

    #endregion

    #region Encoder Architecture

    /// <summary>
    /// Gets or sets the encoder channel progression.
    /// </summary>
    public int[] EncoderChannels { get; set; } = [48, 96, 192, 384];

    /// <summary>
    /// Gets or sets the temporal kernel size.
    /// </summary>
    public int TemporalKernelSize { get; set; } = 8;

    /// <summary>
    /// Gets or sets the temporal stride.
    /// </summary>
    public int TemporalStride { get; set; } = 4;

    #endregion

    #region Transformer

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumTransformerLayers { get; set; } = 5;
    /// <summary>Gets or sets the hidden width of the transformer layers.</summary>
    public int TransformerDim { get; set; } = 384;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 8;
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

    /// <summary>
    /// Gets or sets the learning rate of the fallback AdamW optimizer. HTDemucs declares the paper's Adam recipe
    /// (3e-4 for pre-training), which is used instead whenever no optimizer is passed in.
    /// </summary>
    public double LearningRate { get; set; } = 3e-4;
    /// <summary>Gets or sets a weight decay that is not read: the paper trains with Adam and no weight decay, and the declared recipe follows it.</summary>
    public double WeightDecay { get; set; } = 1e-5;

    #endregion
}
