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

    public int SampleRate { get; set; } = 44100;
    public int FftSize { get; set; } = 4096;
    public int HopLength { get; set; } = 1024;
    public int NumFreqBins { get; set; } = 2049;
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

    public int NumTransformerLayers { get; set; } = 5;
    public int TransformerDim { get; set; } = 384;
    public int NumAttentionHeads { get; set; } = 8;
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

    public double LearningRate { get; set; } = 3e-4;
    public double WeightDecay { get; set; } = 1e-5;

    #endregion
}
