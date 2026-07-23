using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the MP-SENet (Multi-Path Speech Enhancement Network) model.
/// </summary>
/// <remarks>
/// <para>
/// MP-SENet (Lu et al., 2023) predicts both magnitude and phase of the complex spectrogram
/// using parallel magnitude and phase estimation paths with a cross-domain fusion module.
/// It achieves PESQ 3.60 on VoiceBank+DEMAND, surpassing prior single-channel methods.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sound has two components: loudness (magnitude) and timing (phase).
/// Most enhancers only fix loudness and leave timing alone, which limits quality.
/// MP-SENet fixes both simultaneously using two parallel paths that share information,
/// leading to cleaner and more natural-sounding audio.
/// </para>
/// </remarks>
public class MPSENetOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT size for STFT computation.</summary>
    public int FFTSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length for STFT computation.</summary>
    public int HopLength { get; set; } = 128;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small" or "large").</summary>
    public string Variant { get; set; } = "large";

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of frequency bins (FFTSize / 2 + 1).</summary>
    public int NumFreqBins { get; set; } = 257;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
