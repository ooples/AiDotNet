using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Configuration options for the Demucs for Noise model.
/// </summary>
/// <remarks>
/// <para>
/// Demucs for Noise (Defossez et al., 2020, Meta) adapts the Demucs architecture
/// (originally for music source separation) for real-time noise suppression. It operates
/// in the time domain with a U-Net encoder-decoder structure and skip connections,
/// achieving high-quality noise removal at low latency (40ms).
/// </para>
/// <para>
/// <b>For Beginners:</b> Demucs for Noise is like using a music separator to pull apart
/// "speech" and "noise" tracks from a recording. The original Demucs separates vocals,
/// drums, bass, and other instruments. This version separates clean speech from background
/// noise - perfect for cleaning up phone calls, podcasts, and video meetings.
/// </para>
/// </remarks>
public class DemucsNoiseOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of audio channels.</summary>
    public int NumChannels { get; set; } = 1;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small", "base", "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the initial hidden channels.</summary>
    public int HiddenChannels { get; set; } = 48;

    /// <summary>Gets or sets the encoder/decoder depth (number of layers).</summary>
    public int Depth { get; set; } = 5;

    /// <summary>Gets or sets the kernel size for the encoder convolutions.</summary>
    public int KernelSize { get; set; } = 8;

    /// <summary>Gets or sets the stride for the encoder convolutions.</summary>
    public int Stride { get; set; } = 4;

    /// <summary>Gets or sets the LSTM hidden size for the bottleneck.</summary>
    public int LSTMHiddenSize { get; set; } = 512;

    /// <summary>Gets or sets the number of LSTM layers in the bottleneck.</summary>
    public int NumLSTMLayers { get; set; } = 2;

    /// <summary>Gets or sets the channel growth factor per depth level.</summary>
    public int ChannelGrowth { get; set; } = 2;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
