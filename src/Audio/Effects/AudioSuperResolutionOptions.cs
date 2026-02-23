using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Configuration options for the Audio Super-Resolution model.
/// </summary>
/// <remarks>
/// <para>
/// Audio Super-Resolution (Kuleshov et al., 2017; Li et al., 2021) upsamples low-resolution
/// audio to high-resolution audio using neural networks. It predicts the missing high-frequency
/// content that was lost during compression or low-quality recording, effectively converting
/// 8 kHz telephone audio to 44.1 kHz studio quality, or recovering detail lost in MP3 compression.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio Super-Resolution is like AI-powered upscaling for sound.
/// Just as image super-resolution makes blurry photos sharper, audio super-resolution makes
/// low-quality audio sound clearer and more detailed.
///
/// Common uses:
/// - Upscaling old telephone recordings (8 kHz -> 44.1 kHz)
/// - Recovering quality from heavily compressed audio (MP3 at 64 kbps)
/// - Enhancing voice recordings from cheap microphones
/// - Restoring bandwidth-limited historical recordings
/// </para>
/// </remarks>
public class AudioSuperResolutionOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the input sample rate in Hz (low resolution).</summary>
    public int InputSampleRate { get; set; } = 8000;

    /// <summary>Gets or sets the output sample rate in Hz (high resolution).</summary>
    public int OutputSampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the upsampling factor (OutputSampleRate / InputSampleRate).</summary>
    public int UpsampleFactor { get; set; } = 4;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small", "base", "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of residual blocks.</summary>
    public int NumResBlocks { get; set; } = 8;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the number of attention layers.</summary>
    public int NumAttentionLayers { get; set; } = 2;

    /// <summary>Gets or sets the number of mel bins for feature extraction.</summary>
    public int NumMels { get; set; } = 128;

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

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
