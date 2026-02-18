using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Configuration options for the MarbleNet voice activity detection model.
/// </summary>
/// <remarks>
/// <para>
/// MarbleNet (Jia et al., 2021, NVIDIA NeMo) is a lightweight 1D time-channel separable
/// convolutional model for voice activity detection. It uses depth-wise separable convolutions
/// with sub-word modeling to achieve state-of-the-art accuracy while being fast enough for
/// real-time streaming on edge devices.
/// </para>
/// <para>
/// <b>For Beginners:</b> MarbleNet is NVIDIA's efficient voice activity detector. It uses a
/// special type of neural network layer (separable convolutions) that makes it very fast while
/// still being accurate. Think of it as a "speech or not?" classifier that can run in real-time
/// even on a phone or small device.
/// </para>
/// </remarks>
public class MarbleNetOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 64;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Gets or sets the frame duration in milliseconds.</summary>
    public int FrameDurationMs { get; set; } = 31;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the number of initial convolution filters.</summary>
    public int InitialFilters { get; set; } = 128;

    /// <summary>Gets or sets the number of separable conv blocks.</summary>
    public int NumBlocks { get; set; } = 3;

    /// <summary>Gets or sets the number of sub-blocks per block.</summary>
    public int SubBlocksPerBlock { get; set; } = 5;

    /// <summary>Gets or sets the kernel size for separable convolutions.</summary>
    public int KernelSize { get; set; } = 11;

    /// <summary>Gets or sets the detection threshold.</summary>
    public double Threshold { get; set; } = 0.5;

    /// <summary>Gets or sets the minimum speech duration in milliseconds.</summary>
    public int MinSpeechDurationMs { get; set; } = 250;

    /// <summary>Gets or sets the minimum silence duration in milliseconds.</summary>
    public int MinSilenceDurationMs { get; set; } = 100;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    #endregion
}
