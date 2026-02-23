using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for BSVD bidirectional streaming video denoising.
/// </summary>
/// <remarks>
/// <para>
/// BSVD (Qi et al., ACM MM 2022) enables real-time video denoising through bidirectional
/// streaming with efficient buffer management:
/// - Bidirectional streaming: processes video in both forward and backward passes with shared
///   buffers, so each frame benefits from both past and future context
/// - Streaming buffers: maintains compact latent buffers instead of storing full frames,
///   enabling constant-memory processing regardless of video length
/// - Real-time capability: designed for 30+ fps denoising on consumer GPUs through
///   efficient buffer reuse and single-pass-per-direction processing
/// - Noise-adaptive: handles varying noise levels without requiring noise-level input,
///   making it suitable for real-world video with spatially varying noise
/// </para>
/// <para>
/// <b>For Beginners:</b> BSVD cleans up noisy video in real-time by looking at both past and
/// future frames. Unlike methods that need all frames at once, it processes them in a stream
/// using small memory buffers, making it practical for live video and long recordings.
/// </para>
/// </remarks>
public class BSVDOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public BSVDOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BSVDOptions(BSVDOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumRecurrentBlocks = other.NumRecurrentBlocks;
        BufferDim = other.BufferDim;
        NumLevels = other.NumLevels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of recurrent blocks per direction.</summary>
    public int NumRecurrentBlocks { get; set; } = 4;

    /// <summary>Gets or sets the hidden state dimension for streaming buffers.</summary>
    public int BufferDim { get; set; } = 64;

    /// <summary>Gets or sets the number of U-Net encoder/decoder levels.</summary>
    public int NumLevels { get; set; } = 3;

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
