using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the Stream-DiffVSR low-latency streaming video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// Stream-DiffVSR (Li et al., 2025) achieves low-latency online video super-resolution through:
/// - Auto-regressive temporal guidance: uses previously generated HR frames to condition current denoising
/// - 4-step distilled denoiser: compresses many diffusion steps into just 4 for low latency
/// - Causal temporal conditioning: only looks at past frames, enabling streaming applications
/// </para>
/// <para>
/// <b>For Beginners:</b> Stream-DiffVSR is designed for live video upscaling where you can't
/// look at future frames. It uses a trick called "distillation" to reduce the number of
/// processing steps from ~50 to just 4, making it fast enough for real-time streaming.
/// </para>
/// </remarks>
public class StreamDiffVSROptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of denoising steps (distilled).</summary>
    /// <remarks>Original diffusion uses ~50 steps; distillation reduces to 4.</remarks>
    public int NumDenoisingSteps { get; set; } = 4;

    /// <summary>Gets or sets the number of residual blocks in the denoiser.</summary>
    public int NumResBlocks { get; set; } = 16;

    /// <summary>Gets or sets the temporal radius for causal conditioning (past frames only).</summary>
    public int TemporalRadius { get; set; } = 3;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the latent space dimension for the diffusion process.</summary>
    public int LatentDim { get; set; } = 64;

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

    /// <summary>Gets or sets the warmup steps for the learning rate schedule.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
