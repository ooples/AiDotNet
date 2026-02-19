using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FlashVSR real-time streaming video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// FlashVSR (Zhuang et al., 2025) achieves real-time 4x video super-resolution (~17 FPS)
/// through a one-step diffusion framework with three key innovations:
/// - Locality-Constrained Sparse Attention (LCSA): limits attention to local windows for efficiency
/// - Tiny Conditional Decoder: lightweight decoder that generates HR output in a single diffusion step
/// - Flow-guided temporal alignment: deformable convolution guided by optical flow for multi-frame fusion
/// </para>
/// <para>
/// <b>For Beginners:</b> FlashVSR is a video upscaler that makes low-resolution video look sharper
/// in real time. Most diffusion-based methods need 20-50 steps (very slow), but FlashVSR does it
/// in just one step by using a distilled model, making it fast enough for live streaming.
/// </para>
/// </remarks>
public class FlashVSROptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    /// <remarks>
    /// <para>
    /// - Tiny: 32 features, 4 LCSA blocks (fastest, ~30 FPS)
    /// - Small: 48 features, 6 LCSA blocks (~20 FPS)
    /// - Base: 64 features, 8 LCSA blocks (best quality, ~17 FPS)
    /// </para>
    /// </remarks>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the encoder and decoder.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of Locality-Constrained Sparse Attention blocks.</summary>
    /// <remarks>
    /// LCSA restricts attention to local windows, reducing quadratic complexity to linear
    /// while preserving spatial detail needed for super-resolution.
    /// </remarks>
    public int NumLCSABlocks { get; set; } = 8;

    /// <summary>Gets or sets the local window size for LCSA attention.</summary>
    /// <remarks>
    /// Larger windows capture more context but increase computation.
    /// The paper uses 8x8 for the best speed/quality tradeoff.
    /// </remarks>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the number of attention heads in LCSA.</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames for temporal alignment.</summary>
    /// <remarks>
    /// More frames provide better temporal information but increase latency.
    /// The paper uses 5 frames (2 past + current + 2 future).
    /// </remarks>
    public int NumInputFrames { get; set; } = 5;

    /// <summary>Gets or sets the number of residual blocks in the conditional decoder.</summary>
    public int NumDecoderBlocks { get; set; } = 4;

    /// <summary>Gets or sets the feed-forward expansion factor in LCSA blocks.</summary>
    public int FeedForwardExpansion { get; set; } = 4;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the number of warmup steps for the learning rate schedule.</summary>
    public int WarmupSteps { get; set; } = 5000;

    /// <summary>Gets or sets the distillation loss weight (teacher-student).</summary>
    /// <remarks>
    /// FlashVSR is trained via knowledge distillation from a multi-step diffusion teacher.
    /// This weight balances pixel loss vs distillation loss.
    /// </remarks>
    public double DistillationWeight { get; set; } = 0.5;

    #endregion
}
