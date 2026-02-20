using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for UPR-Net unified pyramid recurrent network.
/// </summary>
/// <remarks>
/// <para>
/// UPR-Net (Ma et al., 2023) uses a unified pyramid recurrent architecture:
/// - Unified pyramid: a single encoder-decoder pyramid that performs both optical flow estimation
///   and frame synthesis in one pass, avoiding redundant feature computation and sharing
///   multi-scale representations between the two tasks
/// - Recurrent refinement: at each pyramid level, a ConvLSTM recurrently refines flow and frame
///   predictions, iterating until convergence rather than using a fixed number of steps
/// - Bidirectional estimation: simultaneously estimates forward and backward flows with shared
///   weights, using consistency checks between the two directions to detect occlusions
/// - Lightweight design: the unified architecture removes the need for separate flow and
///   synthesis networks, reducing parameters significantly while maintaining quality
/// </para>
/// <para>
/// <b>For Beginners:</b> UPR-Net combines motion estimation and frame creation into a single
/// efficient network that processes images at multiple scales. At each scale, it repeatedly
/// refines its predictions until they're good enough, like iterating on a drawing.
/// </para>
/// </remarks>
public class UPRNetOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of pyramid levels.</summary>
    public int NumPyramidLevels { get; set; } = 5;

    /// <summary>Gets or sets the number of recurrent refinement iterations per level.</summary>
    public int NumRecurrentIters { get; set; } = 3;

    /// <summary>Gets or sets the number of residual blocks per pyramid level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the ConvLSTM hidden dimension.</summary>
    public int LSTMHiddenDim { get; set; } = 64;

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

    #endregion
}
