using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for VFIMamba state-space model for video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// VFIMamba (2024) applies Mamba (selective state space model) to frame interpolation:
/// - Selective state space: uses Mamba's selective scan mechanism instead of attention, achieving
///   linear complexity O(N) for processing frame features while maintaining global context,
///   enabling efficient processing of high-resolution frames
/// - Bidirectional scanning: scans frame features in both forward (left-to-right, top-to-bottom)
///   and backward directions, ensuring each pixel has full global context from all directions
/// - Cross-frame state propagation: the SSM state from one frame is propagated to condition
///   the processing of the other frame, enabling implicit motion correspondence without
///   explicit flow estimation
/// - Multi-scale Mamba blocks: hierarchical Mamba blocks at different spatial scales, capturing
///   both fine-grained texture details and coarse motion patterns
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIMamba uses a new type of AI architecture called "Mamba" that can
/// process long sequences efficiently (unlike transformers which get slow with long inputs).
/// This means it can handle high-resolution frames without running out of memory, while still
/// understanding how every pixel relates to every other pixel.
/// </para>
/// </remarks>
public class VFIMambaOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of Mamba blocks per stage.</summary>
    public int NumMambaBlocks { get; set; } = 6;

    /// <summary>Gets or sets the SSM state dimension.</summary>
    public int StateDim { get; set; } = 16;

    /// <summary>Gets or sets the SSM expansion factor.</summary>
    public int ExpansionFactor { get; set; } = 2;

    /// <summary>Gets or sets the number of hierarchical stages.</summary>
    public int NumStages { get; set; } = 4;

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
