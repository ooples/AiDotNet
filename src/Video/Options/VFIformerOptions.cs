using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for VFIformer video frame interpolation transformer.
/// </summary>
/// <remarks>
/// <para>
/// VFIformer (Lu et al., CVPR 2022) applies vision transformers to frame interpolation:
/// - Cross-scale attention: transformer attention mechanism that attends across multiple feature
///   scales simultaneously, capturing both local fine-grained correspondences and global scene
///   structure in a single attention operation
/// - Flow-guided deformable attention: attention queries are positioned based on estimated
///   optical flow, so the model attends to motion-relevant regions rather than wasting attention
///   on irrelevant spatial locations
/// - Multi-frame transformer decoder: a transformer decoder that takes tokens from both input
///   frames and generates intermediate frame tokens, with causal masking adapted for spatial
///   rather than temporal ordering
/// - Efficient token design: uses feature pooling and stride patterns that reduce token count
///   by 16x compared to naive patch tokenization, enabling high-resolution processing
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIformer uses transformers (the same technology behind GPT and
/// other AI models) for frame interpolation. The attention mechanism lets every part of the
/// output frame "look at" relevant parts of both input frames, producing better results
/// especially for complex scenes.
/// </para>
/// </remarks>
public class VFIformerOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of transformer encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of transformer decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the number of deformable attention points.</summary>
    public int NumDeformablePoints { get; set; } = 4;

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
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
