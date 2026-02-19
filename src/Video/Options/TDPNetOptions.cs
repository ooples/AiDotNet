using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for TDPNet temporal difference prediction network.
/// </summary>
/// <remarks>
/// <para>
/// TDPNet (2024) predicts temporal differences for efficient interpolation:
/// - Temporal difference prediction: instead of predicting the full intermediate frame, TDPNet
///   predicts only the temporal difference (residual) between the intermediate frame and a
///   linear blend of the two inputs, focusing network capacity on the non-trivial parts
/// - Difference-aware attention: self-attention modules that attend specifically to regions
///   where the temporal difference is large (motion boundaries, occlusions), avoiding wasting
///   computation on static regions where the linear blend is already accurate
/// - Coarse-to-fine difference refinement: multi-scale architecture where coarse-level
///   differences capture global motion corrections and fine-level differences add sharp
///   texture details and boundary refinement
/// - Lightweight backbone: because predicting residuals is easier than predicting full frames,
///   TDPNet can use a significantly lighter backbone while matching the quality of heavier
///   full-frame prediction methods
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of building the whole in-between frame from scratch, TDPNet
/// starts with a simple average of the two frames and then predicts what needs to change.
/// This is much easier (like correcting a rough draft vs writing from blank) and allows
/// a smaller, faster network to achieve good results.
/// </para>
/// </remarks>
public class TDPNetOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 48;

    /// <summary>Gets or sets the number of difference prediction blocks.</summary>
    public int NumDiffBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of refinement scales.</summary>
    public int NumScales { get; set; } = 3;

    /// <summary>Gets or sets the number of attention heads in difference-aware attention.</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the difference threshold for sparse attention.</summary>
    /// <remarks>Regions with temporal difference below this threshold skip attention.</remarks>
    public double DifferenceThreshold { get; set; } = 0.05;

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
