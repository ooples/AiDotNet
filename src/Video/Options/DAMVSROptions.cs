using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DAM-VSR appearance-motion disentangled video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// DAM-VSR (SIGGRAPH 2025) disentangles appearance and motion for cleaner video SR:
/// - Appearance branch: extracts texture and structure features from individual frames
/// - Motion branch: captures temporal dynamics and inter-frame correspondences separately
/// - Appearance-Motion Fusion: combines both branches with learned gating for reconstruction
/// </para>
/// <para>
/// <b>For Beginners:</b> Most video SR models mix up "what things look like" (appearance) with
/// "how things move" (motion). DAM-VSR separates these into two branches so each can focus
/// on what it does best, then combines them for the final high-resolution output.
/// </para>
/// </remarks>
public class DAMVSROptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks in the reconstruction module.</summary>
    public int NumResBlocks { get; set; } = 15;

    /// <summary>Gets or sets the number of attention heads in the motion branch.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the number of deformable groups per attention head.</summary>
    /// <remarks>Each group learns independent sampling offsets.</remarks>
    public int DeformableGroups { get; set; } = 8;

    /// <summary>Gets or sets the number of sampling points per deformable attention head.</summary>
    public int NumSamplingPoints { get; set; } = 4;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames.</summary>
    public int NumFrames { get; set; } = 7;

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

    /// <summary>Gets or sets the warmup steps.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
