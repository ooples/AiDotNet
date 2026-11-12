using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.AMT{T}"/>, all-pairs multi-field transforms
/// for frame interpolation (Li et al., CVPR 2023).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are AMT-S, the release's default configuration (MCG-NKU/AMT, networks/AMT-S.py, cfgs/AMT-S.yaml);
/// <see cref="AmtL"/> returns AMT-L's. Training follows trainers/base_trainer.py.
/// </para>
/// <para>
/// <b>For Beginners:</b> AMT compares every pixel of one frame with every pixel of the other to find where things
/// moved, predicts several candidate motions per pixel, and blends the frames moved along each into the in-between
/// frame.
/// </para>
/// </remarks>
public class AMTOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public AMTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AMTOptions(AMTOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Channels = (int[])other.Channels.Clone();
        SkipChannels = other.SkipChannels;
        NumFlowFields = other.NumFlowFields;
        NumCorrelationLevels = other.NumCorrelationLevels;
        CorrelationRadius = other.CorrelationRadius;
        FeatureEncoderWidths = (int[])other.FeatureEncoderWidths.Clone();
        FeatureDim = other.FeatureDim;
        LargeEncoders = other.LargeEncoders;
        UpdateHiddenDim = other.UpdateHiddenDim;
        UpdateFlowDim = other.UpdateFlowDim;
        UpdateCorrelationDim = other.UpdateCorrelationDim;
        UpdateSecondCorrelationDim = other.UpdateSecondCorrelationDim;
        UpdateFusionDim = other.UpdateFusionDim;
        CombineKernelSize = other.CombineKernelSize;
        LearningRate = other.LearningRate;
        MinLearningRate = other.MinLearningRate;
        TotalIterations = other.TotalIterations;
        WeightDecay = other.WeightDecay;
        TernaryWeight = other.TernaryWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>AMT-L (networks/AMT-L.py, cfgs/AMT-L.yaml): wider, five flow fields, residual encoders.</summary>
    public static AMTOptions AmtL() => new()
    {
        Channels = new[] { 48, 64, 72, 128 },
        SkipChannels = 48,
        NumFlowFields = 5,
        FeatureEncoderWidths = new[] { 64, 72, 128 },
        FeatureDim = 128,
        LargeEncoders = true,
        UpdateHiddenDim = 128,
        UpdateFlowDim = 48,
        UpdateCorrelationDim = 256,
        UpdateSecondCorrelationDim = 160,
        UpdateFusionDim = 124,
        CombineKernelSize = 7,
    };

    #region Architecture

    /// <summary>Gets or sets the context encoder's four widths, 1/2 to 1/16 size (20, 32, 44, 56).</summary>
    public int[] Channels { get; set; } = { 20, 32, 44, 56 };

    /// <summary>Gets or sets the ResBlocks' side channels (20).</summary>
    public int SkipChannels { get; set; } = 20;

    /// <summary>Gets or sets the number of flow fields per pixel at full size (3).</summary>
    public int NumFlowFields { get; set; } = 3;

    /// <summary>Gets or sets the correlation pyramid's levels (4).</summary>
    public int NumCorrelationLevels { get; set; } = 4;

    /// <summary>Gets or sets the correlation lookup radius (3).</summary>
    public int CorrelationRadius { get; set; } = 3;

    /// <summary>Gets or sets the correlation feature encoder's three widths (32, 64, 96).</summary>
    public int[] FeatureEncoderWidths { get; set; } = { 32, 64, 96 };

    /// <summary>Gets or sets the correlation features' width (84).</summary>
    public int FeatureDim { get; set; } = 84;

    /// <summary>
    /// Gets or sets whether the encoders are AMT-L's: residual (not bottleneck) feature-encoder blocks and a 7x7 first
    /// context convolution (false).
    /// </summary>
    public bool LargeEncoders { get; set; }

    /// <summary>Gets or sets the update blocks' hidden width (76).</summary>
    public int UpdateHiddenDim { get; set; } = 76;

    /// <summary>Gets or sets the update blocks' flow-feature width (20).</summary>
    public int UpdateFlowDim { get; set; } = 20;

    /// <summary>Gets or sets the update blocks' correlation-feature width (64).</summary>
    public int UpdateCorrelationDim { get; set; } = 64;

    /// <summary>Gets or sets the width of the second correlation convolution, 0 for none (AMT-S's SmallUpdateBlock).</summary>
    public int UpdateSecondCorrelationDim { get; set; }

    /// <summary>Gets or sets the update blocks' fused motion-feature width (68).</summary>
    public int UpdateFusionDim { get; set; } = 68;

    /// <summary>Gets or sets the kernel of the flow-combination convolutions (3).</summary>
    public int CombineKernelSize { get; set; } = 3;

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's initial learning rate (2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the rate the cosine schedule ends at (2e-5).</summary>
    public double MinLearningRate { get; set; } = 2e-5;

    /// <summary>Gets or sets the cosine schedule's length: 300 epochs of Vimeo90K's 51,312 triplets at batch 24.</summary>
    public int TotalIterations { get; set; } = 641400;

    /// <summary>Gets or sets AdamW's weight decay (0).</summary>
    public double WeightDecay { get; set; }

    /// <summary>Gets or sets the ternary (census) loss's weight against the Charbonnier term (1).</summary>
    public double TernaryWeight { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
