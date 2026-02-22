using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the AMT all-pairs multi-field transforms model.
/// </summary>
/// <remarks>
/// <para>
/// AMT (Li et al., CVPR 2023) uses correlation-based all-pairs transforms:
/// - All-pairs correlation: computes dense 4D cost volume between every pixel pair across
///   two frames at multiple scales, providing exhaustive motion correspondence information
/// - Multi-field transforms: instead of a single flow field, predicts multiple (K) candidate
///   flow fields per pixel, each capturing a plausible motion hypothesis, which are then
///   merged via learned soft selection weights
/// - Iterative refinement: coarse-to-fine correlation lookup with GRU-based iterative
///   updates that progressively refine the multi-field estimates
/// - Efficient correlation: uses separable 1D correlation (H then W) instead of full 2D
///   correlation to reduce the quartic cost to quadratic
/// </para>
/// <para>
/// <b>For Beginners:</b> AMT tries every possible match between pixels in two frames
/// (all-pairs). For each pixel, instead of guessing a single motion direction, it proposes
/// several candidates and lets the network pick the best one. This handles tricky cases
/// like objects moving in front of each other or disappearing behind things.
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
    public AMTOptions(AMTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumFlowFields = other.NumFlowFields;
        NumRefinementIters = other.NumRefinementIters;
        NumCorrelationLevels = other.NumCorrelationLevels;
        CorrelationRadius = other.CorrelationRadius;
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

    /// <summary>Gets or sets the number of candidate flow fields per pixel.</summary>
    /// <remarks>The paper uses K=5 candidate fields in the base model.</remarks>
    public int NumFlowFields { get; set; } = 5;

    /// <summary>Gets or sets the number of GRU refinement iterations.</summary>
    /// <remarks>More iterations improve accuracy at the cost of compute. Paper uses 6.</remarks>
    public int NumRefinementIters { get; set; } = 6;

    /// <summary>Gets or sets the number of correlation pyramid levels.</summary>
    public int NumCorrelationLevels { get; set; } = 4;

    /// <summary>Gets or sets the correlation search radius at each level.</summary>
    public int CorrelationRadius { get; set; } = 4;

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
