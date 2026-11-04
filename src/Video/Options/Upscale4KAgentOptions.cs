using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.Enhancement.Upscale4KAgent{T}"/>, 4KAgent (Zuo et al. 2025) over
/// AiDotNet's native super-resolution experts.
/// </summary>
/// <remarks>
/// <para>
/// The profile fields follow the release's profiles (pipeline/profiles/*.yaml): <see cref="Upscale4K"/>,
/// <see cref="ScaleFactor"/> and <see cref="RestorePreference"/>. The defaults are Gen4K-P, the profile the model is
/// named after: upscale to 4K with the perception-oriented experts.
/// </para>
/// <para>
/// The toolbox experts are AiDotNet's own paper networks, each configured by its options object, so every expert's
/// architecture and training hyperparameters are those of its own paper unless changed here.
/// </para>
/// <para>
/// <b>For Beginners:</b> the agent does not have one network of its own. It runs several existing upscaling networks on
/// the clip, scores each result with a no-reference image-quality measure, keeps the best, and repeats until the clip
/// reaches the target size.
/// </para>
/// </remarks>
public class Upscale4KAgentOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public Upscale4KAgentOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Upscale4KAgentOptions(Upscale4KAgentOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Upscale4K = other.Upscale4K;
        ScaleFactor = other.ScaleFactor;
        RestorePreference = other.RestorePreference;
        RollbackThreshold = other.RollbackThreshold;
        ColorFixMinSide = other.ColorFixMinSide;
        BasicVsr = new BasicVSROptions(other.BasicVsr);
        IconVsr = new IconVSROptions(other.IconVsr);
        RealBasicVsr = new RealBasicVSROptions(other.RealBasicVsr);
        RealViformer = new RealViformerOptions(other.RealViformer);
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Profile

    /// <summary>
    /// Gets or sets whether the agent upscales to 4K (true, Gen4K) or by <see cref="ScaleFactor"/> (false, ExpSR).
    /// </summary>
    /// <remarks>In 4K mode the factor is the smallest of 2, 4, 8 and 16 that brings the longer side to 4000 pixels,
    /// and 1 when it is already there (get_target_factor).</remarks>
    public bool Upscale4K { get; set; } = true;

    /// <summary>Gets or sets the upscaling factor when <see cref="Upscale4K"/> is false: 2, 4, 8 or 16 (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets which experts compete for each super-resolution step (Perception, Gen4K-P's).</summary>
    public FourKAgentRestorePreference RestorePreference { get; set; } = FourKAgentRestorePreference.Perception;

    /// <summary>
    /// Gets or sets the quality score below which a step's best result counts as a failure (0.5, the release's threshold
    /// for its metric score).
    /// </summary>
    public double RollbackThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the longer output side from which a super-resolution result is AdaIN color-corrected to its input
    /// (1024).
    /// </summary>
    public int ColorFixMinSide { get; set; } = 1024;

    #endregion

    #region Experts

    /// <summary>Gets or sets the BasicVSR expert (fidelity).</summary>
    public BasicVSROptions BasicVsr { get; set; } = new();

    /// <summary>Gets or sets the IconVSR expert (fidelity).</summary>
    public IconVSROptions IconVsr { get; set; } = new();

    /// <summary>Gets or sets the RealBasicVSR expert (perception).</summary>
    public RealBasicVSROptions RealBasicVsr { get; set; } = new();

    /// <summary>Gets or sets the RealViformer expert (perception).</summary>
    public RealViformerOptions RealViformer { get; set; } = new();

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to an ONNX model file for inference mode.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
