using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Shared configuration for video models: action recognition, segmentation, generation,
/// super-resolution, frame interpolation, tracking, denoising, inpainting and depth.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Video models work on several frames at once rather than a single
/// picture, so on top of the usual size settings they need to know how many frames they look
/// at together. Each model's own options class ships the values from its paper, so you do not
/// normally set any of these.
/// </para>
/// <para>
/// Derived options classes assign their paper's values in their parameterless constructor.
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// </para>
/// <para>
/// <b>Not to be confused with <c>VideoModelOptions&lt;T&gt;</c>,</b> which is an earlier and
/// now largely abandoned attempt at the same idea: it takes a type parameter it never uses,
/// follows the nullable + <c>Effective*</c> pattern this design moves away from, and is
/// derived from by exactly one of the 108 options classes under <c>src/Video/Options</c>
/// (96 of them derive straight from <c>NeuralNetworkOptions</c> instead). It is left in place
/// for now and removed when the video models are wired to this base.
/// </para>
/// </remarks>
public abstract class VideoHyperparameterOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the width of the model's main feature representation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many numbers the model uses to describe what it sees at
    /// each position. This is the most common size knob across the video models — wider
    /// captures more, and costs proportionally more memory.</para>
    /// </remarks>
    public int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the number of stacked blocks in the model.
    /// </summary>
    public int NumLayers { get; set; }

    /// <summary>
    /// Gets or sets how many consecutive frames the model processes together.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A single frame cannot show motion. Models look at a short
    /// stack of frames — commonly 8 or 16 — so they can tell a person sitting down from a
    /// person standing up.</para>
    /// </remarks>
    public int NumFrames { get; set; }

    /// <summary>
    /// Gets or sets the width of the embedding produced for each token or patch.
    /// </summary>
    public int EmbedDim { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the number of output categories, for classification models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Action-recognition models trained on the Kinetics-400
    /// dataset predict one of 400 actions, so they use 400 here.</para>
    /// </remarks>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the upscaling factor, for super-resolution models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A factor of 4 turns a 480p video into roughly 1920p by
    /// quadrupling each side.</para>
    /// </remarks>
    public int ScaleFactor { get; set; }

    /// <summary>
    /// Gets or sets the number of refinement passes, for models that iterate towards an
    /// answer (optical flow, diffusion sampling).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some models improve their own answer repeatedly rather than
    /// producing it in one go. More passes means better quality and slower inference.</para>
    /// </remarks>
    public int NumIterations { get; set; }

    /// <summary>
    /// Throws if a dimension every video model requires has been left unset.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    protected void ValidateCore()
    {
        Require(NumFeatures, nameof(NumFeatures));
        Require(NumFrames, nameof(NumFrames));
    }
}
