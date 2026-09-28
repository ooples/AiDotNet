using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FlowDiffuser model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FlowDiffuser model for diffusion-based optical flow estimation.
/// Default values follow the original paper recommendations.</para>
/// </remarks>
public class FlowDiffuserOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FlowDiffuserOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public FlowDiffuserOptions()
    {
        NumFeatures = 64;
        NumLayers = 8;
    }

    /// <summary>
    /// Gets or sets an optional Adam learning rate for native training.
    /// </summary>
    /// <remarks>
    /// When <see langword="null"/>, FlowDiffuser retains the framework's default optimizer behavior.
    /// </remarks>
    public double? LearningRate { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
