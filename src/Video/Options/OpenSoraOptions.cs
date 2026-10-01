using AiDotNet.Attributes;
using AiDotNet.Models.Options;

using AiDotNet.Video.Generation;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the OpenSora video generation model.
/// </summary>
[DimensionDivisibility(nameof(HiddenDim), nameof(NumHeads))]
public class OpenSoraOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OpenSoraOptions"/> class carrying
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
    public OpenSoraOptions()
    {
        NumFrames = 16;
        HiddenDim = 1152;
        // STDiT-XL's 16 heads, which the model used to hard-code; HiddenDim must be divisible by it.
        NumHeads = 16;
        NumLayers = 28;
        NumInferenceSteps = 50;
        GuidanceScale = 7.5;
    }


    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num inference steps.
    /// </summary>
    public int NumInferenceSteps { get; set; }

    /// <summary>
    /// Gets or sets the guidance scale.
    /// </summary>
    public double GuidanceScale { get; set; }

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
