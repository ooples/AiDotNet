using AiDotNet.Models.Options;

using AiDotNet.Video;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RealESRGAN super-resolution model.
/// </summary>
public class RealESRGANOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RealESRGANOptions"/> class carrying
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
    public RealESRGANOptions()
    {
        ScaleFactor = 4;
        NumRRDBBlocks = 23;
        NumFeatures = 64;
        ResidualScale = 0.2;
        L1Lambda = 1.0;
        PerceptualLambda = 1.0;
        GanLambda = 0.1;
    }


    /// <summary>
    /// Gets or sets the num r r d b blocks.
    /// </summary>
    public int NumRRDBBlocks { get; set; }

    /// <summary>
    /// Gets or sets the residual scale.
    /// </summary>
    public double ResidualScale { get; set; }

    /// <summary>
    /// Gets or sets the l1 lambda.
    /// </summary>
    public double L1Lambda { get; set; }

    /// <summary>
    /// Gets or sets the perceptual lambda.
    /// </summary>
    public double PerceptualLambda { get; set; }

    /// <summary>
    /// Gets or sets the gan lambda.
    /// </summary>
    public double GanLambda { get; set; }

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
