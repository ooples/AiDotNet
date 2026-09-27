using AiDotNet.Models.Options;

using AiDotNet.Video.ActionRecognition;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoMAE video model.
/// </summary>
public class VideoMAEOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VideoMAEOptions"/> class carrying
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
    public VideoMAEOptions()
    {
        NumClasses = 400;
        NumFrames = 16;
        NumFeatures = 768;
        MaskRatio = 0.9;
    }


    /// <summary>
    /// Gets or sets the mask ratio.
    /// </summary>
    public double MaskRatio { get; set; }

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
    /// <summary>
    /// Gets or sets whether masked-autoencoder pretraining reconstructs per-patch NORMALISED pixels
    /// rather than raw pixels. Default: <c>true</c>, the paper's setting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When on, <c>PretrainMAE</c>'s target for each channel of each masked tubelet patch is that patch's
    /// pixels standardised over its own <c>tubeletSize x 16 x 16</c> values, <c>(x - mean) / (std + 1e-6)</c>
    /// with the unbiased standard deviation. This is the VideoMAE reference implementation's
    /// <c>normlize_target=True</c> default (Tong et al. 2022), carried over from MAE (He et al. 2022,
    /// Sec. 4), where predicting normalised patches improves the learned representation: the model
    /// spends its capacity on local structure rather than on each patch's brightness and contrast.
    /// </para>
    /// <para>
    /// When off, the target is the raw pixel values. Classification (<c>Predict</c>/<c>Train</c>) is not
    /// affected either way.
    /// </para>
    /// <para><b>For Beginners:</b> Leave this on unless you need the reconstruction loss in raw pixel
    /// units (for example, to compare against an older raw-pixel run).</para>
    /// </remarks>
    public bool NormalizeTarget { get; set; } = true;
}
