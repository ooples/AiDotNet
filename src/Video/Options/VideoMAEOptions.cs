using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoMAE video model.
/// </summary>
public class VideoMAEOptions : NeuralNetworkOptions
{
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
