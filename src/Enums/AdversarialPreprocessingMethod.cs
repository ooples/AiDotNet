namespace AiDotNet.Enums;

/// <summary>
/// A built-in input-transformation defense that removes adversarial perturbations before the model sees the input.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each value is a different way of "cleaning" an input so that tiny, carefully
/// crafted changes get erased while the real content survives. To use your own transformation instead, set
/// <see cref="AiDotNet.Models.Options.AdversarialDefenseOptions{T}.CustomPreprocessor"/>.</para>
/// </remarks>
public enum AdversarialPreprocessingMethod
{
    /// <summary>
    /// JPEG compression in the 8 x 8 block DCT domain (Das et al. 2017, "Keeping the Bad Guys Out").
    /// </summary>
    JpegCompression,

    /// <summary>
    /// Feature squeezing by bit-depth reduction (Xu et al. 2018, "Feature Squeezing"): x is mapped to
    /// round(x (2^b - 1)) / (2^b - 1) with b = <see cref="AiDotNet.Models.Options.AdversarialDefenseOptions{T}.BitDepth"/>.
    /// </summary>
    BitDepthReduction,

    /// <summary>
    /// Feature squeezing by local spatial smoothing (Xu et al. 2018): a median filter over a k x k window,
    /// k = <see cref="AiDotNet.Models.Options.AdversarialDefenseOptions{T}.MedianFilterSize"/>, with reflected borders.
    /// </summary>
    MedianSmoothing
}
