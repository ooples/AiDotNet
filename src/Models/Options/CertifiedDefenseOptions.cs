using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for certified defense mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// These options control certified robustness methods that provide provable guarantees
/// about model predictions under adversarial perturbations.
/// </para>
/// <para><b>For Beginners:</b> These settings control how the "guaranteed protection" works.
/// You can adjust how many samples to use, how tight the guarantees should be, and what
/// certification method to apply.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class CertifiedDefenseOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of samples for randomized smoothing.
    /// </summary>
    /// <value>The number of samples, defaulting to 1000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Randomized smoothing makes predictions by averaging over
    /// many noisy versions of the input. More samples give tighter guarantees but take longer.</para>
    /// </remarks>
    public int NumSamples { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the noise standard deviation for randomized smoothing.
    /// </summary>
    /// <value>The noise sigma, defaulting to 0.25.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how much random noise is added. Larger sigma
    /// gives stronger robustness guarantees but might reduce accuracy.</para>
    /// </remarks>
    public double NoiseSigma { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the confidence level for certification.
    /// </summary>
    /// <value>The confidence level (0-1), defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how confident you want to be in the guarantee.
    /// 0.99 means 99% confidence that the guarantee holds. Higher confidence requires more samples.</para>
    /// </remarks>
    public double ConfidenceLevel { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the number of noisy samples randomized smoothing uses to choose the top class
    /// before it estimates that class's probability from <see cref="NumSamples"/> fresh samples.
    /// </summary>
    /// <value>Defaults to 100 (n0 in Cohen et al. 2019, Sec. 4).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Using separate samples to pick the class and to measure how often it
    /// wins keeps the guarantee honest: the same samples would make the winner look more certain.</para>
    /// </remarks>
    public int SelectionSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the certification method to use.
    /// </summary>
    /// <value>Defaults to <see cref="CertifiedDefenseMethod.RandomizedSmoothing"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different certification methods have different trade-offs:
    /// - RandomizedSmoothing: Works for any model, scalable
    /// - IBP (Interval Bound Propagation): Faster but might be looser
    /// - CROWN: Tight bounds but more computationally expensive</para>
    /// </remarks>
    public CertifiedDefenseMethod CertificationMethod { get; set; } = CertifiedDefenseMethod.RandomizedSmoothing;

    /// <summary>
    /// Gets or sets the batch size for certification.
    /// </summary>
    /// <value>The batch size, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Processing multiple inputs at once speeds up certification.
    /// Larger batches are faster but use more memory.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to use tight bounds computation.
    /// </summary>
    /// <value>True for tight bounds, false for faster looser bounds (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tight bounds give stronger guarantees but take longer to compute.
    /// Looser bounds are faster but might certify fewer examples.</para>
    /// </remarks>
    public bool UseTightBounds { get; set; } = false;

    /// <summary>
    /// Gets or sets the norm type for certification.
    /// </summary>
    /// <value>Defaults to <see cref="AdversarialNormType.L2"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This defines what "small perturbation" means for certification.
    /// L2 norm is common for randomized smoothing.</para>
    /// </remarks>
    public AdversarialNormType NormType { get; set; } = AdversarialNormType.L2;

}
