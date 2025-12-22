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
public class CertifiedDefenseOptions<T>
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
    /// Gets or sets the certification method to use.
    /// </summary>
    /// <value>The method name, defaulting to "RandomizedSmoothing".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different certification methods have different trade-offs:
    /// - RandomizedSmoothing: Works for any model, scalable
    /// - IBP (Interval Bound Propagation): Faster but might be looser
    /// - CROWN: Tight bounds but more computationally expensive</para>
    /// </remarks>
    public string CertificationMethod { get; set; } = "RandomizedSmoothing";

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
    /// <value>The norm type, defaulting to "L2".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This defines what "small perturbation" means for certification.
    /// L2 norm is common for randomized smoothing.</para>
    /// </remarks>
    public string NormType { get; set; } = "L2";

    /// <summary>
    /// Gets or sets the random seed for reproducible certification.
    /// </summary>
    /// <value>The random seed, or null for non-deterministic random generation.</value>
    /// <remarks>
    /// <para>
    /// When set to a specific value, the certification process produces reproducible results,
    /// which is useful for testing and debugging. When null, each certification run uses
    /// different random samples for proper statistical validity.
    /// </para>
    /// <para><b>For Beginners:</b> If you need reproducible results (e.g., for testing), set this
    /// to a specific number. For actual robustness certification, leave it as null so each run
    /// uses different random samples.</para>
    /// </remarks>
    public int? RandomSeed { get; set; } = null;
}
