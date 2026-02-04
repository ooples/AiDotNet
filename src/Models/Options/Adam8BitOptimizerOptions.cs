namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the 8-bit Adam optimization algorithm, which reduces memory usage by quantizing optimizer states.
/// </summary>
/// <remarks>
/// <para>
/// 8-bit Adam provides the same optimization behavior as standard Adam but stores the first and second moment
/// estimates (m and v) using 8-bit quantized representations instead of full precision floating point.
/// This reduces memory usage by approximately 4x for these optimizer states, which is significant for large models.
/// </para>
/// <para><b>For Beginners:</b> Training large neural networks requires storing "optimizer state" - extra numbers
/// for each parameter that help the optimizer make better updates. Standard Adam stores two numbers per parameter
/// (momentum and variance), which can use a lot of memory for large models.
///
/// 8-bit Adam compresses these numbers using a technique called quantization, similar to how JPEG compresses images.
/// This reduces memory usage significantly with minimal impact on training quality. It's especially useful when
/// training large models where optimizer memory becomes a bottleneck.
/// </para>
/// <para><b>Memory Savings Example:</b>
/// For a model with 1 billion parameters:
/// - Standard Adam: 8 GB for optimizer states (2 states × 4 bytes × 1B params)
/// - 8-bit Adam: ~2 GB for optimizer states (2 states × 1 byte × 1B params + scaling factors)
/// </para>
/// </remarks>
public class Adam8BitOptimizerOptions<T, TInput, TOutput> : AdamOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the block size for block-wise quantization.
    /// </summary>
    /// <value>The number of elements per quantization block, defaulting to 2048.</value>
    /// <remarks>
    /// <para>
    /// Block-wise quantization divides the optimizer states into blocks of this size, with each block
    /// having its own scaling factor. Smaller blocks provide better precision but require more memory
    /// for scaling factors. Larger blocks use less memory but may have lower precision.
    /// </para>
    /// <para><b>For Beginners:</b> Quantization works by finding a scaling factor that maps numbers to
    /// a smaller range (0-255 for 8-bit). Using one scaling factor per block instead of one for the entire
    /// tensor improves accuracy. A block size of 2048 is a good balance between accuracy and memory overhead.
    ///
    /// Think of it like dividing a large photo into sections and optimizing the compression for each section
    /// separately - you get better quality than using one setting for the whole image.
    /// </para>
    /// </remarks>
    public int BlockSize { get; set; } = 2048;

    /// <summary>
    /// Gets or sets whether to use dynamic quantization that adapts the scale during training.
    /// </summary>
    /// <value>True to use dynamic quantization (default), false for static quantization.</value>
    /// <remarks>
    /// <para>
    /// Dynamic quantization recomputes scaling factors each time the optimizer state is updated,
    /// adapting to the changing distribution of values during training. Static quantization uses
    /// the initial scaling factors throughout training.
    /// </para>
    /// <para><b>For Beginners:</b> As training progresses, the numbers stored by the optimizer change.
    /// Dynamic quantization adjusts how we compress these numbers to match their current range, maintaining
    /// accuracy throughout training. This is recommended for most cases.
    /// </para>
    /// </remarks>
    public bool UseDynamicQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the percentile to use for outlier-aware quantization.
    /// </summary>
    /// <value>The percentile for computing the scale, defaulting to 99.9.</value>
    /// <remarks>
    /// <para>
    /// Instead of using the absolute maximum value to compute the quantization scale (which can be
    /// sensitive to outliers), this option uses a percentile. Values above this percentile are clipped.
    /// Set to 100 to use the absolute maximum (standard quantization).
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes there are a few very large numbers (outliers) that would
    /// cause the quantization to waste precision. By using the 99.9th percentile instead of the maximum,
    /// we ignore extreme outliers and get better precision for the majority of values. Think of it like
    /// adjusting your camera's exposure based on typical brightness rather than the brightest spot.
    /// </para>
    /// </remarks>
    public double QuantizationPercentile { get; set; } = 99.9;

    /// <summary>
    /// Gets or sets the frequency of full-precision state updates.
    /// </summary>
    /// <value>The number of steps between full-precision updates, defaulting to 0 (disabled).</value>
    /// <remarks>
    /// <para>
    /// When enabled, this performs occasional full-precision updates to correct any accumulated
    /// quantization errors. A value of 0 disables this feature. A typical value if enabled is 256 or 512.
    /// </para>
    /// <para><b>For Beginners:</b> Compressing numbers causes small errors that can accumulate over time.
    /// This option periodically does a more accurate update to fix these accumulated errors.
    /// It's usually not needed, but can help if you notice training instability.
    /// </para>
    /// </remarks>
    public int FullPrecisionUpdateFrequency { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use stochastic rounding during quantization.
    /// </summary>
    /// <value>True to use stochastic rounding, false to use standard rounding (default).</value>
    /// <remarks>
    /// <para>
    /// Stochastic rounding rounds up or down randomly based on the fractional part, which provides
    /// unbiased rounding on average. This can help prevent systematic errors from accumulating.
    /// </para>
    /// <para><b>For Beginners:</b> When we round 2.3 to an integer, we always get 2. But over many
    /// rounding operations, we systematically lose 0.3 each time. Stochastic rounding randomly
    /// chooses between 2 and 3 based on the decimal - so 30% of the time we round up to 3.
    /// On average, this gives more accurate results over many operations.
    /// </para>
    /// </remarks>
    public bool UseStochasticRounding { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to compress both first and second moments.
    /// </summary>
    /// <value>True to compress both m and v (default), false to only compress v.</value>
    /// <remarks>
    /// <para>
    /// The second moment (v) is typically more amenable to compression than the first moment (m)
    /// because it contains squared values that are always positive. Setting this to false keeps
    /// the first moment in full precision while only compressing the second moment.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer stores two types of information: momentum (direction)
    /// and variance (how much values have changed). The variance is always positive and compresses better.
    /// If you're concerned about accuracy, you can choose to only compress the variance while keeping
    /// momentum at full precision. This uses less memory savings but may improve training stability.
    /// </para>
    /// </remarks>
    public bool CompressBothMoments { get; set; } = true;
}
