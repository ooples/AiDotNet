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
    /// Gets or sets the percentile of each block's magnitudes used as its quantization scale.
    /// </summary>
    /// <value>The scale percentile. Default 100: the block's absolute maximum, as in the paper.</value>
    /// <remarks>
    /// <para>
    /// Block-wise dynamic quantization normalizes each block by its absolute maximum (Dettmers et al., 2022, and the
    /// bitsandbytes reference). A lower percentile clips the block's largest values to the scale. For the second
    /// moment that is harmful rather than an outlier guard: a clipped v is stored SMALLER than it is, so exactly the
    /// coordinates with the largest gradients get too-small Adam denominators and oversized steps. With the former
    /// default of 99.9, a 40-step training run's parameters diverged to ~14 while full-precision Adam moved 0.14, and
    /// on a four-element quadratic the first parameter stalled at 1.53 where Adam (and this optimizer at 100) reaches
    /// 0.90. Values below 100 remain available for experiments.
    /// </para>
    /// <para><b>For Beginners:</b> Each group of numbers is compressed relative to its largest member. Keeping that
    /// largest member exact (100) is what the original method does; lowering it trades accuracy on the biggest values
    /// for precision on the rest, which for Adam's second moment makes training unstable.</para>
    /// <para><b>Reference:</b> T. Dettmers, M. Lewis, S. Shleifer, L. Zettlemoyer, "8-bit Optimizers via Block-wise
    /// Quantization", ICLR 2022.</para>
    /// </remarks>
    public double QuantizationPercentile { get; set; } = 100.0;

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

    /// <summary>
    /// Stores the optimizer moment state (m and v) as BFloat16 (2 bytes/element) instead of the
    /// default 8-bit block-quantized representation (1 byte/element). Default: false.
    /// </summary>
    /// <value>True to store moments as BFloat16; false (default) to use 8-bit block quantization.</value>
    /// <remarks>
    /// <para>
    /// BFloat16 keeps the full float32 exponent (only the mantissa is shortened), so it preserves
    /// dynamic range without per-block scale factors and changes Adam's convergence far less than the
    /// 8-bit block quantization does — at the cost of using twice the storage (still half of fp32).
    /// This is the gentle, proactive rung of the optimizer-memory ladder: fp32 (4B) → BF16 (2B) →
    /// 8-bit (1B). When true, this overrides <see cref="BlockSize"/>/<see cref="CompressBothMoments"/>
    /// and the block-quant path for the tape (NN) training step.
    /// </para>
    /// <para><b>For Beginners:</b> This halves the memory the optimizer needs to remember each weight's
    /// momentum and variance, with almost no effect on how well the model learns — a safer way to fit a
    /// big model in memory than the more aggressive 8-bit mode.
    /// </para>
    /// </remarks>
    public bool UseBFloat16MomentStorage { get; set; } = false;

    /// <summary>
    /// Parameters with fewer elements than this keep full-precision Adam moments instead of 8-bit ones.
    /// Default: 4096.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is bitsandbytes' <c>min_8bit_size</c>, the reference implementation of Dettmers et al., "8-bit
    /// Optimizers via Block-wise Quantization" (ICLR 2022). Small tensors (biases, normalization scales) add almost
    /// nothing to optimizer memory but are where one block's quantization error touches every element, so they stay
    /// full precision. The rule applies to each parameter tensor in the tape training step and to the whole
    /// parameter vector in the flat-vector path. Set to 0 to quantize every parameter.
    /// </para>
    /// <para><b>For Beginners:</b> Only big weight tensors are compressed; tiny ones are kept exact because
    /// compressing them saves nearly no memory.
    /// </para>
    /// </remarks>
    public int Min8BitSize { get; set; } = 4096;
}
