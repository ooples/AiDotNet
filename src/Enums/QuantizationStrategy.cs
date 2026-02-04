namespace AiDotNet.Enums;

/// <summary>
/// Specifies the quantization strategy (algorithm) to use for model compression.
/// Different strategies offer varying trade-offs between accuracy, speed, and compression ratio.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Think of quantization strategies like different compression algorithms
/// for photos - some preserve more detail, some compress more aggressively. Each strategy uses
/// different math to decide how to convert big numbers (32-bit) to small numbers (8-bit or 4-bit).</para>
///
/// <para><b>Strategy Comparison:</b></para>
/// <list type="table">
/// <listheader>
/// <term>Strategy</term>
/// <description>Best For</description>
/// </listheader>
/// <item>
/// <term>Dynamic</term>
/// <description>Quick deployment, no calibration data available</description>
/// </item>
/// <item>
/// <term>GPTQ</term>
/// <description>Best accuracy at 3-4 bits, especially for large models</description>
/// </item>
/// <item>
/// <term>AWQ</term>
/// <description>Very large models (70B+), preserves important weights</description>
/// </item>
/// <item>
/// <term>SmoothQuant</term>
/// <description>When you need to quantize both weights AND activations (W8A8)</description>
/// </item>
/// </list>
///
/// <para><b>Research References:</b></para>
/// <list type="bullet">
/// <item><description>GPTQ: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023)</description></item>
/// <item><description>AWQ: Lin et al., "AWQ: Activation-aware Weight Quantization" (2024)</description></item>
/// <item><description>SmoothQuant: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs" (2023)</description></item>
/// <item><description>SpinQuant: Liu et al., "SpinQuant: LLM quantization with learned rotations" (ICLR 2025)</description></item>
/// <item><description>QuIP#: Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks" (2024)</description></item>
/// </list>
/// </remarks>
public enum QuantizationStrategy
{
    /// <summary>
    /// Dynamic quantization - computes scale/zero-point at runtime based on actual values.
    /// Fast to apply but slightly slower inference. Good when no calibration data is available.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like auto-adjusting brightness on a camera - it figures out
    /// the right settings on the fly. No setup needed, works immediately.</para>
    /// <para><b>Pros:</b> No calibration needed, easy to use</para>
    /// <para><b>Cons:</b> Slightly slower inference, less optimal compression</para>
    /// </remarks>
    Dynamic,

    /// <summary>
    /// GPTQ (Generative Pre-trained Transformer Quantization) - uses second-order Hessian information
    /// to minimize quantization error. State-of-the-art for 3-4 bit quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPTQ is like a smart packing algorithm that knows which items
    /// are most important. It uses math (Hessian matrix) to figure out which weights matter most
    /// and preserves those more carefully.</para>
    /// <para><b>Key Innovation:</b> Uses approximate second-order information (OBS framework) to
    /// minimize the layer-wise reconstruction error when quantizing weights.</para>
    /// <para><b>Best for:</b> 3-bit and 4-bit weight quantization with minimal accuracy loss</para>
    /// <para><b>Typical accuracy:</b> Within 1-2% of full precision at 4-bit</para>
    /// </remarks>
    GPTQ,

    /// <summary>
    /// AWQ (Activation-aware Weight Quantization) - protects important weights based on activation magnitudes.
    /// Particularly effective for very large models (70B+ parameters).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> AWQ observes which weights are "activated" most often during
    /// inference and protects those from aggressive quantization. It's like knowing which roads
    /// are most traveled and keeping those in better condition.</para>
    /// <para><b>Key Innovation:</b> Scales weights by their corresponding activation magnitudes
    /// before quantization, then compensates during inference.</para>
    /// <para><b>Best for:</b> Very large models where some weights are disproportionately important</para>
    /// <para><b>Advantage over GPTQ:</b> More stable at 4-bit for 70B+ models</para>
    /// </remarks>
    AWQ,

    /// <summary>
    /// SmoothQuant - transfers quantization difficulty from activations to weights using mathematical
    /// smoothing. Enables efficient W8A8 quantization (both weights and activations at 8-bit).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Activations (intermediate values during inference) often have
    /// outliers that are hard to quantize. SmoothQuant "smooths" these outliers by transferring
    /// some of the range to the weights, making both easier to quantize.</para>
    /// <para><b>Key Innovation:</b> Per-channel smoothing factor: Y = (Xdiag(s)^-1) * (diag(s)W)
    /// where s balances the quantization difficulty between X and W.</para>
    /// <para><b>Best for:</b> When you need to quantize BOTH weights AND activations (W8A8)</para>
    /// <para><b>Result:</b> Enables 8-bit inference for both weights and activations with minimal loss</para>
    /// </remarks>
    SmoothQuant,

    /// <summary>
    /// SpinQuant - uses learned rotation matrices to reduce outliers before quantization.
    /// Presented at ICLR 2025, achieves state-of-the-art 4-bit quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> SpinQuant "rotates" the data in a mathematical sense to spread
    /// out outliers more evenly. It learns the best rotation using a small amount of data.</para>
    /// <para><b>Key Innovation:</b> Optimizes rotation matrices using Cayley parameterization to
    /// transform weight distributions into more quantization-friendly shapes.</para>
    /// <para><b>Best for:</b> 4-bit quantization when maximum accuracy is required</para>
    /// <para><b>Improvement:</b> Up to 45% relative improvement over random rotations (QuaRot)</para>
    /// </remarks>
    SpinQuant,

    /// <summary>
    /// QuIP# (Quantization with Incoherence Processing) - extreme 2-bit quantization using
    /// Hadamard transforms and lattice codebooks. State-of-the-art for sub-4-bit quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> QuIP# achieves incredibly aggressive 2-bit quantization
    /// (just 4 possible values per weight!) while maintaining reasonable accuracy. It uses
    /// special mathematical transformations to spread information more evenly.</para>
    /// <para><b>Key Innovations:</b></para>
    /// <list type="bullet">
    /// <item><description>Incoherence processing using Hadamard matrices</description></item>
    /// <item><description>Lattice codebooks for optimal 2-bit value selection</description></item>
    /// <item><description>Vector quantization for groups of weights</description></item>
    /// </list>
    /// <para><b>Best for:</b> Maximum compression (16x) when accuracy loss is acceptable</para>
    /// <para><b>Typical accuracy:</b> Within 5-10% of full precision at 2-bit</para>
    /// </remarks>
    QuIPSharp,

    /// <summary>
    /// Simple MinMax quantization - uses minimum and maximum values to determine scale.
    /// Fast but less accurate than advanced methods. Good baseline for comparison.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The simplest approach - just look at the smallest and largest
    /// values and divide the range evenly. Fast but not optimal.</para>
    /// </remarks>
    MinMax
}
