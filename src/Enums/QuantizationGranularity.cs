namespace AiDotNet.Enums;

/// <summary>
/// Specifies the granularity level for quantization scaling factors.
/// Finer granularity preserves more accuracy but requires more storage for scale/zero-point values.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When compressing numbers, we need "scaling factors" to convert
/// between big and small numbers. Granularity determines how many different scaling factors we use:</para>
///
/// <list type="bullet">
/// <item><description><b>PerTensor:</b> One scaling factor for the entire layer - simplest, fastest, least accurate</description></item>
/// <item><description><b>PerChannel:</b> One scaling factor per output channel - good balance</description></item>
/// <item><description><b>PerGroup:</b> One scaling factor per N elements - most accurate, used by GPTQ/AWQ</description></item>
/// </list>
///
/// <para><b>Analogy:</b> Think of it like setting brightness on a photo:</para>
/// <list type="bullet">
/// <item><description>PerTensor = One brightness setting for the whole image</description></item>
/// <item><description>PerChannel = Different brightness for red, green, blue</description></item>
/// <item><description>PerGroup = Different brightness for each region of the image</description></item>
/// </list>
///
/// <para><b>Memory Overhead:</b> Finer granularity requires storing more scaling factors:</para>
/// <list type="table">
/// <listheader>
/// <term>Granularity</term>
/// <description>Scale Storage (for 1M params)</description>
/// </listheader>
/// <item><term>PerTensor</term><description>1 scale value</description></item>
/// <item><term>PerChannel (1024 channels)</term><description>1,024 scale values</description></item>
/// <item><term>PerGroup (group=128)</term><description>~7,800 scale values</description></item>
/// </list>
///
/// <para><b>Research Reference:</b> K-Quant in llama.cpp uses a two-level scheme (PerBlock with super-blocks)
/// achieving excellent quality with minimal overhead.</para>
/// </remarks>
public enum QuantizationGranularity
{
    /// <summary>
    /// Per-tensor quantization - single scale and zero-point for the entire tensor.
    /// Fastest but least accurate. Good for quick prototyping or when accuracy isn't critical.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses one "ruler" to measure all values in a layer.
    /// Simple and fast, but if some values are very different from others, accuracy suffers.</para>
    /// <para><b>Storage overhead:</b> 2 values per tensor (scale + zero_point)</para>
    /// <para><b>Best for:</b> Quick deployment, INT8 where accuracy loss is acceptable</para>
    /// </remarks>
    PerTensor,

    /// <summary>
    /// Per-channel quantization - separate scale and zero-point for each output channel.
    /// Standard choice for convolutional neural networks. Good balance of accuracy and efficiency.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each output "filter" or "neuron" gets its own ruler.
    /// This works well because different channels often have different value ranges.</para>
    /// <para><b>Storage overhead:</b> 2 values per channel (typically 64-4096 channels)</para>
    /// <para><b>Best for:</b> CNNs, standard INT8 quantization, production deployments</para>
    /// </remarks>
    PerChannel,

    /// <summary>
    /// Per-group quantization - separate scale and zero-point for each group of N consecutive elements.
    /// Used by GPTQ and AWQ for 4-bit quantization. Most accurate but highest overhead.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Divides parameters into small groups (e.g., 128 elements each)
    /// and gives each group its own ruler. More rulers = more accuracy but more storage.</para>
    /// <para><b>Storage overhead:</b> 2 values per group (e.g., 1 scale per 128 weights = +0.125 bits/weight)</para>
    /// <para><b>Typical group sizes:</b> 32, 64, 128, 256 (smaller = more accurate, more overhead)</para>
    /// <para><b>Best for:</b> 4-bit and 3-bit quantization where accuracy is critical</para>
    /// <para><b>Used by:</b> GPTQ (default 128), AWQ (default 128), bitsandbytes QLoRA</para>
    /// </remarks>
    PerGroup,

    /// <summary>
    /// Per-block quantization with super-blocks (K-Quant style from llama.cpp).
    /// Two-level scheme: small blocks grouped into super-blocks with additional scaling.
    /// Achieves near per-group accuracy with lower overhead.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A clever two-level approach: small groups have their own scales,
    /// and groups of groups share a "super-scale". Like having local and regional managers.</para>
    /// <para><b>Structure:</b> Super-block (e.g., 256 elements) contains multiple blocks (e.g., 32 elements each),
    /// with FP16 scale for super-block and smaller scales for each block.</para>
    /// <para><b>Best for:</b> Q4_K_M and Q5_K_M style quantization (llama.cpp)</para>
    /// <para><b>Storage:</b> Slightly less than pure PerGroup with similar accuracy</para>
    /// </remarks>
    PerBlock,

    /// <summary>
    /// Row-wise quantization - separate scale per row of a weight matrix.
    /// Useful for fully-connected layers where each output has different characteristics.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each row (corresponding to one output neuron) gets its own scale.
    /// Similar to PerChannel but specifically for the row dimension of matrices.</para>
    /// <para><b>Best for:</b> Dense/Linear layers, transformer attention projections</para>
    /// </remarks>
    PerRow
}
