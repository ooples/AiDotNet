namespace AiDotNet.Enums;

/// <summary>
/// Specifies the attention variant that determines how Key/Value heads map to Query heads.
/// </summary>
/// <remarks>
/// <para>
/// Modern transformer architectures use different ratios of Query heads to Key/Value heads
/// to trade off between model quality and memory efficiency during inference.
/// </para>
/// <para><b>For Beginners:</b> In standard attention, each "head" has its own Query, Key, and Value.
///
/// Different variants change how many Key/Value heads are shared:
/// - <b>MultiHead (MHA):</b> Each head has its own K and V (standard, most memory)
/// - <b>MultiQuery (MQA):</b> All heads share one K and V (least memory, used by PaLM)
/// - <b>GroupedQuery (GQA):</b> Groups of heads share K and V (balanced, used by Llama 2/3)
///
/// GQA with numKVHeads=8 and numHeads=64 means each K/V head serves 8 Query heads,
/// reducing KV-cache memory by 8x compared to standard MHA.
/// </para>
/// </remarks>
internal enum AttentionVariant
{
    /// <summary>
    /// Standard Multi-Head Attention where numKVHeads == numHeads.
    /// </summary>
    /// <remarks>
    /// Every head has independent Query, Key, and Value projections.
    /// This is the original attention mechanism from Vaswani et al., 2017.
    /// </remarks>
    MultiHead,

    /// <summary>
    /// Multi-Query Attention where numKVHeads == 1 (Shazeer, 2019).
    /// </summary>
    /// <remarks>
    /// All Query heads share a single Key and Value projection.
    /// Maximizes KV-cache memory savings at the cost of some model quality.
    /// Used by PaLM, Falcon (some variants).
    /// </remarks>
    MultiQuery,

    /// <summary>
    /// Grouped-Query Attention where 1 &lt; numKVHeads &lt; numHeads (Ainslie et al., 2023).
    /// </summary>
    /// <remarks>
    /// Groups of Query heads share Key and Value projections.
    /// Provides a configurable trade-off between MHA quality and MQA efficiency.
    /// Used by Llama 2 70B (numKVHeads=8), Llama 3, Mistral.
    /// </remarks>
    GroupedQuery
}
