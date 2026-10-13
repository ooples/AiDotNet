namespace AiDotNet.Enums;

/// <summary>
/// Identifies which attention kernel a layer's most recent forward pass actually executed.
/// </summary>
/// <remarks>
/// <para>
/// The same attention block can be computed by more than one kernel, and the choice is a
/// memory decision rather than a mathematical one: the fused FlashAttention kernel never
/// returns the <c>[batch, heads, seq_q, seq_kv]</c> score matrix, so a forward that needs
/// those scores (the auxiliary losses, or a manual tape-free backward) has to take the
/// score-materializing kernel instead.
/// </para>
/// <para>
/// Reporting the path a forward took makes that decision observable, so a regression that
/// silently reverts a layer to the expensive kernel is a failing assertion rather than a
/// quiet slowdown (#2312).
/// </para>
/// <para><b>For Beginners:</b> Attention can be computed two ways that give the same answer.
/// One of them also hands back a big table of "how much each token looked at each other
/// token"; that table is useful for diagnostics but costs a lot of memory on long sequences.
/// This enum records which of the two was used last.
/// </para>
/// </remarks>
public enum AttentionKernelPath
{
    /// <summary>
    /// No attention forward has run yet on this layer.
    /// </summary>
    None = 0,

    /// <summary>
    /// The fused FlashAttention kernel (<c>IEngine.FlashAttention</c>), which keeps one
    /// log-sum-exp per query row instead of the full score matrix and masks causally in-kernel.
    /// </summary>
    FusedFlashAttention,

    /// <summary>
    /// The score-materializing kernel (<c>IEngine.ScaledDotProductAttention</c>), which returns
    /// the attention weights because something downstream reads them.
    /// </summary>
    ScoreMaterializing,

    /// <summary>
    /// The additive-bias path used when a layer injects an ALiBi positional bias.
    /// </summary>
    AdditiveBias,
}
