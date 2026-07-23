namespace AiDotNet.Augmentation;

/// <summary>
/// Non-generic holder for the process-wide once-per-run latches that gate
/// the ConfigureAugmentation informational messages.
/// </summary>
/// <remarks>
/// <para>
/// Lives on a non-generic class so multiple closed-generic
/// instantiations of <c>AiModelBuilder&lt;T, TInput, TOutput&gt;</c>
/// genuinely share the latch. Putting the static int directly on the
/// generic class would give every closed type its own static slot
/// (e.g. <c>AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;</c>
/// and <c>AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;</c>
/// would each emit the warning once) — defeating the
/// once-per-process intent.
/// </para>
/// <para>
/// Mutated only via <see cref="System.Threading.Interlocked.Exchange(ref int, int)"/>
/// from the augmentation-apply block of <c>BuildSupervisedInternalAsync</c>.
/// Exposed as fields (not properties) because <c>Interlocked.Exchange</c>
/// requires a <c>ref</c>-able lvalue (review #1368 C7HAP).
/// </para>
/// </remarks>
internal static class AugmentationWarningLatch
{
    /// <summary>Latch for "offline augmentation applied once" message.</summary>
    public static int OfflineEmitted;

    /// <summary>Latch for "X-only, not labels" reminder message.</summary>
    public static int XOnlyEmitted;
}
