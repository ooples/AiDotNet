using Xunit;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// xUnit collection for tests that measure model convergence under a fixed
/// wall-clock training budget (<c>MaxTrainingTimeSeconds</c> or similar).
/// <para>
/// Background: tests like NBEATS's <c>R2_ShouldBePositive_OnTrendData</c>
/// give the optimizer 5 wall-clock seconds to fit a synthetic trend-plus-
/// seasonal signal and assert R² &gt; 0. Under xUnit's default
/// <c>parallelizeTestCollections = true</c> with four threads on a 2-core
/// CI runner, those 5 seconds of wall-clock become ~1.25 seconds of
/// effective CPU — not enough Adam steps to converge, so R² lands at 0.
/// The test passes in isolation (full-CPU 5 s budget) and fails under
/// parallel CPU contention.
/// </para>
/// <para>
/// Membership in this collection (applied via <c>[Collection]</c> on the
/// test class) sets <c>DisableParallelization = true</c>, so the
/// convergence-sensitive test runs serially relative to other tests in
/// the collection. Tests in *other* collections still run in parallel —
/// this only prevents thrashing on the specific "needs its full CPU
/// budget" cases where reduced CPU = missed convergence.
/// </para>
/// <para>
/// This is NOT a timeout hack. The model is trained within the user-
/// specified budget; this collection just guarantees the budget is
/// actually available. The underlying training implementation has also
/// been optimized (true batched forward, reduced per-step allocations)
/// so under-CPU convergence is viable; the collection just prevents the
/// CI runner from starving us below the paper-adherence threshold.
/// </para>
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public class ConvergenceSensitiveCollection
{
    /// <summary>
    /// Name used in <c>[Collection(...)]</c> attributes on test classes
    /// whose success depends on actual CPU availability matching their
    /// declared training time budget.
    /// </summary>
    public const string Name = "ConvergenceSensitive";
}
