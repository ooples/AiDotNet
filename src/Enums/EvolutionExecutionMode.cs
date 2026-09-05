namespace AiDotNet.Enums;

/// <summary>Describes how completed worker results are committed to evolution state.</summary>
/// <remarks>
/// <para>
/// The engine evaluates each logical batch of proposals with up to
/// <see cref="AiDotNet.Configuration.EvolutionEngineOptions.MaxDegreeOfParallelism"/> concurrent evaluator calls,
/// waits for the whole batch, and then commits the results one at a time: updating counters, offering completed
/// candidates to their island archive, notifying the selection policy, and retaining failure diagnostics. The
/// execution mode chooses the commit order inside that batch. Order matters because archive replacement,
/// deterministic capacity eviction, fail-fast handling, and adaptive selection policies all react to results
/// sequentially, so a different order can produce a different final archive and a different <c>StateHash</c>.
/// </para>
/// <para><b>For Beginners:</b> Imagine eight graders marking eight exam papers at the same time. In
/// <see cref="Deterministic"/> mode the marks are entered into the register strictly in paper-number order, even if
/// paper 7 was finished before paper 3, so running the same seed again on any machine, with any number of graders,
/// produces exactly the same register. In <see cref="Opportunistic"/> mode marks are entered in the order the graders
/// finished, which mirrors what an observer watching the run saw happen, but two runs with the same seed can now end
/// differently because a faster machine finishes papers in a different order. Keep the default,
/// <see cref="Deterministic"/>, whenever you need reproducible experiments, meaningful state-hash comparisons, or
/// resumable checkpoints; choose <see cref="Opportunistic"/> only when reproducibility is not required and you prefer
/// commits to follow the order in which results actually arrived.</para>
/// <para>
/// Both modes sort a batch of <c>b</c> results in O(b log b) time; neither changes how many evaluations run or how
/// much memory is held, because commits always happen after the batch is fully evaluated.
/// </para>
/// </remarks>
public enum EvolutionExecutionMode
{
    /// <summary>Commit in evaluation-ID order so worker timing cannot change the result.</summary>
    Deterministic = 0,
    /// <summary>Commit in worker-completion order (ties broken by evaluation ID); this can improve responsiveness but is schedule-dependent.</summary>
    Opportunistic = 1
}
