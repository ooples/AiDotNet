namespace AiDotNet.Enums;

/// <summary>Identifies whether an evaluation came from the task or a deterministic cache.</summary>
/// <remarks>
/// <para>
/// When evaluation caching is enabled, the engine remembers each completed task result under its canonical genome
/// identity. A later candidate with the same identity is served from that memory with zero cost units and is
/// marked <see cref="Hit"/>; a canonical identity not yet seen is marked <see cref="Miss"/> and evaluated normally.
/// <see cref="NotChecked"/> is used when caching is disabled or when the candidate never reached the lookup, for
/// example because it was rejected as a duplicate or failed during preparation. The cache is keyed purely on
/// canonical identity, so it is only as reliable as the task's canonicalization.
/// </para>
/// <para><b>For Beginners:</b> Training the same model configuration twice wastes time, so the engine can keep a
/// notebook of configurations it has already scored. This value tells you, for one evaluation, whether that
/// notebook was consulted and whether it had the answer. It matters when you read timing and cost figures: a
/// <see cref="Hit"/> reports the stored score but charges no cost units and essentially no elapsed time, while a
/// <see cref="Miss"/> reflects real work. If you see <see cref="NotChecked"/> everywhere, caching is turned off for
/// the run.</para>
/// </remarks>
public enum EvolutionCacheStatus
{
    /// <summary>No cache lookup was performed.</summary>
    NotChecked = 0,
    /// <summary>The cache did not contain a reusable evaluation.</summary>
    Miss = 1,
    /// <summary>A prior completed evaluation was reused.</summary>
    Hit = 2
}
