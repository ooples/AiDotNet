namespace AiDotNet.Enums;

/// <summary>Classifies observer notifications emitted by the engine.</summary>
/// <remarks>
/// <para>
/// Every <see cref="AiDotNet.Evolution.EvolutionEvent{TGenome}"/> carries one of these kinds plus a monotonically
/// increasing sequence number. <see cref="Proposed"/> is raised once per candidate after refinement and
/// canonicalization, before the duplicate and cache checks. <see cref="Evaluated"/> is raised exactly once per
/// candidate for every terminal <see cref="EvolutionEvaluationStatus"/>, including duplicates, cache hits,
/// rejections, and failures, and carries the evaluation together with the archive insertion result when an
/// insertion was attempted. <see cref="ArchiveChanged"/> follows an <see cref="Evaluated"/> event only when that
/// result was <see cref="EvolutionArchiveInsertionResult.Inserted"/>,
/// <see cref="EvolutionArchiveInsertionResult.Replaced"/>, or
/// <see cref="EvolutionArchiveInsertionResult.InsertedWithEviction"/>. <see cref="Migrated"/>,
/// <see cref="Checkpointed"/>, and <see cref="Stopped"/> carry no candidate; their message holds the number of
/// elite transfers, the checkpoint sequence, and the <see cref="EvolutionStopReason"/> name respectively.
/// </para>
/// <para><b>For Beginners:</b> Think of these as the headlines an evolution run can report while it works. If you
/// only want to know when the search got better, listen for <see cref="ArchiveChanged"/>; if you want a full audit
/// trail of every candidate that was tried and how it ended (completed, duplicate, timed out, and so on), listen
/// for <see cref="Evaluated"/>. <see cref="Checkpointed"/> tells you a resumable snapshot was safely written, and
/// <see cref="Stopped"/> is the final event of a run that ends normally or by cancellation and says why it ended
/// (budget exhausted, time limit, canceled, or a fail-fast failure). A simple console observer might print one
/// line per <see cref="ArchiveChanged"/> event and a summary when <see cref="Stopped"/> arrives.</para>
/// </remarks>
public enum EvolutionEventKind
{
    /// <summary>A candidate was proposed and assigned an identity.</summary>
    Proposed = 0,
    /// <summary>An evaluation reached a terminal status.</summary>
    Evaluated = 1,
    /// <summary>An archive accepted or replaced an elite.</summary>
    ArchiveChanged = 2,
    /// <summary>Migration copied elites between islands.</summary>
    Migrated = 3,
    /// <summary>A checkpoint was durably published.</summary>
    Checkpointed = 4,
    /// <summary>The run stopped.</summary>
    Stopped = 5
}
