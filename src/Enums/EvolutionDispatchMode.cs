namespace AiDotNet.Enums;

/// <summary>Controls whether the engine evaluates in fixed batches or refills workers as evaluations complete.</summary>
/// <remarks>
/// <para>
/// <see cref="Batch"/> proposes <c>ProposalBatchSize</c> candidates, evaluates them together, and commits the whole
/// group as one transaction. It is simple to reason about and checkpoints on a clean boundary, but the group is a
/// barrier: the slowest candidate holds the others' workers idle, and every candidate in the group was proposed from
/// the same archive state, so the last one is proposed from an archive that is a whole batch out of date.
/// </para>
/// <para>
/// <see cref="Continuous"/> keeps a sliding window of evaluations in flight instead. A completed evaluation is
/// committed on its own and a replacement proposal is dispatched immediately, so no worker waits for a straggler and
/// each proposal sees an archive at most one window old rather than one batch old. Determinism is unchanged: under
/// <see cref="EvolutionExecutionMode.Deterministic"/> commits still happen in evaluation-id order, so a run produces
/// the same state hash at any worker count, and the proposal for one evaluation id is prepared only after the id one
/// window earlier has committed, which makes the whole schedule a function of the id sequence rather than of timing.
/// </para>
/// <para><b>For Beginners:</b> Batch mode is like a class where nobody leaves the room until everyone has finished the
/// exercise; continuous mode is like a queue where the next person starts as soon as a seat frees. Continuous mode
/// finishes more work in the same wall-clock time whenever some candidates take much longer than others, which is the
/// normal case when a candidate can time out or fail early. Start with <see cref="Batch"/>, and switch to
/// <see cref="Continuous"/> when your evaluations vary a lot in cost or you are running many workers.</para>
/// </remarks>
public enum EvolutionDispatchMode
{
    /// <summary>Evaluate a fixed group of proposals together and commit the group as one transaction.</summary>
    Batch = 0,

    /// <summary>Keep a window of evaluations in flight, committing and refilling as each one completes.</summary>
    Continuous = 1
}
