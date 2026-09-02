namespace AiDotNet.Enums;

/// <summary>Says what caused a best-program snapshot to be written to disk.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve controller writes the best program twice over a run's lifetime and in two different
/// places: <c>_save_checkpoint</c> writes it into the checkpoint directory it just created, and
/// <c>_save_best_program</c> writes it into a single <c>best/</c> directory when the run finishes. The two carry
/// slightly different metadata - the checkpoint copy records which iteration it was taken at, the final copy
/// records the parent program - so a reader needs to know which one is in front of it.
/// </para>
/// <para><b>For Beginners:</b> The best program found so far gets saved periodically during the run and once more
/// when the run ends. This says which of those two a particular saved file came from, so you can tell a mid-run
/// snapshot from the final answer.</para>
/// </remarks>
public enum ProgramRunOutputTrigger
{
    /// <summary>Written alongside a checkpoint, as a snapshot of the best program at that point.</summary>
    Checkpoint = 0,

    /// <summary>Written when the run stopped, as the run's final answer.</summary>
    RunEnd = 1,

    /// <summary>Written because a caller asked for it directly rather than in response to a run event.</summary>
    Manual = 2
}
