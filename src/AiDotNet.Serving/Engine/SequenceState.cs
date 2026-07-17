namespace AiDotNet.Serving.Engine;

/// <summary>
/// Lifecycle state of a <see cref="Sequence"/> inside the inference engine. Mirrors the states an in-flight
/// batching engine (vLLM, TGI) tracks so the scheduler can move sequences between the waiting queue, the
/// running batch, and (under memory pressure) CPU-swapped or preempted storage.
/// </summary>
public enum SequenceState
{
    /// <summary>Admitted but not yet scheduled — waiting for KV-cache blocks / a batch slot.</summary>
    Waiting,

    /// <summary>Currently in the running batch and being decoded each engine step.</summary>
    Running,

    /// <summary>Preempted under KV memory pressure and swapped to CPU (blocks can be swapped back in).</summary>
    Swapped,

    /// <summary>Preempted by RECOMPUTE — its KV was dropped and will be recomputed from tokens when rescheduled.</summary>
    Preempted,

    /// <summary>Finished because a stop token / EOS was produced.</summary>
    FinishedStopped,

    /// <summary>Finished because it reached its max token budget.</summary>
    FinishedLengthCapped,

    /// <summary>Finished because the caller aborted it (or the request was cancelled).</summary>
    FinishedAborted,
}

/// <summary>Convenience helpers for <see cref="SequenceState"/>.</summary>
public static class SequenceStateExtensions
{
    /// <summary>True for any terminal state (finished/aborted).</summary>
    public static bool IsFinished(this SequenceState state)
        => state is SequenceState.FinishedStopped
            or SequenceState.FinishedLengthCapped
            or SequenceState.FinishedAborted;
}
