namespace AiDotNet.Enums;

/// <summary>Selects the on-disk layout an evolution trace is written in.</summary>
/// <remarks>
/// <para>
/// Both formats carry exactly the same record fields and are read back by <c>EvolutionTraceFile</c> into the same
/// <c>EvolutionTraceRecord</c> objects, so the choice is about how the file is consumed rather than what it contains.
/// <see cref="JsonLines"/> is one self-contained JSON object per line, which streams, appends, greps, and survives a
/// crash with every completed line intact. <see cref="Json"/> is a single document whose <c>records</c> array is
/// written incrementally as the run proceeds, so a crash still leaves every flushed record recoverable; the reader
/// reports <c>IsComplete = false</c> for such a file instead of failing. OpenEvolve also offers <c>json</c> and
/// <c>hdf5</c>, but its <c>flush</c> is a no-op for both (evolution_trace.py:245-251): nothing reaches disk until
/// <c>close</c>, and its buffer is never cleared, so an interrupted run loses every trace and a long run grows without
/// bound. Its <c>hdf5</c> option additionally raises <c>ImportError</c> unless <c>h5py</c> is installed.
/// </para>
/// <para><b>For Beginners:</b> A trace is the diary of an evolution run - one entry per candidate that was evaluated.
/// This setting only decides how that diary is stored. Pick <see cref="JsonLines"/> for anything long-running: each
/// entry is its own line, so you can tail the file while the run is going, load it with pandas, or split it across
/// machines, and an interrupted run still leaves a readable file. Pick <see cref="Json"/> when you want one tidy
/// document to hand to a tool that expects a single JSON object. When in doubt, use <see cref="JsonLines"/>.</para>
/// </remarks>
public enum EvolutionTraceFormat
{
    /// <summary>One JSON object per line, appended as the run proceeds.</summary>
    JsonLines = 0,

    /// <summary>A single JSON document whose record array is written incrementally.</summary>
    Json = 1
}
