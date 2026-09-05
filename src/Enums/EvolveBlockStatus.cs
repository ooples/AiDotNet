namespace AiDotNet.Enums;

/// <summary>Reports how well a program source matched the configured evolve-block markers.</summary>
/// <remarks>
/// <para>
/// Program evolution usually rewrites only a marked region of a file so that imports, harness code, and the
/// evaluator entry point survive every generation. Extraction therefore has to describe not just "found" or
/// "not found" but the exact way a marker pair was malformed, because an LLM that emitted a stray or missing
/// marker can be given precise feedback and asked again. When several anomalies occur in one file the most
/// severe status is reported and every individual anomaly is listed separately as a diagnostic, in the order the
/// lines appear.
/// </para>
/// <para><b>For Beginners:</b> An "evolve block" is the part of a file that a program-evolution run is allowed to
/// change; it is marked with a start comment and an end comment, a bit like the fenced-off area on a building
/// site. This enum is the answer to "did I find a properly fenced-off area?". <see cref="Complete"/> means yes,
/// <see cref="NotPresent"/> means the file has no fence at all, and the remaining values name a specific way the
/// fence was broken, such as a start marker with no matching end marker. Checking this value tells you whether it
/// is safe to rewrite just the block or whether you should ask for the code again.</para>
/// </remarks>
public enum EvolveBlockStatus
{
    /// <summary>No start marker appears anywhere in the source.</summary>
    NotPresent = 0,

    /// <summary>Every start marker was matched by an end marker and no anomaly was detected.</summary>
    Complete = 1,

    /// <summary>An end marker appeared while no block was open; the stray marker was ignored.</summary>
    UnmatchedEnd = 2,

    /// <summary>A second start marker appeared before the open block was closed, discarding the partial block.</summary>
    RestartedBlock = 3,

    /// <summary>A start marker was never followed by an end marker, so the trailing block was discarded.</summary>
    Unterminated = 4
}
