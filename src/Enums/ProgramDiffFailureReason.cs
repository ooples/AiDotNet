namespace AiDotNet.Enums;

/// <summary>Names why one SEARCH/REPLACE block could not be parsed or applied.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation drops a block whose SEARCH text is absent from the parent program
/// without recording anything, so an iteration can silently produce a child identical to its parent and still
/// consume evaluator budget. Every rejection here is instead reported as a typed reason attached to the offending
/// block, which lets a caller decide between retrying with targeted feedback, falling back to a full rewrite, or
/// abandoning the proposal before any evaluation is paid for.
/// </para>
/// <para><b>For Beginners:</b> A language model edits code by sending small "find this text, replace it with that
/// text" instructions. Sometimes those instructions are unusable: the text to find is missing from the file, the
/// instruction itself is cut in half, or it tries to change a part of the file that is off limits. This enum is
/// the short label for each of those situations. Because the label is machine-readable you can write code that
/// reacts differently to "your search text was not found" than to "you sent a broken instruction", and you can
/// tell the model exactly what to fix.</para>
/// </remarks>
public enum ProgramDiffFailureReason
{
    /// <summary>The block markers were incomplete, out of order, or nested, so no usable pair was recovered.</summary>
    MalformedBlock = 0,

    /// <summary>The SEARCH section was empty or whitespace only, which would match an arbitrary blank line.</summary>
    EmptySearchText = 1,

    /// <summary>The SEARCH text does not occur in the target program, so the block cannot be applied.</summary>
    SearchTextNotFound = 2,

    /// <summary>The SEARCH text matched outside every evolve block while out-of-block edits were forbidden.</summary>
    OutsideEvolveBlock = 3,

    /// <summary>The response contained more blocks than the configured maximum, so the surplus was discarded.</summary>
    BlockLimitExceeded = 4,

    /// <summary>The response contained no SEARCH/REPLACE block at all.</summary>
    NoBlocksFound = 5,

    /// <summary>Every block parsed, but applying them left the program byte-identical to the original.</summary>
    ResultUnchanged = 6,

    /// <summary>The response contained a carriage return while strict line-feed-only parsing was requested.</summary>
    CarriageReturnRejected = 7,

    /// <summary>The SEARCH text occurs in both the program and the changes description, so its target is unclear.</summary>
    /// <remarks>
    /// Only possible when a run maintains a changes description alongside the program, because only then does a reply
    /// have two places to edit. Applying such a block to a guess would silently edit the wrong one, so it is refused
    /// and the model is told to make the SEARCH text unique to whichever it meant.
    /// </remarks>
    AmbiguousTarget = 8
}
