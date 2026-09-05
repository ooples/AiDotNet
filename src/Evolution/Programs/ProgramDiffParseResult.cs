using System.Collections.ObjectModel;

namespace AiDotNet.Evolution.Programs;

/// <summary>The blocks recovered from one model response together with every block that was rejected.</summary>
/// <remarks>
/// <para>
/// Parsing is deliberately total: a response with three well-formed blocks and one truncated block yields three
/// entries in <see cref="Blocks"/> and one entry in <see cref="Failures"/>, so a caller can apply what is usable
/// and still tell the model precisely what to fix. That is the difference from the reference implementation, whose
/// regular expression simply does not match a malformed block and reports "no valid diffs found" for the whole
/// response.
/// </para>
/// <para><b>For Beginners:</b> After reading a model's answer, this object says which edit instructions were
/// understood and which were not. <see cref="IsSuccess"/> is <c>true</c> only when at least one instruction was
/// understood and nothing was rejected. If it is <c>false</c>, look at <see cref="Failures"/> to see what went
/// wrong; each entry is short enough to paste straight back into a follow-up prompt.</para>
/// </remarks>
public sealed class ProgramDiffParseResult
{
    private readonly ReadOnlyCollection<ProgramDiffBlock> _blocks;
    private readonly ReadOnlyCollection<ProgramDiffFailure> _failures;

    internal ProgramDiffParseResult(IReadOnlyList<ProgramDiffBlock> blocks, IReadOnlyList<ProgramDiffFailure> failures)
    {
        var blockCopy = new ProgramDiffBlock[blocks.Count];
        for (int index = 0; index < blocks.Count; index++) blockCopy[index] = blocks[index];
        _blocks = Array.AsReadOnly(blockCopy);

        var failureCopy = new ProgramDiffFailure[failures.Count];
        for (int index = 0; index < failures.Count; index++) failureCopy[index] = failures[index];
        _failures = Array.AsReadOnly(failureCopy);
    }

    /// <summary>Gets the well-formed edit blocks, in the order they appeared in the response.</summary>
    public IReadOnlyList<ProgramDiffBlock> Blocks => _blocks;

    /// <summary>Gets every block that could not be parsed, with a typed reason.</summary>
    public IReadOnlyList<ProgramDiffFailure> Failures => _failures;

    /// <summary>Gets whether at least one block parsed and nothing was rejected.</summary>
    public bool IsSuccess => _blocks.Count > 0 && _failures.Count == 0;

    /// <summary>Gets whether at least one usable block was recovered, even if others failed.</summary>
    public bool HasBlocks => _blocks.Count > 0;
}
