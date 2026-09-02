using System.Collections.ObjectModel;

namespace AiDotNet.Evolution.Programs;

/// <summary>The program produced by applying a set of edit blocks, plus every block that did not apply.</summary>
/// <remarks>
/// <para>
/// <see cref="ModifiedSource"/> always holds a usable program: blocks that matched were applied and blocks that
/// did not were left out, so a caller that only wants best effort can use it directly. A caller that wants
/// correctness checks <see cref="IsSuccess"/>, which is <c>false</c> when any block failed, when nothing applied,
/// or when the result came back byte identical to the original while
/// <see cref="AiDotNet.Configuration.ProgramDiffOptions.RejectWhenNoBlockApplied"/> is set. That last check is what
/// stops an evolution run from spending evaluator budget on a child that is a copy of its parent, which the
/// reference implementation does routinely because it never notices that zero blocks applied.
/// </para>
/// <para><b>For Beginners:</b> This is the result of carrying out a model's edit instructions. It gives you the
/// edited program, a count of how many instructions actually took effect, and a list of the ones that did not.
/// <see cref="UnifiedDiff"/> is a human-readable summary of what changed, in the familiar format used by version
/// control tools, which is handy both for logging and for showing the model what its edit really did.</para>
/// </remarks>
public sealed class ProgramDiffApplyResult
{
    private readonly ReadOnlyCollection<ProgramDiffFailure> _failures;

    internal ProgramDiffApplyResult(
        string originalSource,
        string modifiedSource,
        int appliedCount,
        IReadOnlyList<ProgramDiffFailure> failures,
        bool isSuccess)
    {
        OriginalSource = originalSource;
        ModifiedSource = modifiedSource;
        AppliedCount = appliedCount;
        var failureCopy = new ProgramDiffFailure[failures.Count];
        for (int index = 0; index < failures.Count; index++) failureCopy[index] = failures[index];
        _failures = Array.AsReadOnly(failureCopy);
        IsSuccess = isSuccess;
    }

    /// <summary>Gets the program text the blocks were applied to.</summary>
    public string OriginalSource { get; }

    /// <summary>Gets the program text after every block that matched was applied.</summary>
    public string ModifiedSource { get; }

    /// <summary>Gets the number of blocks that matched and were applied.</summary>
    public int AppliedCount { get; }

    /// <summary>Gets every block that could not be applied, with a typed reason.</summary>
    public IReadOnlyList<ProgramDiffFailure> Failures => _failures;

    /// <summary>Gets whether every block applied and the program actually changed.</summary>
    public bool IsSuccess { get; }

    /// <summary>Gets whether the applied edits left the program byte identical to the original.</summary>
    public bool IsUnchanged => string.Equals(OriginalSource, ModifiedSource, StringComparison.Ordinal);

    /// <summary>Renders the change as a unified diff with three lines of context.</summary>
    /// <returns>A unified diff, or an empty string when nothing changed.</returns>
    public string UnifiedDiff => ProgramDiff.CreateUnifiedDiff(OriginalSource, ModifiedSource);
}
