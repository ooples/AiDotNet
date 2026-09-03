using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>The result of routing a reply's edit blocks to the program or to the changes description.</summary>
/// <remarks>
/// <para>
/// A run that maintains a changes description shows the model two documents and asks it to edit both, but a
/// SEARCH/REPLACE reply says only what text to find, never where. Routing decides that: a block whose SEARCH text
/// occurs in exactly one of the two belongs to that one. A block that occurs in both is refused rather than guessed,
/// because applying it to the wrong document is an edit nobody asked for that still looks like success.
/// </para>
/// <para><b>For Beginners:</b> The model is editing a program and a short note describing its own changes. Each
/// edit says "find this text, replace it with that". This works out which of the two documents each edit meant.</para>
/// </remarks>
public sealed class ProgramDiffTargetSplit
{
    private readonly ReadOnlyCollection<ProgramDiffBlock> _programBlocks;
    private readonly ReadOnlyCollection<ProgramDiffBlock> _descriptionBlocks;
    private readonly ReadOnlyCollection<ProgramDiffFailure> _failures;

    /// <summary>Initializes a routing result.</summary>
    /// <param name="programBlocks">The blocks that edit the program.</param>
    /// <param name="descriptionBlocks">The blocks that edit the changes description.</param>
    /// <param name="failures">A failure for each block whose target could not be settled.</param>
    /// <exception cref="ArgumentNullException">An argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A collection contains a <c>null</c> entry.</exception>
    public ProgramDiffTargetSplit(
        IReadOnlyList<ProgramDiffBlock> programBlocks,
        IReadOnlyList<ProgramDiffBlock> descriptionBlocks,
        IReadOnlyList<ProgramDiffFailure> failures)
    {
        Guard.NotNull(programBlocks);
        Guard.NotNull(descriptionBlocks);
        Guard.NotNull(failures);
        if (programBlocks.Any(block => block is null) || descriptionBlocks.Any(block => block is null) ||
            failures.Any(failure => failure is null))
        {
            throw new ArgumentException("A routing result cannot contain null entries.", nameof(programBlocks));
        }

        _programBlocks = Array.AsReadOnly(programBlocks.ToArray());
        _descriptionBlocks = Array.AsReadOnly(descriptionBlocks.ToArray());
        _failures = Array.AsReadOnly(failures.ToArray());
    }

    /// <summary>Gets the blocks that edit the program, in reply order.</summary>
    public IReadOnlyList<ProgramDiffBlock> ProgramBlocks => _programBlocks;

    /// <summary>Gets the blocks that edit the changes description, in reply order.</summary>
    public IReadOnlyList<ProgramDiffBlock> DescriptionBlocks => _descriptionBlocks;

    /// <summary>Gets a failure for each block whose target could not be settled.</summary>
    public IReadOnlyList<ProgramDiffFailure> Failures => _failures;

    /// <summary>Gets whether every block was routed to exactly one target.</summary>
    public bool IsSuccess => _failures.Count == 0;
}
