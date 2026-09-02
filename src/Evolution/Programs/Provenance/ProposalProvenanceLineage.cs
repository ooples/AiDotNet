using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>The full ancestry of one program, rebuilt from a recorded provenance stream.</summary>
/// <remarks>
/// <para>
/// A lineage is the chain of accepted edits that turned a seed program into a final one, oldest first, with the
/// language-model exchange that caused each edit attached to it. It is the answer to "where did this program come
/// from?", assembled after the fact from nothing but the recorded stream — no live engine, no archive, no
/// checkpoint required. The same reconstruction is what the reference implementation performs over the per-program
/// JSON files in a saved checkpoint directory, and it exists for the same two reasons: explaining a finished run,
/// and harvesting successful trajectories as training data.
/// </para>
/// <para>
/// Only accepted steps appear in the chain, because only they moved the program forward. The failed attempts are
/// still in the stream and are still worth reading — they are what a rejection-sampling or preference-training
/// dataset needs — but they are not ancestry, so they are not links in this chain.
/// </para>
/// <para><b>For Beginners:</b> Read this as the family tree of one program, from its earliest ancestor down to
/// itself, with a transcript of the AI conversation that produced each generation. <see cref="Depth"/> tells you
/// how many successful improvements it took to get there.</para>
/// </remarks>
public sealed class ProposalProvenanceLineage
{
    private readonly ProposalProvenanceLineageStep[] _steps;

    /// <summary>Initializes a lineage.</summary>
    /// <param name="finalGenomeId">The canonical identity of the program at the end of the chain.</param>
    /// <param name="steps">The accepted edits, oldest first.</param>
    /// <exception cref="ArgumentNullException"><paramref name="finalGenomeId"/> or <paramref name="steps"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="finalGenomeId"/> is empty or white space, or <paramref name="steps"/> contains a <c>null</c> entry.
    /// </exception>
    public ProposalProvenanceLineage(string finalGenomeId, IReadOnlyList<ProposalProvenanceLineageStep> steps)
    {
        Guard.NotNullOrWhiteSpace(finalGenomeId);
        Guard.NotNull(steps);

        var copy = new ProposalProvenanceLineageStep[steps.Count];
        for (int index = 0; index < steps.Count; index++)
        {
            ProposalProvenanceLineageStep step = steps[index];
            if (step is null)
            {
                throw new ArgumentException("A lineage cannot contain a null step.", nameof(steps));
            }

            copy[index] = step;
        }

        _steps = copy;
        FinalGenomeId = finalGenomeId.Trim();
    }

    /// <summary>Gets the canonical identity of the program at the end of the chain.</summary>
    public string FinalGenomeId { get; }

    /// <summary>Gets the canonical identity of the earliest ancestor, or the final program when there are no steps.</summary>
    public string RootGenomeId => _steps.Length == 0 ? FinalGenomeId : _steps[0].ParentGenomeId;

    /// <summary>Gets the accepted edits, oldest first.</summary>
    public IReadOnlyList<ProposalProvenanceLineageStep> Steps =>
        new ReadOnlyCollection<ProposalProvenanceLineageStep>(_steps);

    /// <summary>Gets how many accepted edits separate the root from the final program.</summary>
    public int Depth => _steps.Length;

    /// <summary>Gets the prompt and answer tokens every step in this lineage cost, as the provider reported them.</summary>
    public long TotalTokens
    {
        get
        {
            long total = 0L;
            foreach (ProposalProvenanceLineageStep step in _steps) total += step.Record.TotalTokens;
            return total;
        }
    }

    /// <summary>Returns the final program, the root, and the chain length.</summary>
    /// <returns>A short description carrying no prompt text or program source.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "ProposalProvenanceLineage({0} <- {1}, depth {2})",
        FinalGenomeId,
        RootGenomeId,
        Depth);
}
