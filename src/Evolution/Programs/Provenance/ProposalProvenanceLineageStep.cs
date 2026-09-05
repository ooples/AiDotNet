using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>One accepted edit in a reconstructed lineage: the parent, the child, and the exchange that joined them.</summary>
/// <remarks>
/// <para>
/// A lineage is a chain of these, oldest first. Each step names the two programs it connects and carries the whole
/// provenance record for the request that produced the child, so the prompt, the answer, the model, the cost, and
/// the timing are all attached to the transition rather than kept in a parallel structure. That is the shape the
/// reference implementation assembles as an "action" attached to each improvement step, and it is what makes a
/// lineage directly usable as a training trajectory.
/// </para>
/// <para><b>For Beginners:</b> One rung on the ladder from the first program to the final one: "program A became
/// program B, and here is the conversation with the AI that did it".</para>
/// </remarks>
public sealed class ProposalProvenanceLineageStep
{
    /// <summary>Initializes one lineage step.</summary>
    /// <param name="stepIndex">The zero-based position of this step within its lineage, oldest first.</param>
    /// <param name="record">The provenance record for the accepted request that produced the child.</param>
    /// <exception cref="ArgumentNullException"><paramref name="record"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="stepIndex"/> is negative.</exception>
    public ProposalProvenanceLineageStep(int stepIndex, ProposalProvenanceRecord record)
    {
        Guard.NotNull(record);
        Guard.NonNegative(stepIndex);
        StepIndex = stepIndex;
        Record = record;
    }

    /// <summary>Gets the zero-based position of this step within its lineage.</summary>
    public int StepIndex { get; }

    /// <summary>Gets the full provenance record for the request that produced the child.</summary>
    public ProposalProvenanceRecord Record { get; }

    /// <summary>Gets the canonical identity of the program this step started from.</summary>
    public string ParentGenomeId => Record.ParentGenomeId;

    /// <summary>Gets the canonical identity of the program this step produced.</summary>
    public string ChildGenomeId => Record.ChildGenomeId;

    /// <summary>Returns the step position and the two program identities it connects.</summary>
    /// <returns>A short description carrying no prompt text or program source.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "Step {0}: {1} -> {2}",
        StepIndex,
        ParentGenomeId,
        ChildGenomeId);
}
