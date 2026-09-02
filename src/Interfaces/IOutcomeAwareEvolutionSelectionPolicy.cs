using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Optional state and outcome feedback implemented by adaptive selection policies.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A plain <see cref="ISelectionPolicy{TGenome}"/> is stateless: it only reads the archive when a parent is
/// requested. A policy that also implements this interface learns from what happened to the candidates it
/// selected. The engine calls <see cref="Observe"/> exactly once per committed evaluation, in commit order,
/// passing <c>null</c> for the insertion result when the evaluation did not complete. It also includes
/// <see cref="CaptureState"/> in every checkpoint and in the run's deterministic state hash, and it calls
/// <see cref="RestoreState"/> before proposing the first candidate of a resumed run, so pausing and resuming
/// cannot change the trajectory of the search.
/// </para>
/// <para><b>For Beginners:</b> A selection policy decides which existing solutions become the "parents" of the
/// next candidates. Most policies choose without remembering anything, but an adaptive policy keeps a small
/// memory, such as "parents whose children recently improved the archive deserve more turns". This interface is
/// how the engine feeds that memory: after every evaluation it reports what happened, and when it saves a
/// checkpoint it asks the policy to write its memory down as a string so the run can stop and later continue
/// exactly where it left off. Implement it only when your policy keeps state; a stateless policy needs nothing
/// beyond <see cref="ISelectionPolicy{TGenome}"/>. <see cref="CuriosityEvolutionSelectionPolicy{TGenome}"/> is
/// the built-in example.</para>
/// <para>
/// Determinism contract: <see cref="CaptureState"/> must return byte-identical text for identical policy state
/// (order dictionary keys with <see cref="StringComparer.Ordinal"/> and format numbers with the invariant
/// culture), and <see cref="RestoreState"/> must validate its input and throw
/// <see cref="System.IO.InvalidDataException"/> for a corrupt or out-of-range payload rather than accept it.
/// </para>
/// </remarks>
public interface IOutcomeAwareEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <summary>Updates policy state after one evaluation is committed.</summary>
    /// <param name="evaluation">The committed evaluation, including its lineage (parent and inspiration identifiers).</param>
    /// <param name="insertionResult">
    /// How the archive responded to the evaluation, or <c>null</c> when the evaluation did not complete and was
    /// therefore never offered to an archive.
    /// </param>
    void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult);

    /// <summary>Captures deterministic policy state for a checkpoint.</summary>
    /// <returns>
    /// A culture-independent string that <see cref="RestoreState"/> can consume; identical state must yield identical text.
    /// </returns>
    string CaptureState();

    /// <summary>Restores deterministic policy state from a checkpoint.</summary>
    /// <param name="state">Text previously produced by <see cref="CaptureState"/> for a compatible policy version.</param>
    /// <exception cref="System.IO.InvalidDataException">The payload is malformed or violates the policy's invariants.</exception>
    void RestoreState(string state);
}
