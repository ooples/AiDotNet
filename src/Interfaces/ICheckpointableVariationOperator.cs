namespace AiDotNet.Interfaces;

/// <summary>Lets a variation operator carry its own state through a checkpoint and a resume.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// Most variation operators are pure: given a parent, a random stream and some inspirations, they propose a child
/// and remember nothing. An operator that also learns from the run - one that shows a language model what it tried
/// before and why it was rejected, for instance - is not pure, and that memory is part of what decides the next
/// proposal. If the memory lives only in the process, a resumed run proposes from a different state than an
/// uninterrupted one, and the engine's determinism guarantee stops covering the operator even though it still holds
/// for everything around it.
/// </para>
/// <para>
/// Implementing this closes that hole: the engine folds <see cref="CaptureState"/> into every checkpoint and into
/// the run's state hash, and calls <see cref="RestoreState"/> before the first proposal of a resumed run. It is the
/// same contract <see cref="IOutcomeAwareEvolutionSelectionPolicy{TGenome}"/> uses, for the same reason.
/// </para>
/// <para>
/// Determinism contract: <see cref="CaptureState"/> must return byte-identical text for identical operator state,
/// on any machine and in any culture, and <see cref="RestoreState"/> must refuse a malformed payload rather than
/// accept it. An operator whose state is genuinely empty should return a stable constant, not an empty string that
/// cannot be told apart from "not captured".
/// </para>
/// <para><b>For Beginners:</b> Implement this only if your operator remembers something between proposals. If it
/// looks at the parent and nothing else, you do not need it - the engine already reproduces those proposals
/// exactly.</para>
/// </remarks>
public interface ICheckpointableVariationOperator<TGenome> : IVariationOperator<TGenome>
{
    /// <summary>Captures deterministic operator state for a checkpoint.</summary>
    /// <returns>
    /// A culture-independent string that <see cref="RestoreState"/> can consume; identical state must yield
    /// identical text.
    /// </returns>
    string CaptureState();

    /// <summary>Restores deterministic operator state from a checkpoint.</summary>
    /// <param name="state">Text previously produced by <see cref="CaptureState"/> for a compatible operator version.</param>
    /// <exception cref="System.IO.InvalidDataException">The payload is malformed or violates the operator's invariants.</exception>
    void RestoreState(string state);
}
