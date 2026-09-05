using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Receives the engine's cross-island elite index before each parent selection.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A plain <see cref="ISelectionPolicy{TGenome}"/> only ever sees the archive of the island a proposal targets, which
/// makes a cross-island exploitation branch impossible to express. A policy that also implements this interface is
/// handed the engine's global elite index, ordered best first, immediately before every
/// <see cref="ISelectionPolicy{TGenome}.Select"/> call, together with the island the proposal targets. The list is a
/// stable snapshot owned by the engine, so a policy may read it but must not retain or mutate it beyond the call it
/// was given for. When <c>EvolutionEngineOptions.GlobalEliteCount</c> is zero the engine keeps no index and passes an
/// empty list, so an implementation must always have a well-defined behaviour for that case.
/// </para>
/// <para><b>For Beginners:</b> An island model keeps several separate populations, and normally the code that picks
/// parents can only see the one island it is working on. Sometimes you want it to be able to say "start from the
/// single best solution found anywhere in this run", and that needs a view across all the islands. This interface is
/// how the engine offers that view: right before it asks your policy for a parent it says "here are the strongest
/// candidates found so far, and here is the island we are filling". Implement it only if your policy wants that
/// cross-island knowledge; <see cref="RatioEvolutionSelectionPolicy{TGenome}"/> is the built-in example.</para>
/// </remarks>
public interface IEliteIndexAwareEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <summary>Supplies the current global elite snapshot and the island the next selection targets.</summary>
    /// <param name="elites">The engine's global elites in best-first order; empty when the index is disabled.</param>
    /// <param name="island">The zero-based island index the next proposal will be assigned to.</param>
    void UseEliteIndex(IReadOnlyList<EvolutionEliteRecord<TGenome>> elites, int island);
}
