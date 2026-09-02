using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Allows an archive whose descriptor ranges widen during a run to checkpoint and restore those ranges.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A descriptor configured with <see cref="AiDotNet.Enums.EvolutionOutOfRangePolicy.Grow"/> extends its range whenever
/// a committed evaluation reports a value outside it, so the grid an archive ends a run with is not the grid it was
/// constructed with. That widened grid is state, and a resumed run has to start from it: rebinning the checkpointed
/// elites against the original narrow ranges would reject them. The engine therefore reads
/// <see cref="IEvolutionArchiveView{TGenome}.Descriptors"/> when it writes a checkpoint and calls
/// <see cref="RestoreDescriptorBounds"/> before <see cref="ICheckpointableEvolutionArchive{TGenome}.Restore"/> when it
/// reads one back, which also removes any dependence on the order in which the restored entries happen to be replayed.
/// </para>
/// <para>
/// Implementations must accept only descriptors that keep the archive's identity: the same count, the same names in the
/// same order, the same policies, and ranges that contain the configured ones on a whole number of bins of the original
/// width. Anything else indicates a checkpoint written by a differently configured archive and must be refused, because
/// the archive definition hash deliberately records the ORIGINAL bounds so that growth alone never invalidates a resume.
/// </para>
/// <para><b>For Beginners:</b> If you let the archive's grid grow to fit surprising values, the saved game has to
/// remember how big the grid became, or reloading it would throw away everything that landed in the new rows. This
/// interface is that memory. You only need it if you write your own growable archive; the built-in
/// <see cref="MapElitesArchive{TGenome}"/> already implements it, and archives with fixed ranges never need it.</para>
/// </remarks>
public interface IGrowableEvolutionArchive<TGenome> : ICheckpointableEvolutionArchive<TGenome>
{
    /// <summary>Adopts previously grown descriptor ranges into an empty archive.</summary>
    /// <param name="descriptors">The descriptor definitions captured at checkpoint time, in their original order.</param>
    /// <exception cref="ArgumentNullException"><paramref name="descriptors"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">The archive already holds entries or has a non-zero version.</exception>
    /// <exception cref="InvalidDataException">
    /// The supplied descriptors are not a widening of the configured ones.
    /// </exception>
    void RestoreDescriptorBounds(IReadOnlyList<EvolutionDescriptorDefinition> descriptors);
}
