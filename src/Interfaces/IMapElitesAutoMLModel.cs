using AiDotNet.AutoML;

namespace AiDotNet.Interfaces;

/// <summary>Adds immutable quality-diversity archive inspection to the standard AutoML contract.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
/// <remarks>
/// <para>
/// A standard <see cref="IAutoMLModel{T,TInput,TOutput}"/> returns one best model. A MAP-Elites search also
/// produces an archive: the best validated specification for every combination of model family and
/// configuration-complexity bin that the search visited. This interface exposes that archive as immutable
/// <see cref="MapElitesAutoMLArchiveEntry"/> values, which hold specifications and scores rather than live trained
/// models, together with a deterministic state hash, so callers can compare two runs, log the diversity that was
/// found, or choose an alternative model family after the fact.
/// </para>
/// <para>
/// <see cref="ArchiveStateHash"/> excludes wall-clock timing and therefore matches across two runs with the same
/// seed, data, candidate models, and search space; it is the cheapest way to assert reproducibility in tests. Both
/// members describe the most recent completed search; the built-in
/// <see cref="MapElitesAutoML{T,TInput,TOutput}"/> returns an empty archive and an empty hash before its first
/// search runs.
/// </para>
/// <para><b>For Beginners:</b> Normal AutoML hands you a single winner. MAP-Elites AutoML also hands you a map of
/// "best in class" runners-up, for example the best small linear model, the best mid-sized tree ensemble, and the
/// best large neural network, each with its validation score. Use this interface when you want more than one
/// answer: perhaps the overall winner is too slow for production and the best entry from a simpler family is a
/// better fit. Select the strategy with <c>AutoMLOptions.SearchStrategy = AutoMLSearchStrategy.MapElites</c> when
/// calling <c>ConfigureAutoML</c> on the model builder, then read <see cref="Archive"/> once the search completes.</para>
/// </remarks>
public interface IMapElitesAutoMLModel<T, TInput, TOutput> : IAutoMLModel<T, TInput, TOutput>
{
    /// <summary>Gets the immutable elite specifications from the most recent completed search.</summary>
    IReadOnlyList<MapElitesAutoMLArchiveEntry> Archive { get; }

    /// <summary>
    /// Gets a deterministic hash of the most recent engine state, excluding wall-clock timing.
    /// </summary>
    string ArchiveStateHash { get; }
}
