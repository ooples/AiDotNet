using AiDotNet.AutoML;

namespace AiDotNet.Interfaces;

/// <summary>Adds immutable quality-diversity archive inspection to the standard AutoML contract.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
public interface IMapElitesAutoMLModel<T, TInput, TOutput> : IAutoMLModel<T, TInput, TOutput>
{
    /// <summary>Gets the immutable elite specifications from the most recent completed search.</summary>
    IReadOnlyList<MapElitesAutoMLArchiveEntry> Archive { get; }

    /// <summary>
    /// Gets a deterministic hash of the most recent engine state, excluding wall-clock timing.
    /// </summary>
    string ArchiveStateHash { get; }
}
