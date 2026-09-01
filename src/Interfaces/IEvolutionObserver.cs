using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Receives structured progress without coupling the engine to a console or logging provider.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionObserver<TGenome>
{
    /// <summary>Observes one immutable event.</summary>
    ValueTask OnEventAsync(EvolutionEvent<TGenome> evolutionEvent, CancellationToken cancellationToken = default);
}
