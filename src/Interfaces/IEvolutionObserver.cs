using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Receives structured progress without coupling the engine to a console or logging provider.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine awaits <see cref="OnEventAsync"/> inline, so events arrive one at a time in
/// <see cref="EvolutionEvent{TGenome}.Sequence"/> order and a slow observer slows the run. Exceptions thrown by
/// an observer are isolated from engine state and do not alter the search or its deterministic identity; only
/// cancellation of the supplied token is propagated. <see cref="EvolutionEvent{TGenome}"/> instances are
/// immutable, so an observer may retain or forward them freely. See <see cref="EvolutionEventKind"/> for the
/// catalogue of notifications and when each is raised.
/// </para>
/// <para><b>For Beginners:</b> An observer is how you watch an evolution run while it is happening without the
/// engine knowing anything about your logger, dashboard, or test harness. Every time something notable happens -
/// a candidate is proposed, an evaluation finishes, the archive improves, a checkpoint is written, or the run
/// stops - the engine hands your observer one small event object describing it. A typical implementation prints
/// the best quality after each <see cref="EvolutionEventKind.ArchiveChanged"/> event, or appends every event to a
/// list so a unit test can assert what happened. Keep the handler fast and avoid throwing from it; if you need to
/// do heavy work, queue the event and process it elsewhere.</para>
/// </remarks>
public interface IEvolutionObserver<TGenome>
{
    /// <summary>Observes one immutable event.</summary>
    /// <param name="evolutionEvent">The event to observe.</param>
    /// <param name="cancellationToken">Cancels the observation.</param>
    ValueTask OnEventAsync(EvolutionEvent<TGenome> evolutionEvent, CancellationToken cancellationToken = default);
}
