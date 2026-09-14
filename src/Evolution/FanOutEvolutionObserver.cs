using System.Runtime.ExceptionServices;

namespace AiDotNet.Evolution;

/// <summary>Delivers an event to both observers without losing either failure.</summary>
internal sealed class FanOutEvolutionObserver<TGenome> : IEvolutionObserver<TGenome>
{
    private readonly IEvolutionObserver<TGenome> _first;
    private readonly IEvolutionObserver<TGenome> _second;

    internal FanOutEvolutionObserver(IEvolutionObserver<TGenome> first, IEvolutionObserver<TGenome> second)
    {
        _first = first ?? throw new ArgumentNullException(nameof(first));
        _second = second ?? throw new ArgumentNullException(nameof(second));
    }

    public async ValueTask OnEventAsync(EvolutionEvent<TGenome> item, CancellationToken cancellationToken = default)
    {
        Exception? firstFailure = null;
        try { await _first.OnEventAsync(item, cancellationToken).ConfigureAwait(false); }
        catch (Exception exception) { firstFailure = exception; }

        try { await _second.OnEventAsync(item, cancellationToken).ConfigureAwait(false); }
        catch (Exception exception) when (firstFailure is not null)
        {
            // The core recursively classifies aggregate causes. A later recoverable failure must never mask
            // an earlier fatal one, as it could with a plain try/finally fan-out.
            throw new AggregateException("Multiple evolution observers failed.", firstFailure, exception);
        }
        if (firstFailure is not null) ExceptionDispatchInfo.Capture(firstFailure).Throw();
    }
}
