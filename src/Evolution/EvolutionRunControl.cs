namespace AiDotNet.Evolution;

/// <summary>A one-run, thread-safe graceful-stop handle for the AiDotNet evolution facade.</summary>
/// <remarks>
/// Stop requests drain the current engine batch and return its result, unlike cancellation, which aborts
/// and may roll back the in-flight batch. A stop is not a durability receipt: checkpoint success must be
/// established separately before calling a run resumable. Use a new handle for every run.
/// </remarks>
public sealed class EvolutionRunControl
{
    private readonly object _gate = new();
    private Action? _stop;
    private bool _used;
    private bool _requested;
    private bool _finished;

    /// <summary>Gets whether a graceful stop has been requested, including before the run starts.</summary>
    public bool IsStopRequested { get { lock (_gate) return _requested; } }

    /// <summary>Gets whether the connected run has left its execution scope, successfully or exceptionally.</summary>
    public bool IsFinished { get { lock (_gate) return _finished; } }

    /// <summary>Requests a graceful stop; returns false only after the connected run has finished.</summary>
    /// <returns>Whether this handle accepted the request; this is not an acknowledgment of a saved checkpoint.</returns>
    public bool RequestStop()
    {
        Action? stop;
        lock (_gate)
        {
            if (_finished) return false;
            _requested = true;
            stop = _stop;
        }
        stop?.Invoke();
        return true;
    }

    internal IDisposable Attach(Action stop)
    {
        if (stop is null) throw new ArgumentNullException(nameof(stop));
        bool requested;
        lock (_gate)
        {
            if (_used) throw new InvalidOperationException("An EvolutionRunControl belongs to exactly one run.");
            _used = true;
            _stop = stop;
            requested = _requested;
        }
        if (requested) stop();
        return new Registration(this);
    }

    private sealed class Registration : IDisposable
    {
        private readonly EvolutionRunControl _owner;
        internal Registration(EvolutionRunControl owner) => _owner = owner;
        public void Dispose()
        {
            lock (_owner._gate)
            {
                _owner._stop = null;
                _owner._finished = true;
            }
        }
    }
}
