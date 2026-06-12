using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Wraps any <see cref="IAgent{T}"/> and records each run as an <see cref="AgentTrajectory"/> in a
/// <see cref="ITrajectoryStore"/>, without altering the agent's behavior. This is how the self-improving
/// layer collects the experience it later evaluates and learns from.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Tracing is transparent and composable: the wrapper returns exactly what the inner agent returns and can
/// itself wrap (or be wrapped by) memory, supervisor, or swarm agents. Capture failures never affect the
/// run — the result is returned regardless of whether the store accepts the trajectory.
/// </para>
/// <para><b>For Beginners:</b> Put this around an agent and every time it runs, a copy of what happened is
/// filed away for later study. The agent itself behaves no differently; you just end up with a logbook.
/// </para>
/// </remarks>
public sealed class TracingAgent<T> : IAgent<T>
{
    private readonly IAgent<T> _inner;
    private readonly ITrajectoryStore _store;
    private readonly IReadOnlyDictionary<string, string>? _metadata;

    /// <summary>
    /// Initializes a new tracing wrapper.
    /// </summary>
    /// <param name="inner">The agent whose runs are recorded.</param>
    /// <param name="store">The store that receives each trajectory.</param>
    /// <param name="metadata">Optional metadata attached to every captured trajectory (e.g., experiment id).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="store"/> is <c>null</c>.</exception>
    public TracingAgent(IAgent<T> inner, ITrajectoryStore store, IReadOnlyDictionary<string, string>? metadata = null)
    {
        Guard.NotNull(inner);
        Guard.NotNull(store);
        _inner = inner;
        _store = store;
        _metadata = metadata;
    }

    /// <inheritdoc/>
    public string Name => _inner.Name;

    /// <inheritdoc/>
    public string Description => _inner.Description;

    /// <inheritdoc/>
    public async Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken cancellationToken = default)
    {
        var result = await _inner.RunAsync(messages, cancellationToken).ConfigureAwait(false);

        var trajectory = new AgentTrajectory(
            Guid.NewGuid().ToString("N"),
            result.AgentName is { } name && name.Trim().Length > 0 ? name : _inner.Name,
            result.Messages,
            result.FinalText,
            result.Iterations,
            result.Usage,
            reward: null,
            _metadata);

        try
        {
            await _store.AddAsync(trajectory, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Cancellation is the caller's signal, not a capture failure — honor it.
            throw;
        }
        catch (Exception)
        {
            // Documented contract: capture failures never affect the run. The
            // store is best-effort observability plumbing; a failing logbook
            // must not turn into a hard dependency for every agent call.
        }

        return result;
    }
}
