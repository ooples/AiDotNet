using System.Runtime.CompilerServices;

namespace AiDotNet.Helpers;

/// <summary>
/// Guarantees that each <see cref="IDisposable"/> instance is disposed
/// <i>at most once</i>, even when it's reachable from multiple owners
/// (shared layer graphs, predictor instances shared between diffusion
/// wrappers, etc.).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists:</b> every cascade <c>Dispose</c> in the codebase
/// previously relied on catching <see cref="ObjectDisposedException"/> as
/// the signal that a shared component had already been disposed. That
/// assumption is wrong for a large fraction of our <see cref="IDisposable"/>
/// implementations — many layers (e.g., <c>DenseLayer</c> returning rented
/// tensors to <c>TensorAllocator</c>) are <i>not</i> idempotent on second
/// dispose. A second <c>Dispose</c> call would silently double-return
/// pooled buffers / native handles and corrupt the pool.
/// </para>
/// <para>
/// This guard uses a <see cref="ConditionalWeakTable{TKey,TValue}"/> so the
/// registry doesn't pin disposables in memory — once a disposable becomes
/// unreachable from the rest of the program, the GC is free to collect it
/// and the registry entry vanishes. Net471-compatible.
/// </para>
/// <para>
/// <b>Thread safety:</b> <see cref="ConditionalWeakTable{TKey,TValue}"/> is
/// thread-safe for concurrent reads and mutations. We additionally hold a
/// lock for the read-then-add sequence so two threads racing on the same
/// instance can't both win the "first to dispose" race.
/// </para>
/// </remarks>
internal static class DisposeOnceGuard
{
    private static readonly ConditionalWeakTable<IDisposable, object> _disposed = new();
    private static readonly object _sync = new();

    /// <summary>
    /// Disposes <paramref name="target"/> iff this guard has not seen it
    /// already. Returns <c>true</c> when Dispose ran, <c>false</c> when the
    /// call was a no-op because another owner already disposed this instance.
    /// </summary>
    /// <remarks>
    /// If <paramref name="target"/>'s <c>Dispose</c> throws anything other
    /// than <see cref="ObjectDisposedException"/>, the guard entry is removed
    /// before re-throwing so a later retry can attempt disposal again.
    /// </remarks>
    public static bool TryDispose(IDisposable? target)
    {
        if (target is null) return false;

        lock (_sync)
        {
            if (_disposed.TryGetValue(target, out _))
                return false;
            _disposed.Add(target, _sentinel);
        }

        try
        {
            target.Dispose();
            return true;
        }
        catch (ObjectDisposedException ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"DisposeOnceGuard: {target.GetType().Name} reported already-disposed: {ex.Message}");
            return true; // It's disposed now, even if by an earlier chain.
        }
        catch
        {
            // Dispose threw a non-ObjectDisposedException. The component is
            // in an indeterminate state — drop the guard entry so an explicit
            // retry can try again if the caller wants to.
            lock (_sync)
            {
                _disposed.Remove(target);
            }
            throw;
        }
    }

    private static readonly object _sentinel = new();
}
