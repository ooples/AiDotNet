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

    /// <summary>
    /// Disposes every <see cref="IDisposable"/> in <paramref name="owned"/> through <see cref="TryDispose"/>,
    /// continuing past failures, then reports them: a single failure is rethrown unchanged, two or more are
    /// thrown together as one <see cref="AggregateException"/>.
    /// </summary>
    /// <param name="owned">The resources to release; nulls and non-disposable entries are skipped.</param>
    /// <param name="owner">Names the owner in the aggregate message.</param>
    /// <remarks>
    /// <para>
    /// An owner that releases its resources in a plain loop leaks every resource after the first one whose
    /// <c>Dispose</c> throws. Collecting the failures and reporting them after the loop releases everything
    /// that can be released. Each instance is still disposed at most once, so the same resource listed twice,
    /// or shared with another owner, is released a single time.
    /// </para>
    /// <para>
    /// Exactly one failure -- by far the common case -- is rethrown as the original exception through
    /// <see cref="System.Runtime.ExceptionServices.ExceptionDispatchInfo"/>, keeping its type and the stack of the
    /// code that threw it. A caller that caught a specific exception type from an owner's <c>Dispose</c> before
    /// that owner started using this method therefore keeps working. Only two or more failures, which cannot be
    /// reported as one exception without losing some, are wrapped in an <see cref="AggregateException"/> -- the
    /// library's convention for multi-resource disposal (compare <c>DataPipeline</c>).
    /// </para>
    /// </remarks>
    /// <exception cref="AggregateException">Two or more resources threw from <c>Dispose</c>.</exception>
    public static void DisposeAll(IEnumerable<object?> owned, string owner)
    {
        if (owned is null) return;

        List<Exception>? failures = null;
        foreach (var item in owned)
        {
            if (item is not IDisposable disposable) continue;
            try
            {
                TryDispose(disposable);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                failures ??= new List<Exception>();
                failures.Add(ex);
            }
        }

        if (failures is null) return;

        if (failures.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failures[0]).Throw();
        }

        throw new AggregateException($"{failures.Count} resources owned by {owner} failed to dispose.", failures);
    }

    private static readonly object _sentinel = new();
}
