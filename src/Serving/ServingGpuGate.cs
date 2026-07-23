using System;
using System.Threading;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Serving;

/// <summary>
/// Process-wide serialization gate for model forwards that run on a SHARED GPU engine. The direct-GPU engine's
/// scratch buffers and compiled-plan cache are process-global and not thread-safe, so when several models are
/// served concurrently in one process (each on its own continuous-batching loop) their forwards would race on
/// that shared state and corrupt each other's outputs. This gate serializes those forwards.
/// </summary>
/// <remarks>
/// It is a no-op on the CPU engine (CPU forwards on separate models are independent), and effectively free for
/// the common single-model case — that model's one batching loop already runs its forwards one at a time, so the
/// gate is uncontended. A single physical GPU executes kernels serially anyway, so serializing here costs no real
/// throughput; it only removes the cross-model data race. Models placed on their own isolated GPU contexts (e.g.
/// tensor-parallel per-rank backends) do not touch the shared engine and are unaffected.
/// </remarks>
internal static class ServingGpuGate
{
    // Reentrant (Monitor): a gated forward never re-enters the gate today, but reentrancy keeps future nesting safe.
    private static readonly object _sync = new();

    private static bool ShouldGate()
    {
        try { return AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.IsGpuAvailable; }
        catch { return false; }
    }

    /// <summary>
    /// Enters the gate when a shared GPU engine is active, returning a scope to release it; returns null (no
    /// serialization) on CPU. Use with <c>using</c>.
    /// </summary>
    public static IDisposable? Enter()
    {
        if (!ShouldGate()) return null;
        Monitor.Enter(_sync);
        return new Scope();
    }

    private sealed class Scope : IDisposable
    {
        private bool _released;
        public void Dispose()
        {
            if (_released) return;
            _released = true;
            Monitor.Exit(_sync);
        }
    }
}
