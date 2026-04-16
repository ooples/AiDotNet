using System;
using System.Runtime;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Shared base class for diffusion unit tests that forces a full two-pass
/// GC cycle WITH Large Object Heap compaction between tests. Addresses the
/// CI shard cancellations tracked in github.com/ooples/AiDotNet#1136.
/// </summary>
/// <remarks>
/// <para>
/// The <c>Unit - 03 Diffusion/Encoding</c> CI shard repeatedly hit the 45-min
/// wall-clock timeout on 16 GB Windows runners because ~200 sequential
/// diffusion-model unit tests each construct a production-default model
/// (DiT-XL, SD15, SD3, Flux1, etc.) with eagerly-allocated weight tensors
/// and let the reference fall out of scope without forcing a compacting
/// collection. .NET's generational GC waits for memory pressure to trigger
/// a gen-2 pass — but the tests allocate fast enough to hit the wall clock
/// before memory pressure kicks in.
/// </para>
/// <para>
/// Implementing <see cref="IAsyncLifetime"/> gives xunit a per-test teardown
/// hook. The teardown runs a blocking compacting Gen-2 collection with
/// explicit Large Object Heap compaction between tests. Two collections
/// around <see cref="GC.WaitForPendingFinalizers"/> ensure finalizer-owned
/// resources (GPU-backing-buffer finalizers return pooled memory) also
/// get reclaimed.
/// </para>
/// <para><b>Why LOH compaction matters specifically:</b> diffusion model
/// weight tensors are typically several hundred MB each, well above the
/// 85KB LOH threshold. Plain <see cref="GC.Collect"/> sweeps the LOH but
/// does NOT compact it — freed regions stay fragmented. Over ~200 sequential
/// tests, LOH fragmentation accumulates until the next allocation can't find
/// a contiguous region even though free bytes exist, producing an
/// <see cref="OutOfMemoryException"/> on a runner that APPEARS to have
/// plenty of memory. Setting <see cref="GCLargeObjectHeapCompactionMode.CompactOnce"/>
/// on the next Gen-2 collection forces the LOH to compact, eliminating
/// fragmentation as an OOM vector. The compaction flag auto-resets to
/// <see cref="GCLargeObjectHeapCompactionMode.Default"/> after each use —
/// safe for test teardowns.
/// </para>
/// <para>
/// This is not a fix for the allocation cost itself — that's the Tensors-
/// side and layer-side lazy-init work tracked in the #1136 PR chain. It's
/// a lifecycle hygiene fix that lets the existing tests fit within the
/// wall-clock budget while the deeper allocation work lands.
/// </para>
/// </remarks>
public abstract class DiffusionUnitTestBase : IAsyncLifetime
{
    /// <summary>
    /// Static lock serializing concurrent teardowns. xunit parallelizes across
    /// test-classes by default (methods in the same class stay sequential),
    /// so two test classes that both inherit from this base can hit
    /// <see cref="DisposeAsync"/> concurrently on different threads.
    /// <see cref="GCSettings.LargeObjectHeapCompactionMode"/> is process-
    /// global, so concurrent toggles race — thread A's set-to-CompactOnce
    /// can be observed by thread B's GC.Collect, or vice versa, producing
    /// non-deterministic LOH behavior. Serializing the whole
    /// mode-set → collect → wait → mode-set → collect sequence keeps LOH
    /// compaction deterministic per teardown.
    /// </summary>
    private static readonly object _lohCompactionGate = new();

    /// <summary>
    /// Before-test hook. No-op — the base has no ambient state to initialize.
    /// </summary>
    public Task InitializeAsync() => Task.CompletedTask;

    /// <summary>
    /// After-test hook. Forces a blocking compacting Gen-2 GC (with explicit
    /// LOH compaction) to reclaim undisposed model weight tensors AND
    /// defragment the LOH between tests. Essential for sequential
    /// diffusion-model unit tests to fit within the 45-min CI wall clock —
    /// without LOH compaction, the heap APPEARS to have free memory but
    /// can't satisfy the next test's multi-hundred-MB weight allocation
    /// because the free bytes are split across fragmented regions.
    /// </summary>
    /// <remarks>
    /// The entire GC sequence runs under <see cref="_lohCompactionGate"/>
    /// so concurrent teardowns from parallel test classes don't race on the
    /// process-global <see cref="GCSettings.LargeObjectHeapCompactionMode"/>
    /// flag.
    /// </remarks>
    public Task DisposeAsync()
    {
        lock (_lohCompactionGate)
        {
            // Compact the LOH on the next Gen-2 collection. CompactOnce auto-resets
            // to Default after each compacting Gen-2 pass, so this is scoped to
            // the single test-teardown call — no process-wide side-effect beyond
            // the lock's critical section.
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect(generation: 2, mode: GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();

            // Second pass reclaims memory freed by finalizers (GPU-pool return
            // paths, for instance). Also compacting so a LOH-allocating finalizer
            // doesn't refragment immediately.
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect(generation: 2, mode: GCCollectionMode.Forced, blocking: true, compacting: true);
        }
        return Task.CompletedTask;
    }
}
