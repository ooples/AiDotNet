using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Parameters;
using Xunit;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Parameters;

/// <summary>
/// Phase 0 gate tests for the parameter-manifest refactor: one per capability that the
/// over-aggressive deletion sweep proved the base did not yet provide.
/// </summary>
/// <remarks>
/// <para>
/// Commit <c>11c602dfb</c> deleted 871 hand-written parameter surfaces on the theory that they were
/// all "(optional mode guard) + (one foreach over Layers)". The audit in <c>e1f8fec13</c> found 125
/// that carried real behaviour and had to restore 122 of them. The deletion was the right idea that
/// ran ahead of the capability.
/// </para>
/// <para>
/// The four bodies the sweep's filter wrongly matched did one of four things beyond the guard and
/// the walk, and those four are the buckets below: <b>E</b> walked a DIFFERENT enumeration
/// (<c>GetAllLayers()</c>, <c>_convLayers</c>, <c>_branches</c>), <b>O</b> ran an optimizer step,
/// <b>C</b> invalidated caches after the walk, <b>F</b> assigned a model-owned field.
/// </para>
/// <para>
/// These are DELETION GATES, not regression tests. The governing rule for the refactor is that no
/// override is removed until the base demonstrably does what that override was doing, proven by a
/// test that fails without it. A bucket's overrides may be deleted when its test here passes with
/// the override absent — and not before. That ordering is the whole lesson of the 122 restorations.
/// </para>
/// </remarks>
public class ParameterCapabilityGateTests
{
    /// <summary>
    /// Bucket E — the manifest must be the SINGLE traversal covering every component source.
    /// </summary>
    /// <remarks>
    /// This is the bucket that most directly argues for a manifest. A model holding components in
    /// more than one place had to hand-write a surface that walked all of them, because the base
    /// walked only <c>Layers</c>; delete that override and the extra components silently leave the
    /// count, the checkpoint and the optimizer. Registration order here is deliberately the reverse
    /// of stable-ID order, so a pass also proves the fold is ordered by identity rather than by the
    /// order somebody happened to call Register.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task BucketE_ManifestFoldsEveryRegisteredSource_NotJustTheFirst()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("branches/00000001", new FakeSource(new[] { 3.0, 4.0 }), ParameterSlotRole.Trainable);
        registry.Register("branches/00000000", new FakeSource(new[] { 1.0, 2.0 }), ParameterSlotRole.Trainable);

        Assert.Equal(4, registry.ParameterCount);

        var values = registry.GetParameters();
        Assert.Equal(4, values.Length);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { values[0], values[1], values[2], values[3] });
    }

    /// <summary>
    /// Bucket O — optimizer semantics as slot DATA, so a model does not override to express them.
    /// </summary>
    /// <remarks>
    /// An optimizer must be able to select only what it is allowed to step. If role is not carried
    /// through to the chunks, the only way to keep a frozen reservoir or an external ONNX buffer out
    /// of the update is a hand-written override — which is the thing being deleted. EchoStateNetwork's
    /// reservoir is the canonical case: fixed by design (Jaeger 2001) and catastrophic to train.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task BucketO_ChunksCarryRole_SoAnOptimizerCanSelectOnlyTrainable()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("model/00000000", new FakeSource(new[] { 1.0, 2.0 }), ParameterSlotRole.Trainable);
        registry.Register("model/00000001", new FakeSource(new[] { 9.0 }), ParameterSlotRole.Frozen);

        var trainable = new List<ParameterChunk<double>>();
        foreach (var chunk in registry.GetParameterStateChunks())
        {
            if (chunk.Role == ParameterSlotRole.Trainable) trainable.Add(chunk);
        }

        Assert.NotEmpty(trainable);
        long trainableValues = 0;
        foreach (var c in trainable) trainableValues += c.Tensor.Length;

        Assert.Equal(2, trainableValues);
    }

    /// <summary>
    /// Bucket C — a restore must be visible to everything derived from the parameters afterwards.
    /// </summary>
    /// <remarks>
    /// The restored overrides invalidated caches AFTER their walk. If the base does not, a model
    /// keeps serving pre-restore values from a cache and the checkpoint appears to load while
    /// changing nothing — the failure mode that is hardest to notice, because nothing throws.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task BucketC_ReadAfterRestore_ReflectsTheRestoredValues()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        var source = new FakeSource(new[] { 1.0, 2.0, 3.0 });
        registry.Register("model/00000000", source, ParameterSlotRole.Trainable);

        _ = registry.ParameterCount;
        _ = registry.GetParameters();

        registry.SetParameters(new Vector<double>(new[] { 7.0, 8.0, 9.0 }));

        var after = registry.GetParameters();
        Assert.Equal(new[] { 7.0, 8.0, 9.0 }, new[] { after[0], after[1], after[2] });
    }

    /// <summary>
    /// Bucket F — restore must write THROUGH to the owner's storage, not into a detached copy.
    /// </summary>
    /// <remarks>
    /// The sweep's filter rejected bodies that assigned a model-owned field, which is not the same
    /// test as "does nothing the base does not already do" — that assignment WAS the write-through.
    /// If the base cannot mutate through a slot accessor, a restore updates the registry's view and
    /// leaves the field the model actually computes with untouched: the checkpoint loads, the model
    /// keeps its old weights, and nothing reports a problem.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task BucketF_Restore_WritesThroughToTheOwnersStorage()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        var source = new FakeSource(new[] { 1.0, 2.0 });
        registry.Register("model/00000000", source, ParameterSlotRole.Trainable);

        registry.SetParameters(new Vector<double>(new[] { 5.0, 6.0 }));

        Assert.Equal(5.0, source.Storage[0]);
        Assert.Equal(6.0, source.Storage[1]);
    }

    /// <summary>
    /// Minimal source that keeps its values in storage the test can inspect directly, so
    /// write-through is observable rather than inferred from the registry's own read path.
    /// </summary>
    private sealed class FakeSource : IParameterSource<double>
    {
        internal FakeSource(double[] values) => Storage = (double[])values.Clone();

        internal double[] Storage { get; }

        public long ParameterCount => Storage.Length;

        public Vector<double> GetParameters()
        {
            var v = new Vector<double>(Storage.Length);
            for (int i = 0; i < Storage.Length; i++) v[i] = Storage[i];
            return v;
        }

        public void SetParameters(Vector<double> parameters)
        {
            for (int i = 0; i < Storage.Length && i < parameters.Length; i++) Storage[i] = parameters[i];
        }
    }
}
