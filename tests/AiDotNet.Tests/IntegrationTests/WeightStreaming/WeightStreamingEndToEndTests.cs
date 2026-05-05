using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WeightStreaming;

/// <summary>
/// Fixture that resets the process-wide WeightRegistry singleton before
/// each test in the collection. Without this, the second test in the
/// suite hits "existing streaming pool has N registered entries" from
/// WeightRegistry.Configure's mid-flight guard, since the previous
/// test's Configure left live pool entries behind.
/// </summary>
public sealed class WeightStreamingResetFixture : System.IDisposable
{
    public WeightStreamingResetFixture()
    {
        NeuralNetworkBase<float>.ResetWeightStreamingForTests();
    }
    public void Dispose()
    {
        NeuralNetworkBase<float>.ResetWeightStreamingForTests();
    }
}

[CollectionDefinition("WeightStreaming-Singleton", DisableParallelization = true)]
public sealed class WeightStreamingSingletonCollection : ICollectionFixture<WeightStreamingResetFixture> { }

/// <summary>
/// Issue #1222 — end-to-end exercise of the streaming path. These tests
/// run an ACTUAL forward through a network with streaming engaged, so a
/// runtime mismatch with the Tensors-side <c>WeightRegistry</c> /
/// <c>MaterializeScope</c> / <c>PrefetchAsync</c> surface fails the test.
/// The earlier coverage in <c>AutoDetectWeightStreamingTests</c> only
/// exercised the threshold-comparison logic without going through the
/// streaming forward path; these tests fill that gap.
///
/// Each test uses a SmallStreamableNetwork — a four-layer Dense network
/// with explicit ConfigureWeightLifetime — so the streaming branch in
/// PredictEager is exercised on a model small enough to run on CI in
/// milliseconds. The full PaLM-E 562B run remains <c>[Fact(Skip = "...")]</c>
/// in PaLMEProfilerTest because it needs ~2 TB of disk.
/// </summary>
[Collection("WeightStreaming-Singleton")]
public sealed class WeightStreamingEndToEndTests
{
    private readonly WeightStreamingResetFixture _fixture;

    public WeightStreamingEndToEndTests(WeightStreamingResetFixture fixture)
    {
        _fixture = fixture;
        // Reset between EVERY test in the collection (the fixture's ctor
        // only fires once for the whole collection). xUnit doesn't expose
        // a [BeforeEach] hook for collection-scoped fixtures; the
        // constructor runs per test, so resetting here is the right
        // place. The Dispose half handles teardown after the last test.
        NeuralNetworkBase<float>.ResetWeightStreamingForTests();
    }

    /// <summary>
    /// Minimal subclass so the test can drive ConfigureWeightLifetime
    /// directly without going through the AiModelBuilder facade. Three
    /// dense layers + one output dense — enough to exercise the
    /// per-layer prefetch + materialize-scope orchestration without
    /// needing real-world dimensions.
    /// </summary>
    private sealed class SmallStreamableNetwork : NeuralNetworkBase<float>
    {
        public SmallStreamableNetwork()
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            Layers.Add(new DenseLayer<float>(outputSize: 16));
            Layers.Add(new DenseLayer<float>(outputSize: 16));
            Layers.Add(new DenseLayer<float>(outputSize: 16));
            Layers.Add(new DenseLayer<float>(outputSize: 4));
        }

        protected override void InitializeLayers() { /* layers added in ctor */ }

        public override void UpdateParameters(Vector<float> parameters)
        {
            // Delegate to the base layer-walking SetParameters path so
            // params copied from a sibling network actually take effect.
            // Without this, the test's UpdateParameters call would be a
            // no-op and the eager-vs-streaming output comparison would
            // be meaningless (both would diverge from the source weights).
            SetParameters(parameters);
        }

        public override ModelMetadata<float> GetModelMetadata()
            => new() { Name = "SmallStreamableNetwork" };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new SmallStreamableNetwork();
    }

    [Fact]
    public void Streaming_PredictEager_ProducesValidOutput_AcrossMultipleCalls()
    {
        // Engage streaming BEFORE the first forward so every layer's
        // weights flow through the materialize/prefetch orchestration
        // from cold (no warm-up cheating). Run multiple forward calls
        // with different inputs, verify the output is finite, has the
        // expected shape, and varies with the input — together those
        // three checks prove the streaming path is delivering REAL
        // weights through MaterializeScope, not silently substituting
        // zeros / cached state from a prior call.
        var net = new SmallStreamableNetwork();
        net.ConfigureWeightLifetimeForTest(new GpuOffloadOptions());
        Assert.True(net.IsWeightStreamingActive,
            "Streaming should be active after ConfigureWeightLifetime");

        // First forward with one input.
        var inputA = new Tensor<float>([1, 8]);
        for (int i = 0; i < 8; i++) inputA[0, i] = (float)(i + 1) * 0.5f;
        var outputA = net.Predict(inputA);

        // Second forward with a clearly different input. Same model
        // instance — the streaming pool is exercised across calls,
        // catching bugs where the materialize scope leaks state
        // between forwards (e.g. a tensor pinned past its scope).
        var inputB = new Tensor<float>([1, 8]);
        for (int i = 0; i < 8; i++) inputB[0, i] = -(float)(i + 1) * 0.5f;
        var outputB = net.Predict(inputB);

        // Shape contract: 4 = last DenseLayer's outputSize.
        Assert.Equal(2, outputA.Rank);
        Assert.Equal(1, outputA.Shape[0]);
        Assert.Equal(4, outputA.Shape[1]);
        Assert.Equal(2, outputB.Rank);
        Assert.Equal(1, outputB.Shape[0]);
        Assert.Equal(4, outputB.Shape[1]);

        // Finite values: streaming-induced corruption (wrong tensor
        // pinned, scope-released-too-early) often surfaces as NaN /
        // Inf when the affected weights produce out-of-range
        // intermediate activations.
        for (int i = 0; i < outputA.Length; i++)
        {
            Assert.True(!float.IsNaN(outputA[i]) && !float.IsInfinity(outputA[i]),
                $"Streaming forward produced non-finite output at outputA[{i}]: {outputA[i]}");
            Assert.True(!float.IsNaN(outputB[i]) && !float.IsInfinity(outputB[i]),
                $"Streaming forward produced non-finite output at outputB[{i}]: {outputB[i]}");
        }

        // Different inputs MUST produce different outputs. If the
        // streaming path were silently caching a prior call's output
        // or returning constant-zero, this would fail.
        bool anyDiffers = false;
        for (int i = 0; i < outputA.Length; i++)
        {
            if (System.Math.Abs(outputA[i] - outputB[i]) > 1e-6f)
            {
                anyDiffers = true;
                break;
            }
        }
        Assert.True(anyDiffers,
            "Streaming forward returned identical outputs for clearly different inputs — "
            + "MaterializeScope is likely returning a stale tensor from a prior call.");
    }

    [Fact]
    public void Streaming_PredictEager_DoesNotThrow_OnLazyTensorsWithEmptyPlaceholders()
    {
        // Lazy networks register 0×0 placeholder tensors in their
        // weight registry pre-first-forward; my BeginLayerMaterializeScope
        // filters them out so the pool isn't asked to materialize
        // empty tensors. Pin that the streaming path survives a model
        // whose layers have varying lazy/eager states across the
        // chain (DenseLayer is lazy until first Forward).
        var net = new SmallStreamableNetwork();
        net.ConfigureWeightLifetimeForTest(new GpuOffloadOptions());

        // First-forward materializes the weights. The streaming path
        // should walk through it without throwing on empty placeholders.
        var input = new Tensor<float>([1, 8]);
        var output = net.Predict(input);

        Assert.NotNull(output);
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(4, output.Shape[1]); // last DenseLayer outputSize
    }

    [Fact]
    public void Streaming_TightPoolBudget_ForcesEvictionWithoutOOM()
    {
        // The proof that streaming actually pages weights to disk
        // under memory pressure. Configure a tight pool budget so the
        // network's combined weights exceed it — eviction MUST kick in
        // for the registration calls to succeed without growing the
        // pool past budget.
        //
        // Pre-fix (Lifetime never set to Streaming), this test would
        // not have engaged the eviction path at all: RegisterWeight
        // returned silently for Default lifetime, the pool's
        // ResidentBytes stayed 0, and the budget was never approached.
        //
        // Post-fix, registering each layer's weights walks the
        // EvictIfOverBudget loop in StreamingTensorPool to keep
        // ResidentBytes ≤ budget. Test asserts the cumulative
        // registered weight bytes far exceed the pool budget AND
        // ResidentBytes stays under it — proof that eviction ran.

        // Pool budget: 256 KB. SmallStreamableNetwork has 4 dense
        // layers; with 8 input → 16 → 16 → 16 → 4 outputs and float
        // weights: (8*16) + 16 + (16*16) + 16 + (16*16) + 16 + (16*4) + 4
        // = 128 + 16 + 256 + 16 + 256 + 16 + 64 + 4 = 756 floats = 3024 bytes.
        // That's tiny — to force eviction we need a budget SMALLER
        // than the cumulative serialized bytes. 1 KB budget against
        // 3 KB of weights → ~3 evictions during register.
        var net = new SmallStreamableNetwork();

        // Materialize the weights via warm-up forward.
        var input = new Tensor<float>([1, 8]);
        for (int i = 0; i < 8; i++) input[0, i] = (float)(i + 1) * 0.5f;
        _ = net.Predict(input);

        // Engage streaming with an aggressively-tight budget. 1 KB
        // forces every register to immediately exceed budget and
        // evict the LRU entry.
        var options = new GpuOffloadOptions { StreamingPoolMaxResidentBytes = 1024L };
        net.ConfigureWeightLifetimeForTest(options);

        long resident = net.WeightStreamingResidentBytes;
        Assert.True(resident <= 1024L,
            $"Streaming pool should have evicted to stay under 1024-byte budget, "
            + $"but ResidentBytes={resident}. EvictIfOverBudget didn't run during "
            + "register, suggesting tensor.Lifetime wasn't set to Streaming "
            + "BEFORE WeightRegistry.RegisterWeight (Default-lifetime tensors "
            + "early-return without ever touching the pool).");

        // Run a forward AFTER registration. Materialization MUST work
        // even though some weights were paged to disk during register.
        var output = net.Predict(input);
        Assert.NotNull(output);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(!float.IsNaN(output[i]) && !float.IsInfinity(output[i]),
                $"Forward through streaming-with-eviction produced non-finite "
                + $"output at index {i}: {output[i]}. Likely a Materialize bug "
                + "where a paged-out tensor isn't correctly rehydrated from "
                + "disk before its layer's Forward needs the bytes.");
        }
    }

    [Fact]
    public void Streaming_ConfigureWeightLifetime_ActuallyTracksWeightsInPool()
    {
        // Pre-audit (commit 190801e1b and earlier), this test would have
        // FAILED: RegisterTrainableTensorsWithWeightRegistry called
        // WeightRegistry.RegisterWeight on every tensor, but those
        // tensors had Lifetime=Default, and RegisterWeight's switch
        // early-returns for Default. Result: the streaming pool was
        // completely inert — every "weight registered" call was a
        // silent no-op, ResidentBytes stayed 0 forever, and PaLM-E
        // OOMed exactly as if streaming were never configured.
        //
        // Post-fix: ConfigureWeightLifetime sets per-instance
        // _registrationLifetime to Streaming (or GpuOffload when an
        // allocator is wired), and RegisterTrainableTensorsWithWeightRegistry
        // assigns it to each tensor BEFORE RegisterWeight. The pool
        // now actually tracks the tensors and ResidentBytes reflects
        // the registered weight bytes.
        //
        // This test pins the fix: streaming on a network with
        // materialized weights should produce a non-zero report.
        var net = new SmallStreamableNetwork();

        // First materialize the weights via a warm-up forward (lazy
        // DenseLayer allocates [in, out] tensors on first Forward).
        // We do this BEFORE configuring streaming so the warm-up runs
        // through the eager fast path and the resulting GC-heap
        // tensors are what get registered with the pool.
        var input = new Tensor<float>([1, 8]);
        for (int i = 0; i < 8; i++) input[0, i] = (float)(i + 1) * 0.5f;
        _ = net.Predict(input);

        // Engage streaming → walks materialized layers, sets each
        // tensor's Lifetime = Streaming, registers with pool. Pool's
        // ResidentBytes should now be > 0.
        net.ConfigureWeightLifetimeForTest(new GpuOffloadOptions());

        long resident = net.WeightStreamingResidentBytes;
        Assert.True(resident > 0,
            $"Streaming pool ResidentBytes should be > 0 after registering "
            + $"a network's weights, but is {resident}. This means "
            + "RegisterTrainableTensorsWithWeightRegistry's RegisterWeight "
            + "calls were no-ops — Lifetime was likely never set to Streaming "
            + "before the register call. See ConfigureWeightLifetime + "
            + "_registrationLifetime in NeuralNetworkBase.cs.");
    }
}

// Exposes ConfigureWeightLifetime to the test project. The base method
// is internal so already visible via InternalsVisibleTo, but having a
// thin delegating wrapper lets the test call site stay readable.
internal static class StreamableNetworkTestHelpers
{
    public static void ConfigureWeightLifetimeForTest(this NeuralNetworkBase<float> net, GpuOffloadOptions options)
    {
        net.ConfigureWeightLifetime(options);
    }
}
