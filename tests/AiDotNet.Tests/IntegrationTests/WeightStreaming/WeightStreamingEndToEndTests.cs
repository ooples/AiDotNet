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
public sealed class WeightStreamingEndToEndTests
{
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

    // The StreamingPoolReport schema (DiskReadCount / EvictionCount /
    // PrefetchHitCount / PrefetchMissCount / PrefetchIssueCount /
    // ResidentBytes / CompressionRatio) is pinned by AiModelBuilder.
    // BuildWeightStreamingReport in src/ — any rename on the Tensors
    // side breaks the src build, which is a stronger guarantee than a
    // test in this assembly could provide (WeightRegistry is internal
    // to Tensors with InternalsVisibleTo on AiDotNet but NOT
    // AiDotNetTests, so a test-side access wouldn't compile anyway).
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
