using System;
using System.IO;
using System.Threading;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.WeightStreaming;

/// <summary>
/// Issue #1222 / task #183 — auto-detect default weight streaming when a
/// model's parameter count crosses the threshold. Pins:
///   - Below threshold: no auto-streaming (zero overhead for small models).
///   - Above threshold: streaming enabled, <c>WeightStreamingAutoDetected</c>
///     is true after explicit <c>TryAutoEnableWeightStreaming</c> call.
///   - <c>DisableAutoStreaming()</c> opts out even when above threshold.
///   - The check is idempotent — repeated <c>TryAutoEnableWeightStreaming</c>
///     calls don't re-pay the <c>ParameterCount</c> walk.
///
/// To test the threshold branch deterministically without the global
/// <c>AIDOTNET_STREAMING_THRESHOLD_PARAMS</c> env var (which is read once
/// at type load and would race across test classes), each test uses
/// <see cref="NeuralNetworkBase{T}.ApplyAutoDetectThresholdOverride"/> to
/// install a per-instance threshold appropriate to the scenario. The
/// stub network exposes the protected base-class internal so the tests
/// can drive it.
/// </summary>
public class AutoDetectWeightStreamingTests
{
    /// <summary>
    /// Stub network that lets the test set ParameterCount directly AND
    /// counts how many times the property's getter has been read. The
    /// counter is the lever we use to assert the auto-detect path is
    /// idempotent — re-pay would show up as multiple ParameterCount
    /// reads on the second / third <c>TryAutoEnableWeightStreaming</c>
    /// call. Overriding is the least-invasive way to drive a deterministic
    /// above/below-threshold check without inflating real layer weights.
    /// </summary>
    private sealed class FixedParamCountNetwork : VectorModelLayoutBase<float>
    {
        private readonly long _fixedCount;
        private long _parameterCountReads;

        public FixedParamCountNetwork(long fixedCount)
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            _fixedCount = fixedCount;
            // Add a trivial layer so the layer-only path's
            // InitializeLayers / ResolveLazyLayerShapes still has something
            // to walk — without any layer the EnsureArchitectureInitialized
            // path skips lazy resolution but still hits TryAutoEnableWeightStreaming.
            Layers.Add(new DenseLayer<float>(outputSize: 1));
        }

        public override long ParameterCount
        {
            get
            {
                Interlocked.Increment(ref _parameterCountReads);
                return _fixedCount;
            }
        }

        public long ParameterCountReadCount => Interlocked.Read(ref _parameterCountReads);

        /// <summary>
        /// Test-only accessor — the base's <c>ApplyAutoDetectThresholdOverride</c>
        /// is internal and visible to AiDotNetTests via InternalsVisibleTo,
        /// but expose it as a public method here so test bodies don't have
        /// to know about that internal-visibility detail.
        /// </summary>
        public void SetThresholdForTest(long thresholdParams) =>
            ApplyAutoDetectThresholdOverride(thresholdParams);

        protected override void InitializeLayers() { /* layer added in ctor */ }
        public override void UpdateParameters(Vector<float> parameters) { /* not exercised */ }
        public override ModelMetadata<float> GetModelMetadata()
            => new() { Name = "FixedParamCountNetwork" };
        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }
    }

    /// <summary>
    /// A model whose PredictCore answers from its own path without the default PredictCore (as
    /// FeedForwardNeuralNetwork's fused dense predict does), so only the Predict funnel can record its first
    /// forward.
    /// </summary>
    private sealed class OwnPredictPathNetwork : VectorModelLayoutBase<float>
    {
        private readonly long _fixedCount;
        private long _parameterCountReads;

        public OwnPredictPathNetwork(long fixedCount)
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            _fixedCount = fixedCount;
            Layers.Add(new DenseLayer<float>(outputSize: 1));
        }

        public override long ParameterCount
        {
            get
            {
                long reads = Interlocked.Increment(ref _parameterCountReads);
                if (reads <= ThrowingReads)
                    throw new InvalidOperationException("Parameter count not readable yet.");
                return _fixedCount;
            }
        }

        /// <summary>How many first ParameterCount reads throw, as a partially built model's can.</summary>
        public int ThrowingReads { get; set; }

        public long ParameterCountReadCount => Interlocked.Read(ref _parameterCountReads);

        public void SetThresholdForTest(long thresholdParams) => ApplyAutoDetectThresholdOverride(thresholdParams);

        protected override Tensor<float> PredictCore(Tensor<float> input) => input;
        protected override void InitializeLayers() { /* layer added in ctor */ }
        public override void UpdateParameters(Vector<float> parameters) { /* not exercised */ }
        public override ModelMetadata<float> GetModelMetadata() => new() { Name = "OwnPredictPathNetwork" };
        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }
    }

    [Fact]
    public void BelowThreshold_OwnPredictPath_FinalizesAfterFirstPredict()
    {
        // Detection below the threshold finalizes only after a first forward. A model answering from its own
        // PredictCore never recorded one, so every Predict re-ran detection: both of its SetTrainingMode transitions
        // walked the parameter count and built (and hashed) the parameter layout again.
        var net = new OwnPredictPathNetwork(fixedCount: 1_000_000_000L);
        net.SetThresholdForTest(10_000_000_000L);
        var input = new Tensor<float>(new[] { 1, 1 });
        net.Predict(input);
        long readsAfterFirst = net.ParameterCountReadCount;
        net.Predict(input);
        net.Predict(input);
        Assert.Equal(readsAfterFirst, net.ParameterCountReadCount);
        Assert.False(net.IsWeightStreamingActive, "A below-threshold model must stay eager.");
    }
    [Fact]
    public void OwnPredictPath_CountThatThrew_IsRetriedUntilDetectionFinalizes()
    {
        // Already in inference mode, so Predict makes no SetTrainingMode transition: the funnel's own retry is the only
        // detection call. A count that threw on the first Predict must be read again on the next one, then never again.
        // Every count read during the first Predict throws (as a partially built model's can). The second Predict must
        // read the count again, succeed and finalize detection; the third reads it no more.
        var net = new OwnPredictPathNetwork(fixedCount: 1_000_000_000L);
        net.SetThresholdForTest(10_000_000_000L);
        net.SetTrainingMode(false);
        net.ThrowingReads = int.MaxValue;   // every read during the first Predict throws
        var input = new Tensor<float>(new[] { 1, 1 });
        long beforeFirst = net.ParameterCountReadCount;
        net.Predict(input);
        long afterFirst = net.ParameterCountReadCount;
        Assert.True(afterFirst > beforeFirst, "The first Predict must attempt detection.");
        net.ThrowingReads = 0;
        net.Predict(input);
        long afterRetry = net.ParameterCountReadCount;
        Assert.True(afterRetry > afterFirst, "The second Predict must retry the count the first one could not read.");
        net.Predict(input);
        Assert.Equal(afterRetry, net.ParameterCountReadCount);
        Assert.False(net.IsWeightStreamingActive, "A below-threshold model must stay eager.");
    }

    [Fact]
    public void BelowThreshold_AutoStreaming_DoesNotEngage()
    {
        // 1B params with a 10B threshold — well below. Explicit
        // TryAutoEnableWeightStreaming call exercises the auto-detect
        // path (the ctor doesn't, since the env var read happens once at
        // type load and we want determinism). Auto-detect should see
        // we're under threshold and leave streaming disabled.
        var net = new FixedParamCountNetwork(fixedCount: 1_000_000_000L);
        net.SetThresholdForTest(10_000_000_000L);
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        Assert.False(net.WeightStreamingAutoDetected,
            "1B-param model should be below the 10B threshold and stay eager.");
        Assert.False(net.IsWeightStreamingActive,
            "Below-threshold models must not have streaming active.");
    }

    [Fact]
    public void AboveThreshold_AutoStreaming_Engages()
    {
        // 50B params with a 10B threshold — above. Explicit
        // TryAutoEnableWeightStreaming call should engage streaming.
        // This is the positive counterpart to BelowThreshold above —
        // without it the test class only exercises the negative branch
        // and a regression that breaks above-threshold engagement would
        // silently pass.
        var net = new FixedParamCountNetwork(fixedCount: 50_000_000_000L);
        net.SetThresholdForTest(10_000_000_000L);
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        Assert.True(net.WeightStreamingAutoDetected,
            "50B-param model with 10B threshold should auto-engage streaming.");
        Assert.True(net.IsWeightStreamingActive,
            "Streaming must be reported active after auto-detect engages.");
    }

    [Fact]
    public void DisableAutoStreaming_PreventsEngagementEvenAboveThreshold()
    {
        // Opt out BEFORE the auto-detect runs. A 50B-param model with
        // 10B threshold would normally engage streaming; the opt-out
        // must veto that. Closes review-comment #1271.rRy1 / .rT-j
        // (test previously had no assertions and would pass even if
        // DisableAutoStreaming silently regressed to a no-op).
        var net = new FixedParamCountNetwork(fixedCount: 50_000_000_000L);
        net.SetThresholdForTest(10_000_000_000L);
        net.DisableAutoStreaming();
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        Assert.False(net.WeightStreamingAutoDetected,
            "DisableAutoStreaming must veto auto-detect even when ParameterCount " +
            "exceeds the threshold — that's the whole point of the opt-out.");
        Assert.False(net.IsWeightStreamingActive,
            "Opt-out must keep streaming inactive on the auto-detect path.");
    }

    [Fact]
    public void Idempotent_RepeatedCalls_DoNotRePayParameterCountWalk()
    {
        // Auto-detect uses the `_streamingAutoDetectFinalized` flag (per
        // NeuralNetworkBase.TryAutoEnableWeightStreaming) to early-return
        // on subsequent calls. Closes review-comment #1271.rT-u (previous
        // assertion only checked boolean stability — could pass even if
        // ParameterCount was re-walked every call).
        //
        // We use the above-threshold scenario because that path finalizes
        // unconditionally on first run (lines 3727-3737 of
        // NeuralNetworkBase.cs). Below-threshold finalization is gated
        // on `_firstForwardCompleted` to support lazy models that report
        // 0 params pre-forward, so it would NOT short-circuit a direct
        // TryAutoEnableWeightStreaming() call without first going through
        // Predict — that's the right production behaviour, just not what
        // this idempotency test is pinning. Above-threshold gives us a
        // clean "finalized = no re-walk" assertion.
        //
        // Lever: ParameterCountReadCount on the stub. Auto-detect reads
        // ParameterCount EXACTLY once when it runs to completion on the
        // above-threshold path; the 2nd, 3rd, 4th calls must short-circuit
        // on the finalized flag and read 0 additional times.
        var net = new FixedParamCountNetwork(fixedCount: 50_000_000_000L);
        net.SetThresholdForTest(10_000_000_000L);

        // Measure deltas around the auto-detect call only (the stub's
        // ctor / layer-add may have read ParameterCount for unrelated
        // reasons before we got here).
        long readsBeforeAutoDetect = net.ParameterCountReadCount;

        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        long readsAfterFirstCall = net.ParameterCountReadCount;
        long firstCallDelta = readsAfterFirstCall - readsBeforeAutoDetect;
        Assert.True(firstCallDelta >= 1,
            "First TryAutoEnableWeightStreaming call must read ParameterCount " +
            "at least once (otherwise auto-detect skipped its decision).");
        Assert.True(net.WeightStreamingAutoDetected,
            "50B-param model with 10B threshold should engage on the first call.");

        // Subsequent calls must short-circuit BEFORE reading ParameterCount.
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        long totalReadsAfter4Calls = net.ParameterCountReadCount;
        long subsequentDelta = totalReadsAfter4Calls - readsAfterFirstCall;

        Assert.Equal(0L, subsequentDelta);
        // And the observable result stays stable across the idempotent
        // calls — auto-detect doesn't oscillate.
        Assert.True(net.WeightStreamingAutoDetected);
    }

    [Fact]
    public void ParameterCountThrows_AutoDetect_DoesNotPropagate()
    {
        // Some partially-constructed models throw from ParameterCount.
        // The auto-detect should swallow the exception so it never
        // crashes the ctor / first Predict — the explicit
        // ConfigureWeightLifetime entry point stays available for those
        // models to call later when their state is ready.
        var net = new ThrowingParamCountNetwork();
        // No throw expected.
        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);
        Assert.False(net.WeightStreamingAutoDetected);
    }

    private sealed class ThrowingParamCountNetwork : VectorModelLayoutBase<float>
    {
        public ThrowingParamCountNetwork()
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            Layers.Add(new DenseLayer<float>(outputSize: 1));
        }

        public override long ParameterCount
            => throw new System.InvalidOperationException("simulated partial-construction failure");

        protected override void InitializeLayers() { }
        public override void UpdateParameters(Vector<float> parameters) { }
        public override ModelMetadata<float> GetModelMetadata() => new() { Name = "Throwing" };
        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }
        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new ThrowingParamCountNetwork();
    }

    [Fact]
    public void AboveThresholdStructuralEstimate_EngagesWithoutParameterCountWalk()
    {
        var net = new StructuralEstimateNetwork();
        net.SetThresholdForTest(10_000_000_000L);

        net.TryAutoEnableWeightStreaming(isTrainingOverride: false);

        Assert.True(net.WeightStreamingAutoDetected);
        Assert.True(net.IsWeightStreamingActive);
        Assert.Equal(0L, net.ParameterCountReadCount);
    }

    [AiDotNet.Attributes.TensorLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Features,
        Direction = AiDotNet.Attributes.TensorLayoutDirection.Input)]
    [AiDotNet.Attributes.TensorLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Features,
        Direction = AiDotNet.Attributes.TensorLayoutDirection.Output)]
    private sealed class StructuralEstimateNetwork : NeuralNetworkBase<float>
    {
        private long _parameterCountReads;

        public StructuralEstimateNetwork()
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            Layers.Add(new DenseLayer<float>(outputSize: 1));
        }

        public override long ParameterCount
        {
            get
            {
                Interlocked.Increment(ref _parameterCountReads);
                return 1L;
            }
        }

        public long ParameterCountReadCount => Interlocked.Read(ref _parameterCountReads);
        public void SetThresholdForTest(long thresholdParams) =>
            ApplyAutoDetectThresholdOverride(thresholdParams);
        protected override long EstimateStructuralParameterCount() => 50_000_000_000L;
        protected override void InitializeLayers() { }
        public override void UpdateParameters(Vector<float> parameters) { }
        public override ModelMetadata<float> GetModelMetadata() => new() { Name = "StructuralEstimate" };
        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }
        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance() =>
            new StructuralEstimateNetwork();
    }
}
