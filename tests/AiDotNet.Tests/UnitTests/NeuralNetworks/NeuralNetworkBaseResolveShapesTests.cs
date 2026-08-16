using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression tests for #1832 — <c>NeuralNetworkBase&lt;T&gt;.ParameterCount</c>'s cache
/// used to go stale after lazy input-shape resolution inside a model-class-owned forward
/// (e.g. NeRF's positional encoding turning <c>[N, 3]</c> positions into <c>[N, 60]</c>
/// before <c>Layers[0]</c> sees them). The stale cache reported the pre-resolution size
/// while <see cref="AiDotNet.Interfaces.IParameterizable{T, TInput, TOutput}.GetParameters"/>
/// (which walks layers fresh) returned the post-resolution size, so a flat-vector
/// round-trip (train → GetParameters → save → fresh model → SetParameters) threw
/// <c>Expected N parameters, got M</c>.
///
/// The completed contract has three arms:
///   1. <c>ParameterCount</c> sums the generated structural manifest on every access — never stale
///      and never dependent on whether weights happen to have been allocated.
///   2. Architecture-known models resolve their declared topology automatically, so a fresh model
///      can restore a checkpoint without a warm-up forward or model-specific parameter override.
///   3. Public <c>ResolveShapes(sampleInput)</c> remains available for genuinely data-dependent
///      topology and is idempotent after automatic resolution.
///
/// These tests pin all three arms.
/// </summary>
public class NeuralNetworkBaseResolveShapesTests
{
    private static NeuralNetwork<double> BuildLazyDenseNetwork() => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 3,
            outputSize: 2,
            layers:
            [
                new DenseLayer<double>(4, (IActivationFunction<double>)new ReLUActivation<double>()),
                new DenseLayer<double>(2, (IActivationFunction<double>)new IdentityActivation<double>())
            ]));

    private static NeRF<float> BuildNeRF() => new NeRF<float>(
        positionEncodingLevels: 10,
        directionEncodingLevels: 4,
        hiddenDim: 64,
        numLayers: 3,
        colorHiddenDim: 32,
        colorNumLayers: 1,
        useHierarchicalSampling: false,
        renderSamples: 8,
        renderNearBound: 1.0,
        renderFarBound: 4.5,
        learningRate: 1e-3);

    private static (Tensor<float> input, Tensor<float> target) DummyBatch(int n, int seed = 0)
    {
        var rng = new Random(seed);
        var input = new float[n * 6];
        var target = new float[n * 4];
        for (int i = 0; i < n; i++)
        {
            // Random position in [-0.5, 0.5]³, random unit direction
            for (int j = 0; j < 3; j++) input[i * 6 + j] = (float)(rng.NextDouble() - 0.5);
            float dx = (float)(rng.NextDouble() - 0.5);
            float dy = (float)(rng.NextDouble() - 0.5);
            float dz = (float)(rng.NextDouble() - 0.5);
            float ln = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
            if (ln < 1e-6f) ln = 1;
            input[i * 6 + 3] = dx / ln;
            input[i * 6 + 4] = dy / ln;
            input[i * 6 + 5] = dz / ln;
            target[i * 4 + 0] = 0.5f;
            target[i * 4 + 1] = 0.5f;
            target[i * 4 + 2] = 0.5f;
            target[i * 4 + 3] = 4f;
        }
        return (new Tensor<float>(new[] { n, 6 }, new Vector<float>(input)),
                new Tensor<float>(new[] { n, 4 }, new Vector<float>(target)));
    }

    [Fact]
    public async Task LazyComposite_ManifestBuildsChildStructureWithoutDeclaringItParameterFree()
    {
        await Task.Yield();
        using var layer = new TransformerEncoderLayer<double>(numHeads: 2, feedForwardDim: 32);
        layer.ResolveShapesOnly(new[] { 4, 16 });

        var layout = new AiDotNet.Models.Parameters.ParameterLayoutSnapshot(
            layer.GetParameterLayout());

        Assert.True(layout.ParameterCount > 0);
        Assert.Equal(
            AiDotNet.Models.Parameters.ParameterReadiness.ShapeResolvedUnmaterialized,
            layout.Readiness);

        long declared = layer.ParameterCount;
        Assert.Equal(declared, layer.GetParameters().Length);
    }

    [Fact]
    public async Task MixedCompositeManifest_PreservesConcreteSiblingsBesideDeferredChildren()
    {
        await Task.Yield();
        using var block = new TransformerEncoderBlock<double>(
            hiddenSize: 24,
            numHeads: 4,
            ffnDim: 48,
            dropoutRate: 0);

        // Attention and normalization are construction-sized. The two Dense FFN children are
        // honestly deferred until a real sequence shape reaches the block.
        var parameters = block.GetParameters();
        var layout = new AiDotNet.Models.Parameters.ParameterLayoutSnapshot(
            block.GetParameterLayout());

        Assert.True(parameters.Length > 0);
        Assert.Equal(AiDotNet.Models.Parameters.ParameterReadiness.ShapeDeferred, layout.Readiness);
        Assert.Equal(parameters.Length, layout.KnownParameterCount);
        Assert.Contains(layout.Slots, slot => slot.Readiness ==
            AiDotNet.Models.Parameters.ParameterReadiness.ShapeDeferred);
        Assert.Contains(layout.Slots, slot => slot.ParameterCount > 0);
    }

    [Fact]
    public async Task ParameterShapeReadyLayer_MaterializesWithDynamicDataAxes()
    {
        await Task.Yield();
        using var layer = ConvolutionalLayer<double>.WithInputDepth(
            inputDepth: 3,
            outputDepth: 8,
            kernelSize: 3);

        Assert.False(layer.IsShapeResolved);
        long declared = layer.ParameterCount;
        Assert.True(declared > 0);
        Assert.Equal(declared, layer.GetParameters().Length);
    }

    [Fact]
    public async Task ParameterCount_UsesDeclaredShapesBeforeTrainAndRemainsStableAfterTrain()
    {
        await Task.Yield();

        // NeRF's topology is non-sequential, but every parameter dimension follows from its
        // architecture. The model hook resolves that topology without allocating weights, so the
        // generated manifest can publish the final count before the first forward.
        var model = BuildNeRF();
        long freshCount = model.ParameterCount;
        Assert.True(freshCount > 0);
        Assert.True(model.HasUninitializedParameters);

        // First Train call materializes DenseLayer inputs via positional encoding
        // (3 → 60 for pos, 3 → 24 for dir) + skip-concat. Layer weight matrices resize.
        var (x, y) = DummyBatch(32);
        model.Train(x, y);

        // Training materializes values but must not change a constructor-known structural count.
        // Pre-#1832 the cache and the materialized vector described different layouts.
        long resolvedCount = model.ParameterCount;
        Assert.Equal(freshCount, resolvedCount);

        // ParameterCount and GetParameters().Length MUST agree post-resolution — that's the
        // invariant SetParameters relies on for length validation.
        Assert.Equal(resolvedCount, (long)model.GetParameters().Length);
    }

    [Fact]
    public async Task AutomaticShapeResolution_UnblocksFlatVectorRoundTrip()
    {
        await Task.Yield();

        // Train a model briefly to materialize its lazy shapes + move some weights.
        var trained = BuildNeRF();
        var (x, y) = DummyBatch(32, seed: 1);
        trained.Train(x, y);
        var savedParams = trained.GetParameters();

        // A fresh sibling reports the same structural width before allocating values. Restore is
        // therefore warm-up-free: model creators do not need a parameter or shape override and
        // users do not need to know which sample tensor would initialize the graph.
        var fresh = BuildNeRF();
        Assert.Equal(savedParams.Length, fresh.ParameterCount);
        Assert.True(fresh.HasUninitializedParameters);

        // NeRF declares its real non-sequential topology in ResolveLazyLayerShapes, so restore can
        // materialize that known layout on demand without requiring a sample input first.
        fresh.SetParameters(savedParams);
        Assert.Equal(savedParams.Length, fresh.GetParameters().Length);

        // ResolveShapes remains a safe, idempotent operation for callers that use it uniformly
        // across architecture-known and genuinely data-dependent models.
        var sample = new Tensor<float>(new[] { 1, 6 }, new Vector<float>(new float[6]));
        fresh.ResolveShapes(sample);
        Assert.Equal(savedParams.Length, fresh.GetParameters().Length);

        // Now the flat-vector round-trip works.
        fresh.SetParameters(savedParams);

        // And the reloaded params match bit-for-bit.
        var reloaded = fresh.GetParameters();
        Assert.Equal(savedParams.Length, reloaded.Length);
        for (int i = 0; i < savedParams.Length; i++)
            Assert.Equal(savedParams[i], reloaded[i]);
    }

    [Fact]
    public void ResolveShapes_NullSampleInput_ThrowsArgumentNullException()
    {
        var model = BuildNeRF();
        Assert.Throws<ArgumentNullException>(() => model.ResolveShapes(null!));
    }

    [Fact]
    public async Task SetParameters_EmptyTrulyShapeDeferredCheckpoint_DoesNotChangeLayoutMidRestore()
    {
        await Task.Yield();

        // No architecture supplies an input width here, so these layers are genuinely deferred.
        // In contrast, BuildLazyDenseNetwork declares inputSize=3 and must now publish/materialize
        // its complete generated parameter surface without a warm-up forward.
        using var source = new DenseLayer<double>(
            4, (IActivationFunction<double>)new ReLUActivation<double>());
        using var target = new DenseLayer<double>(
            4, (IActivationFunction<double>)new ReLUActivation<double>());
        var checkpoint = source.GetParameters();

        Assert.Empty(checkpoint);
        target.SetParameters(checkpoint);
        Assert.Empty(target.GetParameters());
    }

    [Fact]
    public void SetParameters_MaterializedCheckpoint_UsesActualVectorManifest()
    {
        var source = BuildLazyDenseNetwork();
        var target = BuildLazyDenseNetwork();
        source.MaterializeParameters();
        var checkpoint = source.GetParameters();

        Assert.NotEmpty(checkpoint);
        target.SetParameters(checkpoint);

        var restored = target.GetParameters();
        Assert.Equal(checkpoint.Length, restored.Length);
        for (int i = 0; i < checkpoint.Length; i++)
            Assert.Equal(checkpoint[i], restored[i]);
    }

    [Fact]
    public void ShapeContractFallback_DoesNotOverrideRejectedInputGeometry()
    {
        // This is the model-agnostic shape pattern used by SALMONN and similar projection-based
        // architectures. The public input is [..., 4], but the first projection changes the
        // internal feature width to 8 before attention sees it.
        var downstreamProjection = new FullyConnectedLayer<double>(16);
        using var network = new NeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: 3,
                inputWidth: 4,
                outputSize: 8,
                layers:
                [
                    new FullyConnectedLayer<double>(8),
                    new LayerNormalizationLayer<double>(),
                    new FullyConnectedLayer<double>(8),
                    new LayerNormalizationLayer<double>(),
                    new MultiHeadAttentionLayer<double>(headCount: 2, headDimension: 4),
                    new LayerNormalizationLayer<double>(),
                    downstreamProjection,
                    new FullyConnectedLayer<double>(8)
                ]));

        // The speculative architecture walk reaches attention with two invalid candidates:
        // rank-1 [8] and stale [..., 4]. Its declarative output must not overrule either input
        // rejection and pin downstreamProjection to the stale width.
        network.SetTrainingMode(false);
        Assert.False(downstreamProjection.IsShapeResolved);

        var output = network.Predict(new Tensor<double>([1, 3, 4]));

        Assert.Equal(new[] { 1, 3, 8 }, output.Shape.ToArray());
        Assert.True(downstreamProjection.IsShapeResolved);
        Assert.Equal(8, downstreamProjection.GetInputShape()[0]);
    }

    [Fact]
    public void Dispose_DoesNotReadParameterCountOrMaterializeLazyWeights()
    {
        var layer = new ParameterCountBombLayer();
        var model = new NeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 1,
                outputSize: 1,
                layers: [layer]));
        layer.ParameterCountReads = 0;
        layer.ThrowOnParameterCountRead = true;

        model.Dispose();

        Assert.Equal(0, layer.ParameterCountReads);
    }

    [Fact]
    public void DeclarativeContract_CannotOverrideImperativeShapeRejection()
    {
        var model = BuildLazyDenseNetwork();
        using var layer = new RejectingDeclaredLayer();
        int[] running = [32];
        int[] lastGood = running;
        object?[] arguments = [layer, running, lastGood];

        var advance = typeof(NeuralNetworkBase<double>).GetMethod(
            "TryAdvanceLayerShape",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);

        Assert.NotNull(advance);
        bool advanced = (bool)advance!.Invoke(model, arguments)!;

        Assert.False(advanced);
        Assert.Equal(new[] { 32 }, (int[])arguments[1]!);
        Assert.False(layer.IsShapeResolved);
    }

    [ElementWiseShape]
    private sealed class ParameterCountBombLayer : LayerBase<double>
    {
        public ParameterCountBombLayer() : base([1], [1]) { }

        public int ParameterCountReads { get; set; }
        public bool ThrowOnParameterCountRead { get; set; }

        public override long ParameterCount
        {
            get
            {
                ParameterCountReads++;
                if (ThrowOnParameterCountRead)
                    throw new InvalidOperationException("Dispose read ParameterCount.");
                return 0;
            }
        }

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;
        public override bool SupportsTraining => false;
        public override void ResetState() { }
    }

    [TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed class RejectingDeclaredLayer : LayerBase<double>, IShapeContract
    {
        public RejectingDeclaredLayer() : base([-1], [-1])
        {
        }

        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
            => inputRank == 1
                ? [new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(256))]
                : null;

        protected override void OnFirstForward(Tensor<double> input)
            => throw new InvalidOperationException("The imperative implementation rejects this shape.");

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;

        public override bool SupportsTraining => false;

        public override void ResetState()
        {
        }
    }
}
