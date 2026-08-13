using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ActivationFunctions;
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
/// The fix has two arms:
///   1. <c>ParameterCount</c> sums per-layer counts fresh on every access — never goes stale.
///   2. Public <c>ResolveShapes(sampleInput)</c> method lets callers materialize lazy layers
///      up-front so a fresh model's <c>SetParameters</c> sees the same size the trained
///      model's <c>GetParameters</c> returned.
///
/// These tests pin both arms.
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
    public void ParameterCount_ReflectsLazyShapeResolutionAfterTrain()
    {
        // A fresh model has resolved shapes but deliberately unmaterialized weights. Reads stay
        // allocation-free, so the concrete count is honestly zero and readiness carries the fact
        // that parameters are still pending.
        var model = BuildNeRF();
        long freshCount = model.ParameterCount;
        Assert.Equal(0, freshCount);
        Assert.True(model.HasUninitializedParameters);

        // First Train call materializes DenseLayer inputs via positional encoding
        // (3 → 60 for pos, 3 → 24 for dir) + skip-concat. Layer weight matrices resize.
        var (x, y) = DummyBatch(32);
        model.Train(x, y);

        // Post-train ParameterCount MUST reflect the resolved sizes. Pre-#1832 the cache
        // stuck at freshCount and this assertion failed.
        long resolvedCount = model.ParameterCount;
        Assert.True(resolvedCount > freshCount,
            $"ParameterCount should grow after lazy shape resolution during first Train " +
            $"(fresh={freshCount}, resolved={resolvedCount}). If they're equal, the stale " +
            $"cache from pre-#1832 is back.");

        // ParameterCount and GetParameters().Length MUST agree post-resolution — that's the
        // invariant SetParameters relies on for length validation.
        Assert.Equal(resolvedCount, (long)model.GetParameters().Length);
    }

    [Fact]
    public void ResolveShapes_UnblocksFlatVectorRoundTrip()
    {
        // Train a model briefly to materialize its lazy shapes + move some weights.
        var trained = BuildNeRF();
        var (x, y) = DummyBatch(32, seed: 1);
        trained.Train(x, y);
        var savedParams = trained.GetParameters();

        // Fresh sibling — same architecture, no Train yet, lazy shapes still un-materialized.
        var fresh = BuildNeRF();
        Assert.True(fresh.GetParameters().Length < savedParams.Length,
            "Fresh sibling should have FEWER parameters than the trained model (lazy layers " +
            "not yet resolved). If they're equal, the test fixture broke.");

        // NeRF declares its real non-sequential topology in ResolveLazyLayerShapes, so restore can
        // materialize that known layout on demand without requiring a sample input first.
        fresh.SetParameters(savedParams);
        Assert.Equal(savedParams.Length, fresh.GetParameters().Length);

        // ResolveShapes with a sample input drives one forward pass to materialize the
        // lazy layers. After it returns, the fresh model's size matches the trained one.
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
    public void SetParameters_EmptyLazyCheckpoint_DoesNotChangeLayoutMidRestore()
    {
        var source = BuildLazyDenseNetwork();
        var target = BuildLazyDenseNetwork();
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
}
