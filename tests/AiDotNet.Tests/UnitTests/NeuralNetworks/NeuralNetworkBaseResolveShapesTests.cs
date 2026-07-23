using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralRadianceFields.Models;
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
    public void ParameterCount_ReflectsLazyShapeResolutionAfterTrain()
    {
        // Fresh model has lazy input shapes ([1] sentinels for the position/direction
        // input Dense layers). Its parameter count is the pre-resolution total.
        var model = BuildNeRF();
        long freshCount = model.ParameterCount;
        Assert.True(freshCount > 0,
            "Fresh model should have a positive ParameterCount from architecture-based " +
            "shape resolution (layers propagate via ResolveLazyLayerShapes at construction).");

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

        // Pre-ResolveShapes: SetParameters MUST reject the mismatched size with a message
        // that points the caller at ResolveShapes as the fix.
        var mismatch = Assert.Throws<ArgumentException>(() => fresh.SetParameters(savedParams));
        Assert.Contains("ResolveShapes", mismatch.Message);

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
}
