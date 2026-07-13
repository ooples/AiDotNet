using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Per Gong et al. 2019 "SpiralNet++: A Fast and Highly Efficient Mesh Convolution
/// Operator" (arXiv 1911.05856), the model processes 3D mesh data as a rank-3
/// tensor of shape [batch, num_vertices, in_features]. The paper's CoMA / face-mesh
/// experiments use 5023 vertices × 3 coordinate features per vertex; the
/// SpiralNetOptions defaults to a 64-vertex fallback mesh for small-input testing
/// (NumVertices = 64, InputFeatures = 3 = xyz coords).
/// </summary>
public class SpiralNetTests : NeuralNetworkModelTestBase<float>
{
    // [batch, num_vertices, in_features] matches SpiralNetOptions defaults
    // (NumVertices = 64, InputFeatures = 3). The base's default [1, 4] rank-2
    // shape feeds a rank-2 tensor into GlobalPoolingLayer (which requires
    // rank-3, rank-4, or rank-5), producing ArgumentException at line 236
    // of GlobalPoolingLayer.OnFirstForward.
    protected override int[] InputShape => [1, 64, 3];

    // Default output: ModelNet40 classification (NumClasses = 40 per Gong et al.).
    protected override int[] OutputShape => [1, 40];

    // Pin per-layer weight init (and the Dropout RNG derived from it) to a fixed thread-local seed so
    // construction is reproducible. SpiralNet's architecture carries no explicit RandomSeed, so its
    // init + Dropout(0.5) otherwise derive from the process-shared, order-dependent
    // RandomHelper.ThreadSafeRandom; under xUnit's PARALLEL execution that makes MoreData_ShouldNotDegrade
    // nondeterministic (flakes under a loaded shard, passes in isolation) against its tight 2e-3
    // tolerance. AmbientFallbackSeed is [ThreadStatic] → parallel-safe (each worker independent).
    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var previousSeed = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed;
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed = 1337;
        try { return new SpiralNet<float>(); }
        finally { AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed = previousSeed; }
    }

    // MoreData compares a 50-iter run on ONE random sample against a 200-iter run on a
    // DIFFERENT random sample (absolute bound: lossLong <= lossShort + tolerance). This
    // is NOT a dropout-nondeterminism flake: init is seeded (AmbientFallbackSeed) and the
    // Dropout(0.5) mask is per-instance seeded (DropoutLayer derives its mask from the
    // layer RandomSeed + an instance call-counter, closing #1383), so the result is
    // effectively deterministic — two back-to-back local runs measured 0.353176/0.351734
    // and 0.353176/0.351731 (run-to-run delta ~3e-6, negligible).
    //
    // The real drift is intrinsic to comparing the ~0.35 plateau of two DIFFERENT
    // samples, and it is platform-dependent through BLAS/FP ordering: ~1.44e-3 locally
    // (Windows) vs 2.675e-3 on Linux CI (0.353377 vs 0.350702 there) — where the old
    // 2e-3 tolerance failed. It is inherent sample-difficulty noise, not divergence
    // (LossStrictlyDecreasesOnMemorizationTask confirms the loss decreases). Set the
    // tolerance to 5e-3 — ~1.9x the observed Linux drift, enough headroom for other CI
    // runners' FP ordering while a genuinely diverging optimizer (lossLong up O(0.05-0.5)
    // or NaN) still fails hard (the NaN guard above + strict-decrease stay load-bearing).
    protected override double MoreDataTolerance => 5e-3;

    /// <summary>
    /// SpiralConvLayer is lazy — its weight tensor is constructed at [0, 0] in
    /// the ctor and only resolves to its final [outputChannels, inputChannels ×
    /// spiralLength] shape during the first Forward (OnFirstForward at
    /// src/NeuralNetworks/Layers/SpiralConvLayer.cs:485 reads input.Shape to
    /// determine InputChannels). The base
    /// <c>NeuralNetworkBase.ParameterCount</c> calls
    /// <c>ResolveLazyLayerShapes</c> which propagates the architecture's input
    /// shape through generic Dense/Conv chains, but SpiralConv's
    /// vertex-features input contract <c>[B, V, C]</c> doesn't fit that
    /// propagation (the chain expects flat-feature layers), so the lazy
    /// layers stay at length 0 pre-Forward and <c>ParameterCount</c>
    /// returns 0. Override the test with an explicit warm-up Predict to
    /// materialize the weights before the count is read, matching the
    /// pattern used in <c>Training_ShouldChangeParameters</c>.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var input = CreateRandomTensor(InputShape, rng);
        // Warm-up Predict materializes lazy SpiralConvLayer weights.
        network.Predict(input);
        Assert.True(network.ParameterCount > 0,
            "Neural network should have learnable parameters after warm-up Predict.");
    }
}
