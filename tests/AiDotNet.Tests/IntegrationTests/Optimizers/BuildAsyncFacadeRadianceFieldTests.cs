using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Facade-entry integration tests for issue #1826 — passing a radiance-field
/// model (<see cref="NeRF{T}"/>, <see cref="GaussianSplatting{T}"/>,
/// <see cref="InstantNGP{T}"/>) into
/// <c>AiModelBuilder&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;
/// .ConfigureModel(...).ConfigureDataLoader(...).BuildAsync()</c> used to
/// throw <see cref="System.ArgumentOutOfRangeException"/>
/// ("Feature index N exceeds the input dimension 1") from
/// <c>NeuralNetworkBase.SetActiveFeatureIndices</c> during optimizer
/// feature-selection. The root cause was
/// <c>OptimizerBase.HasNonFlatNeuralInput</c> checking only the first
/// layer's shape rank — radiance fields feed their <c>[N, 6]</c> input
/// through a positional-encoding stage into a 1-D vector before the first
/// dense layer, so the shape-rank guard failed to exclude them, the
/// optimizer treated the 6-column tensor as tabular features, and the
/// feature-selection path blew up on flat indices past the (encoded)
/// first-layer input dimension.
///
/// <para>The fix extends <c>HasNonFlatNeuralInput</c> to also return true
/// for any <see cref="AiDotNet.NeuralRadianceFields.Interfaces.IRadianceField{T}"/>.
/// These tests pin that contract at the facade surface — the same surface
/// every documented AiDotNet consumer (and every downstream training-demo
/// / weekly-project code) actually uses.</para>
/// </summary>
[Collection("NonParallelIntegration")]
public class BuildAsyncFacadeRadianceFieldTests
{
    private readonly ITestOutputHelper _output;

    public BuildAsyncFacadeRadianceFieldTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // Tiny fixture — enough to exercise the facade path end-to-end without
    // spending real training time. The feature-selection bug fired on the
    // very first PrepareAndEvaluateSolutionCore call, so we don't need to
    // train to convergence to prove the fix.
    private const int SampleCount = 64;

    private static (Tensor<float> input, Tensor<float> target) SmallCubeBatch(int seed)
    {
        var rng = new System.Random(seed);
        var input = new float[SampleCount * 6];
        var target = new float[SampleCount * 4];
        for (int i = 0; i < SampleCount; i++)
        {
            // Position ∈ [-0.7, 0.7]³
            float x = (float)(rng.NextDouble() * 1.4 - 0.7);
            float y = (float)(rng.NextDouble() * 1.4 - 0.7);
            float z = (float)(rng.NextDouble() * 1.4 - 0.7);
            // Unit-length viewing direction
            float dx, dy, dz, ln;
            do
            {
                dx = (float)(rng.NextDouble() * 2 - 1);
                dy = (float)(rng.NextDouble() * 2 - 1);
                dz = (float)(rng.NextDouble() * 2 - 1);
                ln = System.MathF.Sqrt(dx * dx + dy * dy + dz * dz);
            } while (ln < 1e-4f);
            dx /= ln; dy /= ln; dz /= ln;
            input[i * 6 + 0] = x;  input[i * 6 + 1] = y;  input[i * 6 + 2] = z;
            input[i * 6 + 3] = dx; input[i * 6 + 4] = dy; input[i * 6 + 5] = dz;

            // Colored-cube target: rgb = 0.5, density = 4 if inside unit cube else 0
            bool inside = System.MathF.Abs(x) <= 0.5f && System.MathF.Abs(y) <= 0.5f && System.MathF.Abs(z) <= 0.5f;
            target[i * 4 + 0] = 0.5f;
            target[i * 4 + 1] = 0.5f;
            target[i * 4 + 2] = 0.5f;
            target[i * 4 + 3] = inside ? 4f : 0f;
        }
        return (new Tensor<float>(new[] { SampleCount, 6 }, new Vector<float>(input)),
                new Tensor<float>(new[] { SampleCount, 4 }, new Vector<float>(target)));
    }

    /// <summary>
    /// #1826 — the exact reproducer from the issue. Before the fix,
    /// BuildAsync throws mid-way through the optimizer's feature-selection
    /// pass. After the fix, BuildAsync returns a result and the NeRF's
    /// parameters have moved from their random init.
    /// </summary>
    [Fact(Timeout = 300_000)]
    public async Task BuildAsync_NeRF_ConfigureModel_DoesNotThrowFromFeatureSelection()
    {
        var nerf = new NeRF<float>(
            positionEncodingLevels: 4,
            directionEncodingLevels: 2,
            hiddenDim: 32,
            numLayers: 2,
            colorHiddenDim: 16,
            colorNumLayers: 1,
            useHierarchicalSampling: false,
            renderSamples: 8,
            renderNearBound: 1.0,
            renderFarBound: 4.0,
            learningRate: 1e-3);

        var (xTrain, yTrain) = SmallCubeBatch(seed: 1826);

        // Snapshot the initial parameters so we can verify BuildAsync
        // actually reached the optimizer (proves the fix went beyond
        // "no throw" and the training path is now live).
        var before = nerf.GetParameters();
        var beforeCopy = new float[before.Length];
        for (int i = 0; i < before.Length; i++) beforeCopy[i] = before[i];

        var loader = DataLoaders.FromTensors(xTrain, yTrain);

        // This is the exact call that used to throw
        // "Feature index N exceeds the input dimension 1" from
        // NeuralNetworkBase.SetActiveFeatureIndices during
        // NormalOptimizer.Optimize's feature-selection pass.
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(nerf)
            .BuildAsync();

        Assert.NotNull(result);

        // The passed-in NeRF instance should have been trained in place.
        // Some parameters must have moved from their init values.
        var after = nerf.GetParameters();
        Assert.Equal(before.Length, after.Length);

        int movedCount = 0;
        double maxDelta = 0;
        for (int i = 0; i < after.Length; i++)
        {
            float delta = System.MathF.Abs(after[i] - beforeCopy[i]);
            if (delta > 1e-8f) movedCount++;
            if (delta > maxDelta) maxDelta = delta;
        }
        _output.WriteLine($"NeRF params: {after.Length}, moved: {movedCount}, max Δ = {maxDelta:E3}");
        Assert.True(movedCount > 0,
            "AiModelBuilder.BuildAsync returned without throwing but no NeRF parameters " +
            "moved from init — the facade path is silently dropping the optimizer's " +
            "parameter updates for radiance-field models.");
    }

    /// <summary>
    /// #1826 — same contract for <see cref="InstantNGP{T}"/>, the third shipping
    /// radiance-field model. Like NeRF it is non-sequential (a multiresolution
    /// hash encoding feeds the density MLP; the colour MLP consumes the feature
    /// vector concatenated with the view direction), so it needs the same
    /// through-topology lazy-shape resolution for the facade's post-train
    /// parameter writeback to reach a freshly-constructed instance.
    /// </summary>
    [Fact(Timeout = 300_000)]
    public async Task BuildAsync_InstantNGP_ConfigureModel_DoesNotThrowFromFeatureSelection()
    {
        var ngp = new InstantNGP<float>(
            hashTableSize: 4096,
            numLevels: 4,
            featuresPerLevel: 2,
            finestResolution: 256,
            coarsestResolution: 16,
            mlpHiddenDim: 16,
            mlpNumLayers: 2,
            occupancyGridResolution: 16,
            learningRate: 1e-2);

        var (xTrain, yTrain) = SmallCubeBatch(seed: 18262);

        var before = ngp.GetParameters();
        var beforeCopy = new float[before.Length];
        for (int i = 0; i < before.Length; i++) beforeCopy[i] = before[i];

        var loader = DataLoaders.FromTensors(xTrain, yTrain);
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(ngp)
            .BuildAsync();

        Assert.NotNull(result);

        var after = ngp.GetParameters();
        Assert.Equal(before.Length, after.Length);
        int movedCount = 0;
        for (int i = 0; i < after.Length; i++)
            if (System.MathF.Abs(after[i] - beforeCopy[i]) > 1e-8f) movedCount++;
        _output.WriteLine($"InstantNGP params: {after.Length}, moved: {movedCount}");
        Assert.True(movedCount > 0,
            "AiModelBuilder.BuildAsync returned without throwing but no InstantNGP " +
            "parameters moved from init.");
    }

    /// <summary>
    /// #1826 — same contract for <see cref="GaussianSplatting{T}"/>, which
    /// implements <c>IRadianceField&lt;T&gt;</c> and hit the identical
    /// feature-selection failure. The <c>IRadianceField&lt;T&gt;</c> guard
    /// covers all three shipping radiance-field models at once.
    /// </summary>
    [Fact(Timeout = 300_000)]
    public async Task BuildAsync_GaussianSplatting_ConfigureModel_DoesNotThrowFromFeatureSelection()
    {
        var gs = new GaussianSplatting<float>();
        var (xTrain, yTrain) = SmallCubeBatch(seed: 18261);

        var before = gs.GetParameters();
        var beforeCopy = new float[before.Length];
        for (int i = 0; i < before.Length; i++) beforeCopy[i] = before[i];

        var loader = DataLoaders.FromTensors(xTrain, yTrain);
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(gs)
            .BuildAsync();

        Assert.NotNull(result);

        var after = gs.GetParameters();
        Assert.Equal(before.Length, after.Length);
        int movedCount = 0;
        for (int i = 0; i < after.Length; i++)
            if (System.MathF.Abs(after[i] - beforeCopy[i]) > 1e-8f) movedCount++;
        _output.WriteLine($"GS params: {after.Length}, moved: {movedCount}");
        Assert.True(movedCount > 0,
            "AiModelBuilder.BuildAsync returned without throwing but no GaussianSplatting " +
            "parameters moved from init.");
    }
}
