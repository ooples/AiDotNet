using System;
using System.Linq;
using AiDotNet.NeuralRadianceFields.Encodings;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields.NeuS;

/// <summary>
/// The hash-grid SDF backbone of instant-nsr-pl (Wonder3D's reconstruction): tiny-cuda-nn's grid layout and spatial
/// hash, the trilinear lookup against a hand blend, its spatial Jacobian and the SDF gradient against central
/// differences, the progressive band, the constant-then-exponential schedule and the dynamic ray batch rule.
/// </summary>
public class NeuSHashGridTests
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    private static double[] Values(Tensor<double> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor[i];
        return values;
    }

    private static void Fill(MultiresolutionHashGrid<double> grid, double scale)
    {
        for (int l = 0; l < grid.Levels; l++)
        {
            var span = grid.Tables[l].AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = scale * Math.Sin(0.37 * (i + 1) + l);
        }
    }

    [Fact]
    public void Levels_FollowTinyCudaNnScaleResolutionAndTableSize()
    {
        var grid = new MultiresolutionHashGrid<double>(3, 2, 8, 4, 1.5, RandomHelper.CreateSeededRandom(1));
        // s_l = 4 · 1.5^l − 1, R = ceil(s) + 1, rows = min(2^8, R³ rounded up to 8).
        Assert.Equal(new[] { 3.0, 5.0, 8.0 }, grid.Scales);
        Assert.Equal(new[] { 4, 6, 9 }, grid.Resolutions);
        Assert.Equal(new[] { 64, 216, 256 }, grid.TableSizes);
        Assert.Equal((64 + 216 + 256) * 2, grid.ParameterCount);
        Assert.All(grid.Tables.SelectMany(t => Values(t)), v => Assert.InRange(v, -1e-4, 1e-4));
    }

    [Fact]
    public void VertexIndex_IsDenseWhileTheGridFitsAndHashedOtherwise()
    {
        // Dense: x + y·R + z·R².
        Assert.Equal(1 + 2 * 4 + 3 * 16, MultiresolutionHashGrid<double>.VertexIndex(1, 2, 3, 4, 64));
        // Hashed: (1 · 1) xor (2 · 2654435761) xor (3 · 805459861) = 2892625372 (mod 2^32), then mod 64 = 28.
        Assert.Equal(28, MultiresolutionHashGrid<double>.VertexIndex(1, 2, 3, 100, 64));
    }

    [Fact]
    public void Encode_IsTheTrilinearBlendOfTheCornerRows()
    {
        var grid = new MultiresolutionHashGrid<double>(1, 2, 19, 4, 1.0, RandomHelper.CreateSeededRandom(2));
        Fill(grid, 1.0);
        var table = Values(grid.Tables[0]);
        var u = new[] { 0.3, 0.6, 0.1 };
        var (features, _) = grid.Encode(Engine, u, withJacobian: false);
        var actual = Values(features);

        // Scale 3: grid position 3u + 0.5 = (1.4, 2.3, 0.8), cell (1, 2, 0), fractions (0.4, 0.3, 0.8).
        int[] lower = { 1, 2, 0 };
        double[] frac = { 0.4, 0.3, 0.8 };
        for (int f = 0; f < 2; f++)
        {
            double expected = 0;
            for (int k = 0; k < 8; k++)
            {
                int ox = (k >> 2) & 1, oy = (k >> 1) & 1, oz = k & 1;
                double w = (ox == 1 ? frac[0] : 1 - frac[0]) * (oy == 1 ? frac[1] : 1 - frac[1]) * (oz == 1 ? frac[2] : 1 - frac[2]);
                int row = (lower[0] + ox) + (lower[1] + oy) * 4 + (lower[2] + oz) * 16;
                expected += w * table[row * 2 + f];
            }

            Assert.Equal(expected, actual[f], 1e-12);
        }
    }

    [Fact]
    public void Encode_JacobianMatchesCentralDifferences()
    {
        var grid = new MultiresolutionHashGrid<double>(3, 2, 6, 4, 1.7, RandomHelper.CreateSeededRandom(3));
        Fill(grid, 0.5);
        var u = new[] { 0.31, 0.62, 0.17, 0.83, 0.44, 0.58 };
        var (_, jacobian) = grid.Encode(Engine, u, withJacobian: true);
        Assert.NotNull(jacobian);
        if (jacobian is null) return;
        const double h = 1e-6;
        for (int d = 0; d < 3; d++)
        {
            var plus = (double[])u.Clone();
            var minus = (double[])u.Clone();
            for (int p = 0; p < 2; p++)
            {
                plus[p * 3 + d] += h;
                minus[p * 3 + d] -= h;
            }

            var fp = Values(grid.Encode(Engine, plus, false).Features);
            var fm = Values(grid.Encode(Engine, minus, false).Features);
            var analytic = Values(jacobian[d]);
            for (int i = 0; i < analytic.Length; i++) Assert.Equal((fp[i] - fm[i]) / (2 * h), analytic[i], 1e-6);
        }
    }

    [Fact]
    public void ProgressiveBand_SwitchesOnOneLevelPerUpdateInterval()
    {
        var options = new NeuSOptions
        {
            SdfBackbone = NeuSSdfBackbone.HashGrid, HashGridLevels = 10, HashGridLog2TableSize = 8, HashGridBaseResolution = 4,
            HashGridPerLevelScale = 1.2, HashGridStartLevel = 4, HashGridUpdateIterations = 1000, SdfHiddenDim = 8,
            SdfHiddenLayers = 1, SdfFeatureDim = 2, ColorHiddenDim = 8, ColorHiddenLayers = 1, Seed = 1,
        };
        var field = new NeuSField<double>(options, RandomHelper.CreateSeededRandom(1), Engine);
        var grid = field.HashGrid;
        Assert.NotNull(grid);
        if (grid is null) return;
        Assert.Equal(4, grid.ActiveLevels);
        foreach (var (iteration, levels) in new[] { (999, 4), (1000, 5), (2500, 6), (99999, 10) })
        {
            field.UpdateHashGridLevels(iteration);
            Assert.Equal(levels, grid.ActiveLevels);
        }

        // Inactive levels output zero.
        field.UpdateHashGridLevels(0);
        var features = Values(grid.Encode(Engine, new[] { 0.3, 0.4, 0.5 }, false).Features);
        Assert.All(features.Skip(4 * 2), v => Assert.Equal(0.0, v));
    }

    [Fact]
    public void HashGridSdf_AnalyticGradientMatchesCentralDifferences()
    {
        var options = new NeuSOptions
        {
            SdfBackbone = NeuSSdfBackbone.HashGrid, HashGridLevels = 3, HashGridLog2TableSize = 8, HashGridBaseResolution = 4,
            HashGridPerLevelScale = 1.5, SdfHiddenDim = 16, SdfHiddenLayers = 1, SdfFeatureDim = 4, ColorHiddenDim = 8,
            ColorHiddenLayers = 1, GeometricInit = false, Seed = 4,
        };
        var field = new NeuSField<double>(options, RandomHelper.CreateSeededRandom(4), Engine);
        var grid = field.HashGrid;
        Assert.NotNull(grid);
        if (grid is null) return;
        Fill(grid, 0.8);
        // Hand parameter count: tables + Linear(3 + 6, 16) + Linear(16, 5), weight-normalized (v, g, b).
        int linear = (16 * 9 + 16 + 16) + (5 * 16 + 5 + 5);
        Assert.Equal(grid.ParameterCount + linear, field.GeometryParameters.Sum(p => p.Length));

        var points = new[] { 0.12, -0.31, 0.27, -0.44, 0.18, 0.05 };
        var gradient = Values(field.QueryNetwork(Engine, new Tensor<double>(new[] { 2, 3 }, new Vector<double>(points)), true).Gradient
            ?? throw new InvalidOperationException("No gradient."));
        const double h = 1e-6;
        for (int p = 0; p < 2; p++)
            for (int d = 0; d < 3; d++)
            {
                var plus = (double[])points.Clone();
                var minus = (double[])points.Clone();
                plus[p * 3 + d] += h;
                minus[p * 3 + d] -= h;
                double fp = field.QuerySdf(new Tensor<double>(new[] { 2, 3 }, new Vector<double>(plus)))[p];
                double fm = field.QuerySdf(new Tensor<double>(new[] { 2, 3 }, new Vector<double>(minus)))[p];
                Assert.Equal((fp - fm) / (2 * h), gradient[p * 3 + d], 1e-5);
            }
    }

    [Fact]
    public void ConstantThenExponential_HoldsThenDecaysToAlpha()
    {
        var options = new NeuSOptions
        {
            LearningRateSchedule = NeuSLearningRateSchedule.ConstantThenExponential, Iterations = 3000,
            ConstantLearningRateIterations = 500, LearningRateAlpha = 0.1,
        };
        Assert.Equal(1.0, options.LearningRateFactor(0), 1e-12);
        Assert.Equal(1.0, options.LearningRateFactor(500), 1e-12);
        Assert.Equal(Math.Pow(0.1, 1.0 / 2500), options.LearningRateFactor(501), 1e-12);
        Assert.Equal(Math.Sqrt(0.1), options.LearningRateFactor(1750), 1e-12);
        Assert.Equal(0.1, options.LearningRateFactor(3000), 1e-12);
    }

    [Fact]
    public void DynamicBatch_FollowsInstantNsrPlsRule()
    {
        // Budget 256 rays · 1024 samples; this NeuS evaluates 128 samples per ray.
        long budget = 256L * 1024;
        // int(256 · 262144 / 32768) = 2048; int(0.9 · 256 + 0.1 · 2048) = 435.
        int first = NeuSReconstructor<double>.NextDynamicBatchSize(256, budget, 256L * 128, 8192);
        Assert.Equal(435, first);
        // int(435 · 262144 / 55680) = 2048; int(391.5 + 204.8) = 596.
        Assert.Equal(596, NeuSReconstructor<double>.NextDynamicBatchSize(first, budget, first * 128L, 8192));
        // Capped at max_train_num_rays.
        Assert.Equal(8192, NeuSReconstructor<double>.NextDynamicBatchSize(8000, budget * 100, 8000L * 128, 8192));
        // Shrinks when the samples overshoot: int(0.9 · 1000 + 0.1 · int(1000 · 0.5)) = 950.
        Assert.Equal(950, NeuSReconstructor<double>.NextDynamicBatchSize(1000, 1000, 2000, 8192));
    }

    [Fact]
    public void Reconstructor_WithAdamWGroupsAndDynamicBatch_TrainsAndGrowsTheBatch()
    {
        var options = new NeuSOptions
        {
            SdfBackbone = NeuSSdfBackbone.HashGrid, HashGridLevels = 2, HashGridLog2TableSize = 8, HashGridBaseResolution = 4,
            HashGridPerLevelScale = 1.5, HashGridStartLevel = 1, HashGridUpdateIterations = 1, SdfHiddenDim = 16,
            SdfHiddenLayers = 1, SdfFeatureDim = 4, SdfSkipLayers = Array.Empty<int>(), ColorHiddenDim = 8,
            ColorHiddenLayers = 1, ViewFrequencies = 1, Samples = 8, ImportanceSamples = 0, UpSampleSteps = 1,
            Optimizer = NeuSOptimizer.AdamW, LearningRateSchedule = NeuSLearningRateSchedule.ConstantThenExponential,
            LearningRate = 1e-3, ColorLearningRate = 1e-2, VarianceLearningRate = 1e-3, AdamBeta2 = 0.99,
            AdamEpsilon = 1e-15, ConstantLearningRateIterations = 1, LearningRateAlpha = 0.1, Iterations = 3,
            BatchSize = 8, DynamicRaySampling = true, MaxBatchSize = 32, SamplesPerRayBudget = 64, WarmUpIterations = 0,
            Seed = 5,
        };
        var reconstructor = new NeuSReconstructor<double>(options, Engine);
        var camera = NeuSCamera.Orthographic(new[] { 0.0, -1.3, 0.0 }, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }, 1.0, 4, 4);
        var image = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(Enumerable.Range(0, 48).Select(i => 0.5 + 0.3 * Math.Sin(i)).ToArray()));
        var view = new NeuSView<double>(image, camera);
        var before = reconstructor.Field.Parameters.SelectMany(t => Values(t)).ToArray();
        reconstructor.Fit(new[] { view }, 3);
        var after = reconstructor.Field.Parameters.SelectMany(t => Values(t)).ToArray();
        Assert.Contains(before.Zip(after, (a, b) => Math.Abs(a - b)), d => d > 0);
        // 8 samples per ray against a budget of 8 · 64 = 512 samples: the batch grows 8 -> 13 -> 18 -> 22 (under the cap).
        Assert.InRange(reconstructor.CurrentBatchSize, 17, 23);
        Assert.Equal(2, reconstructor.Field.HashGrid?.ActiveLevels);
    }
}
