using System;
using System.Linq;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Enums;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields.NeuS;

/// <summary>
/// instant-nsr-pl's NeuS pieces (Wonder3D's reconstruction): finite-difference gradients and their progressive step,
/// the <c>VolumeRadiance</c> color network and its spherical-harmonics basis, nerfacc's occupancy grid and ray
/// marching with early-stopped weights.
/// </summary>
public class NeuSInstantNsrTests
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    private static double[] Values(Tensor<double> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor[i];
        return values;
    }

    private static NeuSOptions Smooth(int seed) => new NeuSOptions
    {
        SdfHiddenDim = 16, SdfHiddenLayers = 2, SdfFeatureDim = 4, SdfSkipLayers = Array.Empty<int>(), SdfFrequencies = 2,
        GeometricInit = false, SoftplusBeta = 10.0, ColorHiddenDim = 8, ColorHiddenLayers = 1, ViewFrequencies = 1,
        Seed = seed,
    };

    [Fact]
    public void FiniteDifference_MatchesTheAnalyticGradientToSecondOrder()
    {
        var field = new NeuSField<double>(Smooth(3), RandomHelper.CreateSeededRandom(3), Engine);
        var points = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new[] { 0.11, -0.23, 0.31, -0.42, 0.05, 0.27, 0.36, 0.18, -0.14 }));
        var analytic = Values(field.QueryNetwork(Engine, points, true).Gradient ?? throw new InvalidOperationException());

        double MaxError(double eps)
        {
            field.Options.GradientMode = NeuSGradientMode.FiniteDifference;
            field.Options.FiniteDifferenceEpsilon = eps;
            field.UpdateStep(0);
            Assert.Equal(eps, field.FiniteDifferenceEpsilon);
            var fd = Values(field.QueryNetwork(Engine, points, true).Gradient ?? throw new InvalidOperationException());
            field.Options.GradientMode = NeuSGradientMode.Analytic;
            return fd.Zip(analytic, (a, b) => Math.Abs(a - b)).Max();
        }

        double coarse = MaxError(2e-2), fine = MaxError(1e-2);
        Assert.True(fine < 1e-3, $"FD error {fine}");
        // Central differences: the error is h²/6 · f''' + O(h⁴), so halving h divides it by 4.
        Assert.InRange(coarse / fine, 3.5, 4.5);
    }

    [Fact]
    public void FiniteDifference_UsesTheSixClampedOffsets()
    {
        var options = Smooth(4);
        options.GradientMode = NeuSGradientMode.FiniteDifference;
        options.FiniteDifferenceEpsilon = 0.05;
        var field = new NeuSField<double>(options, RandomHelper.CreateSeededRandom(4), Engine);
        // x sits at the box face: x + ε is clamped to 1, so ∂f/∂x = (f(1) − f(1 − ε)) / 2ε, as instant-nsr-pl computes it.
        var p = new[] { 1.0, 0.2, -0.3 };
        var gradient = Values(field.QueryNetwork(Engine, new Tensor<double>(new[] { 1, 3 }, new Vector<double>(p)), true).Gradient
            ?? throw new InvalidOperationException());
        double F(double x, double y, double z) => field.QuerySdf(new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { x, y, z })))[0];
        Assert.Equal((F(1.0, 0.2, -0.3) - F(0.95, 0.2, -0.3)) / 0.1, gradient[0], 1e-12);
        Assert.Equal((F(1.0, 0.25, -0.3) - F(1.0, 0.15, -0.3)) / 0.1, gradient[1], 1e-12);
        Assert.Equal((F(1.0, 0.2, -0.25) - F(1.0, 0.2, -0.35)) / 0.1, gradient[2], 1e-12);
    }

    [Fact]
    public void ProgressiveStep_IsTheFinestActiveCell()
    {
        // neuralangelo-ortho-wmask: base 32, σ = 1.3195079107728942 (σ⁵ = 4), start level 4, a level per 1000 steps.
        var options = new NeuSOptions
        {
            SdfBackbone = NeuSSdfBackbone.HashGrid, HashGridLevels = 10, HashGridBaseResolution = 32,
            HashGridPerLevelScale = 1.3195079107728942, HashGridStartLevel = 4, HashGridUpdateIterations = 1000,
            GradientMode = NeuSGradientMode.FiniteDifference, FiniteDifferenceStep = NeuSFiniteDifferenceStep.Progressive,
        };
        // ε = 2r / (b · σ^(level − 1)).
        Assert.Equal(2.0 / (32 * Math.Pow(4, 0.6)), options.FiniteDifferenceEpsilonAt(0), 1e-12);
        Assert.Equal(2.0 / (32 * Math.Pow(4, 0.6)), options.FiniteDifferenceEpsilonAt(999), 1e-12);
        Assert.Equal(2.0 / (32 * Math.Pow(4, 0.8)), options.FiniteDifferenceEpsilonAt(1000), 1e-12);
        Assert.Equal(2.0 / 128.0, options.FiniteDifferenceEpsilonAt(2000), 1e-12);
        Assert.Equal(2.0 / (32 * Math.Pow(4, 1.8)), options.FiniteDifferenceEpsilonAt(6000), 1e-12);
        Assert.Equal(2.0 / (32 * Math.Pow(4, 1.8)), options.FiniteDifferenceEpsilonAt(500000), 1e-12);
        Assert.Equal(0.027204705103003882, options.FiniteDifferenceEpsilonAt(0), 1e-15);
        options.SceneRadius = 0.5;
        Assert.Equal(1.0 / 128.0, options.FiniteDifferenceEpsilonAt(2000), 1e-12);
    }

    [Fact]
    public void SphericalHarmonics_MatchTheHandBasis()
    {
        double c0 = 1 / (2 * Math.Sqrt(Math.PI)), c1 = Math.Sqrt(3) / (2 * Math.Sqrt(Math.PI));
        double c2 = Math.Sqrt(15) / (2 * Math.Sqrt(Math.PI)), c20 = Math.Sqrt(5) / (4 * Math.Sqrt(Math.PI));
        double c22 = Math.Sqrt(15) / (4 * Math.Sqrt(Math.PI)), c33 = Math.Sqrt(70) / (8 * Math.Sqrt(Math.PI));
        double c32 = Math.Sqrt(105) / (2 * Math.Sqrt(Math.PI)), c31 = Math.Sqrt(42) / (8 * Math.Sqrt(Math.PI));
        double c30 = Math.Sqrt(7) / (4 * Math.Sqrt(Math.PI)), c32b = Math.Sqrt(105) / (4 * Math.Sqrt(Math.PI));
        double x = 0.48, y = -0.6, z = 0.64; // unit
        double[] expected =
        {
            c0,
            -c1 * y, c1 * z, -c1 * x,
            c2 * x * y, -c2 * y * z, c20 * (3 * z * z - 1), -c2 * x * z, c22 * (x * x - y * y),
            c33 * y * (3 * x * x - y * y) * -1, c32 * x * y * z, c31 * y * (1 - 5 * z * z), c30 * z * (5 * z * z - 3),
            c31 * x * (1 - 5 * z * z), c32b * z * (x * x - y * y), c33 * x * (3 * y * y - x * x),
        };
        var actual = NeuSSphericalHarmonics.Evaluate(new[] { x, y, z }, 4);
        Assert.Equal(16, actual.Length);
        for (int i = 0; i < 16; i++) Assert.Equal(expected[i], actual[i], 1e-12);
        // Lower degrees are prefixes.
        Assert.Equal(actual.Take(4), NeuSSphericalHarmonics.Evaluate(new[] { x, y, z }, 2));
    }

    [Fact]
    public void SphericalHarmonics_AreOrthonormalOverTheSphere()
    {
        // Midpoint rule in cos θ and φ.
        const int nu = 400, nphi = 64;
        var dirs = new double[nu * nphi * 3];
        for (int a = 0; a < nu; a++)
            for (int b = 0; b < nphi; b++)
            {
                double u = -1 + (a + 0.5) * 2.0 / nu, phi = (b + 0.5) * 2 * Math.PI / nphi, s = Math.Sqrt(1 - u * u);
                int i = (a * nphi + b) * 3;
                dirs[i] = s * Math.Cos(phi);
                dirs[i + 1] = s * Math.Sin(phi);
                dirs[i + 2] = u;
            }

        var basis = NeuSSphericalHarmonics.Evaluate(dirs, 4);
        double area = 4 * Math.PI / (nu * nphi);
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
            {
                double dot = 0;
                for (int p = 0; p < nu * nphi; p++) dot += basis[p * 16 + i] * basis[p * 16 + j];
                Assert.Equal(i == j ? 1.0 : 0.0, dot * area, 2e-3);
            }
    }

    [Fact]
    public void VolumeRadiance_HasTheConfiguredInputsAndInitialization()
    {
        // input_feature_dim = feature_dim (13: SDF + 12) + 3 (normal); 64 × 2 hidden; 3 outputs.
        var plain = new NeuSRenderingNetwork<double>(13, NeuSRenderingMode.VolumeRadiance, 64, 2, 4, weightNorm: true,
            squeezeOut: true, RandomHelper.CreateSeededRandom(5));
        Assert.Equal((16 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3), plain.ParameterCount);
        foreach (var layer in plain.Layers)
        {
            Assert.False(layer.WeightNorm);
            double bound = Math.Sqrt(6.0 / layer.InputDims);
            Assert.All(Values(layer.Weight), w => Assert.InRange(w, -bound, bound));
            Assert.All(Values(layer.Bias), b => Assert.Equal(0.0, b));
        }

        // With the view direction: + 16 SH terms.
        var viewDependent = new NeuSRenderingNetwork<double>(13, NeuSRenderingMode.VolumeRadiance, 64, 2, 4, false, true,
            RandomHelper.CreateSeededRandom(5), shDegree: 4);
        Assert.Equal(32, viewDependent.Layers[0].InputDims);

        var features = new Tensor<double>(new[] { 2, 13 }, new Vector<double>(Enumerable.Range(0, 26).Select(i => Math.Sin(i)).ToArray()));
        var normals = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.6, 0.8, 0.0 }));
        var points = new Tensor<double>(new[] { 2, 3 });
        var viewA = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 }));
        var viewB = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 0.0, 0.0, -1.0, 0.0, 0.0, 1.0 }));
        var a = Values(plain.Forward(Engine, points, normals, viewA, features));
        var b = Values(plain.Forward(Engine, points, normals, viewB, features));
        Assert.Equal(a, b);
        Assert.All(a, v => Assert.InRange(v, 0.0, 1.0));
        Assert.NotEqual(Values(viewDependent.Forward(Engine, points, normals, viewA, features)),
            Values(viewDependent.Forward(Engine, points, normals, viewB, features)));
    }

    [Fact]
    public void OccupancyGrid_EmaAndThresholdFollowNerfacc()
    {
        var grid = new NeuSOccupancyGrid(2, 1.0);
        Assert.False(grid.IsInitialized);
        Assert.True(grid.IsOccupied(0.5, 0.5, 0.5));
        Assert.False(grid.IsOccupied(1.5, 0.0, 0.0));

        grid.Apply(Enumerable.Range(0, 8).ToArray(), new[] { 0.5, 0, 0, 0, 0, 0, 0, 0.004 }, 0.01, 0.95);
        // mean 0.063, cut = min(mean, 0.01) = 0.01.
        Assert.Equal(new[] { true, false, false, false, false, false, false, false }, grid.Binary.ToArray());

        grid.Apply(new[] { 0, 1, 2, 2 }, new[] { 0.1, 0.2, 0.3, 0.1 }, 0.01, 0.95);
        // max(0.95 · 0.5, 0.1) = 0.475; max(0, 0.2); the repeated cell takes its last write, max(0, 0.1).
        Assert.Equal(0.475, grid.Occupancy[0], 1e-12);
        Assert.Equal(0.2, grid.Occupancy[1], 1e-12);
        Assert.Equal(0.1, grid.Occupancy[2], 1e-12);
        Assert.Equal(0.004, grid.Occupancy[7], 1e-12);
        Assert.Equal(new[] { true, true, true, false, false, false, false, false }, grid.Binary.ToArray());

        // A high threshold falls back to the mean: (0.475 + 0.2 + 0.1 + 0.004) / 8 = 0.097375.
        grid.Apply(Array.Empty<int>(), Array.Empty<double>(), 1.0, 0.95);
        Assert.Equal(new[] { true, true, true, false, false, false, false, false }, grid.Binary.ToArray());
        // max(0.5 · 0.1, 0.09) = 0.09 is below the new mean (0.475 + 0.2 + 0.09 + 0.004) / 8 = 0.096125.
        grid.Apply(new[] { 2 }, new[] { 0.09 }, 1.0, 0.5);
        Assert.Equal(0.09, grid.Occupancy[2], 1e-12);
        Assert.False(grid.Binary[2]);
        Assert.True(grid.Binary[1]);

        // Cell (x, y, z) = (1, 0, 0) is index 4 and covers x ∈ [0, 1], y, z ∈ [−1, 0].
        Assert.Equal(4, grid.CellIndex(1, 0, 0));
        Assert.Equal(grid.Binary[0], grid.IsOccupied(-0.5, -0.5, -0.5));
        Assert.Equal(grid.Binary[4], grid.IsOccupied(0.5, -0.5, -0.5));
    }

    [Fact]
    public void OccupancyGrid_UpdateDrawsJitteredPointsInEachCell()
    {
        var grid = new NeuSOccupancyGrid(4, 2.0);
        var random = RandomHelper.CreateSeededRandom(6);
        double[]? seen = null;
        // Warm-up: every cell once, its point inside the cell.
        Assert.True(grid.EveryNStep(0, 16, p => { seen = p; return new double[p.Length / 3]; }, 0.01, 0.95, 256, random));
        Assert.NotNull(seen);
        if (seen is null) return;
        Assert.Equal(64 * 3, seen.Length);
        for (int c = 0; c < 64; c++)
        {
            int x = c / 16, y = c / 4 % 4, z = c % 4;
            Assert.InRange(seen[c * 3], -2.0 + x, -1.0 + x);
            Assert.InRange(seen[c * 3 + 1], -2.0 + y, -1.0 + y);
            Assert.InRange(seen[c * 3 + 2], -2.0 + z, -1.0 + z);
        }

        Assert.False(grid.EveryNStep(5, 16, p => throw new InvalidOperationException(), 0.01, 0.95, 256, random));
        // After the warm-up: R³ / 4 uniform cells plus the (here 3) occupied ones.
        grid.Apply(new[] { 1, 2, 3 }, new[] { 1.0, 1.0, 1.0 }, 0.01, 0.95);
        int count = 0;
        grid.Update(300, p => { count = p.Length / 3; return new double[count]; }, 0.01, 0.95, 256, random);
        Assert.Equal(16 + 3, count);
    }

    private static NeuSOptions Marching(bool useGrid) => new NeuSOptions
    {
        SdfHiddenDim = 16, SdfHiddenLayers = 2, SdfFeatureDim = 4, SdfSkipLayers = Array.Empty<int>(), SdfFrequencies = 2,
        ColorHiddenDim = 8, ColorHiddenLayers = 2, RenderingMode = NeuSRenderingMode.VolumeRadiance,
        Sampler = NeuSRaySampler.OccupancyGrid, UseOccupancyGrid = useGrid, OccupancyGridResolution = 16,
        SamplesPerRayBudget = 8, Seed = 7,
    };

    [Fact]
    public void Marching_StepsThroughTheBoxOnTheStepLattice()
    {
        var field = new NeuSField<double>(Marching(useGrid: false), RandomHelper.CreateSeededRandom(7), Engine);
        // δ = 1.732 · 2 / 8 = 0.433; the ray along +z from z = −3 meets the box at t ∈ [2, 4]:
        // steps start at 2 + kδ while the midpoint is before 4: 2, 2.433, 2.866, 3.299, 3.732.
        var (rays, starts, counts) = field.MarchSamples(new[] { 0.0, 0.0, -3.0, 5.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 }, null);
        Assert.Equal(new[] { 5, 0 }, counts);
        Assert.Equal(new[] { 0, 0, 0, 0, 0 }, rays);
        var expected = new[] { 2.0, 2.433, 2.866, 3.299, 3.732 };
        for (int i = 0; i < 5; i++) Assert.Equal(expected[i], starts[i], 1e-12);
        // Stratified: one shift in [0, δ) per ray.
        var (_, shifted, _) = field.MarchSamples(new[] { 0.0, 0.0, -3.0 }, new[] { 0.0, 0.0, 1.0 }, RandomHelper.CreateSeededRandom(1));
        Assert.InRange(shifted[0] - 2.0, 0.0, 0.433);
        for (int i = 1; i < shifted.Length; i++) Assert.Equal(0.433, shifted[i] - shifted[i - 1], 1e-12);
    }

    [Fact]
    public void OccupancyGrid_PrunesEmptySpaceAndKeepsTheSurface()
    {
        // The geometric initialization's sphere of radius 0.5: f ≈ ‖x‖ − 0.5.
        var options = Marching(useGrid: true);
        options.SamplesPerRayBudget = 64;
        var field = new NeuSField<double>(options, RandomHelper.CreateSeededRandom(8), Engine);
        var dense = new NeuSField<double>(WithBudget(Marching(useGrid: false), 64), RandomHelper.CreateSeededRandom(8), Engine);
        Assert.True(field.UpdateOccupancyGrid(0, RandomHelper.CreateSeededRandom(9)));

        // A ray along the cube's diagonal (the box extends to ‖x‖ = √3) and one through a face center.
        var origins = new[] { -2.0, -2.0, -2.0, 0.0, 0.0, -2.0 };
        double k = 1 / Math.Sqrt(3);
        var directions = new[] { k, k, k, 0.0, 0.0, 1.0 };
        var (rays, starts, counts) = field.MarchSamples(origins, directions, null);
        var (denseRays, denseStarts, denseCounts) = dense.MarchSamples(origins, directions, null);
        double step = options.MarchingStepSize;

        double Radius(int ray, double start)
        {
            double t = start + 0.5 * step, r2 = 0;
            for (int c = 0; c < 3; c++) r2 += Math.Pow(origins[ray * 3 + c] + t * directions[ray * 3 + c], 2);
            return Math.Sqrt(r2);
        }

        var kept = rays.Select((ray, i) => Radius(ray, starts[i])).ToArray();
        var all = denseRays.Select((ray, i) => Radius(ray, denseStarts[i])).ToArray();
        Assert.Contains(all, r => r > 1.4);
        Assert.True(counts.Sum() < denseCounts.Sum());
        // With s = exp(10 · 0.3) the step opacity crosses the 0.01 threshold near f = 0.24 (‖x‖ ≈ 0.74) and is ~1e-5
        // far outside; a kept cell's jittered point is within a cell diagonal (2/16 · √3 ≈ 0.22) of it.
        Assert.All(kept, r => Assert.True(r < 1.2, $"kept a sample at radius {r}"));
        // Both rays keep samples at the surface.
        for (int ray = 0; ray < 2; ray++)
            Assert.Contains(rays.Select((rr, i) => (rr, i)).Where(p => p.rr == ray), p => Math.Abs(Radius(ray, starts[p.i]) - 0.5) < step);

        // The render counts exactly the marched samples.
        var render = field.Render(origins, directions, 1.0, null);
        Assert.Equal(counts.Sum(), render.SampleCount);
        Assert.Equal(counts.Max(), render.SamplesPerRay);
        Assert.All(Values(render.WeightSum), w => Assert.InRange(w, 0.0, 1.0 + 1e-9));
    }

    private static NeuSOptions WithBudget(NeuSOptions options, int budget)
    {
        options.SamplesPerRayBudget = budget;
        return options;
    }

    [Fact]
    public void EarlyStop_ZeroesWeightsOnceTransmittanceDropsBelowEpsilon()
    {
        // One ray: the transmittance before each sample is 1, 0.1, 0.001, 1e-5, …: from the fourth on it is below 1e-4.
        var alpha = new[] { 0.9, 0.99, 0.99, 0.5, 0.0, 0.0 };
        Assert.Equal(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }, NeuSField<double>.EarlyStopMask(alpha, 1, 6, 1e-4));
        // Two rays of three: the second ray starts afresh (T before the third sample of ray 0 is 0.001).
        Assert.Equal(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, NeuSField<double>.EarlyStopMask(alpha, 2, 3, 1e-4));
    }

    [Fact]
    public void Wonder3DDefaults_SelectTheInstantNsrPlComponents()
    {
        var o = new Wonder3DOptions();
        Assert.Equal(NeuSGradientMode.FiniteDifference, o.GradientMode);
        Assert.Equal(NeuSFiniteDifferenceStep.Progressive, o.FiniteDifferenceStep);
        Assert.Equal(NeuSColorNetwork.VolumeRadiance, o.ColorNetwork);
        Assert.False(o.TextureUsesViewDirection);
        Assert.Equal(4, o.TextureShDegree);
        Assert.Equal(NeuSRaySampler.OccupancyGrid, o.RaySampler);
        Assert.Equal(128, o.OccupancyGridResolution);
        Assert.Equal(16, o.OccupancyGridUpdateInterval);
        Assert.Equal(0.001, o.OccupancyGridThreshold);
        Assert.Equal(0.95, o.OccupancyGridEmaDecay);
        Assert.Equal(256, o.OccupancyGridWarmUpIterations);
        Assert.Equal(1e-4, o.TransmittanceEarlyStop);
        // Plain NeuS keeps the paper's choices.
        var neus = new NeuSOptions();
        Assert.Equal(NeuSGradientMode.Analytic, neus.GradientMode);
        Assert.Equal(NeuSRaySampler.Hierarchical, neus.Sampler);
        Assert.Equal(NeuSRenderingMode.Idr, neus.RenderingMode);
    }
}
