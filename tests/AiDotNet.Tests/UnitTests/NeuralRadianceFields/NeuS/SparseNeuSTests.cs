using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields.NeuS;

/// <summary>
/// SparseNeuS (Long et al. 2022) at the coarse level: hand-derived parameter counts of every module, the reference
/// state-dict names and a load round trip, exact grid interpolation and its derivative, the latent SDF gradient,
/// and an encoded scene's shapes, determinism and dependence on the input views.
/// </summary>
public class SparseNeuSTests
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    private static Tensor<double> T(double[] values, params int[] shape) => NeuSHost<double>.FromDoubles(values, shape);

    private static double[] H(Tensor<double> tensor) => NeuSHost<double>.ToDoubles(tensor);

    private static SparseNeuSOptions SmallOptions(int seed) => new SparseNeuSOptions
    {
        VolumeDims = new[] { 8, 8, 8 },
        VoxelSize = 2.0 / 7.0,
        Samples = 8,
        ImportanceSamples = 4,
        UpSampleSteps = 4,
        BatchSize = 8,
        SparseLossPoints = 16,
        GridChunkSize = 100,
        Seed = seed,
    };

    private static List<NeuSView<double>> Views(int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var centers = new[] { new[] { 0.0, 0.0, -2.5 }, new[] { 2.5, 0.0, 0.0 }, new[] { 0.0, 0.0, 2.5 } };
        var views = new List<NeuSView<double>>();
        foreach (var center in centers)
        {
            var camera = NeuSCamera.LookAt(center, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, 16.0, 16, 16);
            var pixels = Enumerable.Range(0, 16 * 16 * 3).Select(_ => random.NextDouble()).ToArray();
            views.Add(new NeuSView<double>(T(pixels, 16, 16, 3), camera));
        }

        return views;
    }

    [Fact]
    public void ParameterCountsAreHandDerived()
    {
        var network = new SparseNeuS<double>(new SparseNeuSOptions { Seed = 1 }, Engine);

        // FeatureNet (models/featurenet.py, base 8): ConvBnReLU is a bias-free conv + InPlaceABN (2C); toplayer,
        // lat1, lat0, smooth1 and smooth0 are plain nn.Conv2d with biases.
        int feature = (3 * 8 * 9 + 16) + (8 * 8 * 9 + 16)
            + (8 * 16 * 25 + 32) + 2 * (16 * 16 * 9 + 32)
            + (16 * 32 * 25 + 64) + 2 * (32 * 32 * 9 + 64)
            + (32 * 32 + 32)                         // toplayer
            + (16 * 32 + 32) + (8 * 32 + 32)         // lat1, lat0
            + (32 * 16 * 9 + 16) + (32 * 8 * 9 + 8); // smooth1, smooth0
        Assert.Equal(48976, feature);
        Assert.Equal(feature, network.FeatureNet.ParameterCount);

        Assert.Equal(56 * 16 * 9 + 2 * 16, network.Compress.ParameterCount);

        // SparseCostRegNet(32 → 16): bias-free 3×3×3 convs + BN.
        int regnet = (32 * 16 + 16 * 16 + 16 * 16 + 16 * 32 + 32 * 32 + 32 * 64 + 64 * 64 + 64 * 32 + 32 * 16 + 16 * 16) * 27
            + 2 * (16 + 16 + 16 + 32 + 32 + 64 + 64 + 32 + 16 + 16);
        Assert.Equal(311648, regnet);
        Assert.Equal(regnet, network.CostRegNet.ParameterCount);

        // LatentSDFLayer: dims_in [39, 144, 144], every output 128, weight-normalized.
        int sdf = (39 * 128 + 2 * 128) + 2 * (144 * 128 + 2 * 128);
        Assert.Equal(42624, sdf);
        Assert.Equal(sdf, network.SdfNetwork.ParameterCount);
        Assert.Equal(127, network.SdfNetwork.FeatureDims);

        // GeneralRenderingNetwork: ray_dir_fc 4→16→59, base_fc 193→64→32, vis_fc 32→32→33, vis_fc2 32→32→1,
        // rgb_fc 37→16→8→1, and the pooling sharpness s.
        int rendering = (4 * 16 + 16) + (16 * 59 + 59) + (193 * 64 + 64) + (64 * 32 + 32) + (32 * 32 + 32) + (32 * 33 + 33)
            + (32 * 32 + 32) + (32 + 1) + (37 * 16 + 16) + (16 * 8 + 8) + (8 + 1) + 1;
        Assert.Equal(19567, rendering);
        Assert.Equal(rendering, network.RenderingNetwork.ParameterCount);

        Assert.Equal(feature + network.Compress.ParameterCount + regnet + sdf + 1 + rendering, network.ParameterCount);
        Assert.Equal(network.ParameterCount, network.Parameters.Sum(p => p.Length));
    }

    [Fact]
    public void NamedParametersUseTheReferenceNamesAndRoundTrip()
    {
        var source = new SparseNeuS<double>(SmallOptions(2), Engine);
        var names = source.NamedParameters().Select(p => p.Key).ToList();
        Assert.Equal(names.Count, names.Distinct().Count());
        foreach (var expected in new[]
        {
            "pyramid_feature_network_geometry_lod0.conv0.0.conv.weight",
            "pyramid_feature_network_geometry_lod0.conv0.0.bn.running_var",
            "pyramid_feature_network_geometry_lod0.toplayer.weight",
            "pyramid_feature_network_geometry_lod0.smooth0.bias",
            "sdf_network_lod0.compress_layer.conv.weight",
            "sdf_network_lod0.sparse_costreg_net.conv0.net.0.weight",
            "sdf_network_lod0.sparse_costreg_net.conv11.net.1.running_mean",
            "sdf_network_lod0.sdf_layer.lin0.weight_g",
            "sdf_network_lod0.sdf_layer.lin2.weight_v",
            "variance_network_lod0.variance",
            "rendering_network_lod0.ray_dir_fc.2.weight",
            "rendering_network_lod0.rgb_fc.4.bias",
            "rendering_network_lod0.s",
        })
            Assert.Contains(expected, names);

        var target = new SparseNeuS<double>(SmallOptions(3), Engine);
        var views = Views(4);
        Assert.NotEqual(H(source.Reconstruct(views).Volume), H(target.Reconstruct(views).Volume));
        target.LoadParameters(source.NamedParameters().ToDictionary(p => p.Key, p => p.Value));
        Assert.Equal(H(source.Reconstruct(views).Volume), H(target.Reconstruct(views).Volume));

        Assert.Throws<ArgumentException>(() => target.LoadParameters(
            new Dictionary<string, Tensor<double>> { ["not_a_module.weight"] = T(new[] { 1.0 }, 1) }));
    }

    [Fact]
    public void Trilinear_IsExactForALinearGridAndGivesItsWorldGradient()
    {
        int[] dims = { 3, 4, 5 };
        const double voxel = 0.5;
        var values = new double[60];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 5; k++)
                    values[(i * 4 + j) * 5 + k] = 1 + 2 * i - j + 0.5 * k;
        var grid = T(values, 60, 1);
        var interpolation = NeuSGridInterpolation<double>.Trilinear(new[] { 1.3, 2.6, 0.2, 0.0, 0.0, 0.0 }, dims, voxel, true);
        var sampled = H(interpolation.Sample(Engine, grid));
        Assert.Equal(1 + 2.6 - 2.6 + 0.1, sampled[0], 12);
        Assert.Equal(1.0, sampled[1], 12);
        var derivatives = interpolation.SpatialDerivatives(Engine, grid);
        Assert.Equal(2 / voxel, H(derivatives[0])[0], 10);
        Assert.Equal(-1 / voxel, H(derivatives[1])[0], 10);
        Assert.Equal(0.5 / voxel, H(derivatives[2])[0], 10);
        Assert.Equal(new[] { 1.1, 1.0 }, interpolation.SampleHost(values).Select(v => Math.Round(v, 10)).ToArray());
    }

    [Fact]
    public void Bilinear_AlignsCornersAndZeroPads()
    {
        var image = Enumerable.Range(0, 12).Select(i => 10.0 + i).ToArray(); // [3, 4]
        var interpolation = NeuSGridInterpolation<double>.Bilinear(new[] { 2.0, 0.5, -0.5, 3.0 }, new[] { 1.0, 0.0, 0.0, 2.5 }, 3, 4);
        var sampled = H(interpolation.Sample(Engine, T(image, 12, 1)));
        Assert.Equal(16.0, sampled[0], 12);              // row 1, column 2
        Assert.Equal(10.5, sampled[1], 12);              // halfway between columns 0 and 1
        Assert.Equal(0.5 * 10.0, sampled[2], 12);        // half outside the image
        Assert.Equal(0.5 * 21.0, sampled[3], 12);        // row 2.5: half below the last row
    }

    [Fact]
    public void LatentSdf_GradientIncludesTheLatentJacobian()
    {
        var random = RandomHelper.CreateSeededRandom(9);
        var mlp = NeuSSdfMlp<double>.CreateLatent(24, 4, 4, new[] { 4 }, 2, 0.5, geometricInit: false, weightNorm: true,
            100.0, random, "sdf_layer.");
        int[] dims = { 4, 4, 4 };
        const double voxel = 2.0 / 3.0;
        var grid = T(Enumerable.Range(0, 64 * 4).Select(_ => random.NextDouble() * 2 - 1).ToArray(), 64, 4);
        var points = new double[6 * 3];
        for (int i = 0; i < points.Length; i++) points[i] = -1.0 + (random.Next(3) + 0.2 + 0.6 * random.NextDouble()) * voxel;

        Tensor<double> Sdf(double[] p, bool gradient, out Tensor<double>? grad)
        {
            var coordinates = p.Select(v => (v + 1.0) / voxel).ToArray();
            var interpolation = NeuSGridInterpolation<double>.Trilinear(coordinates, dims, voxel, gradient);
            var latent = interpolation.Sample(Engine, grid);
            var output = mlp.Forward(Engine, T(p, p.Length / 3, 3), latent,
                gradient ? interpolation.SpatialDerivatives(Engine, grid) : null, gradient);
            grad = output.Gradient;
            return output.Sdf;
        }

        Sdf(points, true, out var analyticTensor);
        var analytic = H(analyticTensor ?? throw new InvalidOperationException("No gradient."));
        const double h = 1e-6;
        for (int i = 0; i < points.Length; i++)
        {
            var plus = (double[])points.Clone();
            var minus = (double[])points.Clone();
            plus[i] += h;
            minus[i] -= h;
            double numeric = (H(Sdf(plus, false, out _))[i / 3] - H(Sdf(minus, false, out _))[i / 3]) / (2 * h);
            Assert.Equal(numeric, analytic[i], 5);
        }
    }

    [Fact]
    public void Reconstruct_ShapesDeterminismAndInputDependence()
    {
        var views = Views(5);
        var a = new SparseNeuS<double>(SmallOptions(6), Engine);
        var b = new SparseNeuS<double>(SmallOptions(6), Engine);
        var sceneA = a.Reconstruct(views);
        var sceneB = b.Reconstruct(views);
        Assert.Equal(new[] { 512, 16 }, sceneA.Volume.Shape.ToArray());
        Assert.Equal(512, sceneA.Active.Length);
        Assert.Contains(sceneA.Active, v => v > 0);
        Assert.Equal(3, sceneA.ViewCount);

        var gridA = sceneA.SampleGrid(4, SdfGridBounds.UnitCube);
        Assert.Equal(new[] { 4, 4, 4 }, gridA.Shape.ToArray());
        Assert.All(H(gridA), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        Assert.Equal(H(gridA), H(sceneB.SampleGrid(4, SdfGridBounds.UnitCube)));

        var other = a.Reconstruct(Views(7));
        Assert.NotEqual(H(sceneA.Volume), H(other.Volume));
    }

    [Fact]
    public void Scene_RenderShapes()
    {
        var views = Views(8);
        var network = new SparseNeuS<double>(SmallOptions(8), Engine);
        var scene = network.Reconstruct(views);
        var origins = new[] { 0.0, 0.0, -2.5, 0.0, 0.0, -2.5 };
        var directions = new[] { 0.0, 0.0, 1.0, 0.0, 0.6, 0.8 };
        var result = scene.Render(origins, directions, new[] { 0.0, 0.0, -2.5 }, 0, 1.0, null);
        Assert.Equal(12, result.SamplesPerRay);
        Assert.Equal(new[] { 2, 3 }, result.Color.Shape.ToArray());
        Assert.Equal(new[] { 2, 12 }, result.Weights.Shape.ToArray());
        Assert.Equal(new[] { 2, 12 }, result.Sdf.Shape.ToArray());
        Assert.Equal(2, result.ColorValid.Length);
        Assert.All(H(result.WeightSum), w => Assert.InRange(w, 0.0, 1.0 + 1e-6));
        var image = scene.RenderImage(views[1].Camera, 4, 4, raysPerBatch: 6);
        Assert.Equal(new[] { 4, 4, 3 }, image.Shape.ToArray());
    }

    [Fact]
    public void TrainStep_IsDeterministicAndUpdatesTheWeights()
    {
        var views = Views(10);
        var a = new SparseNeuS<double>(SmallOptions(11), Engine);
        var b = new SparseNeuS<double>(SmallOptions(11), Engine);
        // The sparseness loss reaches the SDF MLP on every step (the color loss only on rays two views see).
        var before = H(a.SdfNetwork.Layers[0].Weight);
        double lossA = a.TrainStep(views);
        double lossB = b.TrainStep(views);
        Assert.False(double.IsNaN(lossA) || double.IsInfinity(lossA));
        Assert.Equal(lossA, lossB);
        Assert.Equal(1, a.Iteration);
        Assert.NotEqual(before, H(a.SdfNetwork.Layers[0].Weight));
        Assert.Equal(a.TrainStep(views), b.TrainStep(views));
    }
}
