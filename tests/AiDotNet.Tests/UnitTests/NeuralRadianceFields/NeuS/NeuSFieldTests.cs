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
/// The NeuS SDF and color networks, the field and the per-scene reconstructor: hand-derived parameter counts, the
/// analytic spatial gradient against central differences, the geometric initialization's sphere, the dense grid
/// sampler's layout, determinism under a seed, and training that responds to the images.
/// </summary>
public class NeuSFieldTests
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    private static Tensor<double> T(double[] values, params int[] shape) => NeuSHost<double>.FromDoubles(values, shape);

    private static double[] H(Tensor<double> tensor) => NeuSHost<double>.ToDoubles(tensor);

    private static NeuSOptions TinyOptions(int seed) => new NeuSOptions
    {
        SdfHiddenDim = 32,
        SdfHiddenLayers = 2,
        SdfFeatureDim = 8,
        SdfSkipLayers = Array.Empty<int>(),
        SdfFrequencies = 2,
        ColorHiddenDim = 16,
        ColorHiddenLayers = 1,
        ViewFrequencies = 1,
        Samples = 8,
        ImportanceSamples = 8,
        UpSampleSteps = 4,
        BatchSize = 32,
        MaskWeight = 0.0,
        WarmUpIterations = 0,
        Iterations = 1000,
        LearningRate = 1e-2,
        GridChunkSize = 10,
        Seed = seed,
    };

    private static List<NeuSView<double>> Views(double color, Tensor<double>? normals = null)
    {
        var views = new List<NeuSView<double>>();
        var centers = new[] { new[] { 0.0, 0.0, -2.5 }, new[] { 2.5, 0.0, 0.0 } };
        foreach (var center in centers)
        {
            var camera = NeuSCamera.LookAt(center, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, 30.0, 8, 8); // the sphere fills the image
            views.Add(new NeuSView<double>(NeuSHost<double>.Filled(color, 8, 8, 3), camera, null, normals));
        }

        return views;
    }

    [Fact]
    public void SdfMlp_ParameterCountIsHandDerived()
    {
        // γ(x) has 3·(1 + 2·2) = 15 channels; dims [15, 32, 32, 32, 5] with a skip into layer 2, so layer 1 narrows
        // to 32 − 15 = 17. Weight norm adds one magnitude per output.
        var mlp = NeuSSdfMlp<double>.CreateNeuS(32, 3, 5, new[] { 2 }, 2, 0.5, 1.0, geometricInit: true,
            weightNorm: true, insideOutside: false, 100.0, RandomHelper.CreateSeededRandom(1));
        int expected = (15 * 32 + 2 * 32) + (32 * 17 + 2 * 17) + (32 * 32 + 2 * 32) + (32 * 5 + 2 * 5);
        Assert.Equal(2380, expected);
        Assert.Equal(expected, mlp.ParameterCount);
        Assert.Equal(15, mlp.EmbeddingDims);
        Assert.Equal(4, mlp.FeatureDims);
        Assert.Equal(new[] { 15, 32, 32, 32 }, mlp.Layers.Select(l => l.InputDims).ToArray());
        Assert.Equal(new[] { 32, 17, 32, 5 }, mlp.Layers.Select(l => l.OutputDims).ToArray());
    }

    [Fact]
    public void Field_DefaultParameterCountMatchesTheReferenceConfig()
    {
        // wmask.conf SDF network: γ(x) 39 channels, 8×256 hidden, skip at 4 (layer 3 outputs 256 − 39 = 217), d_out 257.
        int sdf = (39 * 256 + 512) + 2 * (256 * 256 + 512) + (256 * 217 + 2 * 217) + 4 * (256 * 256 + 512) + (256 * 257 + 2 * 257);
        // IDR color network: point 3 + γ(view) 27 + normal 3 + feature 256 = 289, 4×256 hidden, 3 outputs.
        int color = (289 * 256 + 512) + 3 * (256 * 256 + 512) + (256 * 3 + 6);
        Assert.Equal(529076, sdf);
        Assert.Equal(273414, color);
        var field = new NeuSField<double>(new NeuSOptions { Seed = 3 }, RandomHelper.CreateSeededRandom(3), Engine);
        Assert.Equal(sdf, field.SdfNetwork.ParameterCount);
        Assert.Equal(color, field.ColorNetwork.ParameterCount);
        Assert.Equal(sdf + color + 1, field.ParameterCount);
        Assert.Equal(field.ParameterCount, field.Parameters.Sum(p => p.Length));
        Assert.Contains(field.NamedParameters(), p => p.Key == "sdf_network_fine.lin8.weight_g");
        Assert.Contains(field.NamedParameters(), p => p.Key == "variance_network_fine.variance");
    }

    [Fact]
    public void SdfMlp_AnalyticGradientMatchesCentralDifferences()
    {
        var mlp = NeuSSdfMlp<double>.CreateNeuS(32, 3, 5, new[] { 2 }, 2, 0.5, 1.0, geometricInit: false,
            weightNorm: true, insideOutside: false, 100.0, RandomHelper.CreateSeededRandom(7));
        var random = RandomHelper.CreateSeededRandom(8);
        var points = Enumerable.Range(0, 15).Select(_ => random.NextDouble() * 1.6 - 0.8).ToArray();
        var gradient = H(mlp.Forward(Engine, T(points, 5, 3), null, null, withGradient: true).Gradient
            ?? throw new InvalidOperationException("No gradient."));
        const double h = 1e-6;
        for (int i = 0; i < points.Length; i++)
        {
            var plus = (double[])points.Clone();
            var minus = (double[])points.Clone();
            plus[i] += h;
            minus[i] -= h;
            double fp = H(mlp.Forward(Engine, T(plus, 5, 3), null, null, false).Sdf)[i / 3];
            double fm = H(mlp.Forward(Engine, T(minus, 5, 3), null, null, false).Sdf)[i / 3];
            Assert.Equal((fp - fm) / (2 * h), gradient[i], 5);
        }
    }

    [Fact]
    public void SdfMlp_GeometricInitIsApproximatelyTheRadiusHalfSphere()
    {
        var mlp = NeuSSdfMlp<double>.CreateNeuS(256, 8, 257, new[] { 4 }, 6, 0.5, 1.0, geometricInit: true,
            weightNorm: true, insideOutside: false, 100.0, RandomHelper.CreateSeededRandom(11));
        var random = RandomHelper.CreateSeededRandom(12);
        foreach (double radius in new[] { 0.1, 0.5, 0.9 })
        {
            var points = new double[8 * 3];
            for (int p = 0; p < 8; p++)
            {
                double x = NeuSHost<double>.NextGaussian(random), y = NeuSHost<double>.NextGaussian(random), z = NeuSHost<double>.NextGaussian(random);
                double n = Math.Sqrt(x * x + y * y + z * z);
                points[p * 3] = radius * x / n;
                points[p * 3 + 1] = radius * y / n;
                points[p * 3 + 2] = radius * z / n;
            }

            var sdf = H(mlp.Forward(Engine, T(points, 8, 3), null, null, false).Sdf);
            foreach (double f in sdf)
            {
                if (radius < 0.5) Assert.True(f < -0.2, $"SDF {f} at radius {radius} should be about {radius - 0.5}.");
                else if (radius > 0.5) Assert.True(f > 0.2, $"SDF {f} at radius {radius} should be about {radius - 0.5}.");
                else Assert.True(Math.Abs(f) < 0.1, $"SDF {f} on the initial sphere should be about 0.");
            }
        }
    }

    [Fact]
    public void SampleGrid_IsIndexedXYZAndMatchesQuerySdf()
    {
        var field = new NeuSField<double>(TinyOptions(4), RandomHelper.CreateSeededRandom(4), Engine);
        var bounds = SdfGridBounds.UnitCube;
        var grid = field.SampleGrid(3, bounds);
        Assert.Equal(new[] { 3, 3, 3 }, grid.Shape.ToArray());
        var direct = H(field.QuerySdf(T(bounds.GridPoints(3), 27, 3)));
        var sampled = H(grid);
        for (int i = 0; i < 27; i++) Assert.Equal(direct[i], sampled[i], 10);
        // Sample [2, 0, 1] is the point (1, −1, 0).
        var single = H(field.QuerySdf(T(new[] { 1.0, -1.0, 0.0 }, 1, 3)));
        Assert.Equal(single[0], sampled[(2 * 3 + 0) * 3 + 1], 10);
        Assert.Equal(1.0, bounds.Spacing(3, 0), 12);
    }

    [Fact]
    public void Field_RenderShapesAndRenderImage()
    {
        var options = TinyOptions(5);
        var field = new NeuSField<double>(options, RandomHelper.CreateSeededRandom(5), Engine);
        var origins = new[] { 0.0, 0.0, -2.5, 0.0, 0.0, -2.5 };
        var directions = new[] { 0.0, 0.0, 1.0, 0.6, 0.0, 0.8 };
        var result = field.Render(origins, directions, 1.0, null);
        int samples = options.Samples + options.ImportanceSamples;
        Assert.Equal(samples, result.SamplesPerRay);
        Assert.Equal(new[] { 2, 3 }, result.Color.Shape.ToArray());
        Assert.Equal(new[] { 2, samples }, result.Weights.Shape.ToArray());
        Assert.Equal(new[] { 2, 1 }, result.WeightSum.Shape.ToArray());
        Assert.All(H(result.Color), c => Assert.InRange(c, 0.0, 1.0));
        // The ray through the center crosses the initial sphere, so it is nearly opaque.
        Assert.True(H(result.WeightSum)[0] > 0.9);
        var camera = NeuSCamera.LookAt(new[] { 0.0, 0.0, -2.5 }, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, 4.0, 4, 3);
        var image = field.RenderImage(camera, 4, 3, raysPerBatch: 5);
        Assert.Equal(new[] { 3, 4, 3 }, image.Shape.ToArray());
    }

    [Fact]
    public void Reconstructor_IsDeterministicUnderASeed()
    {
        var a = new NeuSReconstructor<double>(TinyOptions(21), Engine);
        var b = new NeuSReconstructor<double>(TinyOptions(21), Engine);
        var views = Views(0.8);
        for (int i = 0; i < 3; i++) Assert.Equal(a.TrainStep(views), b.TrainStep(views));
        Assert.Equal(H(a.Field.SampleGrid(4, SdfGridBounds.UnitCube)), H(b.Field.SampleGrid(4, SdfGridBounds.UnitCube)));
    }

    [Fact]
    public void Reconstructor_LossFallsAndDependsOnTheImages()
    {
        var views = Views(1.0);
        var reconstructor = new NeuSReconstructor<double>(TinyOptions(31), Engine);
        var losses = new List<double>();
        reconstructor.Fit(views, 30, (_, loss) => losses.Add(loss));
        Assert.Equal(30, losses.Count);
        Assert.Equal(30, reconstructor.Iteration);
        Assert.All(losses, l => Assert.False(double.IsNaN(l) || double.IsInfinity(l)));
        Assert.True(losses.Skip(25).Average() < 0.8 * losses[0],
            $"Loss should fall: first {losses[0]}, last five {losses.Skip(25).Average()}.");

        var other = new NeuSReconstructor<double>(TinyOptions(31), Engine);
        double darkLoss = other.TrainStep(Views(0.0));
        Assert.NotEqual(losses[0], darkLoss);
    }

    [Fact]
    public void Reconstructor_TrainsWithNormalSupervision()
    {
        var options = TinyOptions(41);
        options.NormalWeight = 1.0;
        var normals = NeuSHost<double>.Filled(0.0, 8, 8, 3);
        var data = H(normals);
        for (int p = 0; p < 64; p++) data[p * 3 + 2] = -1.0;
        var views = Views(0.5, T(data, 8, 8, 3));
        var withNormals = new NeuSReconstructor<double>(options, Engine);
        var without = new NeuSReconstructor<double>(TinyOptions(41), Engine);
        double a = withNormals.TrainStep(views);
        double b = without.TrainStep(views);
        Assert.False(double.IsNaN(a));
        Assert.True(a > b, "The normal term adds a non-negative misalignment to the same batch's loss.");
    }

    [Fact]
    public void Reconstructor_RequiresMasksWhenTheMaskLossIsOn()
    {
        var options = TinyOptions(51);
        options.MaskWeight = 0.1;
        var reconstructor = new NeuSReconstructor<double>(options, Engine);
        Assert.Throws<ArgumentException>(() => reconstructor.TrainStep(Views(0.5)));
    }
}
