using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Data;
using AiDotNet.NeuralRadianceFields.Helpers;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// #1834: photometric training must differentiate through volume rendering. The per-model renderers
/// accumulated colors in host scalars, so image-space training reported a loss and updated nothing - a
/// failure every "loss is finite" test passed.
/// </summary>
public class VolumeRenderingGradientTests
{
    private const int Rays = 2, Samples = 4;

    private static (Tensor<double> Rgb, Tensor<double> Density, double[] Near, double[] Far, double[] Ts) Field(int seed)
    {
        var rng = new Random(seed);
        var rgb = new Tensor<double>(new[] { Rays * Samples, 3 });
        var density = new Tensor<double>(new[] { Rays * Samples, 1 });
        for (int i = 0; i < rgb.Length; i++) rgb[i] = rng.NextDouble();
        for (int i = 0; i < density.Length; i++) density[i] = rng.NextDouble() * 2;
        var near = new[] { 1.0, 1.5 };
        var far = new[] { 3.0, 4.0 };
        var ts = new double[Rays * Samples];
        for (int r = 0; r < Rays; r++)
            for (int s = 0; s < Samples; s++)
                ts[r * Samples + s] = near[r] + (far[r] - near[r]) * (s + rng.NextDouble() * 0.5) / Samples;
        return (rgb, density, near, far, ts);
    }

    /// <summary>The quadrature as Mildenhall et al. 2020 eq. 3 writes it, in plain doubles.</summary>
    private static double[] Reference(Tensor<double> rgb, Tensor<double> density, double[] near, double[] far, double[] ts)
    {
        var colors = new double[Rays * 3];
        for (int r = 0; r < Rays; r++)
        {
            double transmittance = 1;
            for (int s = 0; s < Samples; s++)
            {
                int i = r * Samples + s;
                double t1 = s + 1 < Samples ? ts[i + 1] : far[r];
                double alpha = 1 - Math.Exp(-density[i] * Math.Max(0, t1 - ts[i]));
                for (int c = 0; c < 3; c++) colors[r * 3 + c] += transmittance * alpha * rgb[i * 3 + c];
                transmittance *= 1 - alpha;
            }
        }

        return colors;
    }

    [Fact]
    public void Render_MatchesTheReferenceQuadrature()
    {
        var (rgb, density, near, far, ts) = Field(1);
        var rendered = VolumeRenderer<double>.Render(AiDotNetEngine.Current, rgb, density, Rays, Samples, near, far, ts);
        var expected = Reference(rgb, density, near, far, ts);
        Assert.Equal(new[] { Rays, 3 }, rendered.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], rendered[i], 12);
    }

    [Fact]
    public void Render_GradientMatchesCentralDifferences()
    {
        // L = sum of rendered colors weighted by fixed coefficients, differentiated w.r.t. every sample's
        // color and density: the gradient the photometric loss sends into the radiance field.
        var (rgb, density, near, far, ts) = Field(2);
        var coefficients = new double[] { 0.3, -0.7, 1.1, 0.5, 0.9, -0.2 };
        double Loss(Tensor<double> c, Tensor<double> d)
        {
            var colors = Reference(c, d, near, far, ts);
            return colors.Select((v, i) => v * coefficients[i]).Sum();
        }

        var engine = AiDotNetEngine.Current;
        Tensor<double> rgbGrad, densityGrad;
        using (var tape = new GradientTape<double>())
        {
            var rendered = VolumeRenderer<double>.Render(engine, rgb, density, Rays, Samples, near, far, ts);
            var weighted = engine.TensorMultiply(rendered, new Tensor<double>(new[] { Rays, 3 }, new Vector<double>(coefficients)));
            var loss = engine.ReduceSum(weighted, null);
            var gradients = tape.ComputeGradients(loss, new[] { rgb, density });
            rgbGrad = gradients[rgb];
            densityGrad = gradients[density];
        }

        const double h = 1e-6;
        foreach (var (tensor, analytic) in new[] { (rgb, rgbGrad), (density, densityGrad) })
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                double original = tensor[i];
                tensor[i] = original + h;
                double plus = Loss(rgb, density);
                tensor[i] = original - h;
                double minus = Loss(rgb, density);
                tensor[i] = original;
                Assert.Equal((plus - minus) / (2 * h), analytic[i], 6);
            }
        }
    }

    private static ImageView<float>[] Views()
    {
        var views = new ImageView<float>[2];
        for (int v = 0; v < views.Length; v++)
        {
            var photo = new float[4 * 4 * 3];
            for (int i = 0; i < photo.Length; i++) photo[i] = i % 3 == 0 ? 0.9f : 0.2f;
            var rotation = new Matrix<float>(3, 3);
            rotation[0, 0] = 1f; rotation[1, 1] = 1f; rotation[2, 2] = 1f;
            views[v] = new ImageView<float>(new Tensor<float>(new[] { 4, 4, 3 }, new Vector<float>(photo)),
                new Vector<float>(new[] { 0f, 0f, v * -1f }), rotation, focalLength: 0f);
        }

        return views;
    }

    private static NeRF<float> SmallNeRF() => new(options: new NeRFOptions
    {
        PositionEncodingLevels = 2, DirectionEncodingLevels = 2, HiddenDim = 8, NumLayers = 1,
        ColorHiddenDim = 4, ColorNumLayers = 1, UseHierarchicalSampling = false, RenderSamples = 4,
        RenderNearBound = 1.0, RenderFarBound = 3.0, LearningRate = 1e-2,
    });

    [Fact]
    public void NeRF_PhotometricStep_MovesTheFieldWeights()
    {
        var nerf = SmallNeRF();
        var loader = ImageTrainingDataLoaders.FromViews(Views(), seed: 3);
        nerf.TrainOnImageBatch(loader, raysPerBatch: 8, optimizerOptions: null);
        var before = nerf.GetParameters().ToArray();

        nerf.TrainOnImageBatch(loader, raysPerBatch: 8, optimizerOptions: null);

        var after = nerf.GetParameters().ToArray();
        Assert.True(before.Zip(after, (a, b) => a != b).Any(changed => changed),
            "A photometric training step left every NeRF weight unchanged: no gradient reached the field.");
    }

    [Fact]
    public void NeRF_PhotometricTraining_LowersTheLoss()
    {
        // Judged on ONE fixed ray batch: comparing the losses of successive random batches would pass on
        // batch-to-batch noise alone, even if no weight ever moved.
        var nerf = SmallNeRF();
        var loader = ImageTrainingDataLoaders.FromViews(Views(), seed: 5);
        var (_, fixedBatch) = loader.IterateBatches(32).First();
        float Loss()
        {
            var rendered = nerf.RenderRays(fixedBatch.RayOrigins, fixedBatch.RayDirections, 4, 1f, 3f);
            double sum = 0;
            for (int i = 0; i < rendered.Length; i++) sum += Math.Pow(rendered[i] - fixedBatch.TargetColors[i], 2);
            return (float)(sum / rendered.Length);
        }

        float before = Loss();
        for (int step = 0; step < 60; step++) nerf.TrainOnImageBatch(loader, raysPerBatch: 16, optimizerOptions: null);
        float after = Loss();
        Assert.True(after < before, $"Sixty photometric steps did not lower the loss on a fixed batch ({before} -> {after}).");
    }

    [Fact]
    public void InstantNGP_PhotometricStep_MovesTheFieldWeights()
    {
        var ngp = new InstantNGP<float>();
        var loader = ImageTrainingDataLoaders.FromViews(Views(), seed: 9);
        ngp.TrainOnImageBatch(loader, raysPerBatch: 8, optimizerOptions: null);
        var before = ngp.GetParameters().ToArray();

        ngp.TrainOnImageBatch(loader, raysPerBatch: 8, optimizerOptions: null);

        var after = ngp.GetParameters().ToArray();
        Assert.True(before.Zip(after, (a, b) => a != b).Any(changed => changed),
            "A photometric training step left every InstantNGP weight unchanged: no gradient reached the field.");
    }
}
