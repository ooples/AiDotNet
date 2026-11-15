using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// MVDream's parameter surface and its ImageDream-style image path (Wang and Shi, "ImageDream", 2023; the
/// mvdream_diffusers pipeline LGM uses): the image latent rides along as an extra zero-camera frame, the cameras
/// follow the requested elevation, and the result is deterministic under a seed and depends on the image.
/// </summary>
public class MVDreamImageConditioningTests
{
    private const int Seed = 42;

    private static MultiViewUNet<double> TinyMultiViewUNet() => new MultiViewUNet<double>(
        inputChannels: 4, outputChannels: 4, baseChannels: 32, numViews: 4, contextDim: 1024, seed: Seed);

    private static StandardVAE<double> TinyVae() => new StandardVAE<double>(
        inputChannels: 3, latentChannels: 4, baseChannels: 16,
        channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: Seed);

    private static MVDreamModel<double> TinyModel(out MultiViewUNet<double> multiViewUNet, out StandardVAE<double> vae)
    {
        multiViewUNet = TinyMultiViewUNet();
        vae = TinyVae();
        return new MVDreamModel<double>(multiViewUNet: multiViewUNet, imageVAE: vae, seed: Seed);
    }

    internal static Tensor<double> RandomImage(int seed, int size = 32)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var data = new double[3 * size * size];
        for (int i = 0; i < data.Length; i++) data[i] = (2.0 * rng.NextDouble()) - 1.0;
        return new Tensor<double>(data, new[] { 1, 3, size, size });
    }

    [Fact]
    public void ParameterSurface_CountsEveryModuleOnce()
    {
        var model = TinyModel(out var multiViewUNet, out var vae);

        // Base U-Net (the NoisePredictor slot), VAE, multi-view attention (query, key, value and output C x C
        // matrices over the C = 4 predicted channels) and the camera MLP (16 -> E -> E, both with bias), each once.
        int embedding = multiViewUNet.BaseUNet.TimeEmbeddingDim;
        long attention = 4L * 4 * 4;
        long camera = (16L * embedding) + embedding + ((long)embedding * embedding) + embedding;
        long expected = multiViewUNet.BaseUNet.ParameterCount + vae.ParameterCount + attention + camera;

        Assert.Equal(attention, multiViewUNet.MultiViewAttention.ParameterCount);
        Assert.Equal(camera, model.CameraEmbedding.ParameterCount);
        Assert.Equal(expected, model.ParameterCount);
        Assert.Equal(expected, model.GetParameters().Length);
    }

    [Fact]
    public void ImagePath_ReturnsTheRequestedViewsAtTheImageResolution()
    {
        var model = TinyModel(out _, out _);
        var views = model.GenerateNovelViewsFromImage(RandomImage(1), numViews: 4, numInferenceSteps: 2,
            guidanceScale: 3.0, seed: 5);

        Assert.Equal(4, views.Length);
        foreach (var view in views)
        {
            Assert.Equal(new[] { 1, 3, 32, 32 }, view.Shape.ToArray());
            Assert.All(view.ToArray(), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
        }
    }

    [Fact]
    public void ImagePath_IsDeterministicUnderASeedAndDependsOnTheImage()
    {
        var model = TinyModel(out _, out _);
        var image = RandomImage(2);
        var first = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 3.0, seed: 9));
        var second = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 3.0, seed: 9));
        Assert.Equal(first, second);

        var other = Flatten(model.GenerateNovelViewsFromImage(RandomImage(3), 4, 2, 3.0, seed: 9));
        Assert.True(MaxAbsDifference(first, other) > 1e-9, "the generated views must depend on the input image");
    }

    [Fact]
    public void ImagePath_PlacesTheCamerasAtTheRequestedElevation()
    {
        var model = TinyModel(out _, out _);
        var image = RandomImage(4);
        Assert.Equal(30.0, model.Config.DefaultElevation);

        var byDefault = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 1.0, seed: 11));
        var explicitDefault = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 1.0, seed: 11, elevation: 30.0));
        var level = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 1.0, seed: 11, elevation: 0.0));

        Assert.Equal(byDefault, explicitDefault);
        Assert.True(MaxAbsDifference(byDefault, level) > 1e-9, "the elevation must reach the camera embedding");
    }

    private static double[] Flatten(Tensor<double>[] views) => views.SelectMany(view => view.ToArray()).ToArray();

    private static double MaxAbsDifference(double[] a, double[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }
}
