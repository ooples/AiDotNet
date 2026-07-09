using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralRadianceFields.Data;
using AiDotNet.NeuralRadianceFields.Interfaces;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Regression tests for #1834 — the image-space photometric training path.
///   * Loader factory <see cref="ImageTrainingDataLoaders.FromViews{T}"/> produces a valid
///     <c>IDataLoader&lt;ImageView&lt;T&gt;, PixelBatch&lt;T&gt;&gt;</c>.
///   * <c>GaussianSplatting</c> / <c>NeRF</c> / <c>InstantNGP</c> all implement
///     <see cref="IImageTrainable{T}"/> and their <c>TrainOnImageBatch</c> returns a finite
///     MSE loss end-to-end.
/// </summary>
public class ImageTrainingPathTests
{
    private static ImageView<float>[] BuildViews(int count = 2, int H = 4, int W = 4)
    {
        var views = new ImageView<float>[count];
        for (int v = 0; v < count; v++)
        {
            var photo = new float[H * W * 3];
            for (int i = 0; i < photo.Length; i++) photo[i] = 0.5f;
            var pose = new float[] { 0f, 0f, v * -1f };
            var rot = new Matrix<float>(3, 3);
            rot[0, 0] = 1f; rot[1, 1] = 1f; rot[2, 2] = 1f;
            views[v] = new ImageView<float>(
                new Tensor<float>(new[] { H, W, 3 }, new Vector<float>(photo)),
                new Vector<float>(pose),
                rot,
                focalLength: 0f);
        }
        return views;
    }

    [Fact]
    public void ImageTrainingDataLoaders_FromViews_EmitsRayBatchesOfRequestedSize()
    {
        var loader = ImageTrainingDataLoaders.FromViews(BuildViews(), seed: 42);
        Assert.True(loader.IsLoaded);
        Assert.Equal(2, loader.TotalCount);

        var enumerator = loader.IterateBatches(batchSize: 8).GetEnumerator();
        Assert.True(enumerator.MoveNext());
        var (view, pixels) = enumerator.Current;
        Assert.NotNull(view);
        Assert.NotNull(pixels);
        Assert.Equal(8, pixels.Count);
        Assert.Equal(new[] { 8, 3 }, pixels.RayOrigins.Shape);
        Assert.Equal(new[] { 8, 3 }, pixels.RayDirections.Shape);
        Assert.Equal(new[] { 8, 3 }, pixels.TargetColors.Shape);
    }

    [Fact]
    public void ImageTrainingDataLoaders_FromViews_NullThrows()
        => Assert.Throws<ArgumentNullException>(() => ImageTrainingDataLoaders.FromViews<float>(null!));

    [Fact]
    public void ImageTrainingDataLoaders_FromViews_EmptyThrows()
        => Assert.Throws<ArgumentException>(() => ImageTrainingDataLoaders.FromViews<float>(Array.Empty<ImageView<float>>()));

    [Fact]
    public void GaussianSplatting_ImplementsIImageTrainable()
    {
        var gs = new GaussianSplatting<float>(
            new AiDotNet.Models.Options.GaussianSplattingOptions
            {
                EnableDensification = false,
                EnableSpatialIndex = false,
                MaxGaussians = 8,
                ShDegree = 0,
            });
        Assert.IsAssignableFrom<IImageTrainable<float>>(gs);
    }

    [Fact]
    public void NeRF_ImplementsIImageTrainable()
    {
        var nerf = new NeRF<float>(
            positionEncodingLevels: 2, directionEncodingLevels: 2, hiddenDim: 8, numLayers: 1,
            colorHiddenDim: 4, colorNumLayers: 1, useHierarchicalSampling: false,
            renderSamples: 2, renderNearBound: 1.0, renderFarBound: 3.0, learningRate: 1e-3);
        Assert.IsAssignableFrom<IImageTrainable<float>>(nerf);
    }

    [Fact]
    public void NeRF_TrainOnImageBatch_ReturnsFiniteLoss()
    {
        var nerf = new NeRF<float>(
            positionEncodingLevels: 2, directionEncodingLevels: 2, hiddenDim: 8, numLayers: 1,
            colorHiddenDim: 4, colorNumLayers: 1, useHierarchicalSampling: false,
            renderSamples: 2, renderNearBound: 1.0, renderFarBound: 3.0, learningRate: 1e-3);
        var loader = ImageTrainingDataLoaders.FromViews(BuildViews(), seed: 7);

        float loss = nerf.TrainOnImageBatch(loader, raysPerBatch: 4, optimizerOptions: null);
        Assert.True(float.IsFinite(loss), $"loss should be finite; got {loss}.");
        Assert.True(loss >= 0f, "MSE loss can't be negative.");
    }

    [Fact]
    public void ImageTrainable_TrainOnImageBatch_NullLoader_ThrowsArgumentNullException()
    {
        var nerf = new NeRF<float>();
        Assert.Throws<ArgumentNullException>(
            () => nerf.TrainOnImageBatch(null!, raysPerBatch: 16, optimizerOptions: null));
    }

    [Fact]
    public void InstantNGP_TrainOnImageBatch_ReturnsFiniteLoss()
    {
        var ngp = new InstantNGP<float>();
        var loader = ImageTrainingDataLoaders.FromViews(BuildViews(), seed: 11);
        float loss = ngp.TrainOnImageBatch(loader, raysPerBatch: 4, optimizerOptions: null);
        Assert.True(float.IsFinite(loss), $"InstantNGP loss should be finite; got {loss}.");
        Assert.True(loss >= 0f, "MSE loss can't be negative.");
    }

    [Fact]
    public void GaussianSplatting_TrainOnImageBatch_ReturnsFiniteLoss()
    {
        var gs = new GaussianSplatting<float>(
            new AiDotNet.Models.Options.GaussianSplattingOptions
            {
                EnableDensification = false,
                EnableSpatialIndex = false,
                MaxGaussians = 8,
                ShDegree = 0,
            });
        var loader = ImageTrainingDataLoaders.FromViews(BuildViews(count: 1, H: 4, W: 4), seed: 13);
        float loss = gs.TrainOnImageBatch(loader, raysPerBatch: 4, optimizerOptions: null);
        Assert.True(float.IsFinite(loss), $"GS loss should be finite; got {loss}.");
        Assert.True(loss >= 0f, "Loss can't be negative.");
    }
}
