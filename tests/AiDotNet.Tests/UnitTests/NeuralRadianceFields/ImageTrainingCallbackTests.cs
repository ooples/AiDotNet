using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralRadianceFields.Data;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.TrainingMonitoring;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Pins training-callback observability on the image-space path.
/// </summary>
/// <remarks>
/// <para>
/// <c>ConfigureTrainingCallback</c> documents "return <c>false</c> to request an early stop". For every
/// <c>IImageTrainable</c> family that contract was unobservable: the facade owns the epoch loop in
/// <c>RunImageSpaceTrainingLoop</c>, and that loop reported no epochs and consulted no callback, so a caller saw
/// nothing and a veto could not stop anything.
/// </para>
/// <para>
/// This differs from the time-series case, which models own their epoch loop and solve by implementing
/// <c>ITrainingEpochReporter</c>. Here the loop belongs to the facade, so the facade is the only component that
/// can drive the callback — nothing the model implements could have fixed it.
/// </para>
/// </remarks>
public class ImageTrainingCallbackTests
{
    private static ImageView<float>[] BuildViews(int count = 2, int h = 4, int w = 4)
    {
        var views = new ImageView<float>[count];
        for (int v = 0; v < count; v++)
        {
            var photo = new float[h * w * 3];
            for (int i = 0; i < photo.Length; i++) photo[i] = 0.5f;
            var pose = new float[] { 0f, 0f, v * -1f };
            var rot = new Matrix<float>(3, 3);
            rot[0, 0] = 1f; rot[1, 1] = 1f; rot[2, 2] = 1f;
            views[v] = new ImageView<float>(
                new Tensor<float>(new[] { h, w, 3 }, new Vector<float>(photo)),
                new Vector<float>(pose),
                rot,
                focalLength: 0f);
        }

        return views;
    }

    private static AiModelBuilder<float, Tensor<float>, Tensor<float>> BuilderWithEpochs(int epochs)
    {
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            model: null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { MaxIterations = epochs });

        return (AiModelBuilder<float, Tensor<float>, Tensor<float>>)
            new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(new GaussianSplatting<float>())
                .ConfigureDataLoader(ImageTrainingDataLoaders.FromViews(BuildViews(), seed: 42))
                .ConfigureOptimizer(optimizer);
    }

    [Fact]
    public async Task Image_space_training_reports_every_epoch_to_a_configured_callback()
    {
        const int Epochs = 4;
        var observed = new List<int>();

        var builder = BuilderWithEpochs(Epochs)
            .ConfigureTrainingCallback(new DelegateTrainingCallback<float>(
                onEpochEnd: p => { observed.Add(p.Epoch); return true; }));

        await builder.BuildAsync();

        // Previously zero: the loop never reported an epoch to anybody.
        Assert.Equal(Epochs, observed.Count);
        Assert.Equal(new[] { 1, 2, 3, 4 }, observed);
    }

    [Fact]
    public async Task A_callback_returning_false_actually_stops_image_space_training()
    {
        const int Epochs = 10;
        int seen = 0;

        var builder = BuilderWithEpochs(Epochs)
            .ConfigureTrainingCallback(new DelegateTrainingCallback<float>(
                onEpochEnd: _ => { seen++; return seen < 3; }));   // veto on the third epoch

        await builder.BuildAsync();

        // The documented contract: false stops training. Before this fix all 10 epochs ran regardless.
        Assert.Equal(3, seen);
    }

    [Fact]
    public async Task The_callback_sees_the_real_epoch_budget_not_a_synthetic_total()
    {
        const int Epochs = 5;
        var totals = new List<int>();

        var builder = BuilderWithEpochs(Epochs)
            .ConfigureTrainingCallback(new DelegateTrainingCallback<float>(
                onEpochEnd: p => { totals.Add(p.TotalEpochs); return true; }));

        await builder.BuildAsync();

        Assert.NotEmpty(totals);
        Assert.All(totals, t => Assert.Equal(Epochs, t));
    }
}
