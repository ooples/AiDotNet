using System;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// GANDALF's TrainStep used to compute the loss and then call UpdateParameters with no gradients, so the gating
/// network, the trees and the head never moved and the loss never fell. These pin that a step now trains.
/// </summary>
public class GANDALFTrainingTests
{
    private const int NumFeatures = 3;

    private static GANDALFOptions<double> Opts() => new()
    {
        NumTrees = 2,
        TreeDepth = 2,
        GatingHiddenDimension = 8,
        NumGatingLayers = 1,
        DropoutRate = 0.0,
        UseBatchNorm = false,
        InitScale = 0.1,
    };

    // Two separable classes: class = i % 2, encoded in feature[class] with tiny deterministic jitter.
    private static (Tensor<double> x, Vector<int> y) MakeSeparable(int n)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        var x = new Tensor<double>(new[] { n, NumFeatures });
        var y = new Vector<int>(n);
        for (int i = 0; i < n; i++)
        {
            int cls = i % 2;
            y[i] = cls;
            for (int f = 0; f < NumFeatures; f++)
                x[i, f] = (f == cls ? 1.0 : 0.0) + 0.01 * (rng.NextDouble() - 0.5);
        }
        return (x, y);
    }

    [Fact]
    public void ClassifierTrainStep_ChangesParameters_AndLowersTheLoss()
    {
        var model = new GANDALFClassifier<double>(NumFeatures, 2, Opts());
        var (x, y) = MakeSeparable(8);
        _ = model.PredictProbabilities(x);
        var before = model.GetParameters().ToArray();

        double firstLoss = Convert.ToDouble(model.TrainStep(x, y, 0.1));
        var after = model.GetParameters().ToArray();
        int changed = 0;
        for (int i = 0; i < before.Length; i++) if (before[i] != after[i]) changed++;
        Assert.True(changed > before.Length / 2, $"Only {changed} of {before.Length} parameters changed after a training step.");

        double lastLoss = firstLoss;
        for (int step = 0; step < 30; step++) lastLoss = Convert.ToDouble(model.TrainStep(x, y, 0.1));
        Assert.True(!double.IsNaN(lastLoss) && !double.IsInfinity(lastLoss), "Training loss became non-finite.");
        Assert.True(lastLoss < firstLoss, $"Cross-entropy did not fall: first {firstLoss}, last {lastLoss}.");
    }

    [Fact]
    public void RegressionTrainStep_ChangesParameters_AndLowersTheLoss()
    {
        var model = new GANDALFRegression<double>(NumFeatures, 1, Opts());
        var (x, labels) = MakeSeparable(8);
        var targets = new Tensor<double>(new[] { 8, 1 });
        for (int i = 0; i < 8; i++) targets[i, 0] = labels[i];
        _ = model.Predict(x);
        var before = model.GetParameters().ToArray();

        double firstLoss = Convert.ToDouble(model.TrainStep(x, targets, 0.05));
        var after = model.GetParameters().ToArray();
        int changed = 0;
        for (int i = 0; i < before.Length; i++) if (before[i] != after[i]) changed++;
        Assert.True(changed > before.Length / 2, $"Only {changed} of {before.Length} parameters changed after a training step.");

        double lastLoss = firstLoss;
        for (int step = 0; step < 30; step++) lastLoss = Convert.ToDouble(model.TrainStep(x, targets, 0.05));
        Assert.True(!double.IsNaN(lastLoss) && !double.IsInfinity(lastLoss), "Training loss became non-finite.");
        Assert.True(lastLoss < firstLoss, $"MSE did not fall: first {firstLoss}, last {lastLoss}.");
    }
}
