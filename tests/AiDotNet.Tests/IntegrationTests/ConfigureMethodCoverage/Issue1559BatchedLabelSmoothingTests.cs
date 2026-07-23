using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// #1559 guard: label smoothing (Vaswani et al. 2017, ε = 0.1) must preserve
/// batched memorization accuracy while keeping the softmax off the hard 0/1
/// confidence rail. Earlier optimizer behavior caused the un-smoothed path to
/// regress and freeze; now that both paths can reach full top-1 accuracy, the
/// durable contract is equal-or-better accuracy with strictly less overconfidence.
/// </summary>
public sealed class Issue1559BatchedLabelSmoothingTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _o;
    public Issue1559BatchedLabelSmoothingTests(ITestOutputHelper o) { _o = o; }

    [Fact]
    public void BatchedTraining_LabelSmoothing_PreservesAccuracyAndReducesOverconfidence()
    {
        var (features, labels) = MakeMemorizationSet();
        int batch = features.Shape[0], ctx = features.Shape[1], vocab = labels.Shape[1];

        // Per-example one-hot pairs that TrainBatched stacks into a [B, ctx] / [B, vocab] batch.
        var inputs = new Tensor<float>[batch];
        var targets = new Tensor<float>[batch];
        for (int b = 0; b < batch; b++)
        {
            inputs[b] = new Tensor<float>([1, ctx]);
            for (int s = 0; s < ctx; s++) inputs[b][0, s] = features[b, s];
            targets[b] = new Tensor<float>([1, vocab]);
            for (int v = 0; v < vocab; v++) targets[b][0, v] = labels[b, v];
        }

        const int steps = 300;

        // Baseline: plain one-hot CE can converge, but tends to hard saturation.
        var baseline = new Transformer<float>(MakeCanaryArch(42), new CategoricalCrossEntropyLoss<float>());
        baseline.SetTrainingMode(true);
        for (int s = 0; s < steps; s++) baseline.TrainBatched(inputs, targets);
        double baseTop = MeasureTrainingTopOne(baseline, features, labels);
        double baseSpread = MeasurePredictionSpread(baseline, features);

        // Label smoothing ε = 0.1 keeps the softmax away from hard saturation.
        var smoothed = new Transformer<float>(
            MakeCanaryArch(42), new CategoricalCrossEntropyLoss<float>(labelSmoothing: 0.1));
        smoothed.SetTrainingMode(true);
        for (int s = 0; s < steps; s++) smoothed.TrainBatched(inputs, targets);
        double smoothTop = MeasureTrainingTopOne(smoothed, features, labels);
        double smoothSpread = MeasurePredictionSpread(smoothed, features);

        _o.WriteLine($"baseline (no LS):  top1={baseTop:P1} spread={baseSpread:F4}");
        _o.WriteLine($"smoothed (LS 0.1): top1={smoothTop:P1} spread={smoothSpread:F4}");

        // 1) Smoothing must preserve the baseline's classification accuracy.
        Assert.True(smoothTop >= baseTop,
            $"Label smoothing should preserve or improve batched convergence; "
            + $"got smoothed={smoothTop:P1} vs baseline={baseTop:P1}.");

        // 2) It must still reach a healthy top-1 on the memorization set.
        Assert.True(smoothTop >= 0.80,
            $"Batched training with label smoothing should reach >=80% top-1 on the {batch}-example "
            + $"memorization set; got {smoothTop:P1}.");

        // 3) It must strictly reduce overconfidence relative to unsmoothed CE.
        Assert.True(smoothSpread < baseSpread,
            $"Label smoothing should reduce prediction spread below the un-smoothed baseline; "
            + $"got smoothed={smoothSpread:F4} vs baseline={baseSpread:F4}.");

        // 4) It must leave the softmax OFF the hard 0/1 rail.
        Assert.True(smoothSpread < 0.99,
            $"With label smoothing the softmax must stay off the hard 0/1 rail (spread < 0.99); "
            + $"got {smoothSpread:F4}.");
    }
}
