using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SelfSupervisedLearning.Evaluation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SelfSupervisedLearning;

/// <summary>
/// k-NN evaluation of a frozen encoder's features (Wu et al. 2018; the SSL linear-probe alternative).
/// </summary>
public class KNNEvaluatorTests
{
    private static FeedForwardNeuralNetwork<double> Encoder() =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 3));

    // Two classes on opposite rays: any encoder that is not degenerate keeps them apart under cosine similarity,
    // because a linear map sends x and -x to exact opposites before the activations.
    private static (Tensor<double> Data, int[] Labels) Clusters(int perClass, double offset)
    {
        var data = new Tensor<double>(new[] { 2 * perClass, 4 });
        var labels = new int[2 * perClass];
        for (int i = 0; i < 2 * perClass; i++)
        {
            int label = i < perClass ? 0 : 1;
            double sign = label == 0 ? 1.0 : -1.0;
            double scale = 1.0 + offset + 0.1 * (i % perClass);
            data[i, 0] = sign * scale;
            data[i, 1] = sign * 0.5 * scale;
            data[i, 2] = sign * -0.25 * scale;
            data[i, 3] = sign * 0.75 * scale;
            labels[i] = label;
        }

        return (data, labels);
    }

    [Fact]
    public void SeparatedClusters_AreClassifiedPerfectly()
    {
        var evaluator = new KNNEvaluator<double>(Encoder(), k: 3);
        var (train, trainLabels) = Clusters(6, 0.0);
        var (test, testLabels) = Clusters(3, 0.05);

        evaluator.Fit(train, trainLabels);

        Assert.Equal(1.0, evaluator.Evaluate(test, testLabels));
        Assert.Equal(testLabels, evaluator.PredictClasses(test));
    }

    [Fact]
    public void EvaluateMultipleK_ReportsEveryRequestedK()
    {
        var evaluator = new KNNEvaluator<double>(Encoder(), k: 3);
        var (train, trainLabels) = Clusters(6, 0.0);
        var (test, testLabels) = Clusters(3, 0.05);
        evaluator.Fit(train, trainLabels);

        var byK = evaluator.EvaluateMultipleK(test, testLabels, new[] { 1, 3, 5 });

        Assert.Equal(new[] { 1, 3, 5 }, byK.Keys);
        foreach (var accuracy in byK.Values)
            Assert.Equal(1.0, accuracy);
    }

    [Fact]
    public void Evaluating_BeforeFit_Throws()
    {
        var evaluator = new KNNEvaluator<double>(Encoder());
        var (test, labels) = Clusters(2, 0.0);

        Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(test, labels));
        Assert.Throws<InvalidOperationException>(() => evaluator.PredictClasses(test));
    }
}
