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
    public void EvaluateMultipleK_ScoresEachKOnItsOwnNeighbours()
    {
        // A huge temperature makes the similarity weights equal, so the vote is a plain majority. The training set
        // is one exact copy of the query, labelled 1, and two copies of another point, labelled 0. Identical
        // inputs encode identically, so the copy is the nearest neighbour whatever the encoder: k = 1 votes 1 and
        // is right, while k = 3 takes all three points and the two 0s outvote it.
        var evaluator = new KNNEvaluator<double>(Encoder(), k: 3, temperature: 1e6);
        double[] query = { 1.0, 0.5, -0.25, 0.75 };
        double[] other = { -0.5, 1.0, 0.75, -1.0 };
        var train = new Tensor<double>(new[] { 3, 4 });
        var test = new Tensor<double>(new[] { 1, 4 });
        for (int j = 0; j < 4; j++)
        {
            train[0, j] = query[j];
            train[1, j] = other[j];
            train[2, j] = other[j];
            test[0, j] = query[j];
        }

        evaluator.Fit(train, new[] { 1, 0, 0 });
        var byK = evaluator.EvaluateMultipleK(test, new[] { 1 }, new[] { 1, 3 });

        Assert.Equal(1.0, byK[1]);
        Assert.Equal(0.0, byK[3]);
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
