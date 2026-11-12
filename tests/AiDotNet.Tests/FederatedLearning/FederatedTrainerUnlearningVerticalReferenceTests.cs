using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.FederatedLearning.Vertical;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Reference values (worked out independently of the implementation) for the federated trainers,
/// unlearning, verification and vertical components moved onto engine ops.
/// </summary>
public class FederatedTrainerUnlearningVerticalReferenceTests
{
    private const double Tolerance = 1e-9;

    [Fact]
    public void FedYogi_FollowsReddiAlgorithm2_WithoutBiasCorrection()
    {
        // Reddi et al. 2021, Algorithm 2: v_{-1} = tau^2, m = b1 m + (1-b1) g,
        // v = v + (1-b2) sign(g^2 - v) g^2, x = x + lr m / (sqrt(v) + tau); lr 0.1, b1 0.9, b2 0.99, tau 1e-3.
        var optimizer = new FedYogiServerOptimizer<double>(learningRate: 0.1, beta1: 0.9, beta2: 0.99, epsilon: 1e-3);

        var first = optimizer.Step(new Vector<double>(new[] { 0.5, 1.0 }), new Vector<double>(new[] { 1.5, 2.0 }));
        Assert.Equal(0.5990049998750062, first[0], Tolerance);
        Assert.Equal(1.0990049998750062, first[1], Tolerance);

        var second = optimizer.Step(first, new Vector<double>(new[] { 1.0, 1.0 }));
        Assert.Equal(0.7186423017993667, second[0], Tolerance);
        Assert.Equal(1.1786951172907454, second[1], Tolerance);
    }

    private static SplitNeuralNetwork<double> UnitTopModel(VflAggregationMode mode, double[] attention)
    {
        // Top model with 1 hidden unit: W1 = [[1]], b1 = [0], W2 = [[1]], b2 = [0].
        var model = new SplitNeuralNetwork<double>(
            attention.Length, 1, 1, new SplitModelOptions { AggregationMode = mode, TopModelHiddenDimension = 1 }, seed: 7);
        model.SetTopModelParameters(new[]
        {
            new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 }),
            new Tensor<double>(new[] { 0.0 }, new[] { 1 }),
            new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 }),
            new Tensor<double>(new[] { 0.0 }, new[] { 1 }),
            new Tensor<double>(attention, new[] { attention.Length }),
        });
        return model;
    }

    [Fact]
    public void SplitNetwork_AttentionBackward_ScalesByTheSoftmaxWeightUsedInForward()
    {
        // out = sum_p softmax(a)_p e_p, so d out / d e_p = softmax(a)_p. With a = [0, ln 3] that is [0.25, 0.75]
        // (the raw weights 0 and 1.0986 were used before).
        var model = UnitTopModel(VflAggregationMode.Attention, new[] { 0.0, Math.Log(3.0) });
        var embeddings = new[]
        {
            new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 }),
            new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 }),
        };

        var combined = model.AggregateEmbeddings(embeddings);
        Assert.Equal(1.0, combined.GetFlat(0), Tolerance);
        model.ForwardTopModel(combined);

        var grads = model.BackwardTopModel(new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 }), embeddings);
        Assert.Equal(0.25, grads[0].GetFlat(0), Tolerance);
        Assert.Equal(0.75, grads[1].GetFlat(0), Tolerance);
    }

    [Fact]
    public void SplitNetwork_SumOfOneDimensionalEmbeddings_IsOneSample()
    {
        // A 1-D embedding [N] is one sample of width N (as Concatenation already treats it):
        // [1, 2] + [3, 4] = [[4, 6]]. The batch used to be taken as N and read past the end.
        var model = new SplitNeuralNetwork<double>(2, 2, 1, new SplitModelOptions { AggregationMode = VflAggregationMode.Sum }, seed: 7);
        var combined = model.AggregateEmbeddings(new[]
        {
            new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 }),
            new Tensor<double>(new[] { 3.0, 4.0 }, new[] { 2 }),
        });

        Assert.Equal(new[] { 1, 2 }, combined.Shape.ToArray());
        Assert.Equal(new[] { 4.0, 6.0 }, combined.ToArray());
    }
}
