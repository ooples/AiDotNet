using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.ServerOptimizers;
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
}
