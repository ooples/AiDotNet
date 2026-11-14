using System;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// The pFedMe client update (Dinh, Tran and Nguyen 2020, Algorithm 1) on f(theta) = 1/2 (theta - 4)^2, worked by hand.
/// </summary>
public class PFedMeLocalUpdateTests
{
    private static Vector<double> Gradient(Vector<double> theta) => new Vector<double>(new[] { theta[0] - 4.0 });

    private static FederatedPersonalizationOptions Options(int innerSteps) => new FederatedPersonalizationOptions
    {
        Enabled = true,
        Strategy = FederatedPersonalizationStrategy.PFedMe,
        PFedMeMu = 1.0,
        PFedMeInnerSteps = innerSteps,
        PFedMePersonalLearningRate = 0.5,
        PFedMeLearningRate = 0.5
    };

    [Fact]
    public void LocalUpdate_SolvesTheMoreauEnvelopeThenMovesTheLocalModel()
    {
        // lambda = 1, personal rate 0.5, eta = 0.5, K = 2, R = 2, w0 = theta0 = 0.
        // Round 1: theta: 0 -> 0 - 0.5 ((0 - 4) + (0 - 0)) = 2 -> 2 - 0.5 ((2 - 4) + (2 - 0)) = 2; w = 0 - 0.5 (0 - 2) = 1.
        // Round 2: theta: 2 -> 2 - 0.5 ((2 - 4) + (2 - 1)) = 2.5 -> 2.5 - 0.5 (-1.5 + 1.5) = 2.5; w = 1 - 0.5 (1 - 2.5) = 1.75.
        var w = InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>.PFedMeLocalUpdate(
            Gradient, new Vector<double>(new[] { 0.0 }), Options(2), 2, out var theta);

        Assert.Equal(1.75, w[0], 12);
        Assert.Equal(2.5, theta[0], 12);
    }

    [Fact]
    public void LocalUpdate_PersonalizedModelSolvesTheProximalProblem()
    {
        // With many inner steps theta reaches argmin 1/2 (theta - 4)^2 + 1/2 (theta - w)^2 = (4 + w) / 2 = 2 for w = 0,
        // while w moves only eta * lambda of the way: 0 - 0.5 (0 - 2) = 1. The local optimizer's own result (the old
        // update mixed it into theta) plays no part.
        var w = InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>.PFedMeLocalUpdate(
            Gradient, new Vector<double>(new[] { 0.0 }), Options(50), 1, out var theta);

        Assert.Equal(2.0, theta[0], 9);
        Assert.Equal(1.0, w[0], 9);
    }

    [Fact]
    public void LocalUpdate_RejectsAGradientOfTheWrongLength()
    {
        Assert.Throws<InvalidOperationException>(() =>
            InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>.PFedMeLocalUpdate(
                _ => new Vector<double>(2), new Vector<double>(new[] { 0.0 }), Options(1), 1, out _));
    }
}
