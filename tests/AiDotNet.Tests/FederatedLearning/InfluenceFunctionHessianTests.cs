using System;
using System.Collections.Generic;
using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Influence-function unlearning (Koh and Liang 2017) solves H x = v with Hessian-vector products. Each case has a
/// full (non-diagonal) Hessian worked by hand.
/// </summary>
public class InfluenceFunctionHessianTests
{
    private const double Damping = 0.01;

    private static Tensor<double> Vec(params double[] values) => new Tensor<double>(values, new[] { values.Length });

    // H = [[2, 1], [1, 3]]
    private static Tensor<double> Coupled(Tensor<double> v) => Vec(2 * v[0] + v[1], v[0] + 3 * v[1]);

    [Fact]
    public void SolveDamped_InvertsAFullHessian()
    {
        // H^{-1} = (1/5) [[3, -1], [-1, 2]], so H^{-1} [1, 2] = [0.2, 0.6]. A diagonal stand-in would give [0.5, 2/3].
        var x = InfluenceFunctionUnlearner<double>.SolveDamped(Coupled, Vec(1, 2), 0.0, 10, 1e-12);

        Assert.Equal(0.2, x[0], 10);
        Assert.Equal(0.6, x[1], 10);
    }

    [Fact]
    public void Unlearn_TakesTheNewtonStepWithTheSuppliedHessian()
    {
        // v = [1, 2] from the target client, two clients, H + 0.01 I = [[2.01, 1], [1, 3.01]]:
        // (H + 0.01 I)^{-1} v = [3.01 - 2, -1 + 4.02] / (2.01 * 3.01 - 1) and the model moves by half of it.
        var options = new FederatedUnlearningOptions { MaxInfluenceIterations = 10, InfluenceTolerance = 1e-12 };
        var unlearner = new InfluenceFunctionUnlearner<double>(options, (parameters, v) => Coupled(v));
        var histories = new Dictionary<int, List<Tensor<double>>>
        {
            [0] = new List<Tensor<double>> { Vec(1, 2) },
            [1] = new List<Tensor<double>> { Vec(5, 5) },
        };

        var (model, _) = unlearner.Unlearn(0, Vec(0, 0), histories);

        double det = 2.01 * 3.01 - 1.0;
        Assert.Equal(-0.5 * 1.01 / det, model[0], 10);
        Assert.Equal(-0.5 * 3.02 / det, model[1], 10);
    }

    [Fact]
    public void Unlearn_WithoutAHessian_UsesTheFisherOfTheRetainedUpdates()
    {
        // Retained updates [1, 1] and [1, -1]: H = (1/2)([[1, 1], [1, 1]] + [[1, -1], [-1, 1]]) = I, so
        // (H + 0.01 I)^{-1} [1, 2] = [1, 2] / 1.01, halved over two clients.
        var options = new FederatedUnlearningOptions { MaxInfluenceIterations = 10, InfluenceTolerance = 1e-12 };
        var unlearner = new InfluenceFunctionUnlearner<double>(options);
        var histories = new Dictionary<int, List<Tensor<double>>>
        {
            [0] = new List<Tensor<double>> { Vec(1, 2) },
            [1] = new List<Tensor<double>> { Vec(1, 1), Vec(1, -1) },
        };

        var (model, _) = unlearner.Unlearn(0, Vec(0, 0), histories);

        Assert.Equal(-0.5 / (1.0 + Damping), model[0], 10);
        Assert.Equal(-1.0 / (1.0 + Damping), model[1], 10);
    }

    [Fact]
    public void Unlearn_FisherCouplesTheParameters()
    {
        // One retained update [1, 1]: H = [[1, 1], [1, 1]], so H + 0.01 I = [[1.01, 1], [1, 1.01]] with
        // inverse [[1.01, -1], [-1, 1.01]] / (1.01^2 - 1). Applied to v = [1, 0]: [1.01, -1] / 0.0201, halved.
        var options = new FederatedUnlearningOptions { MaxInfluenceIterations = 10, InfluenceTolerance = 1e-12 };
        var unlearner = new InfluenceFunctionUnlearner<double>(options);
        var histories = new Dictionary<int, List<Tensor<double>>>
        {
            [0] = new List<Tensor<double>> { Vec(1, 0) },
            [1] = new List<Tensor<double>> { Vec(1, 1) },
        };

        var (model, _) = unlearner.Unlearn(0, Vec(0, 0), histories);

        double det = 1.01 * 1.01 - 1.0;
        Assert.Equal(-0.5 * 1.01 / det, model[0], 8);
        Assert.Equal(0.5 / det, model[1], 8);
    }
}
