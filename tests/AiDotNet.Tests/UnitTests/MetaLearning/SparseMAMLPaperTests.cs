using AiDotNet.Data.Structures;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="SparseMAMLAlgorithm{T, TInput, TOutput}"/> against von Oswald et al.,
/// "Learning where to learn: Gradient sparsity in meta and continual learning" (NeurIPS 2021), Eq. 4 and 5.
/// </summary>
public class SparseMAMLPaperTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task()
    {
        var sx = new Matrix<double>(new[,] { { 0.5, -1.0, 0.25 }, { 1.0, 0.5, -0.5 } });
        var sy = new Vector<double>(new[] { 1.0, -0.5 });
        var qx = new Matrix<double>(new[,] { { -0.25, 0.75, 1.0 }, { 0.5, 0.5, 0.5 } });
        var qy = new Vector<double>(new[] { 0.5, 1.5 });
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = sx, SupportSetY = sy, QuerySetX = qx, QuerySetY = qy,
            NumWays = 1, NumShots = 2, NumQueryPerClass = 2,
        };
    }

    private static SparseMAMLOptions<double, Matrix<double>, Vector<double>> Options(LinearVectorModel model)
        => new(model)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            GradientClipThreshold = null,
            InnerLearningRate = 0.1,
            GateLearningRate = 0.5,
            AdaptationSteps = 1,
            InitialGateLogit = 1.0,
        };

    [Fact]
    public void Defaults_FollowThePaper()
    {
        var options = new SparseMAMLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3));
        Assert.Equal(SparseMAMLGateFunction.Step, options.GateFunction);
        Assert.Equal(0.1, options.InnerLearningRate);
        Assert.Equal(25, options.AdaptationSteps);
        Assert.Equal(0.0075, options.GateLearningRate);
    }

    [Fact]
    public void StepGate_IsBinary_OneAtZeroAndAbove()
    {
        // Eq. 4: the mask is 1[m >= 0].
        foreach (var (logit, gate) in new[] { (-1e-6, 0.0), (0.0, 1.0), (1e-6, 1.0), (-3.0, 0.0), (3.0, 1.0) })
        {
            var options = Options(new LinearVectorModel(3));
            options.InitialGateLogit = logit;
            Assert.Equal(gate, new SparseMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options).Gate(0));
        }
    }

    [Fact]
    public void GateLogitUpdate_IsEq5_WithTheStraightThroughEstimator()
    {
        var model = new LinearVectorModel(3);
        model.SetParameters(new Vector<double>(new[] { 0.3, -0.2, 0.1, 0.05 }));
        var reference = new LinearVectorModel(3);
        reference.SetParameters(model.GetParameters());
        var loss = new MeanSquaredErrorLoss<double>();
        var task = Task();

        // LinearVectorModel(3) has 3 weights and a bias.
        // phi_1 = theta - alpha * 1 * grad L_in(theta); m <- m + alpha gamma_m grad L_out(phi_1) * grad L_in(theta).
        var gIn = reference.ComputeGradients(task.SupportSetX, task.SupportSetY, loss);
        var phi = new Vector<double>(4);
        for (int d = 0; d < 4; d++) phi[d] = reference.GetParameters()[d] - 0.1 * gIn[d];
        reference.SetParameters(phi);
        var gOut = reference.ComputeGradients(task.QuerySetX, task.QuerySetY, loss);

        var algorithm = new SparseMAMLAlgorithm<double, Matrix<double>, Vector<double>>(Options(model));
        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        for (int d = 0; d < 4; d++)
        {
            Assert.Equal(1.0 + 0.1 * 0.5 * gOut[d] * gIn[d], algorithm.GateLogits[d], 10);
        }
    }

    [Fact]
    public void Sparsity_CountsEveryClosedGateOfALargeHalfModel()
    {
        // 70000 gates, all closed (logit -1 under the step gate): sparsity is 1. Summing the mask in Half overflowed
        // past 65504 to inf, and so did the parameter count, so the ratio was inf / inf = NaN.
        var model = new MockFullModel<Half>(numParameters: 70000);
        var algorithm = new SparseMAMLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new SparseMAMLOptions<Half, Tensor<Half>, Tensor<Half>>(model) { InitialGateLogit = -1.0 });

        Assert.Equal(1.0, (double)(float)algorithm.Sparsity, 6);
    }
}
