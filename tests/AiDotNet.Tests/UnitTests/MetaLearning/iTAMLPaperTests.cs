using System;
using AiDotNet.Data.Structures;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="iTAMLAlgorithm{T, TInput, TOutput}"/> against Rajasegaran et al.,
/// "iTAML: An Incremental Task-Agnostic Meta-learning Approach" (CVPR 2020), Algorithm 1.
/// </summary>
public class iTAMLPaperTests
{
    private static readonly double[] Theta = { 0.3, -0.2, 0.1, 0.05 };

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task(double shift)
        => new()
        {
            SupportSetX = new Matrix<double>(new[,] { { 0.5, -1.0, 0.25 }, { 1.0, 0.5, -0.5 } }),
            SupportSetY = new Vector<double>(new[] { 1.0 + shift, -0.5 }),
            QuerySetX = new Matrix<double>(new[,] { { -0.25, 0.75, 1.0 } }),
            QuerySetY = new Vector<double>(new[] { 0.5 }),
            NumWays = 1, NumShots = 2, NumQueryPerClass = 1,
        };

    /// <summary>One SGD step (r = 1) from <paramref name="start"/> on the task's support set.</summary>
    private static Vector<double> Step(Vector<double> start, MetaLearningTask<double, Matrix<double>, Vector<double>> task)
    {
        var model = new LinearVectorModel(3);
        model.SetParameters(start);
        var g = model.ComputeGradients(task.SupportSetX, task.SupportSetY, new MeanSquaredErrorLoss<double>());
        var p = new Vector<double>(4);
        for (int d = 0; d < 4; d++) p[d] = start[d] - 0.1 * g[d];
        return p;
    }

    [Fact]
    public void Defaults_FollowThePaper()
    {
        var options = new iTAMLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3));
        Assert.Equal(1, options.AdaptationSteps);
        Assert.Equal(1.0, options.EtaDecay);
    }

    [Fact]
    public void MetaUpdate_InterpolatesTowardsTheMeanAdaptedParameters_WithDecayingEta()
    {
        var model = new LinearVectorModel(3);
        model.SetParameters(new Vector<double>(Theta));
        var algorithm = new iTAMLAlgorithm<double, Matrix<double>, Vector<double>>(
            new iTAMLOptions<double, Matrix<double>, Vector<double>>(model)
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                GradientClipThreshold = null,
                InnerLearningRate = 0.1,
                NumMetaIterations = 2,
            });
        var tasks = new[] { Task(0.0), Task(1.0) };
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        var expected = new Vector<double>(Theta);
        foreach (double eta in new[] { 1.0, Math.Exp(-0.5) })
        {
            // Phi = eta * mean_i Phi_i + (1 - eta) * Phi_base, eta = exp(-beta t / T).
            Assert.Equal(eta, algorithm.CurrentEta(), 12);
            var a = Step(expected, tasks[0]);
            var b = Step(expected, tasks[1]);
            var next = new Vector<double>(4);
            for (int d = 0; d < 4; d++) next[d] = eta * 0.5 * (a[d] + b[d]) + (1 - eta) * expected[d];
            expected = next;

            algorithm.MetaTrain(batch);
            var actual = model.GetParameters();
            for (int d = 0; d < 4; d++) Assert.Equal(expected[d], actual[d], 10);
        }
    }
}
