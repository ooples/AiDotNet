using System;
using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Reference tests that pin meta-learning algorithms to the update rules and defaults of their papers.
/// </summary>
public class MetaLearningPaperFidelityTests
{
    private const double Tolerance = 1e-10;

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateRegressionTask(int seed)
    {
        const int rows = 4;
        const int featureCount = 3;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);

        var supportX = new Matrix<double>(rows, featureCount);
        var supportY = new Vector<double>(rows);
        var queryX = new Matrix<double>(rows, featureCount);
        var queryY = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
                queryX[i, j] = random.NextDouble() - 0.5;
            }

            supportY[i] = random.NextDouble() * 2.0 - 1.0;
            queryY[i] = random.NextDouble() * 2.0 - 1.0;
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = 2,
            NumShots = 2,
            NumQueryPerClass = 2,
        };
    }

    /// <summary>k plain SGD steps on the support set from <paramref name="start"/>: W = SGD(L_tau, phi, k).</summary>
    private static Vector<double> AdaptBySgd(
        Vector<double> start, MetaLearningTask<double, Matrix<double>, Vector<double>> task, double learningRate, int steps)
    {
        var model = new LinearVectorModel(3);
        var loss = new MeanSquaredErrorLoss<double>();
        var parameters = start.Clone();
        for (int step = 0; step < steps; step++)
        {
            model.SetParameters(parameters);
            var gradient = model.ComputeGradients(task.SupportSetX, task.SupportSetY, loss);
            var next = new Vector<double>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                next[i] = parameters[i] - learningRate * gradient[i];
            }

            parameters = next;
        }

        return parameters;
    }

    private static ReptileOptions<double, Matrix<double>, Vector<double>> ReptileOptionsFor(
        LinearVectorModel model, bool serial, double? finalOuterLearningRate)
    {
        return new ReptileOptions<double, Matrix<double>, Vector<double>>(model)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.1,
            OuterLearningRate = 1.0,
            FinalOuterLearningRate = finalOuterLearningRate,
            AdaptationSteps = 3,
            InnerBatches = 1,
            Interpolation = 0.5,
            NumMetaIterations = 4,
            UseSerialUpdate = serial,
        };
    }

    /// <summary>Exposes the meta-iteration counter so the annealed outer step can be checked mid-schedule.</summary>
    private sealed class IterationReptile : ReptileAlgorithm<double, Matrix<double>, Vector<double>>
    {
        public IterationReptile(ReptileOptions<double, Matrix<double>, Vector<double>> options)
            : base(options)
        {
        }

        public void SetIteration(int iteration) => _currentIteration = iteration;
    }

    [Fact]
    public void Reptile_SerialUpdate_StepsOncePerTaskFromTheUpdatedInitialization()
    {
        // Nichol et al. (2018), Algorithm 1: for each task, W = SGD(L_tau, phi, k); phi <- phi + epsilon (W - phi).
        var taskA = CreateRegressionTask(3);
        var taskB = CreateRegressionTask(5);
        var model = new LinearVectorModel(3);
        var phi0 = model.GetParameters();
        var algorithm = new ReptileAlgorithm<double, Matrix<double>, Vector<double>>(
            ReptileOptionsFor(model, serial: true, finalOuterLearningRate: null));

        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB }));

        const double step = 1.0 * 0.5; // epsilon = OuterLearningRate * Interpolation
        var wA = AdaptBySgd(phi0, taskA, 0.1, 3);
        var phi1 = new Vector<double>(phi0.Length);
        for (int i = 0; i < phi0.Length; i++)
        {
            phi1[i] = phi0[i] + step * (wA[i] - phi0[i]);
        }

        var wB = AdaptBySgd(phi1, taskB, 0.1, 3);
        var actual = model.GetParameters();
        for (int i = 0; i < phi0.Length; i++)
        {
            Assert.Equal(phi1[i] + step * (wB[i] - phi1[i]), actual[i], Tolerance);
        }
    }

    [Fact]
    public void Reptile_BatchedUpdate_MovesTowardTheMeanAdaptedParameters()
    {
        // Nichol et al. (2018), Algorithm 2: phi <- phi + epsilon (1/n) sum_i (W_i - phi), all W_i from the same phi.
        var taskA = CreateRegressionTask(3);
        var taskB = CreateRegressionTask(5);
        var model = new LinearVectorModel(3);
        var phi0 = model.GetParameters();
        var algorithm = new ReptileAlgorithm<double, Matrix<double>, Vector<double>>(
            ReptileOptionsFor(model, serial: false, finalOuterLearningRate: null));

        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB }));

        var wA = AdaptBySgd(phi0, taskA, 0.1, 3);
        var wB = AdaptBySgd(phi0, taskB, 0.1, 3);
        var actual = model.GetParameters();
        for (int i = 0; i < phi0.Length; i++)
        {
            double expected = phi0[i] + 0.5 * (0.5 * (wA[i] - phi0[i]) + 0.5 * (wB[i] - phi0[i]));
            Assert.Equal(expected, actual[i], Tolerance);
        }
    }

    [Fact]
    public void Reptile_OuterStepSize_AnnealsLinearlyToTheFinalValue()
    {
        // Nichol et al. (2018), Appendix A: the outer step size is annealed linearly from 1 to 0, so at
        // meta-iteration 1 of 4 it is 0.75 (then scaled by Interpolation = 0.5).
        var task = CreateRegressionTask(7);
        var model = new LinearVectorModel(3);
        var phi0 = model.GetParameters();
        Assert.Equal(0.0, new ReptileOptions<double, Matrix<double>, Vector<double>>(model).FinalOuterLearningRate);
        var algorithm = new IterationReptile(ReptileOptionsFor(model, serial: false, finalOuterLearningRate: 0.0));
        algorithm.SetIteration(1);

        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        var w = AdaptBySgd(phi0, task, 0.1, 3);
        var actual = model.GetParameters();
        for (int i = 0; i < phi0.Length; i++)
        {
            Assert.Equal(phi0[i] + 0.75 * 0.5 * (w[i] - phi0[i]), actual[i], Tolerance);
        }
    }
}
