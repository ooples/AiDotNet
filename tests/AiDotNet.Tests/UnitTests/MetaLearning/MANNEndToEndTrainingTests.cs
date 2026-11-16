using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// MANN trained end to end as Santoro et al. 2016 train it: the query cross-entropy is differentiated through the
/// LRUA writes and reads to the controller, the output layer over [controller output; read vector] and the write
/// gate logit alpha (eq. 7's sigmoid(alpha)).
/// </summary>
public class MANNEndToEndTrainingTests
{
    private const int Features = 3;
    private const int Width = 4;
    private const int Classes = 2;

    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTask(int seed, int shots, int queries)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var centres = Enumerable.Range(0, Classes)
            .Select(_ => Enumerable.Range(0, Features).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray()).ToArray();

        (Matrix<double> X, Tensor<double> Y) Rows(int perClass)
        {
            int rows = perClass * Classes;
            var x = new Matrix<double>(rows, Features);
            var y = new Tensor<double>(new[] { rows });
            for (int r = 0; r < rows; r++)
            {
                int label = r % Classes;
                for (int f = 0; f < Features; f++) x[r, f] = centres[label][f] + 0.3 * (rng.NextDouble() - 0.5);
                y[r] = label;
            }

            return (x, y);
        }

        var support = Rows(shots);
        var query = Rows(queries);
        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support.X, SupportSetY = support.Y, QuerySetX = query.X, QuerySetY = query.Y,
            NumWays = Classes, NumShots = shots, NumQueryPerClass = queries, Name = $"mann-{seed}",
        };
    }

    private static MANNAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<MANNOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new MANNOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            MemorySize = 6,
            MemoryKeySize = Width,
            MemoryValueSize = Classes,
            NumClasses = Classes,
            UsageDecay = 0.9,
            WriteGateLogit = 0.3,
            MemoryRegularization = 0.05,
            OuterLearningRate = 0.2,
            GradientClipThreshold = null,
            RandomSeed = 5,
        };
        configure?.Invoke(options);
        return new MANNAlgorithm<double, Matrix<double>, Tensor<double>>(options);
    }

    private static double CentralDifference(Func<double> loss, Action<double> shift, double h)
    {
        shift(h);
        double plus = loss();
        shift(-2 * h);
        double minus = loss();
        shift(h);
        return (plus - minus) / (2 * h);
    }

    private static void CheckVector(Func<double> loss, double[] analytic, Func<double[]> get, Action<double[]> set,
        string name, List<string> failures)
    {
        int reported = 0;
        for (int i = 0; i < analytic.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(loss, h =>
            {
                var shifted = get();
                shifted[index] += h;
                set(shifted);
            }, 1e-6);
            if (Math.Abs(analytic[i] - numeric) > 1e-6 + 1e-4 * Math.Abs(numeric) && reported++ < 3)
            {
                failures.Add($"dL/d{name}[{i}]: analytic {analytic[i]:G10} vs central difference {numeric:G10}.");
            }
        }
    }

    [Fact]
    public void Gradients_ReachTheWriteGateTheOutputLayerAndTheController_ThroughTheMemory()
    {
        var learner = CreateLearner();
        var task = CreateTask(11, shots: 2, queries: 2);

        // One step first, so the memory carries earlier writes and the last read weights are non-zero.
        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(3, 2, 2) }));

        var gradient = learner.EpisodeGradientForTesting(task);
        Assert.Equal(gradient.Loss, learner.EpisodeLossForTesting(task), 10);

        Assert.True(Math.Abs(gradient.GateLogit[0]) > 1e-8, $"The write gate's gradient is {gradient.GateLogit[0]:G6}.");
        Assert.True(gradient.OutputWeights.ToArray().Any(g => Math.Abs(g) > 1e-8), "The output layer got no gradient.");
        Assert.True(gradient.Body.ToArray().Any(g => Math.Abs(g) > 1e-8), "The controller got no gradient.");

        // The read-vector rows of the output layer (after the controller's Width rows) carry gradient: the reads
        // reach the loss.
        var readRows = gradient.OutputWeights.ToArray().Skip(Width * Classes).ToArray();
        Assert.True(readRows.Any(g => Math.Abs(g) > 1e-8), "The read vector's output weights got no gradient.");

        double Loss() => learner.EpisodeLossForTesting(task);
        var controller = (LinearEmbeddingModel)learner.BaseModel;
        var failures = new List<string>();
        CheckVector(Loss, gradient.GateLogit.ToArray(), () => learner.GateLogitForTesting.ToArray(),
            v => learner.GateLogitForTesting = new Vector<double>(v), "alpha", failures);
        CheckVector(Loss, gradient.OutputWeights.ToArray(), () => learner.OutputWeightsForTesting.ToArray(),
            v => learner.OutputWeightsForTesting = new Vector<double>(v), "output weights", failures);
        CheckVector(Loss, gradient.OutputBias.ToArray(), () => learner.OutputBiasForTesting.ToArray(),
            v => learner.OutputBiasForTesting = new Vector<double>(v), "output bias", failures);
        CheckVector(Loss, gradient.Body.ToArray(), () => controller.GetParameters().ToArray(),
            v => controller.SetParameters(new Vector<double>(v)), "controller", failures);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void MetaTraining_UpdatesTheWriteGate()
    {
        var learner = CreateLearner();
        double before = learner.GateLogitForTesting[0];
        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(3, 2, 2) }));
        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(4, 2, 2) }));
        Assert.NotEqual(before, learner.GateLogitForTesting[0]);
    }

    [Fact]
    public void OneShotTask_MetaTrainingLowersItsQueryLoss()
    {
        // Every episode starts from an empty memory, so the loss is a function of the learned parameters alone.
        var learner = CreateLearner(o =>
        {
            o.ClearMemoryBetweenTasks = true;
            o.MemoryRetentionRatio = 0.0;
            o.MemoryRegularization = 0.0;
        });
        var task = CreateTask(21, shots: 1, queries: 2);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        double initial = learner.EpisodeLossForTesting(task);
        for (int i = 0; i < 40; i++) learner.MetaTrain(batch);
        double trained = learner.EpisodeLossForTesting(task);

        Assert.True(trained < 0.8 * initial, $"Query loss went from {initial:G6} to {trained:G6}.");
    }

    [Fact]
    public void SetParameters_OnAFreshInstance_RestoresTheTrainedOutputLayerAndWriteGate()
    {
        // Every episode starts from an empty memory, so the query predictions depend on the parameters alone and a
        // fresh instance holding the restored parameters must reproduce them.
        Action<MANNOptions<double, Matrix<double>, Tensor<double>>> configure = o =>
        {
            o.ClearMemoryBetweenTasks = true;
            o.MemoryRetentionRatio = 0.0;
        };
        var trained = CreateLearner(configure);
        for (int i = 0; i < 5; i++)
        {
            trained.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(30 + i, 2, 2) }));
        }

        var probe = CreateTask(41, shots: 2, queries: 2);
        var untrained = CreateLearner(configure);
        Assert.NotEqual(untrained.EpisodeLossForTesting(probe), trained.EpisodeLossForTesting(probe));

        var parameters = trained.GetParameters();
        int controllerCount = trained.GetMetaModel().GetParameters().Length;
        int outputLayerCount = (Width + Classes) * Classes + Classes;
        Assert.Equal(controllerCount + outputLayerCount + 1, parameters.Length);
        Assert.Equal((long)parameters.Length, trained.ParameterCount);

        var restored = CreateLearner(configure);
        restored.SetParameters(parameters);

        Assert.Equal(parameters.ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(trained.ParameterCount, restored.ParameterCount);
        Assert.Equal(trained.OutputWeightsForTesting.ToArray(), restored.OutputWeightsForTesting.ToArray());
        Assert.Equal(trained.OutputBiasForTesting.ToArray(), restored.OutputBiasForTesting.ToArray());
        Assert.Equal(trained.GateLogitForTesting.ToArray(), restored.GateLogitForTesting.ToArray());
        Assert.Equal(trained.EpisodeLossForTesting(probe), restored.EpisodeLossForTesting(probe), 12);
    }
}
