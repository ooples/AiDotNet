using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Integration tests for Meta-RL algorithms (Phase 5).
/// Verifies: finite loss, parameter change, adaptation, and multi-step stability.
/// </summary>
public class MetaRLTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
    {
        var supportX = new Matrix<double>(4, 3);
        var supportY = new Vector<double>(4);
        var queryX = new Matrix<double>(4, 3);
        var queryY = new Vector<double>(4);
        var rng = new Random(seed);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                supportX[i, j] = rng.NextDouble() - 0.5;
                queryX[i, j] = rng.NextDouble() - 0.5;
            }
            supportY[i] = i % 2;
            queryY[i] = i % 2;
        }
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX, SupportSetY = supportY,
            QuerySetX = queryX, QuerySetY = queryY,
            NumWays = 2, NumShots = 2, NumQueryPerClass = 2
        };
    }

    private static bool ParamsChanged(Vector<double> before, Vector<double> after)
    {
        if (before.Length != after.Length) return true;
        for (int i = 0; i < before.Length; i++)
            if (Math.Abs(before[i] - after[i]) > 1e-15) return true;
        return false;
    }

    [Fact]
    public void PEARL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new PEARLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new PEARLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(7662);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "PEARL loss is NaN");
        Assert.False(double.IsInfinity(loss), "PEARL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "PEARL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.PEARL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void DREAM_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new DREAMOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new DREAMAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(5940);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "DREAM loss is NaN");
        Assert.False(double.IsInfinity(loss), "DREAM loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "DREAM params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.DREAM, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void DiscoRL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new DiscoRLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new DiscoRLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6475);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "DiscoRL loss is NaN");
        Assert.False(double.IsInfinity(loss), "DiscoRL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "DiscoRL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.DiscoRL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void InContextRL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new InContextRLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new InContextRLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6899);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "InContextRL loss is NaN");
        Assert.False(double.IsInfinity(loss), "InContextRL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "InContextRL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.InContextRL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void HyperNetMetaRL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new HyperNetMetaRLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new HyperNetMetaRLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(5022);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "HyperNetMetaRL loss is NaN");
        Assert.False(double.IsInfinity(loss), "HyperNetMetaRL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "HyperNetMetaRL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.HyperNetMetaRL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void ContextMetaRL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new ContextMetaRLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ContextMetaRLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(4968);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ContextMetaRL loss is NaN");
        Assert.False(double.IsInfinity(loss), "ContextMetaRL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ContextMetaRL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ContextMetaRL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void PEARL_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new PEARLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new PEARLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 PEARL steps produced finite loss");
    }
}
