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
/// Integration tests for Cross-Domain Few-Shot algorithms (Phase 4).
/// Verifies: finite loss, parameter change, adaptation, and multi-step stability.
/// </summary>
public class CrossDomainMetaTests
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
    public void MetaFDMixup_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaFDMixupOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaFDMixupAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(8822);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaFDMixup loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaFDMixup loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaFDMixup params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaFDMixup, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void FreqPrior_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new FreqPriorOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new FreqPriorAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(456);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "FreqPrior loss is NaN");
        Assert.False(double.IsInfinity(loss), "FreqPrior loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "FreqPrior params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.FreqPrior, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaCollaborative_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaCollaborativeOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaCollaborativeAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1566);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaCollaborative loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaCollaborative loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaCollaborative params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaCollaborative, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void SDCL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new SDCLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new SDCLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(5677);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "SDCL loss is NaN");
        Assert.False(double.IsInfinity(loss), "SDCL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "SDCL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.SDCL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void FreqPrompt_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new FreqPromptOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new FreqPromptAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6792);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "FreqPrompt loss is NaN");
        Assert.False(double.IsInfinity(loss), "FreqPrompt loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "FreqPrompt params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.FreqPrompt, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void OpenMAMLPlus_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new OpenMAMLPlusOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new OpenMAMLPlusAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(8850);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "OpenMAMLPlus loss is NaN");
        Assert.False(double.IsInfinity(loss), "OpenMAMLPlus loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "OpenMAMLPlus params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.OpenMAMLPlus, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void FreqPrior_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new FreqPriorOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new FreqPriorAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 FreqPrior steps produced finite loss");
    }
}
