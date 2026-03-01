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
/// Integration tests for Bayesian meta-learning extensions (Phase 3).
/// Verifies: finite loss, parameter change, adaptation, and multi-step stability.
/// </summary>
public class BayesianMetaTests
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
    public void PACOH_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new PACOHOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new PACOHAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(7848);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "PACOH loss is NaN");
        Assert.False(double.IsInfinity(loss), "PACOH loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "PACOH params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.PACOH, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaPACOH_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaPACOHOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaPACOHAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(8752);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaPACOH loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaPACOH loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaPACOH params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaPACOH, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void BMAML_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new BMAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new BMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(945);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "BMAML loss is NaN");
        Assert.False(double.IsInfinity(loss), "BMAML loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "BMAML params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.BMAML, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void BayProNet_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new BayProNetOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new BayProNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(5221);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "BayProNet loss is NaN");
        Assert.False(double.IsInfinity(loss), "BayProNet loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "BayProNet params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.BayProNet, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void FlexPACBayes_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new FlexPACBayesOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new FlexPACBayesAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(158);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "FlexPACBayes loss is NaN");
        Assert.False(double.IsInfinity(loss), "FlexPACBayes loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "FlexPACBayes params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.FlexPACBayes, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void BMAML_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new BMAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new BMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 BMAML steps produced finite loss");
    }
}
