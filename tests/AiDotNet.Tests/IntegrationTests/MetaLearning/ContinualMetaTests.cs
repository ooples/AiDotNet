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
/// Integration tests for Continual + Online Meta-Learning algorithms (Phase 6).
/// Verifies: finite loss, parameter change, adaptation, and multi-step stability.
/// </summary>
public class ContinualMetaTests
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
    public void ACL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new ACLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ACLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(2820);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ACL loss is NaN");
        Assert.False(double.IsInfinity(loss), "ACL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ACL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ACL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void iTAML_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new iTAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new iTAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(881);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "iTAML loss is NaN");
        Assert.False(double.IsInfinity(loss), "iTAML loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "iTAML params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.iTAML, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaContinualAL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaContinualALOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaContinualALAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6749);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaContinualAL loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaContinualAL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaContinualAL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaContinualAL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MePo_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MePoOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MePoAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(876);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MePo loss is NaN");
        Assert.False(double.IsInfinity(loss), "MePo loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MePo params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MePo, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void OML_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new OMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new OMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(9011);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "OML loss is NaN");
        Assert.False(double.IsInfinity(loss), "OML loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "OML params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.OML, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MOCA_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MOCAOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MOCAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1626);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MOCA loss is NaN");
        Assert.False(double.IsInfinity(loss), "MOCA loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MOCA params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MOCA, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void ACL_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new ACLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new ACLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 ACL steps produced finite loss");
    }
}
