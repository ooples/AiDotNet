using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearningFailurePathIntegrationTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateVectorTask(
        int numWays,
        int numShots,
        int numQueryPerClass)
    {
        int supportRows = numWays * numShots;
        int queryRows = numWays * numQueryPerClass;

        var supportX = new Matrix<double>(supportRows, 2);
        var supportY = new Vector<double>(supportRows);
        var queryX = new Matrix<double>(queryRows, 2);
        var queryY = new Vector<double>(queryRows);

        for (int i = 0; i < supportRows; i++)
        {
            supportX[i, 0] = i * 0.1;
            supportX[i, 1] = i * 0.2;
            supportY[i] = i % numWays;
        }

        for (int i = 0; i < queryRows; i++)
        {
            queryX[i, 0] = i * 0.3;
            queryX[i, 1] = i * 0.4;
            queryY[i] = i % numWays;
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass,
            Name = $"edge-task-{numWays}-{numShots}-{numQueryPerClass}"
        };
    }

    [Fact]
    public void TaskBatch_Constructor_NullTasks_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void TaskBatch_Constructor_EmptyTasks_Throws()
    {
        var tasks = Array.Empty<IMetaLearningTask<double, Matrix<double>, Vector<double>>>();

        Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(tasks));
    }

    [Fact]
    public void TaskBatch_Constructor_MismatchedConfiguration_Throws()
    {
        var taskA = CreateVectorTask(2, 1, 1);
        var taskB = CreateVectorTask(3, 1, 1);
        var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { taskA, taskB };

        Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(tasks));
    }

    [Fact]
    public void TaskBatch_GetRange_InvalidIndices_Throws()
    {
        var task = CreateVectorTask(2, 1, 1);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { task, task });

        Assert.Throws<ArgumentOutOfRangeException>(() => batch.GetRange(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => batch.GetRange(0, 3));
    }

    [Fact]
    public void TaskBatch_Split_InvalidCount_Throws()
    {
        var task = CreateVectorTask(2, 1, 1);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { task, task });

        Assert.Throws<ArgumentOutOfRangeException>(() => batch.Split(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => batch.Split(3));
    }

    [Fact]
    public void MatchingNetworks_InvalidOptions_Throws()
    {
        var model = new TensorEmbeddingModel(3, 2);
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 2,
            Temperature = 0
        };

        Assert.Throws<ArgumentException>(() =>
            new MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>>(options));
    }

    [Fact]
    public void ProtoNets_InvalidOptions_Throws()
    {
        var model = new TensorEmbeddingModel(3, 2);
        var options = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            Temperature = 0
        };

        Assert.Throws<ArgumentException>(() =>
            new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(options));
    }

    [Fact]
    public void MetaSGD_InvalidOptions_Throws()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(model)
        {
            MinLearningRate = 0
        };

        Assert.Throws<ArgumentException>(() =>
            new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(options));
    }
}
