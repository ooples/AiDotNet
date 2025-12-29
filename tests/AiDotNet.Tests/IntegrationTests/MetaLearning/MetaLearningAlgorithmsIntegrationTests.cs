using System;
using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearningAlgorithmsIntegrationTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateVectorTask(int seed)
    {
        const int numWays = 2;
        const int numShots = 2;
        const int numQueryPerClass = 2;
        const int featureCount = 3;

        int supportRows = numWays * numShots;
        int queryRows = numWays * numQueryPerClass;

        var supportX = new Matrix<double>(supportRows, featureCount);
        var supportY = new Vector<double>(supportRows);
        var queryX = new Matrix<double>(queryRows, featureCount);
        var queryY = new Vector<double>(queryRows);

        var random = new Random(seed);

        for (int i = 0; i < supportRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
            }
            supportY[i] = i % numWays;
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
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
            Name = $"vector-task-{seed}"
        };
    }

    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTensorTask(int seed)
    {
        const int numWays = 2;
        const int numShots = 1;
        const int numQueryPerClass = 2;
        const int featureCount = 3;

        int supportRows = numWays * numShots;
        int queryRows = numWays * numQueryPerClass;

        var supportX = new Matrix<double>(supportRows, featureCount);
        var queryX = new Matrix<double>(queryRows, featureCount);
        var supportY = new Tensor<double>(new[] { supportRows });
        var queryY = new Tensor<double>(new[] { queryRows });

        var random = new Random(seed);

        for (int i = 0; i < supportRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
            }
            supportY[new[] { i }] = i % numWays;
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
            queryY[new[] { i }] = i % numWays;
        }

        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass,
            Name = $"tensor-task-{seed}"
        };
    }

    [Fact]
    public void MAML_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2
        };

        var algorithm = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var initial = model.GetParameters().ToArray();

        var tasks = new[] { CreateVectorTask(1), CreateVectorTask(2) };
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        var loss = algorithm.MetaTrain(batch);

        var updated = model.GetParameters().ToArray();

        Assert.False(double.IsNaN(loss));
        bool changed = false;
        for (int i = 0; i < updated.Length; i++)
        {
            if (Math.Abs(updated[i] - initial[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed);
        Assert.Equal(MetaLearningAlgorithmType.MAML, algorithm.AlgorithmType);
    }

    [Fact]
    public void Reptile_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new ReptileOptions<double, Matrix<double>, Vector<double>>(model)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.02,
            AdaptationSteps = 2,
            InnerBatches = 1,
            Interpolation = 0.5
        };

        var algorithm = new ReptileAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var initial = model.GetParameters().ToArray();

        var tasks = new[] { CreateVectorTask(3) };
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        var loss = algorithm.MetaTrain(batch);

        var updated = model.GetParameters().ToArray();

        Assert.False(double.IsNaN(loss));
        bool changed = false;
        for (int i = 0; i < updated.Length; i++)
        {
            if (Math.Abs(updated[i] - initial[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed);
        Assert.Equal(MetaLearningAlgorithmType.Reptile, algorithm.AlgorithmType);
    }

    [Fact]
    public void MetaSGD_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(model)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            InnerSteps = 2,
            MetaBatchSize = 1,
            NumMetaIterations = 2,
            UseWarmStart = false
        };

        var algorithm = new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(4);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.MetaSGD, algorithm.AlgorithmType);
    }

    [Fact]
    public void ProtoNets_MetaTrainAndAdapt_Run()
    {
        var model = new TensorEmbeddingModel(3, 2);
        var options = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            OuterLearningRate = 0.01,
            AdaptationSteps = 1
        };

        var algorithm = new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorTask(5);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.ProtoNets, algorithm.AlgorithmType);
        Assert.Equal(1, predictions.Shape.Length);
        Assert.Equal(task.NumWays, predictions.Shape[0]);
    }

    [Fact]
    public void MatchingNetworks_MetaTrainAndAdapt_Run()
    {
        var model = new TensorEmbeddingModel(3, 2);
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumClasses = 2,
            AttentionFunction = MatchingNetworksAttentionFunction.DotProduct
        };

        var algorithm = new MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorTask(6);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.MatchingNetworks, algorithm.AlgorithmType);
        Assert.Equal(2, predictions.Shape.Length);
        Assert.Equal(task.QuerySetX.Rows, predictions.Shape[0]);
        Assert.Equal(options.NumClasses, predictions.Shape[1]);
    }
}
