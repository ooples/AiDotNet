using System;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearningAdditionalAlgorithmsIntegrationTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateVectorTask(
        int seed,
        int supportRows,
        int queryRows,
        int featureCount,
        int numWays)
    {
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
            supportY[i] = i % Math.Max(1, numWays);
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
            queryY[i] = i % Math.Max(1, numWays);
        }

        int numShots = Math.Max(1, supportRows / Math.Max(1, numWays));
        int numQueryPerClass = Math.Max(1, queryRows / Math.Max(1, numWays));

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

    private static IMetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTensorLabelTask(
        int seed,
        int supportRows,
        int queryRows,
        int featureCount,
        int numWays)
    {
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
            supportY[new[] { i }] = i % Math.Max(1, numWays);
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
            queryY[new[] { i }] = i % Math.Max(1, numWays);
        }

        int numShots = Math.Max(1, supportRows / Math.Max(1, numWays));
        int numQueryPerClass = Math.Max(1, queryRows / Math.Max(1, numWays));

        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass,
            Name = $"tensor-label-task-{seed}"
        };
    }

    private static IMetaLearningTask<double, Matrix<double>, Tensor<double>> CreateFixedTensorOutputTask(
        int seed,
        int inputRows,
        int featureCount,
        int numClasses)
    {
        var supportX = new Matrix<double>(inputRows, featureCount);
        var queryX = new Matrix<double>(inputRows, featureCount);
        var supportY = new Tensor<double>(new[] { numClasses });
        var queryY = new Tensor<double>(new[] { numClasses });

        var random = new Random(seed);

        for (int i = 0; i < inputRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
                queryX[i, j] = random.NextDouble() - 0.5;
            }
        }

        int supportLabel = 0;
        int queryLabel = numClasses > 1 ? 1 : 0;
        for (int i = 0; i < numClasses; i++)
        {
            supportY[new[] { i }] = i == supportLabel ? 1.0 : 0.0;
            queryY[new[] { i }] = i == queryLabel ? 1.0 : 0.0;
        }

        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = Math.Max(1, numClasses),
            NumShots = 1,
            NumQueryPerClass = 1,
            Name = $"fixed-tensor-task-{seed}"
        };
    }

    [Fact]
    public void ANIL_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new ANILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            FeatureDimension = 2,
            NumClasses = 2,
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            UseHeadBias = false
        };

        var algorithm = new ANILAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(1, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.ANIL, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, predictions.Length);
    }

    [Fact]
    public void BOIL_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new BOILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            FeatureDimension = 2,
            NumClasses = 2,
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            BodyAdaptationFraction = 1.0,
            UseLayerwiseLearningRates = false
        };

        var algorithm = new BOILAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(2, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.BOIL, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, predictions.Length);
    }

    [Fact]
    public void CNAP_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new CNAPOptions<double, Matrix<double>, Vector<double>>(model)
        {
            RepresentationDimension = 4,
            HiddenDimension = 4,
            OuterLearningRate = 0.01,
            NormalizeFastWeights = false,
            FastWeightScale = 0.5,
            FastWeightMode = FastWeightApplicationMode.Additive
        };

        var algorithm = new CNAPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(3, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.CNAP, algorithm.AlgorithmType);
        Assert.Equal(task.QuerySetX.Rows, predictions.Length);
    }

    [Fact]
    public void GNNMeta_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new GNNMetaOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NodeEmbeddingDimension = 4,
            GNNHiddenDimension = 4,
            NumMessagePassingLayers = 1,
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            LearnEdgeWeights = false,
            UseFullyConnectedGraph = true
        };

        var algorithm = new GNNMetaAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var taskA = CreateVectorTask(4, 2, 2, 2, 2);
        var taskB = CreateVectorTask(5, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(taskA);
        var predictions = adapted.Predict(taskA.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.GNNMeta, algorithm.AlgorithmType);
        Assert.Equal(taskA.QuerySetX.Rows, predictions.Length);
    }

    [Fact]
    public void iMAML_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            UseFirstOrder = false,
            ConjugateGradientIterations = 2,
            ConjugateGradientTolerance = 1e-6
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(6, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.iMAML, algorithm.AlgorithmType);
        Assert.Equal(task.QuerySetX.Rows, predictions.Length);
    }

    [Fact]
    public void LEO_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new LEOOptions<double, Matrix<double>, Vector<double>>(model)
        {
            EmbeddingDimension = 2,
            LatentDimension = 2,
            HiddenDimension = 2,
            NumClasses = 2,
            AdaptationSteps = 1,
            InnerLearningRate = 0.1,
            OuterLearningRate = 0.01,
            KLWeight = 0.0,
            UseRelationEncoder = false
        };

        var algorithm = new LEOAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(7, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.LEO, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, predictions.Length);
    }

    [Fact]
    public void MetaOptNet_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaOptNetOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            SolverType = ConvexSolverType.RidgeRegression,
            RegularizationStrength = 0.1,
            OuterLearningRate = 0.01,
            MaxSolverIterations = 3,
            UseLearnedTemperature = false
        };

        var algorithm = new MetaOptNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(8, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.MetaOptNet, algorithm.AlgorithmType);
        Assert.Equal(task.QuerySetX.Rows, predictions.Length);
    }

    [Fact]
    public void SEAL_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new SEALOptions<double, Matrix<double>, Vector<double>>(model)
        {
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            UseAdaptiveInnerLR = true,
            AdaptiveLearningRateMode = SEALAdaptiveLearningRateMode.RunningMean,
            EntropyCoefficient = 0.01,
            Temperature = 1.2
        };

        var algorithm = new SEALAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(9, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.SEAL, algorithm.AlgorithmType);
        Assert.Equal(task.QuerySetX.Rows, predictions.Length);
    }

    [Fact]
    public void MANN_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(2);
        var options = new MANNOptions<double, Matrix<double>, Vector<double>>(model)
        {
            MemorySize = 4,
            MemoryKeySize = 2,
            MemoryValueSize = 2,
            NumClasses = 2,
            NumReadHeads = 1,
            NumWriteHeads = 1,
            OuterLearningRate = 0.01,
            ClearMemoryBetweenTasks = true,
            UseOutputSoftmax = true
        };

        var algorithm = new MANNAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(10, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.MANN, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, predictions.Length);
    }

    [Fact]
    public void NTM_MetaTrainAndAdapt_Run()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new NTMOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            MemorySize = 4,
            MemoryWidth = 2,
            NumReadHeads = 1,
            NumClasses = 2,
            ControllerHiddenSize = 4,
            OuterLearningRate = 0.01,
            ControllerType = NTMControllerType.MLP,
            InitializeMemory = false
        };

        var algorithm = new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateFixedTensorOutputTask(11, 1, 2, options.NumClasses);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.NTM, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, predictions.Shape[0]);
    }

    [Fact]
    public void RelationNetwork_MetaTrainAndAdapt_Run()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new RelationNetworkOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 2,
            RelationHiddenDimension = 4,
            OuterLearningRate = 0.01
        };

        var algorithm = new RelationNetworkAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(12, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.RelationNetwork, algorithm.AlgorithmType);
        Assert.True(predictions.Shape.Length > 0);
    }

    [Fact]
    public void TADAM_MetaTrainAndAdapt_Run()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new TADAMOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            TaskEmbeddingDimension = 2,
            OuterLearningRate = 0.01,
            UseTaskConditioning = true,
            UseMetricScaling = true,
            UseAuxiliaryCoTraining = true,
            AuxiliaryLossWeight = 0.1
        };

        var algorithm = new TADAMAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(13, 2, 2, 2, 2);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(MetaLearningAlgorithmType.TADAM, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, predictions.Shape[0]);
    }

    [Fact]
    public void MANN_InvalidOptions_Throws()
    {
        var model = new LinearVectorModel(2);
        var options = new MANNOptions<double, Matrix<double>, Vector<double>>(model)
        {
            MemorySize = 0
        };

        Assert.Throws<ArgumentException>(() => new MANNAlgorithm<double, Matrix<double>, Vector<double>>(options));
    }

    [Fact]
    public void NTM_InvalidOptions_Throws()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new NTMOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            MemorySize = 0
        };

        Assert.Throws<ArgumentException>(() => new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(options));
    }

    [Fact]
    public void RelationNetwork_InvalidOptions_Throws()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new RelationNetworkOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 0
        };

        Assert.Throws<ArgumentException>(() => new RelationNetworkAlgorithm<double, Matrix<double>, Tensor<double>>(options));
    }

    [Fact]
    public void TADAM_InvalidOptions_Throws()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new TADAMOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 0
        };

        Assert.Throws<ArgumentException>(() => new TADAMAlgorithm<double, Matrix<double>, Tensor<double>>(options));
    }
}
