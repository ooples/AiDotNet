using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Modules;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearningCoverageIntegrationTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateVectorTask(
        int seed,
        int supportRows = 2,
        int queryRows = 2,
        int featureCount = 2,
        int numWays = 2)
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
        int supportRows = 2,
        int queryRows = 2,
        int featureCount = 2,
        int numWays = 2)
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

        return new TestMetaLearningTask<double, Matrix<double>, Tensor<double>>(
            supportX,
            supportY,
            queryX,
            queryY,
            numWays,
            numShots,
            numQueryPerClass,
            $"tensor-task-{seed}");
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

        return new TestMetaLearningTask<double, Matrix<double>, Tensor<double>>(
            supportX,
            supportY,
            queryX,
            queryY,
            Math.Max(1, numClasses),
            1,
            1,
            $"fixed-tensor-task-{seed}");
    }

    private static T InvokePrivate<T>(object instance, Type declaringType, string methodName, params object[] args)
    {
        var method = declaringType.GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return (T)method!.Invoke(instance, args)!;
    }

    [Fact]
    public void TaskBatch_And_TaskWrapper_ExposeMetadata()
    {
        var taskA = CreateVectorTask(1, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var taskB = CreateVectorTask(2, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        taskA.Metadata = new Dictionary<string, object> { ["source"] = "A" };
        taskA.Name = "task-a";

        var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { taskA, taskB };
        var difficulties = new[] { 1.0, 3.0 };
        var similarities = new double[,] { { 1.0, 0.5 }, { 0.5, 1.0 } };

        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            tasks,
            BatchingStrategy.DifficultyBased,
            difficulties,
            similarities,
            CurriculumStage.Medium);

        batch.SetMetadata("note", "meta");

        Assert.Equal(2, batch.BatchSize);
        Assert.Same(taskA, batch[0]);
        Assert.Equal("meta", batch.GetMetadata("note"));
        Assert.Equal(2.0, batch.AverageDifficulty, precision: 6);
        Assert.Equal(1.0, batch.DifficultyVariance, precision: 6);
        Assert.Equal(0.5, batch.AverageTaskSimilarity, precision: 6);
        Assert.True(batch.EstimatedMemoryMB > 0);

        var subset = batch.GetRange(1, 1);
        Assert.Equal(1, subset.BatchSize);
        Assert.Same(taskB, subset[0]);

        var splits = batch.Split(2);
        Assert.Equal(2, splits.Length);
        Assert.Equal(1, splits[0].BatchSize);
        Assert.Equal(1, splits[1].BatchSize);

        var wrapper = new TaskWrapper<double, Matrix<double>, Vector<double>>(taskA);
        Assert.Equal(taskA.Name, wrapper.Name);
        Assert.Same(taskA.Metadata, wrapper.Metadata);
        Assert.Same(taskA.SupportSetX, wrapper.SupportSetX);
        Assert.Same(taskA.SupportSetY, wrapper.SupportSetY);
        Assert.Same(taskA.QuerySetX, wrapper.QuerySetX);
        Assert.Same(taskA.QuerySetY, wrapper.QuerySetY);
    }

    [Fact]
    public void MetaLearnerBase_AccessorsAndFallback_AreCovered()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaLearnerOptionsBase<double>
        {
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.03
        };
        var learner = new TestMetaLearner(model, options);

        Assert.Same(model, learner.GetMetaModel());
        Assert.Same(options, learner.Options);
        Assert.Equal(0.02, learner.InnerLearningRate, precision: 6);
        Assert.Equal(0.03, learner.OuterLearningRate, precision: 6);

        var input = new Matrix<double>(1, 2);
        input[0, 0] = 0.1;
        input[0, 1] = -0.2;
        var expected = new Vector<double>(new[] { 0.4 });

        var gradients = InvokePrivate<Vector<double>>(
            learner,
            typeof(MetaLearnerBase<double, Matrix<double>, Vector<double>>),
            "ComputeGradientsFallback",
            model,
            input,
            expected);

        Assert.Equal(model.ParameterCount, gradients.Length);
    }

    [Fact]
    public void Options_Clone_CoversSpecializedOptions()
    {
        var vectorModel = new LinearVectorModel(2);
        var tensorModel = new TensorEmbeddingModel(2, 2);

        var anilOptions = new ANILOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            FeatureDimension = 2,
            NumClasses = 2
        };
        Assert.IsType<ANILOptions<double, Matrix<double>, Vector<double>>>(anilOptions.Clone());

        var boilOptions = new BOILOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            FeatureDimension = 2,
            NumClasses = 2
        };
        Assert.IsType<BOILOptions<double, Matrix<double>, Vector<double>>>(boilOptions.Clone());

        var cnapOptions = new CNAPOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            RepresentationDimension = 2,
            HiddenDimension = 2
        };
        Assert.IsType<CNAPOptions<double, Matrix<double>, Vector<double>>>(cnapOptions.Clone());

        var gnnOptions = new GNNMetaOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            NodeEmbeddingDimension = 2,
            GNNHiddenDimension = 2,
            NumMessagePassingLayers = 1
        };
        Assert.IsType<GNNMetaOptions<double, Matrix<double>, Vector<double>>>(gnnOptions.Clone());

        var imamlOptions = new iMAMLOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01
        };
        Assert.IsType<iMAMLOptions<double, Matrix<double>, Vector<double>>>(imamlOptions.Clone());

        var leoOptions = new LEOOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            EmbeddingDimension = 2,
            LatentDimension = 2,
            HiddenDimension = 2,
            NumClasses = 2
        };
        Assert.IsType<LEOOptions<double, Matrix<double>, Vector<double>>>(leoOptions.Clone());

        var mamlOptions = new MAMLOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01
        };
        mamlOptions.UseFirstOrderApproximation = true;
        Assert.True(mamlOptions.UseFirstOrderApproximation);
        Assert.IsType<MAMLOptions<double, Matrix<double>, Vector<double>>>(mamlOptions.Clone());

        var mannOptions = new MANNOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            MemorySize = 2,
            MemoryKeySize = 2,
            MemoryValueSize = 2,
            NumClasses = 2
        };
        Assert.IsType<MANNOptions<double, Matrix<double>, Vector<double>>>(mannOptions.Clone());

        var matchingOptions = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(tensorModel)
        {
            NumClasses = 2,
            AttentionFunction = MatchingNetworksAttentionFunction.Cosine
        };
        Assert.IsType<MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>>(matchingOptions.Clone());

        var metaOptNetOptions = new MetaOptNetOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            NumClasses = 2,
            EmbeddingDimension = 2
        };
        Assert.IsType<MetaOptNetOptions<double, Matrix<double>, Vector<double>>>(metaOptNetOptions.Clone());

        var metaSgdOptions = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            UseParameterGrouping = true,
            NumParameterGroups = 3,
            UpdateRuleType = MetaSGDUpdateRuleType.Adam,
            LearnAdamBetas = true
        };
        Assert.IsType<MetaSGDOptions<double, Matrix<double>, Vector<double>>>(metaSgdOptions.Clone());
        Assert.Equal(3, metaSgdOptions.GetEffectiveParameterGroups(10));
        Assert.True(metaSgdOptions.GetTotalMetaParameters(10) > 0);

        var ntmOptions = new NTMOptions<double, Matrix<double>, Tensor<double>>(tensorModel)
        {
            MemorySize = 2,
            MemoryWidth = 2,
            NumReadHeads = 1,
            NumWriteHeads = 1,
            NumClasses = 2
        };
        Assert.IsType<NTMOptions<double, Matrix<double>, Tensor<double>>>(ntmOptions.Clone());

        var protoOptions = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(tensorModel)
        {
            DistanceFunction = ProtoNetsDistanceFunction.Cosine,
            NormalizeFeatures = true
        };
        Assert.IsType<ProtoNetsOptions<double, Matrix<double>, Tensor<double>>>(protoOptions.Clone());

        var relationOptions = new RelationNetworkOptions<double, Matrix<double>, Tensor<double>>(tensorModel)
        {
            NumClasses = 2,
            RelationHiddenDimension = 2
        };
        Assert.IsType<RelationNetworkOptions<double, Matrix<double>, Tensor<double>>>(relationOptions.Clone());

        var reptileOptions = new ReptileOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01
        };
        Assert.IsType<ReptileOptions<double, Matrix<double>, Vector<double>>>(reptileOptions.Clone());

        var sealOptions = new SEALOptions<double, Matrix<double>, Vector<double>>(vectorModel)
        {
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01
        };
        Assert.IsType<SEALOptions<double, Matrix<double>, Vector<double>>>(sealOptions.Clone());

        var tadamOptions = new TADAMOptions<double, Matrix<double>, Tensor<double>>(tensorModel)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            TaskEmbeddingDimension = 2
        };
        Assert.IsType<TADAMOptions<double, Matrix<double>, Tensor<double>>>(tadamOptions.Clone());
    }

    [Fact]
    public void MatchingNetworksModel_UsesCosineSimilarity()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 2,
            AttentionFunction = MatchingNetworksAttentionFunction.Cosine,
            Temperature = 1.2
        };

        var algorithm = new MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(10, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);

        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.Equal(task.QuerySetX.Rows, predictions.Shape[0]);
        Assert.Equal(options.NumClasses, predictions.Shape[1]);

        var modelWrapper = Assert.IsType<MatchingNetworksModel<double, Matrix<double>, Tensor<double>>>(adapted);
        Assert.True(modelWrapper.GetParameters().Length > 0);
        Assert.NotNull(modelWrapper.GetModelMetadata());
    }

    [Fact]
    public void MatchingNetworksModel_UsesEuclideanDistance()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            NumClasses = 2,
            AttentionFunction = MatchingNetworksAttentionFunction.Euclidean,
            Temperature = 1.1
        };

        var algorithm = new MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(11, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);

        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.Equal(task.QuerySetX.Rows, predictions.Shape[0]);
        Assert.Equal(options.NumClasses, predictions.Shape[1]);

        var modelWrapper = Assert.IsType<MatchingNetworksModel<double, Matrix<double>, Tensor<double>>>(adapted);
        Assert.True(modelWrapper.GetParameters().Length > 0);
        Assert.NotNull(modelWrapper.GetModelMetadata());
    }

    [Fact]
    public void ProtoNets_AttentionScaling_And_PrototypicalModel_CosineDistance()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            DistanceFunction = ProtoNetsDistanceFunction.Cosine,
            NormalizeFeatures = true,
            UseAttentionMechanism = true,
            UseAdaptiveClassScaling = true,
            Temperature = 1.2
        };

        var algorithm = new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(12, supportRows: 2, queryRows: 1, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new IMetaLearningTask<double, Matrix<double>, Tensor<double>>[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss));

        var adapted = algorithm.Adapt(task);
        var protoModel = Assert.IsType<PrototypicalModel<double, Matrix<double>, Tensor<double>>>(adapted);
        var predictions = protoModel.Predict(task.QuerySetX);

        Assert.Equal(task.NumWays, predictions.Shape[0]);
        Assert.NotNull(protoModel.GetModelMetadata());
    }

    [Fact]
    public void ProtoNets_Mahalanobis_And_TensorToMatrix_Flattens()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            DistanceFunction = ProtoNetsDistanceFunction.Mahalanobis,
            MahalanobisScaling = 1.5,
            NormalizeFeatures = true
        };

        var algorithm = new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(13, supportRows: 2, queryRows: 1, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new IMetaLearningTask<double, Matrix<double>, Tensor<double>>[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss));

        var adapted = algorithm.Adapt(task);
        var protoModel = Assert.IsType<PrototypicalModel<double, Matrix<double>, Tensor<double>>>(adapted);
        var predictions = protoModel.Predict(task.QuerySetX);

        Assert.Equal(task.NumWays, predictions.Shape[0]);

        var multiTensor = new Tensor<double>(new[] { 2, 2, 2 });
        for (int i = 0; i < multiTensor.Length; i++)
        {
            multiTensor.SetFlat(i, i * 0.1);
        }

        var algoMatrix = InvokePrivate<Matrix<double>>(
            algorithm,
            typeof(ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>),
            "TensorToMatrix",
            multiTensor);
        Assert.Equal(2, algoMatrix.Rows);
        Assert.Equal(4, algoMatrix.Columns);

        var modelMatrix = InvokePrivate<Matrix<double>>(
            protoModel,
            typeof(PrototypicalModel<double, Matrix<double>, Tensor<double>>),
            "TensorToMatrix",
            multiTensor);
        Assert.Equal(2, modelMatrix.Rows);
        Assert.Equal(4, modelMatrix.Columns);
    }

    [Fact]
    public void ANIL_Adapt_WithL2Penalty_And_ModelAccess()
    {
        var model = new LinearVectorModel(2);
        var options = new ANILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            FeatureDimension = 2,
            NumClasses = 2,
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            UseHeadBias = true,
            HeadL2Regularization = 0.1
        };

        var algorithm = new ANILAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(14, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);

        var adapted = algorithm.Adapt(task);
        var anilModel = Assert.IsType<ANILModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = anilModel.Predict(task.QuerySetX);

        Assert.Equal(options.NumClasses, predictions.Length);
        Assert.True(anilModel.GetParameters().Length > 0);
        Assert.NotNull(anilModel.GetModelMetadata());
    }

    [Fact]
    public void BOIL_SecondOrder_And_ModelAccess()
    {
        var model = new LinearVectorModel(2);
        var options = new BOILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            FeatureDimension = 2,
            NumClasses = 2,
            AdaptationSteps = 1,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            UseFirstOrder = false,
            UseLayerwiseLearningRates = true,
            BodyL2Regularization = 0.1,
            ReinitializeBody = true
        };

        var algorithm = new BOILAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(15, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss));

        var adapted = algorithm.Adapt(task);
        var boilModel = Assert.IsType<BOILModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = boilModel.Predict(task.QuerySetX);

        Assert.Equal(options.NumClasses, predictions.Length);
        Assert.True(boilModel.GetParameters().Length > 0);
        Assert.NotNull(boilModel.GetModelMetadata());

        var clone = InvokePrivate<Vector<double>>(
            algorithm,
            typeof(BOILAlgorithm<double, Matrix<double>, Vector<double>>),
            "CloneVector",
            new Vector<double>(new[] { 1.0, 2.0 }));
        Assert.Equal(2, clone.Length);
    }

    [Fact]
    public void LEO_MetaTrain_Accumulates_And_ModelAccess()
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
        var taskA = CreateVectorTask(16, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var taskB = CreateVectorTask(17, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss));

        var adapted = algorithm.Adapt(taskA);
        var leoModel = Assert.IsType<LEOModel<double, Matrix<double>, Vector<double>>>(adapted);
        Assert.True(leoModel.GetParameters().Length > 0);
        Assert.NotNull(leoModel.GetModelMetadata());
    }

    [Fact]
    public void CNAP_NormalizesFastWeights_And_Resizes()
    {
        var model = new LinearVectorModel(2);
        var options = new CNAPOptions<double, Matrix<double>, Vector<double>>(model)
        {
            RepresentationDimension = 4,
            HiddenDimension = 4,
            NormalizeFastWeights = true,
            FastWeightScale = 0.5,
            FastWeightMode = FastWeightApplicationMode.Additive
        };

        var algorithm = new CNAPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(18, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);

        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.Equal(task.QuerySetX.Rows, predictions.Length);

        var resized = InvokePrivate<Vector<double>>(
            algorithm,
            typeof(CNAPAlgorithm<double, Matrix<double>, Vector<double>>),
            "ResizeFastWeights",
            new Vector<double>(new[] { 1.0, 2.0 }),
            5);
        Assert.Equal(5, resized.Length);
    }

    [Fact]
    public void GNNMeta_Attention_And_LearnedSimilarity()
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
            AggregationType = GNNAggregationType.Attention,
            SimilarityMetric = TaskSimilarityMetric.Learned,
            LearnEdgeWeights = false,
            UseFullyConnectedGraph = true
        };

        var algorithm = new GNNMetaAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var taskA = CreateVectorTask(19, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var taskB = CreateVectorTask(20, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss));
    }

    [Fact]
    public void MAML_SecondOrder_BuildsAdaptationSteps()
    {
        var model = new SecondOrderMatrixModel(2);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            UseFirstOrderApproximation = false
        };

        var algorithm = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(21, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new IMetaLearningTask<double, Matrix<double>, Vector<double>>[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss));
    }

    [Fact]
    public void MetaSGD_AdaptedModel_OptimizerSetters()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 1,
            InnerSteps = 1,
            MetaBatchSize = 1,
            NumMetaIterations = 1,
            UseWarmStart = true,
            UpdateRuleType = MetaSGDUpdateRuleType.Adam,
            LearnAdamBetas = true,
            LearnMomentum = true,
            LearnDirection = true
        };

        var algorithm = new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(22, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);

        var adapted = algorithm.Adapt(task);
        var metaModel = Assert.IsType<MetaSGDAdaptedModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = metaModel.Predict(task.QuerySetX);

        Assert.Equal(task.QuerySetX.Rows, predictions.Length);

        metaModel.Train(task.SupportSetX, task.SupportSetY);
        var parameters = metaModel.GetParameters();
        metaModel.SetParameters(parameters);
        Assert.NotNull(metaModel.GetModelMetadata());

        var optimizer = metaModel.Optimizer;
        Assert.True(optimizer.NumParameters > 0);
        optimizer.SetAdamBeta1(0, 0.5);
        optimizer.SetAdamBeta2(0, 0.9);
        optimizer.SetAdamEpsilon(0, 1e-6);
        optimizer.SetDirection(0, 1.0);
        optimizer.SetMomentum(0, 0.1);
    }

    [Fact]
    public void MetaOptNet_LogisticRegression_UsesTemperature()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaOptNetOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            SolverType = ConvexSolverType.LogisticRegression,
            RegularizationStrength = 0.1,
            OuterLearningRate = 0.01,
            MaxSolverIterations = 2,
            UseLearnedTemperature = true,
            InitialTemperature = 1.5
        };

        var algorithm = new MetaOptNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(23, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var metaModel = Assert.IsType<MetaOptNetModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = metaModel.Predict(task.QuerySetX);
        var parameters = metaModel.GetParameters();

        Assert.False(double.IsNaN(loss));
        Assert.Equal(task.QuerySetX.Rows, predictions.Length);
        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void MetaOptNet_Svm_ModelProperties()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaOptNetOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            SolverType = ConvexSolverType.SVM,
            RegularizationStrength = 0.2,
            OuterLearningRate = 0.01,
            MaxSolverIterations = 2,
            UseLearnedTemperature = false,
            InitialTemperature = 1.1
        };

        var algorithm = new MetaOptNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(24, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);

        var adapted = algorithm.Adapt(task);
        var metaModel = Assert.IsType<MetaOptNetModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = metaModel.Predict(task.QuerySetX);

        Assert.Equal(MetaLearningAlgorithmType.MetaOptNet, algorithm.AlgorithmType);
        Assert.Equal(options.NumClasses, metaModel.NumClasses);
        Assert.Equal(options.EmbeddingDimension, metaModel.ClassifierWeights.Columns);
        Assert.Equal(task.QuerySetX.Rows, predictions.Length);
        Assert.NotNull(metaModel.GetModelMetadata());
    }

    [Fact]
    public void RelationNetwork_ModelAndModuleCoverage()
    {
        var module = new RelationModule<double>(2);
        var combined = new Tensor<double>(new[] { 2 });
        combined[0] = 0.25;
        combined[1] = -0.5;

        var moduleOutput = module.Forward(combined);
        var moduleClone = module.Clone();
        module.SetTrainingMode(true);
        var moduleParameters = module.GetParameters();

        Assert.NotSame(module, moduleClone);
        Assert.Equal(1, moduleOutput.Shape[0]);
        Assert.True(moduleParameters.Length > 0);

        var model = new LinearVectorModel(2);
        var options = new RelationNetworkOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            RelationHiddenDimension = 2,
            OuterLearningRate = 0.01,
            AggregationMethod = RelationAggregationMethod.Max,
            FeatureEncoderL2Reg = 0.1,
            RelationModuleL2Reg = 0.1
        };

        var algorithm = new RelationNetworkAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(25, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var relationModel = Assert.IsType<RelationNetworkModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = relationModel.Predict(task.QuerySetX);
        var parameters = relationModel.GetParameters();

        var query = new Vector<double>(2);
        query[0] = 1.0;
        query[1] = 2.0;
        var support = new Vector<double>(2);
        support[0] = 0.5;
        support[1] = -0.5;

        var modelType = typeof(RelationNetworkModel<double, Matrix<double>, Vector<double>>);
        var diff = InvokePrivate<Tensor<double>>(relationModel, modelType, "ComputeDifferenceFeatures", query, support);
        var product = InvokePrivate<Tensor<double>>(relationModel, modelType, "ComputeProductFeatures", query, support);

        var sample = new Tensor<double>(new[] { 2 });
        sample[0] = 0.3;
        sample[1] = 0.7;
        var encoded = InvokePrivate<Vector<double>>(relationModel, modelType, "EncodeSample", sample);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(options.NumClasses, predictions.Length);
        Assert.True(parameters.Length > 0);
        Assert.Equal(2, diff.Length);
        Assert.Equal(2, product.Length);
        Assert.Equal(2, encoded.Length);
        Assert.NotNull(relationModel.GetModelMetadata());
    }

    [Fact]
    public void TADAM_ModelAndRegularizationCoverage()
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
            L2Regularization = 0.1
        };

        var algorithm = new TADAMAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateTensorLabelTask(26, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var tadamModel = Assert.IsType<TADAMModel<double, Matrix<double>, Tensor<double>>>(adapted);
        var predictions = tadamModel.Predict(task.QuerySetX);
        var parameters = tadamModel.GetParameters();

        Assert.False(double.IsNaN(loss));
        Assert.Equal(options.NumClasses, predictions.Shape[0]);
        Assert.True(parameters.Length > 0);
        Assert.NotNull(tadamModel.GetModelMetadata());
    }

    [Fact]
    public void MANN_ModelAndMemoryCoverage()
    {
        var model = new LinearVectorModel(2);
        var options = new MANNOptions<double, Matrix<double>, Vector<double>>(model)
        {
            MemorySize = 3,
            MemoryKeySize = 2,
            MemoryValueSize = 2,
            NumClasses = 2,
            NumReadHeads = 1,
            NumWriteHeads = 1,
            OuterLearningRate = 0.01,
            ClearMemoryBetweenTasks = true,
            MemoryRetentionRatio = 0.5,
            MemoryRegularization = 0.1,
            UseMemoryConsolidation = true,
            MemoryUsageThreshold = 2.0,
            UseMemoryPreInitialization = true,
            UseOutputSoftmax = true
        };

        var algorithm = new MANNAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(27, supportRows: 2, queryRows: 2, featureCount: 2, numWays: 2);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var mannModel = Assert.IsType<MANNModel<double, Matrix<double>, Vector<double>>>(adapted);
        var predictions = mannModel.Predict(task.QuerySetX);
        var parameters = mannModel.GetParameters();

        var memory = new ExternalMemory<double>(2, 2, 2, MathHelper.GetNumericOperations<double>());
        var key = new Vector<double>(2);
        key[0] = 0.1;
        key[1] = -0.2;
        var value = new Vector<double>(2);
        value[0] = 1.0;
        value[1] = 0.0;
        memory.Write(0, key, value);
        memory.Write(1, key, value);
        var penalty = memory.ComputeUsagePenalty();
        var rarelyUsed = memory.FindRarelyUsedSlots(2.0);

        Assert.False(double.IsNaN(loss));
        Assert.Equal(options.NumClasses, predictions.Length);
        Assert.True(parameters.Length > 0);
        Assert.True(penalty > 0);
        Assert.Equal(memory.Size, rarelyUsed.Count);
        Assert.NotNull(mannModel.GetModelMetadata());
    }

    [Fact]
    public void NTM_LstmController_And_MemoryCoverage()
    {
        var model = new TensorEmbeddingModel(2, 2);
        var options = new NTMOptions<double, Matrix<double>, Tensor<double>>(model)
        {
            MemorySize = 3,
            MemoryWidth = 2,
            NumReadHeads = 1,
            NumWriteHeads = 1,
            NumClasses = 2,
            ControllerHiddenSize = 3,
            OuterLearningRate = 0.01,
            ControllerType = NTMControllerType.LSTM,
            InitializeMemory = true,
            MemoryInitialization = NTMMemoryInitialization.Random,
            MemoryUsageRegularization = 0.1,
            MemorySharpnessRegularization = 0.1
        };

        var algorithm = new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var task = CreateFixedTensorOutputTask(28, inputRows: 1, featureCount: 2, numClasses: options.NumClasses);

        var adapted = algorithm.Adapt(task);
        var ntmModel = Assert.IsType<NTMModel<double, Matrix<double>, Tensor<double>>>(adapted);
        var predictions = ntmModel.Predict(task.QuerySetX);
        var parameters = ntmModel.GetParameters();

        var controller = new LSTMNTMController<double, Matrix<double>, Tensor<double>>(options);
        var input = new Tensor<double>(new[] { options.MemoryWidth });
        input[0] = 0.1;
        input[1] = -0.2;
        var readContents = new List<Tensor<double>> { new Tensor<double>(new[] { options.MemoryWidth }) };
        var controllerOutput = controller.Forward(input, readContents);
        var readKeys = controller.GenerateReadKeys(controllerOutput);
        var writeKey = controller.GenerateWriteKey(controllerOutput);
        var eraseVector = controller.GenerateEraseVector(controllerOutput);
        var addVector = controller.GenerateAddVector(controllerOutput);
        var output = controller.GenerateOutput(controllerOutput, readContents);
        var controllerParams = controller.GetParameters();
        controller.SetParameters(controllerParams);
        controller.Reset();

        var memory = new NTMMemory<double>(2, 2, 0);
        memory.InitializeRandom();
        memory.InitializeLearned();
        var writeWeights = new Vector<double>(2);
        writeWeights[0] = 1.0;
        var erase = new Tensor<double>(new[] { 2 });
        erase[0] = 0.5;
        erase[1] = 0.1;
        var add = new Tensor<double>(new[] { 2 });
        add[0] = 0.2;
        add[1] = -0.1;
        memory.Write(writeWeights, erase, add);
        var usage = memory.ComputeUsagePenalty();
        var sharpness = memory.ComputeSharpnessPenalty();

        var memoryField = typeof(NTMAlgorithm<double, Matrix<double>, Tensor<double>>).GetField("_memory", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(memoryField);
        var internalMemory = (NTMMemory<double>)memoryField!.GetValue(algorithm)!;
        var internalWeights = new Vector<double>(options.MemorySize);
        internalWeights[0] = 1.0;
        internalMemory.Write(internalWeights, erase, add);
        var regularized = InvokePrivate<double>(
            algorithm,
            typeof(NTMAlgorithm<double, Matrix<double>, Tensor<double>>),
            "AddMemoryRegularization",
            1.0);

        Assert.Equal(options.NumClasses, predictions.Shape[0]);
        Assert.True(parameters.Length > 0);
        Assert.True(controllerParams.Length > 0);
        Assert.True(readKeys.Count > 0);
        Assert.True(writeKey.Length > 0);
        Assert.True(eraseVector.Length > 0);
        Assert.True(addVector.Length > 0);
        Assert.True(output.Length > 0);
        Assert.True(usage >= 0.0);
        Assert.True(sharpness >= 0.0);
        Assert.True(regularized >= 0.0);
        Assert.NotNull(ntmModel.GetModelMetadata());
    }
}
