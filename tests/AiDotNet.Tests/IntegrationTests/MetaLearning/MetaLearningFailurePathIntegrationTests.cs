using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Modules;
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

    private static Matrix<double> CreateSupportMatrix(int rows, int columns)
    {
        var matrix = new Matrix<double>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = (i + 1) * (j + 1) * 0.1;
            }
        }
        return matrix;
    }

    private static Vector<double> CreateLabelVector(int length, int numClasses)
    {
        var labels = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            labels[i] = i % Math.Max(1, numClasses);
        }
        return labels;
    }

    private static Dictionary<int, Tensor<double>> CreatePrototypeMap(int numClasses, int dimension)
    {
        var prototypes = new Dictionary<int, Tensor<double>>();
        for (int i = 0; i < numClasses; i++)
        {
            var prototype = new Tensor<double>(new[] { dimension });
            for (int j = 0; j < dimension; j++)
            {
                prototype[new[] { j }] = (i + 1) * 0.01 * (j + 1);
            }
            prototypes[i] = prototype;
        }
        return prototypes;
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

    [Fact]
    public void MetaLearningModels_TrainAndUpdateParameters_Throw()
    {
        static void AssertModelNotSupported(
            Action trainAction,
            Action updateAction)
        {
            Assert.Throws<NotSupportedException>(trainAction);
            Assert.Throws<NotSupportedException>(updateAction);
        }

        var trainInput = new Matrix<double>(1, 2);
        var trainOutput = new Vector<double>(1);

        var anilModel = CreateAnilModel();
        var boilModel = CreateBoilModel();
        var leoModel = CreateLeoModel();
        var metaOptNetModel = CreateMetaOptNetModel();
        var relationModel = CreateRelationNetworkModel();
        var tadamModel = CreateTadamModel();
        var matchingModel = CreateMatchingNetworksModel();
        var protoModel = CreatePrototypicalModel();
        var mannModel = CreateMannModel();
        var ntmModel = CreateNtmModel();

        AssertModelNotSupported(
            () => anilModel.Train(trainInput, trainOutput),
            () => anilModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => boilModel.Train(trainInput, trainOutput),
            () => boilModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => leoModel.Train(trainInput, trainOutput),
            () => leoModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => metaOptNetModel.Train(trainInput, trainOutput),
            () => metaOptNetModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => relationModel.Train(trainInput, trainOutput),
            () => relationModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => tadamModel.Train(trainInput, trainOutput),
            () => tadamModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => matchingModel.Train(trainInput, trainOutput),
            () => matchingModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => protoModel.Train(trainInput, trainOutput),
            () => protoModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => mannModel.Train(trainInput, trainOutput),
            () => mannModel.UpdateParameters(new Vector<double>(1)));
        AssertModelNotSupported(
            () => ntmModel.Train(trainInput, trainOutput),
            () => ntmModel.UpdateParameters(new Vector<double>(1)));

        Assert.Throws<NotSupportedException>(() => protoModel.GetParameters());
    }

    [Fact]
    public void MetaLearningAlgorithms_VectorOutputs_Adapt_NullTask_Throws()
    {
        static void AssertNullTaskThrows<TInput, TOutput>(
            IMetaLearner<double, TInput, TOutput> algorithm)
        {
            Assert.Throws<ArgumentNullException>(() => algorithm.Adapt(null!));
        }

        AssertNullTaskThrows(new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(
            new MAMLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                EvaluationTasks = 1
            }));

        AssertNullTaskThrows(new ReptileAlgorithm<double, Matrix<double>, Vector<double>>(
            new ReptileOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                EvaluationTasks = 1,
                InnerBatches = 1,
                Interpolation = 0.5
            }));

        AssertNullTaskThrows(new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(
            new MetaSGDOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                AdaptationSteps = 1,
                InnerSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                EvaluationTasks = 1,
                UseWarmStart = false
            }));

        AssertNullTaskThrows(new ANILAlgorithm<double, Matrix<double>, Vector<double>>(
            new ANILOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                FeatureDimension = 2,
                NumClasses = 2,
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new BOILAlgorithm<double, Matrix<double>, Vector<double>>(
            new BOILOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                FeatureDimension = 2,
                NumClasses = 2,
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(
            new iMAMLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                EvaluationTasks = 1,
                ConjugateGradientIterations = 1,
                ConjugateGradientTolerance = 1e-4,
                NeumannSeriesTerms = 1
            }));

        AssertNullTaskThrows(new CNAPAlgorithm<double, Matrix<double>, Vector<double>>(
            new CNAPOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                RepresentationDimension = 2,
                HiddenDimension = 2,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new SEALAlgorithm<double, Matrix<double>, Vector<double>>(
            new SEALOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                UseAdaptiveInnerLR = false
            }));

        AssertNullTaskThrows(new GNNMetaAlgorithm<double, Matrix<double>, Vector<double>>(
            new GNNMetaOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                NodeEmbeddingDimension = 2,
                GNNHiddenDimension = 2,
                NumMessagePassingLayers = 1,
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new LEOAlgorithm<double, Matrix<double>, Vector<double>>(
            new LEOOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                EmbeddingDimension = 2,
                LatentDimension = 2,
                HiddenDimension = 2,
                NumClasses = 2,
                AdaptationSteps = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                UseRelationEncoder = false
            }));

        AssertNullTaskThrows(new MetaOptNetAlgorithm<double, Matrix<double>, Vector<double>>(
            new MetaOptNetOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                NumClasses = 2,
                EmbeddingDimension = 2,
                SolverType = ConvexSolverType.RidgeRegression,
                OuterLearningRate = 0.01,
                MaxSolverIterations = 1
            }));

        AssertNullTaskThrows(new MANNAlgorithm<double, Matrix<double>, Vector<double>>(
            new MANNOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
            {
                MemorySize = 2,
                MemoryKeySize = 2,
                MemoryValueSize = 2,
                NumClasses = 2,
                NumReadHeads = 1,
                NumWriteHeads = 1,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));
    }

    [Fact]
    public void MetaLearningAlgorithms_TensorOutputs_Adapt_NullTask_Throws()
    {
        static void AssertNullTaskThrows<TInput, TOutput>(
            IMetaLearner<double, TInput, TOutput> algorithm)
        {
            Assert.Throws<ArgumentNullException>(() => algorithm.Adapt(null!));
        }

        AssertNullTaskThrows(new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(
            new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
            {
                MetaBatchSize = 1,
                NumMetaIterations = 1,
                EvaluationTasks = 1
            }));

        AssertNullTaskThrows(new MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>>(
            new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
            {
                NumClasses = 2,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new RelationNetworkAlgorithm<double, Matrix<double>, Tensor<double>>(
            new RelationNetworkOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
            {
                NumClasses = 2,
                RelationHiddenDimension = 2,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new TADAMAlgorithm<double, Matrix<double>, Tensor<double>>(
            new TADAMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
            {
                NumClasses = 2,
                EmbeddingDimension = 2,
                TaskEmbeddingDimension = 2,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));

        AssertNullTaskThrows(new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(
            new NTMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
            {
                MemorySize = 2,
                MemoryWidth = 2,
                NumReadHeads = 1,
                NumWriteHeads = 1,
                NumClasses = 2,
                ControllerHiddenSize = 2,
                ControllerType = NTMControllerType.MLP,
                MetaBatchSize = 1,
                NumMetaIterations = 1
            }));
    }

    private static ANILModel<double, Matrix<double>, Vector<double>> CreateAnilModel()
    {
        var model = new LinearVectorModel(2);
        var options = new ANILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            FeatureDimension = 2,
            NumClasses = 2,
            UseHeadBias = true
        };
        var headWeights = new Vector<double>(options.FeatureDimension * options.NumClasses);
        var headBias = new Vector<double>(options.NumClasses);
        return new ANILModel<double, Matrix<double>, Vector<double>>(model, headWeights, headBias, options);
    }

    private static BOILModel<double, Matrix<double>, Vector<double>> CreateBoilModel()
    {
        var model = new LinearVectorModel(2);
        var options = new BOILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            FeatureDimension = 2,
            NumClasses = 2
        };
        var adaptedBodyParams = model.GetParameters();
        var headWeights = new Vector<double>(options.FeatureDimension * options.NumClasses);
        var headBias = new Vector<double>(options.NumClasses);
        return new BOILModel<double, Matrix<double>, Vector<double>>(
            model,
            adaptedBodyParams,
            headWeights,
            headBias,
            options);
    }

    private static LEOModel<double, Matrix<double>, Vector<double>> CreateLeoModel()
    {
        var model = new LinearVectorModel(2);
        var options = new LEOOptions<double, Matrix<double>, Vector<double>>(model)
        {
            EmbeddingDimension = 2,
            LatentDimension = 2,
            HiddenDimension = 2,
            NumClasses = 2
        };
        var classifierParams = new Vector<double>(options.EmbeddingDimension * options.NumClasses);
        var latentCode = new Vector<double>(options.LatentDimension);
        return new LEOModel<double, Matrix<double>, Vector<double>>(model, classifierParams, latentCode, options);
    }

    private static MetaOptNetModel<double, Matrix<double>, Vector<double>> CreateMetaOptNetModel()
    {
        var model = new LinearVectorModel(2);
        var options = new MetaOptNetOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            SolverType = ConvexSolverType.RidgeRegression
        };
        var classifierWeights = new Matrix<double>(options.NumClasses, options.EmbeddingDimension);
        return new MetaOptNetModel<double, Matrix<double>, Vector<double>>(
            model,
            classifierWeights,
            1.0,
            options);
    }

    private static RelationNetworkModel<double, Matrix<double>, Vector<double>> CreateRelationNetworkModel()
    {
        var model = new LinearVectorModel(2);
        var options = new RelationNetworkOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            RelationHiddenDimension = 2
        };
        var relationModule = new RelationModule<double>(options.RelationHiddenDimension);
        var supportInputs = CreateSupportMatrix(2, 2);
        var supportOutputs = CreateLabelVector(2, options.NumClasses);
        return new RelationNetworkModel<double, Matrix<double>, Vector<double>>(
            model,
            relationModule,
            supportInputs,
            supportOutputs,
            options);
    }

    private static TADAMModel<double, Matrix<double>, Vector<double>> CreateTadamModel()
    {
        var model = new LinearVectorModel(2);
        var options = new TADAMOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            TaskEmbeddingDimension = 2,
            UseMetricScaling = true
        };
        var prototypes = CreatePrototypeMap(options.NumClasses, options.EmbeddingDimension);
        var metricScale = new Vector<double>(options.EmbeddingDimension);
        return new TADAMModel<double, Matrix<double>, Vector<double>>(
            model,
            prototypes,
            metricScale,
            1.0,
            options);
    }

    private static MatchingNetworksModel<double, Matrix<double>, Vector<double>> CreateMatchingNetworksModel()
    {
        var model = new LinearVectorModel(2);
        var options = new MatchingNetworksOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumClasses = 2
        };
        var supportInputs = CreateSupportMatrix(2, 2);
        var supportOutputs = CreateLabelVector(2, options.NumClasses);
        var numOps = MathHelper.GetNumericOperations<double>();
        return new MatchingNetworksModel<double, Matrix<double>, Vector<double>>(
            model,
            supportInputs,
            supportOutputs,
            options,
            numOps);
    }

    private static PrototypicalModel<double, Matrix<double>, Vector<double>> CreatePrototypicalModel()
    {
        var model = new LinearVectorModel(2);
        var options = new ProtoNetsOptions<double, Matrix<double>, Vector<double>>(model)
        {
            NumMetaIterations = 1,
            MetaBatchSize = 1,
            EvaluationTasks = 1
        };
        var supportInputs = CreateSupportMatrix(2, 2);
        var supportOutputs = CreateLabelVector(2, 2);
        var numOps = MathHelper.GetNumericOperations<double>();
        return new PrototypicalModel<double, Matrix<double>, Vector<double>>(
            model,
            supportInputs,
            supportOutputs,
            options,
            numOps);
    }

    private static MANNModel<double, Matrix<double>, Vector<double>> CreateMannModel()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var options = new MANNOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
        {
            MemorySize = 2,
            MemoryKeySize = 2,
            MemoryValueSize = 2,
            NumClasses = 2,
            NumReadHeads = 1,
            NumWriteHeads = 1,
            MetaBatchSize = 1,
            NumMetaIterations = 1
        };
        var memory = new ExternalMemory<double>(
            options.MemorySize,
            options.MemoryKeySize,
            options.MemoryValueSize,
            numOps);
        return new MANNModel<double, Matrix<double>, Vector<double>>(
            new LinearVectorModel(2),
            memory,
            options,
            numOps);
    }

    private static NTMModel<double, Matrix<double>, Vector<double>> CreateNtmModel()
    {
        var options = new NTMOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
        {
            MemorySize = 2,
            MemoryWidth = 2,
            NumReadHeads = 1,
            NumWriteHeads = 1,
            NumClasses = 2,
            ControllerHiddenSize = 2,
            MetaBatchSize = 1,
            NumMetaIterations = 1
        };
        var controller = new TestNtmController(options.MemoryWidth, options.NumReadHeads);
        var memory = new NTMMemory<double>(options.MemorySize, options.MemoryWidth, randomSeed: 1);
        var readHeads = new List<NTMReadHead<double>>
        {
            new NTMReadHead<double>(options.MemoryWidth, options.MemorySize, 0)
        };
        var writeHead = new NTMWriteHead<double>(options.MemoryWidth, options.MemorySize);
        return new NTMModel<double, Matrix<double>, Vector<double>>(
            controller,
            memory,
            readHeads,
            writeHead,
            options);
    }

    private sealed class TestNtmController : INTMController<double>
    {
        private readonly int _memoryWidth;
        private readonly int _numReadHeads;

        public TestNtmController(int memoryWidth, int numReadHeads)
        {
            _memoryWidth = memoryWidth;
            _numReadHeads = numReadHeads;
        }

        public Tensor<double> Forward(Tensor<double> input, List<Tensor<double>> readContents)
        {
            return input;
        }

        public List<Tensor<double>> GenerateReadKeys(Tensor<double> output)
        {
            return Enumerable.Range(0, _numReadHeads)
                .Select(_ => new Tensor<double>(new[] { _memoryWidth }))
                .ToList();
        }

        public Tensor<double> GenerateWriteKey(Tensor<double> output)
        {
            return new Tensor<double>(new[] { _memoryWidth });
        }

        public Tensor<double> GenerateEraseVector(Tensor<double> output)
        {
            return new Tensor<double>(new[] { _memoryWidth });
        }

        public Tensor<double> GenerateAddVector(Tensor<double> output)
        {
            return new Tensor<double>(new[] { _memoryWidth });
        }

        public Tensor<double> GenerateOutput(Tensor<double> output, List<Tensor<double>> readContents)
        {
            if (readContents.Count > 0)
            {
                return readContents[0];
            }
            return new Tensor<double>(new[] { _memoryWidth });
        }

        public Vector<double> GetParameters() => new Vector<double>(1);

        public void SetParameters(Vector<double> parameters)
        {
        }

        public void Reset()
        {
        }
    }
}
