using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Training;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.UnitTests.MetaLearning.TestHelpers;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// End-to-end integration tests for meta-learning algorithms.
/// </summary>
public class E2EMetaLearningTests
{
    [Fact]
    public void SEAL_5Way1Shot_TrainsSuccessfully()
    {
        // Arrange
        const int numWays = 5;
        const int numShots = 1;
        const int numQueryPerClass = 15;
        const int inputDim = 10;
        const int hiddenDim = 32;

        // Create a simple neural network as the base model
        var architecture = new NeuralNetworkArchitecture<double>
        {
            InputSize = inputDim,
            OutputSize = numWays,
            HiddenLayerSizes = new[] { hiddenDim },
            ActivationFunctionType = ActivationFunctionType.ReLU,
            OutputActivationFunctionType = ActivationFunctionType.Softmax,
            TaskType = TaskType.Classification
        };

        var baseModel = new NeuralNetwork<double>(architecture);
        var fullModel = new NeuralNetworkModel<double>(baseModel);

        // Create SEAL algorithm with options
        var sealOptions = new SEALAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = fullModel,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            MetaBatchSize = 4,
            RandomSeed = 42,
            Temperature = 1.0,
            UseFirstOrder = true
        };

        var sealAlgorithm = new SEALAlgorithm<double, Matrix<double>, Vector<double>>(sealOptions);

        // Create mock datasets
        var trainDataset = new MockEpisodicDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20,
            examplesPerClass: 50,
            inputDim: inputDim,
            split: DatasetSplit.Train
        );

        var valDataset = new MockEpisodicDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 10,
            examplesPerClass: 50,
            inputDim: inputDim,
            split: DatasetSplit.Validation
        );

        // Create trainer
        var trainerOptions = new MetaTrainerOptions
        {
            NumEpochs = 5,
            TasksPerEpoch = 100,
            MetaBatchSize = 4,
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass,
            ValInterval = 2,
            ValTasks = 20,
            LogInterval = 1,
            CheckpointInterval = 0, // Disable checkpointing for test
            EarlyStoppingPatience = 10,
            RandomSeed = 42,
            Verbose = false
        };

        var trainer = new MetaTrainer<double, Matrix<double>, Vector<double>>(
            sealAlgorithm,
            trainDataset,
            valDataset,
            trainerOptions
        );

        // Act
        var history = trainer.Train();

        // Assert
        Assert.NotNull(history);
        Assert.Equal(5, history.Count);
        Assert.All(history, metrics => Assert.True(metrics.TrainLoss >= 0));

        // Verify that the algorithm can adapt to a new task
        var testTasks = trainDataset.SampleTasks(1, numWays, numShots, numQueryPerClass);
        var testTask = testTasks[0];
        var adaptedModel = sealAlgorithm.Adapt(testTask);

        Assert.NotNull(adaptedModel);

        // Verify that adapted model can make predictions
        var predictions = adaptedModel.Predict(testTask.QueryInput);
        Assert.NotNull(predictions);
        Assert.Equal(testTask.QueryInput.Rows, predictions.Length);
    }

    [Fact]
    public void MAML_5Way1Shot_TrainsSuccessfully()
    {
        // Arrange
        const int numWays = 5;
        const int numShots = 1;
        const int inputDim = 10;
        const int hiddenDim = 32;

        var architecture = new NeuralNetworkArchitecture<double>
        {
            InputSize = inputDim,
            OutputSize = numWays,
            HiddenLayerSizes = new[] { hiddenDim },
            ActivationFunctionType = ActivationFunctionType.ReLU,
            OutputActivationFunctionType = ActivationFunctionType.Softmax,
            TaskType = TaskType.Classification
        };

        var baseModel = new NeuralNetwork<double>(architecture);
        var fullModel = new NeuralNetworkModel<double>(baseModel);

        var mamlOptions = new MAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = fullModel,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            MetaBatchSize = 4,
            RandomSeed = 42,
            UseFirstOrder = true
        };

        var mamlAlgorithm = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(mamlOptions);

        var trainDataset = new MockEpisodicDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20,
            examplesPerClass: 50,
            inputDim: inputDim
        );

        // Act - Train for a few epochs
        var tasks = trainDataset.SampleTasks(4, numWays, numShots, 15);
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        double initialLoss = Convert.ToDouble(mamlAlgorithm.MetaTrain(taskBatch));

        // Assert
        Assert.True(initialLoss >= 0);
        Assert.True(initialLoss < double.MaxValue);
    }

    [Fact]
    public void Reptile_5Way1Shot_TrainsSuccessfully()
    {
        // Arrange
        const int numWays = 5;
        const int numShots = 1;
        const int inputDim = 10;
        const int hiddenDim = 32;

        var architecture = new NeuralNetworkArchitecture<double>
        {
            InputSize = inputDim,
            OutputSize = numWays,
            HiddenLayerSizes = new[] { hiddenDim },
            ActivationFunctionType = ActivationFunctionType.ReLU,
            OutputActivationFunctionType = ActivationFunctionType.Softmax,
            TaskType = TaskType.Classification
        };

        var baseModel = new NeuralNetwork<double>(architecture);
        var fullModel = new NeuralNetworkModel<double>(baseModel);

        var reptileOptions = new ReptileAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = fullModel,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            MetaBatchSize = 4,
            RandomSeed = 42,
            Interpolation = 1.0,
            InnerBatches = 3
        };

        var reptileAlgorithm = new ReptileAlgorithm<double, Matrix<double>, Vector<double>>(reptileOptions);

        var trainDataset = new MockEpisodicDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20,
            examplesPerClass: 50,
            inputDim: inputDim
        );

        // Act
        var tasks = trainDataset.SampleTasks(4, numWays, numShots, 15);
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        double loss = Convert.ToDouble(reptileAlgorithm.MetaTrain(taskBatch));

        // Assert
        Assert.True(loss >= 0);
        Assert.True(loss < double.MaxValue);
    }

    [Fact]
    public void iMAML_5Way1Shot_TrainsSuccessfully()
    {
        // Arrange
        const int numWays = 5;
        const int numShots = 1;
        const int inputDim = 10;
        const int hiddenDim = 32;

        var architecture = new NeuralNetworkArchitecture<double>
        {
            InputSize = inputDim,
            OutputSize = numWays,
            HiddenLayerSizes = new[] { hiddenDim },
            ActivationFunctionType = ActivationFunctionType.ReLU,
            OutputActivationFunctionType = ActivationFunctionType.Softmax,
            TaskType = TaskType.Classification
        };

        var baseModel = new NeuralNetwork<double>(architecture);
        var fullModel = new NeuralNetworkModel<double>(baseModel);

        var imamlOptions = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = fullModel,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            MetaBatchSize = 4,
            RandomSeed = 42,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 5
        };

        var imamlAlgorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(imamlOptions);

        var trainDataset = new MockEpisodicDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20,
            examplesPerClass: 50,
            inputDim: inputDim
        );

        // Act
        var tasks = trainDataset.SampleTasks(4, numWays, numShots, 15);
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        double loss = Convert.ToDouble(imamlAlgorithm.MetaTrain(taskBatch));

        // Assert
        Assert.True(loss >= 0);
        Assert.True(loss < double.MaxValue);
    }

    [Fact]
    public void MetaLearning_Algorithms_AreComparable()
    {
        // This test verifies that all algorithms can be trained and compared on the same task
        const int numWays = 3;
        const int numShots = 1;
        const int inputDim = 8;

        var dataset = new MockEpisodicDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 15,
            examplesPerClass: 30,
            inputDim: inputDim
        );

        var tasks = dataset.SampleTasks(2, numWays, numShots, 10);
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        var algorithms = new[]
        {
            CreateSEALAlgorithm(inputDim, numWays),
            CreateMAMLAlgorithm(inputDim, numWays),
            CreateReptileAlgorithm(inputDim, numWays),
            CreateiMAMLAlgorithm(inputDim, numWays)
        };

        // Act & Assert
        foreach (var algorithm in algorithms)
        {
            double loss = Convert.ToDouble(algorithm.MetaTrain(taskBatch));
            Assert.True(loss >= 0, $"{algorithm.AlgorithmName} produced negative loss");
            Assert.True(loss < double.MaxValue, $"{algorithm.AlgorithmName} produced infinite loss");
        }
    }

    private IMetaLearningAlgorithm<double, Matrix<double>, Vector<double>> CreateSEALAlgorithm(
        int inputDim, int numWays)
    {
        var arch = CreateArchitecture(inputDim, numWays);
        var model = new NeuralNetworkModel<double>(new NeuralNetwork<double>(arch));
        var options = new SEALAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = model,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 3,
            RandomSeed = 42,
            UseFirstOrder = true
        };
        return new SEALAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    private IMetaLearningAlgorithm<double, Matrix<double>, Vector<double>> CreateMAMLAlgorithm(
        int inputDim, int numWays)
    {
        var arch = CreateArchitecture(inputDim, numWays);
        var model = new NeuralNetworkModel<double>(new NeuralNetwork<double>(arch));
        var options = new MAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = model,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 3,
            RandomSeed = 42,
            UseFirstOrder = true
        };
        return new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    private IMetaLearningAlgorithm<double, Matrix<double>, Vector<double>> CreateReptileAlgorithm(
        int inputDim, int numWays)
    {
        var arch = CreateArchitecture(inputDim, numWays);
        var model = new NeuralNetworkModel<double>(new NeuralNetwork<double>(arch));
        var options = new ReptileAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = model,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 3,
            RandomSeed = 42
        };
        return new ReptileAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    private IMetaLearningAlgorithm<double, Matrix<double>, Vector<double>> CreateiMAMLAlgorithm(
        int inputDim, int numWays)
    {
        var arch = CreateArchitecture(inputDim, numWays);
        var model = new NeuralNetworkModel<double>(new NeuralNetwork<double>(arch));
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = model,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 3,
            RandomSeed = 42
        };
        return new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    private NeuralNetworkArchitecture<double> CreateArchitecture(int inputDim, int outputDim)
    {
        return new NeuralNetworkArchitecture<double>
        {
            InputSize = inputDim,
            OutputSize = outputDim,
            HiddenLayerSizes = new[] { 16 },
            ActivationFunctionType = ActivationFunctionType.ReLU,
            OutputActivationFunctionType = ActivationFunctionType.Softmax,
            TaskType = TaskType.Classification
        };
    }
}
