using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using AiDotNet.Models;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Unit tests for SEALTrainer implementation.
/// </summary>
public class SEALTrainerTests
{
    [Fact]
    public void Constructor_ValidInputs_InitializesSuccessfully()
    {
        // Arrange
        var (dataLoader, model, selfSupervisedLoss) = CreateTestSetup();
        var config = new SEALTrainerConfig<double>(
            selfSupervisedSteps: 5,
            supervisedSteps: 3,
            activeLearningK: 10,
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            metaBatchSize: 2,
            numMetaIterations: 1);

        // Act
        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredErrorLoss<double>(),
            selfSupervisedLoss: selfSupervisedLoss,
            dataLoader: dataLoader,
            config: config);

        // Assert
        Assert.NotNull(trainer);
        Assert.Equal(0, trainer.CurrentIteration);
    }

    [Fact]
    public void Constructor_NullSelfSupervisedLoss_ThrowsArgumentNullException()
    {
        // Arrange
        var (dataLoader, model, _) = CreateTestSetup();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new SEALTrainer<double, Tensor<double>, Tensor<double>>(
                metaModel: model,
                lossFunction: new MeanSquaredErrorLoss<double>(),
                selfSupervisedLoss: null,
                dataLoader: dataLoader,
                config: new SEALTrainerConfig<double>()));
    }

    [Fact]
    public void MetaTrainStep_SingleIteration_CompletesWithoutError()
    {
        // Arrange
        var (dataLoader, model, selfSupervisedLoss) = CreateTestSetup();
        var config = new SEALTrainerConfig<double>(
            selfSupervisedSteps: 5,
            supervisedSteps: 3,
            activeLearningK: 10,
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            metaBatchSize: 2,
            numMetaIterations: 1);

        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredErrorLoss<double>(),
            selfSupervisedLoss: selfSupervisedLoss,
            dataLoader: dataLoader,
            config: config);

        // Act
        var result = trainer.MetaTrainStep(batchSize: 2);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.NumTasks);
        Assert.True(result.TimeMs > 0);
        Assert.False(double.IsNaN(result.MetaLoss));
        Assert.Equal(1, trainer.CurrentIteration);
    }

    [Fact]
    public void MetaTrainStep_UpdatesModelParameters()
    {
        // Arrange
        var (dataLoader, model, selfSupervisedLoss) = CreateTestSetup();
        var initialParams = model.GetParameters().Clone();

        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredErrorLoss<double>(),
            selfSupervisedLoss: selfSupervisedLoss,
            dataLoader: dataLoader,
            config: new SEALTrainerConfig<double>());

        // Act
        trainer.MetaTrainStep(batchSize: 2);
        var updatedParams = model.GetParameters();

        // Assert: Parameters should have changed
        bool parametersChanged = false;
        for (int i = 0; i < initialParams.Length; i++)
        {
            if (Math.Abs(initialParams[i] - updatedParams[i]) > 1e-10)
            {
                parametersChanged = true;
                break;
            }
        }
        Assert.True(parametersChanged, "Model parameters should have been updated after meta-training step");
    }

    [Fact]
    public void MetaTrainStep_InvalidBatchSize_ThrowsArgumentException()
    {
        // Arrange
        var (dataLoader, model, selfSupervisedLoss) = CreateTestSetup();
        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredErrorLoss<double>(),
            selfSupervisedLoss: selfSupervisedLoss,
            dataLoader: dataLoader,
            config: new SEALTrainerConfig<double>());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => trainer.MetaTrainStep(batchSize: 0));
        Assert.Throws<ArgumentException>(() => trainer.MetaTrainStep(batchSize: -1));
    }

    [Fact]
    public void AdaptAndEvaluate_ValidTask_ReturnsMetrics()
    {
        // Arrange
        var (dataLoader, model, selfSupervisedLoss) = CreateTestSetup();
        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredErrorLoss<double>(),
            selfSupervisedLoss: selfSupervisedLoss,
            dataLoader: dataLoader,
            config: new SEALTrainerConfig<double>());

        var task = dataLoader.GetNextTask();

        // Act
        var result = trainer.AdaptAndEvaluate(task);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.AdaptationTimeMs > 0);
        Assert.False(double.IsNaN(result.QueryLoss));
        Assert.False(double.IsNaN(result.QueryAccuracy));
    }

    [Fact]
    public void Train_MultipleIterations_CompletesSuccessfully()
    {
        // Arrange
        var (dataLoader, model, selfSupervisedLoss) = CreateTestSetup();
        var config = new SEALTrainerConfig<double>(
            selfSupervisedSteps: 3,
            supervisedSteps: 2,
            activeLearningK: 5,
            metaBatchSize: 2,
            numMetaIterations: 5);  // Small number for fast test

        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredErrorLoss<double>(),
            selfSupervisedLoss: selfSupervisedLoss,
            dataLoader: dataLoader,
            config: config);

        // Act
        var result = trainer.Train();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(5, result.LossHistory.Length);
        Assert.Equal(5, result.AccuracyHistory.Length);
        Assert.True(result.TrainingTime.TotalMilliseconds > 0);
    }

    /// <summary>
    /// Creates test setup with mock components.
    /// </summary>
    private (IEpisodicDataLoader<double, Tensor<double>, Tensor<double>> dataLoader,
             IFullModel<double, Tensor<double>, Tensor<double>> model,
             ISelfSupervisedLoss<double> selfSupervisedLoss) CreateTestSetup()
    {
        // Create dataset: 10 classes, 20 examples per class
        var (datasetX, datasetY) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, imageSize: 28);

        // Create data loader for 5-way 5-shot tasks with 15 query examples
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            datasetX: datasetX,
            datasetY: datasetY,
            nWay: 5,
            kShot: 5,
            queryShots: 15);

        // Create mock model
        var model = new SEALMockModel(parameterCount: 100);

        // Create self-supervised loss
        var selfSupervisedLoss = new RotationPredictionLoss<double>();

        return (dataLoader, model, selfSupervisedLoss);
    }

    /// <summary>
    /// Creates a synthetic dataset for testing.
    /// </summary>
    private (Tensor<double> X, Tensor<double> Y) CreateTestDataset(int numClasses, int examplesPerClass, int imageSize)
    {
        int totalExamples = numClasses * examplesPerClass;

        // Create images: [N, H, W, C] with C=1 for grayscale
        var datasetX = new Tensor<double>(new[] { totalExamples, imageSize, imageSize, 1 });
        var datasetY = new Tensor<double>(new[] { totalExamples, numClasses });

        var random = new Random(42);

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            for (int exampleIdx = 0; exampleIdx < examplesPerClass; exampleIdx++)
            {
                int idx = classIdx * examplesPerClass + exampleIdx;

                // Generate synthetic pattern based on class
                for (int i = 0; i < imageSize; i++)
                {
                    for (int j = 0; j < imageSize; j++)
                    {
                        double value = Math.Sin((i + classIdx) * 0.3) * Math.Cos((j + classIdx) * 0.3);
                        value = (value + 1.0) / 2.0;  // Normalize to [0, 1]
                        value += random.NextDouble() * 0.1;  // Add noise
                        datasetX[idx, i, j, 0] = value;
                    }
                }

                // One-hot label
                for (int k = 0; k < numClasses; k++)
                {
                    datasetY[idx, k] = (k == classIdx) ? 1.0 : 0.0;
                }
            }
        }

        return (datasetX, datasetY);
    }
}

/// <summary>
/// Mock model that produces predictions with varying confidence for SEAL testing.
/// </summary>
internal class SEALMockModel : IFullModel<double, Tensor<double>, Tensor<double>>
{
    private Vector<double> _parameters;
    private Random _random = new Random(42);
    public int TrainCallCount { get; private set; }
    public int PredictCallCount { get; private set; }

    public SEALMockModel(int parameterCount)
    {
        _parameters = new Vector<double>(parameterCount);
        for (int i = 0; i < parameterCount; i++)
        {
            _parameters[i] = 0.1 * i;
        }
        TrainCallCount = 0;
        PredictCallCount = 0;
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters.Length != _parameters.Length)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        }
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new SEALMockModel(_parameters.Length);
        newModel.SetParameters(parameters);
        return newModel;
    }

    public void Train(Tensor<double> input, Tensor<double> expectedOutput)
    {
        TrainCallCount++;
        // Simple update: add a small value to each parameter
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] += 0.01;
        }
    }

    public Tensor<double> Predict(Tensor<double> input)
    {
        PredictCallCount++;

        // Return predictions with varying confidence (softmax-like distribution)
        int numExamples = input.Shape[0];
        // Assume 5-way classification based on typical 5-way 5-shot setup
        int numClasses = 5;

        var predictions = new Tensor<double>(new[] { numExamples, numClasses });

        for (int i = 0; i < numExamples; i++)
        {
            // Create a probability distribution with one dominant class
            double[] probs = new double[numClasses];
            int predictedClass = _random.Next(numClasses);

            // Make predicted class have high confidence (0.6-0.9)
            probs[predictedClass] = 0.6 + _random.NextDouble() * 0.3;

            // Distribute remaining probability among other classes
            double remaining = 1.0 - probs[predictedClass];
            for (int j = 0; j < numClasses; j++)
            {
                if (j != predictedClass)
                {
                    probs[j] = remaining / (numClasses - 1);
                }
            }

            // Copy to tensor
            for (int j = 0; j < numClasses; j++)
            {
                predictions[i, j] = probs[j];
            }
        }

        return predictions;
    }

    public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }
    public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
    {
        var copy = new SEALMockModel(_parameters.Length);
        copy.SetParameters(_parameters);
        return copy;
    }
    public IFullModel<double, Tensor<double>, Tensor<double>> Clone() => DeepCopy();
    public int InputFeatureCount => 784;  // 28x28 image
    public int OutputFeatureCount => 5;   // 5-way classification
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, InputFeatureCount);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < InputFeatureCount;
    public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
    public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
    {
        return new Vector<double>(ParameterCount);
    }
    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }
}
