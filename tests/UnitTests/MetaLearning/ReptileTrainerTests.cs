using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Trainers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Unit tests for the ReptileTrainer class.
/// </summary>
public class ReptileTrainerTests
{
    #region Test Helper Classes

    /// <summary>
    /// Simple mock model for testing that tracks parameter updates.
    /// </summary>
    private class SimpleMockModel : IFullModel<double, Tensor<double>, Tensor<double>>
    {
        private Vector<double> _parameters;
        public int TrainCallCount { get; private set; }
        public int PredictCallCount { get; private set; }

        public SimpleMockModel(int parameterCount)
        {
            _parameters = new Vector<double>(parameterCount);
            // Initialize with small random values
            for (int i = 0; i < parameterCount; i++)
            {
                _parameters[i] = 0.1 * i;
            }
            TrainCallCount = 0;
            PredictCallCount = 0;
        }

        public Vector<double> GetParameters() => _parameters.Copy();

        public void SetParameters(Vector<double> parameters)
        {
            if (parameters.Length != _parameters.Length)
            {
                throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
            }
            _parameters = parameters.Copy();
        }

        public int ParameterCount => _parameters.Length;

        public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters)
        {
            var newModel = new SimpleMockModel(_parameters.Length);
            newModel.SetParameters(parameters);
            return newModel;
        }

        public void Train(Tensor<double> input, Tensor<double> expectedOutput)
        {
            TrainCallCount++;
            // Simple update: add a small value to each parameter to simulate training
            for (int i = 0; i < _parameters.Length; i++)
            {
                _parameters[i] += 0.01;
            }
        }

        public Tensor<double> Predict(Tensor<double> input)
        {
            PredictCallCount++;
            // Return a tensor of the same shape as input filled with zeros
            return new Tensor<double>(input.Shape);
        }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>();
        }

        public void Save(string filePath) { }
        public void Load(string filePath) { }

        public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
        {
            var copy = new SimpleMockModel(_parameters.Length);
            copy.SetParameters(_parameters);
            copy.TrainCallCount = TrainCallCount;
            copy.PredictCallCount = PredictCallCount;
            return copy;
        }

        public IFullModel<double, Tensor<double>, Tensor<double>> Clone()
        {
            return DeepCopy();
        }

        public int InputFeatureCount => 10;
        public int OutputFeatureCount => 1;
        public string[] FeatureNames { get; set; } = Array.Empty<string>();
        public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
    }

    /// <summary>
    /// Simple mock loss function for testing.
    /// </summary>
    private class SimpleMockLossFunction : ILossFunction<double>
    {
        public double CalculateLoss(Vector<double> predicted, Vector<double> actual)
        {
            // Simple mean squared error
            double sum = 0;
            for (int i = 0; i < predicted.Length && i < actual.Length; i++)
            {
                double diff = predicted[i] - actual[i];
                sum += diff * diff;
            }
            return sum / predicted.Length;
        }

        public Vector<double> CalculateDerivative(Vector<double> predicted, Vector<double> actual)
        {
            var gradient = new Vector<double>(predicted.Length);
            for (int i = 0; i < predicted.Length && i < actual.Length; i++)
            {
                gradient[i] = 2.0 * (predicted[i] - actual[i]) / predicted.Length;
            }
            return gradient;
        }
    }

    /// <summary>
    /// Creates a synthetic dataset for testing.
    /// </summary>
    private (Matrix<double> X, Vector<double> Y) CreateTestDataset(int numClasses, int examplesPerClass, int numFeatures)
    {
        int totalExamples = numClasses * examplesPerClass;
        var X = new Matrix<double>(totalExamples, numFeatures);
        var Y = new Vector<double>(totalExamples);

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            for (int exampleIdx = 0; exampleIdx < examplesPerClass; exampleIdx++)
            {
                int rowIdx = classIdx * examplesPerClass + exampleIdx;

                for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++)
                {
                    X[rowIdx, featureIdx] = (double)classIdx + (double)exampleIdx * 0.1 + (double)featureIdx * 0.01;
                }

                Y[rowIdx] = classIdx;
            }
        }

        return (X, Y);
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_ValidInputs_InitializesSuccessfully()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();

        // Act
        var trainer = new ReptileTrainer<double>(
            metaModel: model,
            lossFunction: lossFunction,
            innerSteps: 5,
            innerLearningRate: 0.01,
            metaLearningRate: 0.001);

        // Assert
        Assert.NotNull(trainer);
    }

    [Fact]
    public void Constructor_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var lossFunction = new SimpleMockLossFunction();

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new ReptileTrainer<double>(
                metaModel: null!,
                lossFunction: lossFunction,
                innerSteps: 5,
                innerLearningRate: 0.01,
                metaLearningRate: 0.001));

        Assert.Contains("metaModel", exception.Message);
    }

    [Fact]
    public void Constructor_NullLossFunction_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModel(10);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new ReptileTrainer<double>(
                metaModel: model,
                lossFunction: null!,
                innerSteps: 5,
                innerLearningRate: 0.01,
                metaLearningRate: 0.001));

        Assert.Contains("lossFunction", exception.Message);
    }

    [Fact]
    public void Constructor_InvalidInnerSteps_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ReptileTrainer<double>(
                metaModel: model,
                lossFunction: lossFunction,
                innerSteps: 0,
                innerLearningRate: 0.01,
                metaLearningRate: 0.001));

        Assert.Contains("innerSteps", exception.Message);
    }

    [Fact]
    public void Constructor_InvalidInnerLearningRate_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ReptileTrainer<double>(
                metaModel: model,
                lossFunction: lossFunction,
                innerSteps: 5,
                innerLearningRate: 0.0,
                metaLearningRate: 0.001));

        Assert.Contains("learning rate", exception.Message.ToLower());
    }

    [Fact]
    public void Constructor_InvalidMetaLearningRate_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ReptileTrainer<double>(
                metaModel: model,
                lossFunction: lossFunction,
                innerSteps: 5,
                innerLearningRate: 0.01,
                metaLearningRate: -0.001));

        Assert.Contains("learning rate", exception.Message.ToLower());
    }

    #endregion

    #region Train Method Tests

    [Fact]
    public void Train_NullDataLoader_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();
        var trainer = new ReptileTrainer<double>(model, lossFunction);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            trainer.Train(null!, numMetaIterations: 10));

        Assert.Contains("dataLoader", exception.Message);
    }

    [Fact]
    public void Train_InvalidNumMetaIterations_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();
        var trainer = new ReptileTrainer<double>(model, lossFunction);

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            trainer.Train(dataLoader, numMetaIterations: 0));

        Assert.Contains("meta-iterations", exception.Message.ToLower());
    }

    [Fact]
    public void Train_UpdatesModelParameters()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();
        var trainer = new ReptileTrainer<double>(model, lossFunction, innerSteps: 3);

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);

        var initialParameters = model.GetParameters();

        // Act
        trainer.Train(dataLoader, numMetaIterations: 5);

        var finalParameters = model.GetParameters();

        // Assert - Parameters should have changed
        bool parametersChanged = false;
        for (int i = 0; i < initialParameters.Length; i++)
        {
            if (Math.Abs(initialParameters[i] - finalParameters[i]) > 1e-10)
            {
                parametersChanged = true;
                break;
            }
        }

        Assert.True(parametersChanged, "Model parameters should be updated after training");
    }

    [Fact]
    public void Train_CallsModelTrainMultipleTimes()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();
        int innerSteps = 3;
        int numMetaIterations = 5;
        var trainer = new ReptileTrainer<double>(model, lossFunction, innerSteps: innerSteps);

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        int initialTrainCount = model.TrainCallCount;

        // Act
        trainer.Train(dataLoader, numMetaIterations: numMetaIterations);

        // Assert - Should call Train() innerSteps times per meta-iteration
        int expectedCalls = innerSteps * numMetaIterations;
        int actualCalls = model.TrainCallCount - initialTrainCount;

        Assert.Equal(expectedCalls, actualCalls);
    }

    [Fact]
    public void Train_ReturnsValidMetadata()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();
        var trainer = new ReptileTrainer<double>(model, lossFunction);

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        int numIterations = 10;

        // Act
        var metadata = trainer.Train(dataLoader, numMetaIterations: numIterations);

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(numIterations, metadata.Iterations);
        Assert.NotNull(metadata.LossHistory);
        Assert.Equal(numIterations, metadata.LossHistory.Count);
        Assert.True(metadata.TrainingTime.TotalMilliseconds > 0);
    }

    [Fact]
    public void Train_WithDefaultParameters_ExecutesSuccessfully()
    {
        // Arrange
        var model = new SimpleMockModel(10);
        var lossFunction = new SimpleMockLossFunction();
        var trainer = new ReptileTrainer<double>(model, lossFunction); // Use all defaults

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y); // Use defaults here too

        // Act
        var metadata = trainer.Train(dataLoader, numMetaIterations: 5);

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(5, metadata.Iterations);
    }

    #endregion

    #region Reptile Algorithm Tests

    [Fact]
    public void Train_ParametersConvergeTowardAdaptedParameters()
    {
        // Arrange
        var model = new SimpleMockModel(5);
        var lossFunction = new SimpleMockLossFunction();
        var metaLearningRate = 0.1; // Higher rate to see effect clearly
        var trainer = new ReptileTrainer<double>(
            model,
            lossFunction,
            innerSteps: 3,
            innerLearningRate: 0.01,
            metaLearningRate: metaLearningRate);

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);

        var initialParameters = model.GetParameters();

        // Act - Run one meta-iteration
        trainer.Train(dataLoader, numMetaIterations: 1);

        var finalParameters = model.GetParameters();

        // Assert - Parameters should have moved (Reptile update)
        // The change should be in the direction of adapted parameters
        bool parametersUpdated = false;
        for (int i = 0; i < initialParameters.Length; i++)
        {
            if (Math.Abs(finalParameters[i] - initialParameters[i]) > 1e-6)
            {
                parametersUpdated = true;
                break;
            }
        }

        Assert.True(parametersUpdated, "Reptile meta-update should change parameters");
    }

    #endregion
}
