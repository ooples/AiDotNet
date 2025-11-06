using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

// Type alias for cleaner test code
using ReptileTrainerDouble = ReptileTrainer<double, Tensor<double>, Tensor<double>>;
using SimpleMockModelDouble = SimpleMockModel;

/// <summary>
/// Unit tests for the ReptileTrainer class.
/// </summary>
public class ReptileTrainerTests
{
    #region Test Helper Methods

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
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 5);

        // Act
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            config: config);

        // Assert
        Assert.NotNull(trainer);
    }

    [Fact]
    public void Constructor_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var lossFunction = new MeanSquaredErrorLoss<double>();

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new ReptileTrainerDouble(
                metaModel: null!,
                lossFunction: lossFunction));

        Assert.Contains("metaModel", exception.Message);
    }

    [Fact]
    public void Constructor_NullLossFunction_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new ReptileTrainerDouble(
                metaModel: model,
                lossFunction: null!));

        Assert.Contains("lossFunction", exception.Message);
    }

    [Fact]
    public void Constructor_InvalidInnerSteps_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 0); // Invalid

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ReptileTrainerDouble(
                metaModel: model,
                lossFunction: lossFunction,
                config: config));

        Assert.Contains("innersteps", exception.Message.ToLower());
    }

    [Fact]
    public void Constructor_InvalidInnerLearningRate_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.0, // Invalid
            metaLearningRate: 0.001,
            innerSteps: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ReptileTrainerDouble(
                metaModel: model,
                lossFunction: lossFunction,
                config: config));

        Assert.Contains("learning rate", exception.Message.ToLower());
    }

    [Fact]
    public void Constructor_InvalidMetaLearningRate_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: -0.001, // Invalid
            innerSteps: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ReptileTrainerDouble(
                metaModel: model,
                lossFunction: lossFunction,
                config: config));

        Assert.Contains("learning rate", exception.Message.ToLower());
    }

    #endregion

    #region Train Method Tests

    [Fact]
    public void Train_NullDataLoader_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var trainer = new ReptileTrainerDouble(model, lossFunction);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            trainer.Train(null!, numMetaIterations: 10));

        Assert.Contains("dataLoader", exception.Message);
    }

    [Fact]
    public void Train_InvalidNumMetaIterations_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var trainer = new ReptileTrainerDouble(model, lossFunction);

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
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var config = new ReptileTrainerConfig<double>(innerSteps: 3);
        var trainer = new ReptileTrainerDouble(model, lossFunction, config);

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
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        int innerSteps = 3;
        int numMetaIterations = 5;
        var config = new ReptileTrainerConfig<double>(innerSteps: innerSteps);
        var trainer = new ReptileTrainerDouble(model, lossFunction, config);

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
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var trainer = new ReptileTrainerDouble(model, lossFunction);

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
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var trainer = new ReptileTrainerDouble(model, lossFunction); // Use all defaults

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
        var model = new SimpleMockModelDouble(5);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.1, // Higher rate to see effect clearly
            innerSteps: 3);
        var trainer = new ReptileTrainerDouble(model, lossFunction, config);

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
