using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using AiDotNet.Models.Results;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

// Type alias for cleaner test code
using MAMLTrainerDouble = MAMLTrainer<double, Tensor<double>, Tensor<double>>;
using SimpleMockModelDouble = SimpleMockModel;

/// <summary>
/// Unit tests for the MAMLTrainer class.
/// </summary>
public class MAMLTrainerTests
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
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 5);

        // Act
        var trainer = new MAMLTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        // Assert
        Assert.NotNull(trainer);
        Assert.NotNull(trainer.BaseModel);
        Assert.NotNull(trainer.Config);
        Assert.Equal(0, trainer.CurrentIteration);
    }

    [Fact]
    public void Constructor_DefaultConfig_UsesMAMLDefaults()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        // Act
        var trainer = new MAMLTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            dataLoader: dataLoader);

        // Assert
        Assert.NotNull(trainer);
        Assert.IsType<MAMLTrainerConfig<double>>(trainer.Config);
        var config = (MAMLTrainerConfig<double>)trainer.Config;
        Assert.Equal(0.01, config.InnerLearningRate);
        Assert.Equal(0.001, config.MetaLearningRate);
        Assert.Equal(5, config.InnerSteps);
        Assert.Equal(4, config.MetaBatchSize);
        Assert.True(config.UseFirstOrderApproximation); // FOMAML by default
    }

    [Fact]
    public void Constructor_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new MAMLTrainerDouble(
                metaModel: null!,
                lossFunction: lossFunction,
                dataLoader: dataLoader));

        Assert.Contains("metaModel", exception.Message);
    }

    [Fact]
    public void Constructor_NullLossFunction_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new MAMLTrainerDouble(
                metaModel: model,
                lossFunction: null!,
                dataLoader: dataLoader));

        Assert.Contains("lossFunction", exception.Message);
    }

    [Fact]
    public void Constructor_NullDataLoader_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new MAMLTrainerDouble(
                metaModel: model,
                lossFunction: lossFunction,
                dataLoader: null!));

        Assert.Contains("dataLoader", exception.Message);
    }

    [Fact]
    public void Constructor_InvalidInnerSteps_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 0); // Invalid

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new MAMLTrainerDouble(
                metaModel: model,
                lossFunction: lossFunction,
                dataLoader: dataLoader,
                config: config));

        Assert.Contains("Configuration validation failed", exception.Message);
    }

    #endregion

    #region MetaTrainStep Tests

    [Fact]
    public void MetaTrainStep_ValidBatchSize_UpdatesParameters()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        var initialParams = model.GetParameters();

        // Act
        var result = trainer.MetaTrainStep(batchSize: 2);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(1, trainer.CurrentIteration);
        Assert.Equal(2, result.NumTasks);

        var finalParams = model.GetParameters();
        Assert.NotEqual(initialParams[0], finalParams[0]); // Parameters should have changed
    }

    [Fact]
    public void MetaTrainStep_ZeroBatchSize_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => trainer.MetaTrainStep(batchSize: 0));
    }

    [Fact]
    public void MetaTrainStep_NegativeBatchSize_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => trainer.MetaTrainStep(batchSize: -1));
    }

    #endregion

    #region AdaptAndEvaluate Tests

    [Fact]
    public void AdaptAndEvaluate_ValidTask_ReturnsResult()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        var task = dataLoader.GetNextTask();

        // Act
        var result = trainer.AdaptAndEvaluate(task);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.AdaptationSteps > 0);
        Assert.NotNull(result.PerStepLosses);
        Assert.True(result.PerStepLosses.Count > 0);
    }

    [Fact]
    public void AdaptAndEvaluate_NullTask_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => trainer.AdaptAndEvaluate(null!));
    }

    #endregion

    #region Train Tests

    [Fact]
    public void Train_CompletesSuccessfully()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);
        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 3,
            metaBatchSize: 2,
            numMetaIterations: 5); // Small for fast test

        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader, config);

        // Act
        var result = trainer.Train();

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.LossHistory);
        Assert.NotNull(result.AccuracyHistory);
        Assert.Equal(5, result.LossHistory.Length);
        Assert.Equal(5, result.AccuracyHistory.Length);
    }

    #endregion

    #region Evaluate Tests

    [Fact]
    public void Evaluate_ValidNumberOfTasks_ReturnsResult()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act
        var result = trainer.Evaluate(numTasks: 10);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.TaskAccuracies);
        Assert.NotNull(result.TaskLosses);
        Assert.Equal(10, result.TaskAccuracies.Length);
        Assert.Equal(10, result.TaskLosses.Length);
    }

    [Fact]
    public void Evaluate_ZeroTasks_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => trainer.Evaluate(numTasks: 0));
    }

    #endregion

    #region Save/Load/Reset Tests

    [Fact]
    public void Save_ValidPath_DoesNotThrow()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act & Assert - just verify it doesn't throw
        // (SimpleMockModel has empty Save implementation)
        trainer.Save("test_model.bin");
    }

    [Fact]
    public void Save_NullPath_ThrowsArgumentException()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => trainer.Save(null!));
    }

    [Fact]
    public void Reset_ResetsIterationCounter()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 10);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: 42);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Perform some iterations
        trainer.MetaTrainStep(batchSize: 2);
        trainer.MetaTrainStep(batchSize: 2);
        Assert.Equal(2, trainer.CurrentIteration);

        // Act
        trainer.Reset();

        // Assert
        Assert.Equal(0, trainer.CurrentIteration);
    }

    #endregion

    #region Configuration-Specific Tests

    [Fact]
    public void Config_FirstOrderApproximation_DefaultsToTrue()
    {
        // Arrange
        var config = new MAMLTrainerConfig<double>();

        // Assert
        Assert.True(config.UseFirstOrderApproximation);
    }

    [Fact]
    public void Config_CanSetFullMAML()
    {
        // Arrange
        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 5,
            metaBatchSize: 4,
            numMetaIterations: 1000,
            useFirstOrderApproximation: false); // Full MAML

        // Assert
        Assert.False(config.UseFirstOrderApproximation);
    }

    #endregion
}
