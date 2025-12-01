using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using AiDotNet.Models.Results;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

// Type alias for cleaner test code
using ReptileTrainerDouble = ReptileTrainer<double, Tensor<double>, Tensor<double>>;
using SimpleMockModelDouble = SimpleMockModel;

/// <summary>
/// Integration tests for ReptileTrainer demonstrating meta-learning framework functionality.
/// </summary>
/// <remarks>
/// These tests verify that the Reptile meta-learning framework operates correctly:
/// - Parameters are updated through meta-training
/// - Training completes without errors
/// - Metadata is properly tracked
/// - The two-loop (inner/outer) structure works as expected
/// </remarks>
public class ReptileTrainerIntegrationTests
{
    #region Helper Methods

    /// <summary>
    /// Generates a classification dataset where features are derived from sine waves with task-specific amplitudes and phases.
    /// Labels are discrete class indices (0 to numTasks-1) for N-way episodic classification tasks.
    /// NOTE: This is classification, not regression - Y contains class labels, not continuous sine values.
    /// </summary>
    private (Matrix<double> X, Vector<double> Y) GenerateSineWaveFeaturesDataset(
        int numTasks,
        int samplesPerTask,
        double minX = 0.0,
        double maxX = 2 * Math.PI)
    {
        var random = new Random(42);
        int totalSamples = numTasks * samplesPerTask;
        var X = new Matrix<double>(totalSamples, 1);
        var Y = new Vector<double>(totalSamples);

        for (int taskIdx = 0; taskIdx < numTasks; taskIdx++)
        {
            // Random amplitude and phase for each task/class
            double amplitude = 0.5 + random.NextDouble(); // Range: [0.5, 1.5]
            double phase = random.NextDouble() * 2 * Math.PI; // Range: [0, 2π]

            for (int sampleIdx = 0; sampleIdx < samplesPerTask; sampleIdx++)
            {
                int rowIdx = taskIdx * samplesPerTask + sampleIdx;

                // Random x value
                double x = minX + random.NextDouble() * (maxX - minX);

                // Feature: sine wave value with task-specific amplitude and phase
                double sineValue = amplitude * Math.Sin(x + phase);

                X[rowIdx, 0] = sineValue;
                Y[rowIdx] = taskIdx; // Class label (0 to numTasks-1)
            }
        }

        return (X, Y);
    }

    /// <summary>
    /// Generates a single classification task with features based on sine wave patterns.
    /// </summary>
    private (Tensor<double> X, Tensor<double> Y) GenerateSineWaveTask(
        double amplitude,
        double phase,
        int classLabel,
        int numSamples,
        double minX = 0.0,
        double maxX = 2 * Math.PI,
        int? seed = null)
    {
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var X = new Tensor<double>(new[] { numSamples, 1 });
        var Y = new Tensor<double>(new[] { numSamples, 1 });

        for (int i = 0; i < numSamples; i++)
        {
            double x = minX + random.NextDouble() * (maxX - minX);
            double sineValue = amplitude * Math.Sin(x + phase);

            X[new[] { i, 0 }] = sineValue;
            Y[new[] { i, 0 }] = classLabel; // Classification label
        }

        return (X, Y);
    }

    /// <summary>
    /// Computes the mean squared error between predictions and targets.
    /// </summary>
    private double ComputeMSE(Tensor<double> predictions, Tensor<double> targets)
    {
        double sum = 0;
        int count = predictions.Shape[0];

        for (int i = 0; i < count; i++)
        {
            double diff = predictions[new[] { i, 0 }] - targets[new[] { i, 0 }];
            sum += diff * diff;
        }

        return sum / count;
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void MetaTrain_WithEpisodicData_UpdatesParametersCorrectly()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();

        // Generate meta-training dataset with 20 tasks
        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);

        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            datasetX: X,
            datasetY: Y,
            nWay: 5,       // 5 different tasks per meta-batch
            kShot: 5,      // 5 samples to learn from
            queryShots: 10, // 10 samples to evaluate
            seed: 42);

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10,
            metaBatchSize: 1,
            numMetaIterations: 50);
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        // Get initial parameters
        var initialParams = model.GetParameters();

        // Act - Meta-train for 50 iterations (as specified in requirements)
        var result = trainer.Train();

        // Assert - Parameters should have changed
        var finalParams = model.GetParameters();
        bool paramsChanged = false;
        for (int i = 0; i < initialParams.Length; i++)
        {
            if (Math.Abs(finalParams[i] - initialParams[i]) > 1e-6)
            {
                paramsChanged = true;
                break;
            }
        }

        Assert.True(paramsChanged, "Meta-training should update model parameters");

        // Assert - Training should complete successfully
        Assert.NotNull(result);
        Assert.Equal(50, result.TotalIterations);
        Assert.NotNull(result.LossHistory);
        Assert.Equal(50, result.LossHistory.Length);
    }

    [Fact]
    public void MetaTrain_TrainsModelOnMultipleTasks()
    {
        // This test verifies the framework correctly processes multiple meta-learning tasks

        // Arrange - Create two identical models
        var metaTrainedModel = new SimpleMockModelDouble(10);
        var baselineModel = new SimpleMockModelDouble(10);

        // Make sure they start with the same parameters
        var initialParams = metaTrainedModel.GetParameters();
        baselineModel.SetParameters(initialParams);

        var lossFunction = new MeanSquaredErrorLoss<double>();

        // Generate meta-training dataset
        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        // Meta-train only one model
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10,
            metaBatchSize: 1,
            numMetaIterations: 100);
        var trainer = new ReptileTrainerDouble(
            metaModel: metaTrainedModel,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        // Act
        var result = trainer.Train();

        // Assert - Meta-training should process all iterations
        Assert.NotNull(result);
        Assert.Equal(100, result.TotalIterations);

        // Meta-trained model should have different parameters than baseline
        var metaTrainedParams = metaTrainedModel.GetParameters();
        var baselineParams = baselineModel.GetParameters();

        bool paramsDifferent = false;
        for (int i = 0; i < metaTrainedParams.Length; i++)
        {
            if (Math.Abs(metaTrainedParams[i] - baselineParams[i]) > 1e-6)
            {
                paramsDifferent = true;
                break;
            }
        }

        Assert.True(paramsDifferent, "Meta-training should change parameters differently than baseline");
    }

    [Fact]
    public void MetaTrain_LongTraining_TracksMetricsCorrectly()
    {
        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();

        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10,
            metaBatchSize: 1,
            numMetaIterations: 100);
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        // Act - Train for 100 iterations to verify metric tracking
        var result = trainer.Train();

        // Assert - Result should be properly populated
        Assert.NotNull(result.LossHistory);
        Assert.Equal(100, result.LossHistory.Length);
        Assert.NotNull(result.AccuracyHistory);
        Assert.Equal(100, result.AccuracyHistory.Length);

        // Check that we have recorded losses
        double firstLoss = result.LossHistory[0];
        double lastLoss = result.LossHistory[result.LossHistory.Length - 1];

        Assert.True(firstLoss >= 0, "Initial loss should be non-negative");
        Assert.True(lastLoss >= 0, "Final loss should be non-negative");

        // Verify system produces reasonable values
        Assert.True(lastLoss < double.MaxValue, "Loss should not explode");
        Assert.True(!double.IsNaN(lastLoss), "Loss should not be NaN");
    }

    [Fact]
    public void MetaTrain_CompletesRequiredIterations()
    {
        // Test specifically for the requirement: "50+ meta-iterations"

        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();

        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10,
            metaBatchSize: 1,
            numMetaIterations: 50);
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        // Act - Run exactly 50 meta-iterations as required
        var result = trainer.Train();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(50, result.TotalIterations);
        Assert.Equal(50, result.LossHistory.Length);
        Assert.Equal(50, result.AccuracyHistory.Length);

        // Verify training completed without errors
        Assert.True(result.TrainingTime.TotalMilliseconds > 0);
        Assert.False(double.IsNaN(result.FinalLoss), "Final loss should not be NaN");
        Assert.False(double.IsPositiveInfinity(result.FinalLoss), "Final loss should not be infinity");
    }

    #endregion
}
