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
    /// Generates a dataset of sine wave tasks with different amplitudes and phases.
    /// Each task is a different sine wave function.
    /// </summary>
    private (Matrix<double> X, Vector<double> Y) GenerateSineWaveDataset(
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
            // Random amplitude and phase for each task
            double amplitude = 0.5 + random.NextDouble(); // Range: [0.5, 1.5]
            double phase = random.NextDouble() * 2 * Math.PI; // Range: [0, 2π]

            for (int sampleIdx = 0; sampleIdx < samplesPerTask; sampleIdx++)
            {
                int rowIdx = taskIdx * samplesPerTask + sampleIdx;

                // Random x value
                double x = minX + random.NextDouble() * (maxX - minX);

                // y = amplitude * sin(x + phase)
                double y = amplitude * Math.Sin(x + phase);

                X[rowIdx, 0] = x;
                Y[rowIdx] = taskIdx; // Use task index as "class" for episodic loader
            }
        }

        return (X, Y);
    }

    /// <summary>
    /// Generates a single sine wave task for testing adaptation.
    /// </summary>
    private (Tensor<double> X, Tensor<double> Y) GenerateSineWaveTask(
        double amplitude,
        double phase,
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
            double y = amplitude * Math.Sin(x + phase);

            X[new[] { i, 0 }] = x;
            Y[new[] { i, 0 }] = y;
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
        var (X, Y) = GenerateSineWaveDataset(numTasks: 20, samplesPerTask: 25);

        var dataLoader = new UniformEpisodicDataLoader<double>(
            datasetX: X,
            datasetY: Y,
            nWay: 5,       // 5 different tasks per meta-batch
            kShot: 5,      // 5 samples to learn from
            queryShots: 10, // 10 samples to evaluate
            seed: 42);

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10);
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            config: config);

        // Get initial parameters
        var initialParams = model.GetParameters();

        // Act - Meta-train for 50 iterations (as specified in requirements)
        var metadata = trainer.Train(dataLoader, numMetaIterations: 50);

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
        Assert.NotNull(metadata);
        Assert.Equal(50, metadata.Iterations);
        Assert.NotNull(metadata.LossHistory);
        Assert.Equal(50, metadata.LossHistory.Count);
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
        var (X, Y) = GenerateSineWaveDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        // Meta-train only one model
        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10);
        var trainer = new ReptileTrainerDouble(
            metaModel: metaTrainedModel,
            lossFunction: lossFunction,
            config: config);

        // Act
        var metadata = trainer.Train(dataLoader, numMetaIterations: 100);

        // Assert - Meta-training should process all iterations
        Assert.NotNull(metadata);
        Assert.Equal(100, metadata.Iterations);

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

        var (X, Y) = GenerateSineWaveDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10);
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            config: config);

        // Act - Train for 100 iterations to verify metric tracking
        var metadata = trainer.Train(dataLoader, numMetaIterations: 100);

        // Assert - Metadata should be properly populated
        Assert.NotNull(metadata.LossHistory);
        Assert.Equal(100, metadata.LossHistory.Count);
        Assert.NotNull(metadata.AccuracyHistory);
        Assert.Equal(100, metadata.AccuracyHistory.Count);

        // Check that we have recorded losses
        double firstLoss = metadata.LossHistory[0];
        double lastLoss = metadata.LossHistory[metadata.LossHistory.Count - 1];

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

        var (X, Y) = GenerateSineWaveDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 10);
        var trainer = new ReptileTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            config: config);

        // Act - Run exactly 50 meta-iterations as required
        var metadata = trainer.Train(dataLoader, numMetaIterations: 50);

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(50, metadata.Iterations);
        Assert.Equal(50, metadata.LossHistory.Count);
        Assert.Equal(50, metadata.AccuracyHistory.Count);

        // Verify training completed without errors
        Assert.True(metadata.TrainingTime.TotalMilliseconds > 0);
        Assert.False(double.IsNaN(metadata.FinalLoss), "Final loss should not be NaN");
        Assert.False(double.IsPositiveInfinity(metadata.FinalLoss), "Final loss should not be infinity");
    }

    #endregion
}
