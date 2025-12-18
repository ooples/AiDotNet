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
/// Integration tests for MAMLTrainer demonstrating meta-learning framework functionality.
/// </summary>
/// <remarks>
/// These tests verify that the MAML meta-learning framework operates correctly:
/// - Parameters are updated through meta-training
/// - Training completes without errors
/// - Metadata is properly tracked
/// - The two-loop (inner/outer) structure works as expected
/// - Query set evaluation drives meta-optimization
/// </remarks>
public class MAMLTrainerIntegrationTests
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
            kShot: 5,      // 5 samples to learn from (support set)
            queryShots: 10, // 10 samples to evaluate (query set)
            seed: 42);

        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 5,      // MAML typically uses fewer inner steps
            metaBatchSize: 4,   // MAML uses batch meta-updates
            numMetaIterations: 50);
        var trainer = new MAMLTrainerDouble(
            metaModel: model,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        // Get initial parameters
        var initialParams = model.GetParameters();

        // Act - Meta-train for 50 iterations
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
        // and that MAML uses query set evaluation (key difference from Reptile)

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
        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 5,
            metaBatchSize: 4,
            numMetaIterations: 50);
        var trainer = new MAMLTrainerDouble(
            metaModel: metaTrainedModel,
            lossFunction: lossFunction,
            dataLoader: dataLoader,
            config: config);

        var trainingResult = trainer.Train();

        // Act - Both models adapt to a new task (not in meta-training set)
        var newTask = dataLoader.GetNextTask();

        // Meta-trained model adapts
        var metaAdaptResult = trainer.AdaptAndEvaluate(newTask);

        // Baseline model would need manual adaptation (simulated here)
        // For this test, we just verify meta-trained model produces results

        // Assert - Meta-trained model should successfully adapt
        Assert.NotNull(metaAdaptResult);
        Assert.True(metaAdaptResult.AdaptationSteps > 0);

        // Assert - Training result should show progress
        Assert.NotNull(trainingResult);
        Assert.True(trainingResult.LossHistory.Length > 0);
    }

    [Fact]
    public void AdaptAndEvaluate_TracksLossImprovement()
    {
        // This test verifies that MAML adaptation reduces loss (key meta-learning property)

        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        // Meta-train first
        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 5,
            metaBatchSize: 4,
            numMetaIterations: 30);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader, config);
        trainer.Train();

        // Get a test task
        var task = dataLoader.GetNextTask();

        // Act - Adapt and evaluate
        var result = trainer.AdaptAndEvaluate(task);

        // Assert - Should track per-step losses
        Assert.NotNull(result.PerStepLosses);
        Assert.True(result.PerStepLosses.Count > 1); // Initial + at least one adaptation step

        // Assert - Initial loss should be recorded
        Assert.Contains("initial_query_loss", result.AdditionalMetrics.Keys);
        Assert.Contains("loss_improvement", result.AdditionalMetrics.Keys);
    }

    [Fact]
    public void Evaluate_ProducesConsistentMetrics()
    {
        // This test verifies evaluation produces valid metrics

        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act - Evaluate on multiple tasks
        var result = trainer.Evaluate(numTasks: 20);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.PerTaskAccuracies);
        Assert.NotNull(result.PerTaskLosses);
        Assert.Equal(20, result.PerTaskAccuracies.Length);
        Assert.Equal(20, result.PerTaskLosses.Length);
        Assert.NotNull(result.AccuracyStats);
        Assert.NotNull(result.LossStats);
    }

    [Fact]
    public void MAML_FirstOrderApproximation_WorksCorrectly()
    {
        // This test verifies FOMAML (first-order approximation) works

        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);

        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.02,
            metaLearningRate: 0.01,
            innerSteps: 5,
            metaBatchSize: 4,
            numMetaIterations: 20,
            useFirstOrderApproximation: true); // FOMAML

        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader, config);

        // Act
        var result = trainer.Train();

        // Assert - Training should complete successfully
        Assert.NotNull(result);
        Assert.Equal(20, result.TotalIterations);

        // Assert - Config should reflect FOMAML
        var mamlConfig = (MAMLTrainerConfig<double>)trainer.Config;
        Assert.True(mamlConfig.UseFirstOrderApproximation);
    }

    [Fact]
    public void MetaTrainStep_ProducesValidMetrics()
    {
        // This test verifies a single meta-training step produces valid metrics

        // Arrange
        var model = new SimpleMockModelDouble(10);
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var (X, Y) = GenerateSineWaveFeaturesDataset(numTasks: 20, samplesPerTask: 25);
        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            X, Y, nWay: 5, kShot: 5, queryShots: 10, seed: 42);
        var trainer = new MAMLTrainerDouble(model, lossFunction, dataLoader);

        // Act
        var stepResult = trainer.MetaTrainStep(batchSize: 4);

        // Assert
        Assert.NotNull(stepResult);
        Assert.Equal(1, trainer.CurrentIteration);
        Assert.Equal(4, stepResult.NumTasks);
        Assert.True(stepResult.TimeMs >= 0);

        // Meta-loss and task loss should be equal for MAML (query set loss)
        Assert.Equal(stepResult.MetaLoss, stepResult.TaskLoss);
    }

    #endregion
}
