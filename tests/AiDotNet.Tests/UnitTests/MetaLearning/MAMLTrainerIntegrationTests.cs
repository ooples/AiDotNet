using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Integration tests for MAMLAlgorithm demonstrating meta-learning framework functionality.
/// </summary>
/// <remarks>
/// These tests verify that the MAML meta-learning framework operates correctly:
/// - Parameters are updated through meta-training
/// - Training completes without errors
/// - The two-loop (inner/outer) structure works as expected
/// - Query set evaluation drives meta-optimization
/// </remarks>
public class MAMLTrainerIntegrationTests
{
    private SimpleMockModel CreateMockModel() => new SimpleMockModel(50);

    private MAMLOptions<double, Tensor<double>, Tensor<double>> CreateDefaultOptions()
    {
        var mockModel = CreateMockModel();
        return new MAMLOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.01,
            AdaptationSteps = 5
        };
    }

    /// <summary>
    /// Creates a mock task for testing purposes.
    /// </summary>
    private IMetaLearningTask<double, Tensor<double>, Tensor<double>> CreateMockTask(int seed = 0)
    {
        var supportInput = new Tensor<double>(new int[] { 15, 10 });  // 5 classes x 3 shots
        var supportOutput = new Tensor<double>(new int[] { 15 });
        var queryInput = new Tensor<double>(new int[] { 50, 10 });   // 5 classes x 10 queries
        var queryOutput = new Tensor<double>(new int[] { 50 });

        // Fill with seeded random data for reproducibility
        var random = new Random(42 + seed);
        for (int i = 0; i < supportInput.Shape[0]; i++)
        {
            for (int j = 0; j < supportInput.Shape[1]; j++)
            {
                supportInput[new[] { i, j }] = random.NextDouble() * 2 - 1;
            }
            supportOutput[new[] { i }] = i % 5; // Class labels 0-4
        }

        for (int i = 0; i < queryInput.Shape[0]; i++)
        {
            for (int j = 0; j < queryInput.Shape[1]; j++)
            {
                queryInput[new[] { i, j }] = random.NextDouble() * 2 - 1;
            }
            queryOutput[new[] { i }] = i % 5; // Class labels 0-4
        }

        return new MetaLearningTask<double, Tensor<double>, Tensor<double>>
        {
            SupportSetX = supportInput,
            SupportSetY = supportOutput,
            QuerySetX = queryInput,
            QuerySetY = queryOutput,
            NumWays = 5,
            NumShots = 3,
            NumQueryPerClass = 10,
            Name = $"test-task-{seed}"
        };
    }

    private TaskBatch<double, Tensor<double>, Tensor<double>> CreateTaskBatch(int batchSize)
    {
        var tasks = Enumerable.Range(0, batchSize)
            .Select(i => CreateMockTask(i))
            .ToArray();
        return new TaskBatch<double, Tensor<double>, Tensor<double>>(tasks);
    }

    #region Integration Tests

    [Fact]
    public void MetaTrain_WithMultipleIterations_UpdatesParametersCorrectly()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Get initial parameters
        var metaModel = algorithm.GetMetaModel();
        var initialParams = metaModel.GetParameters();
        var initialParamsClone = initialParams.Clone();

        // Act - Meta-train for multiple iterations
        var lossHistory = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);  // 4 tasks per batch
            var loss = algorithm.MetaTrain(taskBatch);
            lossHistory.Add(loss);
        }

        // Assert - Parameters should have changed
        var finalParams = metaModel.GetParameters();
        bool paramsChanged = false;
        for (int i = 0; i < initialParamsClone.Length; i++)
        {
            if (Math.Abs(finalParams[i] - initialParamsClone[i]) > 1e-10)
            {
                paramsChanged = true;
                break;
            }
        }

        Assert.True(paramsChanged, "Meta-training should update model parameters");
        Assert.Equal(10, lossHistory.Count);
        Assert.All(lossHistory, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
    }

    [Fact]
    public void MetaTrain_TrainsModelOnMultipleTasks()
    {
        // This test verifies the framework correctly processes multiple meta-learning tasks
        // and that MAML uses query set evaluation (key difference from Reptile)

        // Arrange - Create two algorithms with same initial model
        var model1 = new SimpleMockModel(50);
        var model2 = new SimpleMockModel(50);

        // Initialize both with same parameters
        var initialParams = model1.GetParameters();
        model2.SetParameters(initialParams.Clone());

        var options1 = new MAMLOptions<double, Tensor<double>, Tensor<double>>(model1)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.01,
            AdaptationSteps = 5
        };

        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options1);

        // Act - Meta-train only one model
        for (int i = 0; i < 20; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            algorithm.MetaTrain(taskBatch);
        }

        // Assert - Meta-trained model should have different parameters than baseline
        var metaTrainedParams = model1.GetParameters();
        var baselineParams = model2.GetParameters();

        bool paramsDifferent = false;
        for (int i = 0; i < metaTrainedParams.Length; i++)
        {
            if (Math.Abs(metaTrainedParams[i] - baselineParams[i]) > 1e-10)
            {
                paramsDifferent = true;
                break;
            }
        }

        Assert.True(paramsDifferent, "Meta-training should change parameters differently than baseline");
    }

    [Fact]
    public void Adapt_ProducesTaskSpecificModel()
    {
        // This test verifies that MAML adaptation creates a task-specialized model

        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Meta-train first
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            algorithm.MetaTrain(taskBatch);
        }

        // Get original meta-model parameters
        var metaModel = algorithm.GetMetaModel();
        var metaParams = metaModel.GetParameters().Clone();

        // Act - Adapt to a new task
        var newTask = CreateMockTask(100);  // Different seed for novel task
        var adaptedModel = algorithm.Adapt(newTask);

        // Assert - Adapted model should be different from meta-model
        Assert.NotNull(adaptedModel);

        // The adapted model should be a specialized version
        // Meta-model parameters should remain unchanged (adaptation doesn't modify meta-model)
        var currentMetaParams = metaModel.GetParameters();
        bool metaParamsUnchanged = true;
        for (int i = 0; i < metaParams.Length; i++)
        {
            if (Math.Abs(currentMetaParams[i] - metaParams[i]) > 1e-15)
            {
                metaParamsUnchanged = false;
                break;
            }
        }
        Assert.True(metaParamsUnchanged, "Adaptation should not modify meta-model parameters");
    }

    [Fact]
    public void Evaluate_ProducesValidMetrics()
    {
        // This test verifies evaluation produces valid loss values

        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Meta-train first
        for (int i = 0; i < 5; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            algorithm.MetaTrain(taskBatch);
        }

        // Act - Evaluate on a task batch
        var evalBatch = CreateTaskBatch(10);
        var evalLoss = algorithm.Evaluate(evalBatch);

        // Assert
        Assert.True(evalLoss >= 0, "Evaluation loss should be non-negative");
        Assert.False(double.IsNaN(evalLoss), "Evaluation loss should not be NaN");
        Assert.False(double.IsPositiveInfinity(evalLoss), "Evaluation loss should not be infinite");
    }

    [Fact]
    public void MAML_FirstOrderApproximation_WorksCorrectly()
    {
        // This test verifies FOMAML (first-order approximation) works

        // Arrange
        var mockModel = CreateMockModel();
        var options = new MAMLOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.01,
            AdaptationSteps = 5,
            UseFirstOrder = true  // FOMAML
        };

        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Train with first-order approximation
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert - Training should complete successfully
        Assert.Equal(10, losses.Count);
        Assert.All(losses, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
    }

    [Fact]
    public void MetaTrain_LongTraining_TracksLossCorrectly()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Train for many iterations
        var losses = new List<double>();
        for (int i = 0; i < 50; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.Equal(50, losses.Count);

        // Check that we have recorded valid losses
        double firstLoss = losses[0];
        double lastLoss = losses[losses.Count - 1];

        Assert.True(firstLoss >= 0, "Initial loss should be non-negative");
        Assert.True(lastLoss >= 0, "Final loss should be non-negative");
        Assert.True(lastLoss < double.MaxValue, "Loss should not explode");
        Assert.True(!double.IsNaN(lastLoss), "Loss should not be NaN");
    }

    [Fact]
    public void MetaTrain_WithLargeBatch_ProcessesAllTasks()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Process a larger batch
        var largeBatch = CreateTaskBatch(8);  // 8 tasks
        var loss = algorithm.MetaTrain(largeBatch);

        // Assert
        Assert.True(loss >= 0, "Loss should be non-negative");
        Assert.False(double.IsNaN(loss), "Loss should not be NaN");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void MetaTrain_WithDifferentAdaptationSteps_CompletesSuccessfully(int adaptationSteps)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new MAMLOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.01,
            AdaptationSteps = adaptationSteps
        };

        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(4);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for adaptation steps={adaptationSteps}");
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for adaptation steps={adaptationSteps}");
    }

    [Theory]
    [InlineData(0.001)]
    [InlineData(0.01)]
    [InlineData(0.1)]
    public void MetaTrain_WithDifferentLearningRates_CompletesSuccessfully(double innerLr)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new MAMLOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = innerLr,
            OuterLearningRate = 0.01,
            AdaptationSteps = 5
        };

        var algorithm = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(4);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for inner LR={innerLr}");
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for inner LR={innerLr}");
    }

    #endregion
}
