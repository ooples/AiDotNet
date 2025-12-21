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
/// Integration tests for ReptileAlgorithm demonstrating meta-learning framework functionality.
/// </summary>
/// <remarks>
/// These tests verify that the Reptile meta-learning framework operates correctly:
/// - Parameters are updated through meta-training
/// - Training completes without errors
/// - The two-loop (inner/outer) structure works as expected
/// - Reptile's simpler approach (no query set evaluation) still produces good meta-models
/// </remarks>
public class ReptileTrainerIntegrationTests
{
    private SimpleMockModel CreateMockModel() => new SimpleMockModel(50);

    private ReptileOptions<double, Tensor<double>, Tensor<double>> CreateDefaultOptions()
    {
        var mockModel = CreateMockModel();
        return new ReptileOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 1.0,  // Reptile typically uses higher outer LR (interpolation factor)
            AdaptationSteps = 10
        };
    }

    /// <summary>
    /// Creates a mock task for testing purposes.
    /// </summary>
    private IMetaLearningTask<double, Tensor<double>, Tensor<double>> CreateMockTask(int seed = 0)
    {
        var supportInput = new Tensor<double>(new int[] { 25, 10 });  // 5 classes x 5 shots
        var supportOutput = new Tensor<double>(new int[] { 25 });
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
            NumShots = 5,
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
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Get initial parameters
        var metaModel = algorithm.GetMetaModel();
        var initialParams = metaModel.GetParameters();
        var initialParamsClone = initialParams.Clone();

        // Act - Meta-train for 50 iterations (as specified in requirements)
        var lossHistory = new List<double>();
        for (int i = 0; i < 50; i++)
        {
            var taskBatch = CreateTaskBatch(1);  // Reptile typically uses single task per iteration
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
        Assert.Equal(50, lossHistory.Count);
        Assert.All(lossHistory, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
    }

    [Fact]
    public void MetaTrain_TrainsModelOnMultipleTasks()
    {
        // This test verifies the framework correctly processes multiple meta-learning tasks

        // Arrange - Create two algorithms with same initial model
        var model1 = new SimpleMockModel(50);
        var model2 = new SimpleMockModel(50);

        // Initialize both with same parameters
        var initialParams = model1.GetParameters();
        model2.SetParameters(initialParams.Clone());

        var options1 = new ReptileOptions<double, Tensor<double>, Tensor<double>>(model1)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 1.0,
            AdaptationSteps = 10
        };

        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options1);

        // Act - Meta-train only one model for 100 iterations
        for (int i = 0; i < 100; i++)
        {
            var taskBatch = CreateTaskBatch(1);
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
    public void MetaTrain_LongTraining_TracksMetricsCorrectly()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Train for 100 iterations to verify metric tracking
        var lossHistory = new List<double>();
        for (int i = 0; i < 100; i++)
        {
            var taskBatch = CreateTaskBatch(1);
            var loss = algorithm.MetaTrain(taskBatch);
            lossHistory.Add(loss);
        }

        // Assert
        Assert.Equal(100, lossHistory.Count);

        // Check that we have recorded valid losses
        double firstLoss = lossHistory[0];
        double lastLoss = lossHistory[lossHistory.Count - 1];

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
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Run exactly 50 meta-iterations as required
        var losses = new List<double>();
        for (int i = 0; i < 50; i++)
        {
            var taskBatch = CreateTaskBatch(1);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.Equal(50, losses.Count);
        Assert.All(losses, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
        Assert.All(losses, loss => Assert.False(double.IsPositiveInfinity(loss), "Loss should not be infinity"));
    }

    [Fact]
    public void Adapt_ProducesTaskSpecificModel()
    {
        // This test verifies that Reptile adaptation creates a task-specialized model

        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Meta-train first
        for (int i = 0; i < 20; i++)
        {
            var taskBatch = CreateTaskBatch(1);
            algorithm.MetaTrain(taskBatch);
        }

        // Get original meta-model parameters
        var metaModel = algorithm.GetMetaModel();
        var metaParams = metaModel.GetParameters().Clone();

        // Act - Adapt to a new task
        var newTask = CreateMockTask(100);  // Different seed for novel task
        var adaptedModel = algorithm.Adapt(newTask);

        // Assert - Adapted model should be created
        Assert.NotNull(adaptedModel);

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
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Meta-train first
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(1);
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

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(20)]
    public void MetaTrain_WithDifferentInnerSteps_CompletesSuccessfully(int innerSteps)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new ReptileOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = 1.0,
            AdaptationSteps = innerSteps
        };

        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(1);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for inner steps={innerSteps}");
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for inner steps={innerSteps}");
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void MetaTrain_WithDifferentOuterLearningRates_CompletesSuccessfully(double epsilon)
    {
        // Arrange - Reptile uses epsilon (interpolation factor) as outer learning rate
        var mockModel = CreateMockModel();
        var options = new ReptileOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.02,
            OuterLearningRate = epsilon,
            AdaptationSteps = 10
        };

        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(1);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for epsilon={epsilon}");
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for epsilon={epsilon}");
    }

    [Fact]
    public void MetaTrain_WithMultipleTasksPerBatch_ProcessesCorrectly()
    {
        // Test Reptile with batched updates (less common but valid)

        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Use batch of 4 tasks
        var losses = new List<double>();
        for (int i = 0; i < 20; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.Equal(20, losses.Count);
        Assert.All(losses, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
    }

    [Fact]
    public void Algorithm_HasCorrectName()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Assert
        Assert.Equal(MetaLearningAlgorithmType.Reptile, algorithm.AlgorithmType);
    }

    [Fact]
    public void Algorithm_ExposesCorrectHyperparameters()
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new ReptileOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.05,
            OuterLearningRate = 0.8,
            AdaptationSteps = 15
        };

        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Assert
        Assert.Equal(0.05, algorithm.InnerLearningRate);
        Assert.Equal(0.8, algorithm.OuterLearningRate);
        Assert.Equal(15, algorithm.AdaptationSteps);
    }

    #endregion
}
