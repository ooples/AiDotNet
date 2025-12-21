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
/// Integration tests for SEALAlgorithm demonstrating Sample-Efficient Adaptive Learning functionality.
/// </summary>
/// <remarks>
/// These tests verify that the SEAL meta-learning framework operates correctly:
/// - Parameters are updated through meta-training
/// - Training completes without errors
/// - SEAL-specific features work (temperature, entropy regularization, adaptive LR)
/// - The algorithm adapts to new tasks correctly
/// </remarks>
public class SEALTrainerIntegrationTests
{
    private SimpleMockModel CreateMockModel() => new SimpleMockModel(50);

    private SEALOptions<double, Tensor<double>, Tensor<double>> CreateDefaultOptions()
    {
        var mockModel = CreateMockModel();
        return new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            Temperature = 1.0,
            EntropyCoefficient = 0.0,
            WeightDecay = 0.0
        };
    }

    /// <summary>
    /// Creates a mock task for testing purposes with varying patterns per seed.
    /// </summary>
    private IMetaLearningTask<double, Tensor<double>, Tensor<double>> CreateMockTask(int seed = 0)
    {
        var supportInput = new Tensor<double>(new int[] { 25, 10 });  // 5 classes x 5 shots
        var supportOutput = new Tensor<double>(new int[] { 25 });
        var queryInput = new Tensor<double>(new int[] { 75, 10 });   // 5 classes x 15 queries
        var queryOutput = new Tensor<double>(new int[] { 75 });

        // Fill with seeded random data that varies per task
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
            NumQueryPerClass = 15,
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
    public void SEAL_CompletesTraining_WithoutErrors()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Train for multiple iterations
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.Equal(10, losses.Count);
        Assert.All(losses, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
    }

    [Fact]
    public void SEAL_MetaTrain_UpdatesParametersCorrectly()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Get initial parameters
        var metaModel = algorithm.GetMetaModel();
        var initialParams = metaModel.GetParameters();
        var initialParamsClone = initialParams.Clone();

        // Act - Meta-train for multiple iterations
        for (int i = 0; i < 20; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            algorithm.MetaTrain(taskBatch);
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
    }

    [Fact]
    public void SEAL_WithEntropyRegularization_CompletesSuccessfully()
    {
        // Arrange - SEAL with entropy regularization for better generalization
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            EntropyCoefficient = 0.01
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
        Assert.All(losses, loss => Assert.False(double.IsPositiveInfinity(loss), "Loss should not be infinite"));
    }

    [Fact]
    public void SEAL_WithTemperatureScaling_CompletesSuccessfully()
    {
        // Arrange - SEAL with temperature scaling
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            Temperature = 1.5
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
        Assert.All(losses, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
    }

    [Fact]
    public void SEAL_WithWeightDecay_CompletesSuccessfully()
    {
        // Arrange - SEAL with weight decay for regularization
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            WeightDecay = 0.001
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
    }

    [Fact]
    public void SEAL_WithAdaptiveLearningRate_CompletesSuccessfully()
    {
        // Arrange - SEAL with adaptive inner learning rates
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            UseAdaptiveInnerLR = true,
            AdaptiveLearningRateMode = SEALAdaptiveLearningRateMode.GradientNorm
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
    }

    [Fact]
    public void SEAL_Adapt_ProducesTaskSpecificModel()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

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

        // Assert - Adapted model should be created
        Assert.NotNull(adaptedModel);

        // Meta-model parameters should remain unchanged
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
    public void SEAL_Evaluate_ProducesValidMetrics()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Meta-train first
        for (int i = 0; i < 10; i++)
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
    public void SEAL_LongTraining_TracksLossCorrectly()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act - Train for many iterations (simulating 50+ meta-iterations requirement)
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
    public void SEAL_WithFirstOrderApproximation_CompletesSuccessfully()
    {
        // Arrange - SEAL with first-order approximation (FOMAML-style)
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            UseFirstOrder = true
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var losses = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var taskBatch = CreateTaskBatch(4);
            var loss = algorithm.MetaTrain(taskBatch);
            losses.Add(loss);
        }

        // Assert
        Assert.All(losses, loss => Assert.False(double.IsNaN(loss), "Loss should not be NaN"));
        Assert.All(losses, loss => Assert.True(loss >= 0, "Loss should be non-negative"));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void SEAL_WithDifferentAdaptationSteps_CompletesSuccessfully(int adaptationSteps)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = adaptationSteps
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(4);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for adaptation steps={adaptationSteps}");
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for adaptation steps={adaptationSteps}");
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void SEAL_WithDifferentTemperatures_CompletesSuccessfully(double temperature)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            Temperature = temperature,
            // MinTemperature must be <= Temperature for validation to pass
            MinTemperature = Math.Min(temperature, 1.0)
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(4);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for temperature={temperature}");
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for temperature={temperature}");
    }

    [Theory]
    [InlineData(SEALAdaptiveLearningRateMode.GradientNorm)]
    [InlineData(SEALAdaptiveLearningRateMode.RunningMean)]
    [InlineData(SEALAdaptiveLearningRateMode.PerLayer)]
    public void SEAL_WithDifferentAdaptiveLRModes_CompletesSuccessfully(SEALAdaptiveLearningRateMode mode)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            UseAdaptiveInnerLR = true,
            AdaptiveLearningRateMode = mode
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act
        var taskBatch = CreateTaskBatch(4);
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.False(double.IsNaN(loss), $"Loss should not be NaN for mode={mode}");
    }

    [Fact]
    public void Algorithm_HasCorrectName()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Assert
        Assert.Equal(MetaLearningAlgorithmType.SEAL, algorithm.AlgorithmType);
    }

    [Fact]
    public void Algorithm_ExposesCorrectHyperparameters()
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new SEALOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.05,
            OuterLearningRate = 0.002,
            AdaptationSteps = 8
        };

        var algorithm = new SEALAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Assert
        Assert.Equal(0.05, algorithm.InnerLearningRate);
        Assert.Equal(0.002, algorithm.OuterLearningRate);
        Assert.Equal(8, algorithm.AdaptationSteps);
    }

    [Fact]
    public void Options_IsValid_ReturnsTrueForValidOptions()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Assert
        Assert.True(options.IsValid());
    }

    [Fact]
    public void Options_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var clonedOptions = options.Clone() as SEALOptions<double, Tensor<double>, Tensor<double>>;

        // Assert
        Assert.NotNull(clonedOptions);
        Assert.Equal(options.InnerLearningRate, clonedOptions.InnerLearningRate);
        Assert.Equal(options.OuterLearningRate, clonedOptions.OuterLearningRate);
        Assert.Equal(options.AdaptationSteps, clonedOptions.AdaptationSteps);
        Assert.Equal(options.Temperature, clonedOptions.Temperature);
        Assert.Equal(options.EntropyCoefficient, clonedOptions.EntropyCoefficient);
        Assert.Equal(options.WeightDecay, clonedOptions.WeightDecay);
    }

    #endregion
}
