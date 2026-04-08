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
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Unit tests for the ReptileAlgorithm class.
/// </summary>
public class ReptileTrainerTests
{
    private SimpleMockModel CreateMockModel() => new SimpleMockModel(50);

    private ReptileOptions<double, Tensor<double>, Tensor<double>> CreateDefaultOptions()
    {
        var mockModel = CreateMockModel();
        return new ReptileOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 1.0,
            AdaptationSteps = 5
        };
    }

    /// <summary>
    /// Creates a mock task for testing purposes.
    /// </summary>
    private IMetaLearningTask<double, Tensor<double>, Tensor<double>> CreateMockTask()
    {
        var supportInput = new Tensor<double>(new int[] { 5, 10 });
        var supportOutput = new Tensor<double>(new int[] { 5 });
        var queryInput = new Tensor<double>(new int[] { 15, 10 });
        var queryOutput = new Tensor<double>(new int[] { 15 });

        return new MetaLearningTask<double, Tensor<double>, Tensor<double>>
        {
            SupportSetX = supportInput,
            SupportSetY = supportOutput,
            QuerySetX = queryInput,
            QuerySetY = queryOutput,
            NumWays = 5,
            NumShots = 1,
            NumQueryPerClass = 3,
            Name = "test-task"
        };
    }

    #region Constructor Tests

    [Fact(Timeout = 60000)]
    public async Task Constructor_ValidInputs_InitializesSuccessfully()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Assert
        Assert.NotNull(algorithm);
        Assert.Equal(MetaLearningAlgorithmType.Reptile, algorithm.AlgorithmType);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_NullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(null!));
    }

    #endregion

    #region MetaTrain Tests

    [Fact(Timeout = 60000)]
    public async Task MetaTrain_WithValidTaskBatch_ReturnsNonNegativeLoss()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Tensor<double>, Tensor<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Fact(Timeout = 60000)]
    public async Task MetaTrain_WithMultipleTasks_ReturnsValidLoss()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);
        var tasks = Enumerable.Range(0, 4).Select(_ => CreateMockTask()).ToArray();
        var taskBatch = new TaskBatch<double, Tensor<double>, Tensor<double>>(tasks);

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, "Loss should be non-negative");
        Assert.False(double.IsNaN(loss), "Loss should not be NaN");
    }

    [Fact(Timeout = 60000)]
    public async Task MetaTrain_NullTaskBatch_ThrowsArgumentException()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => algorithm.MetaTrain(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task MetaTrain_EmptyTaskBatch_ThrowsArgumentException()
    {
        // Arrange
        var emptyTasks = Array.Empty<IMetaLearningTask<double, Tensor<double>, Tensor<double>>>();

        // Act & Assert
        // TaskBatch constructor validates that tasks are not empty (fail fast)
        Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Tensor<double>, Tensor<double>>(emptyTasks));
    }

    #endregion

    #region Adapt Tests

    [Fact(Timeout = 60000)]
    public async Task Adapt_ValidTask_ReturnsAdaptedModel()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);
        var task = CreateMockTask();

        // Act
        var adaptedModel = algorithm.Adapt(task);

        // Assert
        Assert.NotNull(adaptedModel);
    }

    [Fact(Timeout = 60000)]
    public async Task Adapt_NullTask_ThrowsArgumentNullException()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => algorithm.Adapt(null!));
    }

    #endregion

    #region Options Tests

    [Fact(Timeout = 60000)]
    public async Task Options_IsValid_ReturnsTrueForValidOptions()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var isValid = options.IsValid();

        // Assert
        Assert.True(isValid);
    }

    [Fact(Timeout = 60000)]
    public async Task Options_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var clonedOptions = options.Clone() as ReptileOptions<double, Tensor<double>, Tensor<double>>;

        // Assert
        Assert.NotNull(clonedOptions);
        Assert.Equal(options.InnerLearningRate, clonedOptions.InnerLearningRate);
        Assert.Equal(options.OuterLearningRate, clonedOptions.OuterLearningRate);
        Assert.Equal(options.AdaptationSteps, clonedOptions.AdaptationSteps);
    }

    #endregion

    #region Parameterized Tests

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void MetaTrain_WithDifferentAdaptationSteps_ReturnsValidLoss(int adaptationSteps)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new ReptileOptions<double, Tensor<double>, Tensor<double>>(mockModel)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 1.0,
            AdaptationSteps = adaptationSteps
        };
        var algorithm = new ReptileAlgorithm<double, Tensor<double>, Tensor<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Tensor<double>, Tensor<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for adaptation steps={adaptationSteps}");
    }

    #endregion
}
