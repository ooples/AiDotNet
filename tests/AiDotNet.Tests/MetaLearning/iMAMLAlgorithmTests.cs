using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.MetaLearning;

/// <summary>
/// Unit tests for the iMAML (implicit Model-Agnostic Meta-Learning) algorithm.
/// </summary>
public class iMAMLAlgorithmTests
{
    private MatrixMockModel CreateMockModel() => new MatrixMockModel(10, 5);
    private MockLossFunction<double> CreateMockLossFunction() => new MockLossFunction<double>();

    private iMAMLOptions<double, Matrix<double>, Vector<double>> CreateDefaultOptions()
    {
        var mockModel = CreateMockModel();
        return new iMAMLOptions<double, Matrix<double>, Vector<double>>(mockModel)
        {
            LossFunction = CreateMockLossFunction(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10,
            ConjugateGradientTolerance = 1e-8
        };
    }

    /// <summary>
    /// Creates a mock task for testing purposes.
    /// </summary>
    private IMetaLearningTask<double, Matrix<double>, Vector<double>> CreateMockTask()
    {
        // Create mock support data (5 examples, 10 features each)
        var supportInput = Matrix<double>.CreateRandom(5, 10, -1, 1);
        var supportOutput = Vector<double>.CreateRandom(5, -1, 1);

        // Create mock query data (15 examples)
        var queryInput = Matrix<double>.CreateRandom(15, 10, -1, 1);
        var queryOutput = Vector<double>.CreateRandom(15, -1, 1);

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
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

    [Fact]
    public void Constructor_WithValidOptions_ReturnsInstance()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(algorithm);
        Assert.Equal(MetaLearningAlgorithmType.iMAML, algorithm.AlgorithmType);
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void MetaTrain_WithNullTaskBatch_ThrowsArgumentException()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => algorithm.MetaTrain(null!));
    }

    [Fact]
    public void MetaTrain_WithEmptyTaskBatch_ThrowsArgumentException()
    {
        // Arrange
        var emptyTasks = Array.Empty<IMetaLearningTask<double, Matrix<double>, Vector<double>>>();

        // Act & Assert
        // TaskBatch constructor validates that tasks are not empty (fail fast)
        Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(emptyTasks));
    }

    [Fact]
    public void MetaTrain_WithSingleTask_ReturnsValidLoss()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Fact]
    public void MetaTrain_WithMultipleTasks_ReturnsValidLoss()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var tasks = Enumerable.Range(0, 5).Select(_ => CreateMockTask()).ToArray();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Fact]
    public void Adapt_WithNullTask_ThrowsArgumentNullException()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => algorithm.Adapt(null!));
    }

    [Fact]
    public void Adapt_WithValidTask_ReturnsAdaptedModel()
    {
        // Arrange
        var options = CreateDefaultOptions();
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();

        // Act
        var adaptedModel = algorithm.Adapt(task);

        // Assert
        Assert.NotNull(adaptedModel);
    }

    [Fact]
    public void MetaTrain_WithNeumannApproximation_ReturnsValidLoss()
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(mockModel)
        {
            LossFunction = CreateMockLossFunction(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            UseNeumannApproximation = true,
            NeumannSeriesTerms = 5
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(!double.IsNaN(loss), "Loss should not be NaN");
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Fact]
    public void MetaTrain_WithFirstOrderApproximation_ReturnsValidLoss()
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(mockModel)
        {
            LossFunction = CreateMockLossFunction(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            UseFirstOrder = true
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(!double.IsNaN(loss), "Loss should not be NaN");
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void MetaTrain_WithDifferentLambdaValues_ReturnsValidLoss(double lambda)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(mockModel)
        {
            LossFunction = CreateMockLossFunction(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = lambda,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for lambda={lambda}");
    }

    [Theory]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(20)]
    public void MetaTrain_WithDifferentCGIterations_ReturnsValidLoss(int cgIterations)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(mockModel)
        {
            LossFunction = CreateMockLossFunction(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = cgIterations
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for CG iterations={cgIterations}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void MetaTrain_WithDifferentAdaptationSteps_ReturnsValidLoss(int adaptationSteps)
    {
        // Arrange
        var mockModel = CreateMockModel();
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(mockModel)
        {
            LossFunction = CreateMockLossFunction(),
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = adaptationSteps,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.True(loss >= 0, $"Loss should be non-negative for adaptation steps={adaptationSteps}");
    }

    [Fact]
    public void Options_IsValid_ReturnsTrueForValidOptions()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var isValid = options.IsValid();

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void Options_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var options = CreateDefaultOptions();

        // Act
        var clonedOptions = options.Clone() as iMAMLOptions<double, Matrix<double>, Vector<double>>;

        // Assert
        Assert.NotNull(clonedOptions);
        Assert.Equal(options.LambdaRegularization, clonedOptions.LambdaRegularization);
        Assert.Equal(options.ConjugateGradientIterations, clonedOptions.ConjugateGradientIterations);
        Assert.Equal(options.AdaptationSteps, clonedOptions.AdaptationSteps);
    }
}
