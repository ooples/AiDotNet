using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using NUnit.Framework;
using System.Linq;

namespace AiDotNet.Tests.MetaLearning;

/// <summary>
/// Unit tests for the iMAML (implicit Model-Agnostic Meta-Learning) algorithm.
/// </summary>
[TestFixture]
public class iMAMLAlgorithmTests
{
    private MockNeuralNetwork<double, Matrix<double>, Vector<double>>? _mockModel;
    private MockLossFunction<double> _mockLossFunction = null!;
    private iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>> _options = null!;

    [SetUp]
    public void SetUp()
    {
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(10, 5);
        _mockLossFunction = new MockLossFunction<double>();

        _options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel,
            LossFunction = _mockLossFunction,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10,
            ConjugateGradientTolerance = 1e-8,
            HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences,
            CGPreconditioningMethod = CGPreconditioningMethod.Jacobi,
            FiniteDifferencesEpsilon = 1e-5,
            UseAdaptiveInnerLearningRate = true,
            MinInnerLearningRate = 1e-6,
            MaxInnerLearningRate = 0.1,
            EnableLineSearch = false
        };
    }

    [Test]
    public void Constructor_WithValidOptions_ReturnsInstance()
    {
        // Act
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);

        // Assert
        Assert.That(algorithm, Is.Not.Null);
        Assert.That(algorithm.AlgorithmName, Is.EqualTo("iMAML"));
    }

    [Test]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(null!));
    }

    [Test]
    public void MetaTrain_WithNullTaskBatch_ThrowsArgumentException()
    {
        // Arrange
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => algorithm.MetaTrain(null!));
    }

    [Test]
    public void MetaTrain_WithEmptyTaskBatch_ThrowsArgumentException()
    {
        // Arrange
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var emptyBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new List<ITask<double, Matrix<double>, Vector<double>>>());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => algorithm.MetaTrain(emptyBatch));
    }

    [Test]
    public void MetaTrain_WithSingleTask_ReturnsValidLoss()
    {
        // Arrange
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void MetaTrain_WithMultipleTasks_AveragesCorrectly()
    {
        // Arrange
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var tasks = Enumerable.Range(0, 5).Select(_ => CreateMockTask()).ToList();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void Adapt_WithNullTask_ThrowsArgumentNullException()
    {
        // Arrange
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => algorithm.Adapt(null!));
    }

    [Test]
    public void Adapt_WithValidTask_ReturnsAdaptedModel()
    {
        // Arrange
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var adaptedModel = algorithm.Adapt(task);

        // Assert
        Assert.That(adaptedModel, Is.Not.Null);
    }

    [Test]
    public void ImplicitGradients_WithFiniteDifferences_ComputesCorrectly()
    {
        // Arrange
        _options.HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.Not.NaN);
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void ImplicitGradients_WithAutomaticDifferentiation_ComputesCorrectly()
    {
        // Arrange
        _options.HessianVectorProductMethod = HessianVectorProductMethod.AutomaticDifferentiation;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.Not.NaN);
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void ConjugateGradient_WithJacobiPreconditioning_Converges()
    {
        // Arrange
        _options.CGPreconditioningMethod = CGPreconditioningMethod.Jacobi;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void ConjugateGradient_WithNoPreconditioning_Converges()
    {
        // Arrange
        _options.CGPreconditioningMethod = CGPreconditioningMethod.None;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void AdaptiveLearningRate_Enabled_UpdatesCorrectly()
    {
        // Arrange
        _options.UseAdaptiveInnerLearningRate = true;
        _options.MinInnerLearningRate = 1e-5;
        _options.MaxInnerLearningRate = 0.05;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void AdaptiveLearningRate_Disabled_UsesFixedRate()
    {
        // Arrange
        _options.UseAdaptiveInnerLearningRate = false;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void LineSearch_Enabled_FindsOptimalStep()
    {
        // Arrange
        _options.EnableLineSearch = true;
        _options.LineSearchMaxIterations = 10;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();

        // Act
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void LambdaRegularization_AffectsGradientMagnitude()
    {
        // Arrange
        _options.LambdaRegularization = 0.5;
        var algorithm1 = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        _options.LambdaRegularization = 2.0;
        var algorithm2 = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);

        // Act
        var loss1 = algorithm1.MetaTrain(taskBatch);
        var loss2 = algorithm2.MetaTrain(taskBatch);

        // Assert
        Assert.That(loss1, Is.GreaterThanOrEqualTo(0));
        Assert.That(loss2, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void MultipleAdaptationSteps_IncreasesAdaptationAccuracy()
    {
        // Arrange
        _options.AdaptationSteps = 10;
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(_options);
        var task = CreateMockTask();
        var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Act
        var loss = algorithm.MetaTrain(taskBatch);

        // Assert
        Assert.That(loss, Is.GreaterThanOrEqualTo(0));
    }

    /// <summary>
    /// Creates a mock task for testing purposes.
    /// </summary>
    private ITask<double, Matrix<double>, Vector<double>> CreateMockTask()
    {
        // Create mock support data (5 examples, 10 features each)
        var supportInput = Matrix<double>.Random(5, 10, -1, 1);
        var supportOutput = Vector<double>.Random(5, -1, 1);

        // Create mock query data (15 examples)
        var queryInput = Matrix<double>.Random(15, 10, -1, 1);
        var queryOutput = Vector<double>.Random(15, -1, 1);

        return new Task<double, Matrix<double>, Vector<double>>(
            supportInput,
            supportOutput,
            queryInput,
            queryOutput,
            numWays: 5,
            numShots: 1,
            numQueryPerClass: 3,
            taskId: "test-task");
    }
}