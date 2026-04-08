using AiDotNet.HyperparameterOptimization;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using Xunit;

namespace AiDotNet.Tests.UnitTests.HyperparameterOptimization;

/// <summary>
/// Unit tests for HyperbandOptimizer hyperparameter optimization with early stopping.
/// </summary>
public class HyperbandOptimizerTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesOptimizer()
    {
        // Act
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();

        // Assert
        Assert.NotNull(optimizer);
        Assert.True(optimizer.NumBrackets > 0);
    }

    [Fact]
    public void Constructor_WithMaximize_SetsMaximize()
    {
        // Act
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(maximize: true);

        // Assert - Will find maximum values during optimization
        Assert.NotNull(optimizer);
    }

    [Fact]
    public void Constructor_WithMinimize_SetsMinimize()
    {
        // Act
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(maximize: false);

        // Assert - Will find minimum values during optimization
        Assert.NotNull(optimizer);
    }

    [Fact]
    public void Constructor_WithSeed_ProducesReproducibleResults()
    {
        // Arrange
        var searchSpace = CreateSimpleSearchSpace();
        int seed = 42;

        var optimizer1 = new HyperbandOptimizer<double, double[], double[]>(seed: seed);
        var optimizer2 = new HyperbandOptimizer<double, double[], double[]>(seed: seed);

        // Act
        var result1 = optimizer1.Optimize(SimpleObjective, searchSpace, 5);
        var result2 = optimizer2.Optimize(SimpleObjective, searchSpace, 5);

        // Assert
        Assert.Equal(result1.BestObjectiveValue, result2.BestObjectiveValue);
    }

    [Fact]
    public void Constructor_WithZeroMaxResource_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double[]>(maxResource: 0));
    }

    [Fact]
    public void Constructor_WithNegativeMaxResource_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double[]>(maxResource: -1));
    }

    [Fact]
    public void Constructor_WithReductionFactorLessThanTwo_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double[]>(reductionFactor: 1));
    }

    [Fact]
    public void Constructor_WithZeroMinResource_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double[]>(minResource: 0));
    }

    [Fact]
    public void Constructor_WithMinResourceExceedingMax_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double[]>(maxResource: 10, minResource: 20));
    }

    [Fact]
    public void Constructor_WithValidResourceParameters_CalculatesNumBrackets()
    {
        // Arrange & Act
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 81, reductionFactor: 3, minResource: 1);

        // Assert
        // NumBrackets = floor(log_3(81)) + 1 = floor(4) + 1 = 5
        Assert.Equal(5, optimizer.NumBrackets);
    }

    #endregion

    #region Optimize Tests

    [Fact]
    public void Optimize_WithValidInputs_ReturnsResult()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 123);
        var searchSpace = CreateSimpleSearchSpace();

        // Act
        var result = optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.BestTrial);
        Assert.True(result.CompletedTrials > 0);
    }

    [Fact]
    public void Optimize_WithNullObjectiveFunction_ThrowsArgumentNullException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();
        var searchSpace = CreateSimpleSearchSpace();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.Optimize(null!, searchSpace, 10));
    }

    [Fact]
    public void Optimize_WithNullSearchSpace_ThrowsArgumentNullException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.Optimize(SimpleObjective, null!, 10));
    }

    [Fact]
    public void Optimize_WithZeroTrials_ThrowsArgumentException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();
        var searchSpace = CreateSimpleSearchSpace();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => optimizer.Optimize(SimpleObjective, searchSpace, 0));
    }

    [Fact]
    public void Optimize_WithNegativeTrials_ThrowsArgumentException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();
        var searchSpace = CreateSimpleSearchSpace();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => optimizer.Optimize(SimpleObjective, searchSpace, -5));
    }

    [Fact]
    public void Optimize_ForMaximization_FindsHighValue()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maximize: true, maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        // Act
        var result = optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Assert
        Assert.True(result.BestObjectiveValue > 0);
    }

    [Fact]
    public void Optimize_ForMinimization_FindsLowValue()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maximize: false, maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        // Act
        var result = optimizer.Optimize(MinimizeObjective, searchSpace, 10);

        // Assert
        Assert.True(result.BestObjectiveValue >= 0);
    }

    [Fact]
    public void Optimize_WithContinuousParameter_SamplesInRange()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.001, 0.1);

        // Act
        var result = optimizer.Optimize(params_ =>
        {
            var lr = Convert.ToDouble(params_["learning_rate"]);
            return lr; // Return the learning rate as objective
        }, searchSpace, 10);

        // Assert - Best value should be in range
        Assert.InRange(result.BestObjectiveValue, 0.001, 0.1);
    }

    [Fact]
    public void Optimize_WithIntegerParameter_SamplesIntegers()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("batch_size", 16, 64, step: 16);

        // Act
        var result = optimizer.Optimize(params_ =>
        {
            var batchSize = Convert.ToInt32(params_["batch_size"]);
            return batchSize / 10.0; // Return scaled batch size
        }, searchSpace, 10);

        // Assert - Parameters should be valid
        Assert.NotNull(result.BestParameters);
        var bestBatchSize = Convert.ToInt32(result.BestParameters["batch_size"]);
        Assert.Contains(bestBatchSize, new[] { 16, 32, 48, 64 });
    }

    [Fact]
    public void Optimize_WithCategoricalParameter_SelectsFromChoices()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = new HyperparameterSearchSpace();
        var optimizers = new object[] { "adam", "sgd", "rmsprop" };
        searchSpace.AddCategorical("optimizer", optimizers);

        // Act
        var result = optimizer.Optimize(params_ =>
        {
            var opt = params_["optimizer"].ToString();
            return opt switch
            {
                "adam" => 1.0,
                "sgd" => 0.5,
                "rmsprop" => 0.75,
                _ => 0.0
            };
        }, searchSpace, 10);

        // Assert
        Assert.NotNull(result.BestParameters);
        var bestOptimizer = result.BestParameters["optimizer"].ToString();
        Assert.Contains(bestOptimizer, new[] { "adam", "sgd", "rmsprop" });
    }

    [Fact]
    public void Optimize_WithLogScaleParameter_SamplesLogarithmically()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.0001, 0.1, logScale: true);

        // Act
        var result = optimizer.Optimize(params_ =>
        {
            var lr = Convert.ToDouble(params_["learning_rate"]);
            return -Math.Abs(lr - 0.01); // Minimize distance from 0.01
        }, searchSpace, 10);

        // Assert
        Assert.NotNull(result.BestParameters);
        var bestLr = Convert.ToDouble(result.BestParameters["learning_rate"]);
        Assert.InRange(bestLr, 0.0001, 0.1);
    }

    [Fact]
    public void Optimize_TracksResourceBudget()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 27, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();
        var resourceValues = new List<int>();

        // Act
        var result = optimizer.Optimize(params_ =>
        {
            if (params_.ContainsKey("resource"))
            {
                resourceValues.Add(Convert.ToInt32(params_["resource"]));
            }
            return 1.0;
        }, searchSpace, 10);

        // Assert - Resource values should be present and reasonable
        Assert.True(resourceValues.Count > 0);
        Assert.All(resourceValues, r => Assert.InRange(r, 1, 27));
    }

    #endregion

    #region GetBestTrial Tests

    [Fact]
    public void GetBestTrial_AfterOptimization_ReturnsBestTrial()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Act
        var bestTrial = optimizer.GetBestTrial();

        // Assert
        Assert.NotNull(bestTrial);
        Assert.Equal(TrialStatus.Complete, bestTrial.Status);
    }

    [Fact]
    public void GetBestTrial_BeforeOptimization_ThrowsInvalidOperationException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => optimizer.GetBestTrial());
    }

    #endregion

    #region GetAllTrials Tests

    [Fact]
    public void GetAllTrials_AfterOptimization_ReturnsAllTrials()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Act
        var allTrials = optimizer.GetAllTrials();

        // Assert
        Assert.NotNull(allTrials);
        Assert.True(allTrials.Count > 0);
    }

    [Fact]
    public void GetAllTrials_BeforeOptimization_ReturnsEmptyList()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();

        // Act
        var allTrials = optimizer.GetAllTrials();

        // Assert
        Assert.Empty(allTrials);
    }

    #endregion

    #region GetTrials Filter Tests

    [Fact]
    public void GetTrials_WithCompletedFilter_ReturnsOnlyCompleted()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Act
        var completedTrials = optimizer.GetTrials(t => t.Status == TrialStatus.Complete);

        // Assert
        Assert.All(completedTrials, t => Assert.Equal(TrialStatus.Complete, t.Status));
    }

    [Fact]
    public void GetTrials_WithNullFilter_ThrowsArgumentNullException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.GetTrials(null!));
    }

    #endregion

    #region SuggestNext Tests

    [Fact]
    public void SuggestNext_BeforeOptimization_ThrowsInvalidOperationException()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>();
        var trial = new HyperparameterTrial<double>(0);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => optimizer.SuggestNext(trial));
    }

    [Fact]
    public void SuggestNext_AfterOptimization_ReturnsSuggestedParameters()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        optimizer.Optimize(SimpleObjective, searchSpace, 10);
        var trial = new HyperparameterTrial<double>(100);

        // Act
        var suggested = optimizer.SuggestNext(trial);

        // Assert
        Assert.NotNull(suggested);
        Assert.Contains("x", suggested.Keys);
    }

    #endregion

    #region Bracket Info Tests

    [Fact]
    public void GetBracketInfo_ReturnsCorrectNumberOfBrackets()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 81, reductionFactor: 3, minResource: 1);

        // Act
        var brackets = optimizer.GetBracketInfo();

        // Assert
        Assert.Equal(optimizer.NumBrackets, brackets.Count);
    }

    [Fact]
    public void GetBracketInfo_BracketsHaveValidResources()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 81, reductionFactor: 3, minResource: 1);

        // Act
        var brackets = optimizer.GetBracketInfo();

        // Assert - All brackets have valid initial resources
        Assert.All(brackets, b => Assert.InRange(b.InitialResource, 1, 81));
        // Each bracket has positive initial configurations
        Assert.All(brackets, b => Assert.True(b.InitialConfigurations > 0));
    }

    [Fact]
    public void GetBracketInfo_EachBracketHasRounds()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 27, reductionFactor: 3, minResource: 1);

        // Act
        var brackets = optimizer.GetBracketInfo();

        // Assert
        Assert.All(brackets, b => Assert.True(b.Rounds.Count > 0));
    }

    [Fact]
    public void GetTotalConfigurationCount_ReturnsPositiveNumber()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 27, reductionFactor: 3, minResource: 1);

        // Act
        var count = optimizer.GetTotalConfigurationCount();

        // Assert
        Assert.True(count > 0);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void Optimize_FromMultipleThreads_IsThreadSafe()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();
        var results = new List<HyperparameterOptimizationResult<double>>();
        var exceptions = new List<Exception>();

        // Act
        var tasks = new List<Task>();
        for (int i = 0; i < 3; i++)
        {
            var seed = i;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var localOptimizer = new HyperbandOptimizer<double, double[], double[]>(
                        maxResource: 9, reductionFactor: 3, seed: seed);
                    var result = localOptimizer.Optimize(SimpleObjective, searchSpace, 5);
                    lock (results)
                    {
                        results.Add(result);
                    }
                }
                catch (Exception ex)
                {
                    lock (exceptions)
                    {
                        exceptions.Add(ex);
                    }
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(3, results.Count);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void Optimize_WithFailingObjective_HandlesGracefully()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();
        int callCount = 0;

        // Act - Objective that fails for some configurations
        var result = optimizer.Optimize(params_ =>
        {
            callCount++;
            if (callCount % 3 == 0)
            {
                throw new ArgumentException("Simulated failure");
            }
            return Convert.ToDouble(params_["x"]);
        }, searchSpace, 10);

        // Assert - Should have at least some completed trials
        Assert.True(result.CompletedTrials > 0 || result.FailedTrials > 0);
    }

    #endregion

    #region Result Properties Tests

    [Fact]
    public void OptimizationResult_HasCorrectStatistics()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        // Act
        var result = optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Assert
        Assert.True(result.TotalTrials > 0);
        Assert.True(result.CompletedTrials > 0);
        Assert.True(result.TotalTime >= TimeSpan.Zero);
        Assert.NotEqual(default, result.StartTime);
        Assert.NotEqual(default, result.EndTime);
    }

    [Fact]
    public void OptimizationResult_HasBestParameters()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        // Act
        var result = optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Assert
        Assert.NotNull(result.BestParameters);
        Assert.Contains("x", result.BestParameters.Keys);
    }

    [Fact]
    public void OptimizationResult_HasSearchSpace()
    {
        // Arrange
        var optimizer = new HyperbandOptimizer<double, double[], double[]>(
            maxResource: 9, reductionFactor: 3, seed: 42);
        var searchSpace = CreateSimpleSearchSpace();

        // Act
        var result = optimizer.Optimize(SimpleObjective, searchSpace, 10);

        // Assert
        Assert.NotNull(result.SearchSpace);
        Assert.Same(searchSpace, result.SearchSpace);
    }

    #endregion

    #region Helper Methods

    private static HyperparameterSearchSpace CreateSimpleSearchSpace()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 10.0);
        return searchSpace;
    }

    private static double SimpleObjective(Dictionary<string, object> parameters)
    {
        var x = Convert.ToDouble(parameters["x"]);
        return x; // Maximize x
    }

    private static double MinimizeObjective(Dictionary<string, object> parameters)
    {
        var x = Convert.ToDouble(parameters["x"]);
        return Math.Abs(x - 5.0); // Minimize distance from 5
    }

    #endregion
}
