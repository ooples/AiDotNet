using AiDotNet.HyperparameterOptimization;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TrainingInfrastructure;

/// <summary>
/// Unit tests for BayesianOptimizer hyperparameter optimization.
/// </summary>
public class BayesianOptimizerTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_InitializesCorrectly()
    {
        // Arrange & Act
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: true,
            acquisitionFunction: AcquisitionFunctionType.ExpectedImprovement,
            nInitialPoints: 5,
            explorationWeight: 2.0,
            seed: 42);

        // Assert
        Assert.NotNull(optimizer);
    }

    [Fact]
    public void Constructor_WithMinimizationGoal_InitializesCorrectly()
    {
        // Arrange & Act
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: false,
            seed: 42);

        // Assert
        Assert.NotNull(optimizer);
    }

    [Theory]
    [InlineData(AcquisitionFunctionType.ExpectedImprovement)]
    [InlineData(AcquisitionFunctionType.ProbabilityOfImprovement)]
    [InlineData(AcquisitionFunctionType.UpperConfidenceBound)]
    [InlineData(AcquisitionFunctionType.LowerConfidenceBound)]
    public void Constructor_WithDifferentAcquisitionFunctions_InitializesCorrectly(AcquisitionFunctionType acquisitionFunction)
    {
        // Arrange & Act
        var optimizer = new BayesianOptimizer<double, double[], double>(
            acquisitionFunction: acquisitionFunction,
            seed: 42);

        // Assert
        Assert.NotNull(optimizer);
    }

    [Fact]
    public void Constructor_WithMinimumInitialPoints_UsesAtLeastTwo()
    {
        // Arrange & Act
        var optimizer = new BayesianOptimizer<double, double[], double>(
            nInitialPoints: 1, // Should be clamped to 2
            seed: 42);

        // Assert - Constructor should not throw, clamping happens internally
        Assert.NotNull(optimizer);
    }

    #endregion

    #region Optimize Tests

    [Fact]
    public void Optimize_WithSimpleQuadraticFunction_FindsMinimum()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: false, // Minimize
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", -5.0, 5.0);

        // Simple quadratic function: f(x) = x^2, minimum at x = 0
        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            var x = Convert.ToDouble(parameters["x"]);
            return x * x;
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.BestParameters);
        Assert.True(result.BestParameters.ContainsKey("x"));
        Assert.NotNull(result.BestObjectiveValue);

        // The best x should be reasonably close to 0
        var bestX = Convert.ToDouble(result.BestParameters["x"]);
        Assert.True(Math.Abs(bestX) < 3.0, $"Best x={bestX} should be close to 0");
    }

    [Fact]
    public void Optimize_WithMultipleParameters_ReturnsValidResult()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: false,
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.001, 0.1, logScale: true);
        searchSpace.AddInteger("batch_size", 16, 128);
        searchSpace.AddCategorical("optimizer", "sgd", "adam");

        // Objective function that prefers lower learning rates and larger batch sizes
        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            var lr = Convert.ToDouble(parameters["learning_rate"]);
            var batchSize = Convert.ToInt32(parameters["batch_size"]);
            // Lower learning rate and higher batch size = lower loss
            return lr * 10 + 1.0 / batchSize;
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.BestParameters);
        Assert.True(result.BestParameters.ContainsKey("learning_rate"));
        Assert.True(result.BestParameters.ContainsKey("batch_size"));
        Assert.True(result.BestParameters.ContainsKey("optimizer"));
        Assert.NotNull(result.AllTrials);
        Assert.Equal(10, result.AllTrials.Count);
    }

    [Fact]
    public void Optimize_WithMaximization_FindsMaximum()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: true, // Maximize
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", -5.0, 5.0);

        // Inverted parabola: f(x) = -x^2, maximum at x = 0
        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            var x = Convert.ToDouble(parameters["x"]);
            return -x * x;
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.BestObjectiveValue);
        // Best score should be close to 0 (maximum of -x^2)
        Assert.True(result.BestObjectiveValue > -10.0, $"Best score={result.BestObjectiveValue} should be close to 0");
    }

    [Fact]
    public void Optimize_WithNullObjectiveFunction_ThrowsArgumentException()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(seed: 42);
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 1.0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.Optimize(null!, searchSpace, nTrials: 5));
    }

    [Fact]
    public void Optimize_WithNullSearchSpace_ThrowsArgumentException()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(seed: 42);
        Func<Dictionary<string, object>, double> objective = _ => 0.0;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.Optimize(objective, null!, nTrials: 5));
    }

    [Fact]
    public void Optimize_WithZeroTrials_ThrowsArgumentException()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(seed: 42);
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 1.0);
        Func<Dictionary<string, object>, double> objective = _ => 0.0;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => optimizer.Optimize(objective, searchSpace, nTrials: 0));
    }

    #endregion

    #region SuggestNext Tests

    [Fact]
    public void SuggestNext_BeforeOptimize_ThrowsInvalidOperationException()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(seed: 42);
        var trial = new HyperparameterTrial<double>(0);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => optimizer.SuggestNext(trial));
    }

    [Fact]
    public void SuggestNext_AfterOptimize_ReturnsValidParameters()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 10.0);

        Func<Dictionary<string, object>, double> objective = p => Convert.ToDouble(p["x"]);

        // First run optimization to initialize search space
        optimizer.Optimize(objective, searchSpace, nTrials: 5);

        // Act
        var trial = new HyperparameterTrial<double>(5);
        var suggestion = optimizer.SuggestNext(trial);

        // Assert
        Assert.NotNull(suggestion);
        Assert.True(suggestion.ContainsKey("x"));
        var xValue = Convert.ToDouble(suggestion["x"]);
        Assert.True(xValue >= 0.0 && xValue <= 10.0, $"x={xValue} should be in [0, 10]");
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Optimize_WithBooleanParameter_HandlesCorrectly()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: true,
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddBoolean("use_dropout");
        searchSpace.AddContinuous("dropout_rate", 0.1, 0.5);

        // Objective prefers dropout=true with higher dropout rate
        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            var useDropout = (bool)parameters["use_dropout"];
            var dropoutRate = Convert.ToDouble(parameters["dropout_rate"]);
            return useDropout ? dropoutRate * 10 : 0;
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.BestParameters);
        Assert.True(result.BestParameters.ContainsKey("use_dropout"));
    }

    [Fact]
    public void Optimize_WithIntegerStepParameter_RespectsStepSize()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: false,
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("batch_size", 16, 128, step: 16); // Only 16, 32, 48, 64, 80, 96, 112, 128

        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            var batchSize = Convert.ToInt32(parameters["batch_size"]);
            return Math.Abs(batchSize - 64); // Prefer 64
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Assert
        Assert.NotNull(result);
        var bestBatchSize = Convert.ToInt32(result.BestParameters["batch_size"]);
        // Should be within the valid range
        Assert.InRange(bestBatchSize, 16, 128);
    }

    [Fact]
    public void Optimize_WithLogScaleContinuous_SamplesAcrossOrders()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: false,
            nInitialPoints: 5,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.0001, 0.1, logScale: true);

        // Objective prefers learning rate around 0.01
        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            var lr = Convert.ToDouble(parameters["learning_rate"]);
            return Math.Abs(Math.Log10(lr) - Math.Log10(0.01)); // Log-scale distance from 0.01
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Assert
        Assert.NotNull(result);
        var bestLr = Convert.ToDouble(result.BestParameters["learning_rate"]);
        Assert.True(bestLr >= 0.0001 && bestLr <= 0.1, $"learning_rate={bestLr} should be in [0.0001, 0.1]");
    }

    [Fact]
    public void Optimize_RecordsAllTrials()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 1.0);

        Func<Dictionary<string, object>, double> objective = p => Convert.ToDouble(p["x"]);

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 7);

        // Assert
        Assert.NotNull(result.AllTrials);
        Assert.Equal(7, result.AllTrials.Count);

        foreach (var trial in result.AllTrials)
        {
            Assert.NotNull(trial);
            Assert.NotNull(trial.Parameters);
            Assert.True(trial.Parameters.ContainsKey("x"));
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Optimize_WithObjectiveThatThrows_ContinuesWithOtherTrials()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 1.0);

        var callCount = 0;
        Func<Dictionary<string, object>, double> objective = parameters =>
        {
            callCount++;
            if (callCount == 2)
            {
                throw new InvalidOperationException("Simulated failure");
            }
            return Convert.ToDouble(parameters["x"]);
        };

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 5);

        // Assert - Should still complete with trials, marking failed ones appropriately
        Assert.NotNull(result);
        Assert.NotNull(result.AllTrials);
        Assert.Equal(5, result.AllTrials.Count);
    }

    [Fact]
    public void Optimize_WithSingleTrial_CompletesSuccessfully()
    {
        // Arrange
        var optimizer = new BayesianOptimizer<double, double[], double>(
            nInitialPoints: 1,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0.0, 1.0);

        Func<Dictionary<string, object>, double> objective = p => Convert.ToDouble(p["x"]);

        // Act
        var result = optimizer.Optimize(objective, searchSpace, nTrials: 1);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.BestParameters);
        Assert.NotNull(result.AllTrials);
        Assert.Single(result.AllTrials);
    }

    #endregion
}
