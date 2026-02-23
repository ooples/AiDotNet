#nullable disable
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class RootMeanSquaredErrorFitnessCalculatorTests
{
    /// <summary>
    /// Creates ErrorStats with the specified actual and predicted values.
    /// Note: Data must include both zeros and non-zeros because ErrorStats calculates AUC
    /// which requires both positive and negative samples (classification-like constraint).
    /// </summary>
    private static ErrorStats<double> CreateErrorStats(double[] actual, double[] predicted)
    {
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = new Vector<double>(actual),
            Predicted = new Vector<double>(predicted),
            PredictionType = PredictionType.Regression
        };
        return new ErrorStats<double>(inputs);
    }

    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        // Perfect predictions: predicted equals actual (RMSE = 0.0)
        // Data includes 0.0 to satisfy AUC calculation requirements (needs both positive and negative samples)
        var actual = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var predicted = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = CreateErrorStats(actual, predicted)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithSmallErrors_ReturnsSmallRMSE()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        // Small errors in predictions
        // Data includes 0.0 to satisfy AUC calculation requirements
        // Errors: 0.5, 0.5, 0, 0, 0.5 -> MSE = (0.25+0.25+0+0+0.25)/5 = 0.15 -> RMSE ≈ 0.387
        var actual = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var predicted = new double[] { 0.5, 1.5, 2.0, 3.0, 3.5 };
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = CreateErrorStats(actual, predicted)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.InRange(result, 0.3, 0.5); // RMSE should be around 0.387
    }

    [Fact]
    public void CalculateFitnessScore_WithLargeErrors_ReturnsLargeRMSE()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        // Large errors in predictions, RMSE ≈ 10
        // Data includes 0.0 to satisfy AUC calculation requirements
        var actual = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var predicted = new double[] { 10.0, 11.0, 12.0, 13.0, 14.0 };
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = CreateErrorStats(actual, predicted)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.InRange(result, 9.5, 10.5); // RMSE should be around 10
    }

    [Fact]
    public void CalculateFitnessScore_RetrievesRMSEFromErrorStats()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        // Create specific errors to get RMSE ≈ 3.0
        // Data includes 0.0 to satisfy AUC calculation requirements
        var actual = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var predicted = new double[] { 3.0, 4.0, 5.0, 6.0, 7.0 };
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = CreateErrorStats(actual, predicted)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert - Should retrieve RMSE from ErrorStats (approximately 3.0)
        Assert.InRange(result, 2.8, 3.2);
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(1.5, 3.0);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(5.0, 2.0);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<float, Vector<float>, Vector<float>>();
        // Create errors to get RMSE ≈ 2.5
        // Data includes 0.0f to satisfy AUC calculation requirements
        var actual = new float[] { 0.0f, 1.0f, 2.0f, 3.0f };
        var predicted = new float[] { 2.5f, 3.5f, 4.5f, 5.5f };
        var inputs = new ErrorStatsInputs<float>
        {
            Actual = new Vector<float>(actual),
            Predicted = new Vector<float>(predicted),
            PredictionType = PredictionType.Regression
        };
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            ErrorStats = new ErrorStats<float>(inputs)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.InRange(result, 2.3f, 2.7f); // RMSE should be around 2.5
    }

    [Fact]
    public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithDefaultDataSetType_UsesValidation()
    {
        // Arrange & Act
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithVerySmallRMSE_HandlesCorrectly()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        // Very small errors, RMSE ≈ 0.0001
        // Data includes 0.0 to satisfy AUC calculation requirements
        var actual = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var predicted = new double[] { 0.0001, 1.0001, 2.0001, 3.0001, 4.0001 };
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = CreateErrorStats(actual, predicted)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result < 0.001, $"Expected RMSE < 0.001, but got {result}");
    }

    [Fact]
    public void CalculateFitnessScore_WithVeryLargeRMSE_HandlesCorrectly()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        // Very large errors, RMSE ≈ 1000000
        // Data includes 0.0 to satisfy AUC calculation requirements
        var actual = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var predicted = new double[] { 1000000.0, 1000001.0, 1000002.0, 1000003.0, 1000004.0 };
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = CreateErrorStats(actual, predicted)
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 999999.0, $"Expected RMSE > 999999, but got {result}");
    }

    [Fact]
    public void IsBetterFitness_ComparesCorrectly()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.True(calculator.IsBetterFitness(0.1, 0.2)); // 0.1 < 0.2
        Assert.True(calculator.IsBetterFitness(1.0, 1.5)); // 1.0 < 1.5
        Assert.False(calculator.IsBetterFitness(2.0, 1.0)); // 2.0 > 1.0
        Assert.False(calculator.IsBetterFitness(5.0, 3.0)); // 5.0 > 3.0
    }
}
