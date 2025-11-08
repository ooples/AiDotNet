using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class RootMeanSquaredErrorFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = new ErrorStats<double> { RMSE = 0.0 }
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
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = new ErrorStats<double> { RMSE = 0.5 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.5, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithLargeErrors_ReturnsLargeRMSE()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = new ErrorStats<double> { RMSE = 10.5 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(10.5, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_RetrievesRMSEFromErrorStats()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        var expectedRMSE = 3.14159;
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = new ErrorStats<double> { RMSE = expectedRMSE }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(expectedRMSE, result, 10);
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
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            ErrorStats = new ErrorStats<float> { RMSE = 2.5f }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(2.5f, result, 5);
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
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = new ErrorStats<double> { RMSE = 0.0001 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0001, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithVeryLargeRMSE_HandlesCorrectly()
    {
        // Arrange
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            ErrorStats = new ErrorStats<double> { RMSE = 1000000.0 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(1000000.0, result, 10);
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
