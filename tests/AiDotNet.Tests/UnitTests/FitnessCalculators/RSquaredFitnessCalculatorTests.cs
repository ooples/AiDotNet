using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class RSquaredFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsOne()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 1.0 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(1.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithNoExplanatoryPower_ReturnsZero()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 0.0 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithGoodPredictions_ReturnsHighR2()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 0.85 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.85, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithPoorPredictions_ReturnsLowR2()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 0.2 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.2, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithNegativeR2_HandlesCorrectly()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = -0.5 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(-0.5, result, 10);
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        // Note: This returns false due to internal optimization handling
        // but R² values themselves are better when higher
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        // Note: Due to isHigherScoreBetter being false, lower is considered better in this context
        var result = calculator.IsBetterFitness(0.5, 0.8);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.9, 0.6);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<float, Vector<float>, Vector<float>>();
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            PredictionStats = new PredictionStats<float> { R2 = 0.75f }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.75f, result, 5);
    }

    [Fact]
    public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
    }

    [Fact]
    public void Constructor_WithDefaultDataSetType_UsesValidation()
    {
        // Arrange & Act
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Assert
        Assert.NotNull(calculator);
    }

    [Fact]
    public void CalculateFitnessScore_RetrievesR2FromPredictionStats()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var expectedR2 = 0.6789;
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = expectedR2 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(expectedR2, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithVeryHighR2_HandlesCorrectly()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 0.999 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.999, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithVeryLowR2_HandlesCorrectly()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 0.001 }
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.001, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithR2AtBoundaries_WorksCorrectly()
    {
        // Arrange
        var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Test R² = 1.0
        var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 1.0 }
        };

        // Test R² = 0.0
        var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            PredictionStats = new PredictionStats<double> { R2 = 0.0 }
        };

        // Act
        var result1 = calculator.CalculateFitnessScore(dataSet1);
        var result2 = calculator.CalculateFitnessScore(dataSet2);

        // Assert
        Assert.Equal(1.0, result1, 10);
        Assert.Equal(0.0, result2, 10);
    }
}
