using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class KullbackLeiblerDivergenceFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.25, 0.25, 0.25, 0.25 }),
            Actual = new Vector<double>(new double[] { 0.25, 0.25, 0.25, 0.25 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithDifferentDistributions_ReturnsPositiveValue()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.5, 0.3, 0.2 }),
            Actual = new Vector<double>(new double[] { 0.3, 0.3, 0.4 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithUniformDistributions_ReturnsZero()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.2, 0.2, 0.2, 0.2, 0.2 }),
            Actual = new Vector<double>(new double[] { 0.2, 0.2, 0.2, 0.2, 0.2 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithSmallDivergence_ReturnsSmallPositiveValue()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.51, 0.49 }),
            Actual = new Vector<double>(new double[] { 0.5, 0.5 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
        Assert.True(result < 0.1); // Small divergence should result in small loss
    }

    [Fact]
    public void CalculateFitnessScore_WithLargeDivergence_ReturnsLargeValue()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.9, 0.1 }),
            Actual = new Vector<double>(new double[] { 0.1, 0.9 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 1.0); // Large divergence should result in larger loss
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.1, 0.5);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.8, 0.3);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<float, Vector<float>, Vector<float>>();
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            Predicted = new Vector<float>(new float[] { 0.25f, 0.25f, 0.25f, 0.25f }),
            Actual = new Vector<float>(new float[] { 0.25f, 0.25f, 0.25f, 0.25f })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0f, result, 5);
    }

    [Fact]
    public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithMultipleClasses_HandlesCorrectly()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }),
            Actual = new Vector<double>(new double[] { 0.2, 0.2, 0.3, 0.3 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_IsAsymmetric_DifferentWhenSwapped()
    {
        // Arrange
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Vector<double>, Vector<double>>();
        var predicted = new Vector<double>(new double[] { 0.7, 0.3 });
        var actual = new Vector<double>(new double[] { 0.3, 0.7 });

        var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = predicted,
            Actual = actual
        };

        var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = actual,
            Actual = predicted
        };

        // Act
        var result1 = calculator.CalculateFitnessScore(dataSet1);
        var result2 = calculator.CalculateFitnessScore(dataSet2);

        // Assert - KL Divergence is asymmetric
        Assert.NotEqual(result1, result2, 5);
    }
}
