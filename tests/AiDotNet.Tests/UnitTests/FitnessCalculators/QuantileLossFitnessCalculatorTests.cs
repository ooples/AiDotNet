using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

/// <summary>
/// Tests for QuantileLossFitnessCalculator.
///
/// Note: The current implementation has a bug where it incorrectly uses OrdinalRegressionLoss
/// with the quantile value (0.0-1.0) converted to an integer as the number of classes.
/// This results in "Number of classes must be at least 2 for ordinal regression" errors.
///
/// These tests only validate the properties and non-calculation functionality until the
/// underlying implementation is fixed to use a proper QuantileLoss function.
/// </summary>
public class QuantileLossFitnessCalculatorTests
{
    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.15, 0.45);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.7, 0.3);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null!));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(dataSetType: DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(dataSetType: DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithValidQuantile_DoesNotThrow()
    {
        // Arrange & Act
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.5);

        // Assert
        Assert.NotNull(calculator);
    }

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }
}
