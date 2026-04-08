using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class QuantileLossFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.5);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithMedianQuantile_WorksCorrectly()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.5);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 3.0, 5.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithHighQuantile_PenalizesUnderPrediction()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.9);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 5.0, 10.0, 15.0 }),
            Actual = new Vector<double>(new double[] { 10.0, 20.0, 30.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithLowQuantile_PenalizesOverPrediction()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.1);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 10.0, 20.0, 30.0 }),
            Actual = new Vector<double>(new double[] { 5.0, 10.0, 15.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithDefaultQuantile_UsesMedian()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

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
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<float, Vector<float>, Vector<float>>(quantile: 0.5f);
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            Predicted = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f }),
            Actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f })
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

    [Fact]
    public void CalculateFitnessScore_WithQ75_WorksCorrectly()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.75);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 5.0, 10.0, 15.0 }),
            Actual = new Vector<double>(new double[] { 6.0, 11.0, 14.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithQ25_WorksCorrectly()
    {
        // Arrange
        var calculator = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.25);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 3.0, 6.0, 9.0 }),
            Actual = new Vector<double>(new double[] { 4.0, 5.0, 10.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_AsymmetricPenalty_DifferentQuantiles()
    {
        // Arrange
        var calculatorLow = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.1);
        var calculatorHigh = new QuantileLossFitnessCalculator<double, Vector<double>, Vector<double>>(quantile: 0.9);

        // Under-prediction scenario
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 5.0, 10.0, 15.0 }),
            Actual = new Vector<double>(new double[] { 10.0, 20.0, 30.0 })
        };

        // Act
        var resultLow = calculatorLow.CalculateFitnessScore(dataSet);
        var resultHigh = calculatorHigh.CalculateFitnessScore(dataSet);

        // Assert - High quantile should penalize under-prediction more
        Assert.True(resultHigh > resultLow);
    }
}
