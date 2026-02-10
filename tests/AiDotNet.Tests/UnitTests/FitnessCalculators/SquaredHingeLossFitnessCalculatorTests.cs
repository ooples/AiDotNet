#nullable disable
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class SquaredHingeLossFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 1.0, -1.0, -1.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, -1.0, -1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithConfidentCorrectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 2.0, 3.0, -2.0, -3.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, -1.0, -1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void CalculateFitnessScore_WithIncorrectPredictions_ReturnsPositiveValue()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { -1.0, -1.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithMixedPredictions_HandlesCorrectly()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, -1.0, 0.5, -0.5 }),
            Actual = new Vector<double>(new double[] { 1.0, -1.0, 1.0, -1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_SquaresPenalty_LargeErrorsMorePenalized()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Small error
        var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.8 }),
            Actual = new Vector<double>(new double[] { 1.0 })
        };

        // Large error
        var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { -1.0 }),
            Actual = new Vector<double>(new double[] { 1.0 })
        };

        // Act
        var result1 = calculator.CalculateFitnessScore(dataSet1);
        var result2 = calculator.CalculateFitnessScore(dataSet2);

        // Assert - Larger errors should be more heavily penalized due to squaring
        Assert.True(result2 > result1 * 4); // Squared relationship
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.1, 0.5);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.7, 0.2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<float, Vector<float>, Vector<float>>();
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            Predicted = new Vector<float>(new float[] { 1.0f, -1.0f }),
            Actual = new Vector<float>(new float[] { 1.0f, -1.0f })
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
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithBinaryClassification_WorksCorrectly()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 1.0, -1.0, -1.0, 1.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, -1.0, -1.0, -1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0); // One misclassification
    }

    [Fact]
    public void CalculateFitnessScore_WithUncertainPredictions_PenalizesCorrectly()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Confident correct predictions
        var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 2.0, -2.0 }),
            Actual = new Vector<double>(new double[] { 1.0, -1.0 })
        };

        // Uncertain predictions
        var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.5, -0.5 }),
            Actual = new Vector<double>(new double[] { 1.0, -1.0 })
        };

        // Act
        var result1 = calculator.CalculateFitnessScore(dataSet1);
        var result2 = calculator.CalculateFitnessScore(dataSet2);

        // Assert - Uncertain predictions should have higher loss
        Assert.True(result2 > result1);
    }

    [Fact]
    public void CalculateFitnessScore_WithAllCorrectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 5.0, 3.0, -4.0, -2.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, -1.0, -1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 10);
    }
}
