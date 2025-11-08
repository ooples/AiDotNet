using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class WeightedCrossEntropyLossFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.Equal(0.0, result, 5);
    }

    [Fact]
    public void CalculateFitnessScore_WithDifferentPredictions_ReturnsPositiveValue()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.1 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithoutWeights_UsesDefaultWeights()
    {
        // Arrange
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.9, 0.1 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithHighWeights_IncreasesLoss()
    {
        // Arrange
        var lowWeights = new Vector<double>(new double[] { 0.5, 0.5 });
        var highWeights = new Vector<double>(new double[] { 2.0, 2.0 });

        var calculatorLow = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(lowWeights);
        var calculatorHigh = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(highWeights);

        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.6, 0.4 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0 })
        };

        // Act
        var resultLow = calculatorLow.CalculateFitnessScore(dataSet);
        var resultHigh = calculatorHigh.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(resultHigh > resultLow);
    }

    [Fact]
    public void CalculateFitnessScore_WithImbalancedWeights_AffectsLoss()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 5.0, 1.0 });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.5, 0.5 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.25, 0.75);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.85, 0.35);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var weights = new Vector<float>(new float[] { 1.0f, 1.0f });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<float, Vector<float>, Vector<float>>(weights);
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            Predicted = new Vector<float>(new float[] { 1.0f, 0.0f }),
            Actual = new Vector<float>(new float[] { 1.0f, 0.0f })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0f);
    }

    [Fact]
    public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
    {
        // Arrange
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(dataSetType: DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(dataSetType: DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithMultiClassPredictions_HandlesCorrectly()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 1.0 });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.05, 0.05 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithEqualProbabilities_HighLoss()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 1.0, 1.0 });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.5, 0.5 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.3); // Should have significant loss for uncertain prediction
    }

    [Fact]
    public void CalculateFitnessScore_WithConfidentPredictions_LowLoss()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 1.0, 1.0 });
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.99, 0.01 }),
            Actual = new Vector<double>(new double[] { 1.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result < 0.1); // Should have low loss for confident correct prediction
    }

    [Fact]
    public void CalculateFitnessScore_AutoCreatesWeights_WhenMismatch()
    {
        // Arrange
        var weights = new Vector<double>(new double[] { 1.0, 1.0 }); // Wrong size
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.8, 0.1, 0.1 }), // Size 3
            Actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert - Should not throw, should auto-create correct size weights
        Assert.True(result >= 0.0);
    }
}
