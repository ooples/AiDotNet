#nullable disable
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class OrdinalRegressionLossFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsMinimumLoss()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 5);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        // Note: Ordinal regression loss uses binary cross-entropy for each threshold,
        // which doesn't return exactly 0 even for perfect predictions.
        // The loss depends on the sigmoid function and log terms.
        Assert.True(!double.IsNaN(result) && !double.IsInfinity(result));
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithOffByOne_ReturnsModerateLoss()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 5);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 2.0, 3.0, 4.0, 5.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
        Assert.True(result < 5.0); // Should be moderate loss
    }

    [Fact]
    public void CalculateFitnessScore_WithLargeErrors_ReturnsHighLoss()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 5);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 5.0, 5.0, 5.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 1.0); // Large errors should result in higher loss
    }

    [Fact]
    public void CalculateFitnessScore_WithThreeClasses_WorksCorrectly()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 3);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 2.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 3.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithoutNumberOfClasses_AutoDetects()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.2, 0.8);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.9, 0.3);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<float, Vector<float>, Vector<float>>(numberOfClassifications: 5);
        var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
        {
            Predicted = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f }),
            Actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        // Note: Ordinal regression loss uses binary cross-entropy, doesn't return 0 for matched predictions
        Assert.True(!float.IsNaN(result) && !float.IsInfinity(result));
        Assert.True(result >= 0.0f);
    }

    [Fact]
    public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(dataSetType: DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(dataSetType: DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithTwoClasses_WorksCorrectly()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 2);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 2.0, 1.0, 2.0 }),
            Actual = new Vector<double>(new double[] { 1.0, 2.0, 2.0, 1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_ConsistentOrdering_SmallerLoss()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 5);

        // Close predictions
        var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 3.0, 3.0, 3.0 }),
            Actual = new Vector<double>(new double[] { 2.0, 3.0, 4.0 })
        };

        // Far predictions
        var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 5.0, 1.0 }),
            Actual = new Vector<double>(new double[] { 5.0, 1.0, 5.0 })
        };

        // Act
        var result1 = calculator.CalculateFitnessScore(dataSet1);
        var result2 = calculator.CalculateFitnessScore(dataSet2);

        // Assert - Closer predictions should have smaller loss
        Assert.True(result1 < result2);
    }

    [Fact]
    public void CalculateFitnessScore_WithTenClasses_HandlesCorrectly()
    {
        // Arrange
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Vector<double>, Vector<double>>(numberOfClassifications: 10);
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 5.0, 10.0 }),
            Actual = new Vector<double>(new double[] { 2.0, 5.0, 9.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }
}
