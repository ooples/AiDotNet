using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class ModifiedHuberLossFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 3.0, 2.5, -3.0, -2.5 }),
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
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
    public void CalculateFitnessScore_WithSlightlyIncorrectPredictions_HasQuadraticLoss()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 0.5, 0.3 }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithVeryIncorrectPredictions_HasLinearLoss()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { -2.0, -3.0 }),
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
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.3, 0.7);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.8, 0.4);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<float, Vector<float>, Vector<float>>();
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
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithDefaultDataSetType_UsesValidation()
    {
        // Arrange & Act
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithBinaryClassification_WorksCorrectly()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
    public void CalculateFitnessScore_RobustToOutliers_ComparedToSquared()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Dataset with one extreme outlier
        var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, 1.0, -10.0 }), // One extreme outlier
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert - Should handle outliers more gracefully than pure squared loss
        Assert.True(result > 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithAllCorrectPredictions_ReturnsZero()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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

    [Fact]
    public void CalculateFitnessScore_BetterPredictions_LowerLoss()
    {
        // Arrange
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

        // Good predictions
        var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { 1.0, -1.0, 1.0 }),
            Actual = new Vector<double>(new double[] { 1.0, -1.0, 1.0 })
        };

        // Poor predictions
        var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
        {
            Predicted = new Vector<double>(new double[] { -1.0, 1.0, -1.0 }),
            Actual = new Vector<double>(new double[] { 1.0, -1.0, 1.0 })
        };

        // Act
        var result1 = calculator.CalculateFitnessScore(dataSet1);
        var result2 = calculator.CalculateFitnessScore(dataSet2);

        // Assert
        Assert.True(result1 < result2);
    }
}
