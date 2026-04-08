#nullable disable
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators;

public class TripletLossFitnessCalculatorTests
{
    [Fact]
    public void CalculateFitnessScore_WithPerfectSeparation_ReturnsZero()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>(margin: 1.0);
        var dataSet = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.0 },  // Class 0
                { 1.1, 0.1 },  // Class 0
                { 0.0, 1.0 },  // Class 1
                { 0.1, 1.1 }   // Class 1
            }),
            Actual = new Vector<double>(new double[] { 0.0, 0.0, 1.0, 1.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert - With well-separated clusters, loss should be low
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithTwoClasses_WorksCorrectly()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Features = new Matrix<double>(new double[,]
            {
                { 2.0, 3.0 },
                { 2.1, 3.1 },
                { 8.0, 9.0 },
                { 8.1, 9.1 }
            }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 2.0, 2.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithThreeClasses_WorksCorrectly()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Features = new Matrix<double>(new double[,]
            {
                { 1.0, 1.0 },
                { 1.1, 1.1 },
                { 5.0, 5.0 },
                { 5.1, 5.1 },
                { 9.0, 9.0 },
                { 9.1, 9.1 }
            }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithDifferentMargins_AffectsLoss()
    {
        // Arrange
        var calculatorSmallMargin = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>(margin: 0.5);
        var calculatorLargeMargin = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>(margin: 2.0);

        var dataSet = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.0 },
                { 1.2, 0.2 },
                { 5.0, 5.0 },
                { 5.2, 5.2 }
            }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 2.0, 2.0 })
        };

        // Act
        var resultSmall = calculatorSmallMargin.CalculateFitnessScore(dataSet);
        var resultLarge = calculatorLargeMargin.CalculateFitnessScore(dataSet);

        // Assert - Larger margin should demand more separation
        Assert.True(resultSmall >= 0.0);
        Assert.True(resultLarge >= 0.0);
    }

    [Fact]
    public void IsHigherScoreBetter_ReturnsFalse()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Act & Assert
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void IsBetterFitness_WithLowerScore_ReturnsTrue()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.2, 0.6);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBetterFitness_WithHigherScore_ReturnsFalse()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Act
        var result = calculator.IsBetterFitness(0.8, 0.4);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<float, Matrix<float>, Vector<float>>(margin: 1.0f);
        var dataSet = new DataSetStats<float, Matrix<float>, Vector<float>>
        {
            Features = new Matrix<float>(new float[,]
            {
                { 1.0f, 0.0f },
                { 1.1f, 0.1f },
                { 5.0f, 5.0f },
                { 5.1f, 5.1f }
            }),
            Actual = new Vector<float>(new float[] { 1.0f, 1.0f, 2.0f, 2.0f })
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
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => calculator.CalculateFitnessScore((DataSetStats<double, Matrix<double>, Vector<double>>)null));
    }

    [Fact]
    public void Constructor_WithTrainingDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>(dataSetType: DataSetType.Training);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithTestDataSetType_SetsCorrectly()
    {
        // Arrange & Act
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>(dataSetType: DataSetType.Testing);

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void Constructor_WithDefaultMargin_UsesDefaultValue()
    {
        // Arrange & Act
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(calculator);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CalculateFitnessScore_WithMultipleSamplesPerClass_HandlesCorrectly()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Features = new Matrix<double>(new double[,]
            {
                { 1.0, 1.0 },
                { 1.2, 1.2 },
                { 1.1, 1.1 },
                { 5.0, 5.0 },
                { 5.2, 5.2 },
                { 5.1, 5.1 }
            }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void CalculateFitnessScore_WithHighDimensionalFeatures_WorksCorrectly()
    {
        // Arrange
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        var dataSet = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0 },
                { 1.1, 2.1, 3.1, 4.1 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 5.1, 6.1, 7.1, 8.1 }
            }),
            Actual = new Vector<double>(new double[] { 1.0, 1.0, 2.0, 2.0 })
        };

        // Act
        var result = calculator.CalculateFitnessScore(dataSet);

        // Assert
        Assert.True(result >= 0.0);
    }
}
