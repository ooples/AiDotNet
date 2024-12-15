using AiDotNet.Models;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class WeightedRegressionTests
{
    private readonly double[] _inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _weights = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    
    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentNullException_When_Inputs_Is_Null()
    {
        // Arrange
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new WeightedRegression<double>(new WeightedRegressionOptions<double>() { Weights = _weights, Order = order }));
    }

    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentNullException_When_Outputs_Is_Null()
    {
        // Arrange
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new WeightedRegression<double>(new WeightedRegressionOptions<double>() { Weights = _weights, Order = order }));
    }

    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Big()
    {
        // Arrange
        const int tooBigTrainingSize = 110;
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new WeightedRegression(_inputs, _outputs, _weights, order, new MultipleRegressionOptions() { TrainingPctSize = tooBigTrainingSize }));
    }

    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Small()
    {
        // Arrange
        const int tooSmallTrainingSize = 0;
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new WeightedRegression(_inputs, _outputs, _weights, order, new MultipleRegressionOptions() { TrainingPctSize = tooSmallTrainingSize }));
    }

    [Fact]
    public void WeightedRegression_Constructor_Returns_Valid_Predictions_With_No_Options()
    {
        // Arrange
        var expectedPredictions = new double[] { 0, 52, 0, 0, 0, 0, 0, 0 };
        const int order = 2;

        // Act
        var weightedRegression = new WeightedRegression(_inputs, _outputs, _weights, order);
        var actualPredictions = weightedRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }
}