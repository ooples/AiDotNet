using AiDotNet.Models;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class WeightedRegressionTests
{
    private readonly double[][] _inputs = new double[][] { new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 } };
    private readonly double[] _outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _weights = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    
    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentNullException_When_Inputs_Is_Null()
    {
        // Arrange

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new WeightedRegression(null, _outputs, _weights));
    }

    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentNullException_When_Outputs_Is_Null()
    {
        // Arrange

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new WeightedRegression(_inputs, null, _weights));
    }

    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Big()
    {
        // Arrange
        const int tooBigTrainingSize = 110;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new WeightedRegression(_inputs, _outputs, _weights, new MultipleRegressionOptions() { TrainingPctSize = tooBigTrainingSize }));
    }

    [Fact]
    public void WeightedRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Small()
    {
        // Arrange
        const int tooSmallTrainingSize = 0;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new WeightedRegression(_inputs, _outputs, _weights, new MultipleRegressionOptions() { TrainingPctSize = tooSmallTrainingSize }));
    }

    [Fact]
    public void WeightedRegression_Constructor_Returns_Valid_Predictions_With_No_Options()
    {
        // Arrange
        var expectedPredictions = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        // Act
        var weightedRegression = new WeightedRegression(_inputs, _outputs, _weights);
        var actualPredictions = weightedRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }
}