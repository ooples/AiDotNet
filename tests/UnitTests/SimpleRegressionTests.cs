using AiDotNet.Models;
using AiDotNet.Regression;

namespace AiDotNetUnitTests.UnitTests;

public class SimpleRegressionTests
{
    private readonly double[] _inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    [Fact]
    public void SimpleRegression_Constructor_Throws_ArgumentNullException_When_Inputs_Is_Null()
    {
        // Arrange

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new SimpleRegression(null, _outputs));
    }

    [Fact]
    public void SimpleRegression_Constructor_Throws_ArgumentNullException_When_Outputs_Is_Null()
    {
        // Arrange

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new SimpleRegression(_inputs, null));
    }

    [Fact]
    public void SimpleRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Big()
    {
        // Arrange
        const int tooBigTrainingSize = 110;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new SimpleRegression(_inputs, _outputs, new SimpleRegressionOptions() { TrainingPctSize = tooBigTrainingSize }));
    }

    [Fact]
    public void SimpleRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Small()
    {
        // Arrange
        const int tooSmallTrainingSize = 0;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new SimpleRegression(_inputs, _outputs, new SimpleRegressionOptions() { TrainingPctSize = tooSmallTrainingSize }));
    }

    [Fact]
    public void SimpleRegression_Constructor_Returns_Valid_Predictions_With_No_Options()
    {
        // Arrange
        var expectedPredictions = new double[] { 3, 4, 5, 6, 7, 8, 9, 10 };

        // Act
        var simpleRegression = new SimpleRegression(_inputs, _outputs);
        var actualPredictions = simpleRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }
}