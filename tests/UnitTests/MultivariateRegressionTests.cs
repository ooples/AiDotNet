using AiDotNet.Models;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class MultivariateRegressionTests
{
    private readonly double[][] _inputs = new double[][] { new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 } };
    private readonly double[][] _outputs = new double[][] { new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 } };

    [Fact]
    public void MultivariateRegression_Constructor_Throws_ArgumentNullException_When_Inputs_Is_Null()
    {
        // Arrange

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new MultivariateRegression(null, _outputs));
    }

    [Fact]
    public void MultivariateRegression_Constructor_Throws_ArgumentNullException_When_Outputs_Is_Null()
    {
        // Arrange

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new MultivariateRegression(_inputs, null));
    }

    [Fact]
    public void MultivariateRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Big()
    {
        // Arrange
        const int tooBigTrainingSize = 110;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new MultivariateRegression(_inputs, _outputs, new MultipleRegressionOptions() { TrainingPctSize = tooBigTrainingSize }));
    }

    [Fact]
    public void MultivariateRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Small()
    {
        // Arrange
        const int tooSmallTrainingSize = 0;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new MultivariateRegression(_inputs, _outputs, new MultipleRegressionOptions() { TrainingPctSize = tooSmallTrainingSize }));
    }

    [Fact]
    public void MultivariateRegression_Constructor_Returns_Valid_Predictions_With_No_Options()
    {
        // Arrange
        var expectedPredictions = new double[][] { new double[] { 3, 4, 5, 6, 7, 8, 9, 10 }, 
            new double[] { 12.2, 26.599999999999998, 40.99999999999999, 55.39999999999999, 69.8, 84.19999999999999, 98.59999999999998, 112.99999999999999 } };

        // Act
        var multivariateRegression = new MultivariateRegression(_inputs, _outputs);
        var actualPredictions = multivariateRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }
}