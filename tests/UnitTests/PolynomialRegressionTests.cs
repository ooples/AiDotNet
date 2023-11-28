using AiDotNet.Models;
using AiDotNet.Normalization;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class PolynomialRegressionTests
{
    private readonly double[] _inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _outputs = new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    [Fact]
    public void PolynomialRegression_Constructor_Throws_ArgumentNullException_When_Inputs_Is_Null()
    {
        // Arrange
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new PolynomialRegression(null, _outputs, order));
    }

    [Fact]
    public void PolynomialRegression_Constructor_Throws_ArgumentNullException_When_Outputs_Is_Null()
    {
        // Arrange
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new PolynomialRegression(_inputs, null, order));
    }

    [Fact]
    public void PolynomialRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Big()
    {
        // Arrange
        const int tooBigTrainingSize = 110;
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new PolynomialRegression(_inputs, _outputs, order, new MultipleRegressionOptions() { TrainingPctSize = tooBigTrainingSize }));
    }

    [Fact]
    public void PolynomialRegression_Constructor_Throws_ArgumentException_When_TrainingSize_Is_Too_Small()
    {
        // Arrange
        const int tooSmallTrainingSize = 0;
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentException>(() => new PolynomialRegression(_inputs, _outputs, order, new MultipleRegressionOptions { TrainingPctSize = tooSmallTrainingSize }));
    }

    [Fact]
    public void PolynomialRegression_Constructor_Returns_Valid_Predictions_With_No_Options()
    {
        // Arrange
        var expectedPredictions = new double[] { 239.19999999999956, 447.20000000000016, -166.39999999999998, 0, 0, 0, 0, 0 };
        const int order = 2;

        // Act
        var polynomialRegression = new PolynomialRegression(_inputs, _outputs, order);
        var actualPredictions = polynomialRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }
}