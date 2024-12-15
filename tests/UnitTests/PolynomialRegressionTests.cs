using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class PolynomialRegressionTests
{
    private readonly double[] _inputs = new double[] { 171, 183, 12, 261, 77, 272, 36, 6, 213, 4, 74, 79, 158, 60, 24, 99, 292, 30, 176, 276, 285, 38, 64, 21, 37, 258, 141, 46, 48, 128, 165, 74, 102, 6, 53, 23, 56, 236, 104, 96, 228, 216, 116, 160, 38, 106, };
    private readonly double[] _outputs = new double[] { 144, 87, 216, 111, 49, 300, 96, 138, 165, 164, 62, 60, 31, 324, 368, 76, 246, 138, 57, 76, 66, 116, 128, 4, 130, 73, 372, 73, 12, 16, 20, 46, 7, 280, 106, 27, 35, 126, 100, 91, 156, 14, 14, 48, 81, 75, };

    [Fact]
    public void PolynomialRegression_Constructor_Throws_ArgumentNullException_When_Inputs_Is_Null()
    {
        // Arrange
        const int order = 2;

        // Act

        // Assert
        Assert.Throws<ArgumentNullException>(() => new PolynomialRegression<double>(_outputs, order));
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
        var polynomialRegression = new PolynomialRegression(_inputs, _outputs, order, new MultipleRegressionOptions() { MatrixDecomposition = AiDotNet.Enums.MatrixDecompositionType.GramSchmidt });
        var actualPredictions = polynomialRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }

    [Fact]
    public void PolynomialRegression_TestCalculationOf_CalculateDeterminantRecursive()
    {
        // Arrange
        const double expectedPredictions1 = 900.014766;
        double actualPredictions1;
        var matrix1 = new double[,] { { 8.34, -5.66, 2.58 }, { -0.74, 5.27, 8.48 }, { -3.97, -8.97, 0.1 } };

        // Act
        //actualPredictions1 = Helpers

        // Assert
        //Assert.Equal(expectedPredictions1, actualPredictions1);
    }

    [Fact]
    public void PolynomialRegression_TestCalculationOf_CalculateCramerMethod()
    {
        // Arrange 
        var expectedPredictions = new double[] { 0.229065559, -0.162800413, 0.027763966 }; //really I mean coefficients here I just copied and pasted.
        var order = 2;
        var actualPredictions = new double[order + 1];
        var inputs = new double[] { -3, -2, -1, -0.2, 1, 3 };
        var outputs = new double[] { 0.9, 0.8, 0.4, 0.2, 0.1, 0 };

        // Act
        var polynomialRegressionCramer = new PolynomialRegression(inputs, outputs, order, new MultipleRegressionOptions() { MatrixDecomposition = AiDotNet.Enums.MatrixDecompositionType.Cramer });
        actualPredictions = polynomialRegressionCramer.Predictions;

        // Assert
        Assert.Equal(expectedPredictions[0], actualPredictions[0]);
        Assert.Equal(expectedPredictions[1], actualPredictions[1]);
        Assert.Equal(expectedPredictions[2], actualPredictions[2]);
    }

}