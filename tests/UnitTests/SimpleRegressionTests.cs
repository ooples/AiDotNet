using AiDotNet.Models;
using AiDotNet.OutlierRemoval;
using AiDotNet.Quartile;
using AiDotNet.Regression;

namespace AiDotNetUnitTests.UnitTests;

public class SimpleRegressionTests
{
    private readonly double[] _inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    private readonly double[] _OddOddInputs = new double[] { 75, 285, 126, 116, 156, 320, 186, 208, 144, 183, 28, 69, 106, 74, 201, 84, 48, 249, 102, 228, 60, 40, 39, 186, 28, 172, 150, 156, 9, 12, 192, 120, 90, 222, 12, 140 };
    private readonly double[] _OddOddOutputs = new double[] { 97, 72, 184, 320, 28, 304, 21, 52, 132, 198, 45, 98, 220, 188, 31, 294, 324, 195, 102, 90, 270, 122, 380, 20, 148, 183, 152, 90, 93, 48, 192, 4, 80, 198, 108, 138 };

    private readonly double[] _OddEvenInputs = new double[] { 168, 14, 20, 112, 100, 158, 55, 207, 150, 198, 66, 268, 252, 136, 66, 5, 14, 50, 368, 184, 171, 288, 136, 90, 282, 46, 43, 216, 76, 15, 177, 116, 42, 152, 130, 176, 100, 276 };
    private readonly double[] _OddEvenOutputs = new double[] { 16, 54, 320, 87, 261, 20, 240, 171, 148, 16, 99, 44, 34, 272, 71, 44, 27, 188, 152, 29, 213, 38, 292, 188, 11, 396, 196, 100, 82, 97, 104, 141, 146, 65, 135, 194, 17, 150 };

    private readonly double[] _EvenOddInputs = new double[] { 90, 57, 78, 25, 36, 60, 94, 270, 380, 240, 104, 15, 268, 261, 8, 219, 78, 74, 42, 63, 80, 51, 98, 116, 43, 332, 328, 65, 264, 380, 76, 260, 29, 231, 240, 45, 90, 180, 60, 84 };
    private readonly double[] _EvenOddOutputs = new double[] { 280, 258, 68, 91, 110, 40, 188, 258, 292, 246, 60, 304, 180, 17, 114, 11, 64, 88, 74, 165, 84, 18, 60, 48, 2, 320, 42, 224, 64, 58, 204, 134, 210, 60, 172, 104, 54, 71, 176, 340 };

    private readonly double[] _EvenEvenInputs = new double[] { 98, 140, 148, 23, 172, 171, 64, 28, 213, 20, 294, 224, 2, 45, 90, 10, 123, 260, 178, 8, 45, 44, 180, 33, 70, 18, 158, 183, 272, 86, 68, 10, 8, 81, 117, 12, 30, 220, 200, 34, 60, 43 };
    private readonly double[] _EvenEvenOutputs = new double[] { 46, 224, 170, 22, 164, 78, 54, 63, 376, 40, 92, 180, 172, 18, 33, 12, 64, 89, 186, 42, 30, 32, 81, 126, 93, 134, 156, 12, 176, 136, 372, 82, 2, 168, 43, 76, 78, 52, 284, 49, 33, 98 };

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

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddOdd_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 117.55507280870674;
        var q2ExpectedValue = 125.05847199082586;
        var q3ExpectedValue = 134.86129995456216;

        // Act
        var sut = new SimpleRegression(_OddOddInputs, _OddOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile())});

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_OddEven_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 88.173752177739459;
        var q2ExpectedValue = 128.84172997826749;
        var q3ExpectedValue = 174.75718878531529;

        // Act
        var sut = new SimpleRegression(_OddEvenInputs, _OddEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_EvenOdd_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 142.70566255778118;
        var q2ExpectedValue = 159.12400616332820;
        var q3ExpectedValue = 236.52476887519259;

        // Act
        var sut = new SimpleRegression(_EvenOddInputs, _EvenOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_EvenEven_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 29.183869666497486;
        var q2ExpectedValue = 75.697713334559751;
        var q3ExpectedValue = 212.13832142754239;

        // Act
        var sut = new SimpleRegression(_EvenEvenInputs, _EvenEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddOdd_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 118.52325334833502;
        var q2ExpectedValue = 125.54256226064001;
        var q3ExpectedValue = 134.86129995456216;

        // Act
        var sut = new SimpleRegression(_OddOddInputs, _OddOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_OddEven_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue =  97.356843939149;
        var q2ExpectedValue = 132.7773407331573;
        var q3ExpectedValue = 185.25215079835479;

        // Act
        var sut = new SimpleRegression(_OddEvenInputs, _OddEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_EvenOdd_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 145.83296610169492;
        var q2ExpectedValue = 159.1240061633282;
        var q3ExpectedValue = 236.52476887519259;

        // Act
        var sut = new SimpleRegression(_EvenOddInputs, _EvenOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_EvenEven_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 29.183869666497486;
        var q2ExpectedValue = 75.697713334559751;
        var q3ExpectedValue = 212.13832142754239;

        // Act
        var sut = new SimpleRegression(_EvenEvenInputs, _EvenEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddOdd_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 118.52325334833502;
        var q2ExpectedValue = 125.54256226064001;
        var q3ExpectedValue = 135.22436765692277;

        // Act
        var sut = new SimpleRegression(_OddOddInputs, _OddOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_OddEven_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 88.173752177739459;
        var q2ExpectedValue = 132.7773407331573;
        var q3ExpectedValue = 185.25215079835479;

        // Act
        var sut = new SimpleRegression(_OddEvenInputs, _OddEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_EvenOdd_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 142.70566255778118;
        var q2ExpectedValue = 159.1240061633282;
        var q3ExpectedValue = 244.08241910631742;

        // Act
        var sut = new SimpleRegression(_EvenOddInputs, _EvenOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_correct_EvenEven_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 29.183869666497486;
        var q2ExpectedValue = 75.697713334559751;
        var q3ExpectedValue = 212.13832142754239;

        // Act
        var sut = new SimpleRegression(_EvenEvenInputs, _EvenEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }
}