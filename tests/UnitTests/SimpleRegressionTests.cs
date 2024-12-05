using AiDotNet.Models;
using AiDotNet.Normalization;
using AiDotNet.OutlierRemoval;
using AiDotNet.Quartile;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class SimpleRegressionTests
{
    private readonly double[] _inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    private readonly double[] _OddEvenInputs = new double[] { 75, 285, 126, 116, 156, 320, 186, 208, 144, 183, 28, 69, 106, 74, 201, 84, 48, 249, 102, 228, 60, 40, 39, 186, 28, 172, 150, 156, 9, 12, 192, 120, 90, 222, 12, 140 };
    private readonly double[] _OddEvenOutputs = new double[] { 97, 72, 184, 320, 28, 304, 21, 52, 132, 198, 45, 98, 220, 188, 31, 294, 324, 195, 102, 90, 270, 122, 380, 20, 148, 183, 152, 90, 93, 48, 192, 4, 80, 198, 108, 138 };

    private readonly double[] _OddOddInputs = new double[] { 168, 14, 20, 112, 100, 158, 55, 207, 150, 198, 66, 268, 252, 136, 66, 5, 14, 50, 368, 184, 171, 288, 136, 90, 282, 46, 43, 216, 76, 15, 177, 116, 42, 152, 130, 176, 100, 276 };
    private readonly double[] _OddOddOutputs = new double[] { 16, 54, 320, 87, 261, 20, 240, 171, 148, 16, 99, 44, 34, 272, 71, 44, 27, 188, 152, 29, 213, 38, 292, 188, 11, 396, 196, 100, 82, 97, 104, 141, 146, 65, 135, 194, 17, 150 };

    private readonly double[] _EvenEvenInputs = new double[] { 90, 57, 78, 25, 36, 60, 94, 270, 380, 240, 104, 15, 268, 261, 8, 219, 78, 74, 42, 63, 80, 51, 98, 116, 43, 332, 328, 65, 264, 380, 76, 260, 29, 231, 240, 45, 90, 180, 60, 84, 31, 112, 40, 300, 205};
    private readonly double[] _EvenEvenOutputs = new double[] { 280, 258, 68, 91, 110, 40, 188, 258, 292, 246, 60, 304, 180, 17, 114, 11, 64, 88, 74, 165, 84, 18, 60, 48, 2, 320, 42, 224, 64, 58, 204, 134, 210, 60, 172, 104, 54, 71, 176, 340, 63, 44, 1, 290, 60};

    private readonly double[] _EvenOddInputs = new double[] { 340, 316, 270, 122, 189, 54, 280, 12, 32, 62, 51, 76, 16, 243, 70, 66, 90, 104, 288, 236, 6, 152, 76, 90, 69, 188, 208, 59, 136, 16, 208, 336, 264, 77, 116, 75, 111, 38, 148 };
    private readonly double[] _EvenOddOutputs = new double[] { 210, 132, 90, 48, 400, 178, 72, 84, 79, 122, 152, 96, 261, 22, 92, 80, 18, 122, 9, 1, 232, 220, 4, 396, 16, 16, 147, 33, 198, 45, 3, 82, 84, 18, 41, 234, 64, 32, 29 };

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
    public void SimpleRegression_Constructor_Returns_Valid_Predictions_With_DecimalNormalization()
    {
        // Arrange
        var expectedPredictions = new double[] { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 };

        // Act
        var simpleRegression = new SimpleRegression(_inputs, _outputs, new SimpleRegressionOptions { Normalization = new DecimalNormalization()});
        var actualPredictions = simpleRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }

    [Fact]
    public void SimpleRegression_Constructor_Returns_Valid_Predictions_With_MinMaxNormalization()
    {
        // Arrange
        var expectedPredictions = new double[] { 2, 3, 4, 5, 6, 7, 8, 9 };

        // Act
        var simpleRegression = new SimpleRegression(_inputs, _outputs, new SimpleRegressionOptions { Normalization = new MinMaxNormalizer() });
        var actualPredictions = simpleRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }

    [Fact]
    public void SimpleRegression_Constructor_Returns_Valid_Predictions_With_LogNormalization()
    {
        // Arrange
        var expectedPredictions = new double[] { 1.0986122886681098, 1.3862943611198906, 1.6094379124341003, 1.791759469228055, 1.9459101490553132, 
            2.0794415416798357, 2.1972245773362196, 2.302585092994046 };

        // Act
        var simpleRegression = new SimpleRegression(_inputs, _outputs, new SimpleRegressionOptions { Normalization = new LogNormalization() });
        var actualPredictions = simpleRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }

    [Fact]
    public void SimpleRegression_Constructor_Returns_Valid_Predictions_With_ZScoreNormalization()
    {
        // Arrange
        var expectedPredictions = new double[] { -1.5275252316519468, -1.091089451179962, -0.6546536707079772, -0.2182178902359924, 
            0.2182178902359924, 0.6546536707079772, 1.091089451179962, 1.5275252316519468 };

        // Act
        var simpleRegression = new SimpleRegression(_inputs, _outputs, new SimpleRegressionOptions { Normalization = new ZScoreNormalizer() });
        var actualPredictions = simpleRegression.Predictions;

        // Assert
        Assert.Equal(expectedPredictions, actualPredictions);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddEven_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 152.0795839065193;
        var q2ExpectedValue = 171.77754652688836;
        var q3ExpectedValue = 195.05695689641541;

        // Act
        var sut = new SimpleRegression(_OddEvenInputs, _OddEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddEven_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 152.0795839065193;
        var q2ExpectedValue = 162.82392715399334;
        var q3ExpectedValue = 195.05695689641541;

        // Act
        var sut = new SimpleRegression(_OddEvenInputs, _OddEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddEven_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 156.85484757206331;
        var q2ExpectedValue = 171.77754652688836;
        var q3ExpectedValue = 195.05695689641541;

        // Act
        var sut = new SimpleRegression(_OddEvenInputs, _OddEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

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
        var q1ExpectedValue = 118.53688757999335;
        var q2ExpectedValue = 138.76510111583207;
        var q3ExpectedValue = 162.96333786898498;

        // Act
        var sut = new SimpleRegression(_OddOddInputs, _OddOddOutputs, new SimpleRegressionOptions { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_OddOdd_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 118.53688757999335;
        var q2ExpectedValue = 138.76510111583207;
        var q3ExpectedValue = 151.99851184021256;

        // Act
        var sut = new SimpleRegression(_OddOddInputs, _OddOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

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
        var q1ExpectedValue = 123.26310569584352;
        var q2ExpectedValue = 138.76510111583207;
        var q3ExpectedValue = 151.99851184021256;

        // Act
        var sut = new SimpleRegression(_OddOddInputs, _OddOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_EvenEven_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 123.72734667709028;
        var q2ExpectedValue = 128.71082261723302;
        var q3ExpectedValue = 140.7328161223196;

        // Act
        var sut = new SimpleRegression(_EvenEvenInputs, _EvenEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_EvenEven_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 123.72734667709028;
        var q2ExpectedValue = 128.71082261723302;
        var q3ExpectedValue = 140.7328161223196;

        // Act
        var sut = new SimpleRegression(_EvenEvenInputs, _EvenEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_EvenEven_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 123.72734667709028;
        var q2ExpectedValue = 128.71082261723302;
        var q3ExpectedValue = 140.7328161223196;

        // Act
        var sut = new SimpleRegression(_EvenEvenInputs, _EvenEvenOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_EvenOdd_ExclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 111.72163717784437;
        var q2ExpectedValue = 123.79567181111241;
        var q3ExpectedValue = 129.41379404863304;

        // Act
        var sut = new SimpleRegression(_EvenOddInputs, _EvenOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new ExclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_EvenOdd_StandardQuartile()
    {
        // Arrange
        var q1ExpectedValue = 111.72163717784437;
        var q2ExpectedValue = 123.79567181111241;
        var q3ExpectedValue = 127.49180486211283;

        // Act
        var sut = new SimpleRegression(_EvenOddInputs, _EvenOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new StandardQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }

    [Fact]
    public void SimpleRegression_Verify_Values_Are_Correct_EvenOdd_InclusiveQuartile()
    {
        // Arrange
        var q1ExpectedValue = 117.43832296339168;
        var q2ExpectedValue = 123.79567181111241;
        var q3ExpectedValue = 127.49180486211283;

        // Act
        var sut = new SimpleRegression(_EvenOddInputs, _EvenOddOutputs, new SimpleRegressionOptions() { OutlierRemoval = new IQROutlierRemoval(new InclusiveQuartile()) });

        // Assert
        var metrics = sut.Metrics;
        Assert.Equal(q1ExpectedValue, metrics.Quartile1Value);
        Assert.Equal(q2ExpectedValue, metrics.Quartile2Value);
        Assert.Equal(q3ExpectedValue, metrics.Quartile3Value);
    }
}