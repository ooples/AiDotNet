using AiDotNet.LinearAlgebra;
using AiDotNet.UncertaintyQuantification.Calibration;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

public class ExpectedCalibrationErrorTests
{
    [Fact]
    public void Constructor_WithDefaultBins_CreatesInstance()
    {
        // Arrange & Act
        var ece = new ExpectedCalibrationError<double>();

        // Assert
        Assert.NotNull(ece);
    }

    [Fact]
    public void Constructor_WithCustomBins_CreatesInstance()
    {
        // Arrange & Act
        var ece = new ExpectedCalibrationError<double>(numBins: 20);

        // Assert
        Assert.NotNull(ece);
    }

    [Fact]
    public void Constructor_WithZeroBins_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new ExpectedCalibrationError<double>(numBins: 0));
    }

    [Fact]
    public void Constructor_WithNegativeBins_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new ExpectedCalibrationError<double>(numBins: -5));
    }

    [Fact]
    public void Compute_WithPerfectCalibration_ReturnsZero()
    {
        // Arrange
        var ece = new ExpectedCalibrationError<double>(numBins: 10);

        // Create perfectly calibrated data (confidence matches empirical accuracy)
        var probabilities = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 });
        var predictions = new Vector<int>(new int[] { 1, 1, 1, 1 });
        var trueLabels = new Vector<int>(new int[] { 1, 1, 0, 0 });

        // Act
        var eceValue = ece.Compute(probabilities, predictions, trueLabels);

        // Assert
        Assert.Equal(0.0, eceValue, precision: 2);
    }

    [Fact]
    public void Compute_WithIncorrectPredictions_ReturnsPositiveECE()
    {
        // Arrange
        var ece = new ExpectedCalibrationError<double>(numBins: 10);

        // Create data with high confidence but wrong predictions
        var probabilities = new Vector<double>(new double[] { 0.9, 0.9, 0.9, 0.9, 0.9 });
        var predictions = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });
        var trueLabels = new Vector<int>(new int[] { 0, 0, 0, 0, 0 }); // All wrong

        // Act
        var eceValue = ece.Compute(probabilities, predictions, trueLabels);

        // Assert - should have high ECE due to overconfidence
        Assert.True(eceValue > 0.5);
    }

    [Fact]
    public void Compute_WithMismatchedLengths_ThrowsException()
    {
        // Arrange
        var ece = new ExpectedCalibrationError<double>();
        var probabilities = new Vector<double>(new double[] { 0.9, 0.8 });
        var predictions = new Vector<int>(new int[] { 1, 1, 1 }); // Different length
        var trueLabels = new Vector<int>(new int[] { 1, 0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => { ece.Compute(probabilities, predictions, trueLabels); });
    }

    [Fact]
    public void GetReliabilityDiagram_ReturnsNonEmptyData()
    {
        // Arrange
        var ece = new ExpectedCalibrationError<double>(numBins: 5);
        var probabilities = new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05 });
        var predictions = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 });
        var trueLabels = new Vector<int>(new int[] { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 });

        // Act
        var diagram = ece.GetReliabilityDiagram(probabilities, predictions, trueLabels);

        // Assert
        Assert.NotNull(diagram);
        Assert.NotEmpty(diagram);

        // Each entry should have valid values
        foreach (var (confidence, accuracy, count) in diagram)
        {
            Assert.True(confidence >= 0.0 && confidence <= 1.0);
            Assert.True(accuracy >= 0.0 && accuracy <= 1.0);
            Assert.True(count > 0);
        }
    }
}
