using AiDotNet.LinearAlgebra;
using AiDotNet.UncertaintyQuantification.Calibration;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

public class TemperatureScalingTests
{
    [Fact]
    public void Constructor_WithDefaultTemperature_InitializesToOne()
    {
        // Arrange & Act
        var tempScaling = new TemperatureScaling<double>();

        // Assert
        Assert.Equal(1.0, tempScaling.Temperature);
    }

    [Fact]
    public void Constructor_WithCustomTemperature_InitializesCorrectly()
    {
        // Arrange & Act
        var tempScaling = new TemperatureScaling<double>(2.0);

        // Assert
        Assert.Equal(2.0, tempScaling.Temperature);
    }

    [Fact]
    public void Temperature_SetToNegative_ThrowsException()
    {
        // Arrange
        var tempScaling = new TemperatureScaling<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tempScaling.Temperature = -1.0);
    }

    [Fact]
    public void Temperature_SetToZero_ThrowsException()
    {
        // Arrange
        var tempScaling = new TemperatureScaling<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tempScaling.Temperature = 0.0);
    }

    [Fact]
    public void ScaleLogits_WithTemperatureOne_ReturnsUnchangedLogits()
    {
        // Arrange
        var tempScaling = new TemperatureScaling<double>(1.0);
        var logits = new Tensor<double>([3], new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));

        // Act
        var scaled = tempScaling.ScaleLogits(logits);

        // Assert
        Assert.NotNull(scaled);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(logits[i], scaled[i], precision: 6);
        }
    }

    [Fact]
    public void ScaleLogits_WithTemperatureTwo_HalvesLogits()
    {
        // Arrange
        var tempScaling = new TemperatureScaling<double>(2.0);
        var logits = new Tensor<double>([3], new Vector<double>(new double[] { 2.0, 4.0, 6.0 }));

        // Act
        var scaled = tempScaling.ScaleLogits(logits);

        // Assert
        Assert.NotNull(scaled);
        Assert.Equal(1.0, scaled[0], precision: 6);
        Assert.Equal(2.0, scaled[1], precision: 6);
        Assert.Equal(3.0, scaled[2], precision: 6);
    }

    [Fact]
    public void ScaleLogits_WithTemperatureHalf_DoublesLogits()
    {
        // Arrange
        var tempScaling = new TemperatureScaling<double>(0.5);
        var logits = new Tensor<double>([3], new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));

        // Act
        var scaled = tempScaling.ScaleLogits(logits);

        // Assert
        Assert.NotNull(scaled);
        Assert.Equal(2.0, scaled[0], precision: 6);
        Assert.Equal(4.0, scaled[1], precision: 6);
        Assert.Equal(6.0, scaled[2], precision: 6);
    }

    [Fact]
    public void Calibrate_WithValidData_UpdatesTemperature()
    {
        // Arrange
        var tempScaling = new TemperatureScaling<double>();
        var logits = new Matrix<double>(5, 3);
        var labels = new Vector<int>(new int[] { 0, 1, 2, 1, 0 });

        // Fill logits with sample data
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                logits[i, j] = (i + j) * 0.5;
            }
        }

        var initialTemp = tempScaling.Temperature;

        // Act
        tempScaling.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 10);

        // Assert - temperature should have changed
        var finalTemp = tempScaling.Temperature;
        Assert.NotEqual(initialTemp, finalTemp);
        Assert.True(finalTemp > 0); // Temperature should remain positive
    }
}
