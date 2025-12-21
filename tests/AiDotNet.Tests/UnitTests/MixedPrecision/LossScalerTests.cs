using AiDotNet.LinearAlgebra;
using AiDotNet.MixedPrecision;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.MixedPrecision;

/// <summary>
/// Unit tests for the LossScaler class.
/// </summary>
public class LossScalerTests
{
    [Fact]
    public void Constructor_WithDefaults_InitializesCorrectly()
    {
        // Arrange & Act
        var scaler = new LossScaler<float>();

        // Assert
        Assert.Equal(65536.0, scaler.Scale);
        Assert.True(scaler.DynamicScaling);
        Assert.Equal(2000, scaler.GrowthInterval);
        Assert.Equal(2.0, scaler.GrowthFactor);
        Assert.Equal(0.5, scaler.BackoffFactor);
    }

    [Fact]
    public void Constructor_WithCustomValues_UsesProvidedValues()
    {
        // Arrange & Act
        var scaler = new LossScaler<float>(
            initialScale: 4096.0,
            dynamicScaling: false,
            growthInterval: 1000,
            growthFactor: 1.5,
            backoffFactor: 0.25
        );

        // Assert
        Assert.Equal(4096.0, scaler.Scale);
        Assert.False(scaler.DynamicScaling);
        Assert.Equal(1000, scaler.GrowthInterval);
        Assert.Equal(1.5, scaler.GrowthFactor);
        Assert.Equal(0.25, scaler.BackoffFactor);
    }

    [Fact]
    public void ScaleLoss_MultipliesLossByScale()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);
        float loss = 0.5f;

        // Act
        float scaledLoss = scaler.ScaleLoss(loss);

        // Assert
        Assert.Equal(50.0f, scaledLoss);
    }

    [Fact]
    public void UnscaleGradient_DividesGradientByScale()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);
        float gradient = 50.0f;

        // Act
        float unscaledGradient = scaler.UnscaleGradient(gradient);

        // Assert
        Assert.Equal(0.5f, unscaledGradient, precision: 5);
    }

    [Fact]
    public void UnscaleGradients_Vector_CorrectlyUnscalesAllElements()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 10.0);
        var gradients = new Vector<float>(new[] { 10.0f, 20.0f, 30.0f });

        // Act
        scaler.UnscaleGradients(gradients);

        // Assert
        Assert.Equal(1.0f, gradients[0], precision: 5);
        Assert.Equal(2.0f, gradients[1], precision: 5);
        Assert.Equal(3.0f, gradients[2], precision: 5);
    }

    [Fact]
    public void UnscaleGradients_Tensor_CorrectlyUnscalesAllElements()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 10.0);
        var gradients = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(new[] { 10.0f, 20.0f, 30.0f, 40.0f }));

        // Act
        scaler.UnscaleGradients(gradients);

        // Assert
        Assert.Equal(1.0f, gradients.GetFlatIndexValue(0), precision: 5);
        Assert.Equal(2.0f, gradients.GetFlatIndexValue(1), precision: 5);
        Assert.Equal(3.0f, gradients.GetFlatIndexValue(2), precision: 5);
        Assert.Equal(4.0f, gradients.GetFlatIndexValue(3), precision: 5);
    }

    [Fact]
    public void HasOverflow_WithNaN_ReturnsTrue()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        float nan = float.NaN;

        // Act
        bool result = scaler.HasOverflow(nan);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void HasOverflow_WithInfinity_ReturnsTrue()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        float inf = float.PositiveInfinity;

        // Act
        bool result = scaler.HasOverflow(inf);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void HasOverflow_WithNormalValue_ReturnsFalse()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        float normal = 123.456f;

        // Act
        bool result = scaler.HasOverflow(normal);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void DetectOverflow_Vector_WithNaN_ReturnsTrue()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        var gradients = new Vector<float>(new[] { 1.0f, 2.0f, float.NaN, 4.0f });

        // Act
        bool result = scaler.DetectOverflow(gradients);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void DetectOverflow_Vector_WithAllNormal_ReturnsFalse()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        var gradients = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f });

        // Act
        bool result = scaler.DetectOverflow(gradients);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void UnscaleGradientsAndCheck_WithOverflow_ReducesScaleAndReturnsFalse()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 1000.0, dynamicScaling: true, backoffFactor: 0.5);
        var gradients = new Vector<float>(new[] { 10.0f, float.NaN, 30.0f });

        // Act
        bool result = scaler.UnscaleGradientsAndCheck(gradients);

        // Assert
        Assert.False(result);
        Assert.Equal(500.0, scaler.Scale); // Should be reduced by backoff factor
        Assert.Equal(1, scaler.SkippedUpdates);
    }

    [Fact]
    public void UnscaleGradientsAndCheck_WithoutOverflow_ReturnsTrue()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0, dynamicScaling: true);
        var gradients = new Vector<float>(new[] { 100.0f, 200.0f, 300.0f });

        // Act
        bool result = scaler.UnscaleGradientsAndCheck(gradients);

        // Assert
        Assert.True(result);
        Assert.Equal(1.0f, gradients[0], precision: 5);
        Assert.Equal(2.0f, gradients[1], precision: 5);
        Assert.Equal(3.0f, gradients[2], precision: 5);
    }

    [Fact]
    public void DynamicScaling_AfterGrowthInterval_IncreasesScale()
    {
        // Arrange
        var growthInterval = 5;
        var scaler = new LossScaler<float>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: growthInterval,
            growthFactor: 2.0
        );

        // Act - Perform successful updates for growth interval
        for (int i = 0; i < growthInterval; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f, 20.0f }));
        }

        // Assert
        Assert.Equal(200.0, scaler.Scale); // Should be doubled after growth interval
    }

    [Fact]
    public void DynamicScaling_Disabled_DoesNotChangeScale()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0, dynamicScaling: false);

        // Act - Perform multiple successful updates
        for (int i = 0; i < 10; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f, 20.0f }));
        }

        // Assert
        Assert.Equal(100.0, scaler.Scale); // Scale should not change
    }

    [Fact]
    public void OverflowRate_CalculatesCorrectly()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);

        // Act - 2 successful, 1 failed
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f }));

        // Assert
        Assert.Equal(3, scaler.TotalUpdates);
        Assert.Equal(1, scaler.SkippedUpdates);
        Assert.Equal(1.0 / 3.0, scaler.OverflowRate, precision: 5);
    }

    [Fact]
    public void Reset_ClearsStatistics()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));

        // Act
        scaler.Reset();

        // Assert
        Assert.Equal(0, scaler.TotalUpdates);
        Assert.Equal(0, scaler.SkippedUpdates);
        Assert.Equal(0.0, scaler.OverflowRate);
    }

    [Fact]
    public void Reset_WithNewScale_UpdatesScale()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);

        // Act
        scaler.Reset(newInitialScale: 500.0);

        // Assert
        Assert.Equal(500.0, scaler.Scale);
    }

    [Fact]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 1000.0);

        // Act
        string result = scaler.ToString();

        // Assert
        Assert.Contains("LossScaler", result);
        Assert.Contains("Scale=1000", result);
        Assert.Contains("Dynamic=True", result);
        Assert.Contains("Total Updates=0", result);
        Assert.Contains("Skipped=0", result);
    }

    [Fact]
    public void MinScale_PreventScaleFromGoingBelowMinimum()
    {
        // Arrange
        var scaler = new LossScaler<float>(
            initialScale: 10.0,
            dynamicScaling: true,
            backoffFactor: 0.1,
            minScale: 5.0
        );

        // Act - Trigger multiple overflows
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));

        // Assert
        Assert.Equal(5.0, scaler.Scale); // Should not go below min
    }

    [Fact]
    public void MaxScale_PreventScaleFromGoingAboveMaximum()
    {
        // Arrange
        var scaler = new LossScaler<float>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 1,
            growthFactor: 10.0,
            maxScale: 500.0
        );

        // Act - Trigger multiple scale increases
        for (int i = 0; i < 10; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f }));
        }

        // Assert
        Assert.Equal(500.0, scaler.Scale); // Should not go above max
    }
}
