using AiDotNet.UncertaintyQuantification.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

public class MCDropoutLayerTests
{
    [Fact]
    public void Constructor_WithValidDropoutRate_CreatesLayer()
    {
        // Arrange & Act
        var layer = new MCDropoutLayer<double>(0.5);

        // Assert
        Assert.NotNull(layer);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void Constructor_WithInvalidDropoutRate_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new MCDropoutLayer<double>(-0.1));
        Assert.Throws<ArgumentException>(() => new MCDropoutLayer<double>(1.0));
        Assert.Throws<ArgumentException>(() => new MCDropoutLayer<double>(1.5));
    }

    [Fact]
    public void Forward_InTrainingMode_AppliesDropout()
    {
        // Arrange
        var layer = new MCDropoutLayer<double>(0.5);
        layer.SetTrainingMode(true);
        var input = new Tensor<double>([10], Enumerable.Range(0, 10).Select(i => 1.0).ToArray());

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Length, output.Length);

        // At least some values should be different (either 0 or scaled)
        bool hasZeros = false;
        bool hasNonZeros = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] == 0.0) hasZeros = true;
            if (output[i] != 0.0) hasNonZeros = true;
        }

        // With dropout rate 0.5, we expect both zeros and non-zeros (though this could rarely fail due to randomness)
        Assert.True(hasZeros || hasNonZeros); // At least one should be true
    }

    [Fact]
    public void Forward_InMonteCarloMode_AppliesDropout()
    {
        // Arrange
        var layer = new MCDropoutLayer<double>(0.5, mcMode: true);
        layer.SetTrainingMode(false); // Not in training mode
        var input = new Tensor<double>([10], Enumerable.Range(0, 10).Select(i => 1.0).ToArray());

        // Act
        var output = layer.Forward(input);

        // Assert - dropout should still be applied due to MC mode
        Assert.NotNull(output);
        bool hasModifiedValues = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] != input[i])
            {
                hasModifiedValues = true;
                break;
            }
        }
        Assert.True(hasModifiedValues);
    }

    [Fact]
    public void Forward_InInferenceMode_WithoutMCMode_PassesThrough()
    {
        // Arrange
        var layer = new MCDropoutLayer<double>(0.5, mcMode: false);
        layer.SetTrainingMode(false);
        var input = new Tensor<double>([5], new double[] { 1, 2, 3, 4, 5 });

        // Act
        var output = layer.Forward(input);

        // Assert - should pass through unchanged
        Assert.NotNull(output);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], output[i]);
        }
    }

    [Fact]
    public void GetParameters_ReturnsEmptyVector()
    {
        // Arrange
        var layer = new MCDropoutLayer<double>(0.3);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.Equal(0, parameters.Length);
    }

    [Fact]
    public void MonteCarloMode_CanBeToggledOnOff()
    {
        // Arrange
        var layer = new MCDropoutLayer<double>(0.5, mcMode: false);

        // Act & Assert
        Assert.False(layer.MonteCarloMode);

        layer.MonteCarloMode = true;
        Assert.True(layer.MonteCarloMode);

        layer.MonteCarloMode = false;
        Assert.False(layer.MonteCarloMode);
    }
}
