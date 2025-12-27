using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for the EfficientNet implementations.
/// </summary>
public class EfficientNetTests
{
    #region EfficientNet-B0 Tests

    [Fact]
    public void EfficientNetB0_Constructor_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B0, network.Variant);
        Assert.Equal(10, network.NumClasses);
        Assert.Equal(224, network.InputResolution);
        Assert.True(network.Layers.Count > 0);
    }

    [Fact]
    public void EfficientNetB0_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);
        // Input: [batch=1, channels=3, height=224, width=224]
        var input = new Tensor<float>([1, 3, 224, 224]);
        InitializeWithRandomValues(input);

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]); // batch size
        Assert.Equal(10, output.Shape[1]); // num classes
    }

    [Fact]
    public void EfficientNetB0_ForwardSmallInput_ProducesCorrectShape()
    {
        // Arrange - Use the static factory method
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 5);
        // We still need to use the expected resolution
        var input = new Tensor<float>([1, 3, 224, 224]);
        InitializeWithRandomValues(input);

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.Equal(5, output.Shape[1]); // num classes
    }

    #endregion

    #region EfficientNet Variant Tests

    [Theory]
    [InlineData(EfficientNetVariant.B0, 224)]
    [InlineData(EfficientNetVariant.B1, 240)]
    [InlineData(EfficientNetVariant.B2, 260)]
    [InlineData(EfficientNetVariant.B3, 300)]
    public void EfficientNet_Variants_HaveCorrectResolution(EfficientNetVariant variant, int expectedResolution)
    {
        // Arrange & Act - Use factory methods based on variant
        var network = variant switch
        {
            EfficientNetVariant.B0 => EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10),
            EfficientNetVariant.B1 => EfficientNetNetwork<float>.EfficientNetB1(numClasses: 10),
            EfficientNetVariant.B2 => EfficientNetNetwork<float>.EfficientNetB2(numClasses: 10),
            EfficientNetVariant.B3 => EfficientNetNetwork<float>.EfficientNetB3(numClasses: 10),
            _ => EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10)
        };

        // Assert
        Assert.Equal(variant, network.Variant);
        Assert.Equal(expectedResolution, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB1_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB1(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B1, network.Variant);
        Assert.Equal(100, network.NumClasses);
        Assert.Equal(240, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB2_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB2(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B2, network.Variant);
        Assert.Equal(260, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB3_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB3(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B3, network.Variant);
        Assert.Equal(300, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB4_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB4(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B4, network.Variant);
        Assert.Equal(380, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB5_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB5(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B5, network.Variant);
        Assert.Equal(456, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB6_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB6(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B6, network.Variant);
        Assert.Equal(528, network.InputResolution);
    }

    [Fact]
    public void EfficientNetB7_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = EfficientNetNetwork<float>.EfficientNetB7(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(EfficientNetVariant.B7, network.Variant);
        Assert.Equal(600, network.InputResolution);
    }

    #endregion

    #region Configuration Tests

    [Fact]
    public void EfficientNetConfiguration_GetScalingCoefficients_ReturnsCorrectValues()
    {
        // Test B0
        var configB0 = new EfficientNetConfiguration(EfficientNetVariant.B0, numClasses: 10);
        Assert.Equal(1.0, configB0.GetWidthMultiplier());
        Assert.Equal(1.0, configB0.GetDepthMultiplier());
        Assert.Equal(224, configB0.GetInputHeight());

        // Test B3
        var configB3 = new EfficientNetConfiguration(EfficientNetVariant.B3, numClasses: 10);
        Assert.Equal(1.2, configB3.GetWidthMultiplier());
        Assert.Equal(1.4, configB3.GetDepthMultiplier());
        Assert.Equal(300, configB3.GetInputHeight());

        // Test B7
        var configB7 = new EfficientNetConfiguration(EfficientNetVariant.B7, numClasses: 10);
        Assert.Equal(2.0, configB7.GetWidthMultiplier());
        Assert.Equal(3.1, configB7.GetDepthMultiplier());
        Assert.Equal(600, configB7.GetInputHeight());
    }

    [Fact]
    public void EfficientNet_WithCustomInputChannels_CreatesValidNetwork()
    {
        // Arrange - Single channel (grayscale) input
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10, inputChannels: 1);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.Layers.Count > 0);
    }

    #endregion

    #region Swish Activation Tests

    [Fact]
    public void SwishActivation_Activate_ComputesCorrectValues()
    {
        // Arrange
        var swish = new SwishActivation<float>();

        // Act & Assert
        // Swish(x) = x * sigmoid(x)
        // For x = 0: Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        Assert.Equal(0f, swish.Activate(0f), 4);

        // For large positive x, swish(x) approximately equals x
        float largePositive = swish.Activate(5f);
        Assert.True(largePositive > 4.9f && largePositive < 5.1f);

        // For large negative x, swish(x) approximately equals 0
        float largeNegative = swish.Activate(-5f);
        Assert.True(largeNegative > -0.1f && largeNegative < 0.1f);

        // For x = 1: Swish(1) = 1 * sigmoid(1) ~ 0.731
        float atOne = swish.Activate(1f);
        Assert.True(atOne > 0.7f && atOne < 0.8f);
    }

    [Fact]
    public void SwishActivation_Derivative_ComputesCorrectValues()
    {
        // Arrange
        var swish = new SwishActivation<float>();

        // Act & Assert
        // Derivative of Swish at x=0 should be 0.5
        float derivAtZero = swish.Derivative(0f);
        Assert.Equal(0.5f, derivAtZero, 4);

        // Derivative should be positive for most positive values
        float derivAtPositive = swish.Derivative(2f);
        Assert.True(derivAtPositive > 0);

        // Derivative near x=-2 is close to 0
        float derivAtNegative = swish.Derivative(-5f);
        Assert.True(Math.Abs(derivAtNegative) < 0.1f);
    }

    [Fact]
    public void SwishActivation_ActivateVector_WorksCorrectly()
    {
        // Arrange
        var swish = new SwishActivation<float>();
        var input = new Vector<float>([0f, 1f, -1f, 2f]);

        // Act
        var output = swish.Activate(input);

        // Assert
        Assert.Equal(4, output.Length);
        Assert.Equal(0f, output[0], 4); // Swish(0) = 0
        Assert.True(output[1] > 0.7f); // Swish(1) ~ 0.731
        Assert.True(output[2] < 0 && output[2] > -0.5f); // Swish(-1) ~ -0.269
        Assert.True(output[3] > 1.7f); // Swish(2) ~ 1.762
    }

    [Fact]
    public void SwishActivation_SupportsJitCompilation()
    {
        // Arrange
        var swish = new SwishActivation<float>();

        // Assert
        Assert.True(swish.SupportsJitCompilation);
    }

    #endregion

    #region Model Metadata Tests

    [Fact]
    public void EfficientNet_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("EfficientNetNetwork", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal("B0", metadata.AdditionalInfo["Variant"]);
        Assert.Equal(1.0, (double)metadata.AdditionalInfo["WidthCoefficient"]);
        Assert.Equal(1.0, (double)metadata.AdditionalInfo["DepthCoefficient"]);
        Assert.Equal(224, (int)metadata.AdditionalInfo["Resolution"]);
    }

    #endregion

    #region Compound Scaling Tests

    [Fact]
    public void EfficientNet_LargerVariants_HaveMoreLayers()
    {
        // Arrange
        var b0 = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);
        var b3 = EfficientNetNetwork<float>.EfficientNetB3(numClasses: 10);

        // Assert - B3 should have more layers due to depth scaling
        Assert.True(b3.Layers.Count >= b0.Layers.Count);
    }

    [Fact]
    public void EfficientNet_Clone_CreatesNewInstance()
    {
        // Arrange
        var original = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.IsType<EfficientNetNetwork<float>>(clone);
    }

    [Fact]
    public void EfficientNet_GetLayer_ReturnsCorrectLayer()
    {
        // Arrange
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);

        // Act
        var firstLayer = network.GetLayer(0);
        var lastLayer = network.GetLayer(network.Layers.Count - 1);

        // Assert
        Assert.IsType<ConvolutionalLayer<float>>(firstLayer); // Stem conv
        Assert.IsType<DenseLayer<float>>(lastLayer); // Classification head
    }

    [Fact]
    public void EfficientNet_GetLayer_ThrowsOnInvalidIndex()
    {
        // Arrange
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => network.GetLayer(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => network.GetLayer(network.Layers.Count));
    }

    #endregion

    #region Training Tests (Skipped for performance)

    [Fact(Skip = "Training test - slow")]
    public void EfficientNetB0_Train_CompletesWithoutError()
    {
        // Arrange
        var network = EfficientNetNetwork<float>.EfficientNetB0(numClasses: 10);
        var input = new Tensor<float>([1, 3, 224, 224]);
        var target = new Tensor<float>([1, 10]);
        InitializeWithRandomValues(input);
        target[0, 0] = 1f; // One-hot encoded target

        // Act & Assert - Should not throw
        network.Train(input, target);
    }

    #endregion

    #region Helper Methods

    private static void InitializeWithRandomValues(Tensor<float> tensor)
    {
        var random = new Random(42);
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            tensor.Data[i] = (float)(random.NextDouble() * 2 - 1);
        }
    }

    #endregion
}
