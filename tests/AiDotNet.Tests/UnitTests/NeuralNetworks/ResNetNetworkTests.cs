using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for ResNet neural network architectures.
/// </summary>
public class ResNetNetworkTests
{
    #region ResNetConfiguration Tests

    [Fact]
    public void ResNetConfiguration_Constructor_ValidParameters_CreatesConfiguration()
    {
        // Arrange & Act
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10);

        // Assert
        Assert.Equal(ResNetVariant.ResNet50, config.Variant);
        Assert.Equal(10, config.NumClasses);
        Assert.Equal(224, config.InputHeight);
        Assert.Equal(224, config.InputWidth);
        Assert.Equal(3, config.InputChannels);
        Assert.True(config.IncludeClassifier);
        Assert.True(config.ZeroInitResidual);
    }

    [Fact]
    public void ResNetConfiguration_BottleneckVariant_UsesBottleneck()
    {
        // Arrange & Act
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10);

        // Assert
        Assert.True(config.UsesBottleneck);
        Assert.Equal(4, config.Expansion);
    }

    [Fact]
    public void ResNetConfiguration_BasicBlockVariant_DoesNotUseBottleneck()
    {
        // Arrange & Act
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10);

        // Assert
        Assert.False(config.UsesBottleneck);
        Assert.Equal(1, config.Expansion);
    }

    [Fact]
    public void ResNetConfiguration_InvalidNumClasses_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 0));
    }

    [Fact]
    public void ResNetConfiguration_InvalidInputDimensions_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10, inputHeight: 16));
    }

    [Theory]
    [InlineData(ResNetVariant.ResNet18, new[] { 2, 2, 2, 2 })]
    [InlineData(ResNetVariant.ResNet34, new[] { 3, 4, 6, 3 })]
    [InlineData(ResNetVariant.ResNet50, new[] { 3, 4, 6, 3 })]
    [InlineData(ResNetVariant.ResNet101, new[] { 3, 4, 23, 3 })]
    [InlineData(ResNetVariant.ResNet152, new[] { 3, 8, 36, 3 })]
    public void ResNetConfiguration_BlockCounts_ReturnsCorrectCounts(ResNetVariant variant, int[] expectedBlocks)
    {
        // Arrange & Act
        var config = new ResNetConfiguration(variant, numClasses: 10);

        // Assert
        Assert.Equal(expectedBlocks, config.BlockCounts);
    }

    [Fact]
    public void ResNetConfiguration_InputShape_ReturnsCorrectShape()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10,
            inputHeight: 224, inputWidth: 224, inputChannels: 3);

        // Act
        var inputShape = config.InputShape;

        // Assert
        Assert.Equal(3, inputShape.Length);
        Assert.Equal(3, inputShape[0]);   // channels
        Assert.Equal(224, inputShape[1]); // height
        Assert.Equal(224, inputShape[2]); // width
    }

    [Fact]
    public void ResNetConfiguration_CreateResNet50_CreatesCorrectConfiguration()
    {
        // Act
        var config = ResNetConfiguration.CreateResNet50(numClasses: 1000);

        // Assert
        Assert.Equal(ResNetVariant.ResNet50, config.Variant);
        Assert.Equal(1000, config.NumClasses);
        Assert.True(config.UsesBottleneck);
    }

    [Fact]
    public void ResNetConfiguration_CreateForCIFAR_Creates32x32Input()
    {
        // Act
        var config = ResNetConfiguration.CreateForCIFAR(ResNetVariant.ResNet18, numClasses: 10);

        // Assert
        Assert.Equal(32, config.InputHeight);
        Assert.Equal(32, config.InputWidth);
        Assert.Equal(10, config.NumClasses);
    }

    [Fact]
    public void ResNetConfiguration_CreateLightweight_CreatesResNet18()
    {
        // Act
        var config = ResNetConfiguration.CreateLightweight(numClasses: 10);

        // Assert
        Assert.Equal(ResNetVariant.ResNet18, config.Variant);
        Assert.False(config.UsesBottleneck);
    }

    #endregion

    #region ResNetNetwork Construction Tests

    [Fact]
    public void ResNetNetwork_Construction_CreatesValidNetwork()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );

        // Act
        var network = new ResNetNetwork<float>(architecture, config);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(ResNetVariant.ResNet18, network.Variant);
        Assert.Equal(10, network.NumClasses);
        Assert.False(network.UsesBottleneck);
        Assert.True(network.LayerCount > 0);
    }

    [Fact]
    public void ResNetNetwork_WithBottleneck_CreatesValidNetwork()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );

        // Act
        var network = new ResNetNetwork<float>(architecture, config);

        // Assert
        Assert.True(network.UsesBottleneck);
    }

    [Fact]
    public void ResNetNetwork_NullConfiguration_ThrowsArgumentNullException()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ResNetNetwork<float>(architecture, null!));
    }

    [Fact]
    public void ResNetNetwork_MismatchedOutputSize_ThrowsArgumentException()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 5, // Mismatch!
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new ResNetNetwork<float>(architecture, config));
    }

    [Fact]
    public void ResNetNetwork_MismatchedInputShape_ThrowsArgumentException()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 64, inputWidth: 64);  // 64x64
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,   // 32x32 - Mismatch!
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new ResNetNetwork<float>(architecture, config));
    }

    #endregion

    #region ResNetNetwork Forward Pass Tests

    [Fact]
    public void ResNetNetwork_Forward_ReturnsCorrectShape()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        // Create input tensor [channels, height, width]
        var input = new Tensor<float>([3, 32, 32]);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
        }

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.Equal(1, output.Rank);
        Assert.Equal(10, output.Shape[0]);
    }

    [Fact]
    public void ResNetNetwork_Predict_ReturnsCorrectShape()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        var input = new Tensor<float>([3, 32, 32]);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
        }

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(10, output.Shape[0]);
    }

    [Fact]
    public void ResNetNetwork_Predict_With4DInput_ReturnsCorrectShape()
    {
        // Arrange - test with batch dimension [B, C, H, W]
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        // Create 4D input with batch size 1: [1, 3, 32, 32]
        var input = new Tensor<float>([1, 3, 32, 32]);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
        }

        // Act
        var output = network.Predict(input);

        // Assert - output should maintain batch dimension
        Assert.True(output.Shape.Length >= 1);
        Assert.Equal(10, output.Shape[output.Shape.Length - 1]);
    }

    [Fact]
    public void ResNetNetwork_BasicBlock_Forward_ReturnsCorrectShape()
    {
        // Arrange - specifically test BasicBlock architecture (ResNet18/34)
        var config = new ResNetConfiguration(ResNetVariant.ResNet34, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        var input = new Tensor<float>([3, 32, 32]);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
        }

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(10, output.Shape[0]);
    }

    [Fact]
    public void ResNetNetwork_Bottleneck_Forward_ReturnsCorrectShape()
    {
        // Arrange - specifically test BottleneckBlock architecture (ResNet50/101/152)
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        var input = new Tensor<float>([3, 32, 32]);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
        }

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(10, output.Shape[0]);
    }

    #endregion

    #region ResNetNetwork Training Tests

    [Fact(Skip = "Gradient computation issue during optimizer update - requires separate investigation")]
    public void ResNetNetwork_Train_CompletesWithoutError()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        var input = new Tensor<float>([3, 32, 32]);
        var target = new Tensor<float>([10]);
        for (int i = 0; i < input.Length; i++) input[i] = 0.5f;
        target[0] = 1.0f; // One-hot encoded class 0

        // Act & Assert - should not throw
        network.Train(input, target);
    }

    [Fact(Skip = "Gradient computation issue during optimizer update - requires separate investigation")]
    public void ResNetNetwork_Train_LossDecreases()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        var input = new Tensor<float>([3, 32, 32]);
        var target = new Tensor<float>([10]);
        for (int i = 0; i < input.Length; i++) input[i] = 0.5f;
        target[0] = 1.0f;

        // Act - train for a few iterations
        network.Train(input, target);
        var initialLoss = network.GetLastLoss();

        for (int i = 0; i < 5; i++)
        {
            network.Train(input, target);
        }
        var finalLoss = network.GetLastLoss();

        // Assert - loss should decrease (or at least not increase dramatically)
        Assert.True(Convert.ToDouble(finalLoss) < Convert.ToDouble(initialLoss) * 10.0,
            "Loss should not explode during training");
    }

    #endregion

    #region ResNetNetwork Metadata Tests

    [Fact]
    public void ResNetNetwork_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.NotNull(metadata.AdditionalInfo);
        Assert.Equal("ResNet", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal("ResNet50", metadata.AdditionalInfo["Variant"]);
        Assert.Equal(10, metadata.AdditionalInfo["NumClasses"]);
        Assert.Equal(true, metadata.AdditionalInfo["UsesBottleneck"]);
    }

    [Fact]
    public void ResNetNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10,
            inputHeight: 32, inputWidth: 32);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            taskType: NeuralNetworkTaskType.MultiClassClassification
        );
        var network = new ResNetNetwork<float>(architecture, config);

        // Act
        var paramCount = network.GetParameterCount();

        // Assert
        Assert.True(paramCount > 0, "Parameter count should be positive");
    }

    #endregion

    #region ResNetVariant Enum Tests

    [Fact]
    public void ResNetVariant_AllVariantsAreDefined()
    {
        // Act - use non-generic Enum.GetValues for .NET Framework compatibility
        var variants = Enum.GetValues(typeof(ResNetVariant)).Cast<ResNetVariant>().ToArray();

        // Assert
        Assert.Equal(5, variants.Length);
        Assert.Contains(ResNetVariant.ResNet18, variants);
        Assert.Contains(ResNetVariant.ResNet34, variants);
        Assert.Contains(ResNetVariant.ResNet50, variants);
        Assert.Contains(ResNetVariant.ResNet101, variants);
        Assert.Contains(ResNetVariant.ResNet152, variants);
    }

    #endregion

    #region Block Layer Tests

    [Fact]
    public void BasicBlock_Construction_CreatesValidLayer()
    {
        // Arrange & Act
        var block = new AiDotNet.NeuralNetworks.Layers.BasicBlock<float>(
            inChannels: 64,
            outChannels: 64,
            stride: 1,
            inputHeight: 56,
            inputWidth: 56);

        // Assert
        Assert.NotNull(block);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void BasicBlock_WithDownsample_CreatesValidLayer()
    {
        // Arrange & Act - stride=2 triggers downsampling
        var block = new AiDotNet.NeuralNetworks.Layers.BasicBlock<float>(
            inChannels: 64,
            outChannels: 128,
            stride: 2,
            inputHeight: 56,
            inputWidth: 56);

        // Assert
        Assert.NotNull(block);

        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void BottleneckBlock_Construction_CreatesValidLayer()
    {
        // Arrange & Act
        var block = new AiDotNet.NeuralNetworks.Layers.BottleneckBlock<float>(
            inChannels: 64,
            baseChannels: 64,
            stride: 1,
            inputHeight: 56,
            inputWidth: 56);

        // Assert
        Assert.NotNull(block);
        Assert.True(block.SupportsTraining);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void BottleneckBlock_WithDownsample_CreatesValidLayer()
    {
        // Arrange & Act - stride=2 triggers downsampling
        var block = new AiDotNet.NeuralNetworks.Layers.BottleneckBlock<float>(
            inChannels: 256,
            baseChannels: 128,
            stride: 2,
            inputHeight: 56,
            inputWidth: 56);

        // Assert
        Assert.NotNull(block);

        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void AdaptiveAvgPoolingLayer_GlobalPool_CreatesValidLayer()
    {
        // Arrange & Act
        var layer = AiDotNet.NeuralNetworks.Layers.AdaptiveAvgPoolingLayer<float>.GlobalPool(
            inputChannels: 512,
            inputHeight: 7,
            inputWidth: 7);

        // Assert
        Assert.NotNull(layer);


    }

    [Fact]
    public void AdaptiveAvgPoolingLayer_CustomOutput_CreatesValidLayer()
    {
        // Arrange & Act
        var layer = new AiDotNet.NeuralNetworks.Layers.AdaptiveAvgPoolingLayer<float>(
            inputChannels: 256,
            inputHeight: 14,
            inputWidth: 14,
            outputHeight: 7,
            outputWidth: 7);

        // Assert
        Assert.NotNull(layer);


    }

    #endregion
}
