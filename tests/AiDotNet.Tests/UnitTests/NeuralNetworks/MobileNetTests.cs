using Xunit;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for MobileNetV2 and MobileNetV3 network implementations.
/// </summary>
public class MobileNetTests
{
    #region MobileNetV2 Configuration Tests

    [Fact]
    public void MobileNetV2Configuration_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var config = new MobileNetV2Configuration<double>();

        // Assert
        Assert.Equal(MobileNetV2WidthMultiplier.Alpha100, config.WidthMultiplier);
        Assert.Equal(3, config.InputChannels);
        Assert.Equal(224, config.InputHeight);
        Assert.Equal(224, config.InputWidth);
        Assert.Equal(1000, config.NumClasses);
        Assert.Equal(1.0, config.Alpha);
    }

    [Theory]
    [InlineData(MobileNetV2WidthMultiplier.Alpha035, 0.35)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha050, 0.5)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha075, 0.75)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha100, 1.0)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha140, 1.4)]
    public void MobileNetV2Configuration_WidthMultiplier_ReturnsCorrectAlpha(
        MobileNetV2WidthMultiplier multiplier, double expectedAlpha)
    {
        // Arrange
        var config = new MobileNetV2Configuration<double> { WidthMultiplier = multiplier };

        // Assert
        Assert.Equal(expectedAlpha, config.Alpha);
    }

    #endregion

    #region MobileNetV2 Construction Tests

    [Fact]
    public void MobileNetV2_100_Construction_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = MobileNetV2Network<double>.MobileNetV2_100(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(MobileNetV2WidthMultiplier.Alpha100, network.WidthMultiplier);
        Assert.Equal(10, network.NumClasses);
        Assert.True(network.LayerCount > 0);
    }

    [Fact]
    public void MobileNetV2_035_Construction_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = MobileNetV2Network<double>.MobileNetV2_035(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(MobileNetV2WidthMultiplier.Alpha035, network.WidthMultiplier);
        Assert.Equal(10, network.NumClasses);
    }

    [Fact]
    public void MobileNetV2_CustomConfig_CreatesValidNetwork()
    {
        // Arrange
        var config = new MobileNetV2Configuration<double>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha075,
            InputChannels = 1,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };

        // Act
        var network = new MobileNetV2Network<double>(config);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.LayerCount > 0);
    }

    [Theory]
    [InlineData(MobileNetV2WidthMultiplier.Alpha035)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha050)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha075)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha100)]
    [InlineData(MobileNetV2WidthMultiplier.Alpha140)]
    public void MobileNetV2_AllWidthMultipliers_AreValid(MobileNetV2WidthMultiplier multiplier)
    {
        // Arrange
        var config = new MobileNetV2Configuration<double>
        {
            WidthMultiplier = multiplier,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };

        // Act & Assert - should not throw
        var network = new MobileNetV2Network<double>(config);
        Assert.NotNull(network);
        Assert.Equal(multiplier, network.WidthMultiplier);
    }

    #endregion

    #region MobileNetV2 Forward Pass Tests

    [Fact]
    public void MobileNetV2_Forward_ProducesCorrectOutputShape()
    {
        // Arrange - use smaller input for faster tests
        var config = new MobileNetV2Configuration<double>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha100,
            InputChannels = 3,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };
        var network = new MobileNetV2Network<double>(config);

        var input = new Tensor<double>([1, 3, 32, 32]);
        var random = new Random(123);
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] = random.NextDouble() * 0.5 + 0.1;

        // Act
        var output = network.Predict(input);

        // Assert - output should have values for 10 classes
        Assert.True(output.Data.Length >= 10, "Output should have at least 10 values for 10 classes");
    }

    [Fact]
    public void MobileNetV2_Forward_ReturnsNonZeroOutput()
    {
        // Arrange
        var config = new MobileNetV2Configuration<double>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha100,
            InputChannels = 3,
            InputHeight = 64,
            InputWidth = 64,
            NumClasses = 10
        };
        var network = new MobileNetV2Network<double>(config);

        var input = new Tensor<double>([1, 3, 64, 64]);
        var random = new Random(42);
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] = random.NextDouble();

        // Act
        var output = network.Predict(input);

        // Assert
        bool hasNonZero = output.Data.Any(v => Math.Abs(v) > 1e-10);
        Assert.True(hasNonZero, "Output should have at least some non-zero values");
    }

    #endregion

    #region MobileNetV3 Configuration Tests

    [Fact]
    public void MobileNetV3Configuration_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var config = new MobileNetV3Configuration<double>();

        // Assert
        Assert.Equal(MobileNetV3Variant.Large, config.Variant);
        Assert.Equal(MobileNetV3WidthMultiplier.Alpha100, config.WidthMultiplier);
        Assert.Equal(3, config.InputChannels);
        Assert.Equal(224, config.InputHeight);
        Assert.Equal(224, config.InputWidth);
        Assert.Equal(1000, config.NumClasses);
    }

    [Theory]
    [InlineData(MobileNetV3WidthMultiplier.Alpha075, 0.75)]
    [InlineData(MobileNetV3WidthMultiplier.Alpha100, 1.0)]
    public void MobileNetV3Configuration_WidthMultiplier_ReturnsCorrectAlpha(
        MobileNetV3WidthMultiplier multiplier, double expectedAlpha)
    {
        // Arrange
        var config = new MobileNetV3Configuration<double> { WidthMultiplier = multiplier };

        // Assert
        Assert.Equal(expectedAlpha, config.Alpha);
    }

    #endregion

    #region MobileNetV3 Construction Tests

    [Fact]
    public void MobileNetV3Large_Construction_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = MobileNetV3Network<double>.MobileNetV3Large(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(MobileNetV3Variant.Large, network.Variant);
        Assert.Equal(10, network.NumClasses);
        Assert.True(network.LayerCount > 0);
    }

    [Fact]
    public void MobileNetV3Small_Construction_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = MobileNetV3Network<double>.MobileNetV3Small(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(MobileNetV3Variant.Small, network.Variant);
        Assert.Equal(10, network.NumClasses);
        Assert.True(network.LayerCount > 0);
    }

    [Fact]
    public void MobileNetV3_CustomConfig_CreatesValidNetwork()
    {
        // Arrange
        var config = new MobileNetV3Configuration<double>
        {
            Variant = MobileNetV3Variant.Large,
            WidthMultiplier = MobileNetV3WidthMultiplier.Alpha075,
            InputChannels = 1,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };

        // Act
        var network = new MobileNetV3Network<double>(config);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.LayerCount > 0);
    }

    [Theory]
    [InlineData(MobileNetV3Variant.Large)]
    [InlineData(MobileNetV3Variant.Small)]
    public void MobileNetV3_AllVariants_AreValid(MobileNetV3Variant variant)
    {
        // Arrange
        var config = new MobileNetV3Configuration<double>
        {
            Variant = variant,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };

        // Act & Assert - should not throw
        var network = new MobileNetV3Network<double>(config);
        Assert.NotNull(network);
        Assert.Equal(variant, network.Variant);
    }

    #endregion

    #region MobileNetV3 Forward Pass Tests

    [Fact]
    public void MobileNetV3Large_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        var config = new MobileNetV3Configuration<double>
        {
            Variant = MobileNetV3Variant.Large,
            InputChannels = 3,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };
        var network = new MobileNetV3Network<double>(config);

        var input = new Tensor<double>([1, 3, 32, 32]);
        var random = new Random(123);
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] = random.NextDouble() * 0.5 + 0.1;

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.True(output.Data.Length >= 10, "Output should have at least 10 values for 10 classes");
    }

    [Fact]
    public void MobileNetV3Small_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        var config = new MobileNetV3Configuration<double>
        {
            Variant = MobileNetV3Variant.Small,
            InputChannels = 3,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };
        var network = new MobileNetV3Network<double>(config);

        var input = new Tensor<double>([1, 3, 32, 32]);
        var random = new Random(123);
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] = random.NextDouble() * 0.5 + 0.1;

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.True(output.Data.Length >= 10, "Output should have at least 10 values for 10 classes");
    }

    #endregion

    #region InvertedResidualBlock Tests

    [Fact]
    public void InvertedResidualBlock_Construction_CreatesValidBlock()
    {
        // Arrange & Act
        var block = new InvertedResidualBlock<double>(
            inChannels: 32,
            outChannels: 64,
            inputHeight: 16,
            inputWidth: 16,
            expansionRatio: 6,
            stride: 2,
            useSE: false);

        // Assert
        Assert.NotNull(block);
        Assert.Equal(32, block.InChannels);
        Assert.Equal(64, block.OutChannels);
        Assert.Equal(6, block.ExpansionRatio);
        Assert.Equal(2, block.Stride);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void InvertedResidualBlock_WithSE_CreatesValidBlock()
    {
        // Arrange & Act
        var block = new InvertedResidualBlock<double>(
            inChannels: 64,
            outChannels: 96,
            inputHeight: 8,
            inputWidth: 8,
            expansionRatio: 6,
            stride: 1,
            useSE: true,
            seRatio: 4);

        // Assert
        Assert.NotNull(block);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void InvertedResidualBlock_Forward_ProducesOutput()
    {
        // Arrange
        var block = new InvertedResidualBlock<double>(
            inChannels: 32,
            outChannels: 32,
            inputHeight: 8,
            inputWidth: 8,
            expansionRatio: 6,
            stride: 1,
            useSE: false);

        var input = new Tensor<double>([1, 32, 8, 8]);
        var random = new Random(42);
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] = random.NextDouble();

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(1, output.Shape[0]); // Batch
        Assert.Equal(32, output.Shape[1]); // Channels
        Assert.Equal(8, output.Shape[2]); // Height (stride=1)
        Assert.Equal(8, output.Shape[3]); // Width
    }

    [Fact]
    public void InvertedResidualBlock_WithStride2_ReducesSpatialDimensions()
    {
        // Arrange
        var block = new InvertedResidualBlock<double>(
            inChannels: 32,
            outChannels: 64,
            inputHeight: 16,
            inputWidth: 16,
            expansionRatio: 6,
            stride: 2,
            useSE: false);

        var input = new Tensor<double>([1, 32, 16, 16]);
        var random = new Random(42);
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] = random.NextDouble();

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(1, output.Shape[0]); // Batch
        Assert.Equal(64, output.Shape[1]); // New channels
        Assert.Equal(8, output.Shape[2]); // Height reduced by stride
        Assert.Equal(8, output.Shape[3]); // Width reduced by stride
    }

    #endregion

    #region ReLU6 Activation Tests

    [Fact]
    public void ReLU6Activation_Activate_ClampsToSix()
    {
        // Arrange
        var activation = new ReLU6Activation<double>();

        // Act & Assert
        Assert.Equal(0.0, activation.Activate(-1.0), 6);
        Assert.Equal(0.0, activation.Activate(0.0), 6);
        Assert.Equal(3.0, activation.Activate(3.0), 6);
        Assert.Equal(6.0, activation.Activate(6.0), 6);
        Assert.Equal(6.0, activation.Activate(10.0), 6);
    }

    [Fact]
    public void ReLU6Activation_Derivative_ReturnsCorrectValues()
    {
        // Arrange
        var activation = new ReLU6Activation<double>();

        // Act & Assert
        Assert.Equal(0.0, activation.Derivative(-1.0), 6); // Out of range
        Assert.Equal(0.0, activation.Derivative(0.0), 6);  // At boundary
        Assert.Equal(1.0, activation.Derivative(3.0), 6);  // In range
        Assert.Equal(0.0, activation.Derivative(6.0), 6);  // At boundary
        Assert.Equal(0.0, activation.Derivative(10.0), 6); // Out of range
    }

    [Fact]
    public void ReLU6Activation_SupportsJitCompilation()
    {
        // Arrange
        var activation = new ReLU6Activation<double>();

        // Assert
        Assert.False(activation.SupportsJitCompilation); // TensorOperations.Minimum not yet implemented
    }

    #endregion

    #region HardSwish Activation Tests

    [Fact]
    public void HardSwishActivation_Activate_ComputesCorrectly()
    {
        // Arrange
        var activation = new HardSwishActivation<double>();

        // Act & Assert
        // x <= -3: output is 0
        Assert.Equal(0.0, activation.Activate(-4.0), 6);
        Assert.Equal(0.0, activation.Activate(-3.0), 6);

        // x >= 3: output equals x
        Assert.Equal(3.0, activation.Activate(3.0), 6);
        Assert.Equal(5.0, activation.Activate(5.0), 6);

        // In between: x * (x + 3) / 6
        // x = 0: 0 * (0 + 3) / 6 = 0
        Assert.Equal(0.0, activation.Activate(0.0), 6);

        // x = 1: 1 * (1 + 3) / 6 = 4/6 = 0.666...
        Assert.Equal(0.6666, activation.Activate(1.0), 3);
    }

    [Fact]
    public void HardSwishActivation_Derivative_ReturnsCorrectValues()
    {
        // Arrange
        var activation = new HardSwishActivation<double>();

        // Act & Assert
        // x <= -3: derivative = 0
        Assert.Equal(0.0, activation.Derivative(-4.0), 6);

        // x >= 3: derivative = 1
        Assert.Equal(1.0, activation.Derivative(4.0), 6);

        // -3 < x < 3: derivative = (2x + 3) / 6
        // x = 0: (0 + 3) / 6 = 0.5
        Assert.Equal(0.5, activation.Derivative(0.0), 6);

        // x = 1: (2 + 3) / 6 = 5/6 ≈ 0.833
        Assert.Equal(0.8333, activation.Derivative(1.0), 3);
    }

    [Fact]
    public void HardSwishActivation_SupportsJitCompilation()
    {
        // Arrange
        var activation = new HardSwishActivation<double>();

        // Assert
        Assert.False(activation.SupportsJitCompilation); // TensorOperations.Minimum not yet implemented
    }

    #endregion

    #region Metadata Tests

    [Fact]
    public void MobileNetV2_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var network = MobileNetV2Network<double>.MobileNetV2_100(numClasses: 10);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("MobileNetV2Network", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal(1.0, (double)metadata.AdditionalInfo["WidthMultiplier"]);
        Assert.Equal(10, metadata.AdditionalInfo["NumClasses"]);
        Assert.True((int)metadata.AdditionalInfo["LayerCount"] > 0);
    }

    [Fact]
    public void MobileNetV3_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var network = MobileNetV3Network<double>.MobileNetV3Large(numClasses: 10);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("MobileNetV3Network", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal("Large", metadata.AdditionalInfo["Variant"]);
        Assert.Equal(10, metadata.AdditionalInfo["NumClasses"]);
        Assert.True((int)metadata.AdditionalInfo["LayerCount"] > 0);
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void MobileNetV2_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = MobileNetV2Network<double>.MobileNetV2_100(numClasses: 10);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<MobileNetV2Network<double>>(clone);

        var clonedNetwork = (MobileNetV2Network<double>)clone;
        Assert.Equal(original.WidthMultiplier, clonedNetwork.WidthMultiplier);
        Assert.Equal(original.NumClasses, clonedNetwork.NumClasses);
    }

    [Fact]
    public void MobileNetV3_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = MobileNetV3Network<double>.MobileNetV3Small(numClasses: 10);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<MobileNetV3Network<double>>(clone);

        var clonedNetwork = (MobileNetV3Network<double>)clone;
        Assert.Equal(original.Variant, clonedNetwork.Variant);
        Assert.Equal(original.NumClasses, clonedNetwork.NumClasses);
    }

    #endregion

    #region Enum Tests

    [Fact]
    public void MobileNetV2WidthMultiplier_EnumValues_AreDistinct()
    {
        // Arrange
        var values = Enum.GetValues(typeof(MobileNetV2WidthMultiplier)).Cast<MobileNetV2WidthMultiplier>().ToArray();

        // Assert
        Assert.Equal(5, values.Length);
        Assert.Contains(MobileNetV2WidthMultiplier.Alpha035, values);
        Assert.Contains(MobileNetV2WidthMultiplier.Alpha050, values);
        Assert.Contains(MobileNetV2WidthMultiplier.Alpha075, values);
        Assert.Contains(MobileNetV2WidthMultiplier.Alpha100, values);
        Assert.Contains(MobileNetV2WidthMultiplier.Alpha140, values);
    }

    [Fact]
    public void MobileNetV3Variant_EnumValues_AreDistinct()
    {
        // Arrange
        var values = Enum.GetValues(typeof(MobileNetV3Variant)).Cast<MobileNetV3Variant>().ToArray();

        // Assert
        Assert.Equal(2, values.Length);
        Assert.Contains(MobileNetV3Variant.Large, values);
        Assert.Contains(MobileNetV3Variant.Small, values);
    }

    #endregion

    #region Training Tests (Skipped for CI)

    [Fact(Skip = "Training tests require significant compute resources")]
    public void MobileNetV2_Train_CompletesWithoutError()
    {
        // Arrange
        var config = new MobileNetV2Configuration<double>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha100,
            InputChannels = 3,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };
        var network = new MobileNetV2Network<double>(config);

        var input = new Tensor<double>([1, 3, 32, 32]);
        var target = new Tensor<double>([10]);
        for (int i = 0; i < input.Data.Length; i++) input.Data[i] = 0.5;
        target.Data[0] = 1.0;

        // Act & Assert - should not throw
        network.Train(input, target);
    }

    [Fact(Skip = "Training tests require significant compute resources")]
    public void MobileNetV3_Train_CompletesWithoutError()
    {
        // Arrange
        var config = new MobileNetV3Configuration<double>
        {
            Variant = MobileNetV3Variant.Small,
            InputChannels = 3,
            InputHeight = 32,
            InputWidth = 32,
            NumClasses = 10
        };
        var network = new MobileNetV3Network<double>(config);

        var input = new Tensor<double>([1, 3, 32, 32]);
        var target = new Tensor<double>([10]);
        for (int i = 0; i < input.Data.Length; i++) input.Data[i] = 0.5;
        target.Data[0] = 1.0;

        // Act & Assert - should not throw
        network.Train(input, target);
    }

    #endregion
}
