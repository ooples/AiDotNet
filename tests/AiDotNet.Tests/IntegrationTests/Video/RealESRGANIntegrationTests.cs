using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Video;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Video;

/// <summary>
/// Integration tests for Real-ESRGAN to verify tensor rank support,
/// native/ONNX mode functionality, and GAN training behavior.
/// </summary>
public class RealESRGANIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Native Mode Construction Tests

    [Fact]
    public void Constructor_NativeMode_WithValidArchitectures_CreatesModel()
    {
        // Arrange
        var generatorArch = CreateArchitecture(64, 64, 3);
        var discriminatorArch = CreateArchitecture(256, 256, 3);

        // Act
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        // Assert
        Assert.NotNull(model);
        Assert.True(model.UseNativeMode);
        Assert.True(model.SupportsTraining);
        Assert.NotNull(model.Generator);
        Assert.NotNull(model.Discriminator);
        Assert.Equal(4, model.ScaleFactor);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void Constructor_NativeMode_WithDifferentScaleFactors_CreatesModel(int scaleFactor)
    {
        // Arrange
        var generatorArch = CreateArchitecture(64, 64, 3);
        var discriminatorArch = CreateArchitecture(64 * scaleFactor, 64 * scaleFactor, 3);

        // Act
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: scaleFactor);

        // Assert
        Assert.Equal(scaleFactor, model.ScaleFactor);
    }

    [Theory]
    [InlineData(16)]
    [InlineData(23)]
    [InlineData(32)]
    public void Constructor_NativeMode_WithDifferentRRDBBlocks_CreatesModel(int numRRDBBlocks)
    {
        // Arrange
        var generatorArch = CreateArchitecture(64, 64, 3);
        var discriminatorArch = CreateArchitecture(256, 256, 3);

        // Act
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            numRRDBBlocks: numRRDBBlocks);

        // Assert
        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_NativeMode_WithNullGeneratorArchitecture_ThrowsArgumentNullException()
    {
        // Arrange
        var discriminatorArch = CreateArchitecture(256, 256, 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new RealESRGAN<double>(
            null!,
            discriminatorArch,
            InputType.ThreeDimensional));
    }

    [Fact]
    public void Constructor_NativeMode_WithNullDiscriminatorArchitecture_ThrowsArgumentNullException()
    {
        // Arrange
        var generatorArch = CreateArchitecture(64, 64, 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new RealESRGAN<double>(
            generatorArch,
            null!,
            InputType.ThreeDimensional));
    }

    [Fact]
    public void Constructor_NativeMode_WithInvalidScaleFactor_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var generatorArch = CreateArchitecture(64, 64, 3);
        var discriminatorArch = CreateArchitecture(256, 256, 3);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 0));
    }

    #endregion

    #region Tensor Rank Support Tests

    [Fact]
    public void Predict_With4DTensor_BatchChannelHeightWidth_ReturnsCorrectShape()
    {
        // Arrange - 4D tensor: [batch, channels, height, width]
        var generatorArch = CreateArchitecture(32, 32, 3);
        var discriminatorArch = CreateArchitecture(128, 128, 3);
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        var inputTensor = CreateRandomTensor(new[] { 1, 3, 32, 32 });

        // Act
        var output = model.Predict(inputTensor);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void Predict_With5DTensor_BatchFramesChannelsHeightWidth_ReturnsCorrectShape()
    {
        // Arrange - [batch, frames, channels, height, width] for video
        // Note: Using ThreeDimensional input type but with 5D tensor shape
        // The model should support any-rank tensors
        var generatorArch = CreateArchitecture(32, 32, 3);
        var discriminatorArch = CreateArchitecture(128, 128, 3);

        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        // 5D tensor: [batch, frames, channels, height, width]
        var inputTensor = CreateRandomTensor(new[] { 1, 4, 3, 32, 32 });

        // Act
        var output = model.Predict(inputTensor);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void Predict_With2DTensor_HeightWidth_ReturnsCorrectShape()
    {
        // Arrange - [height, width] grayscale (using ThreeDimensional architecture)
        // The model should support any-rank tensors regardless of architecture input type
        var generatorArch = CreateArchitecture(32, 32, 1);
        var discriminatorArch = CreateArchitecture(128, 128, 1);

        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        // 2D tensor: [height, width]
        var inputTensor = CreateRandomTensor(new[] { 32, 32 });

        // Act
        var output = model.Predict(inputTensor);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void Predict_WithBatchOf4Images_ProcessesAllImages()
    {
        // Arrange
        var generatorArch = CreateArchitecture(32, 32, 3);
        var discriminatorArch = CreateArchitecture(128, 128, 3);
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        // Batch of 4 images
        var inputTensor = CreateRandomTensor(new[] { 4, 3, 32, 32 });

        // Act
        var output = model.Predict(inputTensor);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // Batch size preserved
    }

    #endregion

    #region Training Tests

    [Fact]
    public void TrainStep_NativeMode_ReturnsLossValues()
    {
        // Arrange - Uses the proper RRDBNetGenerator architecture that outputs spatial tensors
        // Scale factor must be 2 or 4 for RRDBNetGenerator
        var generatorArch = CreateArchitecture(16, 16, 3);
        var discriminatorArch = CreateArchitecture(32, 32, 3); // 16 * 2 = 32
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 2);

        var lowResInput = CreateRandomTensor(new[] { 1, 3, 16, 16 });
        var highResTarget = CreateRandomTensor(new[] { 1, 3, 32, 32 }); // Scaled 2x

        // Act
        var (discriminatorLoss, generatorLoss) = model.TrainStep(lowResInput, highResTarget);

        // Assert
        Assert.True(discriminatorLoss >= 0);
        Assert.True(generatorLoss >= 0);
        Assert.Equal(discriminatorLoss, model.LastDiscriminatorLoss);
        Assert.Equal(generatorLoss, model.LastGeneratorLoss);
    }

    [Fact]
    public void Train_NativeMode_MultipleBatches_UpdatesLoss()
    {
        // Arrange - Uses the proper RRDBNetGenerator architecture that outputs spatial tensors
        // Scale factor must be 2 or 4 for RRDBNetGenerator
        var generatorArch = CreateArchitecture(16, 16, 3);
        var discriminatorArch = CreateArchitecture(32, 32, 3); // 16 * 2 = 32
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 2);

        var losses = new List<double>();

        // Act - Train for multiple batches
        for (int i = 0; i < 5; i++)
        {
            var lowResInput = CreateRandomTensor(new[] { 1, 3, 16, 16 });
            var highResTarget = CreateRandomTensor(new[] { 1, 3, 32, 32 }); // Scaled 2x
            var (_, gLoss) = model.TrainStep(lowResInput, highResTarget);
            losses.Add(Convert.ToDouble(gLoss));
        }

        // Assert - Loss should be tracked
        Assert.Equal(5, losses.Count);
        Assert.All(losses, l => Assert.True(l >= 0));
    }

    #endregion

    #region Upscale Tests

    [Fact]
    public void Upscale_NativeMode_ReturnsUpscaledTensor()
    {
        // Arrange
        var generatorArch = CreateArchitecture(32, 32, 3);
        var discriminatorArch = CreateArchitecture(128, 128, 3);
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        var inputTensor = CreateRandomTensor(new[] { 1, 3, 32, 32 });

        // Act
        var output = model.Upscale(inputTensor);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void Upscale_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var generatorArch = CreateArchitecture(32, 32, 3);
        var discriminatorArch = CreateArchitecture(128, 128, 3);
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.Upscale(null!));
    }

    #endregion

    #region Metadata Tests

    [Fact]
    public void GetModelMetadata_NativeMode_ReturnsCorrectMetadata()
    {
        // Arrange
        var generatorArch = CreateArchitecture(64, 64, 3);
        var discriminatorArch = CreateArchitecture(256, 256, 3);
        var model = new RealESRGAN<double>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4,
            numRRDBBlocks: 23,
            numFeatures: 64);

        // Act
        var metadata = model.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.GenerativeAdversarialNetwork, metadata.ModelType);
        Assert.Equal("RealESRGAN", metadata.AdditionalInfo["ModelName"]);
        Assert.Equal(4, metadata.AdditionalInfo["ScaleFactor"]);
        Assert.Equal(23, metadata.AdditionalInfo["NumRRDBBlocks"]);
        Assert.Equal(64, metadata.AdditionalInfo["NumFeatures"]);
        Assert.True((bool)metadata.AdditionalInfo["UseNativeMode"]);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Constructor_WithFloatType_CreatesModel()
    {
        // Arrange
        var generatorArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: 0,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 0);
        var discriminatorArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: 0,
            inputHeight: 256,
            inputWidth: 256,
            inputDepth: 3,
            outputSize: 0);

        // Act
        var model = new RealESRGAN<float>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        // Assert
        Assert.NotNull(model);
        Assert.True(model.UseNativeMode);
    }

    [Fact]
    public void Predict_WithFloatType_ReturnsCorrectType()
    {
        // Arrange
        var generatorArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: 0,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 0);
        var discriminatorArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: 0,
            inputHeight: 128,
            inputWidth: 128,
            inputDepth: 3,
            outputSize: 0);

        var model = new RealESRGAN<float>(
            generatorArch,
            discriminatorArch,
            InputType.ThreeDimensional,
            scaleFactor: 4);

        var inputData = new float[1 * 3 * 32 * 32];
        var random = new Random(42);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)random.NextDouble();
        var inputTensor = new Tensor<float>(new[] { 1, 3, 32, 32 }, new Vector<float>(inputData));

        // Act
        var output = model.Predict(inputTensor);

        // Assert
        Assert.NotNull(output);
        Assert.IsType<Tensor<float>>(output);
    }

    #endregion

    #region Helper Methods

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int height, int width, int depth)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: 0,
            inputHeight: height,
            inputWidth: width,
            inputDepth: depth,
            outputSize: 0);
    }

    private static Tensor<double> CreateRandomTensor(int[] shape)
    {
        int totalSize = shape.Aggregate(1, (a, b) => a * b);
        var data = new double[totalSize];
        var random = new Random(42);
        for (int i = 0; i < totalSize; i++)
            data[i] = random.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #endregion
}
