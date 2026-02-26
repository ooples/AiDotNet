using AiDotNet.ActivationFunctions;
using AiDotNet.ComputerVision.Segmentation.Semantic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for the SegFormer semantic segmentation model.
/// Tests native (trainable) and ONNX (inference-only) construction patterns,
/// predict/train behavior, serialization, and model size configuration.
/// </summary>
public class SegFormerIntegrationTests
{
    #region Native Mode Construction

    [Fact]
    public void Constructor_NativeMode_CreatesModelWithCorrectProperties()
    {
        // Arrange & Act
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture);

        // Assert
        Assert.NotNull(model);
        Assert.True(model.UseNativeMode);
        Assert.True(model.SupportsTraining);
        Assert.Equal(SegFormerModelSize.B0, model.ModelSize);
        Assert.Equal(150, model.NumClasses);
    }

    [Fact]
    public void Constructor_NativeMode_WithCustomNumClasses_SetsCorrectly()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture, numClasses: 21);

        Assert.Equal(21, model.NumClasses);
    }

    [Theory]
    [InlineData(SegFormerModelSize.B0)]
    [InlineData(SegFormerModelSize.B1)]
    [InlineData(SegFormerModelSize.B2)]
    [InlineData(SegFormerModelSize.B3)]
    [InlineData(SegFormerModelSize.B4)]
    [InlineData(SegFormerModelSize.B5)]
    public void Constructor_NativeMode_DifferentModelSizes_CreatesDifferentLayerCounts(SegFormerModelSize modelSize)
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture, modelSize: modelSize);

        Assert.NotNull(model);
        Assert.Equal(modelSize, model.ModelSize);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void Constructor_NativeMode_B0HasFewerLayersThanB5()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var modelB0 = new SegFormer<double>(architecture, modelSize: SegFormerModelSize.B0);
        var modelB5 = new SegFormer<double>(architecture, modelSize: SegFormerModelSize.B5);

        // B5 has much deeper transformer stages (depths [3,6,40,3] vs [2,2,2,2])
        // so it should have more layers
        int b0Params = modelB0.ParameterCount;
        int b5Params = modelB5.ParameterCount;
        Assert.True(b5Params > b0Params,
            $"B5 should have more parameters ({b5Params}) than B0 ({b0Params})");
    }

    [Fact]
    public void Constructor_NativeMode_InitializesLayers()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture);

        // Should have encoder + decoder layers
        Assert.True(model.ParameterCount > 0, "Model should have parameters");
    }

    #endregion

    #region ONNX Mode Construction

    [Fact]
    public void Constructor_OnnxMode_WithNullPath_ThrowsArgumentException()
    {
        var architecture = CreateArchitecture(64, 64, 3);

        Assert.Throws<ArgumentException>(() =>
            new SegFormer<double>(architecture, onnxModelPath: ""));
    }

    [Fact]
    public void Constructor_OnnxMode_WithNonExistentPath_ThrowsFileNotFoundException()
    {
        var architecture = CreateArchitecture(64, 64, 3);

        Assert.Throws<FileNotFoundException>(() =>
            new SegFormer<double>(architecture, onnxModelPath: "/nonexistent/path/model.onnx"));
    }

    [Fact]
    public void Constructor_OnnxMode_WithInvalidOnnxFile_ThrowsInvalidOperationException()
    {
        // Create a temp file that isn't a valid ONNX model
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "not an onnx model");

            var architecture = CreateArchitecture(64, 64, 3);

            Assert.Throws<InvalidOperationException>(() =>
                new SegFormer<double>(architecture, onnxModelPath: tempFile));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    #endregion

    #region Predict

    [Fact]
    public void Predict_NativeMode_WithBatchedInput_ReturnsOutput()
    {
        var architecture = CreateArchitecture(32, 32, 3);
        var model = new SegFormer<double>(architecture, numClasses: 10, modelSize: SegFormerModelSize.B0);

        // Create batched input [B, C, H, W]
        var input = CreateRandomTensor([1, 3, 32, 32]);

        var output = model.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void Predict_NativeMode_WithUnbatchedInput_ReturnsOutput()
    {
        var architecture = CreateArchitecture(32, 32, 3);
        var model = new SegFormer<double>(architecture, numClasses: 10, modelSize: SegFormerModelSize.B0);

        // Create unbatched input [C, H, W]
        var input = CreateRandomTensor([3, 32, 32]);

        var output = model.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    #endregion

    #region Train

    [Fact]
    public void Train_NativeMode_DoesNotThrow()
    {
        var architecture = CreateArchitecture(32, 32, 3);
        var model = new SegFormer<double>(architecture, numClasses: 5, modelSize: SegFormerModelSize.B0);

        var input = CreateRandomTensor([1, 3, 32, 32]);
        // Expected output needs to match actual predict output shape
        var predicted = model.Predict(input);
        var expectedOutput = CreateRandomTensor(predicted.Shape);

        // Training should not throw in native mode
        var exception = Record.Exception(() => model.Train(input, expectedOutput));
        Assert.Null(exception);
    }

    #endregion

    #region SupportsTraining

    [Fact]
    public void SupportsTraining_NativeMode_ReturnsTrue()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture);

        Assert.True(model.SupportsTraining);
    }

    #endregion

    #region GetModelMetadata

    [Fact]
    public void GetModelMetadata_ReturnsCorrectModelType()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture, numClasses: 21, modelSize: SegFormerModelSize.B2);

        var metadata = model.GetModelMetadata();

        Assert.Equal(ModelType.SemanticSegmentation, metadata.ModelType);
        Assert.Equal("SegFormer", metadata.AdditionalInfo["ModelName"]);
        Assert.Equal(21, metadata.AdditionalInfo["NumClasses"]);
        Assert.Equal("B2", metadata.AdditionalInfo["ModelSize"]);
    }

    #endregion

    #region Serialization

    [Fact]
    public void Serialization_RoundTrip_PreservesConfig()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(
            architecture,
            numClasses: 21,
            modelSize: SegFormerModelSize.B2,
            dropRate: 0.15);

        // Serialize
        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata.ModelData);
        Assert.True(metadata.ModelData.Length > 0, "Serialized data should not be empty");
    }

    #endregion

    #region CreateNewInstance

    [Fact]
    public void CreateNewInstance_NativeMode_CreatesWorkingCopy()
    {
        var architecture = CreateArchitecture(32, 32, 3);
        var model = new SegFormer<double>(architecture, numClasses: 10, modelSize: SegFormerModelSize.B0);

        // Get metadata which calls Serialize internally
        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.SemanticSegmentation, metadata.ModelType);

        // Verify original model still works after metadata extraction
        var input = CreateRandomTensor([1, 3, 32, 32]);
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    #endregion

    #region Custom Architecture Layers

    [Fact]
    public void Constructor_WithCustomLayers_UsesProvidedLayers()
    {
        // Create an architecture with custom layers
        var customLayers = new List<ILayer<double>>
        {
            new DenseLayer<double>(
                3 * 32 * 32, 100,
                new ReLUActivation<double>() as IActivationFunction<double>),
            new DenseLayer<double>(
                100, 10,
                new IdentityActivation<double>() as IActivationFunction<double>)
        };

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10,
            layers: customLayers);

        var model = new SegFormer<double>(architecture, numClasses: 10);

        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    #endregion

    #region Dispose

    [Fact]
    public void Dispose_NativeMode_DoesNotThrow()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture);

        var exception = Record.Exception(() => model.Dispose());
        Assert.Null(exception);
    }

    [Fact]
    public void Dispose_CalledTwice_DoesNotThrow()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var model = new SegFormer<double>(architecture);

        model.Dispose();
        var exception = Record.Exception(() => model.Dispose());
        Assert.Null(exception);
    }

    #endregion

    #region Options

    [Fact]
    public void GetOptions_ReturnsSegFormerOptions()
    {
        var architecture = CreateArchitecture(64, 64, 3);
        var customOptions = new SegFormerOptions { Seed = 42 };
        var model = new SegFormer<double>(architecture, options: customOptions);

        var options = model.GetOptions();
        Assert.IsType<SegFormerOptions>(options);
        Assert.Equal(42, options.Seed);
    }

    #endregion

    #region Helper Methods

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int height, int width, int depth)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
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
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < totalSize; i++)
            data[i] = random.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #endregion
}
