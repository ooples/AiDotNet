using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Integration tests for <see cref="VisionMambaModel{T}"/>.
/// Tests full forward-backward-parameter round-trips with various configurations.
/// </summary>
public class VisionMambaModelTests
{
    private static NeuralNetworkArchitecture<float> CreateArch(
        int height = 32, int width = 32, int channels = 3, int numClasses = 10)
    {
        return new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.ImageClassification,
            inputHeight: height,
            inputWidth: width,
            inputDepth: channels,
            outputSize: numClasses);
    }

    private static NeuralNetworkArchitecture<double> CreateDoubleArch(
        int height = 16, int width = 16, int channels = 1, int numClasses = 3)
    {
        return new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.ImageClassification,
            inputHeight: height,
            inputWidth: width,
            inputDepth: channels,
            outputSize: numClasses);
    }

    [Fact]
    public void Constructor_ValidParameters_CreatesModel()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(),
            imageHeight: 32, imageWidth: 32, patchSize: 8,
            channels: 3, modelDimension: 32, numLayers: 2, numClasses: 10);

        Assert.Equal(32, model.ImageHeight);
        Assert.Equal(32, model.ImageWidth);
        Assert.Equal(8, model.PatchSize);
        Assert.Equal(32, model.ModelDimension);
        Assert.Equal(2, model.NumLayers);
        Assert.Equal(10, model.NumClasses);
        Assert.Equal(16, model.NumPatches); // (32/8) * (32/8) = 4*4 = 16
        Assert.Equal(VisionScanPattern.Bidirectional, model.ScanPattern);
    }

    [Fact]
    public void Constructor_ThrowsWhenImageNotDivisibleByPatch()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(CreateArch(30, 32), imageHeight: 30, imageWidth: 32, patchSize: 8));
    }

    [Fact]
    public void Constructor_ThrowsWhenImageHeightNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(CreateArch(1, 32), imageHeight: 0, imageWidth: 32, patchSize: 8));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumClassesNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(CreateArch(16, 16, numClasses: 1),
                imageHeight: 16, imageWidth: 16, patchSize: 4, numClasses: 0));
    }

    [Theory]
    [InlineData(VisionScanPattern.Bidirectional)]
    [InlineData(VisionScanPattern.CrossScan)]
    [InlineData(VisionScanPattern.Continuous)]
    public void Predict_4D_AllScanPatterns_ProduceValidOutput(VisionScanPattern pattern)
    {
        int batchSize = 2;
        int height = 16;
        int width = 16;
        int channels = 3;
        int patchSize = 4;
        int numClasses = 5;

        var model = new VisionMambaModel<float>(
            CreateArch(height, width, channels, numClasses),
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4, scanPattern: pattern);

        var input = CreateRandomTensor(new[] { batchSize, channels, height, width });
        var output = model.Predict(input);

        Assert.Equal(new[] { batchSize, numClasses }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Predict_3D_ProducesValidOutput()
    {
        int height = 16;
        int width = 16;
        int channels = 3;
        int patchSize = 4;
        int numClasses = 5;

        var model = new VisionMambaModel<float>(
            CreateArch(height, width, channels, numClasses),
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4);

        var input = CreateRandomTensor(new[] { channels, height, width });
        var output = model.Predict(input);

        Assert.Equal(new[] { numClasses }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backpropagate_ProducesValidGradients()
    {
        int height = 16;
        int width = 16;
        int channels = 1;
        int patchSize = 4;
        int numClasses = 3;

        var model = new VisionMambaModel<float>(
            CreateArch(height, width, channels, numClasses),
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4, scanPattern: VisionScanPattern.Continuous);

        model.SetTrainingMode(true);
        var input = CreateRandomTensor(new[] { 1, channels, height, width });
        var output = model.Predict(input);
        model.SetTrainingMode(true); // Re-enable after Predict set it to false
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = model.Backpropagate(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backpropagate_ThrowsWithoutTrainingMode()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(16, 16, 1, 3),
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);
        var grad = CreateRandomTensor(new[] { 1, 3 });

        Assert.Throws<InvalidOperationException>(() => model.Backpropagate(grad));
    }

    [Fact]
    public void Train_ForwardBackwardUpdate_NoErrors()
    {
        int height = 16;
        int width = 16;
        int channels = 1;
        int patchSize = 4;
        int numClasses = 3;

        var model = new VisionMambaModel<float>(
            CreateArch(height, width, channels, numClasses),
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4);

        var input = CreateRandomTensor(new[] { 1, channels, height, width });
        var expected = new Tensor<float>(new[] { 1, numClasses });
        expected[new[] { 0, 0 }] = 1.0f; // one-hot target

        model.Train(input, expected);

        model.ResetState();
        var output2 = model.Predict(input);
        Assert.Equal(new[] { 1, numClasses }, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(16, 16, 1, 3),
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        var params1 = model.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(model.ParameterCount, params1.Length);

        model.SetParameters(params1);
        var params2 = model.GetParameters();

        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SetParameters_ThrowsOnWrongLength()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(16, 16, 1, 3),
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(16, 16, 1, 3),
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void GetModelMetadata_ContainsExpectedKeys()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(32, 32, 3, 10),
            32, 32, 8, 3, modelDimension: 64, numLayers: 4, numClasses: 10,
            scanPattern: VisionScanPattern.CrossScan);

        var metadata = model.GetModelMetadata();

        Assert.Equal(ModelType.NeuralNetwork, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.ContainsKey("ImageHeight"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("ImageWidth"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("PatchSize"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("NumClasses"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("ScanPattern"));
        Assert.Equal(32, metadata.AdditionalInfo["ImageHeight"]);
        Assert.Equal("CrossScan", metadata.AdditionalInfo["ScanPattern"]);
    }

    [Fact]
    public void ResetState_AllowsReuse()
    {
        var model = new VisionMambaModel<float>(
            CreateArch(16, 16, 1, 3),
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        var input = CreateRandomTensor(new[] { 1, 1, 16, 16 });
        var output1 = model.Predict(input);
        model.ResetState();

        var output2 = model.Predict(input);
        Assert.NotNull(output2);
        Assert.False(ContainsNaN(output2));

        // After reset, same input should produce same output
        var arr1 = output1.ToArray();
        var arr2 = output2.ToArray();
        for (int i = 0; i < arr1.Length; i++)
        {
            Assert.True(MathF.Abs(arr1[i] - arr2[i]) < 1e-5f,
                $"ResetState mismatch at {i}: {arr1[i]:G6} vs {arr2[i]:G6}");
        }
    }

    [Fact]
    public void DifferentImageSizes_Work()
    {
        // Rectangular image
        var model = new VisionMambaModel<float>(
            CreateArch(32, 16, 1, 5),
            imageHeight: 32, imageWidth: 16, patchSize: 8,
            channels: 1, modelDimension: 16, numLayers: 2, numClasses: 5,
            stateDimension: 4, scanPattern: VisionScanPattern.Continuous);

        Assert.Equal(8, model.NumPatches); // (32/8) * (16/8) = 4*2 = 8

        var input = CreateRandomTensor(new[] { 1, 1, 32, 16 });
        var output = model.Predict(input);

        Assert.Equal(new[] { 1, 5 }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Predict_Double_ProducesValidOutput()
    {
        var model = new VisionMambaModel<double>(
            CreateDoubleArch(),
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        var input = CreateRandomDoubleTensor(new[] { 1, 1, 16, 16 });
        var output = model.Predict(input);

        Assert.Equal(new[] { 1, 3 }, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    private static Tensor<double> CreateRandomDoubleTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble() * 2 - 1;
        }
        return tensor;
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (float.IsNaN(value)) return true;
        }
        return false;
    }

    private static bool ContainsNaNDouble(Tensor<double> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (double.IsNaN(value)) return true;
        }
        return false;
    }

    #endregion
}
