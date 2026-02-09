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
    [Fact]
    public void Constructor_ValidParameters_CreatesModel()
    {
        var model = new VisionMambaModel<float>(
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
            new VisionMambaModel<float>(imageHeight: 30, imageWidth: 32, patchSize: 8));
    }

    [Fact]
    public void Constructor_ThrowsWhenImageHeightNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(imageHeight: 0, imageWidth: 32, patchSize: 8));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumClassesNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(imageHeight: 16, imageWidth: 16, patchSize: 4, numClasses: 0));
    }

    [Theory]
    [InlineData(VisionScanPattern.Bidirectional)]
    [InlineData(VisionScanPattern.CrossScan)]
    [InlineData(VisionScanPattern.Continuous)]
    public void Forward_4D_AllScanPatterns_ProduceValidOutput(VisionScanPattern pattern)
    {
        int batchSize = 2;
        int height = 16;
        int width = 16;
        int channels = 3;
        int patchSize = 4;
        int numClasses = 5;

        var model = new VisionMambaModel<float>(
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4, scanPattern: pattern);

        var input = CreateRandomTensor(new[] { batchSize, channels, height, width });
        var output = model.Forward(input);

        Assert.Equal(new[] { batchSize, numClasses }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_3D_ProducesValidOutput()
    {
        int height = 16;
        int width = 16;
        int channels = 3;
        int patchSize = 4;
        int numClasses = 5;

        var model = new VisionMambaModel<float>(
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4);

        var input = CreateRandomTensor(new[] { channels, height, width });
        var output = model.Forward(input);

        Assert.Equal(new[] { numClasses }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_ProducesValidGradients()
    {
        int height = 16;
        int width = 16;
        int channels = 1;
        int patchSize = 4;
        int numClasses = 3;

        var model = new VisionMambaModel<float>(
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4, scanPattern: VisionScanPattern.Continuous);

        var input = CreateRandomTensor(new[] { 1, channels, height, width });
        var output = model.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = model.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        var model = new VisionMambaModel<float>(
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);
        var grad = CreateRandomTensor(new[] { 1, 3 });

        Assert.Throws<InvalidOperationException>(() => model.Backward(grad));
    }

    [Fact]
    public void FullTrainingStep_ForwardBackwardUpdate_NoErrors()
    {
        int height = 16;
        int width = 16;
        int channels = 1;
        int patchSize = 4;
        int numClasses = 3;

        var model = new VisionMambaModel<float>(
            height, width, patchSize, channels,
            modelDimension: 16, numLayers: 2, numClasses: numClasses,
            stateDimension: 4);

        var input = CreateRandomTensor(new[] { 1, channels, height, width });
        var output = model.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        model.Backward(grad);
        model.UpdateParameters(0.001f);

        model.ResetState();
        var output2 = model.Forward(input);
        Assert.Equal(output.Shape, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        var model = new VisionMambaModel<float>(
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
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var model = new VisionMambaModel<float>(
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var model = new VisionMambaModel<float>(
            32, 32, 8, 3, modelDimension: 64, numLayers: 4, numClasses: 10,
            scanPattern: VisionScanPattern.CrossScan);

        var metadata = model.GetMetadata();

        Assert.True(metadata.ContainsKey("ImageHeight"));
        Assert.True(metadata.ContainsKey("ImageWidth"));
        Assert.True(metadata.ContainsKey("PatchSize"));
        Assert.True(metadata.ContainsKey("NumClasses"));
        Assert.True(metadata.ContainsKey("ScanPattern"));
        Assert.Equal("32", metadata["ImageHeight"]);
        Assert.Equal("CrossScan", metadata["ScanPattern"]);
    }

    [Fact]
    public void ResetState_AllowsReuse()
    {
        var model = new VisionMambaModel<float>(
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        var input = CreateRandomTensor(new[] { 1, 1, 16, 16 });
        model.Forward(input);
        model.ResetState();

        var output = model.Forward(input);
        Assert.NotNull(output);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void DifferentImageSizes_Work()
    {
        // Rectangular image
        var model = new VisionMambaModel<float>(
            imageHeight: 32, imageWidth: 16, patchSize: 8,
            channels: 1, modelDimension: 16, numLayers: 2, numClasses: 5,
            stateDimension: 4, scanPattern: VisionScanPattern.Continuous);

        Assert.Equal(8, model.NumPatches); // (32/8) * (16/8) = 4*2 = 8

        var input = CreateRandomTensor(new[] { 1, 1, 32, 16 });
        var output = model.Forward(input);

        Assert.Equal(new[] { 1, 5 }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        var model = new VisionMambaModel<double>(
            16, 16, 4, 1, modelDimension: 16, numLayers: 2, numClasses: 3, stateDimension: 4);

        var input = CreateRandomDoubleTensor(new[] { 1, 1, 16, 16 });
        var output = model.Forward(input);

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
