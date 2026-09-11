using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Integration tests for <see cref="VisionMambaModel{T}"/>.
/// Tests full forward-backward-parameter round-trips with various configurations.
/// </summary>
public class VisionMambaModelTests
{
    public VisionMambaModelTests() => TestModuleInitializer.EnsureInitialized();

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

    [Fact(Timeout = 120000)]
    public async Task Constructor_ValidParameters_CreatesModel()
    {
        var model = new VisionMambaModel<float>(CreateArch(), options: new VisionMambaOptions { ImageHeight = 32, ImageWidth = 32, PatchSize = 8, Channels = 3, ModelDimension = 32, NumLayers = 2, NumClasses = 10 });

        Assert.Equal(32, model.ImageHeight);
        Assert.Equal(32, model.ImageWidth);
        Assert.Equal(8, model.PatchSize);
        Assert.Equal(32, model.ModelDimension);
        Assert.Equal(2, model.NumLayers);
        Assert.Equal(10, model.NumClasses);
        Assert.Equal(16, model.NumPatches); // (32/8) * (32/8) = 4*4 = 16
        Assert.Equal(VisionScanPattern.Bidirectional, model.ScanPattern);
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenImageNotDivisibleByPatch()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(CreateArch(30, 32), options: new VisionMambaOptions { ImageHeight = 30, ImageWidth = 32, PatchSize = 8 }));
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenImageHeightNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(CreateArch(1, 32), options: new VisionMambaOptions { ImageHeight = 0, ImageWidth = 32, PatchSize = 8 }));
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenNumClassesNotPositive()
    {
        var options = new VisionMambaOptions { ImageHeight = 16, ImageWidth = 16, PatchSize = 4 };
        options.Validate();
        options.NumClasses = 0;
        var error = Assert.Throws<ArgumentException>(() =>
            new VisionMambaModel<float>(CreateArch(16, 16, numClasses: 1), options));
        Assert.Equal("options", error.ParamName);
        Assert.Contains(nameof(VisionMambaOptions) + "." + nameof(VisionMambaOptions.NumClasses), error.Message);
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

        var model = new VisionMambaModel<float>(CreateArch(height, width, channels, numClasses), options: new VisionMambaOptions { ImageHeight = height, ImageWidth = width, PatchSize = patchSize, Channels = channels, ModelDimension = 16, NumLayers = 2, NumClasses = numClasses, StateDimension = 4, ScanPattern = pattern });

        var input = CreateRandomTensor(new[] { batchSize, channels, height, width });
        var output = model.Predict(input);

        Assert.Equal(new[] { batchSize, numClasses }, output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_3D_ProducesValidOutput()
    {
        int height = 16;
        int width = 16;
        int channels = 3;
        int patchSize = 4;
        int numClasses = 5;

        var model = new VisionMambaModel<float>(CreateArch(height, width, channels, numClasses), options: new VisionMambaOptions { ImageHeight = height, ImageWidth = width, PatchSize = patchSize, Channels = channels, ModelDimension = 16, NumLayers = 2, NumClasses = numClasses, StateDimension = 4 });

        var input = CreateRandomTensor(new[] { channels, height, width });
        var output = model.Predict(input);

        Assert.Equal(new[] { numClasses }, output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }



    [Fact(Timeout = 120000)]
    public async Task Train_ForwardBackwardUpdate_NoErrors()
    {
        int height = 16;
        int width = 16;
        int channels = 1;
        int patchSize = 4;
        int numClasses = 3;

        var model = new VisionMambaModel<float>(CreateArch(height, width, channels, numClasses), options: new VisionMambaOptions { ImageHeight = height, ImageWidth = width, PatchSize = patchSize, Channels = channels, ModelDimension = 16, NumLayers = 2, NumClasses = numClasses, StateDimension = 4 });

        var input = CreateRandomTensor(new[] { 1, channels, height, width });
        var expected = new Tensor<float>(new[] { 1, numClasses });
        expected[new[] { 0, 0 }] = 1.0f; // one-hot target

        model.Train(input, expected);

        model.ResetState();
        var output2 = model.Predict(input);
        Assert.Equal(new[] { 1, numClasses }, output2.Shape.ToArray());
        Assert.False(ContainsNaN(output2));
    }

    [Fact(Timeout = 120000)]
    public async Task GetParameters_SetParameters_RoundTrip()
    {
        var model = new VisionMambaModel<float>(CreateArch(16, 16, 1, 3), options: new VisionMambaOptions { ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1, ModelDimension = 16, NumLayers = 2, NumClasses = 3, StateDimension = 4 });

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

    [Fact(Timeout = 120000)]
    public async Task SetParameters_ThrowsOnWrongLength()
    {
        var model = new VisionMambaModel<float>(CreateArch(16, 16, 1, 3), options: new VisionMambaOptions { ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1, ModelDimension = 16, NumLayers = 2, NumClasses = 3, StateDimension = 4 });

        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact(Timeout = 120000)]
    public async Task SupportsTraining_ReturnsTrue()
    {
        var model = new VisionMambaModel<float>(CreateArch(16, 16, 1, 3), options: new VisionMambaOptions { ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1, ModelDimension = 16, NumLayers = 2, NumClasses = 3, StateDimension = 4 });
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task GetModelMetadata_ContainsExpectedKeys()
    {
        var model = new VisionMambaModel<float>(CreateArch(32, 32, 3, 10), options: new VisionMambaOptions { ImageHeight = 32, ImageWidth = 32, PatchSize = 8, Channels = 3, ModelDimension = 64, NumLayers = 4, NumClasses = 10, ScanPattern = VisionScanPattern.CrossScan });

        var metadata = model.GetModelMetadata();


        Assert.True(metadata.AdditionalInfo.ContainsKey("ImageHeight"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("ImageWidth"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("PatchSize"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("NumClasses"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("ScanPattern"));
        Assert.Equal(32, metadata.AdditionalInfo["ImageHeight"]);
        Assert.Equal("CrossScan", metadata.AdditionalInfo["ScanPattern"]);
    }

    [Fact(Timeout = 120000)]
    public async Task ResetState_AllowsReuse()
    {
        var model = new VisionMambaModel<float>(CreateArch(16, 16, 1, 3), options: new VisionMambaOptions { ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1, ModelDimension = 16, NumLayers = 2, NumClasses = 3, StateDimension = 4 });

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

    [Fact(Timeout = 120000)]
    public async Task DifferentImageSizes_Work()
    {
        // Rectangular image
        var model = new VisionMambaModel<float>(CreateArch(32, 16, 1, 5), options: new VisionMambaOptions { ImageHeight = 32, ImageWidth = 16, PatchSize = 8, Channels = 1, ModelDimension = 16, NumLayers = 2, NumClasses = 5, StateDimension = 4, ScanPattern = VisionScanPattern.Continuous });

        Assert.Equal(8, model.NumPatches); // (32/8) * (16/8) = 4*2 = 8

        var input = CreateRandomTensor(new[] { 1, 1, 32, 16 });
        var output = model.Predict(input);

        Assert.Equal(new[] { 1, 5 }, output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_Double_ProducesValidOutput()
    {
        var model = new VisionMambaModel<double>(CreateDoubleArch(), options: new VisionMambaOptions { ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1, ModelDimension = 16, NumLayers = 2, NumClasses = 3, StateDimension = 4 });

        var input = CreateRandomDoubleTensor(new[] { 1, 1, 16, 16 });
        var output = model.Predict(input);

        Assert.Equal(new[] { 1, 3 }, output.Shape.ToArray());
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
