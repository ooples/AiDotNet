using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="GatedDeltaNetLayer{T}"/>.
/// </summary>
public class GatedDeltaNetLayerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesLayer()
    {
        int seqLen = 16;
        int modelDim = 64;
        int numHeads = 8;
        int convKernel = 4;

        var layer = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads, convKernel);

        Assert.Equal(modelDim, layer.ModelDimension);
        Assert.Equal(numHeads, layer.NumHeads);
        Assert.Equal(modelDim / numHeads, layer.HeadDimension);
        Assert.Equal(convKernel, layer.ConvKernelSize);
    }

    [Fact]
    public void Constructor_DefaultParameters_UsesCorrectDefaults()
    {
        var layer = new GatedDeltaNetLayer<float>(16);

        Assert.Equal(256, layer.ModelDimension);
        Assert.Equal(8, layer.NumHeads);
        Assert.Equal(32, layer.HeadDimension); // 256 / 8
        Assert.Equal(4, layer.ConvKernelSize);
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new GatedDeltaNetLayer<float>(16, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumHeadsNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new GatedDeltaNetLayer<float>(16, modelDimension: 64, numHeads: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimNotDivisibleByHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            new GatedDeltaNetLayer<float>(16, modelDimension: 65, numHeads: 8));
    }

    [Fact]
    public void Constructor_ThrowsWhenConvKernelNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new GatedDeltaNetLayer<float>(16, modelDimension: 64, convKernelSize: 0));
    }

    [Theory]
    [InlineData(64, 8, 4)]
    [InlineData(32, 4, 4)]
    [InlineData(128, 8, 4)]
    [InlineData(64, 4, 8)]
    public void Forward_3D_ProducesValidOutput(int modelDim, int numHeads, int convK)
    {
        int batchSize = 2;
        int seqLen = 8;
        var layer = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads, convK);
        var input = CreateRandomTensor(new[] { batchSize, seqLen, modelDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Theory]
    [InlineData(64, 8)]
    [InlineData(32, 4)]
    public void Forward_2D_ProducesValidOutput(int modelDim, int numHeads)
    {
        int seqLen = 8;
        var layer = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads);
        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });

        var output = layer.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backward_2D_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });

        var output = layer.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        for (int i = 0; i < inputGrad.Length; i++)
            Assert.False(float.IsNaN(inputGrad[i]), $"NaN detected in backward output at index {i}");
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        var layer = new GatedDeltaNetLayer<float>(4, 32, 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 32 });

        Assert.Throws<InvalidOperationException>(() => layer.Backward(grad));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads);

        var params1 = layer.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(layer.ParameterCount, params1.Length);

        layer.SetParameters(params1);

        var params2 = layer.GetParameters();
        Assert.Equal(params1.Length, params2.Length);

        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SetParameters_ThrowsOnWrongLength()
    {
        var layer = new GatedDeltaNetLayer<float>(4, 32, 4);
        var wrongParams = new Vector<float>(10);

        Assert.Throws<ArgumentException>(() => layer.SetParameters(wrongParams));
    }

    [Fact]
    public void ResetState_ClearsInternalState()
    {
        var layer = new GatedDeltaNetLayer<float>(4, 32, 4);
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        layer.Forward(input);

        layer.ResetState();

        var output = layer.Forward(input);
        Assert.NotNull(output);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_DeterministicWithSameParameters()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer1 = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads);
        var layer2 = new GatedDeltaNetLayer<float>(seqLen, modelDim, numHeads);

        layer2.SetParameters(layer1.GetParameters());

        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        var output1 = layer1.Forward(input);
        var output2 = layer2.Forward(input);

        var arr1 = output1.ToArray();
        var arr2 = output2.ToArray();

        Assert.Equal(arr1.Length, arr2.Length);
        for (int i = 0; i < arr1.Length; i++)
        {
            Assert.True(MathF.Abs(arr1[i] - arr2[i]) < 1e-5f,
                $"Output mismatch at index {i}: {arr1[i]:G6} vs {arr2[i]:G6}");
        }
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var layer = new GatedDeltaNetLayer<float>(4, 32, 4);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var layer = new GatedDeltaNetLayer<float>(8, 64, 8, convKernelSize: 4);

        var metadata = layer.GetMetadata();

        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("NumHeads"));
        Assert.True(metadata.ContainsKey("HeadDimension"));
        Assert.True(metadata.ContainsKey("ConvKernelSize"));
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("8", metadata["NumHeads"]);
        Assert.Equal("8", metadata["HeadDimension"]);
        Assert.Equal("4", metadata["ConvKernelSize"]);
    }

    [Fact]
    public void GetOutputProjectionWeights_ReturnsValidTensor()
    {
        int modelDim = 64;
        int numHeads = 8;
        var layer = new GatedDeltaNetLayer<float>(8, modelDim, numHeads);

        var outputWeights = layer.GetOutputProjectionWeights();

        Assert.NotNull(outputWeights);
        Assert.Equal(new[] { modelDim, modelDim }, outputWeights.Shape);
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer = new GatedDeltaNetLayer<double>(seqLen, modelDim, numHeads);
        var input = CreateRandomDoubleTensor(new[] { 1, seqLen, modelDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void Backward_Double_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer = new GatedDeltaNetLayer<double>(seqLen, modelDim, numHeads);
        var input = CreateRandomDoubleTensor(new[] { 1, seqLen, modelDim });

        var output = layer.Forward(input);
        var grad = CreateRandomDoubleTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaNDouble(inputGrad));
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    private static Tensor<double> CreateRandomDoubleTensor(int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(42);
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
