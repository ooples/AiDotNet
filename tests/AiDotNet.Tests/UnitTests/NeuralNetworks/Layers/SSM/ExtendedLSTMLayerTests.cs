using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="ExtendedLSTMLayer{T}"/>.
/// </summary>
public class ExtendedLSTMLayerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesLayer()
    {
        int seqLen = 16;
        int modelDim = 64;
        int numHeads = 8;

        var layer = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);

        Assert.Equal(modelDim, layer.ModelDimension);
        Assert.Equal(numHeads, layer.NumHeads);
        Assert.Equal(modelDim / numHeads, layer.HeadDimension);
    }

    [Fact]
    public void Constructor_DefaultParameters_UsesCorrectDefaults()
    {
        var layer = new ExtendedLSTMLayer<float>(16);

        Assert.Equal(256, layer.ModelDimension);
        Assert.Equal(8, layer.NumHeads);
        Assert.Equal(32, layer.HeadDimension); // 256 / 8
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new ExtendedLSTMLayer<float>(16, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumHeadsNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new ExtendedLSTMLayer<float>(16, modelDimension: 64, numHeads: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimNotDivisibleByHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            new ExtendedLSTMLayer<float>(16, modelDimension: 65, numHeads: 8));
    }

    [Theory]
    [InlineData(64, 8)]
    [InlineData(32, 4)]
    [InlineData(128, 8)]
    [InlineData(64, 4)]
    public void Forward_3D_ProducesValidOutput(int modelDim, int numHeads)
    {
        int batchSize = 2;
        int seqLen = 8;
        var layer = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);
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
        var layer = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);
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
        var layer = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);
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
        var layer = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });

        var output = layer.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        var layer = new ExtendedLSTMLayer<float>(4, 32, 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 32 });

        Assert.Throws<InvalidOperationException>(() => layer.Backward(grad));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);

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
        var layer = new ExtendedLSTMLayer<float>(4, 32, 4);
        var wrongParams = new Vector<float>(10);

        Assert.Throws<ArgumentException>(() => layer.SetParameters(wrongParams));
    }

    [Fact]
    public void ResetState_ClearsInternalState()
    {
        var layer = new ExtendedLSTMLayer<float>(4, 32, 4);
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        var output1 = layer.Forward(input);

        layer.ResetState();

        var output2 = layer.Forward(input);
        Assert.NotNull(output2);
        Assert.False(ContainsNaN(output2));

        var arr1 = output1.ToArray();
        var arr2 = output2.ToArray();
        for (int i = 0; i < arr1.Length; i++)
        {
            Assert.True(MathF.Abs(arr1[i] - arr2[i]) < 1e-5f,
                $"ResetState mismatch at {i}: {arr1[i]:G6} vs {arr2[i]:G6}");
        }
    }

    [Fact]
    public void Forward_DeterministicWithSameParameters()
    {
        int seqLen = 4;
        int modelDim = 32;
        int numHeads = 4;
        var layer1 = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);
        var layer2 = new ExtendedLSTMLayer<float>(seqLen, modelDim, numHeads);

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
        var layer = new ExtendedLSTMLayer<float>(4, 32, 4);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var layer = new ExtendedLSTMLayer<float>(8, 64, 8);

        var metadata = layer.GetMetadata();

        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("NumHeads"));
        Assert.True(metadata.ContainsKey("HeadDimension"));
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("8", metadata["NumHeads"]);
        Assert.Equal("8", metadata["HeadDimension"]);
    }

    [Fact]
    public void GetOutputProjectionWeights_ReturnsValidTensor()
    {
        int modelDim = 64;
        int numHeads = 8;
        var layer = new ExtendedLSTMLayer<float>(8, modelDim, numHeads);

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
        var layer = new ExtendedLSTMLayer<double>(seqLen, modelDim, numHeads);
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
        var layer = new ExtendedLSTMLayer<double>(seqLen, modelDim, numHeads);
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
