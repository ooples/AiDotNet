using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="S4DLayer{T}"/>.
/// </summary>
public class S4DLayerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesLayer()
    {
        int seqLen = 16;
        int modelDim = 64;
        int stateDim = 32;
        int expandFactor = 2;

        var layer = new S4DLayer<float>(seqLen, modelDim, stateDim, expandFactor);

        Assert.Equal(modelDim, layer.ModelDimension);
        Assert.Equal(stateDim, layer.StateDimension);
        Assert.Equal(modelDim * expandFactor, layer.InnerDimension);
    }

    [Fact]
    public void Constructor_DefaultParameters_UsesCorrectDefaults()
    {
        var layer = new S4DLayer<float>(16);

        Assert.Equal(256, layer.ModelDimension);
        Assert.Equal(64, layer.StateDimension);
        Assert.Equal(256, layer.InnerDimension); // 256 * 1 (expandFactor=1)
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new S4DLayer<float>(16, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenStateDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new S4DLayer<float>(16, modelDimension: 64, stateDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenExpandFactorNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new S4DLayer<float>(16, modelDimension: 64, expandFactor: 0));
    }

    [Theory]
    [InlineData(64, 16, 1)]
    [InlineData(32, 8, 1)]
    [InlineData(64, 32, 2)]
    [InlineData(128, 16, 1)]
    public void Forward_3D_ProducesValidOutput(int modelDim, int stateDim, int expand)
    {
        int batchSize = 2;
        int seqLen = 8;
        var layer = new S4DLayer<float>(seqLen, modelDim, stateDim, expand);
        var input = CreateRandomTensor(new[] { batchSize, seqLen, modelDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Theory]
    [InlineData(64, 16)]
    [InlineData(32, 8)]
    public void Forward_2D_ProducesValidOutput(int modelDim, int stateDim)
    {
        int seqLen = 8;
        var layer = new S4DLayer<float>(seqLen, modelDim, stateDim);
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
        int stateDim = 8;
        var layer = new S4DLayer<float>(seqLen, modelDim, stateDim);
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
        int stateDim = 8;
        var layer = new S4DLayer<float>(seqLen, modelDim, stateDim);
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
        var layer = new S4DLayer<float>(4, 32, 8);
        var grad = CreateRandomTensor(new[] { 1, 4, 32 });

        Assert.Throws<InvalidOperationException>(() => layer.Backward(grad));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var layer = new S4DLayer<float>(seqLen, modelDim, stateDim);

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
        var layer = new S4DLayer<float>(4, 32, 8);
        var wrongParams = new Vector<float>(10);

        Assert.Throws<ArgumentException>(() => layer.SetParameters(wrongParams));
    }

    [Fact]
    public void ResetState_ClearsInternalState()
    {
        var layer = new S4DLayer<float>(4, 32, 8);
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        var output1 = layer.Forward(input);

        layer.ResetState();

        var output2 = layer.Forward(input);
        Assert.NotNull(output2);
        Assert.False(ContainsNaN(output2));

        // After reset, same input should produce same output (deterministic)
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
        int stateDim = 8;
        var layer1 = new S4DLayer<float>(seqLen, modelDim, stateDim);
        var layer2 = new S4DLayer<float>(seqLen, modelDim, stateDim);

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
    public void SupportsTraining_ReturnsFalse_UntilFullBackwardImplemented()
    {
        var layer = new S4DLayer<float>(4, 32, 8);
        Assert.False(layer.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var layer = new S4DLayer<float>(8, 64, 16, expandFactor: 2);

        var metadata = layer.GetMetadata();

        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("StateDimension"));
        Assert.True(metadata.ContainsKey("InnerDimension"));
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("16", metadata["StateDimension"]);
        Assert.Equal("128", metadata["InnerDimension"]);
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var layer = new S4DLayer<double>(seqLen, modelDim, stateDim);
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
        int stateDim = 8;
        var layer = new S4DLayer<double>(seqLen, modelDim, stateDim);
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
