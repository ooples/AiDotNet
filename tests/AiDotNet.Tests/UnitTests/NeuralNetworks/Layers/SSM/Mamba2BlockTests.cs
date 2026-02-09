using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="Mamba2Block{T}"/>.
/// </summary>
public class Mamba2BlockTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesBlock()
    {
        int seqLen = 16;
        int modelDim = 64;
        int stateDim = 16;
        int numHeads = 8;
        int expandFactor = 2;
        int convKernel = 4;
        int chunkSize = 16;

        var block = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads, expandFactor, convKernel, chunkSize);

        Assert.Equal(modelDim, block.ModelDimension);
        Assert.Equal(stateDim, block.StateDimension);
        Assert.Equal(modelDim * expandFactor, block.InnerDimension);
        Assert.Equal(numHeads, block.NumHeads);
        Assert.Equal(modelDim * expandFactor / numHeads, block.HeadDimension);
        Assert.Equal(convKernel, block.ConvKernelSize);
        Assert.Equal(chunkSize, block.ChunkSize);
    }

    [Fact]
    public void Constructor_DefaultParameters_UsesCorrectDefaults()
    {
        var block = new Mamba2Block<float>(16);

        Assert.Equal(256, block.ModelDimension);
        Assert.Equal(64, block.StateDimension);
        Assert.Equal(512, block.InnerDimension); // 256 * 2
        Assert.Equal(8, block.NumHeads);
        Assert.Equal(64, block.HeadDimension); // 512 / 8
        Assert.Equal(4, block.ConvKernelSize);
        Assert.Equal(64, block.ChunkSize);
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenStateDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 64, stateDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumHeadsNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 64, numHeads: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenExpandFactorNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 64, expandFactor: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenConvKernelNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 64, convKernelSize: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenChunkSizeNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 64, chunkSize: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenInnerDimNotDivisibleByHeads()
    {
        // modelDim=64, expandFactor=2 -> innerDim=128, numHeads=3 -> 128%3!=0
        Assert.Throws<ArgumentException>(() =>
            new Mamba2Block<float>(16, modelDimension: 64, numHeads: 3, expandFactor: 2));
    }

    [Theory]
    [InlineData(64, 16, 8, 2, 4)]
    [InlineData(32, 8, 4, 2, 4)]
    [InlineData(128, 32, 8, 2, 4)]
    public void Forward_3D_ProducesValidOutput(int modelDim, int stateDim, int numHeads, int expand, int convK)
    {
        int batchSize = 2;
        int seqLen = 8;
        var block = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads, expand, convK);
        var input = CreateRandomTensor(new[] { batchSize, seqLen, modelDim });

        var output = block.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Theory]
    [InlineData(64, 16)]
    [InlineData(32, 8)]
    public void Forward_2D_ProducesValidOutput(int modelDim, int stateDim)
    {
        int seqLen = 8;
        var block = new Mamba2Block<float>(seqLen, modelDim, stateDim);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });

        var output = block.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads: 4);
        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });

        var output = block.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = block.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backward_2D_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads: 4);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });

        var output = block.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = block.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        for (int i = 0; i < inputGrad.Length; i++)
            Assert.False(float.IsNaN(inputGrad[i]), $"NaN detected in backward output at index {i}");
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        var block = new Mamba2Block<float>(4, 32, 8, numHeads: 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 32 });

        Assert.Throws<InvalidOperationException>(() => block.Backward(grad));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads: 4);

        var params1 = block.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(block.ParameterCount, params1.Length);

        block.SetParameters(params1);

        var params2 = block.GetParameters();
        Assert.Equal(params1.Length, params2.Length);

        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SetParameters_ThrowsOnWrongLength()
    {
        var block = new Mamba2Block<float>(4, 32, 8, numHeads: 4);
        var wrongParams = new Vector<float>(10);

        Assert.Throws<ArgumentException>(() => block.SetParameters(wrongParams));
    }

    [Fact]
    public void ResetState_ClearsInternalState()
    {
        var block = new Mamba2Block<float>(4, 32, 8, numHeads: 4);
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        block.Forward(input);

        block.ResetState();

        var output = block.Forward(input);
        Assert.NotNull(output);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_DeterministicWithSameParameters()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block1 = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads: 4);
        var block2 = new Mamba2Block<float>(seqLen, modelDim, stateDim, numHeads: 4);

        block2.SetParameters(block1.GetParameters());

        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        var output1 = block1.Forward(input);
        var output2 = block2.Forward(input);

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
        var block = new Mamba2Block<float>(4, 32, 8, numHeads: 4);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var block = new Mamba2Block<float>(8, 64, 16, numHeads: 8, expandFactor: 2, convKernelSize: 4, chunkSize: 32);

        var metadata = block.GetMetadata();

        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("StateDimension"));
        Assert.True(metadata.ContainsKey("InnerDimension"));
        Assert.True(metadata.ContainsKey("NumHeads"));
        Assert.True(metadata.ContainsKey("HeadDimension"));
        Assert.True(metadata.ContainsKey("ConvKernelSize"));
        Assert.True(metadata.ContainsKey("ChunkSize"));
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("16", metadata["StateDimension"]);
        Assert.Equal("128", metadata["InnerDimension"]);
        Assert.Equal("8", metadata["NumHeads"]);
        Assert.Equal("32", metadata["ChunkSize"]);
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new Mamba2Block<double>(seqLen, modelDim, stateDim, numHeads: 4);
        var input = CreateRandomDoubleTensor(new[] { 1, seqLen, modelDim });

        var output = block.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void Backward_Double_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new Mamba2Block<double>(seqLen, modelDim, stateDim, numHeads: 4);
        var input = CreateRandomDoubleTensor(new[] { 1, seqLen, modelDim });

        var output = block.Forward(input);
        var grad = CreateRandomDoubleTensor(output.Shape);
        var inputGrad = block.Backward(grad);

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
