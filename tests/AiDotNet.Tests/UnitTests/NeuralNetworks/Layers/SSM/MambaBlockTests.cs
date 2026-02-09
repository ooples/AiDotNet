using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="MambaBlock{T}"/>.
/// </summary>
public class MambaBlockTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesBlock()
    {
        int seqLen = 16;
        int modelDim = 64;
        int stateDim = 16;
        int expandFactor = 2;
        int convKernel = 4;

        var block = new MambaBlock<float>(seqLen, modelDim, stateDim, expandFactor, convKernel);

        Assert.Equal(modelDim, block.ModelDimension);
        Assert.Equal(stateDim, block.StateDimension);
        Assert.Equal(modelDim * expandFactor, block.InnerDimension);
        Assert.Equal(convKernel, block.ConvKernelSize);
        Assert.Equal((int)Math.Ceiling((double)modelDim / 16), block.DtRank);
    }

    [Fact]
    public void Constructor_DefaultParameters_UsesCorrectDefaults()
    {
        var block = new MambaBlock<float>(16);

        // Default values from the Mamba paper
        Assert.Equal(256, block.ModelDimension);
        Assert.Equal(16, block.StateDimension);
        Assert.Equal(512, block.InnerDimension); // 256 * 2
        Assert.Equal(4, block.ConvKernelSize);
        Assert.Equal(16, block.DtRank); // ceil(256/16)
    }

    [Fact]
    public void Constructor_CustomDtRank_UsesProvidedValue()
    {
        var block = new MambaBlock<float>(16, modelDimension: 64, dtRank: 8);

        Assert.Equal(8, block.DtRank);
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaBlock<float>(16, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenStateDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaBlock<float>(16, modelDimension: 64, stateDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenExpandFactorNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaBlock<float>(16, modelDimension: 64, expandFactor: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenConvKernelNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaBlock<float>(16, modelDimension: 64, convKernelSize: 0));
    }

    [Theory]
    [InlineData(64, 16, 2, 4)]
    [InlineData(32, 8, 2, 4)]
    [InlineData(128, 32, 2, 4)]
    [InlineData(64, 16, 4, 4)]  // larger expansion
    [InlineData(64, 16, 2, 8)]  // larger conv kernel
    public void Forward_3D_ProducesValidOutput(int modelDim, int stateDim, int expand, int convK)
    {
        int batchSize = 2;
        int seqLen = 8;
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim, expand, convK);
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
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim);
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
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim);
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
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });

        var output = block.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = block.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var grad = CreateRandomTensor(new[] { 1, 4, 32 });

        Assert.Throws<InvalidOperationException>(() => block.Backward(grad));
    }

    [Fact]
    public void ParameterCount_MatchesExpectedFormula()
    {
        int modelDim = 32;
        int stateDim = 8;
        int expandFactor = 2;
        int convKernel = 4;
        int innerDim = modelDim * expandFactor; // 64
        int dtRank = (int)Math.Ceiling((double)modelDim / 16); // 2

        var block = new MambaBlock<float>(8, modelDim, stateDim, expandFactor, convKernel);

        int expectedParams =
            modelDim * (innerDim * 2) + (innerDim * 2) +   // input proj weights + bias
            innerDim * convKernel + innerDim +               // conv weights + bias
            innerDim * (dtRank + stateDim * 2) +            // x_proj weights
            dtRank * innerDim + innerDim +                   // dt_proj weights + bias
            innerDim * stateDim +                            // A_log
            innerDim +                                       // D param
            innerDim * modelDim + modelDim;                  // output proj weights + bias

        Assert.Equal(expectedParams, block.ParameterCount);
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim);

        var params1 = block.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(block.ParameterCount, params1.Length);

        // Set parameters back (should not throw)
        block.SetParameters(params1);

        var params2 = block.GetParameters();
        Assert.Equal(params1.Length, params2.Length);

        // Values should match exactly
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SetParameters_ThrowsOnWrongLength()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var wrongParams = new Vector<float>(10); // wrong length

        Assert.Throws<ArgumentException>(() => block.SetParameters(wrongParams));
    }

    [Fact]
    public void ResetState_ClearsInternalState()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        block.Forward(input);

        block.ResetState();

        // Should not throw and should be usable again
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
        var block1 = new MambaBlock<float>(seqLen, modelDim, stateDim);
        var block2 = new MambaBlock<float>(seqLen, modelDim, stateDim);

        // Copy parameters from block1 to block2
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
    public void ParameterCount_IncreasesWithExpansion()
    {
        int modelDim = 32;
        int stateDim = 8;

        var block2x = new MambaBlock<float>(8, modelDim, stateDim, expandFactor: 2);
        var block4x = new MambaBlock<float>(8, modelDim, stateDim, expandFactor: 4);

        Assert.True(block4x.ParameterCount > block2x.ParameterCount,
            $"4x expand ({block4x.ParameterCount}) should have more params than 2x ({block2x.ParameterCount})");
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var block = new MambaBlock<float>(8, 64, 16, expandFactor: 2, convKernelSize: 4);

        // GetMetadata is internal but accessible via InternalsVisibleTo
        var metadata = block.GetMetadata();

        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("StateDimension"));
        Assert.True(metadata.ContainsKey("InnerDimension"));
        Assert.True(metadata.ContainsKey("ConvKernelSize"));
        Assert.True(metadata.ContainsKey("DtRank"));
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("16", metadata["StateDimension"]);
        Assert.Equal("128", metadata["InnerDimension"]);
    }

    [Fact]
    public void GetWeightAccessors_ReturnCorrectShapes()
    {
        int modelDim = 64;
        int stateDim = 16;
        int expandFactor = 2;
        int innerDim = modelDim * expandFactor;

        var block = new MambaBlock<float>(8, modelDim, stateDim, expandFactor);

        var inputWeights = block.GetInputProjectionWeights();
        Assert.Equal(new[] { modelDim, innerDim * 2 }, inputWeights.Shape);

        var outputWeights = block.GetOutputProjectionWeights();
        Assert.Equal(new[] { innerDim, modelDim }, outputWeights.Shape);

        var aLog = block.GetALogParameter();
        Assert.Equal(new[] { innerDim, stateDim }, aLog.Shape);

        var dParam = block.GetDParameter();
        Assert.Equal(new[] { innerDim }, dParam.Shape);
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new MambaBlock<double>(seqLen, modelDim, stateDim);
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
        var block = new MambaBlock<double>(seqLen, modelDim, stateDim);
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
