using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="S6Scan{T}"/>.
/// </summary>
public class S6ScanTests
{
    [Fact]
    public void SequentialScanForward_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int seqLen = 4;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var (output, hiddenStates) = S6Scan<float>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        Assert.Equal(new[] { batchSize, seqLen, innerDim }, output.Shape);
        Assert.Equal(new[] { batchSize, seqLen + 1, innerDim, stateDim }, hiddenStates.Shape);
    }

    [Fact]
    public void SequentialScanForward_ProducesNoNaN()
    {
        int batchSize = 1;
        int seqLen = 4;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var (output, _) = S6Scan<float>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SequentialScanForward_HiddenStateZeroAtT0()
    {
        int batchSize = 1;
        int seqLen = 2;
        int innerDim = 4;
        int stateDim = 2;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var (_, hiddenStates) = S6Scan<float>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        // t=0 hidden state should be all zeros (initial state)
        var h0 = hiddenStates.GetSliceAlongDimension(0, 1);
        foreach (var val in h0.ToArray())
        {
            Assert.Equal(0f, val);
        }
    }

    [Fact]
    public void SequentialScanBackward_ProducesCorrectGradientShapes()
    {
        int batchSize = 1;
        int seqLen = 4;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var (output, hiddenStates) = S6Scan<float>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        var dOutput = CreateRandomTensor(output.Shape);
        var (dX, dDelta, dALog, dB, dC, dD) = S6Scan<float>.SequentialScanBackward(
            dOutput, x, delta, aLog, b, c, dParam, hiddenStates,
            batchSize, seqLen, innerDim, stateDim);

        Assert.Equal(new[] { batchSize, seqLen, innerDim }, dX.Shape);
        Assert.Equal(new[] { batchSize, seqLen, innerDim }, dDelta.Shape);
        Assert.Equal(new[] { innerDim, stateDim }, dALog.Shape);
        Assert.Equal(new[] { batchSize, seqLen, stateDim }, dB.Shape);
        Assert.Equal(new[] { batchSize, seqLen, stateDim }, dC.Shape);
        Assert.Equal(new[] { innerDim }, dD.Shape);
    }

    [Fact]
    public void SequentialScanBackward_ProducesNoNaN()
    {
        int batchSize = 1;
        int seqLen = 4;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var (output, hiddenStates) = S6Scan<float>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        var dOutput = CreateRandomTensor(output.Shape);
        var (dX, dDelta, dALog, dB, dC, dD) = S6Scan<float>.SequentialScanBackward(
            dOutput, x, delta, aLog, b, c, dParam, hiddenStates,
            batchSize, seqLen, innerDim, stateDim);

        Assert.False(ContainsNaN(dX));
        Assert.False(ContainsNaN(dDelta));
        Assert.False(ContainsNaN(dALog));
        Assert.False(ContainsNaN(dB));
        Assert.False(ContainsNaN(dC));
        Assert.False(ContainsNaN(dD));
    }

    [Fact]
    public void ParallelScan_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int seqLen = 8;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var output = S6Scan<float>.ParallelScan(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        Assert.Equal(new[] { batchSize, seqLen, innerDim }, output.Shape);
    }

    [Fact]
    public void ParallelScan_ProducesNoNaN()
    {
        int batchSize = 1;
        int seqLen = 4;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim });
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomTensor(new[] { innerDim });

        var output = S6Scan<float>.ParallelScan(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ParallelScan_MatchesSequentialScan()
    {
        int batchSize = 1;
        int seqLen = 4;
        int innerDim = 4;
        int stateDim = 2;

        var x = CreateRandomTensor(new[] { batchSize, seqLen, innerDim }, seed: 123);
        var delta = CreatePositiveTensor(new[] { batchSize, seqLen, innerDim }, seed: 124);
        var aLog = CreateRandomTensor(new[] { innerDim, stateDim }, seed: 125);
        var b = CreateRandomTensor(new[] { batchSize, seqLen, stateDim }, seed: 126);
        var c = CreateRandomTensor(new[] { batchSize, seqLen, stateDim }, seed: 127);
        var dParam = CreateRandomTensor(new[] { innerDim }, seed: 128);

        var (seqOutput, _) = S6Scan<float>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        var parOutput = S6Scan<float>.ParallelScan(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        var seqArr = seqOutput.ToArray();
        var parArr = parOutput.ToArray();

        Assert.Equal(seqArr.Length, parArr.Length);
        for (int i = 0; i < seqArr.Length; i++)
        {
            Assert.True(MathF.Abs(seqArr[i] - parArr[i]) < 0.01f,
                $"Mismatch at index {i}: sequential={seqArr[i]:G6}, parallel={parArr[i]:G6}");
        }
    }

    [Fact]
    public void SequentialScan_Double_ProducesValidOutput()
    {
        int batchSize = 1;
        int seqLen = 4;
        int innerDim = 8;
        int stateDim = 4;

        var x = CreateRandomDoubleTensor(new[] { batchSize, seqLen, innerDim });
        var delta = CreatePositiveDoubleTensor(new[] { batchSize, seqLen, innerDim });
        var aLog = CreateRandomDoubleTensor(new[] { innerDim, stateDim });
        var b = CreateRandomDoubleTensor(new[] { batchSize, seqLen, stateDim });
        var c = CreateRandomDoubleTensor(new[] { batchSize, seqLen, stateDim });
        var dParam = CreateRandomDoubleTensor(new[] { innerDim });

        var (output, _) = S6Scan<double>.SequentialScanForward(
            x, delta, aLog, b, c, dParam, batchSize, seqLen, innerDim, stateDim);

        Assert.Equal(new[] { batchSize, seqLen, innerDim }, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void MambaBlock_StillProducesValidOutput_AfterS6ScanRefactor()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim);
        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });

        var output = block.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));

        // Backward should also still work
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = block.Backward(grad);
        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1) * 0.5f;
        }
        return tensor;
    }

    private static Tensor<float> CreatePositiveTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 0.5 + 0.1);
        }
        return tensor;
    }

    private static Tensor<double> CreateRandomDoubleTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (random.NextDouble() * 2 - 1) * 0.5;
        }
        return tensor;
    }

    private static Tensor<double> CreatePositiveDoubleTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble() * 0.5 + 0.1;
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
