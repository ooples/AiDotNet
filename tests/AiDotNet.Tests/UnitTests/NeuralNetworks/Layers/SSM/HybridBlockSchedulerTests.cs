using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="HybridBlockScheduler{T}"/>.
/// </summary>
public class HybridBlockSchedulerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesScheduler()
    {
        int seqLen = 8;
        int modelDim = 32;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 3);
        var isAttention = new bool[] { false, false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        Assert.Equal(modelDim, scheduler.ModelDimension);
        Assert.Equal(3, scheduler.NumBlocks);
        Assert.Equal(HybridSchedulePattern.JambaStyle, scheduler.SchedulePattern);
    }

    [Fact]
    public void Constructor_ThrowsOnEmptyBlocks()
    {
        Assert.Throws<ArgumentException>(() =>
            new HybridBlockScheduler<float>(
                8, Array.Empty<ILayer<float>>(), Array.Empty<bool>(),
                HybridSchedulePattern.JambaStyle, 32));
    }

    [Fact]
    public void Constructor_ThrowsOnMismatchedArrays()
    {
        var blocks = CreateMambaBlocks(8, 32, 3);
        var isAttention = new bool[] { false, false }; // Wrong length

        Assert.Throws<ArgumentException>(() =>
            new HybridBlockScheduler<float>(
                8, blocks, isAttention, HybridSchedulePattern.JambaStyle, 32));
    }

    [Fact]
    public void Constructor_ThrowsOnInvalidModelDimension()
    {
        var blocks = CreateMambaBlocks(8, 32, 2);
        var isAttention = new bool[] { false, false };

        Assert.Throws<ArgumentException>(() =>
            new HybridBlockScheduler<float>(
                8, blocks, isAttention, HybridSchedulePattern.JambaStyle, 0));
    }

    [Fact]
    public void Forward_3D_ProducesValidOutput()
    {
        int batchSize = 2;
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2, stateDim);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var input = CreateRandomTensor(new[] { batchSize, seqLen, modelDim });
        var output = scheduler.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_2D_ProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2, stateDim);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var input = CreateRandomTensor(new[] { seqLen, modelDim });
        var output = scheduler.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2, stateDim);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        var output = scheduler.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = scheduler.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        int seqLen = 4;
        int modelDim = 32;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var grad = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        Assert.Throws<InvalidOperationException>(() => scheduler.Backward(grad));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2, stateDim);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var params1 = scheduler.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(scheduler.ParameterCount, params1.Length);

        scheduler.SetParameters(params1);
        var params2 = scheduler.GetParameters();

        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SetParameters_ThrowsOnWrongLength()
    {
        var blocks = CreateMambaBlocks(4, 32, 2);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            4, blocks, isAttention, HybridSchedulePattern.JambaStyle, 32);

        Assert.Throws<ArgumentException>(() => scheduler.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var blocks = CreateMambaBlocks(4, 32, 2);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            4, blocks, isAttention, HybridSchedulePattern.JambaStyle, 32);

        Assert.True(scheduler.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var blocks = CreateMambaBlocks(4, 32, 3);
        var isAttention = new bool[] { false, true, false };

        var scheduler = new HybridBlockScheduler<float>(
            4, blocks, isAttention, HybridSchedulePattern.ZambaStyle, 32);

        var metadata = scheduler.GetMetadata();

        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("NumBlocks"));
        Assert.True(metadata.ContainsKey("SchedulePattern"));
        Assert.True(metadata.ContainsKey("AttentionBlocks"));
        Assert.True(metadata.ContainsKey("SSMBlocks"));
        Assert.Equal("32", metadata["ModelDimension"]);
        Assert.Equal("3", metadata["NumBlocks"]);
        Assert.Equal("ZambaStyle", metadata["SchedulePattern"]);
        Assert.Equal("1", metadata["AttentionBlocks"]);
        Assert.Equal("2", metadata["SSMBlocks"]);
    }

    [Fact]
    public void ResetState_AllowsReuse()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2, stateDim);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        var output1 = scheduler.Forward(input);
        scheduler.ResetState();

        var output2 = scheduler.Forward(input);
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

    [Theory]
    [InlineData(HybridSchedulePattern.JambaStyle)]
    [InlineData(HybridSchedulePattern.ZambaStyle)]
    [InlineData(HybridSchedulePattern.SambaStyle)]
    [InlineData(HybridSchedulePattern.Custom)]
    public void Forward_AllSchedulePatterns_Succeed(HybridSchedulePattern pattern)
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = CreateMambaBlocks(seqLen, modelDim, 2, stateDim);
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<float>(
            seqLen, blocks, isAttention, pattern, modelDim);

        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        var output = scheduler.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CreateJambaSchedule_CreatesCorrectPattern()
    {
        var scheduler = HybridBlockScheduler<float>.CreateJambaSchedule(
            sequenceLength: 8, modelDimension: 32, numLayers: 8,
            attentionFrequency: 4, stateDimension: 4, numAttentionHeads: 4);

        Assert.Equal(8, scheduler.NumBlocks);
        Assert.Equal(HybridSchedulePattern.JambaStyle, scheduler.SchedulePattern);

        var metadata = scheduler.GetMetadata();
        Assert.Equal("2", metadata["AttentionBlocks"]);  // Layers 4 and 8
        Assert.Equal("6", metadata["SSMBlocks"]);

        // Verify forward pass works
        var input = CreateRandomTensor(new[] { 1, 8, 32 });
        var output = scheduler.Forward(input);
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CreateZambaSchedule_InterleavesAttentionWithMamba()
    {
        var scheduler = HybridBlockScheduler<float>.CreateZambaSchedule(
            sequenceLength: 8, modelDimension: 32, numLayers: 6,
            attentionFrequency: 3, stateDimension: 4, numAttentionHeads: 4);

        Assert.Equal(6, scheduler.NumBlocks);
        Assert.Equal(HybridSchedulePattern.ZambaStyle, scheduler.SchedulePattern);

        var metadata = scheduler.GetMetadata();
        Assert.Equal("2", metadata["AttentionBlocks"]);  // Layers 3 and 6
        Assert.Equal("4", metadata["SSMBlocks"]);

        // Verify forward pass works
        var input = CreateRandomTensor(new[] { 1, 8, 32 });
        var output = scheduler.Forward(input);
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CreateSambaSchedule_AlternatesMambaAndAttention()
    {
        var scheduler = HybridBlockScheduler<float>.CreateSambaSchedule(
            sequenceLength: 8, modelDimension: 32, numLayers: 4,
            stateDimension: 4, numAttentionHeads: 4);

        Assert.Equal(4, scheduler.NumBlocks);
        Assert.Equal(HybridSchedulePattern.SambaStyle, scheduler.SchedulePattern);

        var metadata = scheduler.GetMetadata();
        Assert.Equal("2", metadata["AttentionBlocks"]);  // Odd positions: 1, 3
        Assert.Equal("2", metadata["SSMBlocks"]);         // Even positions: 0, 2

        // Verify forward pass works
        var input = CreateRandomTensor(new[] { 1, 8, 32 });
        var output = scheduler.Forward(input);
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CreateJambaSchedule_ThrowsOnInvalidParameters()
    {
        Assert.Throws<ArgumentException>(() =>
            HybridBlockScheduler<float>.CreateJambaSchedule(8, 32, numLayers: 0));

        Assert.Throws<ArgumentException>(() =>
            HybridBlockScheduler<float>.CreateJambaSchedule(8, 32, 4, attentionFrequency: 0));
    }

    [Fact]
    public void CreateJambaSchedule_FullTrainingStep()
    {
        var scheduler = HybridBlockScheduler<float>.CreateJambaSchedule(
            sequenceLength: 4, modelDimension: 16, numLayers: 4,
            attentionFrequency: 2, stateDimension: 4, numAttentionHeads: 4);

        var input = CreateRandomTensor(new[] { 1, 4, 16 });
        var output = scheduler.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        scheduler.Backward(grad);
        scheduler.UpdateParameters(0.001f);

        scheduler.ResetState();
        var output2 = scheduler.Forward(input);
        Assert.Equal(output.Shape, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var blocks = new ILayer<double>[]
        {
            new MambaBlock<double>(seqLen, modelDim, stateDim),
            new MambaBlock<double>(seqLen, modelDim, stateDim)
        };
        var isAttention = new bool[] { false, false };

        var scheduler = new HybridBlockScheduler<double>(
            seqLen, blocks, isAttention, HybridSchedulePattern.JambaStyle, modelDim);

        var input = CreateRandomDoubleTensor(new[] { 1, seqLen, modelDim });
        var output = scheduler.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    #region Helpers

    private static ILayer<float>[] CreateMambaBlocks(int seqLen, int modelDim, int count, int stateDim = 8)
    {
        var blocks = new ILayer<float>[count];
        for (int i = 0; i < count; i++)
        {
            blocks[i] = new MambaBlock<float>(seqLen, modelDim, stateDim);
        }
        return blocks;
    }

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
