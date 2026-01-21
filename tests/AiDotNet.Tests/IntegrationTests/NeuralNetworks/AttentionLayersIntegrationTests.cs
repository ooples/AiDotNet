namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

/// <summary>
/// Integration tests for attention layer implementations testing any-rank tensor support,
/// forward/backward passes, multi-input scenarios, and cloning.
/// </summary>
public class AttentionLayersIntegrationTests
{
    #region AttentionLayer Tests

    [Fact]
    public void AttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [batch, features]
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AttentionLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seq, features]
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, 10, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([4, 8, inputSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void AttentionLayer_CrossAttention_ProducesValidOutput()
    {
        // Arrange - cross-attention with separate query and key/value inputs
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var query = CreateRandomTensor<float>([2, 8, inputSize]);
        var keyValue = CreateRandomTensor<float>([2, 12, inputSize]);

        // Act
        var output = layer.Forward(query, keyValue);

        // Assert
        Assert.Equal([2, 8, inputSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AttentionLayer_MaskedAttention_ProducesValidOutput()
    {
        // Arrange - attention with mask
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, 6, inputSize]);
        // Mask shape: [batch, queryLen, keyLen]
        var mask = CreateMaskTensor([2, 6, 6]);

        // Act
        var output = layer.Forward(input, mask);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var original = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([4, inputSize]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (AttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    #endregion

    #region SelfAttentionLayer Tests

    [Fact]
    public void SelfAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SelfAttentionLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 8, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SelfAttentionLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange - 4D input [batch1, batch2, seqLen, embedDim]
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, 3, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SelfAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void SelfAttentionLayer_MultiHeadConfiguration_Works()
    {
        // Arrange - embedDim must be divisible by headCount
        int seqLen = 16;
        int embedDim = 96; // 96 / 12 = 8
        int headCount = 12;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SelfAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var original = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (SelfAttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    #endregion

    #region MultiHeadAttentionLayer Tests

    [Fact]
    public void MultiHeadAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount: 8);
        var input = CreateRandomTensor<float>([seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiHeadAttentionLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount: 8);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiHeadAttentionLayer_ForwardPass_5D_ProducesValidOutput()
    {
        // Arrange - 5D input [batch1, batch2, batch3, seqLen, embedDim]
        int seqLen = 4;
        int embedDim = 32;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount: 4);
        var input = CreateRandomTensor<float>([2, 2, 2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiHeadAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount: 4);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void MultiHeadAttentionLayer_CrossAttention_ProducesValidOutput()
    {
        // Arrange - cross-attention with separate query and key/value
        int seqLen = 8;
        int embedDim = 32;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount: 4);
        var query = CreateRandomTensor<float>([2, seqLen, embedDim]);
        var keyValue = CreateRandomTensor<float>([2, 12, embedDim]);

        // Act
        var output = layer.Forward(query, keyValue);

        // Assert
        Assert.Equal([2, seqLen, embedDim], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiHeadAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var original = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount: 4);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (MultiHeadAttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    #endregion

    #region FlashAttentionLayer Tests

    [Fact]
    public void FlashAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [seqLen, embedDim]
        int seqLen = 8;
        int embedDim = 32;
        int headCount = 4;
        var layer = new FlashAttentionLayer<float>(seqLen, embedDim, headCount);
        var input = CreateRandomTensor<float>([seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FlashAttentionLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange - 4D input [batch1, batch2, seqLen, embedDim]
        int seqLen = 6;
        int embedDim = 24;
        int headCount = 4;
        var layer = new FlashAttentionLayer<float>(seqLen, embedDim, headCount);
        var input = CreateRandomTensor<float>([2, 3, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FlashAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        int headCount = 4;
        var layer = new FlashAttentionLayer<float>(seqLen, embedDim, headCount);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    #endregion

    #region CrossAttentionLayer Tests

    [Fact]
    public void CrossAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int queryDim = 64;
        int contextDim = 64;
        int seqLen = 16;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, headCount: 8, sequenceLength: seqLen);
        var query = CreateRandomTensor<float>([2, seqLen, queryDim]);
        var context = CreateRandomTensor<float>([2, seqLen, contextDim]);

        // Act
        var output = layer.Forward(query, context);

        // Assert
        Assert.Equal([2, seqLen, queryDim], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CrossAttentionLayer_ForwardPass_DifferentContextLength_ProducesValidOutput()
    {
        // Arrange - context can have different sequence length
        int queryDim = 32;
        int contextDim = 32;
        int querySeqLen = 8;
        int contextSeqLen = 16;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, headCount: 4, sequenceLength: querySeqLen);
        var query = CreateRandomTensor<float>([2, querySeqLen, queryDim]);
        var context = CreateRandomTensor<float>([2, contextSeqLen, contextDim]);

        // Act
        var output = layer.Forward(query, context);

        // Assert
        Assert.Equal([2, querySeqLen, queryDim], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CrossAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int dim = 32;
        int seqLen = 8;
        var layer = new CrossAttentionLayer<float>(dim, dim, headCount: 4, sequenceLength: seqLen);
        var query = CreateRandomTensor<float>([2, seqLen, dim]);
        var context = CreateRandomTensor<float>([2, seqLen, dim]);

        // Act
        var output = layer.Forward(query, context);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(query.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void CrossAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int dim = 32;
        int seqLen = 8;
        var original = new CrossAttentionLayer<float>(dim, dim, headCount: 4, sequenceLength: seqLen);
        var query = CreateRandomTensor<float>([2, seqLen, dim]);
        var context = CreateRandomTensor<float>([2, seqLen, dim]);
        var originalOutput = original.Forward(query, context);

        // Act
        var cloned = (CrossAttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(query, context);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    #endregion

    #region GraphAttentionLayer Tests

    [Fact]
    public void GraphAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - graph attention with node features and adjacency matrix
        int inputFeatures = 32;
        int outputFeatures = 16;
        int numNodes = 10;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 4);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.Equal([numNodes, outputFeatures], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GraphAttentionLayer_ForwardPass_BatchedInput_ProducesValidOutput()
    {
        // Arrange - batched graph input
        int inputFeatures = 32;
        int outputFeatures = 16;
        int batchSize = 2;
        int numNodes = 8;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2);
        var nodeFeatures = CreateRandomTensor<float>([batchSize, numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.Equal([batchSize, numNodes, outputFeatures], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GraphAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputFeatures = 16;
        int outputFeatures = 8;
        int numNodes = 6;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(nodeFeatures.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void GraphAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputFeatures = 16;
        int outputFeatures = 8;
        int numNodes = 6;
        var original = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        original.SetAdjacencyMatrix(adjacency);
        var originalOutput = original.Forward(nodeFeatures);

        // Act
        var cloned = (GraphAttentionLayer<float>)original.Clone();
        cloned.SetAdjacencyMatrix(adjacency);
        var clonedOutput = cloned.Forward(nodeFeatures);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void AttentionLayer_SingleBatch_Works()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([1, 4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SelfAttentionLayer_SingleHead_Works()
    {
        // Arrange - single attention head
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 1, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiHeadAttentionLayer_LargeHeadCount_Works()
    {
        // Arrange - many attention heads
        int seqLen = 8;
        int embedDim = 64;
        int headCount = 16; // 64 / 16 = 4 per head
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embedDim, headCount);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GraphAttentionLayer_SingleHead_Works()
    {
        // Arrange
        int inputFeatures = 16;
        int outputFeatures = 8;
        int numNodes = 4;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 1);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AttentionLayer_AuxiliaryLoss_Works()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(inputSize, attentionSize, (IActivationFunction<float>?)null);
        layer.UseAuxiliaryLoss = true;
        var input = CreateRandomTensor<float>([2, 8, inputSize]);

        // Act
        var output = layer.Forward(input);
        var auxLoss = layer.ComputeAuxiliaryLoss();

        // Assert
        Assert.False(ContainsNaN(output));
        Assert.False(float.IsNaN(auxLoss));
    }

    [Fact]
    public void SelfAttentionLayer_AuxiliaryLoss_Works()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        layer.UseAuxiliaryLoss = true;
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);
        var auxLoss = layer.ComputeAuxiliaryLoss();

        // Assert
        Assert.False(ContainsNaN(output));
        Assert.False(float.IsNaN(auxLoss));
    }

    #endregion

    #region Helper Methods

    private static Tensor<T> CreateRandomTensor<T>(int[] shape) where T : struct, IComparable<T>
    {
        var tensor = new Tensor<T>(shape);
        var random = new Random(42);
        var span = tensor.AsSpan();

        for (int i = 0; i < span.Length; i++)
        {
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            span[i] = (T)Convert.ChangeType(value, typeof(T));
        }

        return tensor;
    }

    private static Tensor<float> CreateMaskTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);

        // Create a causal mask (lower triangular)
        for (int b = 0; b < shape[0]; b++)
        {
            for (int i = 0; i < shape[1]; i++)
            {
                for (int j = 0; j < shape[2]; j++)
                {
                    // 0 for attended positions, -inf for masked positions
                    tensor[new int[] { b, i, j }] = j <= i ? 0f : float.NegativeInfinity;
                }
            }
        }

        return tensor;
    }

    private static Tensor<float> CreateRandomAdjacencyMatrix(int numNodes)
    {
        var tensor = new Tensor<float>([numNodes, numNodes]);
        var random = new Random(42);

        // Create a random adjacency matrix (sparse, symmetric)
        for (int i = 0; i < numNodes; i++)
        {
            tensor[new int[] { i, i }] = 1f; // Self-loops
            for (int j = i + 1; j < numNodes; j++)
            {
                float edge = random.NextDouble() > 0.5 ? 1f : 0f;
                tensor[new int[] { i, j }] = edge;
                tensor[new int[] { j, i }] = edge; // Symmetric
            }
        }

        return tensor;
    }

    private static bool ContainsNaN<T>(Tensor<T> tensor) where T : struct, IComparable<T>
    {
        foreach (var value in tensor.ToArray())
        {
            if (value is float f && float.IsNaN(f)) return true;
            if (value is double d && double.IsNaN(d)) return true;
        }
        return false;
    }

    #endregion
}
