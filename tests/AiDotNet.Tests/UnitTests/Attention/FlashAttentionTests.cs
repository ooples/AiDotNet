using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Attention;

/// <summary>
/// Unit tests for Flash Attention implementation.
/// </summary>
public class FlashAttentionTests
{
    private const double Tolerance = 1e-4;

    [Fact]
    public void FlashAttention_Forward_ProducesCorrectShape()
    {
        // Arrange
        int batchSize = 2;
        int seqLen = 8;
        int headDim = 16;

        var query = CreateRandomTensor(batchSize, seqLen, headDim);
        var key = CreateRandomTensor(batchSize, seqLen, headDim);
        var value = CreateRandomTensor(batchSize, seqLen, headDim);

        // Act
        var (output, _) = FlashAttention<float>.Forward(query, key, value);

        // Assert
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLen, output.Shape[1]);
        Assert.Equal(headDim, output.Shape[2]);
    }

    [Fact]
    public void FlashAttention_Forward4D_ProducesCorrectShape()
    {
        // Arrange
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int headDim = 16;

        var query = CreateRandomTensor(batchSize, numHeads, seqLen, headDim);
        var key = CreateRandomTensor(batchSize, numHeads, seqLen, headDim);
        var value = CreateRandomTensor(batchSize, numHeads, seqLen, headDim);

        // Act
        var (output, _) = FlashAttention<float>.Forward(query, key, value);

        // Assert
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(numHeads, output.Shape[1]);
        Assert.Equal(seqLen, output.Shape[2]);
        Assert.Equal(headDim, output.Shape[3]);
    }

    [Fact]
    public void FlashAttention_MatchesStandardAttention_3D()
    {
        // Arrange
        int batchSize = 1;
        int seqLen = 4;
        int headDim = 8;

        var query = CreateRandomTensor(batchSize, seqLen, headDim, seed: 42);
        var key = CreateRandomTensor(batchSize, seqLen, headDim, seed: 43);
        var value = CreateRandomTensor(batchSize, seqLen, headDim, seed: 44);

        // Act - Flash Attention
        var (flashOutput, _) = FlashAttention<float>.Forward(query, key, value);

        // Act - Standard Attention (for comparison)
        var standardOutput = ComputeStandardAttention(query, key, value);

        // Assert - Results should match within tolerance
        AssertTensorsEqual(flashOutput, standardOutput, Tolerance);
    }

    [Fact]
    public void FlashAttention_WithCausalMask_MasksCorrectly()
    {
        // Arrange
        int batchSize = 1;
        int seqLen = 4;
        int headDim = 8;

        var query = CreateRandomTensor(batchSize, seqLen, headDim, seed: 42);
        var key = CreateRandomTensor(batchSize, seqLen, headDim, seed: 43);
        var value = CreateRandomTensor(batchSize, seqLen, headDim, seed: 44);

        var config = FlashAttentionConfig.Causal;

        // Act
        var (output, attnWeights) = FlashAttention<float>.Forward(
            query, key, value,
            new FlashAttentionConfig { UseCausalMask = true, ReturnAttentionWeights = true });

        // Assert - Attention weights above diagonal should be zero (masked)
        Assert.NotNull(attnWeights);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = i + 1; j < seqLen; j++)
                {
                    float weight = attnWeights[new[] { b, i, j }];
                    Assert.True(weight < 1e-6f, $"Position ({i}, {j}) should be masked but has weight {weight}");
                }
            }
        }
    }

    [Fact]
    public void FlashAttention_AttentionWeightsRowSumToOne()
    {
        // Arrange
        int batchSize = 1;
        int seqLen = 4;
        int headDim = 8;

        var query = CreateRandomTensor(batchSize, seqLen, headDim, seed: 42);
        var key = CreateRandomTensor(batchSize, seqLen, headDim, seed: 43);
        var value = CreateRandomTensor(batchSize, seqLen, headDim, seed: 44);

        var config = new FlashAttentionConfig { ReturnAttentionWeights = true };

        // Act
        var (_, attnWeights) = FlashAttention<float>.Forward(query, key, value, config);

        // Assert - Each row should sum to 1 (softmax property)
        Assert.NotNull(attnWeights);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                float rowSum = 0;
                for (int j = 0; j < seqLen; j++)
                {
                    rowSum += attnWeights[new[] { b, i, j }];
                }
                Assert.True(Math.Abs(rowSum - 1.0f) < 0.01f, $"Row {i} sums to {rowSum}, expected 1.0");
            }
        }
    }

    [Fact]
    public void FlashAttention_Backward_ProducesCorrectGradientShapes()
    {
        // Arrange
        int batchSize = 2;
        int seqLen = 4;
        int headDim = 8;

        var query = CreateRandomTensor(batchSize, seqLen, headDim, seed: 42);
        var key = CreateRandomTensor(batchSize, seqLen, headDim, seed: 43);
        var value = CreateRandomTensor(batchSize, seqLen, headDim, seed: 44);
        var gradOutput = CreateRandomTensor(batchSize, seqLen, headDim, seed: 45);

        // Forward pass
        var (output, _) = FlashAttention<float>.Forward(query, key, value);

        // Act - Backward pass
        var (gradQuery, gradKey, gradValue) = FlashAttention<float>.Backward(
            gradOutput, query, key, value, output);

        // Assert - Gradient shapes match input shapes
        Assert.Equal(query.Shape, gradQuery.Shape);
        Assert.Equal(key.Shape, gradKey.Shape);
        Assert.Equal(value.Shape, gradValue.Shape);
    }

    [Fact]
    public void FlashAttention_DifferentBlockSizes_ProduceSameResult()
    {
        // Arrange
        int batchSize = 1;
        int seqLen = 16;
        int headDim = 8;

        var query = CreateRandomTensor(batchSize, seqLen, headDim, seed: 42);
        var key = CreateRandomTensor(batchSize, seqLen, headDim, seed: 43);
        var value = CreateRandomTensor(batchSize, seqLen, headDim, seed: 44);

        var config1 = new FlashAttentionConfig { BlockSizeQ = 4, BlockSizeKV = 4 };
        var config2 = new FlashAttentionConfig { BlockSizeQ = 8, BlockSizeKV = 8 };
        var config3 = new FlashAttentionConfig { BlockSizeQ = 16, BlockSizeKV = 16 };

        // Act
        var (output1, _) = FlashAttention<float>.Forward(query, key, value, config1);
        var (output2, _) = FlashAttention<float>.Forward(query, key, value, config2);
        var (output3, _) = FlashAttention<float>.Forward(query, key, value, config3);

        // Assert - All should produce same result
        AssertTensorsEqual(output1, output2, Tolerance);
        AssertTensorsEqual(output2, output3, Tolerance);
    }

    [Fact]
    public void FlashAttentionLayer_Forward_ProducesCorrectShape()
    {
        // Arrange
        int seqLen = 8;
        int embDim = 64;
        int numHeads = 4;
        int batchSize = 2;

        var layer = new FlashAttentionLayer<float>(seqLen, embDim, numHeads);
        var input = CreateRandomTensor(batchSize, seqLen, embDim);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLen, output.Shape[1]);
        Assert.Equal(embDim, output.Shape[2]);
    }

    [Fact]
    public void FlashAttentionLayer_Backward_ProducesCorrectShape()
    {
        // Arrange
        int seqLen = 8;
        int embDim = 64;
        int numHeads = 4;
        int batchSize = 2;

        var layer = new FlashAttentionLayer<float>(seqLen, embDim, numHeads);
        var input = CreateRandomTensor(batchSize, seqLen, embDim, seed: 42);
        var gradOutput = CreateRandomTensor(batchSize, seqLen, embDim, seed: 43);

        // Forward pass
        layer.Forward(input);

        // Act - Backward pass
        var gradInput = layer.Backward(gradOutput);

        // Assert
        Assert.Equal(input.Shape, gradInput.Shape);
    }

    [Fact]
    public void FlashAttentionLayer_UpdateParameters_ChangesWeights()
    {
        // Arrange
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 2;
        int batchSize = 1;

        var layer = new FlashAttentionLayer<float>(seqLen, embDim, numHeads);
        var input = CreateRandomTensor(batchSize, seqLen, embDim, seed: 42);
        var gradOutput = CreateRandomTensor(batchSize, seqLen, embDim, seed: 43);

        var paramsBefore = layer.GetParameters().ToArray();

        // Forward and backward
        layer.Forward(input);
        layer.Backward(gradOutput);

        // Act
        layer.UpdateParameters(0.01f);
        var paramsAfter = layer.GetParameters().ToArray();

        // Assert - Parameters should have changed
        bool anyChanged = false;
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-10)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged, "Parameters should change after update");
    }

    [Fact]
    public void FlashAttentionLayer_GetSetParameters_RoundTrip()
    {
        // Arrange
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 2;

        var layer1 = new FlashAttentionLayer<float>(seqLen, embDim, numHeads);
        var layer2 = new FlashAttentionLayer<float>(seqLen, embDim, numHeads);

        // Act
        var params1 = layer1.GetParameters();
        layer2.SetParameters(params1);
        var params2 = layer2.GetParameters();

        // Assert
        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i], 6);
        }
    }

    [Fact]
    public void FlashAttentionConfig_Presets_HaveExpectedValues()
    {
        // Act
        var defaultConfig = FlashAttentionConfig.Default;
        var causalConfig = FlashAttentionConfig.Causal;
        var memoryEfficientConfig = FlashAttentionConfig.MemoryEfficient;
        var highPerfConfig = FlashAttentionConfig.HighPerformance;

        // Assert - Default
        Assert.Equal(64, defaultConfig.BlockSizeQ);
        Assert.False(defaultConfig.UseCausalMask);

        // Assert - Causal
        Assert.True(causalConfig.UseCausalMask);

        // Assert - Memory Efficient
        Assert.Equal(32, memoryEfficientConfig.BlockSizeQ);
        Assert.True(memoryEfficientConfig.RecomputeInBackward);

        // Assert - High Performance
        Assert.Equal(128, highPerfConfig.BlockSizeQ);
        Assert.True(highPerfConfig.UseGpuKernel);
    }

    #region Helper Methods

    private static Tensor<float> CreateRandomTensor(params int[] shape)
    {
        return CreateRandomTensor(shape, seed: null);
    }

    private static Tensor<float> CreateRandomTensor(int dim1, int dim2, int dim3, int? seed = null)
    {
        return CreateRandomTensor(new[] { dim1, dim2, dim3 }, seed);
    }

    private static Tensor<float> CreateRandomTensor(int dim1, int dim2, int dim3, int dim4, int? seed = null)
    {
        return CreateRandomTensor(new[] { dim1, dim2, dim3, dim4 }, seed);
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, int? seed)
    {
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var tensor = new Tensor<float>(shape);

        int totalElements = 1;
        foreach (var dim in shape) totalElements *= dim;

        for (int i = 0; i < totalElements; i++)
        {
            tensor.SetFlat(i, (float)(random.NextDouble() * 2 - 1));
        }

        return tensor;
    }

    private static void AssertTensorsEqual(Tensor<float> expected, Tensor<float> actual, double tolerance)
    {
        Assert.Equal(expected.Shape.Length, actual.Shape.Length);
        for (int i = 0; i < expected.Shape.Length; i++)
        {
            Assert.Equal(expected.Shape[i], actual.Shape[i]);
        }

        int totalElements = 1;
        foreach (var dim in expected.Shape) totalElements *= dim;

        for (int i = 0; i < totalElements; i++)
        {
            Assert.True(
                Math.Abs(expected.GetFlat(i) - actual.GetFlat(i)) < tolerance,
                $"Tensors differ at index {i}: expected {expected.GetFlat(i)}, actual {actual.GetFlat(i)}");
        }
    }

    /// <summary>
    /// Computes standard attention for comparison: softmax(Q @ K^T / sqrt(d)) @ V
    /// </summary>
    private static Tensor<float> ComputeStandardAttention(Tensor<float> query, Tensor<float> key, Tensor<float> value)
    {
        int batchSize = query.Shape[0];
        int seqLenQ = query.Shape[1];
        int seqLenKV = key.Shape[1];
        int headDim = query.Shape[2];

        float scale = 1.0f / (float)Math.Sqrt(headDim);
        var output = new Tensor<float>(query.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Compute attention scores: Q @ K^T
            var scores = new float[seqLenQ, seqLenKV];
            for (int i = 0; i < seqLenQ; i++)
            {
                for (int j = 0; j < seqLenKV; j++)
                {
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += query[new[] { b, i, d }] * key[new[] { b, j, d }];
                    }
                    scores[i, j] = dot * scale;
                }
            }

            // Apply softmax row-wise
            var attnWeights = new float[seqLenQ, seqLenKV];
            for (int i = 0; i < seqLenQ; i++)
            {
                // Find max for numerical stability
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < seqLenKV; j++)
                {
                    if (scores[i, j] > maxScore) maxScore = scores[i, j];
                }

                // Compute exp and sum
                float sumExp = 0;
                for (int j = 0; j < seqLenKV; j++)
                {
                    attnWeights[i, j] = (float)Math.Exp(scores[i, j] - maxScore);
                    sumExp += attnWeights[i, j];
                }

                // Normalize
                for (int j = 0; j < seqLenKV; j++)
                {
                    attnWeights[i, j] /= sumExp;
                }
            }

            // Compute output: attnWeights @ V
            for (int i = 0; i < seqLenQ; i++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        sum += attnWeights[i, j] * value[new[] { b, j, d }];
                    }
                    output[new[] { b, i, d }] = sum;
                }
            }
        }

        return output;
    }

    #endregion
}
