using System;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Tests for transformer layer shape compatibility - ensuring layers work with both 2D and 3D tensors.
/// </summary>
public class TransformerShapeCompatibilityTests
{
    #region PositionalEncodingLayer Tests

    [Fact]
    public void PositionalEncodingLayer_Forward_With2DInput_ReturnsCorrectShape()
    {
        // Arrange
        int maxSeqLength = 100;
        int embeddingSize = 64;
        var layer = new PositionalEncodingLayer<double>(maxSeqLength, embeddingSize);

        // 2D input: [seq_length, embedding_size]
        var input = new Tensor<double>([10, embeddingSize]);
        InitializeTensor(input, 0.1);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(10, output.Shape[0]); // seq_length preserved
        Assert.Equal(embeddingSize, output.Shape[1]); // embedding_size preserved
    }

    [Fact]
    public void PositionalEncodingLayer_Forward_With3DInput_ReturnsCorrectShape()
    {
        // Arrange
        int maxSeqLength = 100;
        int embeddingSize = 64;
        var layer = new PositionalEncodingLayer<double>(maxSeqLength, embeddingSize);

        // 3D input: [batch_size, seq_length, embedding_size]
        int batchSize = 4;
        int seqLength = 20;
        var input = new Tensor<double>([batchSize, seqLength, embeddingSize]);
        InitializeTensor(input, 0.1);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLength, output.Shape[1]);
        Assert.Equal(embeddingSize, output.Shape[2]);
    }

    [Fact]
    public void PositionalEncodingLayer_Forward_With3DInput_AddsSameEncodingsPerBatch()
    {
        // Arrange
        int maxSeqLength = 100;
        int embeddingSize = 8;
        var layer = new PositionalEncodingLayer<double>(maxSeqLength, embeddingSize);

        // Create input with zeros to see raw positional encodings
        int batchSize = 2;
        int seqLength = 5;
        var input = new Tensor<double>([batchSize, seqLength, embeddingSize]);
        // Leave as zeros

        // Act
        var output = layer.Forward(input);

        // Assert - encodings should be identical across batch dimension
        for (int s = 0; s < seqLength; s++)
        {
            for (int e = 0; e < embeddingSize; e++)
            {
                double firstBatchValue = output[0, s, e];
                double secondBatchValue = output[1, s, e];
                Assert.Equal(firstBatchValue, secondBatchValue, precision: 6);
            }
        }
    }

    #endregion

    #region AttentionLayer Tests

    [Fact]
    public void AttentionLayer_Forward_With2DInput_ReturnsCorrectShape()
    {
        // Arrange
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<double>(inputSize, attentionSize, (IActivationFunction<double>?)null);

        // 2D input: [batch_size, input_size]
        int batchSize = 4;
        var input = new Tensor<double>([batchSize, inputSize]);
        InitializeTensor(input, 0.1);

        // Act
        var output = layer.Forward(input);

        // Assert - should return 2D since input was 2D
        // Output size = inputSize (industry-standard: Wo projects back to input dimension)
        Assert.Equal(2, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(inputSize, output.Shape[1]); // Output projects back to inputSize via Wo
    }

    [Fact]
    public void AttentionLayer_Forward_With3DInput_ReturnsCorrectShape()
    {
        // Arrange
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<double>(inputSize, attentionSize, (IActivationFunction<double>?)null);

        // 3D input: [batch_size, seq_length, input_size]
        int batchSize = 4;
        int seqLength = 10;
        var input = new Tensor<double>([batchSize, seqLength, inputSize]);
        InitializeTensor(input, 0.1);

        // Act
        var output = layer.Forward(input);

        // Assert - should return 3D since input was 3D
        // Output size = inputSize (industry-standard: Wo projects back to input dimension)
        Assert.Equal(3, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLength, output.Shape[1]);
        Assert.Equal(inputSize, output.Shape[2]); // Output projects back to inputSize via Wo
    }

    #endregion

    #region TransformerEncoderLayer Tests

    [Fact]
    public void TransformerEncoderLayer_Forward_With2DInput_ReturnsCorrectShape()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerEncoderLayer<double>(embeddingSize, numHeads, feedForwardDim);

        // 2D input: [seq_length, embedding_size]
        int seqLength = 10;
        var input = new Tensor<double>([seqLength, embeddingSize]);
        InitializeTensor(input, 0.1);

        // Act
        var output = layer.Forward(input);

        // Assert - should return 2D since input was 2D
        Assert.Equal(2, output.Rank);
        Assert.Equal(seqLength, output.Shape[0]);
        Assert.Equal(embeddingSize, output.Shape[1]);
    }

    [Fact]
    public void TransformerEncoderLayer_Forward_With3DInput_ReturnsCorrectShape()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerEncoderLayer<double>(embeddingSize, numHeads, feedForwardDim);

        // 3D input: [batch_size, seq_length, embedding_size]
        int batchSize = 2;
        int seqLength = 10;
        var input = new Tensor<double>([batchSize, seqLength, embeddingSize]);
        InitializeTensor(input, 0.1);

        // Act
        var output = layer.Forward(input);

        // Assert - should return 3D since input was 3D
        Assert.Equal(3, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLength, output.Shape[1]);
        Assert.Equal(embeddingSize, output.Shape[2]);
    }

    #endregion

    #region Transformer Pipeline Integration Tests

    [Fact]
    public void TransformerPipeline_With3DInput_AllLayersProduceCorrectShapes()
    {
        // Arrange - Create a simple transformer pipeline
        int batchSize = 2;
        int seqLength = 8;
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;

        var positionalEncoding = new PositionalEncodingLayer<double>(maxSequenceLength: 100, embeddingSize);
        var encoderLayer = new TransformerEncoderLayer<double>(embeddingSize, numHeads, feedForwardDim);

        // Create input: [batch_size, seq_length, embedding_size]
        var input = new Tensor<double>([batchSize, seqLength, embeddingSize]);
        InitializeTensor(input, 0.1);

        // Act
        var withPositions = positionalEncoding.Forward(input);
        var encoded = encoderLayer.Forward(withPositions);

        // Assert
        Assert.Equal(3, withPositions.Rank);
        Assert.Equal(batchSize, withPositions.Shape[0]);
        Assert.Equal(seqLength, withPositions.Shape[1]);
        Assert.Equal(embeddingSize, withPositions.Shape[2]);

        Assert.Equal(3, encoded.Rank);
        Assert.Equal(batchSize, encoded.Shape[0]);
        Assert.Equal(seqLength, encoded.Shape[1]);
        Assert.Equal(embeddingSize, encoded.Shape[2]);
    }

    [Fact]
    public void DecoderLayer_Forward_With3DInput_ReturnsCorrectShape()
    {
        // Arrange
        // AttentionLayer now has output projection (Wo) per industry-standard,
        // so attentionSize can differ from inputSize while still supporting residual connections
        int inputSize = 64;
        int attentionSize = 32; // Can now differ from inputSize thanks to Wo projection
        int feedForwardSize = 128;
        var layer = new DecoderLayer<double>(inputSize, attentionSize, feedForwardSize, (IActivationFunction<double>?)null);

        // 3D input: [batch_size, seq_length, input_size]
        int batchSize = 2;
        int seqLength = 8;
        var decoderInput = new Tensor<double>([batchSize, seqLength, inputSize]);
        var encoderOutput = new Tensor<double>([batchSize, seqLength, inputSize]);
        InitializeTensor(decoderInput, 0.1);
        InitializeTensor(encoderOutput, 0.2);

        // Act
        var output = layer.Forward(decoderInput, encoderOutput);

        // Assert - should return 3D since input was 3D
        // Output size = inputSize (attention output projects back via Wo)
        Assert.Equal(3, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLength, output.Shape[1]);
        Assert.Equal(inputSize, output.Shape[2]); // Output projects back to inputSize
    }

    #endregion

    #region Higher-Rank Tensor Tests (4D, 5D)

    [Fact]
    public void TensorMatMul_4Dx2D_ReturnsCorrectShape()
    {
        // Arrange - 4D tensor: [batch, heads, seq, dim] @ [dim, out_dim] = [batch, heads, seq, out_dim]
        // Common in multi-head attention: Q @ Wq
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 64;
        int outDim = 32;

        var input = new Tensor<double>([batchSize, numHeads, seqLen, dim]);
        var weights = new Tensor<double>([dim, outDim]);
        InitializeTensor(input, 0.1);
        InitializeTensor(weights, 0.01);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.TensorMatMul(input, weights);

        // Assert
        Assert.Equal(4, result.Rank);
        Assert.Equal(batchSize, result.Shape[0]);
        Assert.Equal(numHeads, result.Shape[1]);
        Assert.Equal(seqLen, result.Shape[2]);
        Assert.Equal(outDim, result.Shape[3]);
    }

    [Fact]
    public void TensorMatMul_4Dx4D_ReturnsCorrectShape()
    {
        // Arrange - 4D x 4D for batched attention scores: Q @ K^T
        // [batch, heads, seq, dim] @ [batch, heads, dim, seq] = [batch, heads, seq, seq]
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 64;

        var Q = new Tensor<double>([batchSize, numHeads, seqLen, dim]);
        var K_transposed = new Tensor<double>([batchSize, numHeads, dim, seqLen]);
        InitializeTensor(Q, 0.1);
        InitializeTensor(K_transposed, 0.01);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.TensorMatMul(Q, K_transposed);

        // Assert
        Assert.Equal(4, result.Rank);
        Assert.Equal(batchSize, result.Shape[0]);
        Assert.Equal(numHeads, result.Shape[1]);
        Assert.Equal(seqLen, result.Shape[2]);
        Assert.Equal(seqLen, result.Shape[3]); // Attention scores: seq x seq
    }

    [Fact]
    public void TensorMatMul_5Dx2D_ReturnsCorrectShape()
    {
        // Arrange - 5D tensor: [outer_batch, inner_batch, heads, seq, dim] @ [dim, out]
        // For multi-level batching (e.g., documents -> sentences -> tokens)
        int outerBatch = 2;
        int innerBatch = 3;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 32;
        int outDim = 16;

        var input = new Tensor<double>([outerBatch, innerBatch, numHeads, seqLen, dim]);
        var weights = new Tensor<double>([dim, outDim]);
        InitializeTensor(input, 0.1);
        InitializeTensor(weights, 0.01);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.TensorMatMul(input, weights);

        // Assert
        Assert.Equal(5, result.Rank);
        Assert.Equal(outerBatch, result.Shape[0]);
        Assert.Equal(innerBatch, result.Shape[1]);
        Assert.Equal(numHeads, result.Shape[2]);
        Assert.Equal(seqLen, result.Shape[3]);
        Assert.Equal(outDim, result.Shape[4]);
    }

    [Fact]
    public void TensorMatMul_5Dx5D_ReturnsCorrectShape()
    {
        // Arrange - Full 5D batched matmul: attention pattern with extra batch dimension
        int outerBatch = 2;
        int innerBatch = 3;
        int numHeads = 2;
        int seqLen = 4;
        int dim = 16;

        var A = new Tensor<double>([outerBatch, innerBatch, numHeads, seqLen, dim]);
        var B = new Tensor<double>([outerBatch, innerBatch, numHeads, dim, seqLen]);
        InitializeTensor(A, 0.1);
        InitializeTensor(B, 0.01);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.TensorMatMul(A, B);

        // Assert
        Assert.Equal(5, result.Rank);
        Assert.Equal(outerBatch, result.Shape[0]);
        Assert.Equal(innerBatch, result.Shape[1]);
        Assert.Equal(numHeads, result.Shape[2]);
        Assert.Equal(seqLen, result.Shape[3]);
        Assert.Equal(seqLen, result.Shape[4]); // seqLen x seqLen attention pattern
    }

    [Fact]
    public void LayerNorm_4DTensor_ReturnsCorrectShape()
    {
        // Arrange - 4D tensor: [batch, heads, seq, dim]
        // LayerNorm normalizes over last N dims defined by gamma shape
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 64;

        var input = new Tensor<double>([batchSize, numHeads, seqLen, dim]);
        var gamma = new Tensor<double>([dim]); // Normalize over last dim only
        var beta = new Tensor<double>([dim]);
        InitializeTensor(input, 0.5);
        InitializeTensor(gamma, 1.0); // Scale = 1
        InitializeTensor(beta, 0.0);  // Shift = 0

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.LayerNorm(input, gamma, beta, 1e-5, out var mean, out var variance);

        // Assert
        Assert.Equal(4, result.Rank);
        Assert.Equal(batchSize, result.Shape[0]);
        Assert.Equal(numHeads, result.Shape[1]);
        Assert.Equal(seqLen, result.Shape[2]);
        Assert.Equal(dim, result.Shape[3]);

        // Mean/variance have batch shape (all dims except normalized dim)
        Assert.Equal(batchSize * numHeads * seqLen, mean.Length);
        Assert.Equal(batchSize * numHeads * seqLen, variance.Length);
    }

    [Fact]
    public void LayerNorm_4DTensor_MultiDimNormalization_ReturnsCorrectShape()
    {
        // Arrange - Normalize over last 2 dimensions (seq, dim)
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 64;

        var input = new Tensor<double>([batchSize, numHeads, seqLen, dim]);
        var gamma = new Tensor<double>([seqLen, dim]); // Normalize over last 2 dims
        var beta = new Tensor<double>([seqLen, dim]);
        InitializeTensor(input, 0.5);
        InitializeTensor(gamma, 1.0);
        InitializeTensor(beta, 0.0);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.LayerNorm(input, gamma, beta, 1e-5, out var mean, out var variance);

        // Assert
        Assert.Equal(4, result.Rank);
        Assert.Equal(batchSize, result.Shape[0]);
        Assert.Equal(numHeads, result.Shape[1]);
        Assert.Equal(seqLen, result.Shape[2]);
        Assert.Equal(dim, result.Shape[3]);

        // Mean/variance have batch shape (batch, heads) = product of non-normalized dims
        Assert.Equal(batchSize * numHeads, mean.Length);
    }

    [Fact]
    public void LayerNorm_5DTensor_ReturnsCorrectShape()
    {
        // Arrange - 5D tensor: [outer, inner, heads, seq, dim]
        int outerBatch = 2;
        int innerBatch = 3;
        int numHeads = 2;
        int seqLen = 4;
        int dim = 32;

        var input = new Tensor<double>([outerBatch, innerBatch, numHeads, seqLen, dim]);
        var gamma = new Tensor<double>([dim]);
        var beta = new Tensor<double>([dim]);
        InitializeTensor(input, 0.5);
        InitializeTensor(gamma, 1.0);
        InitializeTensor(beta, 0.0);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.LayerNorm(input, gamma, beta, 1e-5, out var mean, out var variance);

        // Assert
        Assert.Equal(5, result.Rank);
        Assert.Equal(outerBatch, result.Shape[0]);
        Assert.Equal(innerBatch, result.Shape[1]);
        Assert.Equal(numHeads, result.Shape[2]);
        Assert.Equal(seqLen, result.Shape[3]);
        Assert.Equal(dim, result.Shape[4]);

        // Mean/variance have batch size = product of all dims except last
        Assert.Equal(outerBatch * innerBatch * numHeads * seqLen, mean.Length);
    }

    [Fact]
    public void LayerNorm_4DTensor_OutputIsNormalized()
    {
        // Arrange - Verify that output is actually normalized (mean ≈ 0, var ≈ 1)
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 64;

        var input = new Tensor<double>([batchSize, numHeads, seqLen, dim]);
        var gamma = new Tensor<double>([dim]);
        var beta = new Tensor<double>([dim]);

        // Initialize with non-zero, non-uniform values
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = random.NextDouble() * 10.0 - 5.0; // Values between -5 and 5
        }
        InitializeTensor(gamma, 1.0); // Scale = 1 (preserves variance = 1)
        InitializeTensor(beta, 0.0);  // Shift = 0 (preserves mean = 0)

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.LayerNorm(input, gamma, beta, 1e-5, out _, out _);

        // Assert - Check that output is normalized per feature group
        // For each batch position, mean should be ≈ 0 and std should be ≈ 1
        int totalBatchPositions = batchSize * numHeads * seqLen;
        for (int b = 0; b < Math.Min(totalBatchPositions, 10); b++) // Check first 10 batch positions
        {
            double sum = 0;
            double sumSq = 0;
            for (int f = 0; f < dim; f++)
            {
                double val = result.ToArray()[b * dim + f];
                sum += val;
                sumSq += val * val;
            }
            double mean = sum / dim;
            double variance = sumSq / dim - mean * mean;

            Assert.True(Math.Abs(mean) < 1e-5, $"Mean at batch {b} should be ≈ 0, got {mean}");
            Assert.True(Math.Abs(variance - 1.0) < 1e-4, $"Variance at batch {b} should be ≈ 1, got {variance}");
        }
    }

    [Fact]
    public void BatchMatMul_4DTensors_AttentionPattern_ReturnsCorrectShape()
    {
        // Arrange - BatchMatMul for attention: attention_weights @ V
        // [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, dim] = [batch, heads, seq_q, dim]
        // BatchMatMul now supports any-rank tensors (industry standard)
        int batchSize = 2;
        int numHeads = 4;
        int seqLen = 8;
        int dim = 64;

        var attentionWeights = new Tensor<double>([batchSize, numHeads, seqLen, seqLen]);
        var V = new Tensor<double>([batchSize, numHeads, seqLen, dim]);
        InitializeTensor(attentionWeights, 0.1);
        InitializeTensor(V, 0.1);

        // Act - Use BatchMatMul for 4D tensors (industry-standard any-rank support)
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.BatchMatMul(attentionWeights, V);

        // Assert
        Assert.Equal(4, result.Rank);
        Assert.Equal(batchSize, result.Shape[0]);
        Assert.Equal(numHeads, result.Shape[1]);
        Assert.Equal(seqLen, result.Shape[2]);
        Assert.Equal(dim, result.Shape[3]);
    }

    [Fact]
    public void BatchMatMul_5DTensors_ReturnsCorrectShape()
    {
        // Arrange - 5D BatchMatMul for video or other high-dimensional data
        // [batch, time, heads, seq, features] @ [batch, time, heads, features, out] = [batch, time, heads, seq, out]
        int batch = 2;
        int time = 3;
        int heads = 4;
        int seq = 8;
        int features = 64;
        int outDim = 32;

        var A = new Tensor<double>([batch, time, heads, seq, features]);
        var B = new Tensor<double>([batch, time, heads, features, outDim]);
        InitializeTensor(A, 0.1);
        InitializeTensor(B, 0.1);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.BatchMatMul(A, B);

        // Assert
        Assert.Equal(5, result.Rank);
        Assert.Equal(batch, result.Shape[0]);
        Assert.Equal(time, result.Shape[1]);
        Assert.Equal(heads, result.Shape[2]);
        Assert.Equal(seq, result.Shape[3]);
        Assert.Equal(outDim, result.Shape[4]);
    }

    [Fact]
    public void BatchMatMul_2DTensors_ReturnsCorrectShape()
    {
        // Arrange - 2D BatchMatMul (standard matrix multiplication)
        // [M, K] @ [K, N] = [M, N]
        int m = 8;
        int k = 64;
        int n = 32;

        var A = new Tensor<double>([m, k]);
        var B = new Tensor<double>([k, n]);
        InitializeTensor(A, 0.1);
        InitializeTensor(B, 0.1);

        // Act
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var result = engine.BatchMatMul(A, B);

        // Assert
        Assert.Equal(2, result.Rank);
        Assert.Equal(m, result.Shape[0]);
        Assert.Equal(n, result.Shape[1]);
    }

    [Fact]
    public void BatchMatMul_RankMismatch_ThrowsArgumentException()
    {
        // Arrange - BatchMatMul requires tensors with matching ranks
        var A = new Tensor<double>([2, 4, 8, 64]); // Rank 4
        var B = new Tensor<double>([2, 64, 32]);    // Rank 3

        // Act & Assert
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        Action act = () => engine.BatchMatMul(A, B);
        Assert.Throws<ArgumentException>(act);
    }

    [Fact]
    public void BatchMatMul_BatchDimMismatch_ThrowsArgumentException()
    {
        // Arrange - BatchMatMul requires matching batch dimensions
        var A = new Tensor<double>([2, 4, 8, 64]); // Batch = 2, heads = 4
        var B = new Tensor<double>([3, 4, 64, 32]); // Batch = 3 (doesn't match)

        // Act & Assert
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        Action act = () => engine.BatchMatMul(A, B);
        Assert.Throws<ArgumentException>(act);
    }

    [Fact]
    public void TensorMatMul_DimensionMismatch_ThrowsArgumentException()
    {
        // Arrange - Incompatible dimensions
        var A = new Tensor<double>([2, 4, 8, 64]); // Last dim = 64
        var B = new Tensor<double>([32, 16]);      // First dim = 32, doesn't match

        // Act & Assert
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        Action act = () => engine.TensorMatMul(A, B);
        Assert.Throws<ArgumentException>(act);
    }

    [Fact]
    public void TensorMatMul_RankMismatch_NDxND_ThrowsArgumentException()
    {
        // Arrange - ND x ND with different ranks (not supported)
        var A = new Tensor<double>([2, 4, 8, 64]); // Rank 4
        var B = new Tensor<double>([2, 8, 64, 32]); // Rank 4 but we're testing batch dim mismatch

        // Create with mismatched batch dims to trigger error
        var C = new Tensor<double>([3, 4, 8, 64]); // Batch dim 3 doesn't match A's batch dim 2
        var D = new Tensor<double>([2, 4, 64, 32]);

        // Act & Assert
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        Action act = () => engine.TensorMatMul(C, D);
        Assert.Throws<ArgumentException>(act);
    }

    #endregion

    #region Helper Methods

    private static void InitializeTensor(Tensor<double> tensor, double value)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = value;
        }
    }

    #endregion
}
