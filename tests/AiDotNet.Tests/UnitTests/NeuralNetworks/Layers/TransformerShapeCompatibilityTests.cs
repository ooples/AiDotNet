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
        Assert.Equal(2, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(attentionSize, output.Shape[1]);
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
        Assert.Equal(3, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLength, output.Shape[1]);
        Assert.Equal(attentionSize, output.Shape[2]);
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
        // Note: attentionSize must equal inputSize for residual connections until
        // AttentionLayer gets an output projection layer (Wo) like industry-standard transformers
        int inputSize = 64;
        int attentionSize = 64; // Must match inputSize for residual connection
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
        Assert.Equal(3, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLength, output.Shape[1]);
        // Output size depends on attention layer output
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
