using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    /// <summary>
    /// Integration tests for specialized layers including Dropout, Embedding, Attention,
    /// Flatten, Reshape, and other utility layers.
    /// </summary>
    public class SpecializedLayerIntegrationTests
    {
        private const double Tolerance = 1e-6;

        // ===== DropoutLayer Tests =====

        [Fact]
        public void DropoutLayer_TrainingMode_DropsOutSomeValues()
        {
            // Arrange
            var layer = new DropoutLayer<double>(dropoutRate: 0.5);
            var input = new Tensor<double>([1, 100]);
            for (int i = 0; i < 100; i++)
                input[0, i] = 1.0;

            // Act
            var output = layer.Forward(input);

            // Assert - Some values should be zeroed out (approximately half)
            int zeroCount = 0;
            for (int i = 0; i < 100; i++)
            {
                if (Math.Abs(output[0, i]) < 1e-10)
                    zeroCount++;
            }
            Assert.True(zeroCount > 20 && zeroCount < 80); // Probabilistic test
        }

        [Fact]
        public void DropoutLayer_InferenceMode_PreservesAllValues()
        {
            // Arrange
            var layer = new DropoutLayer<double>(dropoutRate: 0.5);
            layer.SetInferenceMode();
            var input = new Tensor<double>([1, 50]);
            for (int i = 0; i < 50; i++)
                input[0, i] = 1.0;

            // Act
            var output = layer.Forward(input);

            // Assert - All values should be preserved (scaled by 1-rate)
            for (int i = 0; i < 50; i++)
            {
                Assert.True(Math.Abs(output[0, i] - 1.0) < 0.1);
            }
        }

        [Fact]
        public void DropoutLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new DropoutLayer<double>(dropoutRate: 0.3);
            var input = new Tensor<double>([2, 20]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
        }

        [Fact]
        public void DropoutLayer_DifferentRates_ProduceDifferentDropouts()
        {
            // Arrange
            var layer1 = new DropoutLayer<double>(dropoutRate: 0.2);
            var layer2 = new DropoutLayer<double>(dropoutRate: 0.8);
            var input = new Tensor<double>([1, 100]);
            for (int i = 0; i < 100; i++)
                input[0, i] = 1.0;

            // Act
            var output1 = layer1.Forward(input);
            var output2 = layer2.Forward(input);

            // Count zeros
            int zeros1 = 0, zeros2 = 0;
            for (int i = 0; i < 100; i++)
            {
                if (Math.Abs(output1[0, i]) < 1e-10) zeros1++;
                if (Math.Abs(output2[0, i]) < 1e-10) zeros2++;
            }

            // Assert - Higher rate should drop more
            Assert.True(zeros2 > zeros1);
        }

        [Fact]
        public void DropoutLayer_ParameterCount_ReturnsZero()
        {
            // Arrange
            var layer = new DropoutLayer<double>(dropoutRate: 0.5);

            // Act & Assert
            Assert.Equal(0, layer.ParameterCount);
        }

        // ===== EmbeddingLayer Tests =====

        [Fact]
        public void EmbeddingLayer_ForwardPass_ProducesCorrectShape()
        {
            // Arrange - Vocabulary of 100, embedding dim of 16
            var layer = new EmbeddingLayer<double>(vocabularySize: 100, embeddingDim: 16);
            var input = new Tensor<int>([2, 10]); // Batch=2, Sequence=10
            for (int i = 0; i < 20; i++)
                input[i] = i % 100; // Valid indices

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]); // Batch
            Assert.Equal(10, output.Shape[1]); // Sequence
            Assert.Equal(16, output.Shape[2]); // Embedding dimension
        }

        [Fact]
        public void EmbeddingLayer_SameIndex_ProducesSameEmbedding()
        {
            // Arrange
            var layer = new EmbeddingLayer<double>(vocabularySize: 50, embeddingDim: 8);
            var input = new Tensor<int>([1, 2]);
            input[0, 0] = 5;
            input[0, 1] = 5; // Same index twice

            // Act
            var output = layer.Forward(input);

            // Assert - Same index should produce same embedding
            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(output[0, 0, i], output[0, 1, i], precision: 10);
            }
        }

        [Fact]
        public void EmbeddingLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange
            var vocabSize = 100;
            var embeddingDim = 16;
            var layer = new EmbeddingLayer<double>(vocabularySize: vocabSize, embeddingDim: embeddingDim);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert - vocab_size * embedding_dim
            Assert.Equal(1600, paramCount);
        }

        [Fact]
        public void EmbeddingLayer_UpdateParameters_ChangesEmbeddings()
        {
            // Arrange
            var layer = new EmbeddingLayer<double>(vocabularySize: 10, embeddingDim: 4);
            var input = new Tensor<int>([1, 2]);
            input[0, 0] = 3;
            input[0, 1] = 7;

            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
                outputGradient[i] = 0.1;

            layer.Backward(outputGradient);
            var paramsBefore = layer.GetParameters();

            // Act
            layer.UpdateParameters(0.01);
            var paramsAfter = layer.GetParameters();

            // Assert
            bool changed = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-10)
                {
                    changed = true;
                    break;
                }
            }
            Assert.True(changed);
        }

        // ===== FlattenLayer Tests =====

        [Fact]
        public void FlattenLayer_ForwardPass_FlattensMultiDimensionalInput()
        {
            // Arrange - 4D input (batch, channels, height, width)
            var layer = new FlattenLayer<double>();
            var input = new Tensor<double>([2, 3, 4, 5]); // Batch=2, 3x4x5 volume

            // Act
            var output = layer.Forward(input);

            // Assert - Should flatten to [batch, features]
            Assert.Equal(2, output.Shape[0]); // Batch preserved
            Assert.Equal(60, output.Shape[1]); // 3*4*5 = 60
        }

        [Fact]
        public void FlattenLayer_BackwardPass_RestoresOriginalShape()
        {
            // Arrange
            var layer = new FlattenLayer<double>();
            var input = new Tensor<double>([1, 2, 3, 4]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert - Should restore original shape
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
            Assert.Equal(input.Shape[3], inputGradient.Shape[3]);
        }

        [Fact]
        public void FlattenLayer_ParameterCount_ReturnsZero()
        {
            // Arrange
            var layer = new FlattenLayer<double>();

            // Act & Assert
            Assert.Equal(0, layer.ParameterCount);
        }

        // ===== ReshapeLayer Tests =====

        [Fact]
        public void ReshapeLayer_ForwardPass_ReshapesToTargetShape()
        {
            // Arrange - Reshape [2, 12] to [2, 3, 4]
            var layer = new ReshapeLayer<double>(targetShape: [3, 4]);
            var input = new Tensor<double>([2, 12]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]); // Batch preserved
            Assert.Equal(3, output.Shape[1]);
            Assert.Equal(4, output.Shape[2]);
        }

        [Fact]
        public void ReshapeLayer_BackwardPass_RestoresInputShape()
        {
            // Arrange
            var layer = new ReshapeLayer<double>(targetShape: [4, 5]);
            var input = new Tensor<double>([1, 20]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
        }

        // ===== AttentionLayer Tests =====

        [Fact]
        public void AttentionLayer_ForwardPass_ProducesCorrectShape()
        {
            // Arrange
            var layer = new AttentionLayer<double>(embeddingDim: 16);
            var query = new Tensor<double>([2, 10, 16]); // Batch=2, Seq=10, Dim=16
            var key = new Tensor<double>([2, 10, 16]);
            var value = new Tensor<double>([2, 10, 16]);

            // Act
            var output = layer.Forward(query, key, value);

            // Assert
            Assert.Equal(2, output.Shape[0]); // Batch
            Assert.Equal(10, output.Shape[1]); // Sequence
            Assert.Equal(16, output.Shape[2]); // Dimension
        }

        [Fact]
        public void AttentionLayer_SelfAttention_WorksCorrectly()
        {
            // Arrange - Self-attention: Q, K, V are all the same
            var layer = new AttentionLayer<double>(embeddingDim: 8);
            var input = new Tensor<double>([1, 5, 8]);

            // Act - Use same tensor for Q, K, V
            var output = layer.Forward(input, input, input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(5, output.Shape[1]);
            Assert.Equal(8, output.Shape[2]);
        }

        [Fact]
        public void AttentionLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange - Attention has query, key, value, and output projections
            var embeddingDim = 16;
            var layer = new AttentionLayer<double>(embeddingDim: embeddingDim);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert - Should have parameters for Q, K, V, and output projections
            Assert.True(paramCount > 0);
        }

        // ===== MultiHeadAttentionLayer Tests =====

        [Fact]
        public void MultiHeadAttentionLayer_ForwardPass_ProducesCorrectShape()
        {
            // Arrange
            var layer = new MultiHeadAttentionLayer<double>(
                embeddingDim: 64,
                numHeads: 8);

            var query = new Tensor<double>([2, 10, 64]);
            var key = new Tensor<double>([2, 10, 64]);
            var value = new Tensor<double>([2, 10, 64]);

            // Act
            var output = layer.Forward(query, key, value);

            // Assert
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(10, output.Shape[1]);
            Assert.Equal(64, output.Shape[2]);
        }

        [Fact]
        public void MultiHeadAttentionLayer_DifferentHeadCounts_WorkCorrectly()
        {
            // Arrange - Try different numbers of heads
            var layer4 = new MultiHeadAttentionLayer<double>(embeddingDim: 64, numHeads: 4);
            var layer8 = new MultiHeadAttentionLayer<double>(embeddingDim: 64, numHeads: 8);

            var input = new Tensor<double>([1, 5, 64]);

            // Act
            var output4 = layer4.Forward(input, input, input);
            var output8 = layer8.Forward(input, input, input);

            // Assert - Both should produce same output shape
            Assert.Equal(output4.Shape[0], output8.Shape[0]);
            Assert.Equal(output4.Shape[1], output8.Shape[1]);
            Assert.Equal(output4.Shape[2], output8.Shape[2]);
        }

        // ===== ActivationLayer Tests =====

        [Fact]
        public void ActivationLayer_ReLU_AppliesCorrectly()
        {
            // Arrange
            var layer = new ActivationLayer<double>(new ReLUActivation<double>());
            var input = new Tensor<double>([1, 10]);
            for (int i = 0; i < 10; i++)
                input[0, i] = i - 5; // Mix of positive and negative

            // Act
            var output = layer.Forward(input);

            // Assert - Negative values should be zero
            for (int i = 0; i < 5; i++)
                Assert.Equal(0.0, output[0, i], precision: 10);

            // Positive values preserved
            for (int i = 5; i < 10; i++)
                Assert.True(output[0, i] > 0);
        }

        [Fact]
        public void ActivationLayer_Sigmoid_OutputsInRange()
        {
            // Arrange
            var layer = new ActivationLayer<double>(new SigmoidActivation<double>());
            var input = new Tensor<double>([1, 10]);
            for (int i = 0; i < 10; i++)
                input[0, i] = (i - 5) * 2; // Range from -10 to 8

            // Act
            var output = layer.Forward(input);

            // Assert - All outputs in (0, 1)
            for (int i = 0; i < 10; i++)
            {
                Assert.True(output[0, i] > 0);
                Assert.True(output[0, i] < 1);
            }
        }

        [Fact]
        public void ActivationLayer_ParameterCount_ReturnsZero()
        {
            // Arrange
            var layer = new ActivationLayer<double>(new TanhActivation<double>());

            // Act & Assert
            Assert.Equal(0, layer.ParameterCount);
        }

        // ===== PositionalEncodingLayer Tests =====

        [Fact]
        public void PositionalEncodingLayer_ForwardPass_AddsPositionalInfo()
        {
            // Arrange
            var layer = new PositionalEncodingLayer<double>(
                maxSequenceLength: 100,
                embeddingDim: 16);

            var input = new Tensor<double>([2, 10, 16]); // Batch=2, Seq=10, Dim=16

            // Act
            var output = layer.Forward(input);

            // Assert - Shape preserved, but values modified
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(10, output.Shape[1]);
            Assert.Equal(16, output.Shape[2]);
        }

        [Fact]
        public void PositionalEncodingLayer_DifferentPositions_ProduceDifferentEncodings()
        {
            // Arrange
            var layer = new PositionalEncodingLayer<double>(
                maxSequenceLength: 50,
                embeddingDim: 8);

            var input = new Tensor<double>([1, 10, 8]);
            // Set all inputs to same value
            for (int i = 0; i < input.Length; i++)
                input[i] = 1.0;

            // Act
            var output = layer.Forward(input);

            // Assert - Different positions should have different values
            bool hasDifference = false;
            for (int i = 1; i < 10; i++)
            {
                if (Math.Abs(output[0, 0, 0] - output[0, i, 0]) > 0.01)
                {
                    hasDifference = true;
                    break;
                }
            }
            Assert.True(hasDifference);
        }

        // ===== AddLayer Tests =====

        [Fact]
        public void AddLayer_ForwardPass_AddsInputs()
        {
            // Arrange
            var layer = new AddLayer<double>();
            var input1 = new Tensor<double>([1, 10]);
            var input2 = new Tensor<double>([1, 10]);

            for (int i = 0; i < 10; i++)
            {
                input1[0, i] = i;
                input2[0, i] = i * 2;
            }

            // Act
            var output = layer.Forward(input1, input2);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(i * 3, output[0, i], precision: 10);
            }
        }

        [Fact]
        public void AddLayer_ParameterCount_ReturnsZero()
        {
            // Arrange
            var layer = new AddLayer<double>();

            // Act & Assert
            Assert.Equal(0, layer.ParameterCount);
        }

        // ===== MultiplyLayer Tests =====

        [Fact]
        public void MultiplyLayer_ForwardPass_MultipliesInputs()
        {
            // Arrange
            var layer = new MultiplyLayer<double>();
            var input1 = new Tensor<double>([1, 5]);
            var input2 = new Tensor<double>([1, 5]);

            for (int i = 0; i < 5; i++)
            {
                input1[0, i] = i + 1;
                input2[0, i] = 2;
            }

            // Act
            var output = layer.Forward(input1, input2);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal((i + 1) * 2, output[0, i], precision: 10);
            }
        }

        // ===== ConcatenateLayer Tests =====

        [Fact]
        public void ConcatenateLayer_ForwardPass_ConcatenatesAlongAxis()
        {
            // Arrange
            var layer = new ConcatenateLayer<double>(axis: 1);
            var input1 = new Tensor<double>([2, 5]);
            var input2 = new Tensor<double>([2, 3]);

            // Act
            var output = layer.Forward(input1, input2);

            // Assert - Should concatenate along axis 1
            Assert.Equal(2, output.Shape[0]); // Batch preserved
            Assert.Equal(8, output.Shape[1]); // 5 + 3 = 8
        }

        [Fact]
        public void ConcatenateLayer_BackwardPass_SplitsGradients()
        {
            // Arrange
            var layer = new ConcatenateLayer<double>(axis: 1);
            var input1 = new Tensor<double>([1, 4]);
            var input2 = new Tensor<double>([1, 6]);
            var output = layer.Forward(input1, input2);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradients = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(2, inputGradients.Length); // Two input gradients
            Assert.Equal(4, inputGradients[0].Shape[1]); // First input size
            Assert.Equal(6, inputGradients[1].Shape[1]); // Second input size
        }

        // ===== Float Type Tests =====

        [Fact]
        public void DropoutLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new DropoutLayer<float>(dropoutRate: 0.5f);
            var input = new Tensor<float>([1, 50]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(input.Shape[0], output.Shape[0]);
            Assert.Equal(input.Shape[1], output.Shape[1]);
        }

        [Fact]
        public void EmbeddingLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new EmbeddingLayer<float>(vocabularySize: 50, embeddingDim: 8);
            var input = new Tensor<int>([1, 5]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(8, output.Shape[2]);
        }
    }
}
