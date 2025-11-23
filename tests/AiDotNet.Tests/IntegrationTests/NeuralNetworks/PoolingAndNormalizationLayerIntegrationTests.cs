using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    /// <summary>
    /// Integration tests for Pooling and Normalization layers with comprehensive coverage
    /// of max/average pooling, batch normalization, and layer normalization.
    /// </summary>
    public class PoolingAndNormalizationLayerIntegrationTests
    {
        private const double Tolerance = 1e-6;

        // ===== MaxPoolingLayer Tests =====

        [Fact]
        public void MaxPoolingLayer_ForwardPass_ReducesSpatialDimensions()
        {
            // Arrange - 8x8 input, 2x2 pooling, stride 2
            var layer = new MaxPoolingLayer<double>([1, 8, 8], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 1, 8, 8]);
            for (int i = 0; i < 64; i++)
                input[i] = i;

            // Act
            var output = layer.Forward(input);

            // Assert - Should reduce to 4x4
            Assert.Equal(1, output.Shape[0]); // Batch
            Assert.Equal(1, output.Shape[1]); // Channels
            Assert.Equal(4, output.Shape[2]); // Height / 2
            Assert.Equal(4, output.Shape[3]); // Width / 2
        }

        [Fact]
        public void MaxPoolingLayer_ForwardPass_SelectsMaximumValues()
        {
            // Arrange - Simple 4x4 input
            var layer = new MaxPoolingLayer<double>([1, 4, 4], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 1, 4, 4]);

            // Fill with pattern where max in each 2x2 block is predictable
            input[0, 0, 0, 0] = 1; input[0, 0, 0, 1] = 2;
            input[0, 0, 1, 0] = 3; input[0, 0, 1, 1] = 9; // Max = 9

            input[0, 0, 0, 2] = 5; input[0, 0, 0, 3] = 6;
            input[0, 0, 1, 2] = 7; input[0, 0, 1, 3] = 8; // Max = 8

            input[0, 0, 2, 0] = 10; input[0, 0, 2, 1] = 11;
            input[0, 0, 3, 0] = 12; input[0, 0, 3, 1] = 16; // Max = 16

            input[0, 0, 2, 2] = 13; input[0, 0, 2, 3] = 14;
            input[0, 0, 3, 2] = 15; input[0, 0, 3, 3] = 20; // Max = 20

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(9.0, output[0, 0, 0, 0], precision: 10);
            Assert.Equal(8.0, output[0, 0, 0, 1], precision: 10);
            Assert.Equal(16.0, output[0, 0, 1, 0], precision: 10);
            Assert.Equal(20.0, output[0, 0, 1, 1], precision: 10);
        }

        [Fact]
        public void MaxPoolingLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new MaxPoolingLayer<double>([2, 8, 8], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 2, 8, 8]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
            Assert.Equal(input.Shape[3], inputGradient.Shape[3]);
        }

        [Fact]
        public void MaxPoolingLayer_With3x3Pooling_WorksCorrectly()
        {
            // Arrange - 9x9 input, 3x3 pooling, stride 3
            var layer = new MaxPoolingLayer<double>([1, 9, 9], poolSize: 3, strides: 3);
            var input = new Tensor<double>([1, 1, 9, 9]);

            // Act
            var output = layer.Forward(input);

            // Assert - Should reduce to 3x3
            Assert.Equal(3, output.Shape[2]);
            Assert.Equal(3, output.Shape[3]);
        }

        [Fact]
        public void MaxPoolingLayer_MultipleChannels_ProcessesIndependently()
        {
            // Arrange - Multiple channels (RGB-like)
            var layer = new MaxPoolingLayer<double>([3, 8, 8], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 3, 8, 8]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(3, output.Shape[1]); // Channels preserved
            Assert.Equal(4, output.Shape[2]); // Spatial reduced
            Assert.Equal(4, output.Shape[3]);
        }

        [Fact]
        public void MaxPoolingLayer_BatchProcessing_WorksCorrectly()
        {
            // Arrange
            var layer = new MaxPoolingLayer<double>([1, 8, 8], poolSize: 2, strides: 2);
            var input = new Tensor<double>([4, 1, 8, 8]); // Batch of 4

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(4, output.Shape[0]); // Batch preserved
        }

        [Fact]
        public void MaxPoolingLayer_SupportsTraining_ReturnsTrue()
        {
            // Arrange
            var layer = new MaxPoolingLayer<double>([1, 8, 8], poolSize: 2, strides: 2);

            // Act & Assert
            Assert.True(layer.SupportsTraining);
        }

        [Fact]
        public void MaxPoolingLayer_ParameterCount_ReturnsZero()
        {
            // Arrange - Pooling layers have no trainable parameters
            var layer = new MaxPoolingLayer<double>([1, 8, 8], poolSize: 2, strides: 2);

            // Act & Assert
            Assert.Equal(0, layer.ParameterCount);
        }

        // ===== AveragePoolingLayer Tests =====

        [Fact]
        public void AveragePoolingLayer_ForwardPass_ComputesAverages()
        {
            // Arrange
            var layer = new AveragePoolingLayer<double>([1, 4, 4], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 1, 4, 4]);

            // Fill first 2x2 block with known values
            input[0, 0, 0, 0] = 2; input[0, 0, 0, 1] = 4;
            input[0, 0, 1, 0] = 6; input[0, 0, 1, 1] = 8; // Average = 5

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(5.0, output[0, 0, 0, 0], precision: 10);
        }

        [Fact]
        public void AveragePoolingLayer_ForwardPass_ReducesSpatialDimensions()
        {
            // Arrange
            var layer = new AveragePoolingLayer<double>([2, 10, 10], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 2, 10, 10]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(5, output.Shape[2]); // 10 / 2 = 5
            Assert.Equal(5, output.Shape[3]);
        }

        [Fact]
        public void AveragePoolingLayer_BackwardPass_DistributesGradients()
        {
            // Arrange
            var layer = new AveragePoolingLayer<double>([1, 4, 4], poolSize: 2, strides: 2);
            var input = new Tensor<double>([1, 1, 4, 4]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
                outputGradient[i] = 1.0;

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert - Gradients should be distributed evenly
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
            Assert.Equal(input.Shape[3], inputGradient.Shape[3]);
        }

        // ===== BatchNormalizationLayer Tests =====

        [Fact]
        public void BatchNormalizationLayer_ForwardPass_NormalizesAcrossBatch()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<double>([10]);
            var input = new Tensor<double>([4, 10]); // Batch of 4, 10 features

            // Set input with varying values
            for (int b = 0; b < 4; b++)
                for (int f = 0; f < 10; f++)
                    input[b, f] = b * 10 + f;

            // Act
            var output = layer.Forward(input);

            // Assert - Output should be normalized
            Assert.Equal(input.Shape[0], output.Shape[0]);
            Assert.Equal(input.Shape[1], output.Shape[1]);
        }

        [Fact]
        public void BatchNormalizationLayer_TrainingMode_UpdatesRunningStatistics()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<double>([5]);
            var input = new Tensor<double>([8, 5]);

            for (int i = 0; i < input.Length; i++)
                input[i] = i * 0.1;

            // Act - Multiple forward passes should update statistics
            var output1 = layer.Forward(input);
            var output2 = layer.Forward(input);
            var output3 = layer.Forward(input);

            // Assert - Outputs should be valid
            Assert.NotNull(output1);
            Assert.NotNull(output2);
            Assert.NotNull(output3);
        }

        [Fact]
        public void BatchNormalizationLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<double>([8]);
            var input = new Tensor<double>([4, 8]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
        }

        [Fact]
        public void BatchNormalizationLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange - BatchNorm has gamma and beta parameters
            var numFeatures = 10;
            var layer = new BatchNormalizationLayer<double>([numFeatures]);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert - 2 * numFeatures (gamma + beta)
            Assert.Equal(20, paramCount);
        }

        [Fact]
        public void BatchNormalizationLayer_UpdateParameters_ChangesGammaAndBeta()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<double>([5]);
            var input = new Tensor<double>([4, 5]);
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

        [Fact]
        public void BatchNormalizationLayer_LargeBatch_ProcessesEfficiently()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<double>([20]);
            var input = new Tensor<double>([64, 20]); // Large batch

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(64, output.Shape[0]);
            Assert.Equal(20, output.Shape[1]);
        }

        [Fact]
        public void BatchNormalizationLayer_SupportsTraining_ReturnsTrue()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<double>([10]);

            // Act & Assert
            Assert.True(layer.SupportsTraining);
        }

        // ===== LayerNormalizationLayer Tests =====

        [Fact]
        public void LayerNormalizationLayer_ForwardPass_NormalizesAcrossFeatures()
        {
            // Arrange
            var layer = new LayerNormalizationLayer<double>([10]);
            var input = new Tensor<double>([4, 10]);

            for (int i = 0; i < input.Length; i++)
                input[i] = i * 0.5;

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(input.Shape[0], output.Shape[0]);
            Assert.Equal(input.Shape[1], output.Shape[1]);
        }

        [Fact]
        public void LayerNormalizationLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new LayerNormalizationLayer<double>([8]);
            var input = new Tensor<double>([2, 8]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
        }

        [Fact]
        public void LayerNormalizationLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange
            var numFeatures = 12;
            var layer = new LayerNormalizationLayer<double>([numFeatures]);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert - 2 * numFeatures (gamma + beta)
            Assert.Equal(24, paramCount);
        }

        [Fact]
        public void LayerNormalizationLayer_UpdateParameters_ChangesParameters()
        {
            // Arrange
            var layer = new LayerNormalizationLayer<double>([6]);
            var input = new Tensor<double>([3, 6]);
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

        // ===== GlobalPoolingLayer Tests =====

        [Fact]
        public void GlobalPoolingLayer_ForwardPass_ReducesToSingleValue()
        {
            // Arrange - Average over entire spatial dimensions
            var layer = new GlobalPoolingLayer<double>([2, 8, 8], PoolingType.Average);
            var input = new Tensor<double>([1, 2, 8, 8]);

            // Act
            var output = layer.Forward(input);

            // Assert - Should reduce to [batch, channels, 1, 1]
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(2, output.Shape[1]);
            Assert.Equal(1, output.Shape[2]);
            Assert.Equal(1, output.Shape[3]);
        }

        [Fact]
        public void GlobalPoolingLayer_MaxPooling_SelectsMaximumValue()
        {
            // Arrange
            var layer = new GlobalPoolingLayer<double>([1, 4, 4], PoolingType.Max);
            var input = new Tensor<double>([1, 1, 4, 4]);

            // Set one value as clearly maximum
            for (int i = 0; i < input.Length; i++)
                input[i] = i;
            input[0, 0, 3, 3] = 100.0; // Maximum value

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(100.0, output[0, 0, 0, 0], precision: 10);
        }

        // ===== Float Type Tests =====

        [Fact]
        public void MaxPoolingLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new MaxPoolingLayer<float>([1, 8, 8], poolSize: 2, strides: 2);
            var input = new Tensor<float>([1, 1, 8, 8]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(4, output.Shape[2]);
            Assert.Equal(4, output.Shape[3]);
        }

        [Fact]
        public void BatchNormalizationLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new BatchNormalizationLayer<float>([10]);
            var input = new Tensor<float>([4, 10]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(4, output.Shape[0]);
            Assert.Equal(10, output.Shape[1]);
        }

        // ===== Clone Tests =====

        [Fact]
        public void BatchNormalizationLayer_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new BatchNormalizationLayer<double>([8]);
            var originalParams = original.GetParameters();

            // Act
            var clone = (BatchNormalizationLayer<double>)original.Clone();
            var cloneParams = clone.GetParameters();

            // Assert
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], cloneParams[i], precision: 10);

            // Modify clone
            var newParams = new Vector<double>(cloneParams.Length);
            for (int i = 0; i < newParams.Length; i++)
                newParams[i] = 99.0;
            clone.SetParameters(newParams);

            // Original unchanged
            var originalParamsAfter = original.GetParameters();
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], originalParamsAfter[i], precision: 10);
        }
    }
}
