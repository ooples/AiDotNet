using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    /// <summary>
    /// Integration tests for ConvolutionalLayer with comprehensive coverage of convolution operations,
    /// forward/backward passes, and various configuration scenarios.
    /// </summary>
    public class ConvolutionalLayerIntegrationTests
    {
        private const double Tolerance = 1e-6;

        // ===== Forward Pass Tests =====

        [Fact]
        public void ConvolutionalLayer_ForwardPass_SingleChannel_ProducesCorrectShape()
        {
            // Arrange - Single channel 5x5 input, 3 filters, 3x3 kernel
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 3,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5,
                stride: 1,
                padding: 0);

            var input = new Tensor<double>([1, 1, 5, 5]); // Batch=1, Channels=1, H=5, W=5

            // Act
            var output = layer.Forward(input);

            // Assert - Output should be 3x3x3 (3 filters, 3x3 spatial)
            Assert.Equal(1, output.Shape[0]); // Batch
            Assert.Equal(3, output.Shape[1]); // Output channels
            Assert.Equal(3, output.Shape[2]); // Height: (5-3+0)/1 + 1 = 3
            Assert.Equal(3, output.Shape[3]); // Width: (5-3+0)/1 + 1 = 3
        }

        [Fact]
        public void ConvolutionalLayer_ForwardPass_MultipleChannels_ProducesCorrectShape()
        {
            // Arrange - RGB image (3 channels), 16 filters
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 3,
                outputDepth: 16,
                kernelSize: 3,
                inputHeight: 28,
                inputWidth: 28,
                stride: 1,
                padding: 1); // Same padding

            var input = new Tensor<double>([1, 3, 28, 28]);

            // Act
            var output = layer.Forward(input);

            // Assert - With padding=1, output spatial dims should match input
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(16, output.Shape[1]); // 16 filters
            Assert.Equal(28, output.Shape[2]); // Height preserved with padding
            Assert.Equal(28, output.Shape[3]); // Width preserved with padding
        }

        [Fact]
        public void ConvolutionalLayer_ForwardPass_WithStride2_ReducesSpatialDimensions()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 8,
                kernelSize: 3,
                inputHeight: 10,
                inputWidth: 10,
                stride: 2,
                padding: 0);

            var input = new Tensor<double>([1, 1, 10, 10]);

            // Act
            var output = layer.Forward(input);

            // Assert - Stride 2 should halve spatial dimensions
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(8, output.Shape[1]);
            Assert.Equal(4, output.Shape[2]); // (10-3+0)/2 + 1 = 4
            Assert.Equal(4, output.Shape[3]);
        }

        [Fact]
        public void ConvolutionalLayer_ForwardPass_BatchProcessing_WorksCorrectly()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 8,
                inputWidth: 8,
                stride: 1,
                padding: 1);

            var input = new Tensor<double>([8, 1, 8, 8]); // Batch of 8

            // Act
            var output = layer.Forward(input);

            // Assert - Batch dimension preserved
            Assert.Equal(8, output.Shape[0]);
            Assert.Equal(4, output.Shape[1]);
            Assert.Equal(8, output.Shape[2]);
            Assert.Equal(8, output.Shape[3]);
        }

        [Fact]
        public void ConvolutionalLayer_ForwardPass_WithPadding_PreservesDimensions()
        {
            // Arrange - Padding=1 with 3x3 kernel should preserve dims
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 1,
                kernelSize: 3,
                inputHeight: 7,
                inputWidth: 7,
                stride: 1,
                padding: 1);

            var input = new Tensor<double>([1, 1, 7, 7]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(7, output.Shape[2]); // Height preserved
            Assert.Equal(7, output.Shape[3]); // Width preserved
        }

        [Fact]
        public void ConvolutionalLayer_ForwardPass_With5x5Kernel_WorksCorrectly()
        {
            // Arrange - Larger kernel
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 4,
                kernelSize: 5,
                inputHeight: 12,
                inputWidth: 12,
                stride: 1,
                padding: 0);

            var input = new Tensor<double>([1, 1, 12, 12]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(4, output.Shape[1]);
            Assert.Equal(8, output.Shape[2]); // (12-5+0)/1 + 1 = 8
            Assert.Equal(8, output.Shape[3]);
        }

        [Fact]
        public void ConvolutionalLayer_ForwardPass_ReLUActivation_AppliesCorrectly()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5,
                activation: new ReLUActivation<double>());

            var input = new Tensor<double>([1, 1, 5, 5]);

            // Act
            var output = layer.Forward(input);

            // Assert - ReLU ensures non-negative outputs
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] >= 0);
            }
        }

        // ===== Backward Pass Tests =====

        [Fact]
        public void ConvolutionalLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 2,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 8,
                inputWidth: 8,
                stride: 1,
                padding: 1);

            var input = new Tensor<double>([2, 2, 8, 8]);
            var output = layer.Forward(input);

            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert - Gradient should have same shape as input
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
            Assert.Equal(input.Shape[3], inputGradient.Shape[3]);
        }

        [Fact]
        public void ConvolutionalLayer_BackwardPass_MultipleTimes_WorksConsistently()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 6,
                inputWidth: 6);

            var input = new Tensor<double>([1, 1, 6, 6]);

            // Act - Multiple forward/backward cycles
            for (int i = 0; i < 5; i++)
            {
                var output = layer.Forward(input);
                var outputGradient = new Tensor<double>(output.Shape);
                var inputGradient = layer.Backward(outputGradient);

                // Assert
                Assert.NotNull(inputGradient);
                Assert.Equal(input.Shape.Length, inputGradient.Shape.Length);
            }
        }

        // ===== Parameter Management Tests =====

        [Fact]
        public void ConvolutionalLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 3,
                outputDepth: 16,
                kernelSize: 3,
                inputHeight: 28,
                inputWidth: 28);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert
            // Expected: (3 * 16 * 3 * 3) + 16 = 432 + 16 = 448
            // (input_channels * output_channels * kernel_h * kernel_w) + biases
            Assert.Equal(448, paramCount);
        }

        [Fact]
        public void ConvolutionalLayer_GetSetParameters_RoundTrip_PreservesValues()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 2,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 8,
                inputWidth: 8);

            var originalParams = layer.GetParameters();

            // Act
            layer.SetParameters(originalParams);
            var retrievedParams = layer.GetParameters();

            // Assert
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], retrievedParams[i], precision: 10);
        }

        [Fact]
        public void ConvolutionalLayer_UpdateParameters_ChangesKernelsAndBiases()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5);

            var input = new Tensor<double>([1, 1, 5, 5]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
                outputGradient[i] = 0.1;

            layer.Backward(outputGradient);

            var paramsBefore = layer.GetParameters();

            // Act
            layer.UpdateParameters(0.01);
            var paramsAfter = layer.GetParameters();

            // Assert - Parameters should change
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

        // ===== Different Kernel Sizes Tests =====

        [Fact]
        public void ConvolutionalLayer_1x1Kernel_WorksAsPointwiseConvolution()
        {
            // Arrange - 1x1 convolution (pointwise)
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 3,
                outputDepth: 6,
                kernelSize: 1,
                inputHeight: 8,
                inputWidth: 8);

            var input = new Tensor<double>([1, 3, 8, 8]);

            // Act
            var output = layer.Forward(input);

            // Assert - Spatial dimensions unchanged
            Assert.Equal(8, output.Shape[2]);
            Assert.Equal(8, output.Shape[3]);
            Assert.Equal(6, output.Shape[1]); // 6 output channels
        }

        [Fact]
        public void ConvolutionalLayer_7x7Kernel_WorksCorrectly()
        {
            // Arrange - Large kernel
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 4,
                kernelSize: 7,
                inputHeight: 14,
                inputWidth: 14);

            var input = new Tensor<double>([1, 1, 14, 14]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(8, output.Shape[2]); // (14-7+0)/1 + 1 = 8
            Assert.Equal(8, output.Shape[3]);
        }

        // ===== Float Type Tests =====

        [Fact]
        public void ConvolutionalLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new ConvolutionalLayer<float>(
                inputDepth: 1,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 8,
                inputWidth: 8);

            var input = new Tensor<float>([1, 1, 8, 8]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(4, output.Shape[1]);
        }

        // ===== Edge Cases =====

        [Fact]
        public void ConvolutionalLayer_MinimalInput_3x3_WorksCorrectly()
        {
            // Arrange - Minimal input size for 3x3 kernel
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 1,
                kernelSize: 3,
                inputHeight: 3,
                inputWidth: 3);

            var input = new Tensor<double>([1, 1, 3, 3]);

            // Act
            var output = layer.Forward(input);

            // Assert - Should produce 1x1 output
            Assert.Equal(1, output.Shape[2]);
            Assert.Equal(1, output.Shape[3]);
        }

        [Fact]
        public void ConvolutionalLayer_LargeNumberOfFilters_64_WorksCorrectly()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 3,
                outputDepth: 64,
                kernelSize: 3,
                inputHeight: 16,
                inputWidth: 16,
                padding: 1);

            var input = new Tensor<double>([1, 3, 16, 16]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(64, output.Shape[1]);
        }

        [Fact]
        public void ConvolutionalLayer_SupportsTraining_ReturnsTrue()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 8,
                inputWidth: 8);

            // Act & Assert
            Assert.True(layer.SupportsTraining);
        }

        // ===== Reset State Tests =====

        [Fact]
        public void ConvolutionalLayer_ResetState_ClearsInternalState()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5);

            var input = new Tensor<double>([1, 1, 5, 5]);
            layer.Forward(input);

            // Act
            layer.ResetState();

            // Assert - Should work normally after reset
            var output = layer.Forward(input);
            Assert.NotNull(output);
        }

        // ===== Clone Tests =====

        [Fact]
        public void ConvolutionalLayer_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new ConvolutionalLayer<double>(
                inputDepth: 2,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 8,
                inputWidth: 8);

            var originalParams = original.GetParameters();

            // Act
            var clone = (ConvolutionalLayer<double>)original.Clone();
            var cloneParams = clone.GetParameters();

            // Assert - Clone should have same parameters
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], cloneParams[i], precision: 10);

            // Modify clone
            var newParams = new Vector<double>(cloneParams.Length);
            for (int i = 0; i < newParams.Length; i++)
                newParams[i] = 99.0;
            clone.SetParameters(newParams);

            // Original should be unchanged
            var originalParamsAfter = original.GetParameters();
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], originalParamsAfter[i], precision: 10);
        }

        // ===== Training Scenario Tests =====

        [Fact]
        public void ConvolutionalLayer_TrainingIterations_UpdatesParameters()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5);

            var input = new Tensor<double>([1, 1, 5, 5]);
            for (int i = 0; i < input.Length; i++)
                input[i] = (i % 10) * 0.1;

            var initialParams = layer.GetParameters();

            // Act - Training loop
            for (int i = 0; i < 10; i++)
            {
                var output = layer.Forward(input);
                var gradient = new Tensor<double>(output.Shape);
                for (int j = 0; j < gradient.Length; j++)
                    gradient[j] = 0.1;

                layer.Backward(gradient);
                layer.UpdateParameters(0.01);
            }

            var finalParams = layer.GetParameters();

            // Assert - Parameters should have changed
            bool changed = false;
            for (int i = 0; i < initialParams.Length; i++)
            {
                if (Math.Abs(initialParams[i] - finalParams[i]) > 1e-6)
                {
                    changed = true;
                    break;
                }
            }
            Assert.True(changed);
        }

        // ===== Different Activation Functions =====

        [Fact]
        public void ConvolutionalLayer_WithTanhActivation_OutputsInRange()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5,
                activation: new TanhActivation<double>());

            var input = new Tensor<double>([1, 1, 5, 5]);
            for (int i = 0; i < input.Length; i++)
                input[i] = (i - 12) * 2.0; // Mix of positive and negative

            // Act
            var output = layer.Forward(input);

            // Assert - Tanh outputs in (-1, 1)
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] > -1.0);
                Assert.True(output[i] < 1.0);
            }
        }

        [Fact]
        public void ConvolutionalLayer_WithSigmoidActivation_OutputsInRange()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 2,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5,
                activation: new SigmoidActivation<double>());

            var input = new Tensor<double>([1, 1, 5, 5]);

            // Act
            var output = layer.Forward(input);

            // Assert - Sigmoid outputs in (0, 1)
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] > 0.0);
                Assert.True(output[i] < 1.0);
            }
        }

        // ===== Padding Variations =====

        [Fact]
        public void ConvolutionalLayer_WithPadding2_IncreasesOutputSize()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 1,
                kernelSize: 3,
                inputHeight: 5,
                inputWidth: 5,
                padding: 2);

            var input = new Tensor<double>([1, 1, 5, 5]);

            // Act
            var output = layer.Forward(input);

            // Assert - Padding of 2 should increase output size
            Assert.True(output.Shape[2] >= 5);
            Assert.True(output.Shape[3] >= 5);
        }

        // ===== Stride Variations =====

        [Fact]
        public void ConvolutionalLayer_WithStride3_SignificantlyReducesDimensions()
        {
            // Arrange
            var layer = new ConvolutionalLayer<double>(
                inputDepth: 1,
                outputDepth: 4,
                kernelSize: 3,
                inputHeight: 15,
                inputWidth: 15,
                stride: 3);

            var input = new Tensor<double>([1, 1, 15, 15]);

            // Act
            var output = layer.Forward(input);

            // Assert - Stride 3 should significantly reduce dims
            Assert.True(output.Shape[2] <= 5); // (15-3+0)/3 + 1 = 5
            Assert.True(output.Shape[3] <= 5);
        }
    }
}
