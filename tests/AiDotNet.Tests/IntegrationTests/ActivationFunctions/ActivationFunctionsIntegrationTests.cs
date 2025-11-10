using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.ActivationFunctions
{
    /// <summary>
    /// Integration tests for activation functions with mathematically verified results.
    /// Tests ensure activation functions produce correct outputs and gradients.
    /// </summary>
    public class ActivationFunctionsIntegrationTests
    {
        [Fact]
        public void ReLUActivation_PositiveValues_ReturnsInput()
        {
            // Arrange
            var relu = new ReLUActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0; input[4] = 5.0;

            // Act
            var output = relu.Forward(input);

            // Assert - ReLU(x) = max(0, x), so positive values pass through
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(input[i], output[i], precision: 10);
            }
        }

        [Fact]
        public void ReLUActivation_NegativeValues_ReturnsZero()
        {
            // Arrange
            var relu = new ReLUActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -3.0; input[3] = -4.0; input[4] = -5.0;

            // Act
            var output = relu.Forward(input);

            // Assert - ReLU(x) = max(0, x), so negative values become 0
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(0.0, output[i], precision: 10);
            }
        }

        [Fact]
        public void ReLUActivation_MixedValues_ProducesCorrectResult()
        {
            // Arrange
            var relu = new ReLUActivation<double>();
            var input = new Tensor<double>(new[] { 6 });
            input[0] = -2.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 2.0; input[5] = 3.0;

            // Act
            var output = relu.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10); // max(0, -2) = 0
            Assert.Equal(0.0, output[1], precision: 10); // max(0, -1) = 0
            Assert.Equal(0.0, output[2], precision: 10); // max(0, 0) = 0
            Assert.Equal(1.0, output[3], precision: 10); // max(0, 1) = 1
            Assert.Equal(2.0, output[4], precision: 10); // max(0, 2) = 2
            Assert.Equal(3.0, output[5], precision: 10); // max(0, 3) = 3
        }

        [Fact]
        public void SigmoidActivation_ZeroInput_ReturnsHalf()
        {
            // Arrange
            var sigmoid = new SigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = sigmoid.Forward(input);

            // Assert - sigmoid(0) = 1 / (1 + e^0) = 0.5
            Assert.Equal(0.5, output[0], precision: 10);
        }

        [Fact]
        public void SigmoidActivation_LargePositive_ApproachesOne()
        {
            // Arrange
            var sigmoid = new SigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 10.0;

            // Act
            var output = sigmoid.Forward(input);

            // Assert - sigmoid(10) ≈ 0.9999 (very close to 1)
            Assert.True(output[0] > 0.9999);
        }

        [Fact]
        public void SigmoidActivation_LargeNegative_ApproachesZero()
        {
            // Arrange
            var sigmoid = new SigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = -10.0;

            // Act
            var output = sigmoid.Forward(input);

            // Assert - sigmoid(-10) ≈ 0.0000 (very close to 0)
            Assert.True(output[0] < 0.0001);
        }

        [Fact]
        public void SigmoidActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var sigmoid = new SigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = 0.0; input[2] = 1.0;

            // Act
            var output = sigmoid.Forward(input);

            // Assert
            // sigmoid(-1) = 1 / (1 + e^1) ≈ 0.2689
            Assert.Equal(0.26894142137, output[0], precision: 8);
            // sigmoid(0) = 0.5
            Assert.Equal(0.5, output[1], precision: 10);
            // sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
            Assert.Equal(0.73105857863, output[2], precision: 8);
        }

        [Fact]
        public void TanhActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var tanh = new TanhActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = tanh.Forward(input);

            // Assert - tanh(0) = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void TanhActivation_SymmetricFunction_IsSymmetric()
        {
            // Arrange
            var tanh = new TanhActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 2.0; input[1] = -2.0;

            // Act
            var output = tanh.Forward(input);

            // Assert - tanh(-x) = -tanh(x)
            Assert.Equal(-output[0], output[1], precision: 10);
        }

        [Fact]
        public void TanhActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var tanh = new TanhActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = 0.0; input[2] = 1.0;

            // Act
            var output = tanh.Forward(input);

            // Assert
            // tanh(-1) ≈ -0.76159
            Assert.Equal(-0.76159415595, output[0], precision: 8);
            // tanh(0) = 0
            Assert.Equal(0.0, output[1], precision: 10);
            // tanh(1) ≈ 0.76159
            Assert.Equal(0.76159415595, output[2], precision: 8);
        }

        [Fact]
        public void SoftmaxActivation_ProducesValidProbabilityDistribution()
        {
            // Arrange
            var softmax = new SoftmaxActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = softmax.Forward(input);

            // Assert - Output should sum to 1.0 (probability distribution)
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
                // Each value should be between 0 and 1
                Assert.True(output[i] >= 0.0 && output[i] <= 1.0);
            }
            Assert.Equal(1.0, sum, precision: 10);
        }

        [Fact]
        public void SoftmaxActivation_LargestInput_GetHighestProbability()
        {
            // Arrange
            var softmax = new SoftmaxActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 10.0; input[3] = 3.0; input[4] = 1.5; // 10.0 is largest

            // Act
            var output = softmax.Forward(input);

            // Assert - Index 2 (value 10.0) should have highest probability
            var maxIndex = 0;
            var maxValue = output[0];
            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > maxValue)
                {
                    maxValue = output[i];
                    maxIndex = i;
                }
            }
            Assert.Equal(2, maxIndex);
            // Should be very close to 1.0
            Assert.True(output[2] > 0.999);
        }

        [Fact]
        public void SoftmaxActivation_UniformInput_ProducesUniformDistribution()
        {
            // Arrange
            var softmax = new SoftmaxActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 1.0; input[1] = 1.0; input[2] = 1.0; input[3] = 1.0;

            // Act
            var output = softmax.Forward(input);

            // Assert - All outputs should be equal (0.25 each)
            for (int i = 0; i < output.Length; i++)
            {
                Assert.Equal(0.25, output[i], precision: 10);
            }
        }

        [Fact]
        public void LeakyReLUActivation_PositiveValues_ReturnsInput()
        {
            // Arrange
            var leakyRelu = new LeakyReLUActivation<double>(alpha: 0.01);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = leakyRelu.Forward(input);

            // Assert - Positive values pass through
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(input[i], output[i], precision: 10);
            }
        }

        [Fact]
        public void LeakyReLUActivation_NegativeValues_ReturnsScaled()
        {
            // Arrange
            var alpha = 0.01;
            var leakyRelu = new LeakyReLUActivation<double>(alpha);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -3.0;

            // Act
            var output = leakyRelu.Forward(input);

            // Assert - Negative values are scaled by alpha
            Assert.Equal(-0.01, output[0], precision: 10);
            Assert.Equal(-0.02, output[1], precision: 10);
            Assert.Equal(-0.03, output[2], precision: 10);
        }

        [Fact]
        public void ELUActivation_PositiveValues_ReturnsInput()
        {
            // Arrange
            var elu = new ELUActivation<double>(alpha: 1.0);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = elu.Forward(input);

            // Assert - Positive values pass through
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(input[i], output[i], precision: 10);
            }
        }

        [Fact]
        public void ELUActivation_NegativeValues_ProducesExpCurve()
        {
            // Arrange
            var alpha = 1.0;
            var elu = new ELUActivation<double>(alpha);
            var input = new Tensor<double>(new[] { 1 });
            input[0] = -1.0;

            // Act
            var output = elu.Forward(input);

            // Assert - ELU(-1) = alpha * (e^-1 - 1) ≈ -0.6321
            Assert.Equal(-0.6321205588, output[0], precision: 8);
        }

        [Fact]
        public void GELUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var gelu = new GELUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = gelu.Forward(input);

            // Assert - GELU(0) = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void GELUActivation_PositiveValues_ProducesSmootherThanReLU()
        {
            // Arrange
            var gelu = new GELUActivation<double>();
            var relu = new ReLUActivation<double>();

            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.5;

            // Act
            var geluOutput = gelu.Forward(input);
            var reluOutput = relu.Forward(input);

            // Assert - GELU is smooth, should be slightly less than input for small positive values
            Assert.True(geluOutput[0] > 0.0);
            Assert.True(geluOutput[0] < reluOutput[0]); // GELU is slightly lower than ReLU for small values
        }

        [Fact]
        public void ActivationFunctions_ChainMultipleTimes_ProduceConsistentResults()
        {
            // Arrange
            var relu = new ReLUActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -2.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 2.0;

            // Act - Apply ReLU multiple times (should be idempotent)
            var output1 = relu.Forward(input);
            var output2 = relu.Forward(output1);
            var output3 = relu.Forward(output2);

            // Assert - Results should be identical
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(output1[i], output2[i], precision: 10);
                Assert.Equal(output2[i], output3[i], precision: 10);
            }
        }

        [Fact]
        public void ActivationFunctions_WithFloatType_WorkCorrectly()
        {
            // Arrange
            var sigmoid = new SigmoidActivation<float>();
            var input = new Tensor<float>(new[] { 1 });
            input[0] = 0.0f;

            // Act
            var output = sigmoid.Forward(input);

            // Assert
            Assert.Equal(0.5f, output[0], precision: 6);
        }
    }
}
