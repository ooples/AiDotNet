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

        // ===== BinarySpikingActivation Tests =====

        [Fact]
        public void BinarySpikingActivation_InputAboveThreshold_ReturnsOne()
        {
            // Arrange
            var activation = new BinarySpikingActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.5; input[1] = 2.0; input[2] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All inputs are above default threshold of 1.0
            Assert.Equal(1.0, output[0], precision: 10);
            Assert.Equal(1.0, output[1], precision: 10);
            Assert.Equal(1.0, output[2], precision: 10);
        }

        [Fact]
        public void BinarySpikingActivation_InputBelowThreshold_ReturnsZero()
        {
            // Arrange
            var activation = new BinarySpikingActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 0.5; input[1] = 0.0; input[2] = -1.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All inputs are below default threshold of 1.0
            Assert.Equal(0.0, output[0], precision: 10);
            Assert.Equal(0.0, output[1], precision: 10);
            Assert.Equal(0.0, output[2], precision: 10);
        }

        [Fact]
        public void BinarySpikingActivation_InputAtThreshold_ReturnsOne()
        {
            // Arrange
            var activation = new BinarySpikingActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 1.0; // Exactly at threshold

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(1.0, output[0], precision: 10);
        }

        [Fact]
        public void BinarySpikingActivation_CustomThreshold_WorksCorrectly()
        {
            // Arrange
            var activation = new BinarySpikingActivation<double>(threshold: 2.0, derivativeSlope: 1.0, derivativeWidth: 0.2);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.5; input[1] = 2.5; input[2] = 2.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10); // Below threshold
            Assert.Equal(1.0, output[1], precision: 10); // Above threshold
            Assert.Equal(1.0, output[2], precision: 10); // At threshold
        }

        [Fact]
        public void BinarySpikingActivation_MixedValues_ProducesBinaryPattern()
        {
            // Arrange
            var activation = new BinarySpikingActivation<double>();
            var input = new Tensor<double>(new[] { 6 });
            input[0] = -2.0; input[1] = 0.5; input[2] = 1.0;
            input[3] = 1.5; input[4] = 3.0; input[5] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
            Assert.Equal(0.0, output[1], precision: 10);
            Assert.Equal(1.0, output[2], precision: 10);
            Assert.Equal(1.0, output[3], precision: 10);
            Assert.Equal(1.0, output[4], precision: 10);
            Assert.Equal(0.0, output[5], precision: 10);
        }

        // ===== BentIdentityActivation Tests =====

        [Fact]
        public void BentIdentityActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new BentIdentityActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = ((sqrt(1) - 1) / 2) + 0 = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void BentIdentityActivation_PositiveValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new BentIdentityActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(1) = ((sqrt(2) - 1) / 2) + 1 ≈ 1.2071
            Assert.Equal(1.2071067812, output[0], precision: 8);
            // f(2) = ((sqrt(5) - 1) / 2) + 2 ≈ 2.6180
            Assert.Equal(2.6180339887, output[1], precision: 8);
            // f(3) = ((sqrt(10) - 1) / 2) + 3 ≈ 4.0811
            Assert.Equal(4.0811388301, output[2], precision: 8);
        }

        [Fact]
        public void BentIdentityActivation_NegativeValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new BentIdentityActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -3.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(-1) = ((sqrt(2) - 1) / 2) - 1 ≈ -0.7929
            Assert.Equal(-0.7928932188, output[0], precision: 8);
            // f(-2) = ((sqrt(5) - 1) / 2) - 2 ≈ -1.3820
            Assert.Equal(-1.3819660113, output[1], precision: 8);
            // f(-3) = ((sqrt(10) - 1) / 2) - 3 ≈ -1.9189
            Assert.Equal(-1.9188611699, output[2], precision: 8);
        }

        [Fact]
        public void BentIdentityActivation_LargePositiveValue_ApproximatesLinear()
        {
            // Arrange
            var activation = new BentIdentityActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 100.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For large x, f(x) ≈ x + (x/2) = 1.5x
            Assert.True(output[0] > 149.0 && output[0] < 151.0);
        }

        [Fact]
        public void BentIdentityActivation_Derivative_AlwaysPositive()
        {
            // Arrange
            var activation = new BentIdentityActivation<double>();
            var input = new Vector<double>(5);
            input[0] = -2.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 2.0;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative should always be positive
            for (int i = 0; i < 5; i++)
            {
                Assert.True(jacobian[i, i] > 0.0);
            }
        }

        // ===== CELUActivation Tests =====

        [Fact]
        public void CELUActivation_PositiveValues_ReturnsInput()
        {
            // Arrange
            var activation = new CELUActivation<double>(alpha: 1.0);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Positive values pass through
            Assert.Equal(1.0, output[0], precision: 10);
            Assert.Equal(2.0, output[1], precision: 10);
            Assert.Equal(3.0, output[2], precision: 10);
        }

        [Fact]
        public void CELUActivation_NegativeValues_ProducesExpCurve()
        {
            // Arrange
            var alpha = 1.0;
            var activation = new CELUActivation<double>(alpha);
            var input = new Tensor<double>(new[] { 2 });
            input[0] = -1.0; input[1] = -2.0;

            // Act
            var output = activation.Forward(input);

            // Assert - CELU(x) = alpha * (exp(x/alpha) - 1) for x < 0
            // f(-1) = 1.0 * (e^-1 - 1) ≈ -0.6321
            Assert.Equal(-0.6321205588, output[0], precision: 8);
            // f(-2) = 1.0 * (e^-2 - 1) ≈ -0.8647
            Assert.Equal(-0.8646647168, output[1], precision: 8);
        }

        [Fact]
        public void CELUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new CELUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void CELUActivation_DifferentAlpha_ChangesNegativeSaturation()
        {
            // Arrange
            var activation1 = new CELUActivation<double>(alpha: 0.5);
            var activation2 = new CELUActivation<double>(alpha: 2.0);
            var input = new Tensor<double>(new[] { 1 });
            input[0] = -1.0;

            // Act
            var output1 = activation1.Forward(input);
            var output2 = activation2.Forward(input);

            // Assert - Larger alpha allows more negative values
            Assert.True(output2[0] < output1[0]);
        }

        [Fact]
        public void CELUActivation_Derivative_PositiveInputsIsOne()
        {
            // Arrange
            var activation = new CELUActivation<double>();
            var input = new Vector<double>(3);
            input[0] = 1.0; input[1] = 2.0; input[2] = 5.0;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative for positive inputs is 1
            Assert.Equal(1.0, jacobian[0, 0], precision: 10);
            Assert.Equal(1.0, jacobian[1, 1], precision: 10);
            Assert.Equal(1.0, jacobian[2, 2], precision: 10);
        }

        // ===== IdentityActivation Tests =====

        [Fact]
        public void IdentityActivation_ReturnsInputUnchanged()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -2.5; input[1] = -1.0; input[2] = 0.0; input[3] = 1.5; input[4] = 3.7;

            // Act
            var output = activation.Forward(input);

            // Assert - All values remain unchanged
            Assert.Equal(-2.5, output[0], precision: 10);
            Assert.Equal(-1.0, output[1], precision: 10);
            Assert.Equal(0.0, output[2], precision: 10);
            Assert.Equal(1.5, output[3], precision: 10);
            Assert.Equal(3.7, output[4], precision: 10);
        }

        [Fact]
        public void IdentityActivation_Derivative_AlwaysOne()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Vector<double>(4);
            input[0] = -10.0; input[1] = 0.0; input[2] = 5.0; input[3] = 100.0;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative is always 1
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1.0, jacobian[i, i], precision: 10);
            }
        }

        [Fact]
        public void IdentityActivation_LargeValues_PassThrough()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1000000.0; input[1] = -1000000.0; input[2] = 0.0000001;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(1000000.0, output[0], precision: 10);
            Assert.Equal(-1000000.0, output[1], precision: 10);
            Assert.Equal(0.0000001, output[2], precision: 10);
        }

        // ===== HardTanhActivation Tests =====

        [Fact]
        public void HardTanhActivation_WithinBounds_ReturnsInput()
        {
            // Arrange
            var activation = new HardTanhActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -0.5; input[1] = -0.2; input[2] = 0.0; input[3] = 0.3; input[4] = 0.8;

            // Act
            var output = activation.Forward(input);

            // Assert - Values between -1 and 1 pass through
            Assert.Equal(-0.5, output[0], precision: 10);
            Assert.Equal(-0.2, output[1], precision: 10);
            Assert.Equal(0.0, output[2], precision: 10);
            Assert.Equal(0.3, output[3], precision: 10);
            Assert.Equal(0.8, output[4], precision: 10);
        }

        [Fact]
        public void HardTanhActivation_BelowLowerBound_ReturnsMinusOne()
        {
            // Arrange
            var activation = new HardTanhActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.5; input[1] = -2.0; input[2] = -100.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All values clipped to -1
            Assert.Equal(-1.0, output[0], precision: 10);
            Assert.Equal(-1.0, output[1], precision: 10);
            Assert.Equal(-1.0, output[2], precision: 10);
        }

        [Fact]
        public void HardTanhActivation_AboveUpperBound_ReturnsOne()
        {
            // Arrange
            var activation = new HardTanhActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.5; input[1] = 2.0; input[2] = 100.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All values clipped to 1
            Assert.Equal(1.0, output[0], precision: 10);
            Assert.Equal(1.0, output[1], precision: 10);
            Assert.Equal(1.0, output[2], precision: 10);
        }

        [Fact]
        public void HardTanhActivation_AtBoundaries_ReturnsExactValues()
        {
            // Arrange
            var activation = new HardTanhActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = -1.0; input[1] = 1.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(-1.0, output[0], precision: 10);
            Assert.Equal(1.0, output[1], precision: 10);
        }

        [Fact]
        public void HardTanhActivation_Derivative_InsideBoundsIsOne()
        {
            // Arrange
            var activation = new HardTanhActivation<double>();
            var input = new Vector<double>(3);
            input[0] = -0.5; input[1] = 0.0; input[2] = 0.5;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative is 1 inside bounds
            Assert.Equal(1.0, jacobian[0, 0], precision: 10);
            Assert.Equal(1.0, jacobian[1, 1], precision: 10);
            Assert.Equal(1.0, jacobian[2, 2], precision: 10);
        }

        [Fact]
        public void HardTanhActivation_Derivative_OutsideBoundsIsZero()
        {
            // Arrange
            var activation = new HardTanhActivation<double>();
            var input = new Vector<double>(4);
            input[0] = -2.0; input[1] = -1.5; input[2] = 1.5; input[3] = 2.0;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative is 0 outside bounds
            Assert.Equal(0.0, jacobian[0, 0], precision: 10);
            Assert.Equal(0.0, jacobian[1, 1], precision: 10);
            Assert.Equal(0.0, jacobian[2, 2], precision: 10);
            Assert.Equal(0.0, jacobian[3, 3], precision: 10);
        }

        // ===== GaussianActivation Tests =====

        [Fact]
        public void GaussianActivation_ZeroInput_ReturnsOne()
        {
            // Arrange
            var activation = new GaussianActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Gaussian(0) = exp(0) = 1
            Assert.Equal(1.0, output[0], precision: 10);
        }

        [Fact]
        public void GaussianActivation_SymmetricInputs_ProduceSameOutput()
        {
            // Arrange
            var activation = new GaussianActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 2.0; input[1] = -2.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(x) = f(-x) for Gaussian
            Assert.Equal(output[0], output[1], precision: 10);
        }

        [Fact]
        public void GaussianActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new GaussianActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(1) = exp(-1) ≈ 0.3679
            Assert.Equal(0.3678794412, output[0], precision: 8);
            // f(2) = exp(-4) ≈ 0.0183
            Assert.Equal(0.0183156389, output[1], precision: 8);
            // f(3) = exp(-9) ≈ 0.0001
            Assert.Equal(0.0001234098, output[2], precision: 8);
        }

        [Fact]
        public void GaussianActivation_OutputBetweenZeroAndOne()
        {
            // Arrange
            var activation = new GaussianActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -5.0; input[1] = -2.0; input[2] = 0.0; input[3] = 2.0; input[4] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs between 0 and 1
            for (int i = 0; i < 5; i++)
            {
                Assert.True(output[i] >= 0.0 && output[i] <= 1.0);
            }
        }

        [Fact]
        public void GaussianActivation_LargeInputs_ApproachZero()
        {
            // Arrange
            var activation = new GaussianActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 10.0; input[1] = -10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Very small values approaching 0
            Assert.True(output[0] < 0.00001);
            Assert.True(output[1] < 0.00001);
        }

        // ===== HardSigmoidActivation Tests =====

        [Fact]
        public void HardSigmoidActivation_ZeroInput_ReturnsHalf()
        {
            // Arrange
            var activation = new HardSigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = (0 + 1) / 2 = 0.5
            Assert.Equal(0.5, output[0], precision: 10);
        }

        [Fact]
        public void HardSigmoidActivation_WithinRange_LinearTransformation()
        {
            // Arrange
            var activation = new HardSigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -0.5; input[1] = 0.0; input[2] = 0.5;

            // Act
            var output = activation.Forward(input);

            // Assert - Linear between -1 and 1
            Assert.Equal(0.25, output[0], precision: 10); // (-0.5 + 1) / 2
            Assert.Equal(0.50, output[1], precision: 10); // (0 + 1) / 2
            Assert.Equal(0.75, output[2], precision: 10); // (0.5 + 1) / 2
        }

        [Fact]
        public void HardSigmoidActivation_BelowRange_ReturnsZero()
        {
            // Arrange
            var activation = new HardSigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -2.0; input[1] = -5.0; input[2] = -10.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
            Assert.Equal(0.0, output[1], precision: 10);
            Assert.Equal(0.0, output[2], precision: 10);
        }

        [Fact]
        public void HardSigmoidActivation_AboveRange_ReturnsOne()
        {
            // Arrange
            var activation = new HardSigmoidActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 2.0; input[1] = 5.0; input[2] = 10.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(1.0, output[0], precision: 10);
            Assert.Equal(1.0, output[1], precision: 10);
            Assert.Equal(1.0, output[2], precision: 10);
        }

        [Fact]
        public void HardSigmoidActivation_Derivative_InsideRangeIsHalf()
        {
            // Arrange
            var activation = new HardSigmoidActivation<double>();
            var input = new Vector<double>(3);
            input[0] = -0.5; input[1] = 0.0; input[2] = 0.5;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative is 0.5 inside range
            Assert.Equal(0.5, jacobian[0, 0], precision: 10);
            Assert.Equal(0.5, jacobian[1, 1], precision: 10);
            Assert.Equal(0.5, jacobian[2, 2], precision: 10);
        }

        // ===== ISRUActivation Tests =====

        [Fact]
        public void ISRUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new ISRUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = 0 / sqrt(1) = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void ISRUActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new ISRUActivation<double>(alpha: 1.0);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = -1.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(1) = 1 / sqrt(1 + 1) = 1 / sqrt(2) ≈ 0.7071
            Assert.Equal(0.7071067812, output[0], precision: 8);
            // f(2) = 2 / sqrt(1 + 4) = 2 / sqrt(5) ≈ 0.8944
            Assert.Equal(0.8944271910, output[1], precision: 8);
            // f(-1) = -1 / sqrt(1 + 1) = -1 / sqrt(2) ≈ -0.7071
            Assert.Equal(-0.7071067812, output[2], precision: 8);
        }

        [Fact]
        public void ISRUActivation_OutputBounded()
        {
            // Arrange
            var activation = new ISRUActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -10.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs between -1 and 1
            for (int i = 0; i < 5; i++)
            {
                Assert.True(output[i] >= -1.0 && output[i] <= 1.0);
            }
        }

        [Fact]
        public void ISRUActivation_LargeInputs_ApproachUnitBounds()
        {
            // Arrange
            var activation = new ISRUActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 100.0; input[1] = -100.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Approach ±1
            Assert.True(output[0] > 0.99);
            Assert.True(output[1] < -0.99);
        }

        // ===== HierarchicalSoftmaxActivation Tests =====

        [Fact]
        public void HierarchicalSoftmaxActivation_OutputSumsToApproximatelyOne()
        {
            // Arrange
            var activation = new HierarchicalSoftmaxActivation<double>(numClasses: 4);
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 0.5;

            // Act
            var output = activation.Activate(input);

            // Assert - Outputs should approximately sum to 1
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }
            Assert.True(Math.Abs(sum - 1.0) < 0.1); // Allow some tolerance
        }

        [Fact]
        public void HierarchicalSoftmaxActivation_AllOutputsPositive()
        {
            // Arrange
            var activation = new HierarchicalSoftmaxActivation<double>(numClasses: 8);
            var input = new Vector<double>(8);
            for (int i = 0; i < 8; i++)
            {
                input[i] = i - 4.0; // Mix of negative and positive
            }

            // Act
            var output = activation.Activate(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] >= 0.0);
            }
        }

        // ===== LogSoftmaxActivation Tests =====

        [Fact]
        public void LogSoftmaxActivation_OutputsAreNegative()
        {
            // Arrange
            var activation = new LogSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Log of probabilities should be negative or zero
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] <= 0.0);
            }
        }

        [Fact]
        public void LogSoftmaxActivation_ExponentialSumsToOne()
        {
            // Arrange
            var activation = new LogSoftmaxActivation<double>();
            var input = new Vector<double>(3);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Activate(input);

            // Assert - exp(log_softmax) should sum to 1
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += Math.Exp(output[i]);
            }
            Assert.Equal(1.0, sum, precision: 8);
        }

        [Fact]
        public void LogSoftmaxActivation_LargestInputHasLeastNegativeOutput()
        {
            // Arrange
            var activation = new LogSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 5.0; input[2] = 2.0; input[3] = 1.5;

            // Act
            var output = activation.Activate(input);

            // Assert - Index 1 has largest input, should have least negative (closest to 0) output
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
            Assert.Equal(1, maxIndex);
        }

        // ===== LogSoftminActivation Tests =====

        [Fact]
        public void LogSoftminActivation_OutputsAreNegative()
        {
            // Arrange
            var activation = new LogSoftminActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Log of probabilities should be negative or zero
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] <= 0.0);
            }
        }

        [Fact]
        public void LogSoftminActivation_SmallestInputHasLeastNegativeOutput()
        {
            // Arrange
            var activation = new LogSoftminActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 5.0; input[1] = 1.0; input[2] = 3.0; input[3] = 2.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Index 1 has smallest input, should have least negative output
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
            Assert.Equal(1, maxIndex);
        }

        // ===== LiSHTActivation Tests =====

        [Fact]
        public void LiSHTActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new LiSHTActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = 0 * tanh(0) = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void LiSHTActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new LiSHTActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = -1.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(1) = 1 * tanh(1) ≈ 0.7616
            Assert.Equal(0.7615941560, output[0], precision: 8);
            // f(2) = 2 * tanh(2) ≈ 1.9281
            Assert.Equal(1.9280552448, output[1], precision: 8);
            // f(-1) = -1 * tanh(-1) ≈ 0.7616 (note: positive result)
            Assert.Equal(0.7615941560, output[2], precision: 8);
        }

        [Fact]
        public void LiSHTActivation_PositiveInputs_ProducePositiveOutputs()
        {
            // Arrange
            var activation = new LiSHTActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 0.5; input[1] = 1.0; input[2] = 2.0; input[3] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < 4; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        [Fact]
        public void LiSHTActivation_NegativeInputs_ProducePositiveOutputs()
        {
            // Arrange
            var activation = new LiSHTActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -0.5; input[1] = -1.0; input[2] = -2.0;

            // Act
            var output = activation.Forward(input);

            // Assert - LiSHT is symmetric, so negative inputs also produce positive outputs
            for (int i = 0; i < 3; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        // ===== MaxoutActivation Tests =====

        [Fact]
        public void MaxoutActivation_TwoPieces_SelectsMaximum()
        {
            // Arrange
            var activation = new MaxoutActivation<double>(numPieces: 2);
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 3.0; input[2] = 2.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Should select max from each pair
            Assert.Equal(2, output.Length);
            Assert.Equal(3.0, output[0], precision: 10); // max(1.0, 3.0)
            Assert.Equal(4.0, output[1], precision: 10); // max(2.0, 4.0)
        }

        [Fact]
        public void MaxoutActivation_ThreePieces_SelectsMaximum()
        {
            // Arrange
            var activation = new MaxoutActivation<double>(numPieces: 3);
            var input = new Vector<double>(6);
            input[0] = 1.0; input[1] = 5.0; input[2] = 2.0;
            input[3] = 3.0; input[4] = 1.5; input[5] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert
            Assert.Equal(2, output.Length);
            Assert.Equal(5.0, output[0], precision: 10); // max(1.0, 5.0, 2.0)
            Assert.Equal(4.0, output[1], precision: 10); // max(3.0, 1.5, 4.0)
        }

        [Fact]
        public void MaxoutActivation_NegativeValues_SelectsLeastNegative()
        {
            // Arrange
            var activation = new MaxoutActivation<double>(numPieces: 2);
            var input = new Vector<double>(4);
            input[0] = -5.0; input[1] = -2.0; input[2] = -10.0; input[3] = -3.0;

            // Act
            var output = activation.Activate(input);

            // Assert
            Assert.Equal(2, output.Length);
            Assert.Equal(-2.0, output[0], precision: 10); // max(-5.0, -2.0)
            Assert.Equal(-3.0, output[1], precision: 10); // max(-10.0, -3.0)
        }

        // ===== MishActivation Tests =====

        [Fact]
        public void MishActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new MishActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) ≈ 0
            Assert.True(Math.Abs(output[0]) < 0.01); // Should be very close to 0
        }

        [Fact]
        public void MishActivation_LargePositiveValues_ApproachLinear()
        {
            // Arrange
            var activation = new MishActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 10.0; input[1] = 20.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For large x, Mish(x) ≈ x
            Assert.True(output[0] > 9.9);
            Assert.True(output[1] > 19.9);
        }

        [Fact]
        public void MishActivation_NegativeValues_Damped()
        {
            // Arrange
            var activation = new MishActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Negative values are dampened but not zeroed
            for (int i = 0; i < 3; i++)
            {
                Assert.True(output[i] < 0.0); // Still negative
                Assert.True(output[i] > input[i]); // But less negative than input
            }
        }

        [Fact]
        public void MishActivation_Smooth_NoDicsontinuities()
        {
            // Arrange
            var activation = new MishActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -0.1; input[1] = -0.01; input[2] = 0.0; input[3] = 0.01; input[4] = 0.1;

            // Act
            var output = activation.Forward(input);

            // Assert - Check smooth transition around zero
            for (int i = 0; i < 4; i++)
            {
                var diff = Math.Abs(output[i + 1] - output[i]);
                Assert.True(diff < 0.1); // Gradual change
            }
        }

        // ===== PReLUActivation Tests =====

        [Fact]
        public void PReLUActivation_PositiveValues_ReturnInput()
        {
            // Arrange
            var activation = new PReLUActivation<double>(alpha: 0.01);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Positive values pass through
            Assert.Equal(1.0, output[0], precision: 10);
            Assert.Equal(2.0, output[1], precision: 10);
            Assert.Equal(5.0, output[2], precision: 10);
        }

        [Fact]
        public void PReLUActivation_NegativeValues_ScaledByAlpha()
        {
            // Arrange
            var alpha = 0.01;
            var activation = new PReLUActivation<double>(alpha);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -5.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(-0.01, output[0], precision: 10);
            Assert.Equal(-0.02, output[1], precision: 10);
            Assert.Equal(-0.05, output[2], precision: 10);
        }

        [Fact]
        public void PReLUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new PReLUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void PReLUActivation_DifferentAlpha_ProducesDifferentScaling()
        {
            // Arrange
            var activation1 = new PReLUActivation<double>(alpha: 0.01);
            var activation2 = new PReLUActivation<double>(alpha: 0.1);
            var input = new Tensor<double>(new[] { 1 });
            input[0] = -10.0;

            // Act
            var output1 = activation1.Forward(input);
            var output2 = activation2.Forward(input);

            // Assert
            Assert.Equal(-0.1, output1[0], precision: 10);
            Assert.Equal(-1.0, output2[0], precision: 10);
        }

        // ===== RReLUActivation Tests =====

        [Fact]
        public void RReLUActivation_PositiveValues_ReturnInput()
        {
            // Arrange
            var activation = new RReLUActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Positive values pass through
            Assert.Equal(1.0, output[0], precision: 10);
            Assert.Equal(2.0, output[1], precision: 10);
            Assert.Equal(5.0, output[2], precision: 10);
        }

        [Fact]
        public void RReLUActivation_NegativeValues_ScaledRandomly()
        {
            // Arrange
            var activation = new RReLUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = -10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Should be scaled, between -10*upperBound and -10*lowerBound
            Assert.True(output[0] < 0.0); // Still negative
            Assert.True(output[0] > -10.0); // But less negative
            Assert.True(output[0] <= -10.0 * (1.0 / 8.0)); // Within bounds
        }

        [Fact]
        public void RReLUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new RReLUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
        }

        // ===== SELUActivation Tests =====

        [Fact]
        public void SELUActivation_PositiveValues_ScaledByLambda()
        {
            // Arrange
            var activation = new SELUActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For x >= 0: SELU(x) = lambda * x
            var lambda = 1.0507009873554804934193349852946;
            Assert.Equal(lambda * 1.0, output[0], precision: 8);
            Assert.Equal(lambda * 2.0, output[1], precision: 8);
            Assert.Equal(lambda * 3.0, output[2], precision: 8);
        }

        [Fact]
        public void SELUActivation_NegativeValues_ExponentialCurve()
        {
            // Arrange
            var activation = new SELUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = -1.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For x < 0: SELU(x) = lambda * alpha * (e^x - 1)
            var lambda = 1.0507009873554804934193349852946;
            var alpha = 1.6732632423543772848170429916717;
            var expected = lambda * alpha * (Math.Exp(-1.0) - 1.0);
            Assert.Equal(expected, output[0], precision: 8);
        }

        [Fact]
        public void SELUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new SELUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void SELUActivation_MixedValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new SELUActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -2.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 2.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.True(output[0] < 0.0); // Negative inputs produce negative outputs
            Assert.True(output[1] < 0.0);
            Assert.Equal(0.0, output[2], precision: 10);
            Assert.True(output[3] > 0.0); // Positive inputs produce positive outputs
            Assert.True(output[4] > 0.0);
        }

        // ===== SignActivation Tests =====

        [Fact]
        public void SignActivation_PositiveValues_ReturnsOne()
        {
            // Arrange
            var activation = new SignActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 0.1; input[1] = 1.0; input[2] = 5.0; input[3] = 100.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1.0, output[i], precision: 10);
            }
        }

        [Fact]
        public void SignActivation_NegativeValues_ReturnsMinusOne()
        {
            // Arrange
            var activation = new SignActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = -0.1; input[1] = -1.0; input[2] = -5.0; input[3] = -100.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(-1.0, output[i], precision: 10);
            }
        }

        [Fact]
        public void SignActivation_ZeroValue_ReturnsZero()
        {
            // Arrange
            var activation = new SignActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void SignActivation_MixedValues_ProducesCorrectSigns()
        {
            // Arrange
            var activation = new SignActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -5.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(-1.0, output[0], precision: 10);
            Assert.Equal(-1.0, output[1], precision: 10);
            Assert.Equal(0.0, output[2], precision: 10);
            Assert.Equal(1.0, output[3], precision: 10);
            Assert.Equal(1.0, output[4], precision: 10);
        }

        [Fact]
        public void SignActivation_Derivative_AlwaysZero()
        {
            // Arrange
            var activation = new SignActivation<double>();
            var input = new Vector<double>(5);
            input[0] = -5.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 5.0;

            // Act
            var jacobian = activation.Backward(input);

            // Assert - Derivative is always 0
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(0.0, jacobian[i, i], precision: 10);
            }
        }

        // ===== SiLUActivation Tests =====

        [Fact]
        public void SiLUActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new SiLUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void SiLUActivation_PositiveValues_ProducesPositiveOutputs()
        {
            // Arrange
            var activation = new SiLUActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 0.5; input[1] = 1.0; input[2] = 2.0; input[3] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < 4; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        [Fact]
        public void SiLUActivation_LargePositive_ApproachesLinear()
        {
            // Arrange
            var activation = new SiLUActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For large x, SiLU(x) ≈ x
            Assert.True(output[0] > 9.9);
        }

        [Fact]
        public void SiLUActivation_NegativeValues_AllowSomeThrough()
        {
            // Arrange
            var activation = new SiLUActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - SiLU allows some negative values through (unlike ReLU)
            for (int i = 0; i < 3; i++)
            {
                Assert.True(output[i] < 0.0); // Still negative
                Assert.True(output[i] > input[i]); // But less negative than input
            }
        }

        // ===== SoftPlusActivation Tests =====

        [Fact]
        public void SoftPlusActivation_ZeroInput_ReturnsLnTwo()
        {
            // Arrange
            var activation = new SoftPlusActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = ln(1 + e^0) = ln(2) ≈ 0.6931
            Assert.Equal(Math.Log(2.0), output[0], precision: 8);
        }

        [Fact]
        public void SoftPlusActivation_LargePositive_ApproximatesInput()
        {
            // Arrange
            var activation = new SoftPlusActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 10.0; input[1] = 20.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For large x, softplus(x) ≈ x
            Assert.True(Math.Abs(output[0] - 10.0) < 0.01);
            Assert.True(Math.Abs(output[1] - 20.0) < 0.01);
        }

        [Fact]
        public void SoftPlusActivation_NegativeValues_ApproachZero()
        {
            // Arrange
            var activation = new SoftPlusActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = -10.0; input[1] = -20.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For large negative x, softplus(x) ≈ 0
            Assert.True(output[0] < 0.001);
            Assert.True(output[1] < 0.000001);
        }

        [Fact]
        public void SoftPlusActivation_AlwaysPositive()
        {
            // Arrange
            var activation = new SoftPlusActivation<double>();
            var input = new Tensor<double>(new[] { 7 });
            input[0] = -10.0; input[1] = -5.0; input[2] = -1.0; input[3] = 0.0;
            input[4] = 1.0; input[5] = 5.0; input[6] = 10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < 7; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        // ===== SoftSignActivation Tests =====

        [Fact]
        public void SoftSignActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new SoftSignActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = 0 / (1 + 0) = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void SoftSignActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new SoftSignActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 1.0; input[1] = 2.0; input[2] = -1.0; input[3] = -2.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(1) = 1 / (1 + 1) = 0.5
            Assert.Equal(0.5, output[0], precision: 10);
            // f(2) = 2 / (1 + 2) = 0.6667
            Assert.Equal(0.6666666667, output[1], precision: 8);
            // f(-1) = -1 / (1 + 1) = -0.5
            Assert.Equal(-0.5, output[2], precision: 10);
            // f(-2) = -2 / (1 + 2) = -0.6667
            Assert.Equal(-0.6666666667, output[3], precision: 8);
        }

        [Fact]
        public void SoftSignActivation_OutputBounded()
        {
            // Arrange
            var activation = new SoftSignActivation<double>();
            var input = new Tensor<double>(new[] { 6 });
            input[0] = -100.0; input[1] = -10.0; input[2] = -1.0;
            input[3] = 1.0; input[4] = 10.0; input[5] = 100.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs between -1 and 1
            for (int i = 0; i < 6; i++)
            {
                Assert.True(output[i] >= -1.0 && output[i] <= 1.0);
            }
        }

        [Fact]
        public void SoftSignActivation_Symmetric()
        {
            // Arrange
            var activation = new SoftSignActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 5.0; input[1] = -5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(-x) = -f(x)
            Assert.Equal(-output[0], output[1], precision: 10);
        }

        // ===== SoftminActivation Tests =====

        [Fact]
        public void SoftminActivation_OutputSumsToOne()
        {
            // Arrange
            var activation = new SoftminActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }
            Assert.Equal(1.0, sum, precision: 10);
        }

        [Fact]
        public void SoftminActivation_SmallestValueGetsHighestProbability()
        {
            // Arrange
            var activation = new SoftminActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 5.0; input[1] = 1.0; input[2] = 3.0; input[3] = 2.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Index 1 (value 1.0) should have highest probability
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
            Assert.Equal(1, maxIndex);
        }

        [Fact]
        public void SoftminActivation_AllPositive()
        {
            // Arrange
            var activation = new SoftminActivation<double>();
            var input = new Vector<double>(5);
            input[0] = -5.0; input[1] = -2.0; input[2] = 0.0; input[3] = 2.0; input[4] = 5.0;

            // Act
            var output = activation.Activate(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        // ===== SparsemaxActivation Tests =====

        [Fact]
        public void SparsemaxActivation_OutputSumsToOne()
        {
            // Arrange
            var activation = new SparsemaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }
            Assert.Equal(1.0, sum, precision: 10);
        }

        [Fact]
        public void SparsemaxActivation_ProducesSparseOutput()
        {
            // Arrange
            var activation = new SparsemaxActivation<double>();
            var input = new Vector<double>(5);
            input[0] = 1.0; input[1] = 5.0; input[2] = 2.0; input[3] = 1.5; input[4] = 1.2;

            // Act
            var output = activation.Activate(input);

            // Assert - Some outputs should be exactly zero
            var zeroCount = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] == 0.0)
                {
                    zeroCount++;
                }
            }
            Assert.True(zeroCount > 0);
        }

        [Fact]
        public void SparsemaxActivation_LargestValuesGetNonZeroProbability()
        {
            // Arrange
            var activation = new SparsemaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 10.0; input[2] = 2.0; input[3] = 1.5;

            // Act
            var output = activation.Activate(input);

            // Assert - Largest value (index 1) should be non-zero
            Assert.True(output[1] > 0.0);
        }

        // ===== SphericalSoftmaxActivation Tests =====

        [Fact]
        public void SphericalSoftmaxActivation_OutputSumsToOne()
        {
            // Arrange
            var activation = new SphericalSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }
            Assert.Equal(1.0, sum, precision: 10);
        }

        [Fact]
        public void SphericalSoftmaxActivation_AllPositive()
        {
            // Arrange
            var activation = new SphericalSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = -2.0; input[1] = -1.0; input[2] = 1.0; input[3] = 2.0;

            // Act
            var output = activation.Activate(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        [Fact]
        public void SphericalSoftmaxActivation_LargestInputGetHighestProbability()
        {
            // Arrange
            var activation = new SphericalSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 5.0; input[2] = 2.0; input[3] = 1.5;

            // Act
            var output = activation.Activate(input);

            // Assert - Index 1 should have highest probability
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
            Assert.Equal(1, maxIndex);
        }

        // ===== SquashActivation Tests =====

        [Fact]
        public void SquashActivation_OutputMagnitudeBounded()
        {
            // Arrange
            var activation = new SquashActivation<double>();
            var input = new Vector<double>(3);
            input[0] = 10.0; input[1] = 20.0; input[2] = 30.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Output magnitude should be between 0 and 1
            var magnitude = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                magnitude += output[i] * output[i];
            }
            magnitude = Math.Sqrt(magnitude);
            Assert.True(magnitude <= 1.0);
        }

        [Fact]
        public void SquashActivation_PreservesDirection()
        {
            // Arrange
            var activation = new SquashActivation<double>();
            var input = new Vector<double>(3);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Output should point in same direction as input
            // Check if output/||output|| = input/||input||
            var inputMag = Math.Sqrt(input[0] * input[0] + input[1] * input[1] + input[2] * input[2]);
            var outputMag = Math.Sqrt(output[0] * output[0] + output[1] * output[1] + output[2] * output[2]);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(input[i] / inputMag, output[i] / outputMag, precision: 6);
            }
        }

        // ===== ScaledTanhActivation Tests =====

        [Fact]
        public void ScaledTanhActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new ScaledTanhActivation<double>(beta: 1.0);
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void ScaledTanhActivation_OutputBounded()
        {
            // Arrange
            var activation = new ScaledTanhActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -10.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs between -1 and 1
            for (int i = 0; i < 5; i++)
            {
                Assert.True(output[i] >= -1.0 && output[i] <= 1.0);
            }
        }

        [Fact]
        public void ScaledTanhActivation_Symmetric()
        {
            // Arrange
            var activation = new ScaledTanhActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 2.0; input[1] = -2.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(-x) = -f(x)
            Assert.Equal(-output[0], output[1], precision: 10);
        }

        // ===== SQRBFActivation Tests =====

        [Fact]
        public void SQRBFActivation_ZeroInput_ReturnsOne()
        {
            // Arrange
            var activation = new SQRBFActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = exp(0) = 1
            Assert.Equal(1.0, output[0], precision: 10);
        }

        [Fact]
        public void SQRBFActivation_Symmetric()
        {
            // Arrange
            var activation = new SQRBFActivation<double>();
            var input = new Tensor<double>(new[] { 2 });
            input[0] = 2.0; input[1] = -2.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(x) = f(-x)
            Assert.Equal(output[0], output[1], precision: 10);
        }

        [Fact]
        public void SQRBFActivation_KnownValues_ProducesCorrectResults()
        {
            // Arrange
            var activation = new SQRBFActivation<double>(beta: 1.0);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            // f(1) = exp(-1) ≈ 0.3679
            Assert.Equal(0.3678794412, output[0], precision: 8);
            // f(2) = exp(-4) ≈ 0.0183
            Assert.Equal(0.0183156389, output[1], precision: 8);
            // f(3) = exp(-9) ≈ 0.0001
            Assert.Equal(0.0001234098, output[2], precision: 8);
        }

        [Fact]
        public void SQRBFActivation_OutputBetweenZeroAndOne()
        {
            // Arrange
            var activation = new SQRBFActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -5.0; input[1] = -1.0; input[2] = 0.0; input[3] = 1.0; input[4] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs between 0 and 1
            for (int i = 0; i < 5; i++)
            {
                Assert.True(output[i] >= 0.0 && output[i] <= 1.0);
            }
        }

        // ===== ThresholdedReLUActivation Tests =====

        [Fact]
        public void ThresholdedReLUActivation_AboveThreshold_ReturnsInput()
        {
            // Arrange
            var activation = new ThresholdedReLUActivation<double>(theta: 1.0);
            var input = new Tensor<double>(new[] { 3 });
            input[0] = 1.5; input[1] = 2.0; input[2] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(1.5, output[0], precision: 10);
            Assert.Equal(2.0, output[1], precision: 10);
            Assert.Equal(5.0, output[2], precision: 10);
        }

        [Fact]
        public void ThresholdedReLUActivation_BelowThreshold_ReturnsZero()
        {
            // Arrange
            var activation = new ThresholdedReLUActivation<double>(theta: 1.0);
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 0.0; input[1] = 0.5; input[2] = 1.0; input[3] = -1.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10);
            Assert.Equal(0.0, output[1], precision: 10);
            Assert.Equal(0.0, output[2], precision: 10); // At threshold
            Assert.Equal(0.0, output[3], precision: 10);
        }

        [Fact]
        public void ThresholdedReLUActivation_CustomThreshold_WorksCorrectly()
        {
            // Arrange
            var activation = new ThresholdedReLUActivation<double>(theta: 2.0);
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 1.0; input[1] = 2.0; input[2] = 2.5; input[3] = 3.0;

            // Act
            var output = activation.Forward(input);

            // Assert
            Assert.Equal(0.0, output[0], precision: 10); // Below threshold
            Assert.Equal(0.0, output[1], precision: 10); // At threshold
            Assert.Equal(2.5, output[2], precision: 10); // Above threshold
            Assert.Equal(3.0, output[3], precision: 10); // Above threshold
        }

        // ===== TaylorSoftmaxActivation Tests =====

        [Fact]
        public void TaylorSoftmaxActivation_OutputSumsToOne()
        {
            // Arrange
            var activation = new TaylorSoftmaxActivation<double>(order: 2);
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }
            Assert.Equal(1.0, sum, precision: 10);
        }

        [Fact]
        public void TaylorSoftmaxActivation_AllPositive()
        {
            // Arrange
            var activation = new TaylorSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = -2.0; input[1] = -1.0; input[2] = 1.0; input[3] = 2.0;

            // Act
            var output = activation.Activate(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        [Fact]
        public void TaylorSoftmaxActivation_LargestInputGetHighestProbability()
        {
            // Arrange
            var activation = new TaylorSoftmaxActivation<double>();
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 5.0; input[2] = 2.0; input[3] = 1.5;

            // Act
            var output = activation.Activate(input);

            // Assert - Index 1 should have highest probability
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
            Assert.Equal(1, maxIndex);
        }

        // ===== SwishActivation Tests =====

        [Fact]
        public void SwishActivation_ZeroInput_ReturnsZero()
        {
            // Arrange
            var activation = new SwishActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 0.0;

            // Act
            var output = activation.Forward(input);

            // Assert - f(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
            Assert.Equal(0.0, output[0], precision: 10);
        }

        [Fact]
        public void SwishActivation_PositiveValues_ProducePositiveOutputs()
        {
            // Arrange
            var activation = new SwishActivation<double>();
            var input = new Tensor<double>(new[] { 4 });
            input[0] = 0.5; input[1] = 1.0; input[2] = 2.0; input[3] = 5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < 4; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }

        [Fact]
        public void SwishActivation_LargePositive_ApproachesLinear()
        {
            // Arrange
            var activation = new SwishActivation<double>();
            var input = new Tensor<double>(new[] { 1 });
            input[0] = 10.0;

            // Act
            var output = activation.Forward(input);

            // Assert - For large x, Swish(x) ≈ x
            Assert.True(output[0] > 9.9);
        }

        [Fact]
        public void SwishActivation_NegativeValues_Damped()
        {
            // Arrange
            var activation = new SwishActivation<double>();
            var input = new Tensor<double>(new[] { 3 });
            input[0] = -1.0; input[1] = -2.0; input[2] = -5.0;

            // Act
            var output = activation.Forward(input);

            // Assert - Swish allows some negative values through
            for (int i = 0; i < 3; i++)
            {
                Assert.True(output[i] < 0.0); // Still negative
                Assert.True(output[i] > input[i]); // But less negative than input
            }
        }

        [Fact]
        public void SwishActivation_Smooth_NoDicsontinuities()
        {
            // Arrange
            var activation = new SwishActivation<double>();
            var input = new Tensor<double>(new[] { 5 });
            input[0] = -0.1; input[1] = -0.01; input[2] = 0.0; input[3] = 0.01; input[4] = 0.1;

            // Act
            var output = activation.Forward(input);

            // Assert - Check smooth transition around zero
            for (int i = 0; i < 4; i++)
            {
                var diff = Math.Abs(output[i + 1] - output[i]);
                Assert.True(diff < 0.1); // Gradual change
            }
        }

        // ===== GumbelSoftmaxActivation Tests =====

        [Fact]
        public void GumbelSoftmaxActivation_OutputSumsToApproximatelyOne()
        {
            // Arrange - Use fixed seed for reproducibility
            var activation = new GumbelSoftmaxActivation<double>(temperature: 1.0, seed: 42);
            var input = new Vector<double>(4);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0; input[3] = 4.0;

            // Act
            var output = activation.Activate(input);

            // Assert - Outputs should sum to 1
            var sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }
            Assert.Equal(1.0, sum, precision: 8);
        }

        [Fact]
        public void GumbelSoftmaxActivation_AllOutputsPositive()
        {
            // Arrange
            var activation = new GumbelSoftmaxActivation<double>(seed: 42);
            var input = new Vector<double>(4);
            input[0] = -2.0; input[1] = -1.0; input[2] = 1.0; input[3] = 2.0;

            // Act
            var output = activation.Activate(input);

            // Assert - All outputs should be positive
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] > 0.0);
            }
        }
    }
}
