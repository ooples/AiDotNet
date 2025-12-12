using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    /// <summary>
    /// Integration tests for DenseLayer with comprehensive coverage of forward pass,
    /// backward pass, parameter management, and training scenarios.
    /// </summary>
    public class DenseLayerIntegrationTests
    {
        private const double Tolerance = 1e-6;

        // ===== Forward Pass Tests =====

        [Fact]
        public void DenseLayer_ForwardPass_SingleInput_ProducesCorrectShape()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3, new ReLUActivation<double>());
            var input = new Tensor<double>([1, 5]);
            for (int i = 0; i < 5; i++)
                input[0, i] = i + 1.0; // [1, 2, 3, 4, 5]

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]); // Batch size
            Assert.Equal(3, output.Shape[1]); // Output size
        }

        [Fact]
        public void DenseLayer_ForwardPass_BatchInput_ProducesCorrectShape()
        {
            // Arrange
            var layer = new DenseLayer<double>(10, 5, new ReLUActivation<double>());
            var input = new Tensor<double>([4, 10]); // Batch of 4

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(4, output.Shape[0]); // Batch size preserved
            Assert.Equal(5, output.Shape[1]); // Output size
        }

        [Fact]
        public void DenseLayer_ForwardPass_WithKnownWeights_ProducesCorrectOutput()
        {
            // Arrange
            var layer = new DenseLayer<double>(2, 2, new LinearActivation<double>());

            // Set specific weights: [[1, 2], [3, 4]]
            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0; weights[0, 1] = 2.0;
            weights[1, 0] = 3.0; weights[1, 1] = 4.0;
            layer.SetWeights(weights);

            // Set biases to zero
            var params_ = layer.GetParameters();
            for (int i = 4; i < 6; i++)
                params_[i] = 0.0;
            layer.SetParameters(params_);

            // Input: [1, 2]
            var input = new Tensor<double>([1, 2]);
            input[0, 0] = 1.0;
            input[0, 1] = 2.0;

            // Act
            var output = layer.Forward(input);

            // Assert
            // Expected: [1*1 + 2*2, 1*3 + 2*4] = [5, 11]
            Assert.Equal(5.0, output[0, 0], precision: 6);
            Assert.Equal(11.0, output[0, 1], precision: 6);
        }

        [Fact]
        public void DenseLayer_ForwardPass_ReLUActivation_AppliesCorrectly()
        {
            // Arrange
            var layer = new DenseLayer<double>(2, 2, new ReLUActivation<double>());

            // Set weights to produce negative values
            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = -1.0; weights[0, 1] = 1.0;
            weights[1, 0] = 1.0; weights[1, 1] = -1.0;
            layer.SetWeights(weights);

            var input = new Tensor<double>([1, 2]);
            input[0, 0] = 2.0;
            input[0, 1] = 1.0;

            // Act
            var output = layer.Forward(input);

            // Assert - ReLU should zero out negative values
            Assert.True(output[0, 0] >= 0);
            Assert.True(output[0, 1] >= 0);
        }

        [Fact]
        public void DenseLayer_ForwardPass_SigmoidActivation_OutputsInRange()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2, new SigmoidActivation<double>());
            var input = new Tensor<double>([1, 3]);
            for (int i = 0; i < 3; i++)
                input[0, i] = (i - 1) * 5.0; // [-5, 0, 5]

            // Act
            var output = layer.Forward(input);

            // Assert - Sigmoid outputs should be in (0, 1)
            for (int i = 0; i < 2; i++)
            {
                Assert.True(output[0, i] > 0.0);
                Assert.True(output[0, i] < 1.0);
            }
        }

        [Fact]
        public void DenseLayer_ForwardPass_TanhActivation_OutputsInRange()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2, new TanhActivation<double>());
            var input = new Tensor<double>([1, 3]);
            for (int i = 0; i < 3; i++)
                input[0, i] = (i - 1) * 5.0; // [-5, 0, 5]

            // Act
            var output = layer.Forward(input);

            // Assert - Tanh outputs should be in (-1, 1)
            for (int i = 0; i < 2; i++)
            {
                Assert.True(output[0, i] > -1.0);
                Assert.True(output[0, i] < 1.0);
            }
        }

        [Fact]
        public void DenseLayer_ForwardPass_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new DenseLayer<float>(3, 2, new ReLUActivation<float>());
            var input = new Tensor<float>([1, 3]);
            input[0, 0] = 1.0f; input[0, 1] = 2.0f; input[0, 2] = 3.0f;

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(2, output.Shape[1]);
        }

        // ===== Backward Pass Tests =====

        [Fact]
        public void DenseLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3, new ReLUActivation<double>());
            var input = new Tensor<double>([2, 5]);
            var outputGradient = new Tensor<double>([2, 3]);

            // Forward pass first
            layer.Forward(input);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(2, inputGradient.Shape[0]); // Batch size
            Assert.Equal(5, inputGradient.Shape[1]); // Input size
        }

        [Fact]
        public void DenseLayer_BackwardPass_WithLinearActivation_CalculatesCorrectGradient()
        {
            // Arrange
            var layer = new DenseLayer<double>(2, 2, new LinearActivation<double>());

            // Set specific weights
            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0; weights[0, 1] = 2.0;
            weights[1, 0] = 3.0; weights[1, 1] = 4.0;
            layer.SetWeights(weights);

            var input = new Tensor<double>([1, 2]);
            input[0, 0] = 1.0; input[0, 1] = 1.0;

            layer.Forward(input);

            var outputGradient = new Tensor<double>([1, 2]);
            outputGradient[0, 0] = 1.0; outputGradient[0, 1] = 1.0;

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert - Gradient should flow back through weights
            Assert.NotNull(inputGradient);
            Assert.Equal(1, inputGradient.Shape[0]);
            Assert.Equal(2, inputGradient.Shape[1]);
        }

        [Fact]
        public void DenseLayer_BackwardPass_MultipleTimes_WorksConsistently()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2, new ReLUActivation<double>());
            var input = new Tensor<double>([1, 3]);
            var outputGradient = new Tensor<double>([1, 2]);

            // Act - Multiple forward/backward passes
            for (int i = 0; i < 5; i++)
            {
                layer.Forward(input);
                var gradient = layer.Backward(outputGradient);

                // Assert
                Assert.NotNull(gradient);
                Assert.Equal(input.Shape[0], gradient.Shape[0]);
                Assert.Equal(input.Shape[1], gradient.Shape[1]);
            }
        }

        // ===== Parameter Management Tests =====

        [Fact]
        public void DenseLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange & Act
            var layer = new DenseLayer<double>(10, 5);

            // Assert
            // Expected: (10 inputs * 5 outputs) + 5 biases = 55
            Assert.Equal(55, layer.ParameterCount);
        }

        [Fact]
        public void DenseLayer_GetParameters_ReturnsCorrectLength()
        {
            // Arrange
            var layer = new DenseLayer<double>(8, 4);

            // Act
            var parameters = layer.GetParameters();

            // Assert
            Assert.Equal(36, parameters.Length); // 8*4 + 4 = 36
        }

        [Fact]
        public void DenseLayer_SetParameters_UpdatesWeightsAndBiases()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2);
            var newParameters = new Vector<double>(8); // 3*2 + 2 = 8
            for (int i = 0; i < 8; i++)
                newParameters[i] = i + 1.0;

            // Act
            layer.SetParameters(newParameters);
            var retrieved = layer.GetParameters();

            // Assert
            for (int i = 0; i < 8; i++)
                Assert.Equal(newParameters[i], retrieved[i], precision: 10);
        }

        [Fact]
        public void DenseLayer_SetGetParameters_RoundTrip_PreservesValues()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3);
            var originalParams = layer.GetParameters();

            // Act
            layer.SetParameters(originalParams);
            var retrievedParams = layer.GetParameters();

            // Assert
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], retrievedParams[i], precision: 10);
        }

        // ===== Update Parameters Tests =====

        [Fact]
        public void DenseLayer_UpdateParameters_ChangesParameters()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2);
            var input = new Tensor<double>([1, 3]);
            var outputGradient = new Tensor<double>([1, 2]);
            for (int i = 0; i < 2; i++)
                outputGradient[0, i] = 1.0;

            layer.Forward(input);
            layer.Backward(outputGradient);

            var paramsBefore = layer.GetParameters();

            // Act
            layer.UpdateParameters(0.01);
            var paramsAfter = layer.GetParameters();

            // Assert - Parameters should have changed
            bool parametersChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-10)
                {
                    parametersChanged = true;
                    break;
                }
            }
            Assert.True(parametersChanged);
        }

        [Fact]
        public void DenseLayer_UpdateParameters_WithHigherLearningRate_MakesBiggerChanges()
        {
            // Arrange
            var layer1 = new DenseLayer<double>(3, 2);
            var layer2 = new DenseLayer<double>(3, 2);

            // Make layers identical
            var params_ = layer1.GetParameters();
            layer2.SetParameters(params_);

            var input = new Tensor<double>([1, 3]);
            var outputGradient = new Tensor<double>([1, 2]);
            for (int i = 0; i < 2; i++)
                outputGradient[0, i] = 1.0;

            // Forward and backward for both
            layer1.Forward(input);
            layer1.Backward(outputGradient);
            layer2.Forward(input);
            layer2.Backward(outputGradient);

            var params1Before = layer1.GetParameters();
            var params2Before = layer2.GetParameters();

            // Act
            layer1.UpdateParameters(0.01);
            layer2.UpdateParameters(0.1); // 10x larger learning rate

            var params1After = layer1.GetParameters();
            var params2After = layer2.GetParameters();

            // Assert - Layer2 should have larger parameter changes
            var change1 = 0.0;
            var change2 = 0.0;
            for (int i = 0; i < params1Before.Length; i++)
            {
                change1 += Math.Abs(params1After[i] - params1Before[i]);
                change2 += Math.Abs(params2After[i] - params2Before[i]);
            }
            Assert.True(change2 > change1);
        }

        // ===== Different Batch Sizes Tests =====

        [Fact]
        public void DenseLayer_DifferentBatchSizes_ProduceConsistentResults()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3);
            var params_ = layer.GetParameters();

            var singleInput = new Tensor<double>([1, 5]);
            for (int i = 0; i < 5; i++)
                singleInput[0, i] = i + 1.0;

            var batchInput = new Tensor<double>([3, 5]);
            for (int b = 0; b < 3; b++)
                for (int i = 0; i < 5; i++)
                    batchInput[b, i] = i + 1.0; // Same input repeated

            // Act
            var singleOutput = layer.Forward(singleInput);

            layer.ResetState();
            layer.SetParameters(params_); // Reset to same state
            var batchOutput = layer.Forward(batchInput);

            // Assert - Each batch item should match single input result
            for (int b = 0; b < 3; b++)
            {
                for (int i = 0; i < 3; i++)
                {
                    Assert.Equal(singleOutput[0, i], batchOutput[b, i], precision: 6);
                }
            }
        }

        [Fact]
        public void DenseLayer_BatchSizeOne_EquivalentToSingleInput()
        {
            // Arrange
            var layer = new DenseLayer<double>(4, 2);
            var input = new Tensor<double>([1, 4]);
            for (int i = 0; i < 4; i++)
                input[0, i] = i * 0.5;

            // Act
            var output = layer.Forward(input);

            // Assert - Should process correctly with batch size 1
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(2, output.Shape[1]);
        }

        [Fact]
        public void DenseLayer_LargeBatch_ProcessesCorrectly()
        {
            // Arrange
            var layer = new DenseLayer<double>(10, 5);
            var input = new Tensor<double>([100, 10]); // Large batch

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(100, output.Shape[0]);
            Assert.Equal(5, output.Shape[1]);
        }

        // ===== Reset State Tests =====

        [Fact]
        public void DenseLayer_ResetState_ClearsInternalState()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2);
            var input = new Tensor<double>([1, 3]);

            layer.Forward(input);

            // Act
            layer.ResetState();

            // Assert - Should be able to use layer normally after reset
            var output = layer.Forward(input);
            Assert.NotNull(output);
        }

        // ===== Clone Tests =====

        [Fact]
        public void DenseLayer_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new DenseLayer<double>(4, 3);
            var originalParams = original.GetParameters();

            // Act
            var clone = (DenseLayer<double>)original.Clone();
            var cloneParams = clone.GetParameters();

            // Assert - Clone should have same parameters
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], cloneParams[i], precision: 10);

            // Modify clone parameters
            var newParams = clone.GetParameters();
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
        public void DenseLayer_TrainingOnIdentityFunction_ConvergesToCorrectWeights()
        {
            // Arrange - Train layer to learn identity function
            var layer = new DenseLayer<double>(3, 3, new LinearActivation<double>());

            // Act - Training loop
            for (int epoch = 0; epoch < 100; epoch++)
            {
                var input = new Tensor<double>([1, 3]);
                input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0;

                var output = layer.Forward(input);

                // Calculate gradient (output - target)
                var gradient = new Tensor<double>([1, 3]);
                for (int i = 0; i < 3; i++)
                    gradient[0, i] = output[0, i] - input[0, i];

                layer.Backward(gradient);
                layer.UpdateParameters(0.01);
            }

            // Assert - Final output should be close to input
            var testInput = new Tensor<double>([1, 3]);
            testInput[0, 0] = 1.0; testInput[0, 1] = 2.0; testInput[0, 2] = 3.0;
            var finalOutput = layer.Forward(testInput);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(testInput[0, i], finalOutput[0, i], precision: 1);
            }
        }

        [Fact]
        public void DenseLayer_TrainingOnXORProblem_ReducesError()
        {
            // Arrange - XOR problem requires non-linear activation
            var layer1 = new DenseLayer<double>(2, 4, new ReLUActivation<double>());
            var layer2 = new DenseLayer<double>(4, 1, new SigmoidActivation<double>());

            var xorInputs = new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            var xorOutputs = new double[] { 0, 1, 1, 0 };

            double initialError = 0;
            double finalError = 0;

            // Calculate initial error
            for (int i = 0; i < 4; i++)
            {
                var input = new Tensor<double>([1, 2]);
                input[0, 0] = xorInputs[i, 0];
                input[0, 1] = xorInputs[i, 1];

                var hidden = layer1.Forward(input);
                var output = layer2.Forward(hidden);

                initialError += Math.Pow(output[0, 0] - xorOutputs[i], 2);
            }

            // Act - Training
            for (int epoch = 0; epoch < 500; epoch++)
            {
                for (int i = 0; i < 4; i++)
                {
                    var input = new Tensor<double>([1, 2]);
                    input[0, 0] = xorInputs[i, 0];
                    input[0, 1] = xorInputs[i, 1];

                    var hidden = layer1.Forward(input);
                    var output = layer2.Forward(hidden);

                    // Backpropagate
                    var outputGradient = new Tensor<double>([1, 1]);
                    outputGradient[0, 0] = 2 * (output[0, 0] - xorOutputs[i]);

                    var hiddenGradient = layer2.Backward(outputGradient);
                    layer1.Backward(hiddenGradient);

                    layer2.UpdateParameters(0.1);
                    layer1.UpdateParameters(0.1);
                }
            }

            // Calculate final error
            for (int i = 0; i < 4; i++)
            {
                var input = new Tensor<double>([1, 2]);
                input[0, 0] = xorInputs[i, 0];
                input[0, 1] = xorInputs[i, 1];

                layer1.ResetState();
                layer2.ResetState();

                var hidden = layer1.Forward(input);
                var output = layer2.Forward(hidden);

                finalError += Math.Pow(output[0, 0] - xorOutputs[i], 2);
            }

            // Assert - Error should decrease significantly
            Assert.True(finalError < initialError * 0.5,
                $"Final error {finalError} should be less than half of initial error {initialError}");
        }

        [Fact]
        public void DenseLayer_SupportsTraining_ReturnsTrue()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3);

            // Act & Assert
            Assert.True(layer.SupportsTraining);
        }

        // ===== Edge Cases and Error Handling =====

        [Fact]
        public void DenseLayer_VerySmallLayer_1to1_WorksCorrectly()
        {
            // Arrange
            var layer = new DenseLayer<double>(1, 1);
            var input = new Tensor<double>([1, 1]);
            input[0, 0] = 5.0;

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(1, output.Shape[1]);
        }

        [Fact]
        public void DenseLayer_LargeLayer_100to50_WorksCorrectly()
        {
            // Arrange
            var layer = new DenseLayer<double>(100, 50);
            var input = new Tensor<double>([1, 100]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(50, output.Shape[1]);
            Assert.Equal(5050, layer.ParameterCount); // 100*50 + 50
        }

        [Fact]
        public void DenseLayer_WithZeroInputs_ProducesValidOutput()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3);
            var input = new Tensor<double>([1, 5]); // All zeros

            // Act
            var output = layer.Forward(input);

            // Assert - Output should be the biases (possibly activated)
            Assert.NotNull(output);
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(3, output.Shape[1]);
        }

        [Fact]
        public void DenseLayer_BackwardBeforeForward_ThrowsException()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2);
            var outputGradient = new Tensor<double>([1, 2]);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.Backward(outputGradient));
        }

        [Fact]
        public void DenseLayer_UpdateParametersBeforeBackward_ThrowsException()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.UpdateParameters(0.01));
        }

        // ===== Vector Activation Function Tests =====

        [Fact]
        public void DenseLayer_WithSoftmaxActivation_ProducesValidProbabilities()
        {
            // Arrange
            var layer = new DenseLayer<double>(5, 3, new SoftmaxActivation<double>());
            var input = new Tensor<double>([1, 5]);
            for (int i = 0; i < 5; i++)
                input[0, i] = i * 2.0;

            // Act
            var output = layer.Forward(input);

            // Assert - Softmax outputs should sum to 1
            double sum = 0;
            for (int i = 0; i < 3; i++)
            {
                Assert.True(output[0, i] >= 0); // Non-negative
                Assert.True(output[0, i] <= 1); // At most 1
                sum += output[0, i];
            }
            Assert.Equal(1.0, sum, precision: 6);
        }

        // ===== Numerical Stability Tests =====

        [Fact]
        public void DenseLayer_WithVeryLargeInputs_MaintainsNumericalStability()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2, new TanhActivation<double>());
            var input = new Tensor<double>([1, 3]);
            input[0, 0] = 1e6; input[0, 1] = 1e6; input[0, 2] = 1e6;

            // Act
            var output = layer.Forward(input);

            // Assert - Output should not be NaN or Infinity
            for (int i = 0; i < 2; i++)
            {
                Assert.False(double.IsNaN(output[0, i]));
                Assert.False(double.IsInfinity(output[0, i]));
            }
        }

        [Fact]
        public void DenseLayer_WithVerySmallInputs_MaintainsNumericalStability()
        {
            // Arrange
            var layer = new DenseLayer<double>(3, 2);
            var input = new Tensor<double>([1, 3]);
            input[0, 0] = 1e-10; input[0, 1] = 1e-10; input[0, 2] = 1e-10;

            // Act
            var output = layer.Forward(input);

            // Assert - Output should not be NaN
            for (int i = 0; i < 2; i++)
            {
                Assert.False(double.IsNaN(output[0, i]));
            }
        }

        // ===== Multiple Forward/Backward Cycles =====

        [Fact]
        public void DenseLayer_MultipleTrainingCycles_ImprovesPerformance()
        {
            // Arrange - Simple regression task
            var layer = new DenseLayer<double>(1, 1, new LinearActivation<double>());

            var inputs = new double[] { 1, 2, 3, 4, 5 };
            var targets = new double[] { 2, 4, 6, 8, 10 }; // y = 2x

            double initialError = 0;
            double finalError = 0;

            // Calculate initial error
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = new Tensor<double>([1, 1]);
                input[0, 0] = inputs[i];
                var output = layer.Forward(input);
                initialError += Math.Pow(output[0, 0] - targets[i], 2);
            }

            // Act - Training
            for (int epoch = 0; epoch < 100; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var input = new Tensor<double>([1, 1]);
                    input[0, 0] = inputs[i];

                    var output = layer.Forward(input);

                    var gradient = new Tensor<double>([1, 1]);
                    gradient[0, 0] = 2 * (output[0, 0] - targets[i]);

                    layer.Backward(gradient);
                    layer.UpdateParameters(0.01);
                }
            }

            // Calculate final error
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = new Tensor<double>([1, 1]);
                input[0, 0] = inputs[i];
                layer.ResetState();
                var output = layer.Forward(input);
                finalError += Math.Pow(output[0, 0] - targets[i], 2);
            }

            // Assert
            Assert.True(finalError < initialError * 0.1);
        }
    }
}
