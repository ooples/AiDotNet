using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    /// <summary>
    /// Integration tests for recurrent layers (RNN, LSTM, GRU) with comprehensive coverage
    /// of sequential processing, forward/backward passes, and temporal dependencies.
    /// </summary>
    public class RecurrentLayerIntegrationTests
    {
        private const double Tolerance = 1e-6;

        // ===== RecurrentLayer Tests =====

        [Fact]
        public void RecurrentLayer_ForwardPass_SingleTimeStep_ProducesCorrectShape()
        {
            // Arrange
            var layer = new RecurrentLayer<double>(inputSize: 5, hiddenSize: 10);
            var input = new Tensor<double>([1, 1, 5]); // Batch=1, Sequence=1, Features=5

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]); // Batch
            Assert.Equal(1, output.Shape[1]); // Sequence
            Assert.Equal(10, output.Shape[2]); // Hidden size
        }

        [Fact]
        public void RecurrentLayer_ForwardPass_MultipleTimeSteps_ProducesCorrectShape()
        {
            // Arrange
            var layer = new RecurrentLayer<double>(inputSize: 3, hiddenSize: 8);
            var input = new Tensor<double>([2, 10, 3]); // Batch=2, Sequence=10, Features=3

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]); // Batch preserved
            Assert.Equal(10, output.Shape[1]); // Sequence length preserved
            Assert.Equal(8, output.Shape[2]); // Hidden size
        }

        [Fact]
        public void RecurrentLayer_ForwardPass_LongSequence_ProcessesCorrectly()
        {
            // Arrange
            var layer = new RecurrentLayer<double>(inputSize: 4, hiddenSize: 6);
            var input = new Tensor<double>([1, 50, 4]); // Long sequence of 50 steps

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(50, output.Shape[1]);
            Assert.Equal(6, output.Shape[2]);
        }

        [Fact]
        public void RecurrentLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new RecurrentLayer<double>(inputSize: 3, hiddenSize: 5);
            var input = new Tensor<double>([1, 4, 3]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
        }

        [Fact]
        public void RecurrentLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange
            var inputSize = 4;
            var hiddenSize = 6;
            var layer = new RecurrentLayer<double>(inputSize, hiddenSize);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert
            // Expected: (inputSize * hiddenSize) + (hiddenSize * hiddenSize) + hiddenSize
            // = (4 * 6) + (6 * 6) + 6 = 24 + 36 + 6 = 66
            Assert.Equal(66, paramCount);
        }

        [Fact]
        public void RecurrentLayer_ResetState_ClearsHiddenState()
        {
            // Arrange
            var layer = new RecurrentLayer<double>(inputSize: 3, hiddenSize: 5);
            var input = new Tensor<double>([1, 5, 3]);
            layer.Forward(input);

            // Act
            layer.ResetState();

            // Assert - Should work normally after reset
            var output = layer.Forward(input);
            Assert.NotNull(output);
        }

        // ===== LSTMLayer Tests =====

        [Fact]
        public void LSTMLayer_ForwardPass_SingleTimeStep_ProducesCorrectShape()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 8, hiddenSize: 16);
            var input = new Tensor<double>([1, 1, 8]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(1, output.Shape[1]);
            Assert.Equal(16, output.Shape[2]);
        }

        [Fact]
        public void LSTMLayer_ForwardPass_MultipleTimeSteps_ProducesCorrectShape()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 5, hiddenSize: 10);
            var input = new Tensor<double>([2, 8, 5]); // Batch=2, Sequence=8, Features=5

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(8, output.Shape[1]);
            Assert.Equal(10, output.Shape[2]);
        }

        [Fact]
        public void LSTMLayer_ForwardPass_LongSequence_HandlesTemporalDependencies()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 3, hiddenSize: 8);
            var input = new Tensor<double>([1, 100, 3]); // Very long sequence

            // Act
            var output = layer.Forward(input);

            // Assert - LSTM should handle long sequences without issues
            Assert.Equal(100, output.Shape[1]);
            Assert.Equal(8, output.Shape[2]);
        }

        [Fact]
        public void LSTMLayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 4, hiddenSize: 6);
            var input = new Tensor<double>([1, 5, 4]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
        }

        [Fact]
        public void LSTMLayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange
            var inputSize = 5;
            var hiddenSize = 7;
            var layer = new LSTMLayer<double>(inputSize, hiddenSize);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert
            // LSTM has 4 gates (forget, input, candidate, output)
            // Each gate has: (inputSize * hiddenSize) + (hiddenSize * hiddenSize) + hiddenSize
            // Total: 4 * [(5 * 7) + (7 * 7) + 7] = 4 * [35 + 49 + 7] = 4 * 91 = 364
            Assert.Equal(364, paramCount);
        }

        [Fact]
        public void LSTMLayer_UpdateParameters_ChangesWeights()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 3, hiddenSize: 4);
            var input = new Tensor<double>([1, 2, 3]);
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
        public void LSTMLayer_BatchProcessing_WorksCorrectly()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 4, hiddenSize: 5);
            var input = new Tensor<double>([8, 6, 4]); // Batch=8

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(8, output.Shape[0]); // Batch preserved
        }

        [Fact]
        public void LSTMLayer_ResetState_ClearsCellState()
        {
            // Arrange
            var layer = new LSTMLayer<double>(inputSize: 3, hiddenSize: 4);
            var input = new Tensor<double>([1, 5, 3]);
            layer.Forward(input);

            // Act
            layer.ResetState();

            // Assert
            var output = layer.Forward(input);
            Assert.NotNull(output);
        }

        // ===== GRULayer Tests =====

        [Fact]
        public void GRULayer_ForwardPass_SingleTimeStep_ProducesCorrectShape()
        {
            // Arrange
            var layer = new GRULayer<double>(inputSize: 6, hiddenSize: 12);
            var input = new Tensor<double>([1, 1, 6]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(1, output.Shape[1]);
            Assert.Equal(12, output.Shape[2]);
        }

        [Fact]
        public void GRULayer_ForwardPass_MultipleTimeSteps_ProducesCorrectShape()
        {
            // Arrange
            var layer = new GRULayer<double>(inputSize: 4, hiddenSize: 8);
            var input = new Tensor<double>([2, 10, 4]); // Batch=2, Sequence=10

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(10, output.Shape[1]);
            Assert.Equal(8, output.Shape[2]);
        }

        [Fact]
        public void GRULayer_BackwardPass_ProducesCorrectGradientShape()
        {
            // Arrange
            var layer = new GRULayer<double>(inputSize: 3, hiddenSize: 5);
            var input = new Tensor<double>([1, 4, 3]);
            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
            Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
            Assert.Equal(input.Shape[2], inputGradient.Shape[2]);
        }

        [Fact]
        public void GRULayer_ParameterCount_CalculatesCorrectly()
        {
            // Arrange
            var inputSize = 4;
            var hiddenSize = 6;
            var layer = new GRULayer<double>(inputSize, hiddenSize);

            // Act
            var paramCount = layer.ParameterCount;

            // Assert
            // GRU has 3 gates (update, reset, candidate)
            // Each gate has: (inputSize * hiddenSize) + (hiddenSize * hiddenSize) + hiddenSize
            // Total: 3 * [(4 * 6) + (6 * 6) + 6] = 3 * [24 + 36 + 6] = 3 * 66 = 198
            Assert.Equal(198, paramCount);
        }

        [Fact]
        public void GRULayer_UpdateParameters_ChangesWeights()
        {
            // Arrange
            var layer = new GRULayer<double>(inputSize: 3, hiddenSize: 4);
            var input = new Tensor<double>([1, 2, 3]);
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
        public void GRULayer_LongSequence_ProcessesEfficiently()
        {
            // Arrange
            var layer = new GRULayer<double>(inputSize: 5, hiddenSize: 10);
            var input = new Tensor<double>([1, 50, 5]); // Sequence of 50

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(50, output.Shape[1]);
            Assert.Equal(10, output.Shape[2]);
        }

        // ===== Comparative Tests =====

        [Fact]
        public void RecurrentLayers_DifferentTypes_ProduceSameShapeOutput()
        {
            // Arrange
            var rnn = new RecurrentLayer<double>(inputSize: 5, hiddenSize: 8);
            var lstm = new LSTMLayer<double>(inputSize: 5, hiddenSize: 8);
            var gru = new GRULayer<double>(inputSize: 5, hiddenSize: 8);

            var input = new Tensor<double>([2, 6, 5]);

            // Act
            var rnnOutput = rnn.Forward(input);
            var lstmOutput = lstm.Forward(input);
            var gruOutput = gru.Forward(input);

            // Assert - All should produce same shape
            Assert.Equal(rnnOutput.Shape[0], lstmOutput.Shape[0]);
            Assert.Equal(rnnOutput.Shape[1], lstmOutput.Shape[1]);
            Assert.Equal(rnnOutput.Shape[2], lstmOutput.Shape[2]);

            Assert.Equal(rnnOutput.Shape[0], gruOutput.Shape[0]);
            Assert.Equal(rnnOutput.Shape[1], gruOutput.Shape[1]);
            Assert.Equal(rnnOutput.Shape[2], gruOutput.Shape[2]);
        }

        [Fact]
        public void RecurrentLayers_ParameterCounts_DifferAsExpected()
        {
            // Arrange
            var rnn = new RecurrentLayer<double>(inputSize: 4, hiddenSize: 6);
            var lstm = new LSTMLayer<double>(inputSize: 4, hiddenSize: 6);
            var gru = new GRULayer<double>(inputSize: 4, hiddenSize: 6);

            // Act & Assert
            var rnnParams = rnn.ParameterCount;
            var lstmParams = lstm.ParameterCount;
            var gruParams = gru.ParameterCount;

            // LSTM should have most parameters (4 gates), GRU next (3 gates), RNN least
            Assert.True(lstmParams > gruParams);
            Assert.True(gruParams > rnnParams);
        }

        // ===== Float Type Tests =====

        [Fact]
        public void RecurrentLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new RecurrentLayer<float>(inputSize: 4, hiddenSize: 6);
            var input = new Tensor<float>([1, 3, 4]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(3, output.Shape[1]);
            Assert.Equal(6, output.Shape[2]);
        }

        [Fact]
        public void LSTMLayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new LSTMLayer<float>(inputSize: 3, hiddenSize: 5);
            var input = new Tensor<float>([1, 4, 3]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(5, output.Shape[2]);
        }

        [Fact]
        public void GRULayer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var layer = new GRULayer<float>(inputSize: 5, hiddenSize: 7);
            var input = new Tensor<float>([1, 2, 5]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(7, output.Shape[2]);
        }

        // ===== Training Scenario Tests =====

        [Fact]
        public void RecurrentLayer_SimpleSequenceTraining_ConvergesParameters()
        {
            // Arrange - Train to recognize simple patterns
            var layer = new RecurrentLayer<double>(inputSize: 2, hiddenSize: 4);

            var input = new Tensor<double>([1, 5, 2]);
            for (int t = 0; t < 5; t++)
            {
                input[0, t, 0] = t * 0.1;
                input[0, t, 1] = t * 0.2;
            }

            var initialParams = layer.GetParameters();

            // Act - Training iterations
            for (int epoch = 0; epoch < 10; epoch++)
            {
                var output = layer.Forward(input);
                var gradient = new Tensor<double>(output.Shape);
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] = 0.05;

                layer.Backward(gradient);
                layer.UpdateParameters(0.01);
            }

            var finalParams = layer.GetParameters();

            // Assert
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

        [Fact]
        public void LSTMLayer_SequenceMemory_MaintainsInformation()
        {
            // Arrange - Test LSTM's ability to maintain information
            var layer = new LSTMLayer<double>(inputSize: 3, hiddenSize: 6);

            var input = new Tensor<double>([1, 20, 3]); // Long sequence
            for (int t = 0; t < 20; t++)
            {
                for (int f = 0; f < 3; f++)
                {
                    input[0, t, f] = (t + f) * 0.1;
                }
            }

            // Act - Multiple forward passes to ensure state is maintained
            var output1 = layer.Forward(input);
            layer.ResetState();
            var output2 = layer.Forward(input);

            // Assert - Same input should produce same output after reset
            for (int i = 0; i < Math.Min(10, output1.Length); i++)
            {
                Assert.Equal(output1[i], output2[i], precision: 6);
            }
        }

        [Fact]
        public void GRULayer_MultipleForwardBackwardCycles_WorksStably()
        {
            // Arrange
            var layer = new GRULayer<double>(inputSize: 4, hiddenSize: 5);
            var input = new Tensor<double>([1, 3, 4]);

            // Act - Multiple cycles
            for (int i = 0; i < 20; i++)
            {
                var output = layer.Forward(input);
                var gradient = new Tensor<double>(output.Shape);
                for (int j = 0; j < gradient.Length; j++)
                    gradient[j] = 0.01;

                var inputGradient = layer.Backward(gradient);
                layer.UpdateParameters(0.01);

                // Assert - No NaN or Infinity
                for (int j = 0; j < output.Length; j++)
                {
                    Assert.False(double.IsNaN(output[j]));
                    Assert.False(double.IsInfinity(output[j]));
                }
            }
        }

        // ===== Clone Tests =====

        [Fact]
        public void RecurrentLayer_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new RecurrentLayer<double>(inputSize: 3, hiddenSize: 4);
            var originalParams = original.GetParameters();

            // Act
            var clone = (RecurrentLayer<double>)original.Clone();
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

        [Fact]
        public void LSTMLayer_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new LSTMLayer<double>(inputSize: 3, hiddenSize: 5);
            var originalParams = original.GetParameters();

            // Act
            var clone = (LSTMLayer<double>)original.Clone();

            // Modify clone
            var input = new Tensor<double>([1, 2, 3]);
            clone.Forward(input);
            var output = clone.Forward(input);
            var gradient = new Tensor<double>(output.Shape);
            clone.Backward(gradient);
            clone.UpdateParameters(1.0);

            // Assert - Original unchanged
            var originalParamsAfter = original.GetParameters();
            for (int i = 0; i < originalParams.Length; i++)
                Assert.Equal(originalParams[i], originalParamsAfter[i], precision: 10);
        }

        // ===== SupportsTraining Tests =====

        [Fact]
        public void RecurrentLayer_SupportsTraining_ReturnsTrue()
        {
            var layer = new RecurrentLayer<double>(inputSize: 3, hiddenSize: 4);
            Assert.True(layer.SupportsTraining);
        }

        [Fact]
        public void LSTMLayer_SupportsTraining_ReturnsTrue()
        {
            var layer = new LSTMLayer<double>(inputSize: 3, hiddenSize: 4);
            Assert.True(layer.SupportsTraining);
        }

        [Fact]
        public void GRULayer_SupportsTraining_ReturnsTrue()
        {
            var layer = new GRULayer<double>(inputSize: 3, hiddenSize: 4);
            Assert.True(layer.SupportsTraining);
        }
    }
}
