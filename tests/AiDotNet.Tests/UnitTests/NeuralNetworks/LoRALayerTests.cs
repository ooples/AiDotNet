using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks
{
    public class LoRALayerTests
    {
        [Fact]
        public void Constructor_WithValidParameters_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new LoRALayer<double>(inputSize: 10, outputSize: 5, rank: 3);

            // Assert
            Assert.Equal(10, layer.GetInputShape()[0]);
            Assert.Equal(5, layer.GetOutputShape()[0]);
            Assert.Equal(3, layer.Rank);
            Assert.True(layer.SupportsTraining);
            Assert.Equal((10 * 3) + (3 * 5), layer.ParameterCount);
        }

        [Fact]
        public void Constructor_WithZeroRank_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LoRALayer<double>(10, 5, rank: 0));
        }

        [Fact]
        public void Constructor_WithNegativeRank_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LoRALayer<double>(10, 5, rank: -1));
        }

        [Fact]
        public void Constructor_WithRankExceedingDimensions_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LoRALayer<double>(10, 5, rank: 11));
        }

        [Fact]
        public void Constructor_WithCustomAlpha_UsesSpecifiedAlpha()
        {
            // Arrange & Act
            var layer = new LoRALayer<double>(10, 5, rank: 3, alpha: 16);

            // Assert
            Assert.Equal(16.0, layer.Alpha);
            Assert.Equal(16.0 / 3.0, layer.Scaling);
        }

        [Fact]
        public void Constructor_WithDefaultAlpha_UsesRankAsAlpha()
        {
            // Arrange & Act
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Assert
            Assert.Equal(3.0, layer.Alpha);
            Assert.Equal(1.0, layer.Scaling);
        }

        [Fact]
        public void Forward_WithValidInput_ProducesCorrectOutputShape()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 2, 10 }); // Batch size 2, input size 10

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]); // Batch size preserved
            Assert.Equal(5, output.Shape[1]); // Output size correct
        }

        [Fact]
        public void Forward_WithInvalidInputSize_ThrowsArgumentException()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 2, 8 }); // Wrong input size

            // Act & Assert
            Assert.Throws<ArgumentException>(() => layer.Forward(input));
        }

        [Fact]
        public void Forward_InitiallyProducesZeroOutput_DueToZeroInitializationOfB()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[0, i] = 1.0;
            }

            // Act
            var output = layer.Forward(input);

            // Assert - Output should be near zero because B is initialized to zero
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(Math.Abs(output.GetFlat(i)) < 1e-10);
            }
        }

        [Fact]
        public void Backward_WithoutForward_ThrowsInvalidOperationException()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var outputGradient = new Tensor<double>(new[] { 2, 5 });

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.Backward(outputGradient));
        }

        [Fact]
        public void Backward_WithValidGradient_ProducesCorrectInputGradientShape()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 2, 10 });
            layer.Forward(input);
            var outputGradient = new Tensor<double>(new[] { 2, 5 });

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(2, inputGradient.Shape[0]);
            Assert.Equal(10, inputGradient.Shape[1]);
        }

        [Fact]
        public void GetParameters_ReturnsCorrectParameterCount()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var parameters = layer.GetParameters();

            // Assert
            Assert.Equal((10 * 3) + (3 * 5), parameters.Length);
        }

        [Fact]
        public void SetParameters_ThenGetParameters_ReturnsSetValues()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var newParams = new Vector<double>((10 * 3) + (3 * 5));
            for (int i = 0; i < newParams.Length; i++)
            {
                newParams[i] = i * 0.1;
            }

            // Act
            layer.SetParameters(newParams);
            var retrievedParams = layer.GetParameters();

            // Assert
            Assert.Equal(newParams.Length, retrievedParams.Length);
            for (int i = 0; i < newParams.Length; i++)
            {
                Assert.Equal(newParams[i], retrievedParams[i], precision: 10);
            }
        }

        [Fact]
        public void SetParameters_WithWrongSize_ThrowsArgumentException()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var wrongParams = new Vector<double>(100); // Wrong size

            // Act & Assert
            Assert.Throws<ArgumentException>(() => layer.SetParameters(wrongParams));
        }

        [Fact]
        public void UpdateParameters_UpdatesParametersCorrectly()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[0, i] = 1.0;
            }
            layer.Forward(input);

            var outputGradient = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                outputGradient[0, i] = 0.1;
            }
            layer.Backward(outputGradient);

            var paramsBefore = layer.GetParameters();

            // Act
            layer.UpdateParameters(0.01);
            var paramsAfter = layer.GetParameters();

            // Assert - At least some parameters should have changed
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
        public void MergeWeights_ProducesCorrectDimensions()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var mergedWeights = layer.MergeWeights();

            // Assert
            Assert.Equal(5, mergedWeights.Rows);
            Assert.Equal(10, mergedWeights.Columns);
        }

        [Fact]
        public void MergeWeights_InitiallyProducesZeroMatrix()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var mergedWeights = layer.MergeWeights();

            // Assert - Should be near zero because B is initialized to zero
            for (int i = 0; i < mergedWeights.Rows; i++)
            {
                for (int j = 0; j < mergedWeights.Columns; j++)
                {
                    Assert.True(Math.Abs(mergedWeights[i, j]) < 1e-10);
                }
            }
        }

        [Fact]
        public void GetMatrixA_ReturnsCorrectDimensions()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var matrixA = layer.GetMatrixA();

            // Assert
            Assert.Equal(10, matrixA.Rows);
            Assert.Equal(3, matrixA.Columns);
        }

        [Fact]
        public void GetMatrixB_ReturnsCorrectDimensions()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var matrixB = layer.GetMatrixB();

            // Assert
            Assert.Equal(3, matrixB.Rows);
            Assert.Equal(5, matrixB.Columns);
        }

        [Fact]
        public void GetMatrixA_ReturnsClone_NotOriginal()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var matrixA1 = layer.GetMatrixA();
            matrixA1[0, 0] = 999.0;
            var matrixA2 = layer.GetMatrixA();

            // Assert - Original should not be affected
            Assert.NotEqual(999.0, matrixA2[0, 0]);
        }

        [Fact]
        public void GetMatrixB_InitializedToZero()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);

            // Act
            var matrixB = layer.GetMatrixB();

            // Assert
            for (int i = 0; i < matrixB.Rows; i++)
            {
                for (int j = 0; j < matrixB.Columns; j++)
                {
                    Assert.Equal(0.0, matrixB[i, j]);
                }
            }
        }

        [Fact]
        public void GetParameterGradients_AfterBackward_ReturnsValidGradients()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[0, i] = 1.0;
            }
            layer.Forward(input);

            var outputGradient = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                outputGradient[0, i] = 0.1;
            }

            // Act
            layer.Backward(outputGradient);
            var gradients = layer.GetParameterGradients();

            // Assert
            Assert.Equal(layer.ParameterCount, gradients.Length);
            // At least some gradients should be non-zero
            Assert.Contains(gradients, g => Math.Abs(g) > 1e-10);
        }

        [Fact]
        public void ParameterCount_ReflectsCorrectFormula()
        {
            // Arrange & Act
            var layer = new LoRALayer<double>(inputSize: 100, outputSize: 50, rank: 8);

            // Assert
            // Parameters = (inputSize * rank) + (rank * outputSize)
            Assert.Equal((100 * 8) + (8 * 50), layer.ParameterCount);
        }

        [Theory]
        [InlineData(10, 10, 1)]
        [InlineData(100, 50, 8)]
        [InlineData(1000, 500, 64)]
        public void ParameterCount_WithVariousConfigurations_IsCorrect(int inputSize, int outputSize, int rank)
        {
            // Arrange & Act
            var layer = new LoRALayer<double>(inputSize, outputSize, rank);

            // Assert
            Assert.Equal((inputSize * rank) + (rank * outputSize), layer.ParameterCount);
        }

        [Fact]
        public void ForwardAndBackward_MultipleIterations_MaintainsGradientFlow()
        {
            // Arrange
            var layer = new LoRALayer<double>(10, 5, rank: 3);
            var input = new Tensor<double>(new[] { 2, 10 });
            var outputGradient = new Tensor<double>(new[] { 2, 5 });

            // Act & Assert - Multiple iterations should not throw
            for (int i = 0; i < 10; i++)
            {
                var output = layer.Forward(input);
                var inputGrad = layer.Backward(outputGradient);

                Assert.NotNull(output);
                Assert.NotNull(inputGrad);
            }
        }

        [Fact]
        public void LoRALayer_WithFloat_WorksCorrectly()
        {
            // Arrange & Act
            var layer = new LoRALayer<float>(inputSize: 10, outputSize: 5, rank: 3);
            var input = new Tensor<float>(new[] { 1, 10 });

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(5, output.Shape[1]);
        }
    }
}
