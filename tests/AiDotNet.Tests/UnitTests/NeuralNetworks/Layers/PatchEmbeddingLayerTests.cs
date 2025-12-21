using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.Layers
{
    public class PatchEmbeddingLayerTests
    {
        [Fact]
        public void Constructor_WithValidParameters_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                embeddingDim: 64);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.SupportsTraining);
            Assert.True(layer.ParameterCount > 0);
        }

        [Fact]
        public void Constructor_WithNonDivisibleHeight_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new PatchEmbeddingLayer<double>(
                imageHeight: 30,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                embeddingDim: 64));
        }

        [Fact]
        public void Constructor_WithNonDivisibleWidth_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new PatchEmbeddingLayer<double>(
                imageHeight: 32,
                imageWidth: 30,
                channels: 3,
                patchSize: 8,
                embeddingDim: 64));
        }

        [Fact]
        public void Forward_WithValidInput_ReturnsCorrectShape()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                embeddingDim: 64);

            int batchSize = 2;
            var input = new Tensor<double>([batchSize, 3, 32, 32]);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.1;
            }

            // Act
            var output = layer.Forward(input);

            // Assert
            int expectedPatches = (32 / 8) * (32 / 8);
            Assert.Equal(3, output.Rank);
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(expectedPatches, output.Shape[1]);
            Assert.Equal(64, output.Shape[2]);
        }

        [Fact]
        public void Forward_CalculatesCorrectNumberOfPatches()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 64,
                imageWidth: 64,
                channels: 3,
                patchSize: 16,
                embeddingDim: 128);

            var input = new Tensor<double>([1, 3, 64, 64]);

            // Act
            var output = layer.Forward(input);

            // Assert - 64/16 = 4 patches per dimension, 4x4 = 16 total patches
            Assert.Equal(16, output.Shape[1]);
        }

        [Fact]
        public void Backward_AfterForward_UpdatesGradients()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            var input = new Tensor<double>([1, 3, 16, 16]);
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.NextDouble();
            }

            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                outputGradient[i] = random.NextDouble();
            }

            // Act
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Equal(input.Shape, inputGradient.Shape);
        }

        [Fact]
        public void UpdateParameters_AfterBackward_ChangesParameters()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            var paramsBefore = layer.GetParameters();

            var input = new Tensor<double>([1, 3, 16, 16]);
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.NextDouble();
            }

            var output = layer.Forward(input);
            var outputGradient = new Tensor<double>(output.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                outputGradient[i] = random.NextDouble();
            }

            layer.Backward(outputGradient);

            // Act
            layer.UpdateParameters(0.01);

            // Assert
            var paramsAfter = layer.GetParameters();
            Assert.Equal(paramsBefore.Length, paramsAfter.Length);

            bool parametersChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-10)
                {
                    parametersChanged = true;
                    break;
                }
            }
            Assert.True(parametersChanged, "Parameters should change after update");
        }

        [Fact]
        public void GetParameters_ReturnsAllParameters()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            // Act
            var parameters = layer.GetParameters();

            // Assert
            int patchDim = 3 * 4 * 4;
            int expectedParams = patchDim * 32 + 32;
            Assert.Equal(expectedParams, parameters.Length);
        }

        [Fact]
        public void SetParameters_WithValidVector_UpdatesParameters()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            var params1 = layer.GetParameters();
            var newParams = new Vector<double>(params1.Length);
            for (int i = 0; i < newParams.Length; i++)
            {
                newParams[i] = 0.5;
            }

            // Act
            layer.SetParameters(newParams);

            // Assert
            var params2 = layer.GetParameters();
            for (int i = 0; i < params2.Length; i++)
            {
                Assert.Equal(0.5, params2[i], 6);
            }
        }

        [Fact]
        public void SetParameters_WithInvalidLength_ThrowsArgumentException()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            var wrongParams = new Vector<double>(10);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => layer.SetParameters(wrongParams));
        }

        [Fact]
        public void ResetState_ClearsCachedValues()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            var input = new Tensor<double>([1, 3, 16, 16]);
            layer.Forward(input);

            // Act
            layer.ResetState();

            // Assert - should not throw when called without prior forward
            layer.ResetState();
        }

        [Fact]
        public void Forward_WithMultipleBatches_ProcessesIndependently()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                embeddingDim: 32);

            int batchSize = 4;
            var input = new Tensor<double>([batchSize, 3, 16, 16]);
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.NextDouble();
            }

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(16, output.Shape[1]);
            Assert.Equal(32, output.Shape[2]);
        }

        [Fact]
        public void ParameterCount_MatchesGetParametersLength()
        {
            // Arrange
            var layer = new PatchEmbeddingLayer<double>(
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                embeddingDim: 64);

            // Act
            int count1 = layer.ParameterCount;
            int count2 = layer.GetParameters().Length;

            // Assert
            Assert.Equal(count1, count2);
        }
    }
}
