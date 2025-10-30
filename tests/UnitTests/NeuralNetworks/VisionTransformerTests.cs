using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks
{
    public class VisionTransformerTests
    {
        [Fact]
        public void Constructor_WithValidParameters_InitializesCorrectly()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 10,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            // Act
            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                numClasses: 10,
                hiddenDim: 64,
                numLayers: 2,
                numHeads: 4,
                mlpDim: 128);

            // Assert
            Assert.NotNull(vit);
            Assert.True(vit.SupportsTraining);
            Assert.True(vit.ParameterCount > 0);
        }

        [Fact]
        public void Constructor_WithInvalidPatchSize_ThrowsArgumentException()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 10,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 7,
                numClasses: 10));
        }

        [Fact]
        public void Predict_WithValidInput_ReturnsCorrectShape()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 10,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                numClasses: 10,
                hiddenDim: 64,
                numLayers: 2,
                numHeads: 4,
                mlpDim: 128);

            int batchSize = 2;
            var input = new Tensor<double>([batchSize, 3, 32, 32]);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.1;
            }

            // Act
            var output = vit.Predict(input);

            // Assert
            Assert.Equal(2, output.Rank);
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(10, output.Shape[1]);
        }

        [Fact]
        public void Predict_OutputSumsToOne_ForEachBatchItem()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 5,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 5,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            int batchSize = 3;
            var input = new Tensor<double>([batchSize, 3, 16, 16]);
            var random = new Random(42);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.NextDouble();
            }

            // Act
            var output = vit.Predict(input);

            // Assert - softmax output should sum to 1 for each batch item
            for (int b = 0; b < batchSize; b++)
            {
                double sum = 0;
                for (int c = 0; c < 5; c++)
                {
                    sum += output[b, c];
                }
                Assert.True(Math.Abs(sum - 1.0) < 0.001, $"Sum for batch {b} is {sum}, expected 1.0");
            }
        }

        [Fact]
        public void Train_UpdatesParameters()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 3,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 3,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            var parametersBefore = vit.GetParameters();

            var input = new Tensor<double>([1, 3, 16, 16]);
            var random = new Random(42);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.NextDouble();
            }

            var expectedOutput = new Tensor<double>([1, 3]);
            expectedOutput[0, 1] = 1.0;

            // Act
            vit.Train(input, expectedOutput);

            // Assert
            var parametersAfter = vit.GetParameters();
            Assert.Equal(parametersBefore.Length, parametersAfter.Length);

            bool parametersChanged = false;
            for (int i = 0; i < parametersBefore.Length; i++)
            {
                if (Math.Abs(parametersBefore[i] - parametersAfter[i]) > 1e-10)
                {
                    parametersChanged = true;
                    break;
                }
            }
            Assert.True(parametersChanged, "Parameters should change after training");
        }

        [Fact]
        public void GetModelMetadata_ReturnsCorrectInformation()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 10,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 32,
                imageWidth: 32,
                channels: 3,
                patchSize: 8,
                numClasses: 10,
                hiddenDim: 64,
                numLayers: 2,
                numHeads: 4,
                mlpDim: 128);

            // Act
            var metadata = vit.GetModelMetadata();

            // Assert
            Assert.Equal("VisionTransformer", metadata.ModelType);
            Assert.True(metadata.ParameterCount > 0);
            Assert.Equal(NeuralNetworkTaskType.Classification, metadata.TaskType);
            Assert.Contains("ImageHeight", metadata.Features.Keys);
            Assert.Equal(32, metadata.Features["ImageHeight"]);
            Assert.Equal(8, metadata.Features["PatchSize"]);
            Assert.Equal(64, metadata.Features["HiddenDim"]);
        }

        [Fact]
        public void SaveAndLoad_PreservesModelState()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 5,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 5,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            var input = new Tensor<double>([1, 3, 16, 16]);
            var random = new Random(42);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.NextDouble();
            }

            var predictionBefore = vit.Predict(input);
            var serialized = vit.Serialize();

            // Act
            var vitLoaded = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 5,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);
            vitLoaded.Deserialize(serialized);
            var predictionAfter = vitLoaded.Predict(input);

            // Assert
            Assert.Equal(predictionBefore.Shape, predictionAfter.Shape);
            for (int i = 0; i < predictionBefore.Length; i++)
            {
                Assert.True(Math.Abs(predictionBefore[i] - predictionAfter[i]) < 1e-6,
                    $"Prediction mismatch at index {i}: {predictionBefore[i]} vs {predictionAfter[i]}");
            }
        }

        [Fact]
        public void UpdateParameters_WithValidVector_UpdatesSuccessfully()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 3,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 3,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            var paramsBefore = vit.GetParameters();
            var newParams = new Vector<double>(paramsBefore.Length);
            for (int i = 0; i < newParams.Length; i++)
            {
                newParams[i] = 0.5;
            }

            // Act
            vit.UpdateParameters(newParams);

            // Assert
            var paramsAfter = vit.GetParameters();
            Assert.Equal(newParams.Length, paramsAfter.Length);
            for (int i = 0; i < newParams.Length; i++)
            {
                Assert.Equal(0.5, paramsAfter[i], 6);
            }
        }

        [Fact]
        public void UpdateParameters_WithInvalidLength_ThrowsArgumentException()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 3,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 3,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            var wrongParams = new Vector<double>(10);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => vit.UpdateParameters(wrongParams));
        }

        [Fact]
        public void DeepCopy_CreatesIndependentCopy()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 3,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 3,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            var paramsBefore = vit.GetParameters();

            // Act
            var vitCopy = (VisionTransformer<double>)vit.DeepCopy();
            var paramsCopy = vitCopy.GetParameters();

            // Modify original
            var newParams = new Vector<double>(paramsBefore.Length);
            for (int i = 0; i < newParams.Length; i++)
            {
                newParams[i] = 999.0;
            }
            vit.UpdateParameters(newParams);

            // Assert
            var paramsAfter = vit.GetParameters();
            for (int i = 0; i < paramsAfter.Length; i++)
            {
                Assert.Equal(999.0, paramsAfter[i], 6);
                Assert.NotEqual(999.0, paramsCopy[i]);
            }
        }

        [Fact]
        public void ParameterCount_ReturnsConsistentValue()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                outputSize: 5,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3,
                taskType: NeuralNetworkTaskType.Classification);

            var vit = new VisionTransformer<double>(
                architecture: architecture,
                imageHeight: 16,
                imageWidth: 16,
                channels: 3,
                patchSize: 4,
                numClasses: 5,
                hiddenDim: 32,
                numLayers: 1,
                numHeads: 2,
                mlpDim: 64);

            // Act
            int count1 = vit.ParameterCount;
            int count2 = vit.GetParameters().Length;

            // Assert
            Assert.Equal(count1, count2);
            Assert.True(count1 > 0);
        }
    }
}
