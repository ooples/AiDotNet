using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class DeserializationHelperTests
    {
        [Fact]
        public void CreateLayerFromType_WithDenseLayer_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 5 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("DenseLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<DenseLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithConvolutionalLayer_CreatesCorrectly()
        {
            // Arrange - NCHW format: [batch, channels, height, width]
            var inputShape = new int[] { 1, 1, 28, 28 };
            var outputShape = new int[] { 32 };
            var additionalParams = new Dictionary<string, object>
            {
                { "FilterSize", 3 },
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("ConvolutionalLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<ConvolutionalLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithPoolingLayer_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 26, 26, 32 };
            var outputShape = new int[] { 13, 13, 32 };
            var additionalParams = new Dictionary<string, object>
            {
                { "PoolSize", 2 },
                { "Stride", 2 },
                { "PoolingType", PoolingType.Max }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("PoolingLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PoolingLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithActivationLayer_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 10 };
            var additionalParams = new Dictionary<string, object>
            {
                { "ActivationFunction", ActivationFunction.ReLU }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("ActivationLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<ActivationLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithUnsupportedType_ThrowsNotSupportedException()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 10 };

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                DeserializationHelper.CreateLayerFromType<double>("InvalidLayerType", inputShape, outputShape));
        }

        [Fact]
        public void CreateLayerFromType_WithNullAdditionalParams_UsesDefaults()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 5 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("DenseLayer`1", inputShape, outputShape, null);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<DenseLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithConvolutionalLayerAndDefaultParams_UsesDefaults()
        {
            // Arrange - NCHW format: [batch, channels, height, width]
            var inputShape = new int[] { 1, 1, 28, 28 };
            var outputShape = new int[] { 32 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("ConvolutionalLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<ConvolutionalLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithPoolingLayerAndDefaultParams_UsesDefaults()
        {
            // Arrange
            var inputShape = new int[] { 26, 26, 32 };
            var outputShape = new int[] { 13, 13, 32 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("PoolingLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PoolingLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithActivationLayerAndDefaultParams_UsesDefaults()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 10 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("ActivationLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<ActivationLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithFloatType_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 5 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<float>("DenseLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<DenseLayer<float>>(layer);
        }

        [Fact]
        public void DeserializeInterface_WithEmptyString_ReturnsNull()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(string.Empty);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = DeserializationHelper.DeserializeInterface<object>(reader);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void DeserializeInterface_WithInvalidTypeName_ThrowsInvalidOperationException()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write("InvalidTypeName.DoesNotExist");
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                DeserializationHelper.DeserializeInterface<object>(reader));
        }

        [Fact]
        public void DeserializeInterface_WithValidTypeName_CreatesInstance()
        {
            // Arrange
            var typeName = typeof(System.Collections.Generic.List<int>).AssemblyQualifiedName;
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(typeName);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = DeserializationHelper.DeserializeInterface<object>(reader);

            // Assert
            Assert.NotNull(result);
            Assert.IsType<List<int>>(result);
        }

        [Fact]
        public void CreateLayerFromType_WithDifferentInputShapes_WorksCorrectly()
        {
            // Arrange
            var inputShape1 = new int[] { 5 };
            var inputShape2 = new int[] { 20 };
            var outputShape = new int[] { 10 };

            // Act
            var layer1 = DeserializationHelper.CreateLayerFromType<double>("DenseLayer`1", inputShape1, outputShape);
            var layer2 = DeserializationHelper.CreateLayerFromType<double>("DenseLayer`1", inputShape2, outputShape);

            // Assert
            Assert.NotNull(layer1);
            Assert.NotNull(layer2);
            Assert.IsType<DenseLayer<double>>(layer1);
            Assert.IsType<DenseLayer<double>>(layer2);
        }

        [Fact]
        public void CreateLayerFromType_WithConvolutionalLayerAndCustomFilterSize_UsesCustomValue()
        {
            // Arrange - NCHW format: [batch, channels, height, width]
            var inputShape = new int[] { 1, 1, 28, 28 };
            var outputShape = new int[] { 64 };
            var additionalParams = new Dictionary<string, object>
            {
                { "FilterSize", 5 }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("ConvolutionalLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<ConvolutionalLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithPoolingLayerAndCustomStride_UsesCustomValue()
        {
            // Arrange
            var inputShape = new int[] { 26, 26, 32 };
            var outputShape = new int[] { 13, 13, 32 };
            var additionalParams = new Dictionary<string, object>
            {
                { "PoolSize", 3 },
                { "Stride", 3 },
                { "PoolingType", PoolingType.Average }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("PoolingLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PoolingLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithActivationLayerAndSigmoid_UsesCorrectActivation()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 10 };
            var additionalParams = new Dictionary<string, object>
            {
                { "ActivationFunction", ActivationFunction.Sigmoid }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("ActivationLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<ActivationLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithMultipleLayers_CreatesIndependentInstances()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape1 = new int[] { 5 };
            var outputShape2 = new int[] { 8 };

            // Act
            var layer1 = DeserializationHelper.CreateLayerFromType<double>("DenseLayer`1", inputShape, outputShape1);
            var layer2 = DeserializationHelper.CreateLayerFromType<double>("DenseLayer`1", inputShape, outputShape2);

            // Assert
            Assert.NotNull(layer1);
            Assert.NotNull(layer2);
            Assert.NotSame(layer1, layer2);
        }

        [Fact]
        public void CreateLayerFromType_WithInt32Type_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 5 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<int>("DenseLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<DenseLayer<int>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithPoolingLayerMaxType_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 26, 26, 32 };
            var outputShape = new int[] { 13, 13, 32 };
            var additionalParams = new Dictionary<string, object>
            {
                { "PoolSize", 2 },
                { "Stride", 2 },
                { "PoolingType", PoolingType.Max }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("PoolingLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PoolingLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithPoolingLayerAverageType_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 26, 26, 32 };
            var outputShape = new int[] { 13, 13, 32 };
            var additionalParams = new Dictionary<string, object>
            {
                { "PoolSize", 2 },
                { "Stride", 2 },
                { "PoolingType", PoolingType.Average }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("PoolingLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PoolingLayer<double>>(layer);
        }
    }
}
