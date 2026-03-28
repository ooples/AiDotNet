#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
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

        #region GraphAttentionLayer Deserialization Tests

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_SetsInputAndOutputFeatures()
        {
            // Arrange
            var inputShape = new int[] { 32 };
            var outputShape = new int[] { 16 };

            // Act
            var layer = (GraphAttentionLayer<double>)DeserializationHelper.CreateLayerFromType<double>(
                "GraphAttentionLayer`1", inputShape, outputShape);

            // Assert
            Assert.Equal(32, layer.InputFeatures);
            Assert.Equal(16, layer.OutputFeatures);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_DefaultParams_UsesDefaultValues()
        {
            // Arrange
            var inputShape = new int[] { 8 };
            var outputShape = new int[] { 4 };

            // Act — no additionalParams: numHeads defaults to 1, alpha to 0.2, dropout to 0.0
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape, null);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_WithNumHeads_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };
            var additionalParams = new Dictionary<string, object>
            {
                { "NumHeads", 4 }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_WithAlpha_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };
            var additionalParams = new Dictionary<string, object>
            {
                { "Alpha", 0.1 }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_WithDropoutRate_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };
            var additionalParams = new Dictionary<string, object>
            {
                { "DropoutRate", 0.5 }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_WithAllParams_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 32 };
            var outputShape = new int[] { 16 };
            var additionalParams = new Dictionary<string, object>
            {
                { "NumHeads", 2 },
                { "Alpha", 0.3 },
                { "DropoutRate", 0.2 }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_WithSemicolonEncodedNumHeads_CreatesCorrectly()
        {
            // Arrange — semicolon-encoded constructor metadata in the layer type string
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(
                "GraphAttentionLayer`1;NumHeads=4", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_WithSemicolonEncodedAllParams_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(
                "GraphAttentionLayer`1;NumHeads=2;Alpha=0.1;DropoutRate=0.3", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_FloatType_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<float>("GraphAttentionLayer`1", inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<float>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_ImplementsILayer()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape);

            // Assert — returned instance must implement ILayer<T>
            Assert.IsAssignableFrom<ILayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_EmptyInputShape_ThrowsArgumentException()
        {
            // Arrange
            var inputShape = new int[] { };
            var outputShape = new int[] { 8 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape));
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_EmptyOutputShape_ThrowsArgumentException()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                DeserializationHelper.CreateLayerFromType<double>("GraphAttentionLayer`1", inputShape, outputShape));
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_SemicolonParamsMergeWithAdditionalParams()
        {
            // Arrange — NumHeads in the type string, Alpha in additionalParams; merged result drives construction
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };
            var additionalParams = new Dictionary<string, object>
            {
                { "Alpha", 0.15 }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(
                "GraphAttentionLayer`1;NumHeads=2", inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_SupportsTraining()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = (GraphAttentionLayer<double>)DeserializationHelper.CreateLayerFromType<double>(
                "GraphAttentionLayer`1", inputShape, outputShape);

            // Assert — deserialized layer must support training
            Assert.True(layer.SupportsTraining);
        }

        #endregion
    }
}