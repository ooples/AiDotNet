using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Inference;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Attention;
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
            // Arrange
            var inputShape = new int[] { 28, 28, 1 };
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
            // Arrange
            var inputShape = new int[] { 28, 28, 1 };
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
            // Arrange
            var inputShape = new int[] { 28, 28, 1 };
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

        [Fact]
        public void DeserializeInterface_WhenTypeDoesNotImplementInterface_Throws()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(typeof(List<int>).AssemblyQualifiedName);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                DeserializationHelper.DeserializeInterface<IDisposable>(reader));
        }

        private sealed class NoDefaultCtorDisposable : IDisposable
        {
            public NoDefaultCtorDisposable(int _) { }
            public void Dispose() { }
        }

        [Fact]
        public void DeserializeInterface_WhenNoParameterlessCtor_ReturnsNull()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(typeof(NoDefaultCtorDisposable).AssemblyQualifiedName);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = DeserializationHelper.DeserializeInterface<IDisposable>(reader);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void CreateLayerFromType_WithAttentionLayers_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 8, 16 };
            var outputShape = new int[] { 8, 16 };

            // Act & Assert
            Assert.IsType<MultiHeadAttentionLayer<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(MultiHeadAttentionLayer<>).Name, inputShape, outputShape));

            Assert.IsType<SelfAttentionLayer<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(SelfAttentionLayer<>).Name, inputShape, outputShape));

            Assert.IsType<FlashAttentionLayer<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(FlashAttentionLayer<>).Name, inputShape, outputShape,
                    new Dictionary<string, object> { { "UseCausalMask", true } }));

            Assert.IsType<CachedMultiHeadAttention<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(CachedMultiHeadAttention<>).Name, inputShape, outputShape));

            Assert.IsType<PagedCachedMultiHeadAttention<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(PagedCachedMultiHeadAttention<>).Name, inputShape, outputShape));
        }

        [Fact]
        public void CreateLayerFromType_WithAttentionLayer_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(typeof(AttentionLayer<>).Name, inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<AttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithGraphAttentionLayer_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 8 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(typeof(GraphAttentionLayer<>).Name, inputShape, outputShape,
                new Dictionary<string, object> { { "NumHeads", 2 } });

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<GraphAttentionLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithDropoutAndLayerNorm_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 16 };
            var outputShape = new int[] { 16 };

            // Act & Assert
            Assert.IsType<DropoutLayer<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(DropoutLayer<>).Name, inputShape, outputShape));

            Assert.IsType<LayerNormalizationLayer<double>>(
                DeserializationHelper.CreateLayerFromType<double>(typeof(LayerNormalizationLayer<>).Name, inputShape, outputShape));
        }

        [Fact]
        public void CreateLayerFromType_WithPositionalEncoding_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 128, 16 };
            var outputShape = new int[] { 128, 16 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(typeof(PositionalEncodingLayer<>).Name, inputShape, outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PositionalEncodingLayer<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithEncodedParamsInIdentifier_Works()
        {
            // Arrange
            var inputShape = new int[] { 8, 16 };
            var outputShape = new int[] { 8, 16 };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(
                $"{typeof(PagedCachedMultiHeadAttention<>).Name};HeadCount=2;UseCausalMask=false",
                inputShape,
                outputShape);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<PagedCachedMultiHeadAttention<double>>(layer);
        }

        [Fact]
        public void CreateLayerFromType_WithMultiLoRAAdapter_CreatesCorrectly()
        {
            // Arrange
            var inputShape = new int[] { 10 };
            var outputShape = new int[] { 5 };

            var additionalParams = new Dictionary<string, object>
            {
                { "Tasks", "taskA|taskB" },
                { "TaskRanks", "2|4" },
                { "TaskAlphas", "1.0|2.0" },
                { "CurrentTask", Uri.EscapeDataString("taskB") }
            };

            // Act
            var layer = DeserializationHelper.CreateLayerFromType<double>(typeof(MultiLoRAAdapter<>).Name, inputShape, outputShape, additionalParams);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<MultiLoRAAdapter<double>>(layer);
        }
    }
}
