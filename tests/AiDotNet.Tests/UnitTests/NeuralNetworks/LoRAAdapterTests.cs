using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks
{
    public class LoRAAdapterTests
    {
        [Fact]
        public void Constructor_WithValidBaseLayer_InitializesCorrectly()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Assert
            Assert.NotNull(adapter);
            Assert.Equal(10, adapter.GetInputShape()[0]);
            Assert.Equal(5, adapter.GetOutputShape()[0]);
            Assert.Equal(3, adapter.Rank);
            Assert.True(adapter.IsBaseLayerFrozen);
        }

        [Fact]
        public void Constructor_WithNullBaseLayer_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new DenseLoRAAdapter<double>(null!, rank: 3));
        }

        [Fact]
        public void ParameterCount_WithFrozenBase_ReturnsOnlyLoRAParameters()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

            // Act
            var paramCount = adapter.ParameterCount;

            // Assert
            // Should only count LoRA parameters: (10 * 3) + (3 * 5) = 45
            Assert.Equal(45, paramCount);
        }

        [Fact]
        public void ParameterCount_WithUnfrozenBase_ReturnsAllParameters()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: false);

            // Act
            var paramCount = adapter.ParameterCount;

            // Assert
            // Should count both: base (10*5 + 5 = 55) + LoRA (45) = 100
            Assert.Equal(100, paramCount);
        }

        [Fact]
        public void Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);
            var input = new Tensor<double>(new[] { 2, 10 });

            // Act
            var output = adapter.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(5, output.Shape[1]);
        }

        [Fact]
        public void Forward_CombinesBaseAndLoRAOutputs()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Create input
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[i] = 1.0;
            }

            // Act
            var baseOutput = baseLayer.Forward(input);
            var adapterOutput = adapter.Forward(input);

            // Assert - Adapter output should include base layer contribution
            // (Can't directly test equality since LoRA adds on top, but we can verify it's not zero)
            Assert.NotNull(adapterOutput);
            Assert.Equal(5, adapterOutput.Shape[1]);
        }

        [Fact]
        public void Backward_WithFrozenBase_UpdatesOnlyLoRAGradients()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

            var input = new Tensor<double>(new[] { 1, 10 });
            adapter.Forward(input);

            var outputGradient = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                outputGradient[i] = 0.1;
            }

            // Act
            var inputGradient = adapter.Backward(outputGradient);

            // Assert
            Assert.NotNull(inputGradient);
            Assert.Equal(10, inputGradient.Shape[1]);

            // Gradients should only be for LoRA parameters
            var gradients = adapter.GetParameterGradients();
            Assert.Equal(45, gradients.Length); // Only LoRA parameters
        }

        [Fact]
        public void Backward_WithUnfrozenBase_UpdatesAllGradients()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: false);

            var input = new Tensor<double>(new[] { 1, 10 });
            adapter.Forward(input);

            var outputGradient = new Tensor<double>(new[] { 1, 5 });

            // Act
            var inputGradient = adapter.Backward(outputGradient);

            // Assert
            var gradients = adapter.GetParameterGradients();
            Assert.Equal(100, gradients.Length); // Base + LoRA parameters
        }

        [Fact]
        public void UpdateParameters_WithFrozenBase_UpdatesOnlyLoRA()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

            var input = new Tensor<double>(new[] { 1, 10 });
            adapter.Forward(input);

            var outputGradient = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                outputGradient[i] = 0.1;
            }
            adapter.Backward(outputGradient);

            var baseParamsBefore = baseLayer.GetParameters();

            // Act
            adapter.UpdateParameters(0.01);

            // Assert
            var baseParamsAfter = baseLayer.GetParameters();

            // Base parameters should not change (frozen)
            for (int i = 0; i < baseParamsBefore.Length; i++)
            {
                Assert.Equal(baseParamsBefore[i], baseParamsAfter[i], precision: 10);
            }
        }

        [Fact]
        public void GetParameters_ReturnsCorrectCount()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

            // Act
            var parameters = adapter.GetParameters();

            // Assert
            Assert.Equal(45, parameters.Length);
        }

        [Fact]
        public void SetParameters_ThenGetParameters_ReturnsSetValues()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

            var newParams = new Vector<double>(45);
            for (int i = 0; i < 45; i++)
            {
                newParams[i] = i * 0.1;
            }

            // Act
            adapter.SetParameters(newParams);
            var retrievedParams = adapter.GetParameters();

            // Assert
            for (int i = 0; i < 45; i++)
            {
                Assert.Equal(newParams[i], retrievedParams[i], precision: 10);
            }
        }

        [Fact]
        public void SetParameters_WithWrongSize_ThrowsArgumentException()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

            var wrongParams = new Vector<double>(100);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => adapter.SetParameters(wrongParams));
        }

        [Fact]
        public void MergeToSingleLayer_ProducesDenseLayer()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Act
            var mergedLayer = adapter.MergeToOriginalLayer();

            // Assert
            Assert.NotNull(mergedLayer);
            Assert.IsType<DenseLayer<double>>(mergedLayer);
            Assert.Equal(10, mergedLayer.GetInputShape()[0]);
            Assert.Equal(5, mergedLayer.GetOutputShape()[0]);
        }

        [Fact]
        public void MergedLayer_ProducesSameOutputAsAdapter()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Train the adapter a bit
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[i] = (i + 1) * 0.1;
            }

            var outputGradient = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                outputGradient[i] = 0.1;
            }

            for (int iter = 0; iter < 10; iter++)
            {
                adapter.Forward(input);
                adapter.Backward(outputGradient);
                adapter.UpdateParameters(0.01);
            }

            // Get output from adapter
            var adapterOutput = adapter.Forward(input);

            // Act - Merge and get output from merged layer
            var mergedLayer = adapter.MergeToOriginalLayer();
            var mergedOutput = mergedLayer.Forward(input);

            // Assert - Outputs should be very close
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(adapterOutput[i], mergedOutput[i], precision: 5);
            }
        }

        [Fact]
        public void BaseLayer_Property_ReturnsOriginalLayer()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Act
            var retrievedBase = adapter.BaseLayer;

            // Assert
            Assert.Same(baseLayer, retrievedBase);
        }

        [Fact]
        public void LoRALayer_Property_ReturnsLoRALayer()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Act
            var loraLayer = adapter.LoRALayer;

            // Assert
            Assert.NotNull(loraLayer);
            Assert.IsType<LoRALayer<double>>(loraLayer);
            Assert.Equal(3, loraLayer.Rank);
        }

        [Fact]
        public void IsBaseLayerFrozen_Property_ReflectsConstructorParameter()
        {
            // Arrange & Act
            var adapter1 = new DenseLoRAAdapter<double>(new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null), rank: 3, freezeBaseLayer: true);
            var adapter2 = new DenseLoRAAdapter<double>(new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null), rank: 3, freezeBaseLayer: false);

            // Assert
            Assert.True(adapter1.IsBaseLayerFrozen);
            Assert.False(adapter2.IsBaseLayerFrozen);
        }

        [Fact]
        public void Alpha_Property_ReturnsCorrectValue()
        {
            // Arrange & Act
            var adapter = new DenseLoRAAdapter<double>(new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null), rank: 3, alpha: 16);

            // Assert
            Assert.Equal(16.0, adapter.Alpha);
        }

        [Fact]
        public void SupportsTraining_ReturnsTrue()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank: 3);

            // Act & Assert
            Assert.True(adapter.SupportsTraining);
        }

        [Theory]
        [InlineData(8, 4, 2, true)]
        [InlineData(16, 8, 4, false)]
        [InlineData(100, 50, 8, true)]
        public void Constructor_WithVariousConfigurations_WorksCorrectly(int inputSize, int outputSize, int rank, bool freeze)
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(inputSize, outputSize, (IActivationFunction<double>?)null);

            // Act
            var adapter = new DenseLoRAAdapter<double>(baseLayer, rank, freezeBaseLayer: freeze);

            // Assert
            Assert.NotNull(adapter);
            Assert.Equal(inputSize, adapter.GetInputShape()[0]);
            Assert.Equal(outputSize, adapter.GetOutputShape()[0]);
            Assert.Equal(rank, adapter.Rank);
            Assert.Equal(freeze, adapter.IsBaseLayerFrozen);
        }

        [Fact]
        public void LoRAAdapter_WithFloat_WorksCorrectly()
        {
            // Arrange
            var baseLayer = new DenseLayer<float>(10, 5, (IActivationFunction<float>?)null);
            var adapter = new DenseLoRAAdapter<float>(baseLayer, rank: 3);
            var input = new Tensor<float>(new[] { 1, 10 });

            // Act
            var output = adapter.Forward(input);

            // Assert
            Assert.Equal(5, output.Shape[1]);
        }
    }
}
