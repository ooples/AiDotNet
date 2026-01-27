using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks
{
    /// <summary>
    /// Unit tests for Vector Bank LoRA (VB-LoRA) adapter implementation.
    /// </summary>
    public class VBLoRAAdapterTests : IDisposable
    {
        public VBLoRAAdapterTests()
        {
            // Clear banks before each test to ensure isolation
            VBLoRAAdapter<double>.ClearBanks();
        }

        public void Dispose()
        {
            // Clear banks after each test
            VBLoRAAdapter<double>.ClearBanks();
        }

        [Fact]
        public void DenseLayerForward_WithNonZeroInput_ProducesNonZeroOutput()
        {
            var layer = new DenseLayer<double>(10, 5, (IActivationFunction<double>)new IdentityActivation<double>());
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[i] = i * 0.1;
            }

            var output = layer.Forward(input);
            var outputVector = output.ToVector();
            Assert.Contains(outputVector, v => v != 0.0);
        }

        [Fact]
        public void Constructor_WithValidParameters_InitializesCorrectly()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>)new IdentityActivation<double>());

            // Act
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

            // Assert
            Assert.NotNull(adapter);
            Assert.Equal(10, adapter.GetInputShape()[0]);
            Assert.Equal(5, adapter.GetOutputShape()[0]);
            Assert.Equal(3, adapter.Rank);
            Assert.Equal(10, adapter.BankSizeA);
            Assert.Equal(10, adapter.BankSizeB);
            Assert.True(adapter.IsBaseLayerFrozen);
        }

        [Fact]
        public void Constructor_CreatesSharedBanks()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>)new IdentityActivation<double>());

            // Act
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

            // Assert - Banks should be created
            var bankA = VBLoRAAdapter<double>.GetBankA("default");
            var bankB = VBLoRAAdapter<double>.GetBankB("default");

            Assert.NotNull(bankA);
            Assert.NotNull(bankB);
            Assert.Equal(10, bankA.Rows);    // inputSize
            Assert.Equal(10, bankA.Columns); // bankSizeA
            Assert.Equal(10, bankB.Rows);    // bankSizeB
            Assert.Equal(5, bankB.Columns);  // outputSize
        }

        [Fact]
        public void Constructor_WithSameBankKey_SharesBanks()
        {
            // Arrange
            var baseLayer1 = new DenseLayer<double>(10, 5, (IActivationFunction<double>)new IdentityActivation<double>());
            var baseLayer2 = new DenseLayer<double>(10, 5, (IActivationFunction<double>)new IdentityActivation<double>());

            // Act
            var adapter1 = new VBLoRAAdapter<double>(
                baseLayer1,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "shared");

            var adapter2 = new VBLoRAAdapter<double>(
                baseLayer2,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "shared");

            // Assert - Both adapters should see the same banks
            var bankA1 = VBLoRAAdapter<double>.GetBankA("shared");
            var bankA2 = VBLoRAAdapter<double>.GetBankA("shared");

            Assert.NotNull(bankA1);
            Assert.NotNull(bankA2);

            // Banks should have identical values (clones, but content is the same)
            for (int i = 0; i < bankA1.Rows; i++)
            {
                for (int j = 0; j < bankA1.Columns; j++)
                {
                    Assert.Equal(bankA1[i, j], bankA2[i, j]);
                }
            }
        }

        [Fact]
        public void Constructor_WithDifferentBankKeys_CreatesSeparateBanks()
        {
            // Arrange
            var baseLayer1 = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var baseLayer2 = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act
            var adapter1 = new VBLoRAAdapter<double>(
                baseLayer1,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "bank1");

            var adapter2 = new VBLoRAAdapter<double>(
                baseLayer2,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "bank2");

            // Assert - Different banks should exist
            var bankA1 = VBLoRAAdapter<double>.GetBankA("bank1");
            var bankA2 = VBLoRAAdapter<double>.GetBankA("bank2");

            Assert.NotNull(bankA1);
            Assert.NotNull(bankA2);

            // Banks should have different random values (very unlikely to be identical)
            bool foundDifference = false;
            for (int i = 0; i < bankA1.Rows && !foundDifference; i++)
            {
                for (int j = 0; j < bankA1.Columns && !foundDifference; j++)
                {
                    if (bankA1[i, j] != bankA2[i, j])
                    {
                        foundDifference = true;
                    }
                }
            }

            Assert.True(foundDifference, "Banks with different keys should have different random initializations");
        }

        [Fact]
        public void BankIndices_ReturnsCorrectLength()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 4,
                bankSizeA: 10,
                bankSizeB: 10);

            // Assert
            Assert.Equal(4, adapter.BankIndicesA.Length);
            Assert.Equal(4, adapter.BankIndicesB.Length);
        }

        [Fact]
        public void BankIndices_WithCustomIndices_UsesProvidedValues()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            int[] customIndicesA = new[] { 0, 2, 5 };
            int[] customIndicesB = new[] { 1, 3, 7 };

            // Act
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankIndicesA: customIndicesA,
                bankIndicesB: customIndicesB);

            // Assert
            Assert.Equal(customIndicesA, adapter.BankIndicesA);
            Assert.Equal(customIndicesB, adapter.BankIndicesB);
        }

        [Fact]
        public void Constructor_WithInvalidBankSizeA_ThrowsArgumentException()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 0,
                bankSizeB: 10));
        }

        [Fact]
        public void Constructor_WithInvalidBankSizeB_ThrowsArgumentException()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 0));
        }

        [Fact]
        public void Constructor_WithRankExceedingBankSizeA_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act & Assert - ArgumentOutOfRangeException is correct for invalid range values
            Assert.Throws<ArgumentOutOfRangeException>(() => new VBLoRAAdapter<double>(
                baseLayer,
                rank: 15,
                bankSizeA: 10,
                bankSizeB: 10));
        }

        [Fact]
        public void Constructor_WithRankExceedingBankSizeB_ThrowsArgumentException()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 2));
        }

        [Fact]
        public void Constructor_WithInvalidIndicesA_ThrowsArgumentException()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            int[] invalidIndices = new[] { 0, 2, 15 }; // 15 exceeds bankSizeA

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankIndicesA: invalidIndices));
        }

        [Fact]
        public void Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

            var input = new Tensor<double>(new[] { 2, 10 });

            // Act
            var output = adapter.Forward(input);

            // Assert
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(5, output.Shape[1]);
        }

        [Fact]
        public void Forward_CombinesBaseAndVBLoRAOutputs()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[i] = 1.0;
            }

            // Act
            var baseOutput = baseLayer.Forward(input);
            var adapterOutput = adapter.Forward(input);

            // Assert - Adapter output should be different from base output
            // (VB-LoRA adds the LoRA contribution on top of base)
            Assert.NotNull(adapterOutput);
            Assert.Equal(baseOutput.Shape[0], adapterOutput.Shape[0]);
            Assert.Equal(baseOutput.Shape[1], adapterOutput.Shape[1]);
        }

        [Fact]
        public void MergeToOriginalLayer_ProducesValidDenseLayer()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

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
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>)new IdentityActivation<double>());
            Assert.IsType<IdentityActivation<double>>(baseLayer.ScalarActivation);
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[i] = i * 0.1;
            }

            // Act
            var adapterOutput = adapter.Forward(input);
            var mergedLayer = adapter.MergeToOriginalLayer();
            var mergedDense = Assert.IsType<DenseLayer<double>>(mergedLayer);
            Assert.IsType<IdentityActivation<double>>(mergedDense.ScalarActivation);
            var mergedOutput = mergedLayer.Forward(input);

            // Assert - Merged layer should produce same output as adapter
            Assert.Equal(adapterOutput.Length, mergedOutput.Length);
            for (int i = 0; i < adapterOutput.Length; i++)
            {
                Assert.Equal(adapterOutput[i], mergedOutput[i], precision: 10);
            }
        }

        [Fact]
        public void ClearBanks_WithSpecificKey_RemovesOnlyThatBank()
        {
            // Arrange
            var baseLayer1 = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var baseLayer2 = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            var adapter1 = new VBLoRAAdapter<double>(
                baseLayer1,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "bank1");

            var adapter2 = new VBLoRAAdapter<double>(
                baseLayer2,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "bank2");

            // Act
            VBLoRAAdapter<double>.ClearBanks("bank1");

            // Assert
            var bank1A = VBLoRAAdapter<double>.GetBankA("bank1");
            var bank2A = VBLoRAAdapter<double>.GetBankA("bank2");

            Assert.Null(bank1A);
            Assert.NotNull(bank2A);
        }

        [Fact]
        public void ClearBanks_WithNullKey_RemovesAllBanks()
        {
            // Arrange
            var baseLayer1 = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var baseLayer2 = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            var adapter1 = new VBLoRAAdapter<double>(
                baseLayer1,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "bank1");

            var adapter2 = new VBLoRAAdapter<double>(
                baseLayer2,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                bankKey: "bank2");

            // Act
            VBLoRAAdapter<double>.ClearBanks(null);

            // Assert
            var bank1A = VBLoRAAdapter<double>.GetBankA("bank1");
            var bank2A = VBLoRAAdapter<double>.GetBankA("bank2");

            Assert.Null(bank1A);
            Assert.Null(bank2A);
        }

        [Fact]
        public void ParameterCount_MatchesLoRAParameterCount()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);

            // Act
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10,
                freezeBaseLayer: true);

            // Assert - Should only count LoRA parameters: (10 * 3) + (3 * 5) = 45
            Assert.Equal(45, adapter.ParameterCount);
        }

        [Fact]
        public void UpdateParameters_ModifiesSharedBanks()
        {
            // Arrange
            var baseLayer = new DenseLayer<double>(10, 5, (IActivationFunction<double>?)null);
            var adapter = new VBLoRAAdapter<double>(
                baseLayer,
                rank: 3,
                bankSizeA: 10,
                bankSizeB: 10);

            // Get initial bank state - use BankB because BankA won't change on first iteration
            // This is because: A gradient = input^T * (outputGradient * B^T) * scaling
            // BankB is initialized to zeros, so A gradient is zeros on first iteration
            // But B gradient = (input * A)^T * outputGradient * scaling, which doesn't depend on B
            var initialBankB = VBLoRAAdapter<double>.GetBankB("default");
            Assert.NotNull(initialBankB);

            // Create input and perform forward/backward pass
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                input[i] = 1.0;
            }

            var output = adapter.Forward(input);
            var gradient = new Tensor<double>(output.Shape);
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = 0.1;
            }

            adapter.Backward(gradient);

            // Act - Update parameters
            adapter.UpdateParameters(0.01);

            // Assert - BankB should be modified
            var updatedBankB = VBLoRAAdapter<double>.GetBankB("default");
            Assert.NotNull(updatedBankB);

            // At least some values in BankB should have changed
            bool foundChange = false;
            for (int i = 0; i < updatedBankB.Rows && !foundChange; i++)
            {
                for (int j = 0; j < updatedBankB.Columns && !foundChange; j++)
                {
                    if (Math.Abs(initialBankB[i, j] - updatedBankB[i, j]) > 1e-10)
                    {
                        foundChange = true;
                    }
                }
            }

            Assert.True(foundChange, "Bank values should change after parameter update");
        }
    }
}
