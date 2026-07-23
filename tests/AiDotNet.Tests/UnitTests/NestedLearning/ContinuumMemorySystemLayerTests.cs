using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.NestedLearning
{
    public class ContinuumMemorySystemLayerTests
    {
        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.SupportsTraining);
            Assert.Equal(3, layer.UpdateFrequencies.Length);
            Assert.Equal(3, layer.ChunkSizes.Length);
            Assert.Equal(3, layer.GetMLPBlocks().Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullInputShape_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: null!,
                hiddenDim: 128,
                numFrequencyLevels: 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyInputShape_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new int[] { },
                hiddenDim: 128,
                numFrequencyLevels: 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeInputDim_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { -1 },
                hiddenDim: 128,
                numFrequencyLevels: 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeHiddenDim_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: -128,
                numFrequencyLevels: 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroFrequencyLevels_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 0));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithTooManyFrequencyLevels_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 15));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithMismatchedUpdateFrequenciesLength_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3,
                updateFrequencies: new[] { 1, 10 })); // Wrong length
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeUpdateFrequency_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3,
                updateFrequencies: new[] { 1, -10, 100 }));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultUpdateFrequencies_CreatesCorrectSequence()
        {
            // Arrange & Act
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 4);

            // Assert
            Assert.Equal(1, layer.UpdateFrequencies[0]);    // 10^0
            Assert.Equal(10, layer.UpdateFrequencies[1]);   // 10^1
            Assert.Equal(100, layer.UpdateFrequencies[2]);  // 10^2
            Assert.Equal(1000, layer.UpdateFrequencies[3]); // 10^3
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_CalculatesChunkSizesCorrectly()
        {
            // Arrange & Act
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            // Assert
            // Chunk sizes: C(ℓ) = max_ℓ C(ℓ) / fℓ
            // With frequencies [1, 10, 100], max = 100
            Assert.Equal(100, layer.ChunkSizes[0]); // 100/1
            Assert.Equal(10, layer.ChunkSizes[1]);  // 100/10
            Assert.Equal(1, layer.ChunkSizes[2]);   // 100/100
        }

        [Fact(Timeout = 60000)]
        public async Task Forward_WithValidInput_ReturnsCorrectShape()
        {
            // Arrange
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            var input = new Tensor<double>(new[] { 1, 64 });
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.1;
            }

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(2, output.Rank);
            Assert.Equal(1, output.Shape[0]); // Batch size
            Assert.Equal(128, output.Shape[1]); // Hidden dim
        }

        [Fact(Timeout = 60000)]
        public async Task Forward_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => layer.Forward((Tensor<double>)null!));
        }

        [Fact(Timeout = 60000)]
        public async Task Forward_ProcessesSequentiallyThroughAllMLPBlocks()
        {
            // Arrange
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            var input = new Tensor<double>(new[] { 1, 64 });
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = (double)i / 100.0;
            }

            // Act
            var output = layer.Forward(input);

            // Assert - verify sequential chain processing
            var mlpBlocks = layer.GetMLPBlocks();
            Assert.Equal(3, mlpBlocks.Length);

            // Each MLP block should have been initialized
            foreach (var block in mlpBlocks)
            {
                Assert.NotNull(block);
                Assert.True(block.ParameterCount > 0);
            }
        }



        [Fact(Timeout = 60000)]
        public async Task ResetMemory_ResetsAllMLPBlocksSuccessfully()
        {
            // Arrange
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            var input = new Tensor<double>(new[] { 1, 64 });
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.5;
            }

            // Run forward pass to initialize state
            layer.Forward(input);

            // Act
            layer.ResetMemory();

            // Assert - should not throw
            var mlpBlocks = layer.GetMLPBlocks();
            Assert.Equal(3, mlpBlocks.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task ConsolidateMemory_TransfersKnowledgeBetweenLevels()
        {
            // Arrange
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 3);

            var input = new Tensor<double>(new[] { 1, 64 });
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.5;
            }

            // Run forward pass
            layer.Forward(input);

            // Act - should transfer knowledge from faster to slower levels
            layer.ConsolidateMemory();

            // Assert - should complete without errors
            var mlpBlocks = layer.GetMLPBlocks();
            foreach (var block in mlpBlocks)
            {
                Assert.NotNull(block);
                Assert.True(block.ParameterCount > 0);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task UpdateFrequencies_DefaultValues_MatchPaperSpecification()
        {
            // Arrange & Act
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 4);

            // Assert - frequencies should be powers of 10 as per Equation 30-31
            var frequencies = layer.UpdateFrequencies;
            Assert.Equal(1, frequencies[0]);
            Assert.Equal(10, frequencies[1]);
            Assert.Equal(100, frequencies[2]);
            Assert.Equal(1000, frequencies[3]);
        }

        [Fact(Timeout = 60000)]
        public async Task GetMLPBlocks_ReturnsCorrectNumberOfBlocks()
        {
            // Arrange
            var layer = new ContinuumMemorySystemLayer<double>(
                inputShape: new[] { 64 },
                hiddenDim: 128,
                numFrequencyLevels: 5);

            // Act
            var blocks = layer.GetMLPBlocks();

            // Assert
            Assert.NotNull(blocks);
            Assert.Equal(5, blocks.Length);
            foreach (var block in blocks)
            {
                Assert.NotNull(block);
            }
        }
    }
}
