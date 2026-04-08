using System;
using System.Collections.Generic;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the AttentiveNAS (Attention-based Neural Architecture Search) algorithm.
    /// </summary>
    public class AttentiveNASTests
    {
        [Fact]
        public void AttentiveNAS_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);

            // Assert
            Assert.NotNull(attentive);
        }

        [Fact]
        public void AttentiveNAS_Constructor_WithCustomElasticDimensions_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 4, 6, 8 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0, 1.5 },
                elasticKernelSizes: new List<int> { 3, 5, 7 },
                attentionHiddenSize: 64);

            // Assert
            Assert.NotNull(attentive);
        }

        [Fact]
        public void AttentiveNAS_AttentiveSample_ReturnsValidConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var context = new Vector<double>(128);
            for (int i = 0; i < context.Length; i++)
                context[i] = 0.1;

            // Act
            var config = attentive.AttentiveSample(context);

            // Assert
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
            Assert.True(config.WidthMultiplier > 0);
            Assert.True(config.KernelSize > 0);
        }

        [Fact]
        public void AttentiveNAS_AttentiveSample_ReturnsConfigWithEmbedding()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace, attentionHiddenSize: 64);
            var context = new Vector<double>(64);
            for (int i = 0; i < context.Length; i++)
                context[i] = 0.0;

            // Act
            var config = attentive.AttentiveSample(context);

            // Assert
            Assert.NotNull(config.Embedding);
            Assert.Equal(64, config.Embedding.Length);
        }

        [Fact]
        public void AttentiveNAS_CreateContextVector_ReturnsCorrectSize()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            int hiddenSize = 256;
            var attentive = new AttentiveNAS<double>(searchSpace, attentionHiddenSize: hiddenSize);

            // Act
            var context = attentive.CreateContextVector();

            // Assert
            Assert.NotNull(context);
            Assert.Equal(hiddenSize, context.Length);
        }

        [Fact]
        public void AttentiveNAS_CreateContextVector_WithoutPerformanceMemory_ReturnsExplorationContext()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);

            // Act
            var context = attentive.CreateContextVector();

            // Assert
            Assert.NotNull(context);
            // Context values should be within exploration range
            for (int i = 0; i < context.Length; i++)
            {
                Assert.True(Math.Abs(context[i]) <= 1.0);
            }
        }

        [Fact]
        public void AttentiveNAS_UpdateAttention_UpdatesPerformanceMemory()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var context = attentive.CreateContextVector();
            var config = attentive.AttentiveSample(context);

            // Act
            double performance = 0.95;
            double learningRate = 0.001;
            attentive.UpdateAttention(config, performance, learningRate);
            var memory = attentive.GetPerformanceMemory();

            // Assert
            Assert.True(memory.Count > 0);
        }

        [Fact]
        public void AttentiveNAS_GetAttentionWeights_ReturnsValidMatrix()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            int hiddenSize = 64;
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0 },
                elasticKernelSizes: new List<int> { 3, 5 },
                attentionHiddenSize: hiddenSize);

            // Act
            var weights = attentive.GetAttentionWeights();

            // Assert
            Assert.NotNull(weights);
            Assert.Equal(hiddenSize, weights.Rows);
            // Columns = total elastic choices (3 depths + 2 widths + 2 kernels = 7)
            Assert.Equal(7, weights.Columns);
        }

        [Fact]
        public void AttentiveNAS_GetPerformanceMemory_InitiallyEmpty()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);

            // Act
            var memory = attentive.GetPerformanceMemory();

            // Assert
            Assert.NotNull(memory);
            Assert.Empty(memory);
        }

        [Fact]
        public void AttentiveNAS_Search_ReturnsValidConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 100.0,
                MaxMemory = 50.0
            };

            // Act
            var config = attentive.Search(
                constraints,
                inputChannels: 32,
                spatialSize: 14,
                numIterations: 10);

            // Assert
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
            Assert.True(config.WidthMultiplier > 0);
            Assert.True(config.KernelSize > 0);
        }

        [Fact]
        public void AttentiveNAS_Search_PopulatesPerformanceMemory()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 100.0
            };

            // Act
            var config = attentive.Search(
                constraints,
                inputChannels: 32,
                spatialSize: 14,
                numIterations: 20);

            var memory = attentive.GetPerformanceMemory();

            // Assert
            Assert.True(memory.Count > 0);
        }

        [Fact]
        public void AttentiveNAS_AttentiveSample_ConfigWithinElasticRange()
        {
            // Arrange
            var elasticDepths = new List<int> { 2, 4, 6 };
            var elasticWidths = new List<double> { 0.5, 1.0, 1.5 };
            var elasticKernels = new List<int> { 3, 5, 7 };

            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: elasticDepths,
                elasticWidthMultipliers: elasticWidths,
                elasticKernelSizes: elasticKernels);

            var context = attentive.CreateContextVector();

            // Act
            var config = attentive.AttentiveSample(context);

            // Assert
            Assert.Contains(config.Depth, elasticDepths);
            Assert.Contains(config.WidthMultiplier, elasticWidths);
            Assert.Contains(config.KernelSize, elasticKernels);
        }

        [Fact]
        public void AttentiveNAS_MultipleSamples_ProducesVariedConfigs()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);

            // Act - sample multiple times
            var allDepths = new HashSet<int>();
            var allKernels = new HashSet<int>();
            for (int i = 0; i < 50; i++)
            {
                var context = attentive.CreateContextVector();
                var config = attentive.AttentiveSample(context);
                allDepths.Add(config.Depth);
                allKernels.Add(config.KernelSize);
            }

            // Assert - should have varied configurations
            Assert.True(allDepths.Count >= 2);
            Assert.True(allKernels.Count >= 2);
        }

        [Fact]
        public void AttentiveNAS_AttentiveNASConfig_HasCorrectDefaults()
        {
            // Arrange & Act
            var config = new AttentiveNASConfig<double>();

            // Assert
            Assert.Equal(0, config.Depth);
            Assert.Equal(0.0, config.WidthMultiplier);
            Assert.Equal(0, config.KernelSize);
        }

        [Fact]
        public void AttentiveNAS_UpdateAttention_WithMultipleConfigs_AccumulatesMemory()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidthMultipliers: new List<double> { 0.75, 1.0 },
                elasticKernelSizes: new List<int> { 3, 5 });

            // Act - update with different configs
            var context = attentive.CreateContextVector();
            var config1 = attentive.AttentiveSample(context);
            attentive.UpdateAttention(config1, 0.8, 0.001);

            var config2 = attentive.AttentiveSample(context);
            attentive.UpdateAttention(config2, 0.9, 0.001);

            var memory = attentive.GetPerformanceMemory();

            // Assert
            Assert.True(memory.Count >= 1); // At least one entry (could be same config)
        }

        [Fact]
        public void AttentiveNAS_Search_WithTightConstraints_ReturnsValidConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 5.0,  // Very tight
                MaxMemory = 5.0
            };

            // Act
            var config = attentive.Search(
                constraints,
                inputChannels: 32,
                spatialSize: 14,
                numIterations: 15);

            // Assert
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
        }

        [Fact]
        public void AttentiveNAS_CreateContextVector_AfterUpdates_IncludesPerformanceInfo()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);

            // First update with a config
            var initialContext = attentive.CreateContextVector();
            var config = attentive.AttentiveSample(initialContext);
            attentive.UpdateAttention(config, 0.85, 0.001);

            // Act
            var contextAfterUpdate = attentive.CreateContextVector();

            // Assert - first element should reflect average performance
            Assert.True(Math.Abs(contextAfterUpdate[0] - 0.85) < 0.01);
        }

        #region Edge Case Tests

        [Fact]
        public void AttentiveNAS_SingleElementElasticLists_SamplesCorrectly()
        {
            // Arrange - boundary condition with single choices
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 3 },
                elasticWidthMultipliers: new List<double> { 1.0 },
                elasticKernelSizes: new List<int> { 5 },
                attentionHiddenSize: 64);

            var context = attentive.CreateContextVector();

            // Act
            var config = attentive.AttentiveSample(context);

            // Assert - should always return the only choices
            Assert.Equal(3, config.Depth);
            Assert.Equal(1.0, config.WidthMultiplier);
            Assert.Equal(5, config.KernelSize);
        }

        [Fact]
        public void AttentiveNAS_UpdateAttention_WithNegativePerformance_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var context = attentive.CreateContextVector();
            var config = attentive.AttentiveSample(context);

            // Act - negative performance (e.g., penalty-adjusted score)
            double negativePerformance = -5.0;
            attentive.UpdateAttention(config, negativePerformance, 0.001);
            var memory = attentive.GetPerformanceMemory();

            // Assert - should still record the performance
            Assert.True(memory.Count > 0);
            Assert.Contains(memory.Values, v => v < 0);
        }

        [Fact]
        public void AttentiveNAS_UpdateAttention_WithConfigNotInElasticLists_ReturnsEarly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0 },
                elasticKernelSizes: new List<int> { 3, 5 });

            // Create a config with values NOT in the elastic lists
            var invalidConfig = new AttentiveNASConfig<double>
            {
                Depth = 99,  // Not in elastic list
                WidthMultiplier = 99.0,  // Not in elastic list
                KernelSize = 99,  // Not in elastic list
                Embedding = new Vector<double>(128)
            };

            var weightsBefore = attentive.GetAttentionWeights()[0, 0];

            // Act
            attentive.UpdateAttention(invalidConfig, 0.9, 0.001);

            var weightsAfter = attentive.GetAttentionWeights()[0, 0];

            // Assert - weights should not change for invalid config
            Assert.Equal(weightsBefore, weightsAfter);
        }

        [Fact]
        public void AttentiveNAS_Search_ZeroIterations_ReturnsDefaultConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var constraints = new HardwareConstraints<double> { MaxLatency = 100.0 };

            // Act
            var config = attentive.Search(
                constraints,
                inputChannels: 32,
                spatialSize: 14,
                numIterations: 0);

            // Assert - should return default config (not crash)
            Assert.NotNull(config);
        }

        [Fact]
        public void AttentiveNAS_AttentiveSample_WithShortContextVector_HandlesGracefully()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace, attentionHiddenSize: 128);
            var shortContext = new Vector<double>(5); // Much shorter than hidden size
            for (int i = 0; i < shortContext.Length; i++)
                shortContext[i] = 0.1;

            // Act
            var config = attentive.AttentiveSample(shortContext);

            // Assert - should still produce valid config
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
        }

        [Fact]
        public void AttentiveNAS_UpdateAttention_WithZeroLearningRate_NoWeightChange()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0 },
                elasticKernelSizes: new List<int> { 3, 5 });
            var context = attentive.CreateContextVector();
            var config = attentive.AttentiveSample(context);

            var weightsBefore = attentive.GetAttentionWeights()[0, 0];

            // Act - zero learning rate should cause no weight update
            attentive.UpdateAttention(config, 0.9, 0.0);

            var weightsAfter = attentive.GetAttentionWeights()[0, 0];

            // Assert
            Assert.Equal(weightsBefore, weightsAfter);
        }

        [Fact]
        public void AttentiveNAS_AttentiveSample_WithVeryLargeContext_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace, attentionHiddenSize: 64);
            var largeContext = new Vector<double>(1000); // Much larger than hidden size
            for (int i = 0; i < largeContext.Length; i++)
                largeContext[i] = 0.01;

            // Act
            var config = attentive.AttentiveSample(largeContext);

            // Assert - should still work (uses min of context and hidden size)
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
        }

        [Fact]
        public void AttentiveNAS_UpdateAttention_WithZeroPerformance_RecordsCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var attentive = new AttentiveNAS<double>(searchSpace);
            var context = attentive.CreateContextVector();
            var config = attentive.AttentiveSample(context);

            // Act
            attentive.UpdateAttention(config, 0.0, 0.001);
            var memory = attentive.GetPerformanceMemory();

            // Assert
            Assert.True(memory.Count > 0);
        }

        #endregion
    }
}
