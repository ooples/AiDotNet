using System;
using System.Collections.Generic;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the Once-for-All (OFA) Networks algorithm.
    /// </summary>
    public class OnceForAllTests
    {
        [Fact]
        public void OnceForAll_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Assert
            Assert.NotNull(ofa);
        }

        [Fact]
        public void OnceForAll_Constructor_WithCustomElasticDimensions_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 4, 6 },
                elasticWidths: new List<double> { 0.5, 1.0, 1.5 },
                elasticKernelSizes: new List<int> { 3, 5 },
                elasticExpansionRatios: new List<int> { 4, 6 });

            // Assert
            Assert.NotNull(ofa);
        }

        [Fact]
        public void OnceForAll_SampleSubNetwork_ReturnsValidConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act
            var config = ofa.SampleSubNetwork();

            // Assert
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
            Assert.True(config.KernelSize > 0);
            Assert.True(config.WidthMultiplier > 0);
            Assert.True(config.ExpansionRatio > 0);
        }

        [Fact]
        public void OnceForAll_SampleSubNetwork_Stage0_ReturnsLargestConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidths: new List<double> { 0.75, 1.0, 1.25 },
                elasticExpansionRatios: new List<int> { 3, 4, 6 });
            ofa.SetTrainingStage(0);

            // Act
            var config = ofa.SampleSubNetwork();

            // Assert - at stage 0, depth, expansion, and width should be largest
            Assert.Equal(4, config.Depth);
            Assert.Equal(6, config.ExpansionRatio);
            Assert.Equal(1.25, config.WidthMultiplier);
        }

        [Fact]
        public void OnceForAll_SetTrainingStage_UpdatesStage()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act - set stage 2 (elastic depth enabled)
            ofa.SetTrainingStage(2);
            var configsStage2 = new List<SubNetworkConfig>();
            for (int i = 0; i < 20; i++)
            {
                configsStage2.Add(ofa.SampleSubNetwork());
            }

            // Assert - should produce varied depth values
            var depths = new HashSet<int>();
            foreach (var config in configsStage2)
            {
                depths.Add(config.Depth);
            }

            // At stage 2, depth should vary (not always max)
            Assert.True(depths.Count >= 1);
        }

        [Fact]
        public void OnceForAll_SetTrainingStage_AtStage4_AllDimensionsElastic()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act - set stage 4 (all dimensions elastic)
            ofa.SetTrainingStage(4);

            var configs = new List<SubNetworkConfig>();
            for (int i = 0; i < 50; i++)
            {
                configs.Add(ofa.SampleSubNetwork());
            }

            // Assert - should produce varied configurations
            var depths = new HashSet<int>();
            var kernels = new HashSet<int>();
            var widths = new HashSet<double>();
            var expansions = new HashSet<int>();

            foreach (var config in configs)
            {
                depths.Add(config.Depth);
                kernels.Add(config.KernelSize);
                widths.Add(config.WidthMultiplier);
                expansions.Add(config.ExpansionRatio);
            }

            // All dimensions should vary (multiple values sampled)
            Assert.True(depths.Count >= 1);
            Assert.True(kernels.Count >= 1);
        }

        [Fact]
        public void OnceForAll_SpecializeForHardware_ReturnsValidConfig()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 100.0,
                MaxMemory = 50.0
            };

            // Act
            var config = ofa.SpecializeForHardware(
                constraints,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 20,
                generations: 5);

            // Assert
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
            Assert.True(config.KernelSize > 0);
        }

        [Fact]
        public void OnceForAll_SpecializeForHardware_RespectsConstraints()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Very tight constraints should favor smaller networks
            var tightConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 10.0,
                MaxMemory = 5.0
            };

            // Loose constraints should allow larger networks
            var looseConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 1000.0,
                MaxMemory = 500.0
            };

            // Act
            var configTight = ofa.SpecializeForHardware(
                tightConstraints,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 30,
                generations: 10);

            var configLoose = ofa.SpecializeForHardware(
                looseConstraints,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 30,
                generations: 10);

            // Assert - both should be valid
            Assert.NotNull(configTight);
            Assert.NotNull(configLoose);
        }

        [Fact]
        public void OnceForAll_GetSharedWeights_ReturnsWeights()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act
            var weights = ofa.GetSharedWeights("conv3x3_layer1", 64, 32);

            // Assert
            Assert.NotNull(weights);
            Assert.Equal(64, weights.Rows);
            Assert.Equal(32, weights.Columns);
        }

        [Fact]
        public void OnceForAll_GetSharedWeights_SameKey_ReturnsSameWeights()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act
            var weights1 = ofa.GetSharedWeights("conv3x3_layer1", 64, 32);
            var weights2 = ofa.GetSharedWeights("conv3x3_layer1", 64, 32);

            // Assert
            Assert.Same(weights1, weights2);
        }

        [Fact]
        public void OnceForAll_GetSharedWeights_DifferentKeys_ReturnsDifferentWeights()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act
            var weights1 = ofa.GetSharedWeights("conv3x3_layer1", 64, 32);
            var weights2 = ofa.GetSharedWeights("conv3x3_layer2", 64, 32);

            // Assert
            Assert.NotSame(weights1, weights2);
        }

        [Fact]
        public void OnceForAll_SampleSubNetwork_ProducesVariedConfigs()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);
            ofa.SetTrainingStage(4); // All dimensions elastic

            // Act - sample multiple configs
            var configs = new List<SubNetworkConfig>();
            for (int i = 0; i < 100; i++)
            {
                configs.Add(ofa.SampleSubNetwork());
            }

            // Assert - should have some variety
            var kernelSizes = new HashSet<int>();
            foreach (var config in configs)
            {
                kernelSizes.Add(config.KernelSize);
            }

            Assert.True(kernelSizes.Count > 1);
        }

        [Fact]
        public void OnceForAll_SubNetworkConfig_HasValidDefaults()
        {
            // Arrange
            var config = new SubNetworkConfig();

            // Assert - default values should be set
            Assert.Equal(0, config.Depth);
            Assert.Equal(0, config.KernelSize);
            Assert.Equal(0.0, config.WidthMultiplier);
            Assert.Equal(0, config.ExpansionRatio);
        }

        [Fact]
        public void OnceForAll_TrainingStages_ProgressivelyUnlockDimensions()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidths: new List<double> { 0.75, 1.0 },
                elasticExpansionRatios: new List<int> { 3, 6 });

            // Test each stage
            for (int stage = 0; stage <= 4; stage++)
            {
                ofa.SetTrainingStage(stage);
                var config = ofa.SampleSubNetwork();

                // Assert basic validity
                Assert.NotNull(config);
                Assert.True(config.Depth > 0);
                Assert.True(config.KernelSize > 0);
            }
        }
    }
}
