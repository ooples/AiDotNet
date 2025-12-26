using System;
using System.Collections.Generic;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the BigNAS (Scaling Up Neural Architecture Search) algorithm.
    /// </summary>
    public class BigNASTests
    {
        [Fact]
        public void BigNAS_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);

            // Assert
            Assert.NotNull(bignas);
        }

        [Fact]
        public void BigNAS_Constructor_WithCustomElasticDimensions_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 4, 6, 8 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0, 1.5 },
                elasticKernelSizes: new List<int> { 3, 5, 7 },
                elasticExpansionRatios: new List<int> { 3, 6 },
                elasticResolutions: new List<int> { 128, 224, 320 });

            // Assert
            Assert.NotNull(bignas);
        }

        [Fact]
        public void BigNAS_SandwichSample_WithSandwichSampling_ReturnsFourConfigs()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace, useSandwichSampling: true);

            // Act
            var samples = bignas.SandwichSample();

            // Assert
            Assert.NotNull(samples);
            Assert.Equal(4, samples.Count);
        }

        [Fact]
        public void BigNAS_SandwichSample_FirstIsTeacher()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace, useSandwichSampling: true);

            // Act
            var samples = bignas.SandwichSample();

            // Assert
            Assert.True(samples[0].IsTeacher);
        }

        [Fact]
        public void BigNAS_SandwichSample_FirstIsLargest()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 4, 6 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0, 1.5 },
                elasticKernelSizes: new List<int> { 3, 5, 7 },
                elasticExpansionRatios: new List<int> { 3, 4, 6 },
                elasticResolutions: new List<int> { 128, 192, 256 },
                useSandwichSampling: true);

            // Act
            var samples = bignas.SandwichSample();
            var largest = samples[0];

            // Assert - should be the largest configuration
            Assert.Equal(6, largest.Depth);
            Assert.Equal(1.5, largest.WidthMultiplier);
            Assert.Equal(7, largest.KernelSize);
            Assert.Equal(6, largest.ExpansionRatio);
            Assert.Equal(256, largest.Resolution);
        }

        [Fact]
        public void BigNAS_SandwichSample_SecondIsSmallest()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 4, 6 },
                elasticWidthMultipliers: new List<double> { 0.5, 1.0, 1.5 },
                elasticKernelSizes: new List<int> { 3, 5, 7 },
                elasticExpansionRatios: new List<int> { 3, 4, 6 },
                elasticResolutions: new List<int> { 128, 192, 256 },
                useSandwichSampling: true);

            // Act
            var samples = bignas.SandwichSample();
            var smallest = samples[1];

            // Assert - should be the smallest configuration
            Assert.Equal(2, smallest.Depth);
            Assert.Equal(0.5, smallest.WidthMultiplier);
            Assert.Equal(3, smallest.KernelSize);
            Assert.Equal(3, smallest.ExpansionRatio);
            Assert.Equal(128, smallest.Resolution);
        }

        [Fact]
        public void BigNAS_SandwichSample_WithoutSandwichSampling_ReturnsRandomConfigs()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace, useSandwichSampling: false);

            // Act
            var samples = bignas.SandwichSample();

            // Assert
            Assert.Equal(4, samples.Count);
            // None should be marked as teacher
            foreach (var sample in samples)
            {
                Assert.False(sample.IsTeacher);
            }
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_ReturnsValidLoss()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var teacherLogits = new Vector<double>(5);
            var studentLogits = new Vector<double>(5);
            var random = RandomHelper.CreateSeededRandom(42);

            for (int i = 0; i < 5; i++)
            {
                teacherLogits[i] = random.NextDouble() * 2 - 1;
                studentLogits[i] = random.NextDouble() * 2 - 1;
            }

            // Act
            double temperature = 3.0;
            var loss = bignas.ComputeDistillationLoss(teacherLogits, studentLogits, temperature);

            // Assert
            Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_SameLogits_ReturnsZeroish()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var logits = new Vector<double>(5);
            for (int i = 0; i < 5; i++)
                logits[i] = i * 0.2;

            // Act - same logits should give ~0 loss
            var loss = bignas.ComputeDistillationLoss(logits, logits, 2.0);

            // Assert - loss should be near zero for identical distributions
            Assert.True(Math.Abs(loss) < 0.1);
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_DifferentLengths_ThrowsException()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var teacherLogits = new Vector<double>(5);
            var studentLogits = new Vector<double>(3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                bignas.ComputeDistillationLoss(teacherLogits, studentLogits, 2.0));
        }

        [Fact]
        public void BigNAS_MultiObjectiveSearch_ReturnsConfigsForAllDevices()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);

            var targetDevices = new List<(string name, HardwareConstraints<double> constraints)>
            {
                ("mobile", new HardwareConstraints<double> { MaxLatency = 30.0, MaxMemory = 50.0 }),
                ("tablet", new HardwareConstraints<double> { MaxLatency = 50.0, MaxMemory = 100.0 }),
                ("desktop", new HardwareConstraints<double> { MaxLatency = 100.0, MaxMemory = 500.0 })
            };

            // Act
            var results = bignas.MultiObjectiveSearch(
                targetDevices,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 10,
                generations: 3);

            // Assert
            Assert.NotNull(results);
            Assert.Equal(3, results.Count);
            Assert.True(results.ContainsKey("mobile"));
            Assert.True(results.ContainsKey("tablet"));
            Assert.True(results.ContainsKey("desktop"));
        }

        [Fact]
        public void BigNAS_MultiObjectiveSearch_ReturnsValidConfigs()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);

            var targetDevices = new List<(string name, HardwareConstraints<double> constraints)>
            {
                ("device1", new HardwareConstraints<double> { MaxLatency = 50.0 })
            };

            // Act
            var results = bignas.MultiObjectiveSearch(
                targetDevices,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 10,
                generations: 3);

            // Assert
            var config = results["device1"];
            Assert.True(config.Depth > 0);
            Assert.True(config.WidthMultiplier > 0);
            Assert.True(config.KernelSize > 0);
            Assert.True(config.ExpansionRatio > 0);
            Assert.True(config.Resolution > 0);
        }

        [Fact]
        public void BigNAS_BigNASConfig_HasCorrectDefaults()
        {
            // Arrange & Act
            var config = new BigNASConfig();

            // Assert
            Assert.Equal(0, config.Depth);
            Assert.Equal(0.0, config.WidthMultiplier);
            Assert.Equal(0, config.KernelSize);
            Assert.Equal(0, config.ExpansionRatio);
            Assert.Equal(0, config.Resolution);
            Assert.False(config.IsTeacher);
        }

        [Fact]
        public void BigNAS_SandwichSample_ProducesVariedConfigs()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace, useSandwichSampling: true);

            // Act - collect multiple samples
            var allDepths = new HashSet<int>();
            for (int i = 0; i < 10; i++)
            {
                var samples = bignas.SandwichSample();
                foreach (var sample in samples)
                {
                    allDepths.Add(sample.Depth);
                }
            }

            // Assert - should have varied depths (at least largest and smallest)
            Assert.True(allDepths.Count >= 2);
        }

        [Fact]
        public void BigNAS_WithDifferentDistillationWeights_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var bignasLow = new BigNAS<double>(searchSpace, distillationWeight: 0.1);
            var bignasHigh = new BigNAS<double>(searchSpace, distillationWeight: 0.9);

            // Assert
            Assert.NotNull(bignasLow);
            Assert.NotNull(bignasHigh);
        }

        #region Edge Case Tests

        [Fact]
        public void BigNAS_SingleElementElasticLists_SamplesCorrectly()
        {
            // Arrange - single element in each list means only one possible config
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 3 },
                elasticWidthMultipliers: new List<double> { 1.0 },
                elasticKernelSizes: new List<int> { 5 },
                elasticExpansionRatios: new List<int> { 4 },
                elasticResolutions: new List<int> { 224 },
                useSandwichSampling: false);

            // Act
            var samples = bignas.SandwichSample();

            // Assert - all samples should have the same values
            foreach (var sample in samples)
            {
                Assert.Equal(3, sample.Depth);
                Assert.Equal(1.0, sample.WidthMultiplier);
                Assert.Equal(5, sample.KernelSize);
                Assert.Equal(4, sample.ExpansionRatio);
                Assert.Equal(224, sample.Resolution);
            }
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_WithVeryLowTemperature_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var teacherLogits = new Vector<double>(5);
            var studentLogits = new Vector<double>(5);

            for (int i = 0; i < 5; i++)
            {
                teacherLogits[i] = i * 0.5;
                studentLogits[i] = i * 0.3;
            }

            // Act - very low temperature makes softmax sharper
            double veryLowTemperature = 0.1;
            var loss = bignas.ComputeDistillationLoss(teacherLogits, studentLogits, veryLowTemperature);

            // Assert - should produce valid (non-NaN, non-Infinity) result
            Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_WithHighTemperature_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var teacherLogits = new Vector<double>(5);
            var studentLogits = new Vector<double>(5);

            for (int i = 0; i < 5; i++)
            {
                teacherLogits[i] = i * 0.5;
                studentLogits[i] = i * 0.3;
            }

            // Act - high temperature makes softmax smoother
            double highTemperature = 100.0;
            var loss = bignas.ComputeDistillationLoss(teacherLogits, studentLogits, highTemperature);

            // Assert - should produce valid result
            Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_SingleElementLogits_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var teacherLogits = new Vector<double>(1);
            var studentLogits = new Vector<double>(1);
            teacherLogits[0] = 1.0;
            studentLogits[0] = 0.5;

            // Act
            var loss = bignas.ComputeDistillationLoss(teacherLogits, studentLogits, 2.0);

            // Assert - single element should give valid loss
            Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
        }

        [Fact]
        public void BigNAS_MultiObjectiveSearch_EmptyDeviceList_ReturnsEmptyResults()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var emptyDevices = new List<(string name, HardwareConstraints<double> constraints)>();

            // Act
            var results = bignas.MultiObjectiveSearch(
                emptyDevices,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 10,
                generations: 3);

            // Assert
            Assert.NotNull(results);
            Assert.Empty(results);
        }

        [Fact]
        public void BigNAS_MultiObjectiveSearch_MinimalPopulationAndGenerations_Works()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var devices = new List<(string name, HardwareConstraints<double> constraints)>
            {
                ("test", new HardwareConstraints<double> { MaxLatency = 50.0 })
            };

            // Act - minimal viable population/generations
            var results = bignas.MultiObjectiveSearch(
                devices,
                inputChannels: 32,
                spatialSize: 14,
                populationSize: 2,
                generations: 1);

            // Assert
            Assert.NotNull(results);
            Assert.Single(results);
            Assert.True(results.ContainsKey("test"));
        }

        [Fact]
        public void BigNAS_SandwichSample_WithSingleElementLists_LargestEqualsSmallest()
        {
            // Arrange - single element in each list
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(
                searchSpace,
                elasticDepths: new List<int> { 4 },
                elasticWidthMultipliers: new List<double> { 1.0 },
                elasticKernelSizes: new List<int> { 5 },
                elasticExpansionRatios: new List<int> { 4 },
                elasticResolutions: new List<int> { 224 },
                useSandwichSampling: true);

            // Act
            var samples = bignas.SandwichSample();

            // Assert - largest (index 0) and smallest (index 1) should be identical
            Assert.Equal(samples[0].Depth, samples[1].Depth);
            Assert.Equal(samples[0].WidthMultiplier, samples[1].WidthMultiplier);
            Assert.Equal(samples[0].KernelSize, samples[1].KernelSize);
            Assert.Equal(samples[0].ExpansionRatio, samples[1].ExpansionRatio);
            Assert.Equal(samples[0].Resolution, samples[1].Resolution);
        }

        [Fact]
        public void BigNAS_ComputeDistillationLoss_WithZeroLogits_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var bignas = new BigNAS<double>(searchSpace);
            var teacherLogits = new Vector<double>(5);
            var studentLogits = new Vector<double>(5);

            // All zeros
            for (int i = 0; i < 5; i++)
            {
                teacherLogits[i] = 0.0;
                studentLogits[i] = 0.0;
            }

            // Act
            var loss = bignas.ComputeDistillationLoss(teacherLogits, studentLogits, 2.0);

            // Assert - same distributions should give near-zero loss
            Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
            Assert.True(Math.Abs(loss) < 0.1);
        }

        #endregion
    }
}
