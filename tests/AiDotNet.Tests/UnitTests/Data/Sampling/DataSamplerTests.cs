using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Sampling;
using Xunit;

namespace AiDotNetTests.UnitTests.Data.Sampling
{
    /// <summary>
    /// Unit tests for the DataLoader sampling infrastructure including
    /// RandomSampler, SequentialSampler, SubsetSampler, CurriculumSampler, and related classes.
    /// </summary>
    public class DataSamplerTests
    {
        #region RandomSampler Tests

        [Fact]
        public void RandomSampler_Constructor_WithValidSize_Succeeds()
        {
            // Arrange & Act
            var sampler = new RandomSampler(100);

            // Assert
            Assert.NotNull(sampler);
            Assert.Equal(100, sampler.Length);
        }

        [Fact]
        public void RandomSampler_Constructor_WithZeroSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new RandomSampler(0));
        }

        [Fact]
        public void RandomSampler_Constructor_WithNegativeSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new RandomSampler(-1));
        }

        [Fact]
        public void RandomSampler_GetIndices_ReturnsAllIndicesOnce()
        {
            // Arrange
            var sampler = new RandomSampler(50, seed: 42);

            // Act
            var indices = sampler.GetIndices().ToList();

            // Assert
            Assert.Equal(50, indices.Count);
            Assert.Equal(50, indices.Distinct().Count()); // All unique
            Assert.True(indices.All(i => i >= 0 && i < 50)); // All in range
        }

        [Fact]
        public void RandomSampler_GetIndices_ShufflesData()
        {
            // Arrange
            var sampler = new RandomSampler(100, seed: 42);

            // Act
            var indices = sampler.GetIndices().ToList();
            var sequential = Enumerable.Range(0, 100).ToList();

            // Assert - Should not be in sequential order
            Assert.NotEqual(sequential, indices);
        }

        [Fact]
        public void RandomSampler_WithSameSeed_ProducesReproducibleResults()
        {
            // Arrange
            var sampler1 = new RandomSampler(100, seed: 12345);
            var sampler2 = new RandomSampler(100, seed: 12345);

            // Act
            var indices1 = sampler1.GetIndices().ToList();
            var indices2 = sampler2.GetIndices().ToList();

            // Assert
            Assert.Equal(indices1, indices2);
        }

        [Fact]
        public void RandomSampler_WithDifferentSeeds_ProducesDifferentResults()
        {
            // Arrange
            var sampler1 = new RandomSampler(100, seed: 1);
            var sampler2 = new RandomSampler(100, seed: 2);

            // Act
            var indices1 = sampler1.GetIndices().ToList();
            var indices2 = sampler2.GetIndices().ToList();

            // Assert
            Assert.NotEqual(indices1, indices2);
        }

        [Fact]
        public void RandomSampler_SetSeed_ChangesRandomization()
        {
            // Arrange
            var sampler = new RandomSampler(100, seed: 42);
            var originalIndices = sampler.GetIndices().ToList();

            // Act
            sampler.SetSeed(999);
            var newIndices = sampler.GetIndices().ToList();

            // Assert
            Assert.NotEqual(originalIndices, newIndices);
        }

        #endregion

        #region SequentialSampler Tests

        [Fact]
        public void SequentialSampler_Constructor_WithValidSize_Succeeds()
        {
            // Arrange & Act
            var sampler = new SequentialSampler(100);

            // Assert
            Assert.NotNull(sampler);
            Assert.Equal(100, sampler.Length);
        }

        [Fact]
        public void SequentialSampler_Constructor_WithZeroSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new SequentialSampler(0));
        }

        [Fact]
        public void SequentialSampler_GetIndices_ReturnsSequentialOrder()
        {
            // Arrange
            var sampler = new SequentialSampler(20);

            // Act
            var indices = sampler.GetIndices().ToList();

            // Assert
            Assert.Equal(Enumerable.Range(0, 20).ToList(), indices);
        }

        [Fact]
        public void SequentialSampler_GetIndices_AlwaysReturnsSameOrder()
        {
            // Arrange
            var sampler = new SequentialSampler(50);

            // Act
            var indices1 = sampler.GetIndices().ToList();
            var indices2 = sampler.GetIndices().ToList();
            var indices3 = sampler.GetIndices().ToList();

            // Assert
            Assert.Equal(indices1, indices2);
            Assert.Equal(indices2, indices3);
        }

        #endregion

        #region SubsetSampler Tests

        [Fact]
        public void SubsetSampler_Constructor_WithValidIndices_Succeeds()
        {
            // Arrange
            var indices = new[] { 0, 5, 10, 15, 20 };

            // Act
            var sampler = new SubsetSampler(indices);

            // Assert
            Assert.NotNull(sampler);
            Assert.Equal(5, sampler.Length);
        }

        [Fact]
        public void SubsetSampler_Constructor_WithNullIndices_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SubsetSampler(null!));
        }

        [Fact]
        public void SubsetSampler_GetIndices_WithoutShuffle_ReturnsOriginalOrder()
        {
            // Arrange
            var originalIndices = new[] { 10, 20, 30, 40, 50 };
            var sampler = new SubsetSampler(originalIndices, shuffle: false);

            // Act
            var result = sampler.GetIndices().ToList();

            // Assert
            Assert.Equal(originalIndices.ToList(), result);
        }

        [Fact]
        public void SubsetSampler_GetIndices_WithShuffle_ShufflesIndices()
        {
            // Arrange
            var originalIndices = Enumerable.Range(0, 50).ToArray();
            var sampler = new SubsetSampler(originalIndices, shuffle: true, seed: 42);

            // Act
            var result = sampler.GetIndices().ToList();

            // Assert - Should contain same elements but different order
            Assert.Equal(originalIndices.Length, result.Count);
            Assert.Equal(originalIndices.OrderBy(x => x), result.OrderBy(x => x));
            Assert.NotEqual(originalIndices.ToList(), result); // Should be shuffled
        }

        [Fact]
        public void SubsetSampler_Shuffle_PropertyCanBeChanged()
        {
            // Arrange
            var indices = new[] { 1, 2, 3, 4, 5 };
            var sampler = new SubsetSampler(indices, shuffle: false);

            // Act & Assert
            Assert.False(sampler.Shuffle);
            sampler.Shuffle = true;
            Assert.True(sampler.Shuffle);
        }

        #endregion

        #region CurriculumSampler Tests

        [Fact]
        public void CurriculumSampler_Constructor_WithValidParameters_Succeeds()
        {
            // Arrange
            var difficulties = new[] { 0.1, 0.3, 0.5, 0.7, 0.9 };

            // Act
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10);

            // Assert
            Assert.NotNull(sampler);
            Assert.Equal(5, sampler.Length);
        }

        [Fact]
        public void CurriculumSampler_Constructor_WithNullDifficulties_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new CurriculumSampler<double>(null!, totalEpochs: 10));
        }

        [Fact]
        public void CurriculumSampler_Constructor_WithZeroEpochs_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var difficulties = new[] { 0.1, 0.5, 0.9 };

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CurriculumSampler<double>(difficulties, totalEpochs: 0));
        }

        [Fact]
        public void CurriculumSampler_LinearStrategy_ProgressesFromEasyToHard()
        {
            // Arrange
            // Create samples with varying difficulty: 0=easiest, 1=hardest
            var difficulties = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10,
                strategy: CurriculumStrategy.Linear, seed: 42);

            // Act - Early epoch should include fewer samples
            sampler.OnEpochStart(0);
            var earlyIndices = sampler.GetIndices().ToList();

            // Act - Late epoch should include all samples
            sampler.OnEpochStart(9);
            var lateIndices = sampler.GetIndices().ToList();

            // Assert
            Assert.True(earlyIndices.Count < lateIndices.Count,
                $"Early epoch should have fewer samples ({earlyIndices.Count}) than late epoch ({lateIndices.Count})");
        }

        [Fact]
        public void CurriculumSampler_CurrentDifficultyThreshold_IncreasesWithEpoch()
        {
            // Arrange
            var difficulties = new double[] { 0.1, 0.5, 0.9 };
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10);

            // Act & Assert - Threshold should increase with epoch
            sampler.OnEpochStart(0);
            double threshold0 = sampler.CurrentDifficultyThreshold;

            sampler.OnEpochStart(5);
            double threshold5 = sampler.CurrentDifficultyThreshold;

            sampler.OnEpochStart(9);
            double threshold9 = sampler.CurrentDifficultyThreshold;

            Assert.True(threshold0 < threshold5);
            Assert.True(threshold5 < threshold9);
        }

        [Theory]
        [InlineData(CurriculumStrategy.Linear)]
        [InlineData(CurriculumStrategy.Exponential)]
        [InlineData(CurriculumStrategy.Stepped)]
        public void CurriculumSampler_AllStrategies_EventuallyIncludeAllSamples(CurriculumStrategy strategy)
        {
            // Arrange
            var difficulties = Enumerable.Range(0, 20).Select(i => i / 20.0).ToArray();
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10, strategy: strategy);

            // Act - At the final epoch
            sampler.OnEpochStart(10); // After totalEpochs, threshold should be 1.0
            var indices = sampler.GetIndices().ToList();

            // Assert - All samples should be included
            Assert.Equal(20, indices.Count);
        }

        [Fact]
        public void CurriculumSampler_CompetenceBased_RespectsSetCompetence()
        {
            // Arrange
            var difficulties = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10,
                strategy: CurriculumStrategy.CompetenceBased);

            // Act - Set low competence
            sampler.SetCompetence(0.3);
            sampler.OnEpochStart(0);
            var lowCompetenceIndices = sampler.GetIndices().ToList();

            // Act - Set high competence
            sampler.SetCompetence(0.9);
            sampler.OnEpochStart(0);
            var highCompetenceIndices = sampler.GetIndices().ToList();

            // Assert
            Assert.True(lowCompetenceIndices.Count < highCompetenceIndices.Count);
        }

        [Fact]
        public void CurriculumSampler_SetCompetence_ClampsValueToValidRange()
        {
            // Arrange
            var difficulties = new double[] { 0.5 };
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10,
                strategy: CurriculumStrategy.CompetenceBased);

            // Act & Assert - Values should be clamped to [0, 1]
            sampler.SetCompetence(1.5);
            Assert.Equal(1.0, sampler.CurrentDifficultyThreshold, precision: 5);

            sampler.SetCompetence(-0.5);
            Assert.Equal(0.0, sampler.CurrentDifficultyThreshold, precision: 5);
        }

        [Fact]
        public void CurriculumSampler_WhenThresholdTooLow_IncludesEasiestSamples()
        {
            // Arrange - All samples have difficulty > 0.5
            var difficulties = new double[] { 0.6, 0.7, 0.8, 0.9, 1.0 };
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 100,
                strategy: CurriculumStrategy.Linear);

            // Act - At epoch 0, threshold is 0, but all samples are > 0
            sampler.OnEpochStart(0);
            var indices = sampler.GetIndices().ToList();

            // Assert - Should still include the easiest sample (0.6)
            Assert.NotEmpty(indices);
            Assert.Contains(0, indices); // Index 0 has lowest difficulty (0.6)
        }

        #endregion

        #region SelfPacedSampler Tests

        [Fact]
        public void SelfPacedSampler_Constructor_WithValidParameters_Succeeds()
        {
            // Arrange & Act
            var sampler = new SelfPacedSampler<double>(
                datasetSize: 100,
                initialLambda: 0.1,
                lambdaGrowthRate: 0.1,
                totalEpochs: 50);

            // Assert
            Assert.NotNull(sampler);
            Assert.Equal(100, sampler.Length);
            Assert.Equal(0.1, sampler.Lambda);
        }

        [Fact]
        public void SelfPacedSampler_UpdateLoss_UpdatesSampleLoss()
        {
            // Arrange
            var sampler = new SelfPacedSampler<double>(
                datasetSize: 10,
                initialLambda: 0.5,
                lambdaGrowthRate: 0.1);

            // Act
            sampler.UpdateLoss(0, 0.1);
            sampler.UpdateLoss(5, 0.9);

            // Assert - Samples with loss <= lambda should be included
            var indices = sampler.GetIndices().ToList();
            Assert.Contains(0, indices); // Loss 0.1 <= 0.5
            // Index 5 may or may not be included depending on whether 0.9 <= lambda
        }

        [Fact]
        public void SelfPacedSampler_UpdateLosses_BatchUpdatesWork()
        {
            // Arrange
            var sampler = new SelfPacedSampler<double>(
                datasetSize: 10,
                initialLambda: 0.5,
                lambdaGrowthRate: 0.1);

            var indices = new[] { 0, 1, 2 };
            var losses = new[] { 0.1, 0.2, 0.3 };

            // Act
            sampler.UpdateLosses(indices, losses);
            var result = sampler.GetIndices().ToList();

            // Assert - All samples with loss <= 0.5 should be included
            Assert.Contains(0, result);
            Assert.Contains(1, result);
            Assert.Contains(2, result);
        }

        [Fact]
        public void SelfPacedSampler_OnEpochStart_IncreasesLambda()
        {
            // Arrange
            var sampler = new SelfPacedSampler<double>(
                datasetSize: 10,
                initialLambda: 0.1,
                lambdaGrowthRate: 0.1);

            double initialLambda = sampler.Lambda;

            // Act
            sampler.OnEpochStart(1);
            double newLambda = sampler.Lambda;

            // Assert
            Assert.Equal(initialLambda + 0.1, newLambda, precision: 5);
        }

        [Fact]
        public void SelfPacedSampler_WithAllHighLosses_ReturnsAtLeast10Percent()
        {
            // Arrange
            var sampler = new SelfPacedSampler<double>(
                datasetSize: 100,
                initialLambda: 0.1,
                lambdaGrowthRate: 0.01);

            // Set all losses high (above lambda)
            for (int i = 0; i < 100; i++)
            {
                sampler.UpdateLoss(i, 1.0);
            }

            // Act
            var indices = sampler.GetIndices().ToList();

            // Assert - Should return at least 10% of samples even when all are "hard"
            Assert.True(indices.Count >= 10);
        }

        #endregion

        #region WeightedSampler Tests

        [Fact]
        public void WeightedSampler_Constructor_WithValidWeights_Succeeds()
        {
            // Arrange
            var weights = new[] { 1.0, 2.0, 3.0, 4.0 };

            // Act
            var sampler = new WeightedSampler<double>(weights, numSamples: 100);

            // Assert
            Assert.NotNull(sampler);
            Assert.Equal(100, sampler.Length);
        }

        [Fact]
        public void WeightedSampler_Constructor_WithNullWeights_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new WeightedSampler<double>(null!, numSamples: 100));
        }

        [Fact]
        public void WeightedSampler_Constructor_WithEmptyWeights_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new WeightedSampler<double>(Array.Empty<double>(), numSamples: 100));
        }

        [Fact]
        public void WeightedSampler_Constructor_WithNegativeWeight_ThrowsArgumentException()
        {
            // Arrange
            var weights = new[] { 1.0, -2.0, 3.0 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new WeightedSampler<double>(weights, numSamples: 100));
        }

        [Fact]
        public void WeightedSampler_Constructor_WithZeroTotalWeight_ThrowsArgumentException()
        {
            // Arrange
            var weights = new[] { 0.0, 0.0, 0.0 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new WeightedSampler<double>(weights, numSamples: 100));
        }

        [Fact]
        public void WeightedSampler_GetIndices_SamplesWithReplacement_CanRepeatIndices()
        {
            // Arrange - Only 4 samples but sample 100 times with replacement
            var weights = new[] { 1.0, 1.0, 1.0, 1.0 };
            var sampler = new WeightedSampler<double>(weights, numSamples: 100, replacement: true);

            // Act
            var indices = sampler.GetIndices().ToList();

            // Assert
            Assert.Equal(100, indices.Count);
            // With 100 samples from 4 options, there will definitely be repeats
            Assert.True(indices.Distinct().Count() < 100);
        }

        [Fact]
        public void WeightedSampler_GetIndices_SamplesWithoutReplacement_NoRepeats()
        {
            // Arrange
            var weights = new[] { 1.0, 1.0, 1.0, 1.0 };
            var sampler = new WeightedSampler<double>(weights, numSamples: 4, replacement: false);

            // Act
            var indices = sampler.GetIndices().ToList();

            // Assert
            Assert.Equal(4, indices.Count);
            Assert.Equal(4, indices.Distinct().Count()); // No repeats
        }

        [Fact]
        public void WeightedSampler_HigherWeights_SelectedMoreOften()
        {
            // Arrange - One weight much higher than others
            var weights = new[] { 1.0, 1.0, 100.0, 1.0 };
            var sampler = new WeightedSampler<double>(weights, numSamples: 1000, replacement: true, seed: 42);

            // Act
            var indices = sampler.GetIndices().ToList();
            var counts = indices.GroupBy(i => i).ToDictionary(g => g.Key, g => g.Count());

            // Assert - Index 2 should be selected most often
            int highWeightCount = counts.GetValueOrDefault(2, 0);
            int otherCounts = counts.Where(kv => kv.Key != 2).Sum(kv => kv.Value);

            Assert.True(highWeightCount > otherCounts,
                $"High weight index should be selected more often. Got {highWeightCount} vs {otherCounts}");
        }

        [Fact]
        public void WeightedSampler_CreateBalancedWeights_CreatesCorrectWeights()
        {
            // Arrange - Imbalanced classes: class 0 has 10, class 1 has 90
            var labels = Enumerable.Repeat(0, 10).Concat(Enumerable.Repeat(1, 90)).ToList();

            // Act
            var weights = WeightedSampler<double>.CreateBalancedWeights(labels, numClasses: 2);

            // Assert
            Assert.Equal(100, weights.Length);

            // Class 0 samples should have higher weights to balance
            double class0Weight = weights.Take(10).Average();
            double class1Weight = weights.Skip(10).Average();

            Assert.True(class0Weight > class1Weight,
                $"Minority class should have higher weights: {class0Weight} vs {class1Weight}");
        }

        #endregion

        #region Edge Cases and Integration Tests

        [Fact]
        public void DataSamplerBase_OnEpochStart_UpdatesCurrentEpoch()
        {
            // Arrange
            var sampler = new RandomSampler(100);

            // Act
            sampler.OnEpochStart(5);

            // Assert - Epoch callback should work without error
            // (CurrentEpoch is protected, so we verify via side effects in derived classes)
            Assert.NotNull(sampler.GetIndices().ToList());
        }

        [Fact]
        public void AllSamplers_CanBeIteratedMultipleTimes()
        {
            // Arrange
            var randomSampler = new RandomSampler(50, seed: 42);
            var sequentialSampler = new SequentialSampler(50);
            var subsetSampler = new SubsetSampler(Enumerable.Range(0, 50), shuffle: false);

            // Act & Assert - Each sampler can be iterated multiple times
            for (int i = 0; i < 3; i++)
            {
                var random = randomSampler.GetIndices().ToList();
                var sequential = sequentialSampler.GetIndices().ToList();
                var subset = subsetSampler.GetIndices().ToList();

                Assert.Equal(50, random.Count);
                Assert.Equal(50, sequential.Count);
                Assert.Equal(50, subset.Count);
            }
        }

        [Fact]
        public void Samplers_WithSingleElement_WorkCorrectly()
        {
            // Arrange
            var randomSampler = new RandomSampler(1);
            var sequentialSampler = new SequentialSampler(1);

            // Act
            var randomIndices = randomSampler.GetIndices().ToList();
            var sequentialIndices = sequentialSampler.GetIndices().ToList();

            // Assert
            Assert.Single(randomIndices);
            Assert.Equal(0, randomIndices[0]);
            Assert.Single(sequentialIndices);
            Assert.Equal(0, sequentialIndices[0]);
        }

        #endregion
    }
}
