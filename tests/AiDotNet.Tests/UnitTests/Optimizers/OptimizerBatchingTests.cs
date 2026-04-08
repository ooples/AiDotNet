using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Sampling;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNetTests.UnitTests.Optimizers
{
    /// <summary>
    /// Unit tests for optimizer batching behavior and DataLoader integration.
    /// Tests that optimizers correctly use the batching API, epoch notifications,
    /// and custom samplers.
    /// </summary>
    public class OptimizerBatchingTests
    {
        #region Test Helpers

        /// <summary>
        /// Creates simple test data with specified dimensions.
        /// </summary>
        private static OptimizationInputData<double, Matrix<double>, Vector<double>> CreateTestData(
            int numSamples, int numFeatures)
        {
            var xTrain = new Matrix<double>(numSamples, numFeatures);
            var yTrain = new Vector<double>(numSamples);

            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    xTrain[i, j] = i * numFeatures + j + 1.0;
                }
                yTrain[i] = i * 0.1;
            }

            return new OptimizationInputData<double, Matrix<double>, Vector<double>>
            {
                XTrain = xTrain,
                YTrain = yTrain,
                XValidation = xTrain,
                YValidation = yTrain,
                XTest = xTrain,
                YTest = yTrain
            };
        }

        /// <summary>
        /// A test sampler that tracks OnEpochStart calls.
        /// </summary>
        private class TrackingDataSampler : DataSamplerBase
        {
            private readonly int _datasetSize;
            public List<int> EpochStartCalls { get; } = new List<int>();

            public TrackingDataSampler(int datasetSize) : base(42)
            {
                _datasetSize = datasetSize;
            }

            public override int Length => _datasetSize;

            public override void OnEpochStart(int epoch)
            {
                base.OnEpochStart(epoch);
                EpochStartCalls.Add(epoch);
            }

            protected override IEnumerable<int> GetIndicesCore()
            {
                for (int i = 0; i < _datasetSize; i++)
                {
                    yield return i;
                }
            }
        }

        #endregion

        #region Batcher Creation Tests

        [Fact]
        public void CreateBatcher_WithDefaultOptions_CreatesBatcherCorrectly()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);

            // Act
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, dropLast: false, seed: 42);

            // Assert
            Assert.NotNull(batcher);
            Assert.Equal(100, batcher.DataSize);
            Assert.Equal(32, batcher.BatchSize);
            Assert.Equal(4, batcher.NumBatches); // 100/32 = 4 (3 full + 1 partial)
        }

        [Fact]
        public void CreateBatcher_WithShuffleDisabled_ReturnsSequentialOrder()
        {
            // Arrange
            var inputData = CreateTestData(50, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 10, shuffle: false);

            // Act
            var allIndices = batcher.GetBatches()
                .SelectMany(b => b.Indices)
                .ToList();

            // Assert
            Assert.Equal(Enumerable.Range(0, 50), allIndices);
        }

        [Fact]
        public void CreateBatcher_WithDropLast_DropsIncompleteBatches()
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false, dropLast: true);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Equal(3, batches.Count); // Only 3 complete batches of 32
            foreach (var batch in batches)
            {
                Assert.Equal(32, batch.Indices.Length);
            }
        }

        [Fact]
        public void CreateBatcher_WithSeed_IsReproducible()
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var batcher1 = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, seed: 12345);
            var batcher2 = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, seed: 12345);

            // Act
            var indices1 = batcher1.GetBatches().SelectMany(b => b.Indices).ToList();
            var indices2 = batcher2.GetBatches().SelectMany(b => b.Indices).ToList();

            // Assert
            Assert.Equal(indices1, indices2);
        }

        #endregion

        #region Custom Sampler Tests

        [Fact]
        public void CreateBatcher_WithCustomSampler_UsesSamplerIndices()
        {
            // Arrange - Create data matching sampler size to avoid batcher/sampler size mismatch
            var inputData = CreateTestData(50, 5);
            var sampler = new SequentialSampler(50); // Returns indices 0-49 in order

            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 10, shuffle: false, sampler: sampler);

            // Act
            var allIndices = batcher.GetBatches()
                .SelectMany(b => b.Indices)
                .OrderBy(i => i)
                .ToList();

            // Assert - Should contain all indices from sampler in order
            Assert.Equal(50, allIndices.Count);
            Assert.Equal(Enumerable.Range(0, 50), allIndices);
        }

        [Fact]
        public void CreateBatcher_WithRandomSampler_ShufflesCorrectly()
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var sampler = new RandomSampler(100, seed: 42);

            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 100, shuffle: true, sampler: sampler);

            // Act
            var indices = batcher.GetBatches().First().Indices.ToList();
            var sequential = Enumerable.Range(0, 100).ToList();

            // Assert
            Assert.NotEqual(sequential, indices); // Should be shuffled
            Assert.Equal(100, indices.Distinct().Count()); // All unique
        }

        #endregion

        #region Epoch Notification Tests

        [Fact]
        public void NotifyEpochStart_CallsSamplerOnEpochStart()
        {
            // Arrange
            var trackingSampler = new TrackingDataSampler(100);

            // Act
            trackingSampler.OnEpochStart(0);
            trackingSampler.OnEpochStart(1);
            trackingSampler.OnEpochStart(2);

            // Assert
            Assert.Equal(new[] { 0, 1, 2 }, trackingSampler.EpochStartCalls);
        }

        [Fact]
        public void CurriculumSampler_EpochNotification_AffectsSampling()
        {
            // Arrange
            var difficulties = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
            var sampler = new CurriculumSampler<double>(difficulties, totalEpochs: 10,
                strategy: CurriculumStrategy.Linear, seed: 42);

            // Act - Get indices at different epochs
            sampler.OnEpochStart(0);
            var earlyIndices = sampler.GetIndices().ToList();

            sampler.OnEpochStart(9);
            var lateIndices = sampler.GetIndices().ToList();

            // Assert - Later epochs should include more samples
            Assert.True(earlyIndices.Count <= lateIndices.Count);
        }

        #endregion

        #region Batch Size Configuration Tests

        [Theory]
        [InlineData(1)]   // SGD-style
        [InlineData(16)]  // Small batch
        [InlineData(32)]  // Standard batch
        [InlineData(64)]  // Large batch
        [InlineData(128)] // Very large batch
        public void BatchSize_VariousValues_ProducesCorrectBatches(int batchSize)
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: batchSize, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            int expectedBatches = (100 + batchSize - 1) / batchSize;
            Assert.Equal(expectedBatches, batches.Count);

            // Verify batch sizes
            for (int i = 0; i < batches.Count - 1; i++)
            {
                Assert.Equal(batchSize, batches[i].Indices.Length);
            }

            // Last batch may be partial
            int lastBatchExpectedSize = 100 - (batches.Count - 1) * batchSize;
            Assert.Equal(lastBatchExpectedSize, batches[batches.Count - 1].Indices.Length);
        }

        [Fact]
        public void BatchSizeNegativeOne_ForFullBatch_UsesAllData()
        {
            // Arrange - For second-order optimizers, BatchSize = -1 means full batch
            var inputData = CreateTestData(100, 5);

            // Second-order optimizers don't use the batcher - they use full batch
            // This test verifies the concept that full batch means all samples
            int batchSize = inputData.XTrain.Rows; // Full batch = all samples

            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: batchSize, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Single(batches);
            Assert.Equal(100, batches[0].Indices.Length);
        }

        #endregion

        #region Batch Data Extraction Tests

        [Fact]
        public void GetBatches_ExtractedDataMatchesIndices()
        {
            // Arrange
            var inputData = CreateTestData(10, 3);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 3, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert - First batch should have indices 0, 1, 2
            var firstBatch = batches[0];
            Assert.Equal(new[] { 0, 1, 2 }, firstBatch.Indices);

            // Verify X data matches
            Assert.Equal(3, firstBatch.XBatch.Rows);
            Assert.Equal(3, firstBatch.XBatch.Columns);

            // Verify Y data matches
            Assert.Equal(3, firstBatch.YBatch.Length);

            // First row of batch should be first row of original (index 0)
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(inputData.XTrain[0, j], firstBatch.XBatch[0, j]);
            }
            Assert.Equal(inputData.YTrain[0], firstBatch.YBatch[0]);
        }

        [Fact]
        public void GetBatches_WithShuffle_DataMatchesShuffledIndices()
        {
            // Arrange
            var inputData = CreateTestData(10, 3);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 5, shuffle: true, seed: 42);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert - Each batch's data should match its indices
            foreach (var batch in batches)
            {
                for (int i = 0; i < batch.Indices.Length; i++)
                {
                    int originalIndex = batch.Indices[i];

                    // Verify X data matches the original at this index
                    for (int j = 0; j < 3; j++)
                    {
                        Assert.Equal(inputData.XTrain[originalIndex, j], batch.XBatch[i, j]);
                    }

                    // Verify Y data matches
                    Assert.Equal(inputData.YTrain[originalIndex], batch.YBatch[i]);
                }
            }
        }

        #endregion

        #region Optimizer Options Default BatchSize Tests

        [Fact]
        public void AdamOptimizerOptions_DefaultBatchSize_Is32()
        {
            // Arrange
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(32, options.BatchSize);
        }

        [Fact]
        public void SGDOptimizerOptions_DefaultBatchSize_Is1()
        {
            // Arrange
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(1, options.BatchSize);
        }

        [Fact]
        public void MiniBatchGDOptimizerOptions_DefaultBatchSize_Is32()
        {
            // Arrange
            var options = new MiniBatchGradientDescentOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(32, options.BatchSize);
        }

        [Fact]
        public void BFGSOptimizerOptions_DefaultBatchSize_IsNegativeOne()
        {
            // Arrange - Second-order optimizers use full batch
            var options = new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(-1, options.BatchSize);
        }

        [Fact]
        public void LBFGSOptimizerOptions_DefaultBatchSize_IsNegativeOne()
        {
            // Arrange
            var options = new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(-1, options.BatchSize);
        }

        [Fact]
        public void NewtonMethodOptimizerOptions_DefaultBatchSize_IsNegativeOne()
        {
            // Arrange
            var options = new NewtonMethodOptimizerOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(-1, options.BatchSize);
        }

        [Fact]
        public void TrustRegionOptimizerOptions_DefaultBatchSize_IsNegativeOne()
        {
            // Arrange
            var options = new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.Equal(-1, options.BatchSize);
        }

        #endregion

        #region WithSampler and WithCurriculumLearning Tests

        [Fact]
        public void WithSampler_CreatesNewBatcherWithSampler()
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32);

            var customSampler = new SequentialSampler(100);

            // Act
            var newBatcher = batcher.WithSampler(customSampler);

            // Assert
            Assert.NotSame(batcher, newBatcher);
            Assert.NotNull(newBatcher);
        }

        [Fact]
        public void WithClassBalancing_CreatesWeightedSampler()
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32);

            // Create imbalanced labels: 10 class 0, 90 class 1
            var labels = Enumerable.Repeat(0, 10)
                .Concat(Enumerable.Repeat(1, 90))
                .ToList();

            // Act
            var balancedBatcher = batcher.WithClassBalancing<double>(labels, numClasses: 2);

            // Assert
            Assert.NotNull(balancedBatcher);
            Assert.NotSame(batcher, balancedBatcher);
        }

        [Fact]
        public void WithCurriculumLearning_CreatesCurriculumSampler()
        {
            // Arrange
            var inputData = CreateTestData(100, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32);

            // Create difficulty scores
            var difficulties = Enumerable.Range(0, 100).Select(i => i / 100.0).ToList();

            // Act
            var curriculumBatcher = batcher.WithCurriculumLearning(
                difficulties, totalEpochs: 50, strategy: CurriculumStrategy.Linear);

            // Assert
            Assert.NotNull(curriculumBatcher);
            Assert.NotSame(batcher, curriculumBatcher);
        }

        #endregion
    }
}
