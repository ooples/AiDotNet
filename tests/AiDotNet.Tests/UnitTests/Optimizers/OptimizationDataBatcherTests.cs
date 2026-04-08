using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Optimizers
{
    /// <summary>
    /// Unit tests for the OptimizationDataBatcher class which provides batch iteration
    /// utilities for optimization input data as part of the DataLoader API.
    /// </summary>
    public class OptimizationDataBatcherTests
    {
        #region Test Helpers

        /// <summary>
        /// Creates a simple optimization input data with the specified number of samples and features.
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
                    xTrain[i, j] = i * numFeatures + j;
                }
                yTrain[i] = i;
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

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_InitializesSuccessfully()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);

            // Act
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32);

            // Assert
            Assert.NotNull(batcher);
            Assert.Equal(100, batcher.DataSize);
            Assert.Equal(32, batcher.BatchSize);
        }

        [Fact]
        public void Constructor_WithNullInputData_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                    null!, batchSize: 32));
        }

        [Fact]
        public void Constructor_WithZeroBatchSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                    inputData, batchSize: 0));
        }

        [Fact]
        public void Constructor_WithNegativeBatchSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                    inputData, batchSize: -1));
        }

        #endregion

        #region Property Tests

        [Fact]
        public void DataSize_ReturnsCorrectNumberOfSamples()
        {
            // Arrange
            var inputData = CreateTestData(150, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32);

            // Act & Assert
            Assert.Equal(150, batcher.DataSize);
        }

        [Fact]
        public void BatchSize_ReturnsConfiguredBatchSize()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 64);

            // Act & Assert
            Assert.Equal(64, batcher.BatchSize);
        }

        [Theory]
        [InlineData(100, 32, false, 4)]  // 100/32 = 3.125 -> 4 batches (last partial)
        [InlineData(100, 32, true, 3)]   // 100/32 = 3 full batches, drop partial
        [InlineData(100, 50, false, 2)]  // 100/50 = 2 full batches
        [InlineData(100, 100, false, 1)] // Exactly one batch
        [InlineData(100, 150, false, 1)] // Batch larger than data
        public void NumBatches_ReturnsCorrectCount(int numSamples, int batchSize, bool dropLast, int expectedBatches)
        {
            // Arrange
            var inputData = CreateTestData(numSamples, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize, dropLast: dropLast);

            // Act & Assert
            Assert.Equal(expectedBatches, batcher.NumBatches);
        }

        #endregion

        #region GetBatches Tests

        [Fact]
        public void GetBatches_ReturnsCorrectNumberOfBatches()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Equal(4, batches.Count); // 100 samples / 32 batch size = 4 batches (3 full + 1 partial)
        }

        [Fact]
        public void GetBatches_WithDropLast_DropsIncompleteBatch()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false, dropLast: true);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Equal(3, batches.Count); // Only 3 full batches
            foreach (var batch in batches)
            {
                Assert.Equal(32, batch.Indices.Length);
            }
        }

        [Fact]
        public void GetBatches_WithoutShuffle_ReturnsDataInOrder()
        {
            // Arrange
            var inputData = CreateTestData(10, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 3, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Equal(4, batches.Count);

            // First batch should have indices 0, 1, 2
            Assert.Equal(new[] { 0, 1, 2 }, batches[0].Indices);

            // Second batch should have indices 3, 4, 5
            Assert.Equal(new[] { 3, 4, 5 }, batches[1].Indices);

            // Third batch should have indices 6, 7, 8
            Assert.Equal(new[] { 6, 7, 8 }, batches[2].Indices);

            // Fourth batch should have index 9
            Assert.Equal(new[] { 9 }, batches[3].Indices);
        }

        [Fact]
        public void GetBatches_WithShuffle_ReturnsShuffledData()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher1 = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, seed: 42);
            var batcher2 = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false);

            // Act
            var shuffledBatches = batcher1.GetBatches().ToList();
            var orderedBatches = batcher2.GetBatches().ToList();

            // Assert - shuffled should be different from ordered
            bool anyDifferent = false;
            for (int i = 0; i < shuffledBatches.Count && i < orderedBatches.Count; i++)
            {
                if (!shuffledBatches[i].Indices.SequenceEqual(orderedBatches[i].Indices))
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.True(anyDifferent, "Shuffled batches should differ from ordered batches");
        }

        [Fact]
        public void GetBatches_WithSameSeed_ProducesReproducibleResults()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher1 = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, seed: 42);
            var batcher2 = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, seed: 42);

            // Act
            var batches1 = batcher1.GetBatches().ToList();
            var batches2 = batcher2.GetBatches().ToList();

            // Assert - same seed should produce identical results
            Assert.Equal(batches1.Count, batches2.Count);
            for (int i = 0; i < batches1.Count; i++)
            {
                Assert.Equal(batches1[i].Indices, batches2[i].Indices);
            }
        }

        [Fact]
        public void GetBatches_ReturnsCorrectBatchData()
        {
            // Arrange
            var inputData = CreateTestData(10, 3);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 2, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert - verify first batch has correct data
            var firstBatch = batches[0];
            Assert.Equal(2, firstBatch.XBatch.Rows);
            Assert.Equal(3, firstBatch.XBatch.Columns);
            Assert.Equal(2, firstBatch.YBatch.Length);

            // Verify the data matches the expected values
            Assert.Equal(0.0, firstBatch.YBatch[0]); // First sample label
            Assert.Equal(1.0, firstBatch.YBatch[1]); // Second sample label
        }

        [Fact]
        public void GetBatches_AllIndicesAreCovered()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: true, seed: 42);

            // Act
            var allIndices = batcher.GetBatches()
                .SelectMany(b => b.Indices)
                .OrderBy(i => i)
                .ToList();

            // Assert - all indices should be present exactly once
            Assert.Equal(100, allIndices.Count);
            Assert.Equal(Enumerable.Range(0, 100), allIndices);
        }

        #endregion

        #region GetBatchIndices Tests

        [Fact]
        public void GetBatchIndices_ReturnsCorrectIndicesOnly()
        {
            // Arrange
            var inputData = CreateTestData(10, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 3, shuffle: false);

            // Act
            var batchIndices = batcher.GetBatchIndices().ToList();

            // Assert
            Assert.Equal(4, batchIndices.Count);
            Assert.Equal(new[] { 0, 1, 2 }, batchIndices[0]);
            Assert.Equal(new[] { 3, 4, 5 }, batchIndices[1]);
            Assert.Equal(new[] { 6, 7, 8 }, batchIndices[2]);
            Assert.Equal(new[] { 9 }, batchIndices[3]);
        }

        [Fact]
        public void GetBatchIndices_WithDropLast_DropsIncompleteBatch()
        {
            // Arrange
            var inputData = CreateTestData(10, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 3, shuffle: false, dropLast: true);

            // Act
            var batchIndices = batcher.GetBatchIndices().ToList();

            // Assert
            Assert.Equal(3, batchIndices.Count);
            foreach (var indices in batchIndices)
            {
                Assert.Equal(3, indices.Length);
            }
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void GetBatches_WithBatchSizeLargerThanData_ReturnsSingleBatch()
        {
            // Arrange
            var inputData = CreateTestData(10, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 100, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Single(batches);
            Assert.Equal(10, batches[0].Indices.Length);
        }

        [Fact]
        public void GetBatches_WithExactlyOneBatch_Works()
        {
            // Arrange
            var inputData = CreateTestData(32, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Single(batches);
            Assert.Equal(32, batches[0].Indices.Length);
        }

        [Fact]
        public void GetBatches_WithSingleSample_Works()
        {
            // Arrange
            var inputData = CreateTestData(1, 5);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 1, shuffle: false);

            // Act
            var batches = batcher.GetBatches().ToList();

            // Assert
            Assert.Single(batches);
            Assert.Equal(new[] { 0 }, batches[0].Indices);
        }

        #endregion

        #region WithSampler Tests

        [Fact]
        public void WithSampler_CreatesNewBatcherWithSampler()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32);

            // Act
            var newBatcher = batcher.WithSampler(null!);

            // Assert
            Assert.NotNull(newBatcher);
            Assert.NotSame(batcher, newBatcher);
        }

        #endregion

        #region Extension Method Tests

        [Fact]
        public void CreateBatcher_ExtensionMethod_CreatesValidBatcher()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);

            // Act
            var batcher = inputData.CreateBatcher(batchSize: 32);

            // Assert
            Assert.NotNull(batcher);
            Assert.Equal(100, batcher.DataSize);
            Assert.Equal(32, batcher.BatchSize);
        }

        [Fact]
        public void CreateBatcher_ExtensionMethod_WithAllOptions_Works()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);

            // Act
            var batcher = inputData.CreateBatcher(
                batchSize: 32,
                shuffle: true,
                dropLast: true,
                seed: 42);

            // Assert
            Assert.NotNull(batcher);
            Assert.Equal(3, batcher.NumBatches); // 100/32 = 3 full batches (dropLast = true)
        }

        #endregion

        #region Vector Input Limitation Tests

        [Fact]
        public void Constructor_WithVectorInput_ThrowsArgumentException()
        {
            // Arrange - Vector<T> as input is not supported by InputHelper.GetBatchSize
            // This is expected because Vector represents 1D data without a batch dimension
            var xTrain = new Vector<double>(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var yTrain = new Vector<double>(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            var inputData = new OptimizationInputData<double, Vector<double>, Vector<double>>
            {
                XTrain = xTrain,
                YTrain = yTrain,
                XValidation = xTrain,
                YValidation = yTrain,
                XTest = xTrain,
                YTest = yTrain
            };

            // Act & Assert - Vector input is not supported
            Assert.Throws<ArgumentException>(() =>
                new OptimizationDataBatcher<double, Vector<double>, Vector<double>>(
                    inputData, batchSize: 3, shuffle: false));
        }

        #endregion

        #region Performance and Memory Tests

        [Fact]
        public void GetBatches_IsLazilyEvaluated()
        {
            // Arrange
            var inputData = CreateTestData(1000, 100);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false);

            // Act - Get the enumerable but don't materialize it
            var batchEnumerable = batcher.GetBatches();

            // Assert - Should be able to get type without full iteration
            Assert.NotNull(batchEnumerable);

            // Only take first batch
            var firstBatch = batchEnumerable.First();
            Assert.Equal(32, firstBatch.Indices.Length);
        }

        [Fact]
        public void GetBatches_CanIterateMultipleTimes()
        {
            // Arrange
            var inputData = CreateTestData(100, 10);
            var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
                inputData, batchSize: 32, shuffle: false);

            // Act - Iterate twice
            var firstIteration = batcher.GetBatches().ToList();
            var secondIteration = batcher.GetBatches().ToList();

            // Assert - Both iterations should have same number of batches
            Assert.Equal(firstIteration.Count, secondIteration.Count);

            // With shuffle: false, both iterations should be identical
            for (int i = 0; i < firstIteration.Count; i++)
            {
                Assert.Equal(firstIteration[i].Indices, secondIteration[i].Indices);
            }
        }

        #endregion
    }
}
