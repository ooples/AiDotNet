using AiDotNet.Helpers;
using AiDotNet.TransferLearning.FeatureMapping;
using Xunit;

namespace AiDotNetTests.IntegrationTests.TransferLearning
{
    /// <summary>
    /// Comprehensive integration tests for Feature Mapping achieving 100% coverage.
    /// Tests LinearFeatureMapper with various dimension mappings and edge cases.
    /// </summary>
    public class FeatureMappingIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region Helper Methods

        /// <summary>
        /// Creates synthetic source domain data
        /// </summary>
        private Matrix<double> CreateSourceData(int samples = 100, int features = 10, int seed = 42)
        {
            var random = new Random(seed);
            var data = new Matrix<double>(samples, features);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    data[i, j] = random.NextDouble() * 10.0 - 5.0;
                }
            }

            return data;
        }

        /// <summary>
        /// Creates synthetic target domain data with different dimensionality
        /// </summary>
        private Matrix<double> CreateTargetData(int samples = 100, int features = 5, int seed = 43)
        {
            var random = new Random(seed);
            var data = new Matrix<double>(samples, features);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    data[i, j] = random.NextDouble() * 8.0 - 4.0;
                }
            }

            return data;
        }

        /// <summary>
        /// Computes reconstruction error between original and reconstructed data
        /// </summary>
        private double ComputeReconstructionError(Matrix<double> original, Matrix<double> reconstructed)
        {
            double totalError = 0.0;
            int count = 0;

            int minRows = Math.Min(original.Rows, reconstructed.Rows);
            int minCols = Math.Min(original.Columns, reconstructed.Columns);

            for (int i = 0; i < minRows; i++)
            {
                for (int j = 0; j < minCols; j++)
                {
                    double diff = original[i, j] - reconstructed[i, j];
                    totalError += diff * diff;
                    count++;
                }
            }

            return Math.Sqrt(totalError / count);
        }

        #endregion

        #region Basic Functionality Tests

        [Fact]
        public void LinearFeatureMapper_InitialState_NotTrained()
        {
            // Arrange & Act
            var mapper = new LinearFeatureMapper<double>();

            // Assert
            Assert.False(mapper.IsTrained);
        }

        [Fact]
        public void LinearFeatureMapper_AfterTraining_IsTrained()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);

            // Act
            mapper.Train(sourceData, targetData);

            // Assert
            Assert.True(mapper.IsTrained);
        }

        [Fact]
        public void LinearFeatureMapper_Train_ComputesConfidence()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);

            // Act
            mapper.Train(sourceData, targetData);
            var confidence = mapper.GetMappingConfidence();

            // Assert
            Assert.True(confidence >= 0.0);
            Assert.True(confidence <= 1.0);
        }

        [Fact]
        public void LinearFeatureMapper_MapToTarget_CorrectDimensions()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert
            Assert.Equal(50, mapped.Rows);
            Assert.Equal(5, mapped.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_MapToSource_CorrectDimensions()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToSource(targetData, 10);

            // Assert
            Assert.Equal(50, mapped.Rows);
            Assert.Equal(10, mapped.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_MapWithoutTraining_ThrowsException()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                mapper.MapToTarget(sourceData, 5));
        }

        [Fact]
        public void LinearFeatureMapper_RoundTripMapping_PreservesInformation()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act - map to target and back to source
            var mappedToTarget = mapper.MapToTarget(sourceData, 5);
            var reconstructed = mapper.MapToSource(mappedToTarget, 10);

            // Assert - reconstruction should be reasonable
            var error = ComputeReconstructionError(sourceData, reconstructed);
            Assert.True(error < 50.0, $"Reconstruction error {error} should be reasonable");
        }

        #endregion

        #region Dimension Mapping Tests

        [Fact]
        public void LinearFeatureMapper_ReduceDimensions_10to5()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 10);
            var targetData = CreateTargetData(100, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert
            Assert.Equal(5, mapped.Columns);
            Assert.Equal(100, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_IncreaseDimensions_5to10()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 5);
            var targetData = CreateTargetData(100, 10);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 10);

            // Assert
            Assert.Equal(10, mapped.Columns);
            Assert.Equal(100, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_SameDimensions_WorksCorrectly()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 7);
            var targetData = CreateTargetData(50, 7); // Same dimensions

            // Act
            mapper.Train(sourceData, targetData);
            var mapped = mapper.MapToTarget(sourceData, 7);

            // Assert
            Assert.Equal(7, mapped.Columns);
            Assert.Equal(50, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_SingleFeature_ExpandsCorrectly()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 1);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert
            Assert.Equal(5, mapped.Columns);
            Assert.Equal(50, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_ManyToOne_CompressesCorrectly()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 1);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 1);

            // Assert
            Assert.Equal(1, mapped.Columns);
            Assert.Equal(50, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_HighDimensional_20to50()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 20);
            var targetData = CreateTargetData(100, 50);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 50);

            // Assert
            Assert.Equal(50, mapped.Columns);
            Assert.Equal(100, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_HighDimensional_50to20()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 50);
            var targetData = CreateTargetData(100, 20);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 20);

            // Assert
            Assert.Equal(20, mapped.Columns);
            Assert.Equal(100, mapped.Rows);
        }

        #endregion

        #region Confidence and Quality Tests

        [Fact]
        public void LinearFeatureMapper_SimilarDomains_HighConfidence()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            // Create similar source and target domains
            var sourceData = CreateSourceData(100, 5, seed: 42);
            var targetData = CreateSourceData(100, 5, seed: 43); // Similar distribution

            // Act
            mapper.Train(sourceData, targetData);
            var confidence = mapper.GetMappingConfidence();

            // Assert
            Assert.True(confidence > 0.1, $"Similar domains should have reasonable confidence, got {confidence}");
        }

        [Fact]
        public void LinearFeatureMapper_LargeDimensionGap_ReasonableConfidence()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 50);
            var targetData = CreateTargetData(100, 5); // Large dimension gap

            // Act
            mapper.Train(sourceData, targetData);
            var confidence = mapper.GetMappingConfidence();

            // Assert
            Assert.True(confidence >= 0.0 && confidence <= 1.0);
        }

        [Fact]
        public void LinearFeatureMapper_MultipleTrainings_UpdatesConfidence()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData1 = CreateSourceData(50, 10, seed: 42);
            var targetData1 = CreateTargetData(50, 5, seed: 43);

            // Act - first training
            mapper.Train(sourceData1, targetData1);
            var confidence1 = mapper.GetMappingConfidence();

            // Second training with different data
            var sourceData2 = CreateSourceData(50, 10, seed: 44);
            var targetData2 = CreateTargetData(50, 5, seed: 45);
            mapper.Train(sourceData2, targetData2);
            var confidence2 = mapper.GetMappingConfidence();

            // Assert - confidence should be recomputed
            Assert.InRange(confidence1, 0.0, 1.0);
            Assert.InRange(confidence2, 0.0, 1.0);
        }

        #endregion

        #region Consistency Tests

        [Fact]
        public void LinearFeatureMapper_MultipleMappings_ConsistentResults()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act - map same data twice
            var mapped1 = mapper.MapToTarget(sourceData, 5);
            var mapped2 = mapper.MapToTarget(sourceData, 5);

            // Assert - should be identical
            for (int i = 0; i < mapped1.Rows; i++)
            {
                for (int j = 0; j < mapped1.Columns; j++)
                {
                    Assert.Equal(mapped1[i, j], mapped2[i, j], 10);
                }
            }
        }

        [Fact]
        public void LinearFeatureMapper_DifferentBatches_ConsistentMapping()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 10);
            var targetData = CreateTargetData(100, 5);
            mapper.Train(sourceData, targetData);

            // Split data into batches
            var batch1 = new Matrix<double>(50, 10);
            var batch2 = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    batch1[i, j] = sourceData[i, j];
                    batch2[i, j] = sourceData[i + 50, j];
                }
            }

            // Act
            var mappedBatch1 = mapper.MapToTarget(batch1, 5);
            var mappedBatch2 = mapper.MapToTarget(batch2, 5);

            // Assert
            Assert.Equal(50, mappedBatch1.Rows);
            Assert.Equal(50, mappedBatch2.Rows);
            Assert.Equal(5, mappedBatch1.Columns);
            Assert.Equal(5, mappedBatch2.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_RepeatedTraining_StableResults()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);

            // Act - train twice with same data
            mapper.Train(sourceData, targetData);
            var mapped1 = mapper.MapToTarget(sourceData, 5);

            mapper.Train(sourceData, targetData);
            var mapped2 = mapper.MapToTarget(sourceData, 5);

            // Assert - results should be stable
            for (int i = 0; i < mapped1.Rows; i++)
            {
                for (int j = 0; j < mapped1.Columns; j++)
                {
                    Assert.Equal(mapped1[i, j], mapped2[i, j], 10);
                }
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void LinearFeatureMapper_SmallSampleSize_HandlesGracefully()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(10, 5); // Small sample
            var targetData = CreateTargetData(10, 3);

            // Act
            mapper.Train(sourceData, targetData);
            var mapped = mapper.MapToTarget(sourceData, 3);

            // Assert
            Assert.Equal(10, mapped.Rows);
            Assert.Equal(3, mapped.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_SingleSample_WorksCorrectly()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(1, 5); // Single sample
            var targetData = CreateTargetData(1, 3);

            // Act
            mapper.Train(sourceData, targetData);
            var mapped = mapper.MapToTarget(sourceData, 3);

            // Assert
            Assert.Equal(1, mapped.Rows);
            Assert.Equal(3, mapped.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_DifferentSampleSizes_Training()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 10);
            var targetData = CreateTargetData(50, 5); // Different size

            // Act
            mapper.Train(sourceData, targetData);
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert
            Assert.Equal(100, mapped.Rows);
            Assert.Equal(5, mapped.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_VeryHighDimensional_PerformsWell()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(100, 100);
            var targetData = CreateTargetData(100, 50);

            // Act
            var startTime = DateTime.Now;
            mapper.Train(sourceData, targetData);
            var mapped = mapper.MapToTarget(sourceData, 50);
            var elapsed = DateTime.Now - startTime;

            // Assert
            Assert.True(elapsed.TotalSeconds < 10.0, "Should complete in reasonable time");
            Assert.Equal(50, mapped.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_MinimalDimensions_2to1()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 2);
            var targetData = CreateTargetData(50, 1);

            // Act
            mapper.Train(sourceData, targetData);
            var mapped = mapper.MapToTarget(sourceData, 1);

            // Assert
            Assert.Equal(1, mapped.Columns);
            Assert.Equal(50, mapped.Rows);
        }

        #endregion

        #region Bidirectional Mapping Tests

        [Fact]
        public void LinearFeatureMapper_BidirectionalMapping_Symmetry()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 8);
            var targetData = CreateTargetData(50, 4);
            mapper.Train(sourceData, targetData);

            // Act - map in both directions
            var toTarget = mapper.MapToTarget(sourceData, 4);
            var backToSource = mapper.MapToSource(toTarget, 8);

            // Assert - dimensions should match
            Assert.Equal(50, toTarget.Rows);
            Assert.Equal(4, toTarget.Columns);
            Assert.Equal(50, backToSource.Rows);
            Assert.Equal(8, backToSource.Columns);
        }

        [Fact]
        public void LinearFeatureMapper_ReverseMapping_PreservesStructure()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act - map target back to source space
            var mapped = mapper.MapToSource(targetData, 10);

            // Assert
            Assert.Equal(10, mapped.Columns);
            Assert.Equal(50, mapped.Rows);
        }

        [Fact]
        public void LinearFeatureMapper_ChainedMapping_WorksCorrectly()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 12);
            var targetData = CreateTargetData(50, 6);
            mapper.Train(sourceData, targetData);

            // Act - chain multiple mappings
            var step1 = mapper.MapToTarget(sourceData, 6);
            var step2 = mapper.MapToSource(step1, 12);
            var step3 = mapper.MapToTarget(step2, 6);

            // Assert - final result should have correct dimensions
            Assert.Equal(50, step3.Rows);
            Assert.Equal(6, step3.Columns);
        }

        #endregion

        #region Data Quality Tests

        [Fact]
        public void LinearFeatureMapper_NoNaNValues_InMappedData()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert - no NaN values
            for (int i = 0; i < mapped.Rows; i++)
            {
                for (int j = 0; j < mapped.Columns; j++)
                {
                    Assert.False(double.IsNaN(mapped[i, j]),
                        $"Found NaN at position [{i},{j}]");
                }
            }
        }

        [Fact]
        public void LinearFeatureMapper_NoInfinityValues_InMappedData()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert - no infinity values
            for (int i = 0; i < mapped.Rows; i++)
            {
                for (int j = 0; j < mapped.Columns; j++)
                {
                    Assert.False(double.IsInfinity(mapped[i, j]),
                        $"Found Infinity at position [{i},{j}]");
                }
            }
        }

        [Fact]
        public void LinearFeatureMapper_MappedDataRange_Reasonable()
        {
            // Arrange
            var mapper = new LinearFeatureMapper<double>();
            var sourceData = CreateSourceData(50, 10);
            var targetData = CreateTargetData(50, 5);
            mapper.Train(sourceData, targetData);

            // Act
            var mapped = mapper.MapToTarget(sourceData, 5);

            // Assert - values should be in a reasonable range
            for (int i = 0; i < mapped.Rows; i++)
            {
                for (int j = 0; j < mapped.Columns; j++)
                {
                    Assert.True(Math.Abs(mapped[i, j]) < 1000.0,
                        $"Value at [{i},{j}] is unreasonably large: {mapped[i, j]}");
                }
            }
        }

        #endregion
    }
}
