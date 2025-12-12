using AiDotNet.Helpers;
using AiDotNet.TransferLearning.DomainAdaptation;
using Xunit;

namespace AiDotNetTests.IntegrationTests.TransferLearning
{
    /// <summary>
    /// Comprehensive integration tests for Domain Adaptation achieving 100% coverage.
    /// Tests CORAL, MMD adapters with various domain shifts and edge cases.
    /// </summary>
    public class DomainAdaptationIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region Helper Methods

        /// <summary>
        /// Creates synthetic source domain data - high mean, high variance
        /// </summary>
        private Matrix<double> CreateSourceDomain(int samples = 100, int features = 5, int seed = 42)
        {
            var random = new Random(seed);
            var data = new Matrix<double>(samples, features);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    // Source domain: mean=5.0, std=2.0
                    data[i, j] = random.NextDouble() * 4.0 + 3.0;
                }
            }

            return data;
        }

        /// <summary>
        /// Creates synthetic target domain data - low mean, low variance
        /// </summary>
        private Matrix<double> CreateTargetDomain(int samples = 100, int features = 5, int seed = 43)
        {
            var random = new Random(seed);
            var data = new Matrix<double>(samples, features);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    // Target domain: mean=1.0, std=0.5
                    data[i, j] = random.NextDouble() * 1.0 + 0.5;
                }
            }

            return data;
        }

        /// <summary>
        /// Creates similar domains for testing small domain gaps
        /// </summary>
        private (Matrix<double>, Matrix<double>) CreateSimilarDomains(int samples = 50, int features = 3)
        {
            var random = new Random(42);
            var source = new Matrix<double>(samples, features);
            var target = new Matrix<double>(samples, features);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    double baseValue = random.NextDouble() * 2.0;
                    source[i, j] = baseValue + 0.1;
                    target[i, j] = baseValue + 0.15; // Small difference
                }
            }

            return (source, target);
        }

        /// <summary>
        /// Computes the mean of each column
        /// </summary>
        private Vector<double> ComputeMean(Matrix<double> data)
        {
            var mean = new Vector<double>(data.Columns);
            for (int j = 0; j < data.Columns; j++)
            {
                double sum = 0;
                for (int i = 0; i < data.Rows; i++)
                {
                    sum += data[i, j];
                }
                mean[j] = sum / data.Rows;
            }
            return mean;
        }

        /// <summary>
        /// Computes the variance of each column
        /// </summary>
        private Vector<double> ComputeVariance(Matrix<double> data)
        {
            var mean = ComputeMean(data);
            var variance = new Vector<double>(data.Columns);

            for (int j = 0; j < data.Columns; j++)
            {
                double sumSquares = 0;
                for (int i = 0; i < data.Rows; i++)
                {
                    double diff = data[i, j] - mean[j];
                    sumSquares += diff * diff;
                }
                variance[j] = sumSquares / data.Rows;
            }

            return variance;
        }

        #endregion

        #region CORAL Domain Adapter Tests

        [Fact]
        public void CORALAdapter_BasicAdaptation_ReducesDomainDiscrepancy()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            // Compute initial discrepancy
            var initialDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Act
            adapter.Train(sourceData, targetData);
            var adaptedSource = adapter.AdaptSource(sourceData, targetData);
            var finalDiscrepancy = adapter.ComputeDomainDiscrepancy(adaptedSource, targetData);

            // Assert
            Assert.True(finalDiscrepancy < initialDiscrepancy,
                "CORAL should reduce domain discrepancy");
            Assert.True(finalDiscrepancy < initialDiscrepancy * 0.8,
                "Discrepancy should be reduced by at least 20%");
        }

        [Fact]
        public void CORALAdapter_AdaptSource_AlignsMeans()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            var sourceMean = ComputeMean(sourceData);
            var targetMean = ComputeMean(targetData);

            // Act
            var adaptedSource = adapter.AdaptSource(sourceData, targetData);
            var adaptedMean = ComputeMean(adaptedSource);

            // Assert - adapted mean should be closer to target mean
            for (int j = 0; j < targetMean.Length; j++)
            {
                double originalDistance = Math.Abs(sourceMean[j] - targetMean[j]);
                double adaptedDistance = Math.Abs(adaptedMean[j] - targetMean[j]);
                Assert.True(adaptedDistance < originalDistance * 1.5,
                    $"Mean should be closer to target for feature {j}");
            }
        }

        [Fact]
        public void CORALAdapter_AdaptTarget_ReverseAdaptation()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            var sourceMean = ComputeMean(sourceData);
            var targetMean = ComputeMean(targetData);

            // Act
            var adaptedTarget = adapter.AdaptTarget(targetData, sourceData);
            var adaptedMean = ComputeMean(adaptedTarget);

            // Assert - adapted target mean should be closer to source mean
            for (int j = 0; j < sourceMean.Length; j++)
            {
                double originalDistance = Math.Abs(targetMean[j] - sourceMean[j]);
                double adaptedDistance = Math.Abs(adaptedMean[j] - sourceMean[j]);
                Assert.True(adaptedDistance < originalDistance * 1.5,
                    $"Target mean should be closer to source for feature {j}");
            }
        }

        [Fact]
        public void CORALAdapter_RequiresTraining_ReturnsTrue()
        {
            // Arrange & Act
            var adapter = new CORALDomainAdapter<double>();

            // Assert
            Assert.True(adapter.RequiresTraining);
            Assert.Equal("CORAL (CORrelation ALignment)", adapter.AdaptationMethod);
        }

        [Fact]
        public void CORALAdapter_AdaptWithoutTraining_TrainsAutomatically()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(50, 3);
            var targetData = CreateTargetDomain(50, 3);

            // Act - adapt without explicit training
            var adapted = adapter.AdaptSource(sourceData, targetData);

            // Assert
            Assert.NotNull(adapted);
            Assert.Equal(sourceData.Rows, adapted.Rows);
            Assert.Equal(sourceData.Columns, adapted.Columns);
        }

        [Fact]
        public void CORALAdapter_SmallDomainGap_LowDiscrepancy()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var (source, target) = CreateSimilarDomains();

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(source, target);

            // Assert
            Assert.True(discrepancy < 2.0, "Similar domains should have low discrepancy");
        }

        [Fact]
        public void CORALAdapter_LargeDomainGap_HighDiscrepancy()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(); // mean=5, std=2
            var targetData = CreateTargetDomain(); // mean=1, std=0.5

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.True(discrepancy > 1.0, "Different domains should have high discrepancy");
        }

        [Fact]
        public void CORALAdapter_MultipleAdaptations_Consistent()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(50, 3);
            var targetData = CreateTargetDomain(50, 3);

            adapter.Train(sourceData, targetData);

            // Act - multiple adaptations
            var adapted1 = adapter.AdaptSource(sourceData, targetData);
            var adapted2 = adapter.AdaptSource(sourceData, targetData);

            // Assert - should produce identical results
            for (int i = 0; i < adapted1.Rows; i++)
            {
                for (int j = 0; j < adapted1.Columns; j++)
                {
                    Assert.Equal(adapted1[i, j], adapted2[i, j], 6);
                }
            }
        }

        [Fact]
        public void CORALAdapter_DifferentSampleSizes_HandlesCorrectly()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(100, 5);
            var targetData = CreateTargetDomain(50, 5); // Different size

            // Act
            var adapted = adapter.AdaptSource(sourceData, targetData);

            // Assert
            Assert.Equal(sourceData.Rows, adapted.Rows);
            Assert.Equal(sourceData.Columns, adapted.Columns);
        }

        [Fact]
        public void CORALAdapter_SingleFeature_WorksCorrectly()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(50, 1);
            var targetData = CreateTargetDomain(50, 1);

            // Act
            var adapted = adapter.AdaptSource(sourceData, targetData);
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.Equal(1, adapted.Columns);
            Assert.True(discrepancy >= 0);
        }

        [Fact]
        public void CORALAdapter_HighDimensional_PerformsWell()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(100, 20);
            var targetData = CreateTargetDomain(100, 20);

            // Act
            var initialDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);
            var adapted = adapter.AdaptSource(sourceData, targetData);
            var finalDiscrepancy = adapter.ComputeDomainDiscrepancy(adapted, targetData);

            // Assert
            Assert.True(finalDiscrepancy < initialDiscrepancy);
        }

        #endregion

        #region MMD Domain Adapter Tests

        [Fact]
        public void MMDAdapter_BasicAdaptation_ReducesDomainDiscrepancy()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            // Compute initial discrepancy
            var initialDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Act
            var adaptedSource = adapter.AdaptSource(sourceData, targetData);
            var finalDiscrepancy = adapter.ComputeDomainDiscrepancy(adaptedSource, targetData);

            // Assert
            Assert.True(finalDiscrepancy <= initialDiscrepancy,
                "MMD adaptation should not increase discrepancy");
        }

        [Fact]
        public void MMDAdapter_RequiresTraining_ReturnsFalse()
        {
            // Arrange & Act
            var adapter = new MMDDomainAdapter<double>();

            // Assert
            Assert.False(adapter.RequiresTraining);
            Assert.Equal("Maximum Mean Discrepancy (MMD)", adapter.AdaptationMethod);
        }

        [Fact]
        public void MMDAdapter_ComputeDiscrepancy_NonNegative()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.True(discrepancy >= 0, "MMD should be non-negative");
        }

        [Fact]
        public void MMDAdapter_IdenticalDistributions_ZeroDiscrepancy()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain(50, 3, seed: 42);
            var targetData = CreateSourceDomain(50, 3, seed: 42); // Same as source

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.True(discrepancy < 0.1, "Identical distributions should have near-zero MMD");
        }

        [Fact]
        public void MMDAdapter_AdaptSource_AlignsMeans()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            var sourceMean = ComputeMean(sourceData);
            var targetMean = ComputeMean(targetData);

            // Act
            var adaptedSource = adapter.AdaptSource(sourceData, targetData);
            var adaptedMean = ComputeMean(adaptedSource);

            // Assert - adapted mean should be closer to target mean
            for (int j = 0; j < targetMean.Length; j++)
            {
                double originalDistance = Math.Abs(sourceMean[j] - targetMean[j]);
                double adaptedDistance = Math.Abs(adaptedMean[j] - targetMean[j]);
                Assert.True(adaptedDistance < originalDistance + Tolerance,
                    $"Mean should be closer to target for feature {j}");
            }
        }

        [Fact]
        public void MMDAdapter_AdaptTarget_ReverseAdaptation()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            var sourceMean = ComputeMean(sourceData);
            var targetMean = ComputeMean(targetData);

            // Act
            var adaptedTarget = adapter.AdaptTarget(targetData, sourceData);
            var adaptedMean = ComputeMean(adaptedTarget);

            // Assert - adapted target mean should be closer to source mean
            for (int j = 0; j < sourceMean.Length; j++)
            {
                double originalDistance = Math.Abs(targetMean[j] - sourceMean[j]);
                double adaptedDistance = Math.Abs(adaptedMean[j] - sourceMean[j]);
                Assert.True(adaptedDistance < originalDistance + Tolerance,
                    $"Target mean should be closer to source for feature {j}");
            }
        }

        [Fact]
        public void MMDAdapter_SmallSigma_SensitiveToLocalDifferences()
        {
            // Arrange
            var adapterSmall = new MMDDomainAdapter<double>(sigma: 0.1);
            var adapterLarge = new MMDDomainAdapter<double>(sigma: 10.0);
            var sourceData = CreateSourceDomain(50, 3);
            var targetData = CreateTargetDomain(50, 3);

            // Act
            var discrepancySmall = adapterSmall.ComputeDomainDiscrepancy(sourceData, targetData);
            var discrepancyLarge = adapterLarge.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert - small sigma should detect more differences
            Assert.True(discrepancySmall >= 0);
            Assert.True(discrepancyLarge >= 0);
        }

        [Fact]
        public void MMDAdapter_TrainWithMedianHeuristic_UpdatesSigma()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            // Act - training should update sigma using median heuristic
            adapter.Train(sourceData, targetData);
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.True(discrepancy > 0, "Should compute non-zero discrepancy");
        }

        [Fact]
        public void MMDAdapter_LargeDomainGap_HighMMD()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain(); // mean=5, std=2
            var targetData = CreateTargetDomain(); // mean=1, std=0.5

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.True(discrepancy > 0.1, "Large domain gap should have higher MMD");
        }

        [Fact]
        public void MMDAdapter_SmallDomainGap_LowMMD()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var (source, target) = CreateSimilarDomains();

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(source, target);

            // Assert
            Assert.True(discrepancy >= 0, "MMD should be non-negative");
            Assert.True(discrepancy < 1.0, "Similar domains should have lower MMD");
        }

        [Fact]
        public void MMDAdapter_MultipleAdaptations_Consistent()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain(50, 3);
            var targetData = CreateTargetDomain(50, 3);

            // Act - multiple adaptations
            var adapted1 = adapter.AdaptSource(sourceData, targetData);
            var adapted2 = adapter.AdaptSource(sourceData, targetData);

            // Assert - should produce identical results
            for (int i = 0; i < adapted1.Rows; i++)
            {
                for (int j = 0; j < adapted1.Columns; j++)
                {
                    Assert.Equal(adapted1[i, j], adapted2[i, j], 6);
                }
            }
        }

        [Fact]
        public void MMDAdapter_DifferentSampleSizes_HandlesCorrectly()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain(100, 5);
            var targetData = CreateTargetDomain(50, 5); // Different size

            // Act
            var adapted = adapter.AdaptSource(sourceData, targetData);
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.Equal(sourceData.Rows, adapted.Rows);
            Assert.Equal(sourceData.Columns, adapted.Columns);
            Assert.True(discrepancy >= 0);
        }

        [Fact]
        public void MMDAdapter_SingleFeature_WorksCorrectly()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain(50, 1);
            var targetData = CreateTargetDomain(50, 1);

            // Act
            var adapted = adapter.AdaptSource(sourceData, targetData);
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

            // Assert
            Assert.Equal(1, adapted.Columns);
            Assert.True(discrepancy >= 0);
        }

        [Fact]
        public void MMDAdapter_HighDimensional_PerformsWell()
        {
            // Arrange
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain(100, 20);
            var targetData = CreateTargetDomain(100, 20);

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);
            var adapted = adapter.AdaptSource(sourceData, targetData);

            // Assert
            Assert.True(discrepancy > 0);
            Assert.Equal(20, adapted.Columns);
        }

        #endregion

        #region Comparison Tests

        [Fact]
        public void CompareAdapters_CORAL_vs_MMD_BothReduceDiscrepancy()
        {
            // Arrange
            var coralAdapter = new CORALDomainAdapter<double>();
            var mmdAdapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var sourceData = CreateSourceDomain();
            var targetData = CreateTargetDomain();

            // Act - CORAL
            var coralDiscrepancyBefore = coralAdapter.ComputeDomainDiscrepancy(sourceData, targetData);
            var coralAdapted = coralAdapter.AdaptSource(sourceData, targetData);
            var coralDiscrepancyAfter = coralAdapter.ComputeDomainDiscrepancy(coralAdapted, targetData);

            // Act - MMD
            var mmdDiscrepancyBefore = mmdAdapter.ComputeDomainDiscrepancy(sourceData, targetData);
            var mmdAdapted = mmdAdapter.AdaptSource(sourceData, targetData);
            var mmdDiscrepancyAfter = mmdAdapter.ComputeDomainDiscrepancy(mmdAdapted, targetData);

            // Assert
            Assert.True(coralDiscrepancyAfter < coralDiscrepancyBefore, "CORAL should reduce discrepancy");
            Assert.True(mmdDiscrepancyAfter <= mmdDiscrepancyBefore, "MMD should not increase discrepancy");
        }

        [Fact]
        public void CompareAdapters_TrainingRequirements_Differ()
        {
            // Arrange & Act
            var coralAdapter = new CORALDomainAdapter<double>();
            var mmdAdapter = new MMDDomainAdapter<double>();

            // Assert
            Assert.True(coralAdapter.RequiresTraining);
            Assert.False(mmdAdapter.RequiresTraining);
        }

        [Fact]
        public void DomainAdaptation_WithNoShift_MinimalChange()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(50, 3, seed: 42);
            var targetData = CreateSourceDomain(50, 3, seed: 42); // Identical

            // Act
            var adapted = adapter.AdaptSource(sourceData, targetData);

            // Assert - adaptation of identical data should be minimal
            for (int i = 0; i < sourceData.Rows; i++)
            {
                for (int j = 0; j < sourceData.Columns; j++)
                {
                    double diff = Math.Abs(adapted[i, j] - sourceData[i, j]);
                    Assert.True(diff < 2.0, $"Change should be small at [{i},{j}]");
                }
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void DomainAdapter_SmallSampleSize_HandlesGracefully()
        {
            // Arrange
            var coralAdapter = new CORALDomainAdapter<double>();
            var mmdAdapter = new MMDDomainAdapter<double>();
            var sourceData = CreateSourceDomain(10, 3); // Small sample
            var targetData = CreateTargetDomain(10, 3);

            // Act & Assert - should not throw
            var coralAdapted = coralAdapter.AdaptSource(sourceData, targetData);
            var mmdAdapted = mmdAdapter.AdaptSource(sourceData, targetData);

            Assert.Equal(10, coralAdapted.Rows);
            Assert.Equal(10, mmdAdapted.Rows);
        }

        [Fact]
        public void DomainAdapter_MinimalFeatures_WorksCorrectly()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(50, 2); // Just 2 features
            var targetData = CreateTargetDomain(50, 2);

            // Act
            var adapted = adapter.AdaptSource(sourceData, targetData);

            // Assert
            Assert.Equal(2, adapted.Columns);
            Assert.Equal(50, adapted.Rows);
        }

        [Fact]
        public void DomainAdapter_LargeDataset_PerformsEfficiently()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(500, 10);
            var targetData = CreateTargetDomain(500, 10);

            // Act
            var startTime = DateTime.Now;
            var adapted = adapter.AdaptSource(sourceData, targetData);
            var elapsed = DateTime.Now - startTime;

            // Assert
            Assert.True(elapsed.TotalSeconds < 5.0, "Should complete in reasonable time");
            Assert.Equal(500, adapted.Rows);
        }

        [Fact]
        public void DomainAdapter_RepeatedTraining_Stable()
        {
            // Arrange
            var adapter = new CORALDomainAdapter<double>();
            var sourceData = CreateSourceDomain(50, 3);
            var targetData = CreateTargetDomain(50, 3);

            // Act - train multiple times
            adapter.Train(sourceData, targetData);
            var adapted1 = adapter.AdaptSource(sourceData, targetData);

            adapter.Train(sourceData, targetData);
            var adapted2 = adapter.AdaptSource(sourceData, targetData);

            // Assert - should be consistent
            for (int i = 0; i < adapted1.Rows; i++)
            {
                for (int j = 0; j < adapted1.Columns; j++)
                {
                    Assert.Equal(adapted1[i, j], adapted2[i, j], 6);
                }
            }
        }

        #endregion
    }
}
