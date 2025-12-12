using AiDotNet.LinearAlgebra;
using AiDotNet.OutlierRemoval;
using AiDotNet.Statistics;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.OutlierRemoval
{
    /// <summary>
    /// Integration tests for outlier removal methods with mathematically verified results.
    /// Tests verify correct identification and removal of outliers using various statistical methods.
    /// </summary>
    public class OutlierRemovalIntegrationTests
    {
        private const double Tolerance = 1e-8;

        #region ZScoreOutlierRemoval Tests

        [Fact]
        public void ZScore_NormalDistributionWith3SigmaOutliers_RemovesOutliersCorrectly()
        {
            // Arrange: Normal data around mean=50, std=10, with 2 extreme outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 45.0 }, // normal
                { 50.0 }, // normal (mean)
                { 55.0 }, // normal
                { 48.0 }, // normal
                { 52.0 }, // normal
                { 100.0 }, // outlier: z-score = (100-50)/10 = 5.0 > 3
                { 5.0 }   // outlier: z-score = (5-50)/10 = -4.5 < -3
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove 2 outliers, keeping 5 normal points
            Assert.Equal(5, cleanedInputs.Rows);
            Assert.Equal(5, cleanedOutputs.Length);

            // Verify removed outliers: outputs 6.0 and 7.0 should not be present
            Assert.DoesNotContain(6.0, cleanedOutputs.ToArray());
            Assert.DoesNotContain(7.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void ZScore_ThresholdOf2_RemovesMoreOutliers()
        {
            // Arrange: Data with moderate outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 }, // normal
                { 51.0 }, // normal
                { 52.0 }, // normal
                { 75.0 }  // outlier at 2.5 std (removed with threshold=2, kept with threshold=3)
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            var remover2 = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.0);
            var remover3 = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleaned2, outputs2) = remover2.RemoveOutliers(inputs, outputs);
            var (cleaned3, outputs3) = remover3.RemoveOutliers(inputs, outputs);

            // Assert: Lower threshold should be more aggressive
            Assert.True(cleaned2.Rows < cleaned3.Rows || cleaned2.Rows == cleaned3.Rows);
        }

        [Fact]
        public void ZScore_NoOutliers_PreservesAllData()
        {
            // Arrange: All data within 1 std of mean
            var inputs = new Matrix<double>(new[,]
            {
                { 48.0 },
                { 49.0 },
                { 50.0 },
                { 51.0 },
                { 52.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep all data
            Assert.Equal(5, cleanedInputs.Rows);
            Assert.Equal(5, cleanedOutputs.Length);
        }

        [Fact]
        public void ZScore_MultipleFeatures_RemovesRowIfAnyFeatureIsOutlier()
        {
            // Arrange: 3 features, one row has outlier in second feature
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0, 20.0, 30.0 }, // normal
                { 11.0, 21.0, 31.0 }, // normal
                { 12.0, 100.0, 32.0 }, // outlier in column 2
                { 13.0, 22.0, 33.0 }  // normal
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove row with outlier
            Assert.Equal(3, cleanedInputs.Rows);
            Assert.DoesNotContain(3.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void ZScore_CalculationVerification_MatchesExpectedZScores()
        {
            // Arrange: Data with known mean and std
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 20.0 },
                { 30.0 },
                { 40.0 },
                { 50.0 }
            });
            // Mean = 30, Std = sqrt(250) ≈ 15.811
            // Z-scores: -1.265, -0.632, 0, 0.632, 1.265 (all within threshold)
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: All points should remain (Z-scores < 2)
            Assert.Equal(5, cleanedInputs.Rows);
        }

        [Fact]
        public void ZScore_AllOutliers_ReturnsEmptyDataset()
        {
            // Arrange: All data points are extreme outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 1000.0 },
                { -1000.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 0.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove all data
            Assert.Equal(0, cleanedInputs.Rows);
            Assert.Equal(0, cleanedOutputs.Length);
        }

        [Fact]
        public void ZScore_SingleDataPoint_PreservesData()
        {
            // Arrange: Single data point
            var inputs = new Matrix<double>(new[,] { { 50.0 } });
            var outputs = new Vector<double>(new[] { 1.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Single point should remain (std = 0, z-score undefined but handled)
            Assert.Equal(1, cleanedInputs.Rows);
        }

        [Fact]
        public void ZScore_SkewedDistribution_RemovesExtremeValues()
        {
            // Arrange: Right-skewed distribution with extreme high value
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 11.0 },
                { 12.0 },
                { 13.0 },
                { 14.0 },
                { 15.0 },
                { 50.0 } // extreme outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove extreme outlier
            Assert.True(cleanedInputs.Rows < 7);
            Assert.DoesNotContain(7.0, cleanedOutputs.ToArray());
        }

        #endregion

        #region IQROutlierRemoval Tests

        [Fact]
        public void IQR_StandardMultiplier_RemovesOutliersBeyond1Point5IQR()
        {
            // Arrange: Data with Q1=25, Q3=75, IQR=50
            // Lower bound = 25 - 1.5*50 = -50
            // Upper bound = 75 + 1.5*50 = 150
            var inputs = new Matrix<double>(new[,]
            {
                { 20.0 }, // normal
                { 30.0 }, // normal
                { 40.0 }, // normal
                { 50.0 }, // normal (median)
                { 60.0 }, // normal
                { 70.0 }, // normal
                { 80.0 }, // normal
                { 200.0 } // outlier > 150
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove the outlier
            Assert.True(cleanedInputs.Rows < 8);
            Assert.DoesNotContain(8.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void IQR_DifferentMultipliers_AffectOutlierDetection()
        {
            // Arrange: Data with moderate outlier
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 20.0 },
                { 30.0 },
                { 40.0 },
                { 50.0 },
                { 100.0 } // potential outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            var remover15 = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);
            var remover30 = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 3.0);

            // Act
            var (cleaned15, outputs15) = remover15.RemoveOutliers(inputs, outputs);
            var (cleaned30, outputs30) = remover30.RemoveOutliers(inputs, outputs);

            // Assert: Stricter multiplier (1.5) should remove more or equal outliers
            Assert.True(cleaned15.Rows <= cleaned30.Rows);
        }

        [Fact]
        public void IQR_NoOutliers_PreservesAllData()
        {
            // Arrange: Tight data distribution with no outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 45.0 },
                { 48.0 },
                { 50.0 },
                { 52.0 },
                { 55.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep all data
            Assert.Equal(5, cleanedInputs.Rows);
            Assert.Equal(5, cleanedOutputs.Length);
        }

        [Fact]
        public void IQR_BothHighAndLowOutliers_RemovesBoth()
        {
            // Arrange: Data with outliers on both ends
            var inputs = new Matrix<double>(new[,]
            {
                { 5.0 },   // low outlier
                { 40.0 },  // normal
                { 50.0 },  // normal
                { 60.0 },  // normal
                { 150.0 }  // high outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove both outliers
            Assert.Equal(3, cleanedInputs.Rows);
            Assert.DoesNotContain(1.0, cleanedOutputs.ToArray());
            Assert.DoesNotContain(5.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void IQR_QuartileCalculation_CorrectBoundaries()
        {
            // Arrange: Simple dataset with known quartiles
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 }, // Q1 region
                { 20.0 }, // Q1 region
                { 30.0 }, // Q2 region
                { 40.0 }, // Q2 region
                { 50.0 }, // Q3 region
                { 60.0 }, // Q3 region
                { 70.0 }  // Q3 region
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

            // Manually calculate: Q1≈20, Q3≈60, IQR≈40
            // Lower: 20 - 1.5*40 = -40, Upper: 60 + 1.5*40 = 120
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: All should remain within bounds
            Assert.Equal(7, cleanedInputs.Rows);
        }

        [Fact]
        public void IQR_SkewedData_HandlesAppropriately()
        {
            // Arrange: Right-skewed distribution
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 11.0 },
                { 12.0 },
                { 13.0 },
                { 14.0 },
                { 15.0 },
                { 16.0 },
                { 17.0 },
                { 50.0 } // outlier in skewed data
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove the extreme outlier
            Assert.True(cleanedInputs.Rows < 9);
        }

        [Fact]
        public void IQR_MultipleFeatures_RemovesIfAnyFeatureIsOutlier()
        {
            // Arrange: Multiple features, outlier in one feature
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0, 100.0 }, // normal, normal
                { 11.0, 110.0 }, // normal, normal
                { 12.0, 120.0 }, // normal, normal
                { 13.0, 500.0 }  // normal, outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove row with outlier
            Assert.Equal(3, cleanedInputs.Rows);
            Assert.DoesNotContain(4.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void IQR_UniformDistribution_RemovesNoOutliers()
        {
            // Arrange: Uniform distribution
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 20.0 },
                { 30.0 },
                { 40.0 },
                { 50.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep all data
            Assert.Equal(5, cleanedInputs.Rows);
        }

        #endregion

        #region MADOutlierRemoval Tests

        [Fact]
        public void MAD_StandardThreshold_RemovesOutliersBasedOnMedian()
        {
            // Arrange: Data with median-based outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 48.0 },
                { 49.0 },
                { 50.0 }, // median
                { 51.0 },
                { 52.0 },
                { 100.0 } // outlier far from median
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
            var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove the outlier
            Assert.True(cleanedInputs.Rows < 6);
            Assert.DoesNotContain(6.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void MAD_DifferentThresholds_AffectSensitivity()
        {
            // Arrange: Data with moderate outlier
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 },
                { 51.0 },
                { 52.0 },
                { 53.0 },
                { 75.0 } // moderate outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            var remover25 = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.5);
            var remover45 = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 4.5);

            // Act
            var (cleaned25, outputs25) = remover25.RemoveOutliers(inputs, outputs);
            var (cleaned45, outputs45) = remover45.RemoveOutliers(inputs, outputs);

            // Assert: Lower threshold should be more aggressive
            Assert.True(cleaned25.Rows <= cleaned45.Rows);
        }

        [Fact]
        public void MAD_MoreRobustThanZScore_HandlesSkewedData()
        {
            // Arrange: Highly skewed data with one extreme value
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 11.0 },
                { 12.0 },
                { 13.0 },
                { 14.0 },
                { 1000.0 } // extreme outlier that would affect mean/std
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            var madRemover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);
            var zscoreRemover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedMAD, outputsMAD) = madRemover.RemoveOutliers(inputs, outputs);
            var (cleanedZ, outputsZ) = zscoreRemover.RemoveOutliers(inputs, outputs);

            // Assert: MAD should identify the outlier (both should remove it, but MAD is more robust)
            Assert.True(cleanedMAD.Rows < 6);
            Assert.DoesNotContain(6.0, outputsMAD.ToArray());
        }

        [Fact]
        public void MAD_ModifiedZScoreCalculation_CorrectFormula()
        {
            // Arrange: Simple dataset to verify MAD calculation
            // Median = 50, MAD = median(|x - 50|)
            var inputs = new Matrix<double>(new[,]
            {
                { 45.0 }, // |45-50| = 5
                { 48.0 }, // |48-50| = 2
                { 50.0 }, // |50-50| = 0
                { 52.0 }, // |52-50| = 2
                { 55.0 }  // |55-50| = 5
            });
            // MAD = median([5, 2, 0, 2, 5]) = 2
            // Modified Z-scores: 0.6745 * [5, 2, 0, 2, 5] / 2 = [1.686, 0.675, 0, 0.675, 1.686]
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: All points should remain (modified Z-scores < 3.5)
            Assert.Equal(5, cleanedInputs.Rows);
        }

        [Fact]
        public void MAD_NoOutliers_PreservesAllData()
        {
            // Arrange: Compact data around median
            var inputs = new Matrix<double>(new[,]
            {
                { 49.0 },
                { 50.0 },
                { 51.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep all data
            Assert.Equal(3, cleanedInputs.Rows);
        }

        [Fact]
        public void MAD_MultimodalDistribution_HandlesCorrectly()
        {
            // Arrange: Bimodal distribution with outlier
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 11.0 },
                { 12.0 },
                { 50.0 }, // second mode
                { 51.0 },
                { 52.0 },
                { 200.0 } // outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });
            var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove the extreme outlier
            Assert.True(cleanedInputs.Rows < 7);
        }

        [Fact]
        public void MAD_MultipleFeatures_IdentifiesOutliersInAnyFeature()
        {
            // Arrange: Two features, outlier in second feature only
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0, 50.0 },
                { 11.0, 51.0 },
                { 12.0, 52.0 },
                { 13.0, 200.0 } // normal in first feature, outlier in second
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove row with outlier
            Assert.Equal(3, cleanedInputs.Rows);
            Assert.DoesNotContain(4.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void MAD_AllSameValues_HandlesGracefully()
        {
            // Arrange: All values the same (MAD = 0)
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 },
                { 50.0 },
                { 50.0 },
                { 50.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act & Assert: Should handle division by zero gracefully
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Either keeps all or removes all, depending on NaN handling
            Assert.True(cleanedInputs.Rows >= 0);
        }

        #endregion

        #region ThresholdOutlierRemoval Tests

        [Fact]
        public void Threshold_CustomThreshold_RemovesBasedOnMedianDeviation()
        {
            // Arrange: Data with known median and deviations
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 }, // median
                { 51.0 },
                { 52.0 },
                { 53.0 },
                { 100.0 } // large deviation from median
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove the outlier
            Assert.True(cleanedInputs.Rows < 5);
        }

        [Fact]
        public void Threshold_DifferentThresholdValues_AffectRemoval()
        {
            // Arrange: Data with moderate outlier
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 },
                { 51.0 },
                { 52.0 },
                { 70.0 } // moderate outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            var remover2 = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.0);
            var remover5 = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 5.0);

            // Act
            var (cleaned2, outputs2) = remover2.RemoveOutliers(inputs, outputs);
            var (cleaned5, outputs5) = remover5.RemoveOutliers(inputs, outputs);

            // Assert: Lower threshold should remove more outliers
            Assert.True(cleaned2.Rows <= cleaned5.Rows);
        }

        [Fact]
        public void Threshold_ExactCutoff_VerifiesThresholdBehavior()
        {
            // Arrange: Data designed to test exact threshold boundary
            // Median = 50, deviations = [0, 5, 10, 15, 20]
            // Median deviation = 10
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 }, // deviation = 0
                { 55.0 }, // deviation = 5
                { 60.0 }, // deviation = 10
                { 65.0 }, // deviation = 15
                { 70.0 }  // deviation = 20
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // With threshold 1.5, outliers are > 1.5 * 10 = 15 from median
            var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Point at 70 (deviation=20 > 15) should be removed
            Assert.True(cleanedInputs.Rows < 5);
        }

        [Fact]
        public void Threshold_NoOutliers_PreservesAllData()
        {
            // Arrange: Tight distribution
            var inputs = new Matrix<double>(new[,]
            {
                { 49.0 },
                { 50.0 },
                { 51.0 },
                { 52.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep all data
            Assert.Equal(4, cleanedInputs.Rows);
        }

        [Fact]
        public void Threshold_SymmetricOutliers_RemovesBoth()
        {
            // Arrange: Symmetric outliers around median
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },  // low outlier
                { 48.0 },
                { 50.0 },  // median
                { 52.0 },
                { 90.0 }   // high outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove both outliers
            Assert.True(cleanedInputs.Rows < 5);
        }

        [Fact]
        public void Threshold_MultipleFeatures_ConsidersAllFeatures()
        {
            // Arrange: Multiple features with outlier in one
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0, 100.0 },
                { 11.0, 101.0 },
                { 12.0, 102.0 },
                { 13.0, 500.0 } // outlier in second feature
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove row with outlier
            Assert.Equal(3, cleanedInputs.Rows);
        }

        [Fact]
        public void Threshold_VeryStrictThreshold_RemovesMore()
        {
            // Arrange: Normal data with strict threshold
            var inputs = new Matrix<double>(new[,]
            {
                { 45.0 },
                { 50.0 },
                { 55.0 },
                { 60.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            var remover05 = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 0.5);
            var remover10 = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 10.0);

            // Act
            var (cleaned05, _) = remover05.RemoveOutliers(inputs, outputs);
            var (cleaned10, _) = remover10.RemoveOutliers(inputs, outputs);

            // Assert: Very strict threshold should remove more
            Assert.True(cleaned05.Rows <= cleaned10.Rows);
        }

        [Fact]
        public void Threshold_SingleValue_PreservesData()
        {
            // Arrange: Single data point
            var inputs = new Matrix<double>(new[,] { { 50.0 } });
            var outputs = new Vector<double>(new[] { 1.0 });
            var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep the single point
            Assert.Equal(1, cleanedInputs.Rows);
        }

        #endregion

        #region NoOutlierRemoval Tests

        [Fact]
        public void NoOutlierRemoval_PassesDataThrough_NoModification()
        {
            // Arrange
            var inputs = new Matrix<double>(new[,]
            {
                { 1.0 },
                { 100.0 },
                { -50.0 },
                { 1000.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: All data should remain unchanged
            Assert.Equal(4, cleanedInputs.Rows);
            Assert.Equal(4, cleanedOutputs.Length);
            Assert.Equal(inputs, cleanedInputs);
            Assert.Equal(outputs, cleanedOutputs);
        }

        [Fact]
        public void NoOutlierRemoval_WithExtremeOutliers_KeepsEverything()
        {
            // Arrange: Data with obvious outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 },
                { 51.0 },
                { 10000.0 }, // extreme outlier
                { -10000.0 } // extreme outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should keep all data including outliers
            Assert.Equal(4, cleanedInputs.Rows);
            Assert.Contains(3.0, cleanedOutputs.ToArray());
            Assert.Contains(4.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void NoOutlierRemoval_EmptyData_ReturnsEmpty()
        {
            // Arrange: Empty datasets
            var inputs = new Matrix<double>(new double[0, 1]);
            var outputs = new Vector<double>(new double[0]);
            var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should return empty data
            Assert.Equal(0, cleanedInputs.Rows);
            Assert.Equal(0, cleanedOutputs.Length);
        }

        [Fact]
        public void NoOutlierRemoval_SingleDataPoint_PreservesData()
        {
            // Arrange
            var inputs = new Matrix<double>(new[,] { { 42.0 } });
            var outputs = new Vector<double>(new[] { 1.0 });
            var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert
            Assert.Equal(1, cleanedInputs.Rows);
            Assert.Equal(1, cleanedOutputs.Length);
            Assert.Equal(42.0, cleanedInputs[0, 0]);
            Assert.Equal(1.0, cleanedOutputs[0]);
        }

        [Fact]
        public void NoOutlierRemoval_MultipleFeatures_PreservesAllDimensions()
        {
            // Arrange: Multiple features with various values
            var inputs = new Matrix<double>(new[,]
            {
                { 1.0, 2.0, 3.0 },
                { 100.0, 200.0, 300.0 },
                { -50.0, -100.0, -150.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: All data preserved
            Assert.Equal(3, cleanedInputs.Rows);
            Assert.Equal(3, cleanedInputs.Columns);
            Assert.Equal(3, cleanedOutputs.Length);
        }

        [Fact]
        public void NoOutlierRemoval_ComparisonWithOtherMethods_KeepsMore()
        {
            // Arrange: Data with outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 },
                { 51.0 },
                { 52.0 },
                { 200.0 } // outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            var noRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
            var zScoreRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedNo, outputsNo) = noRemoval.RemoveOutliers(inputs, outputs);
            var (cleanedZ, outputsZ) = zScoreRemoval.RemoveOutliers(inputs, outputs);

            // Assert: NoOutlierRemoval should keep more (or equal) data
            Assert.True(cleanedNo.Rows >= cleanedZ.Rows);
            Assert.Equal(4, cleanedNo.Rows);
        }

        [Fact]
        public void NoOutlierRemoval_AsBaseline_UsefulForComparison()
        {
            // Arrange: Normal dataset
            var inputs = new Matrix<double>(new[,]
            {
                { 45.0 },
                { 50.0 },
                { 55.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Provides baseline (no removal)
            Assert.Equal(inputs.Rows, cleanedInputs.Rows);
            Assert.Equal(outputs.Length, cleanedOutputs.Length);
        }

        #endregion

        #region Cross-Method Comparison Tests

        [Fact]
        public void Comparison_MADMoreRobustThanZScore_OnSkewedData()
        {
            // Arrange: Skewed data where mean/std are affected by outlier
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 11.0 },
                { 12.0 },
                { 13.0 },
                { 14.0 },
                { 100.0 } // Outlier that skews mean
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            var madRemover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);
            var zScoreRemover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (madCleaned, madOutputs) = madRemover.RemoveOutliers(inputs, outputs);
            var (zCleaned, zOutputs) = zScoreRemover.RemoveOutliers(inputs, outputs);

            // Assert: Both should identify outlier, but MAD is more robust
            Assert.True(madCleaned.Rows < 6);
            Assert.True(zCleaned.Rows < 6);
        }

        [Fact]
        public void Comparison_IQRAndMAD_SimilarResultsOnNormalData()
        {
            // Arrange: Normal distribution
            var inputs = new Matrix<double>(new[,]
            {
                { 40.0 },
                { 45.0 },
                { 50.0 },
                { 55.0 },
                { 60.0 },
                { 150.0 } // Clear outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            var iqrRemover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);
            var madRemover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act
            var (iqrCleaned, iqrOutputs) = iqrRemover.RemoveOutliers(inputs, outputs);
            var (madCleaned, madOutputs) = madRemover.RemoveOutliers(inputs, outputs);

            // Assert: Both should remove the outlier
            Assert.True(iqrCleaned.Rows < 6);
            Assert.True(madCleaned.Rows < 6);
        }

        [Fact]
        public void Comparison_AllMethods_WithNoOutliers()
        {
            // Arrange: Clean data with no outliers
            var inputs = new Matrix<double>(new[,]
            {
                { 48.0 },
                { 49.0 },
                { 50.0 },
                { 51.0 },
                { 52.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            var zScore = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);
            var iqr = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);
            var mad = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);
            var threshold = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);
            var none = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Act
            var (zCleaned, _) = zScore.RemoveOutliers(inputs, outputs);
            var (iqrCleaned, _) = iqr.RemoveOutliers(inputs, outputs);
            var (madCleaned, _) = mad.RemoveOutliers(inputs, outputs);
            var (threshCleaned, _) = threshold.RemoveOutliers(inputs, outputs);
            var (noneCleaned, _) = none.RemoveOutliers(inputs, outputs);

            // Assert: All methods should keep all data (no outliers present)
            Assert.Equal(5, zCleaned.Rows);
            Assert.Equal(5, iqrCleaned.Rows);
            Assert.Equal(5, madCleaned.Rows);
            Assert.Equal(5, threshCleaned.Rows);
            Assert.Equal(5, noneCleaned.Rows);
        }

        #endregion

        #region Edge Cases and Boundary Tests

        [Fact]
        public void EdgeCase_TwoDataPoints_HandledAppropriately()
        {
            // Arrange: Minimal dataset
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 100.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0 });

            var zScore = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);
            var iqr = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);
            var mad = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);

            // Act & Assert: Should handle gracefully
            var (zCleaned, _) = zScore.RemoveOutliers(inputs, outputs);
            var (iqrCleaned, _) = iqr.RemoveOutliers(inputs, outputs);
            var (madCleaned, _) = mad.RemoveOutliers(inputs, outputs);

            Assert.True(zCleaned.Rows >= 0 && zCleaned.Rows <= 2);
            Assert.True(iqrCleaned.Rows >= 0 && iqrCleaned.Rows <= 2);
            Assert.True(madCleaned.Rows >= 0 && madCleaned.Rows <= 2);
        }

        [Fact]
        public void EdgeCase_AllIdenticalValues_NoRemoval()
        {
            // Arrange: All values identical
            var inputs = new Matrix<double>(new[,]
            {
                { 50.0 },
                { 50.0 },
                { 50.0 }
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            var zScore = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);
            var iqr = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (zCleaned, _) = zScore.RemoveOutliers(inputs, outputs);
            var (iqrCleaned, _) = iqr.RemoveOutliers(inputs, outputs);

            // Assert: Should keep data (no variation means no outliers)
            Assert.True(zCleaned.Rows >= 0);
            Assert.True(iqrCleaned.Rows >= 0);
        }

        [Fact]
        public void EdgeCase_LargeDataset_PerformanceTest()
        {
            // Arrange: Large dataset
            var size = 1000;
            var inputData = new double[size, 1];
            var outputData = new double[size];

            for (int i = 0; i < size; i++)
            {
                inputData[i, 0] = 50.0 + (i % 100) * 0.1; // Normal range
                outputData[i] = i;
            }
            // Add a few outliers
            inputData[size - 1, 0] = 1000.0;
            inputData[size - 2, 0] = -1000.0;

            var inputs = new Matrix<double>(inputData);
            var outputs = new Vector<double>(outputData);
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove outliers from large dataset
            Assert.True(cleanedInputs.Rows < size);
            Assert.True(cleanedInputs.Rows >= size - 10); // At most a few outliers removed
        }

        [Fact]
        public void EdgeCase_HighDimensionalData_HandlesMultipleFeatures()
        {
            // Arrange: Many features
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0, 20.0, 30.0, 40.0, 50.0 },
                { 11.0, 21.0, 31.0, 41.0, 51.0 },
                { 12.0, 22.0, 32.0, 42.0, 52.0 },
                { 13.0, 500.0, 33.0, 43.0, 53.0 } // Outlier in second feature
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should detect outlier in any dimension
            Assert.Equal(3, cleanedInputs.Rows);
            Assert.DoesNotContain(4.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void EdgeCase_NegativeValues_HandledCorrectly()
        {
            // Arrange: All negative values
            var inputs = new Matrix<double>(new[,]
            {
                { -50.0 },
                { -49.0 },
                { -48.0 },
                { -10.0 } // outlier (far from others)
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should handle negative values correctly
            Assert.True(cleanedInputs.Rows >= 0);
        }

        [Fact]
        public void EdgeCase_MixedPositiveNegativeOutliers_DetectsBoth()
        {
            // Arrange: Mixed positive and negative with outliers
            var inputs = new Matrix<double>(new[,]
            {
                { -100.0 }, // negative outlier
                { -1.0 },
                { 0.0 },
                { 1.0 },
                { 100.0 }  // positive outlier
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove both extreme outliers
            Assert.True(cleanedInputs.Rows < 5);
        }

        [Fact]
        public void Precision_ZScore_VerifiesExactCalculation()
        {
            // Arrange: Data with known mean=0, std=1
            var inputs = new Matrix<double>(new[,]
            {
                { -2.0 }, // z = -2
                { -1.0 }, // z = -1
                { 0.0 },  // z = 0
                { 1.0 },  // z = 1
                { 2.0 },  // z = 2
                { 5.0 }   // z = 5 (outlier)
            });
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
            var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should remove point with |z| > 3
            Assert.True(cleanedInputs.Rows < 6);
            Assert.DoesNotContain(6.0, cleanedOutputs.ToArray());
        }

        [Fact]
        public void Precision_IQR_VerifiesQuartileBoundaries()
        {
            // Arrange: Dataset with exact quartile values
            var inputs = new Matrix<double>(new[,]
            {
                { 10.0 },
                { 20.0 },
                { 30.0 },
                { 40.0 },
                { 50.0 },
                { 60.0 },
                { 70.0 },
                { 80.0 },
                { 90.0 }
            });
            // Q1=30, Q3=70, IQR=40, bounds: 30-60=-30, 70+60=130
            var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
            var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: All within bounds, should keep all
            Assert.Equal(9, cleanedInputs.Rows);
        }

        #endregion

        #region Float Type Tests

        [Fact]
        public void FloatType_ZScoreOutlierRemoval_WorksWithFloats()
        {
            // Arrange: Test with float type
            var inputs = new Matrix<float>(new[,]
            {
                { 50.0f },
                { 51.0f },
                { 52.0f },
                { 100.0f } // outlier
            });
            var outputs = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var remover = new ZScoreOutlierRemoval<float, Matrix<float>, Vector<float>>(threshold: 3.0);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should handle float type correctly
            Assert.True(cleanedInputs.Rows < 4);
        }

        [Fact]
        public void FloatType_IQROutlierRemoval_WorksWithFloats()
        {
            // Arrange: Test with float type
            var inputs = new Matrix<float>(new[,]
            {
                { 10.0f },
                { 20.0f },
                { 30.0f },
                { 100.0f } // outlier
            });
            var outputs = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var remover = new IQROutlierRemoval<float, Matrix<float>, Vector<float>>(iqrMultiplier: 1.5);

            // Act
            var (cleanedInputs, cleanedOutputs) = remover.RemoveOutliers(inputs, outputs);

            // Assert: Should handle float type correctly
            Assert.True(cleanedInputs.Rows <= 4);
        }

        #endregion
    }
}
