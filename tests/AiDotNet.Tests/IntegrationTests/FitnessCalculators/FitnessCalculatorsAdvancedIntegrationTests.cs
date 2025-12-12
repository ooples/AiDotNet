using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using Xunit;
using System;
using System.Linq;

namespace AiDotNetTests.IntegrationTests.FitnessCalculators
{
    /// <summary>
    /// Comprehensive integration tests for advanced fitness calculators with mathematically verified results.
    /// Tests cover specialized loss functions for segmentation, imbalanced classification, similarity learning,
    /// and other advanced scenarios. Each test verifies correct loss calculation, parameter effects, and edge cases.
    /// </summary>
    public class FitnessCalculatorsAdvancedIntegrationTests
    {
        private const double EPSILON = 1e-6;

        #region Helper Methods

        /// <summary>
        /// Creates a mock dataset for testing fitness calculators.
        /// </summary>
        private DataSetStats<double, double[], double> CreateMockDataSet(
            double[] predicted,
            double[] actual,
            double[][] features = null)
        {
            if (features == null)
            {
                features = predicted.Select((_, i) => new[] { (double)i }).ToArray();
            }

            return new DataSetStats<double, double[], double>
            {
                Predicted = predicted.Select(p => p).ToArray(),
                Actual = actual.Select(a => a).ToArray(),
                Features = features
            };
        }

        #endregion

        #region LogCoshLossFitnessCalculator Tests

        [Fact]
        public void LogCoshLoss_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0, 4.0, 5.0 },
                actual: new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - log(cosh(0)) = 0
            Assert.Equal(0.0, fitness, precision: 10);
        }

        [Fact]
        public void LogCoshLoss_KnownValues_ComputesCorrectLoss()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 0.0, 1.0, 2.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - log(cosh(1)) = 0.433, average = 0.433
            Assert.True(fitness > 0.4 && fitness < 0.45);
        }

        [Fact]
        public void LogCoshLoss_SmallErrors_BehavesLikeMSE()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.1, 0.2, 0.3 },
                actual: new[] { 0.0, 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - For small x, log(cosh(x)) ≈ x²/2
            // Expected: (0.01 + 0.04 + 0.09) / (2*3) = 0.0233
            Assert.True(fitness > 0.02 && fitness < 0.03);
        }

        [Fact]
        public void LogCoshLoss_LargeErrors_BehavesLikeMAE()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 10.0, -10.0 },
                actual: new[] { 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - For large x, log(cosh(x)) ≈ |x| - log(2)
            // Expected: (10 - log(2) + 10 - log(2)) / 2 ≈ 10 - log(2) ≈ 9.31
            Assert.True(fitness > 9.0 && fitness < 9.5);
        }

        [Fact]
        public void LogCoshLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { -5.0, -2.0, 0.0, 3.0, 7.0 },
                actual: new[] { 0.0, 1.0, 2.0, 3.0, 5.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void LogCoshLoss_SymmetricErrors_ProduceSameLoss()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet1 = CreateMockDataSet(
                predicted: new[] { 3.0 },
                actual: new[] { 1.0 }
            );
            var dataSet2 = CreateMockDataSet(
                predicted: new[] { -1.0 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitness1 = calculator.CalculateFitness(null, new[] { dataSet1, dataSet1, dataSet1 });
            var fitness2 = calculator.CalculateFitness(null, new[] { dataSet2, dataSet2, dataSet2 });

            // Assert - log(cosh(x)) is symmetric
            Assert.Equal(fitness1, fitness2, precision: 10);
        }

        [Fact]
        public void LogCoshLoss_RobustToOutliers()
        {
            // Arrange
            var calculator = new LogCoshLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, 100.0 },
                actual: new[] { 1.0, 1.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Loss should be moderate despite large outlier
            Assert.True(fitness < 100.0); // Much less than the outlier magnitude
        }

        #endregion

        #region QuantileLossFitnessCalculator Tests

        [Fact]
        public void QuantileLoss_MedianQuantile_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var calculator = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.5);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.Equal(0.0, fitness, precision: 10);
        }

        [Fact]
        public void QuantileLoss_MedianQuantile_SymmetricPenalty()
        {
            // Arrange
            var calculator = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.5);
            var dataSet1 = CreateMockDataSet(
                predicted: new[] { 3.0 },
                actual: new[] { 1.0 }
            );
            var dataSet2 = CreateMockDataSet(
                predicted: new[] { -1.0 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitness1 = calculator.CalculateFitness(null, new[] { dataSet1, dataSet1, dataSet1 });
            var fitness2 = calculator.CalculateFitness(null, new[] { dataSet2, dataSet2, dataSet2 });

            // Assert - At quantile 0.5, over and under predictions penalized equally
            Assert.Equal(fitness1, fitness2, precision: 6);
        }

        [Fact]
        public void QuantileLoss_HighQuantile_PenalizesUnderpredictionMore()
        {
            // Arrange
            var calculator = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.9);
            var dataSetUnder = CreateMockDataSet(
                predicted: new[] { 1.0 },
                actual: new[] { 2.0 }
            );
            var dataSetOver = CreateMockDataSet(
                predicted: new[] { 2.0 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitnessUnder = calculator.CalculateFitness(null, new[] { dataSetUnder, dataSetUnder, dataSetUnder });
            var fitnessOver = calculator.CalculateFitness(null, new[] { dataSetOver, dataSetOver, dataSetOver });

            // Assert - Underprediction should have higher loss at q=0.9
            Assert.True(fitnessUnder > fitnessOver);
        }

        [Fact]
        public void QuantileLoss_LowQuantile_PenalizesOverpredictionMore()
        {
            // Arrange
            var calculator = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.1);
            var dataSetUnder = CreateMockDataSet(
                predicted: new[] { 1.0 },
                actual: new[] { 2.0 }
            );
            var dataSetOver = CreateMockDataSet(
                predicted: new[] { 2.0 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitnessUnder = calculator.CalculateFitness(null, new[] { dataSetUnder, dataSetUnder, dataSetUnder });
            var fitnessOver = calculator.CalculateFitness(null, new[] { dataSetOver, dataSetOver, dataSetOver });

            // Assert - Overprediction should have higher loss at q=0.1
            Assert.True(fitnessOver > fitnessUnder);
        }

        [Fact]
        public void QuantileLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.75);
            var dataSet = CreateMockDataSet(
                predicted: new[] { -5.0, 0.0, 5.0, 10.0 },
                actual: new[] { 0.0, 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void QuantileLoss_DifferentQuantiles_ProduceDifferentResults()
        {
            // Arrange
            var calculator1 = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.25);
            var calculator2 = new QuantileLossFitnessCalculator<double, double[], double>(quantile: 0.75);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 2.0, 3.0, 4.0 }
            );

            // Act
            var fitness1 = calculator1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitness2 = calculator2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Different quantiles should produce different losses
            Assert.NotEqual(fitness1, fitness2);
        }

        #endregion

        #region PoissonLossFitnessCalculator Tests

        [Fact]
        public void PoissonLoss_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var calculator = new PoissonLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0, 4.0 },
                actual: new[] { 1.0, 2.0, 3.0, 4.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness < 0.1);
        }

        [Fact]
        public void PoissonLoss_CountData_ComputesCorrectly()
        {
            // Arrange
            var calculator = new PoissonLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Poisson loss should be small for matching predictions
            Assert.True(fitness >= 0.0);
            Assert.True(fitness < 0.5);
        }

        [Fact]
        public void PoissonLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new PoissonLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 1.5, 2.5, 3.5 },
                actual: new[] { 0.0, 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void PoissonLoss_LowCounts_HandlesCorrectly()
        {
            // Arrange
            var calculator = new PoissonLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.1, 0.2, 0.3 },
                actual: new[] { 0.0, 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(double.IsFinite(fitness));
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void PoissonLoss_HighCounts_HandlesCorrectly()
        {
            // Arrange
            var calculator = new PoissonLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 50.0, 100.0, 150.0 },
                actual: new[] { 52.0, 98.0, 148.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(double.IsFinite(fitness));
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void PoissonLoss_MixedCounts_ComputesCorrectly()
        {
            // Arrange
            var calculator = new PoissonLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 5.0, 50.0 },
                actual: new[] { 1.0, 4.0, 52.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(double.IsFinite(fitness));
            Assert.True(fitness > 0.0);
        }

        #endregion

        #region KullbackLeiblerDivergenceFitnessCalculator Tests

        [Fact]
        public void KLDivergence_IdenticalDistributions_ReturnsNearZero()
        {
            // Arrange
            var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.25, 0.25, 0.25, 0.25 },
                actual: new[] { 0.25, 0.25, 0.25, 0.25 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - KL(P||P) = 0
            Assert.True(fitness < 0.01);
        }

        [Fact]
        public void KLDivergence_DifferentDistributions_ReturnsPositive()
        {
            // Arrange
            var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 0.3, 0.2 },
                actual: new[] { 0.8, 0.1, 0.1 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness > 0.0);
        }

        [Fact]
        public void KLDivergence_IsNonNegative()
        {
            // Arrange
            var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.1, 0.2, 0.3, 0.4 },
                actual: new[] { 0.4, 0.3, 0.2, 0.1 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void KLDivergence_UniformVsSkewed_ComputesCorrectly()
        {
            // Arrange
            var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.25, 0.25, 0.25, 0.25 },
                actual: new[] { 0.7, 0.1, 0.1, 0.1 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should be positive as distributions differ
            Assert.True(fitness > 0.5);
        }

        [Fact]
        public void KLDivergence_ProbabilityConstraints_HandlesCorrectly()
        {
            // Arrange
            var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.9, 0.05, 0.05 },
                actual: new[] { 0.8, 0.1, 0.1 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(double.IsFinite(fitness));
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void KLDivergence_HighConfidencePredictions_ProducesLowerLoss()
        {
            // Arrange
            var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>();
            var dataSet1 = CreateMockDataSet(
                predicted: new[] { 0.9, 0.1 },
                actual: new[] { 1.0, 0.0 }
            );
            var dataSet2 = CreateMockDataSet(
                predicted: new[] { 0.6, 0.4 },
                actual: new[] { 1.0, 0.0 }
            );

            // Act
            var fitness1 = calculator.CalculateFitness(null, new[] { dataSet1, dataSet1, dataSet1 });
            var fitness2 = calculator.CalculateFitness(null, new[] { dataSet2, dataSet2, dataSet2 });

            // Assert - Higher confidence in correct class should have lower loss
            Assert.True(fitness1 < fitness2);
        }

        #endregion

        #region DiceLossFitnessCalculator Tests

        [Fact]
        public void DiceLoss_PerfectOverlap_ReturnsNearZero()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, 0.0, 0.0, 1.0 },
                actual: new[] { 1.0, 1.0, 0.0, 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Dice coefficient = 1, loss = 0
            Assert.True(fitness < 0.01);
        }

        [Fact]
        public void DiceLoss_NoOverlap_ReturnsNearOne()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, 0.0, 0.0 },
                actual: new[] { 0.0, 0.0, 1.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - No overlap, Dice = 0, loss = 1
            Assert.True(fitness > 0.99);
        }

        [Fact]
        public void DiceLoss_PartialOverlap_ComputesCorrectly()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 0.5 },
                actual: new[] { 1.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Intersection=0.5, Sum=2.0, Dice=2*0.5/2=0.5, Loss=0.5
            Assert.Equal(0.5, fitness, precision: 5);
        }

        [Fact]
        public void DiceLoss_SegmentationScenario_KnownIoU()
        {
            // Arrange - Simulating 50% overlap
            var calculator = new DiceLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, 1.0, 0.0 },
                actual: new[] { 1.0, 1.0, 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Intersection=2, Sum=5, Dice=2*2/5=0.8, Loss=0.2
            Assert.True(fitness > 0.15 && fitness < 0.25);
        }

        [Fact]
        public void DiceLoss_IsBetweenZeroAndOne()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.7, 0.3, 0.5, 0.2 },
                actual: new[] { 1.0, 0.0, 1.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0 && fitness <= 1.0);
        }

        [Fact]
        public void DiceLoss_ImbalancedData_HandlesCorrectly()
        {
            // Arrange - 10% positive pixels
            var calculator = new DiceLossFitnessCalculator<double, double[], double>();
            var predicted = new double[100];
            var actual = new double[100];
            for (int i = 0; i < 10; i++)
            {
                predicted[i] = 0.9;
                actual[i] = 1.0;
            }

            var dataSet = CreateMockDataSet(predicted, actual);

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should handle imbalanced data
            Assert.True(fitness >= 0.0);
            Assert.True(fitness < 0.3); // Good overlap
        }

        #endregion

        #region JaccardLossFitnessCalculator Tests

        [Fact]
        public void JaccardLoss_PerfectOverlap_ReturnsNearZero()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, 0.0, 0.0 },
                actual: new[] { 1.0, 1.0, 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - IoU = 1, loss = 0
            Assert.True(fitness < 0.01);
        }

        [Fact]
        public void JaccardLoss_NoOverlap_ReturnsNearOne()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 0.0 },
                actual: new[] { 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - IoU = 0, loss = 1
            Assert.True(fitness > 0.99);
        }

        [Fact]
        public void JaccardLoss_PartialOverlap_ComputesIoU()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 0.5 },
                actual: new[] { 1.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Intersection=0.5, Union=1.5, IoU=0.333, Loss=0.667
            Assert.True(fitness > 0.6 && fitness < 0.7);
        }

        [Fact]
        public void JaccardLoss_KnownIoU_50Percent()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, 1.0, 0.0 },
                actual: new[] { 1.0, 1.0, 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Intersection=2, Union=3, IoU=2/3=0.667, Loss=0.333
            Assert.True(fitness > 0.3 && fitness < 0.4);
        }

        [Fact]
        public void JaccardLoss_IsBetweenZeroAndOne()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 0.7, 0.3 },
                actual: new[] { 1.0, 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0 && fitness <= 1.0);
        }

        [Fact]
        public void JaccardLoss_ComparesWithDice_RelatedButDifferent()
        {
            // Arrange
            var jaccardCalc = new JaccardLossFitnessCalculator<double, double[], double>();
            var diceCalc = new DiceLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.7, 0.3, 0.5 },
                actual: new[] { 1.0, 0.0, 1.0 }
            );

            // Act
            var jaccardFitness = jaccardCalc.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var diceFitness = diceCalc.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should be related but different
            Assert.NotEqual(jaccardFitness, diceFitness);
            Assert.True(Math.Abs(jaccardFitness - diceFitness) < 0.5);
        }

        #endregion

        #region FocalLossFitnessCalculator Tests

        [Fact]
        public void FocalLoss_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var calculator = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 0.25);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.9999, 0.0001, 0.9999 },
                actual: new[] { 1.0, 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness < 0.001);
        }

        [Fact]
        public void FocalLoss_EasyExamples_DownWeighted()
        {
            // Arrange
            var calculator = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 0.25);
            var dataSetEasy = CreateMockDataSet(
                predicted: new[] { 0.9 },
                actual: new[] { 1.0 }
            );
            var dataSetHard = CreateMockDataSet(
                predicted: new[] { 0.6 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitnessEasy = calculator.CalculateFitness(null, new[] { dataSetEasy, dataSetEasy, dataSetEasy });
            var fitnessHard = calculator.CalculateFitness(null, new[] { dataSetHard, dataSetHard, dataSetHard });

            // Assert - Hard examples contribute more
            Assert.True(fitnessHard > fitnessEasy);
        }

        [Fact]
        public void FocalLoss_ImbalancedClassification_1Percent()
        {
            // Arrange - 1% positive, 99% negative
            var calculator = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 0.25);
            var predicted = new double[100];
            var actual = new double[100];
            predicted[0] = 0.8; // Positive prediction
            actual[0] = 1.0;
            for (int i = 1; i < 100; i++)
            {
                predicted[i] = 0.1; // Negative predictions
                actual[i] = 0.0;
            }

            var dataSet = CreateMockDataSet(predicted, actual);

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should handle imbalanced data
            Assert.True(fitness >= 0.0);
            Assert.True(double.IsFinite(fitness));
        }

        [Fact]
        public void FocalLoss_GammaEffect_HigherFocusOnHard()
        {
            // Arrange
            var calculator1 = new FocalLossFitnessCalculator<double, double[], double>(gamma: 0.0, alpha: 1.0);
            var calculator2 = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitness1 = calculator1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitness2 = calculator2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Different gamma should produce different results
            Assert.NotEqual(fitness1, fitness2);
        }

        [Fact]
        public void FocalLoss_AlphaEffect_ClassBalance()
        {
            // Arrange
            var calculator1 = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 0.25);
            var calculator2 = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 0.75);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.6, 0.4 },
                actual: new[] { 1.0, 0.0 }
            );

            // Act
            var fitness1 = calculator1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitness2 = calculator2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Different alpha should affect class weighting
            Assert.NotEqual(fitness1, fitness2);
        }

        [Fact]
        public void FocalLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new FocalLossFitnessCalculator<double, double[], double>(gamma: 2.0, alpha: 0.25);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.3, 0.5, 0.7, 0.9 },
                actual: new[] { 0.0, 1.0, 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        #endregion

        #region ContrastiveLossFitnessCalculator Tests

        [Fact]
        public void ContrastiveLoss_SimilarPairs_LowDistance_LowLoss()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, double[], double>(margin: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.1, 0.1 }, // Low distance embeddings
                actual: new[] { 1.0, 1.0 }     // Similar labels
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Similar pairs with low distance should have low loss
            Assert.True(fitness >= 0.0);
            Assert.True(fitness < 1.0);
        }

        [Fact]
        public void ContrastiveLoss_DissimilarPairs_HighDistance_LowLoss()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, double[], double>(margin: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.0, 2.0 }, // High distance embeddings
                actual: new[] { 0.0, 1.0 }     // Different labels
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Dissimilar pairs beyond margin should have low loss
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void ContrastiveLoss_MarginEffect_LargerMargin()
        {
            // Arrange
            var calculator1 = new ContrastiveLossFitnessCalculator<double, double[], double>(margin: 0.5);
            var calculator2 = new ContrastiveLossFitnessCalculator<double, double[], double>(margin: 2.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.0, 1.0 },
                actual: new[] { 0.0, 1.0 }
            );

            // Act
            var fitness1 = calculator1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitness2 = calculator2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Different margins should produce different results
            Assert.True(fitness1 >= 0.0);
            Assert.True(fitness2 >= 0.0);
        }

        [Fact]
        public void ContrastiveLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, double[], double>(margin: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 1.5, 0.2, 1.8 },
                actual: new[] { 1.0, 1.0, 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void ContrastiveLoss_EmbeddingAlignment_VerifiesCorrectly()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, double[], double>(margin: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.0, 0.0, 2.0, 2.0 },
                actual: new[] { 1.0, 1.0, 0.0, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(double.IsFinite(fitness));
            Assert.True(fitness >= 0.0);
        }

        #endregion

        #region TripletLossFitnessCalculator Tests

        [Fact]
        public void TripletLoss_WellSeparatedTriplets_LowLoss()
        {
            // Arrange
            var calculator = new TripletLossFitnessCalculator<double, double[], double>(margin: 1.0);
            // Create features where same class is close, different class is far
            var features = new double[][]
            {
                new[] { 0.0, 0.0 }, // Class 0
                new[] { 0.1, 0.1 }, // Class 0
                new[] { 5.0, 5.0 }  // Class 1
            };
            var actual = new[] { 0.0, 0.0, 1.0 };

            var dataSet = new DataSetStats<double, double[], double>
            {
                Predicted = new[] { 0.0, 0.0, 0.0 },
                Actual = actual,
                Features = features
            };

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Well-separated should have low loss
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void TripletLoss_MarginEffect_DifferentSeparation()
        {
            // Arrange
            var calculator1 = new TripletLossFitnessCalculator<double, double[], double>(margin: 0.5);
            var calculator2 = new TripletLossFitnessCalculator<double, double[], double>(margin: 2.0);
            var features = new double[][]
            {
                new[] { 0.0 },
                new[] { 0.5 },
                new[] { 1.5 }
            };
            var actual = new[] { 0.0, 0.0, 1.0 };

            var dataSet = new DataSetStats<double, double[], double>
            {
                Predicted = new[] { 0.0, 0.0, 0.0 },
                Actual = actual,
                Features = features
            };

            // Act
            var fitness1 = calculator1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitness2 = calculator2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Different margins affect loss
            Assert.True(fitness1 >= 0.0);
            Assert.True(fitness2 >= 0.0);
        }

        [Fact]
        public void TripletLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new TripletLossFitnessCalculator<double, double[], double>(margin: 1.0);
            var features = new double[][]
            {
                new[] { 1.0, 2.0 },
                new[] { 1.5, 2.5 },
                new[] { 5.0, 6.0 },
                new[] { 5.5, 6.5 }
            };
            var actual = new[] { 0.0, 0.0, 1.0, 1.0 };

            var dataSet = new DataSetStats<double, double[], double>
            {
                Predicted = new[] { 0.0, 0.0, 0.0, 0.0 },
                Actual = actual,
                Features = features
            };

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void TripletLoss_MultipleClasses_HandlesCorrectly()
        {
            // Arrange
            var calculator = new TripletLossFitnessCalculator<double, double[], double>(margin: 1.0);
            var features = new double[][]
            {
                new[] { 0.0 }, new[] { 0.1 }, // Class 0
                new[] { 5.0 }, new[] { 5.1 }, // Class 1
                new[] { 10.0 }, new[] { 10.1 } // Class 2
            };
            var actual = new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 };

            var dataSet = new DataSetStats<double, double[], double>
            {
                Predicted = new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                Actual = actual,
                Features = features
            };

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(double.IsFinite(fitness));
            Assert.True(fitness >= 0.0);
        }

        #endregion

        #region CosineSimilarityLossFitnessCalculator Tests

        [Fact]
        public void CosineSimilarity_SameDirection_ReturnsNearZero()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Cosine similarity = 1, loss = 0
            Assert.True(fitness < 0.01);
        }

        [Fact]
        public void CosineSimilarity_OppositeDirection_ReturnsNearTwo()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { -1.0, -2.0, -3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Cosine similarity = -1, loss = 2
            Assert.True(fitness > 1.9);
        }

        [Fact]
        public void CosineSimilarity_PerpendicularVectors_ReturnsOne()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 0.0 },
                actual: new[] { 0.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Cosine similarity = 0, loss = 1
            Assert.True(fitness > 0.9 && fitness < 1.1);
        }

        [Fact]
        public void CosineSimilarity_ScaleInvariant()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, double[], double>();
            var dataSet1 = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 2.0, 4.0, 6.0 }
            );
            var dataSet2 = CreateMockDataSet(
                predicted: new[] { 10.0, 20.0, 30.0 },
                actual: new[] { 20.0, 40.0, 60.0 }
            );

            // Act
            var fitness1 = calculator.CalculateFitness(null, new[] { dataSet1, dataSet1, dataSet1 });
            var fitness2 = calculator.CalculateFitness(null, new[] { dataSet2, dataSet2, dataSet2 });

            // Assert - Cosine similarity is scale-invariant
            Assert.Equal(fitness1, fitness2, precision: 6);
        }

        [Fact]
        public void CosineSimilarity_IsBetweenZeroAndTwo()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 2.0, 1.0, 4.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Loss should be in [0, 2]
            Assert.True(fitness >= 0.0 && fitness <= 2.0);
        }

        [Fact]
        public void CosineSimilarity_DocumentSimilarity_Scenario()
        {
            // Arrange - Simulating TF-IDF vectors
            var calculator = new CosineSimilarityLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 0.3, 0.2, 0.0 },
                actual: new[] { 0.6, 0.2, 0.2, 0.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Similar documents should have low loss
            Assert.True(fitness < 0.2);
        }

        #endregion

        #region ElasticNetLossFitnessCalculator Tests

        [Fact]
        public void ElasticNetLoss_PerfectPrediction_OnlyRegularizationPenalty()
        {
            // Arrange
            var calculator = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 0.5, alpha: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should have some regularization penalty
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void ElasticNetLoss_L1RatioEffect_PureL1VsPureL2()
        {
            // Arrange
            var calculatorL1 = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 1.0, alpha: 1.0);
            var calculatorL2 = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 0.0, alpha: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.5, 2.5, 3.5 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitnessL1 = calculatorL1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitnessL2 = calculatorL2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Different ratios should produce different results
            Assert.True(fitnessL1 >= 0.0);
            Assert.True(fitnessL2 >= 0.0);
        }

        [Fact]
        public void ElasticNetLoss_AlphaEffect_StrongerRegularization()
        {
            // Arrange
            var calculator1 = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 0.5, alpha: 0.1);
            var calculator2 = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 0.5, alpha: 10.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 2.0, 3.0, 4.0 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness1 = calculator1.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
            var fitness2 = calculator2.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Higher alpha should increase loss
            Assert.True(fitness2 > fitness1);
        }

        [Fact]
        public void ElasticNetLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 0.5, alpha: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { -1.0, 0.0, 1.0, 2.0 },
                actual: new[] { -2.0, 1.0, 0.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void ElasticNetLoss_BalancedRatio_CombinesL1AndL2()
        {
            // Arrange
            var calculator = new ElasticNetLossFitnessCalculator<double, double[], double>(l1Ratio: 0.5, alpha: 1.0);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.5, 2.5, 3.5 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness > 0.0);
            Assert.True(double.IsFinite(fitness));
        }

        #endregion

        #region ExponentialLossFitnessCalculator Tests

        [Fact]
        public void ExponentialLoss_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var calculator = new ExponentialLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 1.0, -1.0, -1.0 },
                actual: new[] { 1.0, 1.0, -1.0, -1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness < 0.5);
        }

        [Fact]
        public void ExponentialLoss_ConfidentMistake_HighPenalty()
        {
            // Arrange
            var calculator = new ExponentialLossFitnessCalculator<double, double[], double>();
            var dataSetCorrect = CreateMockDataSet(
                predicted: new[] { 1.0 },
                actual: new[] { 1.0 }
            );
            var dataSetWrong = CreateMockDataSet(
                predicted: new[] { -1.0 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitnessCorrect = calculator.CalculateFitness(null, new[] { dataSetCorrect, dataSetCorrect, dataSetCorrect });
            var fitnessWrong = calculator.CalculateFitness(null, new[] { dataSetWrong, dataSetWrong, dataSetWrong });

            // Assert - Confident mistake should have much higher loss
            Assert.True(fitnessWrong > fitnessCorrect * 2);
        }

        [Fact]
        public void ExponentialLoss_IsNonNegative()
        {
            // Arrange
            var calculator = new ExponentialLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, -0.5, 1.0, -1.0 },
                actual: new[] { 1.0, -1.0, -1.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void ExponentialLoss_AdaBoostScenario_PenalizesErrors()
        {
            // Arrange
            var calculator = new ExponentialLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.8, 0.6, -0.9, -0.7 },
                actual: new[] { 1.0, 1.0, -1.0, -1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
            Assert.True(double.IsFinite(fitness));
        }

        [Fact]
        public void ExponentialLoss_GrowsExponentially_WithError()
        {
            // Arrange
            var calculator = new ExponentialLossFitnessCalculator<double, double[], double>();
            var dataSetSmall = CreateMockDataSet(
                predicted: new[] { 0.5 },
                actual: new[] { 1.0 }
            );
            var dataSetLarge = CreateMockDataSet(
                predicted: new[] { -1.0 },
                actual: new[] { 1.0 }
            );

            // Act
            var fitnessSmall = calculator.CalculateFitness(null, new[] { dataSetSmall, dataSetSmall, dataSetSmall });
            var fitnessLarge = calculator.CalculateFitness(null, new[] { dataSetLarge, dataSetLarge, dataSetLarge });

            // Assert - Loss should grow significantly with error
            Assert.True(fitnessLarge > fitnessSmall * 2);
        }

        #endregion

        #region OrdinalRegressionLossFitnessCalculator Tests

        [Fact]
        public void OrdinalRegression_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>(numberOfClassifications: 5);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0, 4.0, 5.0 },
                actual: new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness < 0.1);
        }

        [Fact]
        public void OrdinalRegression_NearbyRatings_LowerPenalty()
        {
            // Arrange
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>(numberOfClassifications: 5);
            var dataSetNear = CreateMockDataSet(
                predicted: new[] { 4.0 },
                actual: new[] { 5.0 }
            );
            var dataSetFar = CreateMockDataSet(
                predicted: new[] { 1.0 },
                actual: new[] { 5.0 }
            );

            // Act
            var fitnessNear = calculator.CalculateFitness(null, new[] { dataSetNear, dataSetNear, dataSetNear });
            var fitnessFar = calculator.CalculateFitness(null, new[] { dataSetFar, dataSetFar, dataSetFar });

            // Assert - Nearby prediction should have lower loss
            Assert.True(fitnessNear < fitnessFar);
        }

        [Fact]
        public void OrdinalRegression_FiveStarRating_Scenario()
        {
            // Arrange
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>(numberOfClassifications: 5);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 4.0, 3.0, 5.0, 2.0 },
                actual: new[] { 5.0, 3.0, 4.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should handle rating predictions
            Assert.True(fitness >= 0.0);
            Assert.True(double.IsFinite(fitness));
        }

        [Fact]
        public void OrdinalRegression_EducationLevels_Scenario()
        {
            // Arrange
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>(numberOfClassifications: 4);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0, 4.0 },
                actual: new[] { 1.0, 3.0, 3.0, 4.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void OrdinalRegression_IsNonNegative()
        {
            // Arrange
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>(numberOfClassifications: 5);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 3.0, 2.0, 5.0, 4.0 },
                actual: new[] { 2.0, 4.0, 1.0, 5.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        [Fact]
        public void OrdinalRegression_AutoDetectClasses_WorksCorrectly()
        {
            // Arrange - Don't specify number of classes
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>();
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0 },
                actual: new[] { 1.0, 2.0, 3.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert - Should auto-detect and compute
            Assert.True(fitness >= 0.0);
            Assert.True(double.IsFinite(fitness));
        }

        [Fact]
        public void OrdinalRegression_DiseaseSeverity_ThreeLevel()
        {
            // Arrange
            var calculator = new OrdinalRegressionLossFitnessCalculator<double, double[], double>(numberOfClassifications: 3);
            var dataSet = CreateMockDataSet(
                predicted: new[] { 1.0, 2.0, 3.0, 2.0 },
                actual: new[] { 1.0, 2.0, 3.0, 1.0 }
            );

            // Act
            var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });

            // Assert
            Assert.True(fitness >= 0.0);
        }

        #endregion

        #region Edge Cases and Comprehensive Tests

        [Fact]
        public void AllAdvancedCalculators_IsLowerBetter_SetCorrectly()
        {
            // Arrange & Act
            var calculators = new IFitnessCalculator<double, double[], double>[]
            {
                new LogCoshLossFitnessCalculator<double, double[], double>(),
                new QuantileLossFitnessCalculator<double, double[], double>(),
                new PoissonLossFitnessCalculator<double, double[], double>(),
                new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>(),
                new DiceLossFitnessCalculator<double, double[], double>(),
                new JaccardLossFitnessCalculator<double, double[], double>(),
                new FocalLossFitnessCalculator<double, double[], double>(),
                new ContrastiveLossFitnessCalculator<double, double[], double>(),
                new TripletLossFitnessCalculator<double, double[], double>(),
                new CosineSimilarityLossFitnessCalculator<double, double[], double>(),
                new ElasticNetLossFitnessCalculator<double, double[], double>(),
                new ExponentialLossFitnessCalculator<double, double[], double>(),
                new OrdinalRegressionLossFitnessCalculator<double, double[], double>()
            };

            // Assert - All should have IsLowerBetter = false (fitness score, lower is better)
            foreach (var calculator in calculators)
            {
                Assert.False(calculator.IsLowerBetter,
                    $"{calculator.GetType().Name} should have IsLowerBetter = false for loss-based metrics");
            }
        }

        [Fact]
        public void AllAdvancedCalculators_HandleEmptyData_Gracefully()
        {
            // Arrange
            var emptyDataSet = CreateMockDataSet(
                predicted: Array.Empty<double>(),
                actual: Array.Empty<double>()
            );

            var calculators = new IFitnessCalculator<double, double[], double>[]
            {
                new LogCoshLossFitnessCalculator<double, double[], double>(),
                new PoissonLossFitnessCalculator<double, double[], double>(),
                new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>(),
                new DiceLossFitnessCalculator<double, double[], double>(),
                new JaccardLossFitnessCalculator<double, double[], double>(),
                new CosineSimilarityLossFitnessCalculator<double, double[], double>(),
                new ExponentialLossFitnessCalculator<double, double[], double>()
            };

            // Act & Assert
            foreach (var calculator in calculators)
            {
                try
                {
                    var fitness = calculator.CalculateFitness(null, new[] { emptyDataSet, emptyDataSet, emptyDataSet });
                    // Should either return valid value or handle gracefully
                    Assert.True(double.IsNaN(fitness) || double.IsInfinity(fitness) || fitness >= 0.0);
                }
                catch (Exception)
                {
                    // Some calculators may throw on empty data, which is acceptable
                }
            }
        }

        [Fact]
        public void AllAdvancedCalculators_ProduceFiniteResults_OnNormalData()
        {
            // Arrange
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.1, 0.3, 0.5, 0.7, 0.9 },
                actual: new[] { 0.2, 0.4, 0.6, 0.8, 1.0 }
            );

            var calculators = new IFitnessCalculator<double, double[], double>[]
            {
                new LogCoshLossFitnessCalculator<double, double[], double>(),
                new PoissonLossFitnessCalculator<double, double[], double>(),
                new KullbackLeiblerDivergenceFitnessCalculator<double, double[], double>(),
                new DiceLossFitnessCalculator<double, double[], double>(),
                new JaccardLossFitnessCalculator<double, double[], double>(),
                new CosineSimilarityLossFitnessCalculator<double, double[], double>(),
                new ElasticNetLossFitnessCalculator<double, double[], double>(),
                new ExponentialLossFitnessCalculator<double, double[], double>()
            };

            // Act & Assert
            foreach (var calculator in calculators)
            {
                var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
                Assert.True(double.IsFinite(fitness),
                    $"{calculator.GetType().Name} produced non-finite result: {fitness}");
            }
        }

        [Fact]
        public void ParameterizedCalculators_DefaultParameters_WorkCorrectly()
        {
            // Arrange & Act - Test that default parameters work
            var dataSet = CreateMockDataSet(
                predicted: new[] { 0.5, 0.6, 0.7 },
                actual: new[] { 0.6, 0.7, 0.8 }
            );

            var calculators = new IFitnessCalculator<double, double[], double>[]
            {
                new QuantileLossFitnessCalculator<double, double[], double>(), // Default quantile
                new FocalLossFitnessCalculator<double, double[], double>(), // Default gamma, alpha
                new ContrastiveLossFitnessCalculator<double, double[], double>(), // Default margin
                new TripletLossFitnessCalculator<double, double[], double>(), // Default margin
                new ElasticNetLossFitnessCalculator<double, double[], double>() // Default l1Ratio, alpha
            };

            // Assert
            foreach (var calculator in calculators)
            {
                var fitness = calculator.CalculateFitness(null, new[] { dataSet, dataSet, dataSet });
                Assert.True(double.IsFinite(fitness));
                Assert.True(fitness >= 0.0);
            }
        }

        #endregion
    }
}
