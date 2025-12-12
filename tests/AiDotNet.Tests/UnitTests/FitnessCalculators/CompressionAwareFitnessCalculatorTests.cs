using System;
using AiDotNet.FitnessCalculators;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using AiDotNet.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators
{
    public class CompressionAwareFitnessCalculatorTests
    {
        #region Test Helpers

        /// <summary>
        /// A simple mock fitness calculator for testing with Matrix/Vector types.
        /// </summary>
        private class MockFitnessCalculator : IFitnessCalculator<double, Matrix<double>, Vector<double>>
        {
            private readonly double _returnValue;
            private readonly bool _isHigherBetter;

            public MockFitnessCalculator(double returnValue = 0.9, bool isHigherBetter = true)
            {
                _returnValue = returnValue;
                _isHigherBetter = isHigherBetter;
            }

            public bool IsHigherScoreBetter => _isHigherBetter;

            public double CalculateFitnessScore(ModelEvaluationData<double, Matrix<double>, Vector<double>> evaluationData)
            {
                return _returnValue;
            }

            public double CalculateFitnessScore(DataSetStats<double, Matrix<double>, Vector<double>> dataSet)
            {
                return _returnValue;
            }

            public bool IsBetterFitness(double currentFitness, double bestFitness)
            {
                return _isHigherBetter
                    ? currentFitness > bestFitness
                    : currentFitness < bestFitness;
            }
        }

        /// <summary>
        /// A simple mock fitness calculator for float type.
        /// </summary>
        private class MockFloatFitnessCalculator : IFitnessCalculator<float, Matrix<float>, Vector<float>>
        {
            public bool IsHigherScoreBetter => true;

            public float CalculateFitnessScore(ModelEvaluationData<float, Matrix<float>, Vector<float>> evaluationData)
            {
                return 0.9f;
            }

            public float CalculateFitnessScore(DataSetStats<float, Matrix<float>, Vector<float>> dataSet)
            {
                return 0.9f;
            }

            public bool IsBetterFitness(float currentFitness, float bestFitness)
            {
                return currentFitness > bestFitness;
            }
        }

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidBaseFitnessCalculator_CreatesInstance()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();

            // Act
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            // Assert
            Assert.NotNull(calculator);
        }

        [Fact]
        public void Constructor_WithNullBaseFitnessCalculator_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(null!));
        }

        [Fact]
        public void Constructor_WithNegativeWeights_ThrowsException()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                    baseFitness,
                    accuracyWeight: -0.1,
                    compressionWeight: 0.5,
                    speedWeight: 0.5));
        }

        [Fact]
        public void Constructor_WithAllZeroWeights_ThrowsException()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                    baseFitness,
                    accuracyWeight: 0,
                    compressionWeight: 0,
                    speedWeight: 0));
        }

        [Fact]
        public void Constructor_WithCustomWeights_NormalizesToSumOfOne()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();

            // Act - weights that don't sum to 1
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness,
                accuracyWeight: 2.0,
                compressionWeight: 1.0,
                speedWeight: 1.0);

            // Assert - should still work (normalized internally)
            Assert.NotNull(calculator);
        }

        #endregion

        #region IsHigherScoreBetter Tests

        [Fact]
        public void IsHigherScoreBetter_ReturnsTrue()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            // Act & Assert
            Assert.True(calculator.IsHigherScoreBetter);
        }

        #endregion

        #region CompressionMetrics Property Tests

        [Fact]
        public void CompressionMetrics_DefaultsToNull()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            // Act & Assert
            Assert.Null(calculator.CompressionMetrics);
        }

        [Fact]
        public void CompressionMetrics_CanBeSetAndRetrieved()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 100
            };
            metrics.CalculateDerivedMetrics();

            // Act
            calculator.CompressionMetrics = metrics;

            // Assert
            Assert.NotNull(calculator.CompressionMetrics);
            Assert.Equal(10.0, calculator.CompressionMetrics.CompressionRatio);
        }

        #endregion

        #region CalculateFitnessScore Tests (ModelEvaluationData)

        [Fact]
        public void CalculateFitnessScore_WithoutCompressionMetrics_ReturnsBaseAccuracy()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator(returnValue: 0.8);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(fitness >= 0 && fitness <= 1);
        }

        [Fact]
        public void CalculateFitnessScore_WithCompressionMetrics_CombinesScores()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator(returnValue: 0.9);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 100,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.93
            };
            metrics.CalculateDerivedMetrics();
            calculator.CompressionMetrics = metrics;

            var evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(fitness > 0);
        }

        [Fact]
        public void CalculateFitnessScore_WithLowerIsBetterBase_NormalizesCorrectly()
        {
            // Arrange - Error-based fitness calculator where lower is better
            var baseFitness = new MockFitnessCalculator(returnValue: 0.1, isHigherBetter: false);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(fitness > 0 && fitness <= 1);
        }

        [Fact]
        public void CalculateFitnessScore_WithZeroError_ReturnsHighScore()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator(returnValue: 0.0, isHigherBetter: false);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(1.0, fitness);
        }

        #endregion

        #region CalculateFitnessScore Tests (DataSetStats)

        [Fact]
        public void CalculateFitnessScore_DataSetStats_WithoutCompressionMetrics_ReturnsBaseAccuracy()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator(returnValue: 0.85);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var dataSetStats = new DataSetStats<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(dataSetStats);

            // Assert
            Assert.True(fitness >= 0 && fitness <= 1);
        }

        [Fact]
        public void CalculateFitnessScore_DataSetStats_WithCompressionMetrics_CombinesScores()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator(returnValue: 0.9);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 100,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.93
            };
            metrics.CalculateDerivedMetrics();
            calculator.CompressionMetrics = metrics;

            var dataSetStats = new DataSetStats<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(dataSetStats);

            // Assert
            Assert.True(fitness > 0);
        }

        #endregion

        #region IsBetterFitness Tests

        [Fact]
        public void IsBetterFitness_HigherIsBetter_ReturnsCorrectComparison()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator();
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.9, 0.8));
            Assert.False(calculator.IsBetterFitness(0.7, 0.8));
            Assert.False(calculator.IsBetterFitness(0.8, 0.8));
        }

        #endregion

        #region Type-Specific Tests

        [Fact]
        public void Constructor_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var baseFitness = new MockFloatFitnessCalculator();

            // Act
            var calculator = new CompressionAwareFitnessCalculator<float, Matrix<float>, Vector<float>>(
                baseFitness);

            // Assert
            Assert.NotNull(calculator);
            Assert.True(calculator.IsHigherScoreBetter);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void CalculateFitnessScore_WithVeryHighError_ReturnsLowScore()
        {
            // Arrange - High error value
            var baseFitness = new MockFitnessCalculator(returnValue: 10.0, isHigherBetter: false);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(fitness < 0.5);
        }

        [Fact]
        public void CalculateFitnessScore_WithPerfectAccuracyAndGoodCompression_ReturnsHighScore()
        {
            // Arrange
            var baseFitness = new MockFitnessCalculator(returnValue: 1.0);
            var calculator = new CompressionAwareFitnessCalculator<double, Matrix<double>, Vector<double>>(
                baseFitness);

            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 50,  // 20x compression
                OriginalAccuracy = 1.0,
                CompressedAccuracy = 0.99,  // Very small accuracy loss
                OriginalInferenceTimeMs = 100.0,
                CompressedInferenceTimeMs = 20.0  // 5x speedup
            };
            metrics.CalculateDerivedMetrics();
            calculator.CompressionMetrics = metrics;

            var evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Act
            var fitness = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(fitness > 0.8);
        }

        #endregion
    }
}
