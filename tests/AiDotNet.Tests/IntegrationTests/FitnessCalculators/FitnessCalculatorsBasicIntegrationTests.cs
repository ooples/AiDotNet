using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Statistics;
using Xunit;
using System;

namespace AiDotNetTests.IntegrationTests.FitnessCalculators
{
    /// <summary>
    /// Comprehensive integration tests for basic fitness calculators (Part 1 of 2).
    /// Tests regression metrics (MSE, MAE, RMSE, R², Adjusted R²) and loss-based metrics
    /// (BCE, CCE, Cross-Entropy, Weighted CE, Hinge, Squared Hinge, Huber, Modified Huber).
    /// Verifies correct calculation, edge cases, and mathematical properties.
    /// </summary>
    public class FitnessCalculatorsBasicIntegrationTests
    {
        private const double EPSILON = 1e-10;

        #region Helper Methods

        /// <summary>
        /// Creates DataSetStats with error statistics for testing regression metrics.
        /// </summary>
        private DataSetStats<double, Vector<double>, Vector<double>> CreateDataSetStatsWithErrorStats(
            double mse, double mae, double rmse)
        {
            var stats = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                ErrorStats = CreateErrorStats(mse, mae, rmse)
            };
            return stats;
        }

        /// <summary>
        /// Creates DataSetStats with prediction statistics for testing R² metrics.
        /// </summary>
        private DataSetStats<double, Vector<double>, Vector<double>> CreateDataSetStatsWithPredictionStats(
            double r2, double adjustedR2)
        {
            var stats = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                PredictionStats = CreatePredictionStats(r2, adjustedR2)
            };
            return stats;
        }

        /// <summary>
        /// Creates DataSetStats with actual and predicted values for testing loss functions.
        /// </summary>
        private DataSetStats<double, Vector<double>, Vector<double>> CreateDataSetStatsWithVectors(
            Vector<double> predicted, Vector<double> actual)
        {
            return new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = predicted,
                Actual = actual
            };
        }

        /// <summary>
        /// Creates ModelEvaluationData with validation set for testing.
        /// </summary>
        private ModelEvaluationData<double, Vector<double>, Vector<double>> CreateEvaluationData(
            DataSetStats<double, Vector<double>, Vector<double>> validationSet)
        {
            return new ModelEvaluationData<double, Vector<double>, Vector<double>>
            {
                ValidationSet = validationSet
            };
        }

        /// <summary>
        /// Creates ErrorStats using reflection to set private properties.
        /// </summary>
        private ErrorStats<double> CreateErrorStats(double mse, double mae, double rmse)
        {
            var errorStats = ErrorStats<double>.Empty();
            var type = errorStats.GetType();

            type.GetProperty("MSE")?.SetValue(errorStats, mse);
            type.GetProperty("MAE")?.SetValue(errorStats, mae);
            type.GetProperty("RMSE")?.SetValue(errorStats, rmse);

            return errorStats;
        }

        /// <summary>
        /// Creates PredictionStats using reflection to set private properties.
        /// </summary>
        private PredictionStats<double> CreatePredictionStats(double r2, double adjustedR2)
        {
            var predictionStats = PredictionStats<double>.Empty();
            var type = predictionStats.GetType();

            type.GetProperty("R2")?.SetValue(predictionStats, r2);
            type.GetProperty("AdjustedR2")?.SetValue(predictionStats, adjustedR2);

            return predictionStats;
        }

        #endregion

        #region MSE (Mean Squared Error) Tests

        [Fact]
        public void MSE_KnownValues_ReturnsCorrectScore()
        {
            // Arrange
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 4.0, mae: 2.0, rmse: 2.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(4.0, score, precision: 10);
        }

        [Fact]
        public void MSE_PerfectPredictions_ReturnsZero()
        {
            // Arrange - Perfect predictions should have MSE = 0
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 0.0, mae: 0.0, rmse: 0.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, score, precision: 10);
        }

        [Fact]
        public void MSE_LargeErrors_ReturnsLargeValue()
        {
            // Arrange - Large errors should result in very large MSE (due to squaring)
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 100.0, mae: 10.0, rmse: 10.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(100.0, score, precision: 10);
        }

        [Fact]
        public void MSE_IsAlwaysNonNegative_Property()
        {
            // Arrange - MSE must always be >= 0
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 0.001, mae: 0.03, rmse: 0.0316);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(score >= 0.0, "MSE must be non-negative");
        }

        [Fact]
        public void MSE_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter, "MSE should have lower values indicating better performance");
        }

        [Fact]
        public void MSE_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert - Lower MSE is better
            Assert.True(calculator.IsBetterFitness(1.0, 2.0), "MSE of 1.0 should be better than 2.0");
            Assert.False(calculator.IsBetterFitness(2.0, 1.0), "MSE of 2.0 should not be better than 1.0");
        }

        [Fact]
        public void MSE_SingleValue_CalculatesCorrectly()
        {
            // Arrange - Test with minimal data (single value)
            var calculator = new MeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 2.25, mae: 1.5, rmse: 1.5);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert - Error of 1.5, squared = 2.25
            Assert.Equal(2.25, score, precision: 10);
        }

        #endregion

        #region MAE (Mean Absolute Error) Tests

        [Fact]
        public void MAE_KnownValues_ReturnsCorrectScore()
        {
            // Arrange
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 4.0, mae: 2.0, rmse: 2.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(2.0, score, precision: 10);
        }

        [Fact]
        public void MAE_PerfectPredictions_ReturnsZero()
        {
            // Arrange
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 0.0, mae: 0.0, rmse: 0.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, score, precision: 10);
        }

        [Fact]
        public void MAE_SymmetricErrors_CalculatesAverage()
        {
            // Arrange - Symmetric errors (+1, -1) should average to 1.0
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 1.0, mae: 1.0, rmse: 1.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(1.0, score, precision: 10);
        }

        [Fact]
        public void MAE_IsAlwaysNonNegative_Property()
        {
            // Arrange
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 0.01, mae: 0.1, rmse: 0.1);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(score >= 0.0, "MAE must be non-negative");
        }

        [Fact]
        public void MAE_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void MAE_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.5, 1.0));
            Assert.False(calculator.IsBetterFitness(1.0, 0.5));
        }

        [Fact]
        public void MAE_LessSensitiveToOutliers_ThanMSE()
        {
            // Arrange - MAE should be smaller relative to MSE when outliers present
            // For errors [1, 1, 10]: MAE = 4, MSE = 34
            var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 34.0, mae: 4.0, rmse: 5.83);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var mae = calculator.CalculateFitnessScore(evaluationData);

            // Assert - MAE is much smaller than MSE, showing less sensitivity to outlier
            Assert.Equal(4.0, mae, precision: 10);
            Assert.True(mae < dataSet.ErrorStats.MSE, "MAE should be less than MSE when outliers present");
        }

        #endregion

        #region RMSE (Root Mean Squared Error) Tests

        [Fact]
        public void RMSE_KnownValues_ReturnsCorrectScore()
        {
            // Arrange
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 4.0, mae: 2.0, rmse: 2.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(2.0, score, precision: 10);
        }

        [Fact]
        public void RMSE_PerfectPredictions_ReturnsZero()
        {
            // Arrange
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 0.0, mae: 0.0, rmse: 0.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, score, precision: 10);
        }

        [Fact]
        public void RMSE_EqualsSquareRootOfMSE_Property()
        {
            // Arrange - RMSE should equal sqrt(MSE)
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var mseValue = 9.0;
            var rmseExpected = Math.Sqrt(mseValue);
            var dataSet = CreateDataSetStatsWithErrorStats(mse: mseValue, mae: 3.0, rmse: rmseExpected);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(rmseExpected, score, precision: 10);
            Assert.Equal(Math.Sqrt(mseValue), score, precision: 10);
        }

        [Fact]
        public void RMSE_IsAlwaysNonNegative_Property()
        {
            // Arrange
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 0.25, mae: 0.5, rmse: 0.5);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.True(score >= 0.0, "RMSE must be non-negative");
        }

        [Fact]
        public void RMSE_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void RMSE_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(1.5, 2.5));
            Assert.False(calculator.IsBetterFitness(2.5, 1.5));
        }

        [Fact]
        public void RMSE_InSameUnitsAsData_Property()
        {
            // Arrange - RMSE is in same units as original data (unlike MSE which is squared)
            var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithErrorStats(mse: 16.0, mae: 4.0, rmse: 4.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var rmse = calculator.CalculateFitnessScore(evaluationData);

            // Assert - RMSE (4.0) is in original units, MSE (16.0) is in squared units
            Assert.Equal(4.0, rmse, precision: 10);
            Assert.True(rmse < dataSet.ErrorStats.MSE, "RMSE should be less than MSE for errors > 1");
        }

        #endregion

        #region R² (R-Squared) Tests

        [Fact]
        public void RSquared_PerfectFit_ReturnsOne()
        {
            // Arrange - Perfect fit has R² = 1.0
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 1.0, adjustedR2: 1.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(1.0, score, precision: 10);
        }

        [Fact]
        public void RSquared_NoFit_ReturnsZero()
        {
            // Arrange - Model no better than mean has R² = 0.0
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 0.0, adjustedR2: 0.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, score, precision: 10);
        }

        [Fact]
        public void RSquared_WorseAsNaive_CanBeNegative()
        {
            // Arrange - Model worse than predicting mean can have R² < 0
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: -0.5, adjustedR2: -0.6);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(-0.5, score, precision: 10);
            Assert.True(score < 0.0, "R² can be negative for very poor fits");
        }

        [Fact]
        public void RSquared_GoodFit_ReturnsBetweenZeroAndOne()
        {
            // Arrange - Good fit typically has R² between 0 and 1
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 0.85, adjustedR2: 0.84);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.85, score, precision: 10);
            Assert.InRange(score, 0.0, 1.0);
        }

        [Fact]
        public void RSquared_IsLowerScoreBetter_Property()
        {
            // Arrange - Note: IsHigherScoreBetter is false due to optimization convention
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void RSquared_IsBetterFitness_UsesMinimizationConvention()
        {
            // Arrange - Lower R² is considered "better" due to minimization convention
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert - Due to minimization convention
            Assert.True(calculator.IsBetterFitness(0.7, 0.9));
            Assert.False(calculator.IsBetterFitness(0.9, 0.7));
        }

        [Fact]
        public void RSquared_InterpretationAsVarianceExplained_Property()
        {
            // Arrange - R² of 0.75 means 75% of variance explained
            var calculator = new RSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 0.75, adjustedR2: 0.73);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.75, score, precision: 10);
        }

        #endregion

        #region Adjusted R² Tests

        [Fact]
        public void AdjustedRSquared_PerfectFit_ReturnsOne()
        {
            // Arrange
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 1.0, adjustedR2: 1.0);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(1.0, score, precision: 10);
        }

        [Fact]
        public void AdjustedRSquared_CanBeNegative()
        {
            // Arrange
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: -0.3, adjustedR2: -0.5);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(-0.5, score, precision: 10);
            Assert.True(score < 0.0);
        }

        [Fact]
        public void AdjustedRSquared_LowerThanRSquared_WhenPenalizingComplexity()
        {
            // Arrange - Adjusted R² penalizes for additional features
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 0.90, adjustedR2: 0.85);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.85, score, precision: 10);
            Assert.True(score < dataSet.PredictionStats.R2, "Adjusted R² should be <= R²");
        }

        [Fact]
        public void AdjustedRSquared_GoodFit_ReturnsBetweenZeroAndOne()
        {
            // Arrange
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 0.80, adjustedR2: 0.78);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.78, score, precision: 10);
            Assert.InRange(score, 0.0, 1.0);
        }

        [Fact]
        public void AdjustedRSquared_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void AdjustedRSquared_IsBetterFitness_UsesMinimizationConvention()
        {
            // Arrange
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.6, 0.8));
            Assert.False(calculator.IsBetterFitness(0.8, 0.6));
        }

        [Fact]
        public void AdjustedRSquared_AccountsForModelComplexity_Property()
        {
            // Arrange - Adjusted R² accounts for number of predictors
            var calculator = new AdjustedRSquaredFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = CreateDataSetStatsWithPredictionStats(r2: 0.95, adjustedR2: 0.88);
            var evaluationData = CreateEvaluationData(dataSet);

            // Act
            var score = calculator.CalculateFitnessScore(evaluationData);

            // Assert - Larger penalty suggests more features relative to sample size
            Assert.Equal(0.88, score, precision: 10);
            var penalty = dataSet.PredictionStats.R2 - score;
            Assert.True(penalty > 0.0, "Should have positive penalty for model complexity");
        }

        #endregion

        #region Binary Cross-Entropy Loss Tests

        [Fact]
        public void BinaryCrossEntropy_PerfectPredictions_ReturnsNearZero()
        {
            // Arrange - Perfect binary predictions
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.9999, 0.0001, 0.9999, 0.0001 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should be very close to 0
            Assert.True(score < 0.001, $"Expected near zero, got {score}");
        }

        [Fact]
        public void BinaryCrossEntropy_CompletelyWrong_ReturnsLargeValue()
        {
            // Arrange - Completely wrong predictions
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.01, 0.99, 0.01, 0.99 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should be large (> 4 for very confident wrong predictions)
            Assert.True(score > 4.0, $"Expected large value, got {score}");
        }

        [Fact]
        public void BinaryCrossEntropy_UncertainPredictions_ReturnsModerateValue()
        {
            // Arrange - Uncertain predictions (0.5) for all
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - BCE(0.5) = -log(0.5) ≈ 0.693
            Assert.Equal(Math.Log(2), score, precision: 2);
        }

        [Fact]
        public void BinaryCrossEntropy_KnownCalculation_MatchesFormula()
        {
            // Arrange - Manual calculation for verification
            // BCE = -mean(y*log(p) + (1-y)*log(1-p))
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.8, 0.2 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Manual: -(1*log(0.8) + 0*log(0.2) + 0*log(0.2) + 1*log(0.8)) / 2
            // = -log(0.8) ≈ 0.223
            var expected = -Math.Log(0.8);
            Assert.Equal(expected, score, precision: 3);
        }

        [Fact]
        public void BinaryCrossEntropy_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void BinaryCrossEntropy_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.1, 0.5));
            Assert.False(calculator.IsBetterFitness(0.5, 0.1));
        }

        [Fact]
        public void BinaryCrossEntropy_SingleSample_CalculatesCorrectly()
        {
            // Arrange - Single prediction
            var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.7 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - BCE = -log(0.7) ≈ 0.357
            var expected = -Math.Log(0.7);
            Assert.Equal(expected, score, precision: 3);
        }

        #endregion

        #region Categorical Cross-Entropy Loss Tests

        [Fact]
        public void CategoricalCrossEntropy_PerfectPredictions_ReturnsNearZero()
        {
            // Arrange - Perfect categorical predictions
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.99, 0.005, 0.005, 0.005, 0.99, 0.005 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score < 0.02, $"Expected near zero, got {score}");
        }

        [Fact]
        public void CategoricalCrossEntropy_CompletelyWrong_ReturnsLargeValue()
        {
            // Arrange
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.01, 0.99, 0.01, 0.99 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score > 4.0, $"Expected large value, got {score}");
        }

        [Fact]
        public void CategoricalCrossEntropy_UniformPredictions_ReturnsLogOfClasses()
        {
            // Arrange - Uniform distribution over 3 classes
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.33, 0.33, 0.34 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - CCE for uniform distribution ≈ log(3) ≈ 1.099
            Assert.True(score > 1.0 && score < 1.2, $"Expected ~1.1, got {score}");
        }

        [Fact]
        public void CategoricalCrossEntropy_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void CategoricalCrossEntropy_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.2, 0.8));
            Assert.False(calculator.IsBetterFitness(0.8, 0.2));
        }

        [Fact]
        public void CategoricalCrossEntropy_MultipleClasses_CalculatesCorrectly()
        {
            // Arrange - 4-class problem
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.7, 0.1, 0.1, 0.1 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - CCE = -log(0.7) ≈ 0.357
            var expected = -Math.Log(0.7);
            Assert.Equal(expected, score, precision: 2);
        }

        [Fact]
        public void CategoricalCrossEntropy_PartiallyCorrect_ReturnsModerateLoss()
        {
            // Arrange - Somewhat confident correct prediction
            var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.6, 0.3, 0.1 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - CCE = -log(0.6) ≈ 0.511
            var expected = -Math.Log(0.6);
            Assert.Equal(expected, score, precision: 2);
        }

        #endregion

        #region Cross-Entropy Loss Tests

        [Fact]
        public void CrossEntropy_PerfectPredictions_ReturnsNearZero()
        {
            // Arrange
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.999, 0.001, 0.999 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score < 0.01, $"Expected near zero, got {score}");
        }

        [Fact]
        public void CrossEntropy_CompletelyWrong_ReturnsLargeValue()
        {
            // Arrange
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.001, 0.999, 0.001 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score > 6.0, $"Expected large value, got {score}");
        }

        [Fact]
        public void CrossEntropy_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void CrossEntropy_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.3, 0.9));
            Assert.False(calculator.IsBetterFitness(0.9, 0.3));
        }

        [Fact]
        public void CrossEntropy_MixedPredictions_CalculatesAverageLoss()
        {
            // Arrange - Mix of good and bad predictions
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.9, 0.1, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should be between perfect (0) and completely wrong (>6)
            Assert.InRange(score, 0.0, 2.0);
        }

        [Fact]
        public void CrossEntropy_SingleValue_CalculatesCorrectly()
        {
            // Arrange
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.8 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - CE = -log(0.8) ≈ 0.223
            var expected = -Math.Log(0.8);
            Assert.Equal(expected, score, precision: 2);
        }

        [Fact]
        public void CrossEntropy_LargeBatch_HandlesCorrectly()
        {
            // Arrange - Large number of predictions
            var size = 100;
            var predictedValues = new double[size];
            var actualValues = new double[size];
            for (int i = 0; i < size; i++)
            {
                predictedValues[i] = 0.9;
                actualValues[i] = 1.0;
            }
            var calculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(predictedValues);
            var actual = new Vector<double>(actualValues);
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            var expected = -Math.Log(0.9);
            Assert.Equal(expected, score, precision: 2);
        }

        #endregion

        #region Weighted Cross-Entropy Loss Tests

        [Fact]
        public void WeightedCrossEntropy_PerfectPredictions_ReturnsNearZero()
        {
            // Arrange
            var weights = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
            var predicted = new Vector<double>(new[] { 0.999, 0.001, 0.999 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score < 0.01, $"Expected near zero, got {score}");
        }

        [Fact]
        public void WeightedCrossEntropy_EqualWeights_EqualsRegularCrossEntropy()
        {
            // Arrange - Equal weights should give same result as unweighted
            var weights = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var weightedCalculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
            var regularCalculator = new CrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.8, 0.2, 0.9 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var weightedScore = weightedCalculator.CalculateFitnessScore(dataSet);
            var regularScore = regularCalculator.CalculateFitnessScore(dataSet);

            // Assert - Should be approximately equal
            Assert.Equal(regularScore, weightedScore, precision: 2);
        }

        [Fact]
        public void WeightedCrossEntropy_HigherWeightOnErrors_IncreasesLoss()
        {
            // Arrange - Higher weight on misclassified samples
            var lowWeights = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
            var highWeights = new Vector<double>(new[] { 2.0, 2.0, 2.0 });
            var lowCalculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(lowWeights);
            var highCalculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(highWeights);
            var predicted = new Vector<double>(new[] { 0.6, 0.4, 0.7 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var lowScore = lowCalculator.CalculateFitnessScore(dataSet);
            var highScore = highCalculator.CalculateFitnessScore(dataSet);

            // Assert - Higher weights should give higher loss
            Assert.True(highScore > lowScore, "Higher weights should increase loss");
        }

        [Fact]
        public void WeightedCrossEntropy_NullWeights_UsesDefaultWeights()
        {
            // Arrange - Null weights should use default (all 1s)
            var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(null);
            var predicted = new Vector<double>(new[] { 0.9, 0.1 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should not throw and should return reasonable value
            Assert.True(score >= 0.0 && score < 5.0);
        }

        [Fact]
        public void WeightedCrossEntropy_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void WeightedCrossEntropy_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.4, 0.8));
            Assert.False(calculator.IsBetterFitness(0.8, 0.4));
        }

        [Fact]
        public void WeightedCrossEntropy_DifferentWeightsPerSample_AppliesCorrectly()
        {
            // Arrange - Different weights for each sample
            var weights = new Vector<double>(new[] { 3.0, 1.0, 2.0 });
            var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Vector<double>, Vector<double>>(weights);
            var predicted = new Vector<double>(new[] { 0.7, 0.7, 0.7 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should compute weighted average
            Assert.True(score > 0.0 && score < 1.0);
        }

        #endregion

        #region Hinge Loss Tests

        [Fact]
        public void HingeLoss_PerfectSeparation_ReturnsZero()
        {
            // Arrange - Perfect separation with margin > 1
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 2.0, -2.0, 3.0, -1.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0, -1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - All y*ŷ > 1, so loss = 0
            Assert.Equal(0.0, score, precision: 6);
        }

        [Fact]
        public void HingeLoss_WrongSidePredictions_ReturnsPositiveValue()
        {
            // Arrange - Predictions on wrong side
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { -0.5, 0.5, -0.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should be positive due to violations
            Assert.True(score > 0.0, $"Expected positive loss, got {score}");
        }

        [Fact]
        public void HingeLoss_OnMargin_ReturnsZero()
        {
            // Arrange - Exactly on margin (y*ŷ = 1)
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 1.0, -1.0 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - max(0, 1 - 1*1) = 0
            Assert.Equal(0.0, score, precision: 6);
        }

        [Fact]
        public void HingeLoss_InsideMargin_ReturnsPositiveValue()
        {
            // Arrange - Correct side but inside margin (0 < y*ŷ < 1)
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.5, -0.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - max(0, 1 - 0.5) = 0.5 for each, average = 0.5
            Assert.Equal(0.5, score, precision: 6);
        }

        [Fact]
        public void HingeLoss_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void HingeLoss_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.2, 0.6));
            Assert.False(calculator.IsBetterFitness(0.6, 0.2));
        }

        [Fact]
        public void HingeLoss_LinearPenalty_VerifyProperty()
        {
            // Arrange - Hinge loss increases linearly for violations
            var calculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.0 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - max(0, 1 - 1*0) = 1
            Assert.Equal(1.0, score, precision: 6);
        }

        #endregion

        #region Squared Hinge Loss Tests

        [Fact]
        public void SquaredHingeLoss_PerfectSeparation_ReturnsZero()
        {
            // Arrange
            var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 2.0, -2.0, 3.0 });
            var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.Equal(0.0, score, precision: 6);
        }

        [Fact]
        public void SquaredHingeLoss_WrongSidePredictions_ReturnsPositiveValue()
        {
            // Arrange
            var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { -0.5, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score > 0.0, $"Expected positive loss, got {score}");
        }

        [Fact]
        public void SquaredHingeLoss_InsideMargin_ReturnsSquaredPenalty()
        {
            // Arrange - Inside margin: max(0, 1 - y*ŷ)²
            var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.5 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - max(0, 1 - 0.5)² = 0.5² = 0.25
            Assert.Equal(0.25, score, precision: 6);
        }

        [Fact]
        public void SquaredHingeLoss_LargerPenaltyThanHinge_ForViolations()
        {
            // Arrange - Squared hinge should penalize more than hinge for same violation
            var squaredCalculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var hingeCalculator = new HingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { -1.0 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var squaredScore = squaredCalculator.CalculateFitnessScore(dataSet);
            var hingeScore = hingeCalculator.CalculateFitnessScore(dataSet);

            // Assert - Squared: max(0, 1-(-1))² = 4, Hinge: max(0, 1-(-1)) = 2
            Assert.True(squaredScore > hingeScore, "Squared hinge should penalize more");
            Assert.Equal(4.0, squaredScore, precision: 5);
            Assert.Equal(2.0, hingeScore, precision: 5);
        }

        [Fact]
        public void SquaredHingeLoss_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void SquaredHingeLoss_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.1, 0.5));
            Assert.False(calculator.IsBetterFitness(0.5, 0.1));
        }

        [Fact]
        public void SquaredHingeLoss_QuadraticPenalty_VerifyProperty()
        {
            // Arrange - Verify quadratic growth
            var calculator = new SquaredHingeLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.0 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - max(0, 1 - 0)² = 1
            Assert.Equal(1.0, score, precision: 6);
        }

        #endregion

        #region Huber Loss Tests

        [Fact]
        public void HuberLoss_PerfectPredictions_ReturnsZero()
        {
            // Arrange
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.Equal(0.0, score, precision: 6);
        }

        [Fact]
        public void HuberLoss_SmallErrors_UsesMSE()
        {
            // Arrange - Errors within delta use MSE formula
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 1.0, 2.0 });
            var actual = new Vector<double>(new[] { 1.5, 2.5 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Errors of 0.5 each, squared: 0.25, mean = 0.125 (half MSE for Huber)
            Assert.True(score > 0.0 && score < 0.5, $"Expected small quadratic loss, got {score}");
        }

        [Fact]
        public void HuberLoss_LargeErrors_UsesMAE()
        {
            // Arrange - Errors beyond delta use MAE formula
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 0.0, 0.0 });
            var actual = new Vector<double>(new[] { 5.0, 5.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Large errors use linear penalty
            Assert.True(score > 0.0, $"Expected positive loss, got {score}");
        }

        [Fact]
        public void HuberLoss_DefaultDelta_UsesOne()
        {
            // Arrange - Default delta should be 1.0
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.0 });
            var actual = new Vector<double>(new[] { 2.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Error of 2.0 with delta=1.0
            Assert.True(score > 0.0);
        }

        [Fact]
        public void HuberLoss_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void HuberLoss_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.3, 0.7));
            Assert.False(calculator.IsBetterFitness(0.7, 0.3));
        }

        [Fact]
        public void HuberLoss_RobustToOutliers_ComparedToMSE()
        {
            // Arrange - Huber should be more robust than MSE
            var calculator = new HuberLossFitnessCalculator<double, Vector<double>, Vector<double>>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
            var actual = new Vector<double>(new[] { 1.1, 1.1, 1.1, 10.0 }); // One outlier
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should not be dominated by outlier
            Assert.True(score < 5.0, $"Huber should be robust to outlier, got {score}");
        }

        #endregion

        #region Modified Huber Loss Tests

        [Fact]
        public void ModifiedHuberLoss_PerfectSeparation_ReturnsZero()
        {
            // Arrange - Perfect separation with large margin
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 2.0, -2.0, 3.0 });
            var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.Equal(0.0, score, precision: 6);
        }

        [Fact]
        public void ModifiedHuberLoss_WrongPredictions_ReturnsPositiveValue()
        {
            // Arrange
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { -0.5, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.True(score > 0.0, $"Expected positive loss, got {score}");
        }

        [Fact]
        public void ModifiedHuberLoss_InsideMargin_UsesQuadraticPenalty()
        {
            // Arrange - For -1 < y*ŷ < 1, uses quadratic
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.5 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should use quadratic penalty
            Assert.True(score > 0.0 && score < 1.0, $"Expected moderate loss, got {score}");
        }

        [Fact]
        public void ModifiedHuberLoss_VeryWrong_UsesLinearPenalty()
        {
            // Arrange - For y*ŷ < -1, uses linear penalty
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { -2.0 });
            var actual = new Vector<double>(new[] { 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should use linear penalty for large errors
            Assert.True(score > 1.0, $"Expected large loss, got {score}");
        }

        [Fact]
        public void ModifiedHuberLoss_IsLowerScoreBetter_Property()
        {
            // Arrange
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void ModifiedHuberLoss_IsBetterFitness_LowerScoreIsBetter()
        {
            // Arrange
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.True(calculator.IsBetterFitness(0.2, 0.8));
            Assert.False(calculator.IsBetterFitness(0.8, 0.2));
        }

        [Fact]
        public void ModifiedHuberLoss_RobustToOutliers_Property()
        {
            // Arrange - Modified Huber should be robust to outliers
            var calculator = new ModifiedHuberLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var predicted = new Vector<double>(new[] { 0.9, 0.9, -5.0 }); // One major outlier
            var actual = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var dataSet = CreateDataSetStatsWithVectors(predicted, actual);

            // Act
            var score = calculator.CalculateFitnessScore(dataSet);

            // Assert - Should not be dominated by outlier
            Assert.True(score < 10.0, $"Should be robust to outlier, got {score}");
        }

        #endregion
    }
}
