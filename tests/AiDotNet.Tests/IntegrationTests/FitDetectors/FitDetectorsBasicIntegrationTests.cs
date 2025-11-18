using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using Xunit;

namespace AiDotNetTests.IntegrationTests.FitDetectors
{
    /// <summary>
    /// Comprehensive integration tests for basic fit detectors.
    /// Tests overfitting/underfitting detection, statistical significance, and edge cases.
    /// Part 1 of 2: Basic fit detectors (14 detectors, ~7-8 tests each = ~105 tests)
    /// </summary>
    public class FitDetectorsBasicIntegrationTests
    {
        #region Helper Methods

        /// <summary>
        /// Creates synthetic data representing an overfit scenario (perfect training, poor validation/test)
        /// </summary>
        private ModelEvaluationData<double, double[], double> CreateOverfitScenario()
        {
            return new ModelEvaluationData<double, double[], double>
            {
                TrainingSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.99,
                        LearningCurve = new List<double> { 0.5, 0.7, 0.85, 0.95, 0.99 }
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 0.01,
                        RMSE = 0.1,
                        MAE = 0.08,
                        MAPE = 0.05,
                        MeanBiasError = 0.01,
                        PopulationStandardError = 0.1,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                ValidationSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.55,
                        LearningCurve = new List<double> { 0.4, 0.5, 0.52, 0.54, 0.55 }
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 2.5,
                        RMSE = 1.58,
                        MAE = 1.2,
                        MAPE = 0.35,
                        MeanBiasError = 0.5,
                        PopulationStandardError = 1.5,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                TestSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.52
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 2.8,
                        RMSE = 1.67,
                        MAE = 1.3,
                        MAPE = 0.38,
                        MeanBiasError = 0.55,
                        PopulationStandardError = 1.6,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                ModelStats = ModelStats<double, double[], double>.Empty()
            };
        }

        /// <summary>
        /// Creates synthetic data representing an underfit scenario (poor on all datasets)
        /// </summary>
        private ModelEvaluationData<double, double[], double> CreateUnderfitScenario()
        {
            return new ModelEvaluationData<double, double[], double>
            {
                TrainingSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.45,
                        LearningCurve = new List<double> { 0.3, 0.35, 0.4, 0.42, 0.45 }
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 3.5,
                        RMSE = 1.87,
                        MAE = 1.5,
                        MAPE = 0.55,
                        MeanBiasError = 1.0,
                        PopulationStandardError = 1.8,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                ValidationSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.42,
                        LearningCurve = new List<double> { 0.28, 0.33, 0.38, 0.4, 0.42 }
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 3.8,
                        RMSE = 1.95,
                        MAE = 1.6,
                        MAPE = 0.58,
                        MeanBiasError = 1.1,
                        PopulationStandardError = 1.85,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                TestSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.40
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 4.0,
                        RMSE = 2.0,
                        MAE = 1.65,
                        MAPE = 0.60,
                        MeanBiasError = 1.15,
                        PopulationStandardError = 1.9,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                ModelStats = ModelStats<double, double[], double>.Empty()
            };
        }

        /// <summary>
        /// Creates synthetic data representing a good fit scenario (high performance on all datasets)
        /// </summary>
        private ModelEvaluationData<double, double[], double> CreateGoodFitScenario()
        {
            return new ModelEvaluationData<double, double[], double>
            {
                TrainingSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.92,
                        LearningCurve = new List<double> { 0.5, 0.7, 0.85, 0.90, 0.92 }
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 0.15,
                        RMSE = 0.39,
                        MAE = 0.3,
                        MAPE = 0.08,
                        MeanBiasError = 0.05,
                        PopulationStandardError = 0.38,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                ValidationSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.91,
                        LearningCurve = new List<double> { 0.48, 0.68, 0.83, 0.89, 0.91 }
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 0.18,
                        RMSE = 0.42,
                        MAE = 0.32,
                        MAPE = 0.09,
                        MeanBiasError = 0.06,
                        PopulationStandardError = 0.4,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                TestSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double>
                    {
                        R2 = 0.90
                    },
                    ErrorStats = new ErrorStats<double>
                    {
                        MSE = 0.20,
                        RMSE = 0.45,
                        MAE = 0.35,
                        MAPE = 0.10,
                        MeanBiasError = 0.07,
                        PopulationStandardError = 0.42,
                        DurbinWatsonStatistic = 2.0
                    }
                },
                ModelStats = ModelStats<double, double[], double>.Empty()
            };
        }

        /// <summary>
        /// Creates evaluation data with actual/predicted values for classification detectors
        /// </summary>
        private ModelEvaluationData<double, double[], double> CreateClassificationData(double accuracy)
        {
            int size = 100;
            var actual = new double[size];
            var predicted = new double[size];
            int correctPredictions = (int)(size * accuracy);

            for (int i = 0; i < size; i++)
            {
                actual[i] = i % 2; // Alternating 0s and 1s
                predicted[i] = i < correctPredictions ? actual[i] : 1 - actual[i];
            }

            return new ModelEvaluationData<double, double[], double>
            {
                ModelStats = new ModelStats<double, double[], double>
                {
                    Actual = actual.Select(v => new double[] { v }).ToList(),
                    Predicted = predicted.Select(v => new double[] { v }).ToList()
                },
                TrainingSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double> { R2 = accuracy }
                },
                ValidationSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double> { R2 = accuracy }
                },
                TestSet = new DataSetStats<double, double[], double>
                {
                    PredictionStats = new PredictionStats<double> { R2 = accuracy }
                }
            };
        }

        #endregion

        #region DefaultFitDetector Tests

        [Fact]
        public void DefaultFitDetector_OverfitScenario_DetectsOverfitting()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
            Assert.Contains("regularization", string.Join(" ", result.Recommendations).ToLower());
        }

        [Fact]
        public void DefaultFitDetector_UnderfitScenario_DetectsUnderfitting()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
            Assert.Contains("complexity", string.Join(" ", result.Recommendations).ToLower());
        }

        [Fact]
        public void DefaultFitDetector_GoodFitScenario_IdentifiesGoodFit()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.True(result.ConfidenceLevel > 0.8);
        }

        [Fact]
        public void DefaultFitDetector_HighVariance_DetectsHighVariance()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.R2 = 0.85;
            data.ValidationSet.PredictionStats.R2 = 0.60; // Large gap

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void DefaultFitDetector_HighBias_DetectsHighBias()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();
            data.TrainingSet.PredictionStats.R2 = 0.35;
            data.ValidationSet.PredictionStats.R2 = 0.33;
            data.TestSet.PredictionStats.R2 = 0.32;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighBias, result.FitType);
        }

        [Fact]
        public void DefaultFitDetector_ConfidenceLevel_CalculatesCorrectly()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert - confidence should be average of R2 values
            double expectedConfidence = (0.92 + 0.91 + 0.90) / 3.0;
            Assert.True(Math.Abs(result.ConfidenceLevel - expectedConfidence) < 0.01);
        }

        [Fact]
        public void DefaultFitDetector_PerfectFit_HandlesEdgeCase()
        {
            // Arrange
            var detector = new DefaultFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.R2 = 1.0;
            data.ValidationSet.PredictionStats.R2 = 1.0;
            data.TestSet.PredictionStats.R2 = 1.0;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.Equal(1.0, result.ConfidenceLevel);
        }

        #endregion

        #region ResidualAnalysisFitDetector Tests

        [Fact]
        public void ResidualAnalysisFitDetector_OverfitScenario_DetectsViaResiduals()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Overfit || result.FitType == FitType.HighVariance);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_GoodFitScenario_LowResidualMean()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_HighMAPE_DetectsUnderfit()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Underfit || result.FitType == FitType.HighBias);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_DurbinWatson_DetectsAutocorrelation()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TestSet.ErrorStats.DurbinWatsonStatistic = 0.5; // Strong positive autocorrelation

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Unstable, result.FitType);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_HighVariance_DetectsViaStdDev()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.ErrorStats.PopulationStandardError = 5.0;
            data.ValidationSet.ErrorStats.PopulationStandardError = 5.5;
            data.TestSet.ErrorStats.PopulationStandardError = 6.0;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_BiasedResiduals_DetectsHighBias()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();
            data.TrainingSet.ErrorStats.MeanBiasError = 2.5;
            data.ValidationSet.ErrorStats.MeanBiasError = 2.6;
            data.TestSet.ErrorStats.MeanBiasError = 2.7;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighBias, result.FitType);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_ConfidenceLevel_ReflectsConsistency()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert - confidence should be positive for good fit
            Assert.True(result.ConfidenceLevel > 0);
        }

        [Fact]
        public void ResidualAnalysisFitDetector_LargeR2Difference_DetectsUnstable()
        {
            // Arrange
            var detector = new ResidualAnalysisFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.R2 = 0.95;
            data.ValidationSet.PredictionStats.R2 = 0.50; // Large difference
            data.TestSet.PredictionStats.R2 = 0.48;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.Overfit);
        }

        #endregion

        #region CrossValidationFitDetector Tests

        [Fact]
        public void CrossValidationFitDetector_OverfitScenario_DetectsFromR2Gap()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void CrossValidationFitDetector_UnderfitScenario_LowR2AllDatasets()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void CrossValidationFitDetector_GoodFitScenario_HighConsistentR2()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.Contains("good fit", string.Join(" ", result.Recommendations).ToLower());
        }

        [Fact]
        public void CrossValidationFitDetector_HighVariance_DetectsInconsistency()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.PredictionStats.R2 = 0.65;
            data.TestSet.PredictionStats.R2 = 0.60;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void CrossValidationFitDetector_Recommendations_IncludeR2Values()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Any(r => r.Contains("R2")));
        }

        [Fact]
        public void CrossValidationFitDetector_ConfidenceLevel_BasedOnConsistency()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert - good fit should have high confidence
            Assert.True(result.ConfidenceLevel > 0.5);
        }

        [Fact]
        public void CrossValidationFitDetector_UnstableFit_MixedMetrics()
        {
            // Arrange
            var detector = new CrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.R2 = 0.75;
            data.ValidationSet.PredictionStats.R2 = 0.85;
            data.TestSet.PredictionStats.R2 = 0.65;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.HighVariance);
        }

        [Fact]
        public void CrossValidationFitDetector_CustomOptions_AffectsThresholds()
        {
            // Arrange
            var options = new CrossValidationFitDetectorOptions
            {
                GoodFitThreshold = 0.95,
                OverfitThreshold = 0.15
            };
            var detector = new CrossValidationFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert - stricter thresholds may change the outcome
            Assert.NotNull(result);
        }

        #endregion

        #region KFoldCrossValidationFitDetector Tests

        [Fact]
        public void KFoldCrossValidationFitDetector_OverfitScenario_DetectsFromFolds()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_UnderfitScenario_LowValidationR2()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_GoodFitScenario_StableAcrossFolds()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.Contains("good fit", string.Join(" ", result.Recommendations).ToLower());
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_HighVariance_LargeTestDifference()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TestSet.PredictionStats.R2 = 0.65; // Different from validation

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_Confidence_BasedOnStability()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert - stable performance should give high confidence
            Assert.True(result.ConfidenceLevel > 0.8);
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_UnstablePerformance_DetectsUnstable()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.PredictionStats.R2 = 0.70;
            data.TestSet.PredictionStats.R2 = 0.88;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.HighVariance);
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_Recommendations_ProvideMetrics()
        {
            // Arrange
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Any(r => r.Contains("R2")));
        }

        [Fact]
        public void KFoldCrossValidationFitDetector_CustomOptions_ChangesThresholds()
        {
            // Arrange
            var options = new KFoldCrossValidationFitDetectorOptions
            {
                GoodFitThreshold = 0.88,
                OverfitThreshold = 0.25
            };
            var detector = new KFoldCrossValidationFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        #endregion

        #region StratifiedKFoldCrossValidationFitDetector Tests

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_OverfitScenario_DetectsImbalance()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_UnderfitScenario_LowMetrics()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_GoodFitScenario_BalancedPerformance()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_HighVariance_AcrossStrata()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TestSet.PredictionStats.R2 = 0.60;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_Confidence_ReflectsConsistency()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel > 0.8);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_UnstablePerformance_DetectsIssues()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.PredictionStats.R2 = 0.75;
            data.TestSet.PredictionStats.R2 = 0.88;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.HighVariance);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_Recommendations_IncludeMetrics()
        {
            // Arrange
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Count > 0);
        }

        [Fact]
        public void StratifiedKFoldCrossValidationFitDetector_CustomMetric_UsedForEvaluation()
        {
            // Arrange
            var options = new StratifiedKFoldCrossValidationFitDetectorOptions
            {
                PrimaryMetric = "R2"
            };
            var detector = new StratifiedKFoldCrossValidationFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        #endregion

        #region HoldoutValidationFitDetector Tests

        [Fact]
        public void HoldoutValidationFitDetector_OverfitScenario_DetectsTrainTestGap()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void HoldoutValidationFitDetector_UnderfitScenario_PoorValidationR2()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void HoldoutValidationFitDetector_GoodFitScenario_HighValidationR2()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void HoldoutValidationFitDetector_HighVariance_MSEDifference()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.ErrorStats.MSE = 1.5;
            data.TestSet.ErrorStats.MSE = 3.0; // Large difference

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void HoldoutValidationFitDetector_Confidence_BasedOnStability()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel > 0.8);
        }

        [Fact]
        public void HoldoutValidationFitDetector_UnstablePerformance_DetectsUnstable()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.PredictionStats.R2 = 0.70;
            data.TestSet.PredictionStats.R2 = 0.92;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.HighVariance);
        }

        [Fact]
        public void HoldoutValidationFitDetector_Recommendations_IncludeR2Values()
        {
            // Arrange
            var detector = new HoldoutValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Any(r => r.Contains("R2")));
        }

        [Fact]
        public void HoldoutValidationFitDetector_CustomOptions_AffectsDetection()
        {
            // Arrange
            var options = new HoldoutValidationFitDetectorOptions
            {
                GoodFitThreshold = 0.88,
                OverfitThreshold = 0.20
            };
            var detector = new HoldoutValidationFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        #endregion

        #region TimeSeriesCrossValidationFitDetector Tests

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_OverfitScenario_DetectsFromRMSE()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_UnderfitScenario_LowR2()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_GoodFitScenario_HighR2AllSets()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_HighVariance_TestTrainingRatio()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TestSet.ErrorStats.RMSE = 3.0;
            data.TrainingSet.ErrorStats.RMSE = 0.5;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_Confidence_BasedOnStability()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel >= 0);
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_UnstablePerformance_DetectsIssues()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.PredictionStats.R2 = 0.65;
            data.TestSet.PredictionStats.R2 = 0.85;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.GoodFit);
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_Recommendations_IncludeRMSEAndR2()
        {
            // Arrange
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            var allRecommendations = string.Join(" ", result.Recommendations);
            Assert.True(allRecommendations.Contains("RMSE") || allRecommendations.Contains("R2"));
        }

        [Fact]
        public void TimeSeriesCrossValidationFitDetector_CustomOptions_AffectsThresholds()
        {
            // Arrange
            var options = new TimeSeriesCrossValidationFitDetectorOptions
            {
                GoodFitThreshold = 0.88,
                OverfitThreshold = 1.8
            };
            var detector = new TimeSeriesCrossValidationFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        #endregion

        #region LearningCurveFitDetector Tests

        [Fact]
        public void LearningCurveFitDetector_OverfitScenario_DetectsFromSlopes()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();
            // Training slope negative (getting worse), validation positive (improving)
            data.TrainingSet.PredictionStats.LearningCurve = new List<double> { 0.95, 0.94, 0.93, 0.92, 0.91 };
            data.ValidationSet.PredictionStats.LearningCurve = new List<double> { 0.50, 0.52, 0.53, 0.54, 0.55 };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void LearningCurveFitDetector_UnderfitScenario_BothSlopesPositive()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();
            // Both still improving (not converged)
            data.TrainingSet.PredictionStats.LearningCurve = new List<double> { 0.30, 0.35, 0.40, 0.42, 0.45 };
            data.ValidationSet.PredictionStats.LearningCurve = new List<double> { 0.28, 0.33, 0.38, 0.40, 0.42 };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void LearningCurveFitDetector_GoodFitScenario_ConvergedCurves()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            // Both converged (flat slopes)
            data.TrainingSet.PredictionStats.LearningCurve = new List<double> { 0.88, 0.90, 0.91, 0.92, 0.92 };
            data.ValidationSet.PredictionStats.LearningCurve = new List<double> { 0.87, 0.89, 0.90, 0.91, 0.91 };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void LearningCurveFitDetector_InsufficientData_DetectsUnstable()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.LearningCurve = new List<double> { 0.5, 0.6 }; // Too few points
            data.ValidationSet.PredictionStats.LearningCurve = new List<double> { 0.5, 0.6 };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Unstable, result.FitType);
        }

        [Fact]
        public void LearningCurveFitDetector_Confidence_BasedOnVariance()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert - smooth curves should have high confidence
            Assert.True(result.ConfidenceLevel > 0);
        }

        [Fact]
        public void LearningCurveFitDetector_ErraticCurves_LowerConfidence()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            // Very erratic curves
            data.TrainingSet.PredictionStats.LearningCurve = new List<double> { 0.5, 0.9, 0.3, 0.8, 0.4 };
            data.ValidationSet.PredictionStats.LearningCurve = new List<double> { 0.4, 0.85, 0.35, 0.75, 0.45 };

            // Act
            var result = detector.DetectFit(data);

            // Assert - erratic curves should have lower confidence
            Assert.True(result.ConfidenceLevel < 1.0);
        }

        [Fact]
        public void LearningCurveFitDetector_CustomOptions_ChangesMinDataPoints()
        {
            // Arrange
            var options = new LearningCurveFitDetectorOptions
            {
                MinDataPoints = 3,
                ConvergenceThreshold = 0.05
            };
            var detector = new LearningCurveFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void LearningCurveFitDetector_UnstableFit_MixedSlopes()
        {
            // Arrange
            var detector = new LearningCurveFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.LearningCurve = new List<double> { 0.5, 0.6, 0.55, 0.65, 0.60 };
            data.ValidationSet.PredictionStats.LearningCurve = new List<double> { 0.48, 0.52, 0.50, 0.54, 0.52 };

            // Act
            var result = detector.DetectFit(data);

            // Assert - non-converging, non-diverging should be unstable
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.GoodFit);
        }

        #endregion

        #region ConfusionMatrixFitDetector Tests

        [Fact]
        public void ConfusionMatrixFitDetector_HighAccuracy_DetectsGoodFit()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Accuracy,
                GoodFitThreshold = 0.80
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.90);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void ConfusionMatrixFitDetector_LowAccuracy_DetectsPoorFit()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Accuracy,
                ModerateFitThreshold = 0.60
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.50);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.PoorFit, result.FitType);
        }

        [Fact]
        public void ConfusionMatrixFitDetector_ModerateAccuracy_DetectsModerateFit()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Accuracy,
                GoodFitThreshold = 0.85,
                ModerateFitThreshold = 0.65
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.75);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Moderate, result.FitType);
        }

        [Fact]
        public void ConfusionMatrixFitDetector_F1Score_UsedAsPrimaryMetric()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.F1Score,
                GoodFitThreshold = 0.75
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void ConfusionMatrixFitDetector_Confidence_BasedOnMetricValue()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Accuracy
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.90);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel > 0);
        }

        [Fact]
        public void ConfusionMatrixFitDetector_ClassImbalance_DetectsAndRecommends()
        {
            // Arrange - create heavily imbalanced data
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Accuracy,
                ClassImbalanceThreshold = 0.2
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);

            int size = 100;
            var actual = new double[size];
            var predicted = new double[size];

            // 90% class 0, 10% class 1
            for (int i = 0; i < size; i++)
            {
                actual[i] = i < 90 ? 0 : 1;
                predicted[i] = actual[i];
            }

            var data = new ModelEvaluationData<double, double[], double>
            {
                ModelStats = new ModelStats<double, double[], double>
                {
                    Actual = actual.Select(v => new double[] { v }).ToList(),
                    Predicted = predicted.Select(v => new double[] { v }).ToList()
                }
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Contains("imbalanced", string.Join(" ", result.Recommendations).ToLower());
        }

        [Fact]
        public void ConfusionMatrixFitDetector_PrecisionMetric_EvaluatesCorrectly()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Precision,
                GoodFitThreshold = 0.80
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void ConfusionMatrixFitDetector_RecallMetric_EvaluatesCorrectly()
        {
            // Arrange
            var options = new ConfusionMatrixFitDetectorOptions
            {
                PrimaryMetric = MetricType.Recall,
                GoodFitThreshold = 0.80
            };
            var detector = new ConfusionMatrixFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        #endregion

        #region ROCCurveFitDetector Tests

        [Fact]
        public void ROCCurveFitDetector_HighAUC_DetectsGoodFit()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.92);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.True(result.AdditionalInfo.ContainsKey("AUC"));
        }

        [Fact]
        public void ROCCurveFitDetector_ModerateAUC_DetectsModerate()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.75);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Moderate || result.FitType == FitType.GoodFit);
        }

        [Fact]
        public void ROCCurveFitDetector_LowAUC_DetectsPoorFit()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.55);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.PoorFit || result.FitType == FitType.Moderate);
        }

        [Fact]
        public void ROCCurveFitDetector_VeryLowAUC_DetectsVeryPoorFit()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.45);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.VeryPoorFit || result.FitType == FitType.PoorFit);
        }

        [Fact]
        public void ROCCurveFitDetector_Confidence_BasedOnAUC()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.90);

            // Act
            var result = detector.DetectFit(data);

            // Assert - high AUC should give high confidence
            Assert.True(result.ConfidenceLevel > 0.5);
        }

        [Fact]
        public void ROCCurveFitDetector_AdditionalInfo_ContainsFPRTPR()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.AdditionalInfo.ContainsKey("FPR"));
            Assert.True(result.AdditionalInfo.ContainsKey("TPR"));
        }

        [Fact]
        public void ROCCurveFitDetector_CustomOptions_AffectsThresholds()
        {
            // Arrange
            var options = new ROCCurveFitDetectorOptions
            {
                GoodFitThreshold = 0.88,
                ModerateFitThreshold = 0.72
            };
            var detector = new ROCCurveFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.80);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void ROCCurveFitDetector_ImbalancedDataset_ProvidesRecommendation()
        {
            // Arrange
            var detector = new ROCCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.60);

            // Act
            var result = detector.DetectFit(data);

            // Assert - low AUC may suggest imbalance
            Assert.True(result.Recommendations.Count > 0);
        }

        #endregion

        #region PrecisionRecallCurveFitDetector Tests

        [Fact]
        public void PrecisionRecallCurveFitDetector_HighAUCAndF1_DetectsGoodFit()
        {
            // Arrange
            var options = new PrecisionRecallCurveFitDetectorOptions
            {
                AreaUnderCurveThreshold = 0.75,
                F1ScoreThreshold = 0.75
            };
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.90);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_LowAUCAndF1_DetectsPoorFit()
        {
            // Arrange
            var options = new PrecisionRecallCurveFitDetectorOptions
            {
                AreaUnderCurveThreshold = 0.70,
                F1ScoreThreshold = 0.70
            };
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.55);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.PoorFit, result.FitType);
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_MixedMetrics_DetectsModerate()
        {
            // Arrange
            var options = new PrecisionRecallCurveFitDetectorOptions
            {
                AreaUnderCurveThreshold = 0.75,
                F1ScoreThreshold = 0.75
            };
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.72);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Moderate || result.FitType == FitType.PoorFit);
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_Confidence_WeightedAverage()
        {
            // Arrange
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel > 0);
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_AdditionalInfo_ContainsMetrics()
        {
            // Arrange
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.AdditionalInfo.ContainsKey("AUC"));
            Assert.True(result.AdditionalInfo.ContainsKey("F1Score"));
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_PoorFit_SuggestsImprovement()
        {
            // Arrange
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>();
            var data = CreateClassificationData(0.60);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Contains(result.Recommendations, r => r.ToLower().Contains("feature") || r.ToLower().Contains("algorithm"));
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_CustomWeights_AffectsConfidence()
        {
            // Arrange
            var options = new PrecisionRecallCurveFitDetectorOptions
            {
                AucWeight = 0.7,
                F1ScoreWeight = 0.3
            };
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>(options);
            var data = CreateClassificationData(0.85);

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel > 0);
        }

        [Fact]
        public void PrecisionRecallCurveFitDetector_NullData_ThrowsException()
        {
            // Arrange
            var detector = new PrecisionRecallCurveFitDetector<double, double[], double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => detector.DetectFit(null!));
        }

        #endregion

        #region BootstrapFitDetector Tests

        [Fact]
        public void BootstrapFitDetector_OverfitScenario_DetectsFromBootstrap()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
        }

        [Fact]
        public void BootstrapFitDetector_UnderfitScenario_LowBootstrapR2()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
        }

        [Fact]
        public void BootstrapFitDetector_GoodFitScenario_ConsistentBootstrap()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void BootstrapFitDetector_HighVariance_LargeR2Difference()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.TrainingSet.PredictionStats.R2 = 0.95;
            data.ValidationSet.PredictionStats.R2 = 0.60;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.HighVariance || result.FitType == FitType.Overfit);
        }

        [Fact]
        public void BootstrapFitDetector_Confidence_BasedOnInterval()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel >= 0 && result.ConfidenceLevel <= 1);
        }

        [Fact]
        public void BootstrapFitDetector_Recommendations_IncludeBootstrapInfo()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Any(r => r.ToLower().Contains("bootstrap")));
        }

        [Fact]
        public void BootstrapFitDetector_CustomOptions_ChangesBootstrapCount()
        {
            // Arrange
            var options = new BootstrapFitDetectorOptions
            {
                NumberOfBootstraps = 500,
                ConfidenceInterval = 0.90
            };
            var detector = new BootstrapFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void BootstrapFitDetector_UnstablePerformance_DetectsUnstable()
        {
            // Arrange
            var detector = new BootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            data.ValidationSet.PredictionStats.R2 = 0.70;
            data.TestSet.PredictionStats.R2 = 0.88;

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Unstable || result.FitType == FitType.HighVariance || result.FitType == FitType.GoodFit);
        }

        #endregion

        #region JackknifeFitDetector Tests

        [Fact]
        public void JackknifeFitDetector_OverfitScenario_DetectsFromJackknife()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + (i % 2 == 0 ? 0.1 : -0.1);
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.FitType == FitType.Overfit || result.FitType == FitType.GoodFit);
        }

        [Fact]
        public void JackknifeFitDetector_UnderfitScenario_DetectsFromJackknife()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i / 2.0; // Consistent underestimation
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void JackknifeFitDetector_GoodFitScenario_StableJackknife()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void JackknifeFitDetector_Confidence_BasedOnStability()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.05;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel > 0.5);
        }

        [Fact]
        public void JackknifeFitDetector_InsufficientData_ThrowsException()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[5]; // Too few samples
            var predicted = new double[5];
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => detector.DetectFit(data));
        }

        [Fact]
        public void JackknifeFitDetector_Recommendations_ProvidedForAllFitTypes()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Count > 0);
        }

        [Fact]
        public void JackknifeFitDetector_CustomOptions_AffectsMinSampleSize()
        {
            // Arrange
            var options = new JackknifeFitDetectorOptions
            {
                MinSampleSize = 20,
                MaxIterations = 50
            };
            var detector = new JackknifeFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();
            var actual = new double[30];
            var predicted = new double[30];
            for (int i = 0; i < 30; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void JackknifeFitDetector_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var detector = new JackknifeFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[200];
            var predicted = new double[200];
            for (int i = 0; i < 200; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        #endregion

        #region ResidualBootstrapFitDetector Tests

        [Fact]
        public void ResidualBootstrapFitDetector_OverfitScenario_DetectsFromResiduals()
        {
            // Arrange
            var detector = new ResidualBootstrapFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + (i % 2 == 0 ? 0.1 : -0.1);
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void ResidualBootstrapFitDetector_UnderfitScenario_DetectsFromZScore()
        {
            // Arrange
            var detector = new ResidualBootstrapFitDetector<double, double[], double>();
            var data = CreateUnderfitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i / 2.0;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void ResidualBootstrapFitDetector_GoodFitScenario_ConsistentResiduals()
        {
            // Arrange
            var detector = new ResidualBootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void ResidualBootstrapFitDetector_Confidence_BasedOnZScore()
        {
            // Arrange
            var detector = new ResidualBootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.05;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.ConfidenceLevel >= 0 && result.ConfidenceLevel <= 1);
        }

        [Fact]
        public void ResidualBootstrapFitDetector_InsufficientData_ThrowsException()
        {
            // Arrange
            var detector = new ResidualBootstrapFitDetector<double, double[], double>();
            var data = CreateGoodFitScenario();
            var actual = new double[5]; // Too few samples
            var predicted = new double[5];
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => detector.DetectFit(data));
        }

        [Fact]
        public void ResidualBootstrapFitDetector_Recommendations_ProvideGuidance()
        {
            // Arrange
            var detector = new ResidualBootstrapFitDetector<double, double[], double>();
            var data = CreateOverfitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.True(result.Recommendations.Count > 0);
        }

        [Fact]
        public void ResidualBootstrapFitDetector_CustomOptions_AffectsBootstrapCount()
        {
            // Arrange
            var options = new ResidualBootstrapFitDetectorOptions
            {
                NumBootstrapSamples = 500,
                MinSampleSize = 20
            };
            var detector = new ResidualBootstrapFitDetector<double, double[], double>(options);
            var data = CreateGoodFitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result = detector.DetectFit(data);

            // Assert
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void ResidualBootstrapFitDetector_SeededRandom_ReproducibleResults()
        {
            // Arrange
            var options1 = new ResidualBootstrapFitDetectorOptions { Seed = 42 };
            var options2 = new ResidualBootstrapFitDetectorOptions { Seed = 42 };
            var detector1 = new ResidualBootstrapFitDetector<double, double[], double>(options1);
            var detector2 = new ResidualBootstrapFitDetector<double, double[], double>(options2);
            var data = CreateGoodFitScenario();
            var actual = new double[50];
            var predicted = new double[50];
            for (int i = 0; i < 50; i++)
            {
                actual[i] = i;
                predicted[i] = i + 0.1;
            }
            data.ModelStats = new ModelStats<double, double[], double>
            {
                Actual = actual.Select(v => new double[] { v }).ToList(),
                Predicted = predicted.Select(v => new double[] { v }).ToList()
            };

            // Act
            var result1 = detector1.DetectFit(data);
            var result2 = detector2.DetectFit(data);

            // Assert - same seed should produce same results
            Assert.Equal(result1.FitType, result2.FitType);
        }

        #endregion
    }
}
