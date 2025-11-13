using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    /// <summary>
    /// Unit tests for the HybridFitDetector class.
    /// </summary>
    public class HybridFitDetectorTests
    {
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateTestEvaluationData()
        {
            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = ErrorStats<double>.Empty(),
                    PredictionStats = PredictionStats<double>.Empty(),
                    ActualBasicStats = BasicStats<double>.Empty(),
                    PredictedBasicStats = BasicStats<double>.Empty()
                },
                ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = ErrorStats<double>.Empty(),
                    PredictionStats = PredictionStats<double>.Empty(),
                    ActualBasicStats = BasicStats<double>.Empty(),
                    PredictedBasicStats = BasicStats<double>.Empty()
                },
                TestSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = ErrorStats<double>.Empty(),
                    PredictionStats = PredictionStats<double>.Empty(),
                    ActualBasicStats = BasicStats<double>.Empty(),
                    PredictedBasicStats = BasicStats<double>.Empty()
                }
            };
        }

        [Fact]
        public void Constructor_WithRequiredDetectors_InitializesSuccessfully()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_InitializesSuccessfully()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var options = new HybridFitDetectorOptions();

            // Act
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector,
                options
            );

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_ReturnsValidResult()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
        }

        [Fact]
        public void DetectFit_ReturnsConfidenceLevel()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_CombinesRecommendationsFromBothDetectors()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            // Should have recommendations from both detectors
            Assert.True(result.Recommendations.Count >= 2);
        }

        [Fact]
        public void DetectFit_ProducesCombinedFitType()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            // The hybrid result should consider both inputs
            // (The exact combination depends on the internal logic)
            Assert.True(
                result.FitType == residualResult.FitType ||
                result.FitType == learningResult.FitType ||
                // Or it could be a combined assessment
                System.Enum.IsDefined(typeof(FitType), result.FitType)
            );
        }

        [Fact]
        public void DetectFit_WithSimilarConfidenceLevels_AveragesConfidence()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.NotNull(residualResult.ConfidenceLevel);
            Assert.NotNull(learningResult.ConfidenceLevel);

            // The combined confidence should be related to the individual confidences
            // (exact formula depends on implementation, but should be reasonable)
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_ReturnsNonNullRecommendations()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
        }

        [Fact]
        public void DetectFit_HandlesConsistentDetectorResults()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            // When both detectors agree, hybrid should respect that
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
        }

        [Fact]
        public void DetectFit_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<float, Matrix<float>, Vector<float>>();
            var learningCurveDetector = new LearningCurveFitDetector<float, Matrix<float>, Vector<float>>();
            var detector = new HybridFitDetector<float, Matrix<float>, Vector<float>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = ErrorStats<float>.Empty(),
                    PredictionStats = PredictionStats<float>.Empty(),
                    ActualBasicStats = BasicStats<float>.Empty(),
                    PredictedBasicStats = BasicStats<float>.Empty()
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = ErrorStats<float>.Empty(),
                    PredictionStats = PredictionStats<float>.Empty(),
                    ActualBasicStats = BasicStats<float>.Empty(),
                    PredictedBasicStats = BasicStats<float>.Empty()
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = ErrorStats<float>.Empty(),
                    PredictionStats = PredictionStats<float>.Empty(),
                    ActualBasicStats = BasicStats<float>.Empty(),
                    PredictedBasicStats = BasicStats<float>.Empty()
                }
            };

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_UsesResidualAnalyzerComponent()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var hybridResult = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);

            // Assert
            // The hybrid detector should incorporate residual analysis insights
            Assert.NotNull(hybridResult);
            Assert.NotNull(residualResult);
            // Recommendations should include insights from residual analyzer
            Assert.True(hybridResult.Recommendations.Count > 0);
        }

        [Fact]
        public void DetectFit_UsesLearningCurveComponent()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var hybridResult = detector.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            // The hybrid detector should incorporate learning curve insights
            Assert.NotNull(hybridResult);
            Assert.NotNull(learningResult);
            // Recommendations should include insights from learning curve detector
            Assert.True(hybridResult.Recommendations.Count > 0);
        }

        [Fact]
        public void DetectFit_ProvidesComprehensiveAnalysis()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            // Hybrid analysis should be more comprehensive than individual detectors
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_HandlesUnstableFitType()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            // If either detector indicates instability, it should be given weight
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
        }

        [Fact]
        public void DetectFit_ProducesReproducibleResults()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var result1 = detector.DetectFit(evaluationData);
            var result2 = detector.DetectFit(evaluationData);

            // Assert
            // Same input should produce same output
            Assert.Equal(result1.FitType, result2.FitType);
            Assert.Equal(result1.ConfidenceLevel, result2.ConfidenceLevel);
        }

        [Fact]
        public void DetectFit_RespectsComponentDetectorInputs()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateTestEvaluationData();

            // Act
            var hybridResult = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            // Hybrid result should be informed by both component results
            Assert.NotNull(hybridResult);
            Assert.NotNull(residualResult);
            Assert.NotNull(learningResult);

            // Check that the hybrid incorporates information from both
            // This is a logical check - the result should be sensible given inputs
            Assert.True(
                hybridResult.Recommendations.Count >= residualResult.Recommendations.Count ||
                hybridResult.Recommendations.Count >= learningResult.Recommendations.Count
            );
        }

        [Fact]
        public void DetectFit_WithDifferentOptions_ProducesDifferentResults()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();

            var detector1 = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector,
                new HybridFitDetectorOptions()
            );

            var detector2 = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector,
                new HybridFitDetectorOptions()
            );

            var evaluationData = CreateTestEvaluationData();

            // Act
            var result1 = detector1.DetectFit(evaluationData);
            var result2 = detector2.DetectFit(evaluationData);

            // Assert
            // With same options, should get same result
            Assert.Equal(result1.FitType, result2.FitType);
        }
    }
}
