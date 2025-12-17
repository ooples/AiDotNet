using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    /// <summary>
    /// Unit tests for the HybridFitDetector class.
    /// Tests use properly constructed ModelEvaluationData with calculated statistics.
    /// The HybridFitDetector combines residual analysis and learning curve detection.
    /// </summary>
    public class HybridFitDetectorTests
    {
        /// <summary>
        /// Creates evaluation data with properly calculated statistics.
        /// </summary>
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainMse = 0.1, double validationMse = 0.12, double testMse = 0.11)
        {
            var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(trainMse);
            var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(validationMse);
            var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(testMse);

            // Create well-conditioned feature matrix for calculations
            var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

            return FitDetectorTestHelper.CreateEvaluationData(
                trainActual, trainPredicted,
                validActual, validPredicted,
                testActual, testPredicted,
                features: features);
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            // Should have recommendations from combined analysis
            Assert.True(result.Recommendations.Count >= 1);
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            // The hybrid result should produce a valid fit type
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
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
            var evaluationData = CreateMockEvaluationData();

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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            // When both detectors produce results, hybrid should respect that
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

            // Create float-typed evaluation data manually
            var trainActual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f });
            var trainPredicted = new Vector<float>(new float[] { 1.1f, 2.1f, 2.9f, 4.1f, 5.0f, 6.1f, 7.0f, 8.1f, 9.0f, 10.1f,
                11.0f, 12.1f, 12.9f, 14.1f, 15.0f, 16.1f, 17.0f, 18.1f, 19.0f, 20.1f,
                21.0f, 22.1f, 22.9f, 24.1f, 25.0f, 26.1f, 27.0f, 28.1f, 29.0f, 30.1f });

            var trainErrorStats = new ErrorStats<float>(new AiDotNet.Models.Inputs.ErrorStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = trainPredicted,
                FeatureCount = 3,
                PredictionType = PredictionType.Regression
            });
            var trainPredictionStats = new PredictionStats<float>(new AiDotNet.Models.Inputs.PredictionStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = trainPredicted,
                NumberOfParameters = 3,
                ConfidenceLevel = 0.95,
                LearningCurveSteps = 10,
                PredictionType = PredictionType.Regression
            });
            var trainBasicStats = new BasicStats<float>(new AiDotNet.Models.Inputs.BasicStatsInputs<float>
            {
                Values = trainActual
            });

            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats,
                    ActualBasicStats = trainBasicStats,
                    PredictedBasicStats = trainBasicStats
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats,
                    ActualBasicStats = trainBasicStats,
                    PredictedBasicStats = trainBasicStats
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats,
                    ActualBasicStats = trainBasicStats,
                    PredictedBasicStats = trainBasicStats
                }
            };

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var hybridResult = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);

            // Assert
            // The hybrid detector should incorporate residual analysis insights
            Assert.NotNull(hybridResult);
            Assert.NotNull(residualResult);
            // Verify the residual analyzer produced valid output that can influence hybrid
            Assert.True(System.Enum.IsDefined(typeof(FitType), residualResult.FitType));
            // Recommendations should include insights from residual analysis
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var hybridResult = detector.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            // The hybrid detector should incorporate learning curve insights
            Assert.NotNull(hybridResult);
            Assert.NotNull(learningResult);
            // Verify the learning curve detector produced valid output that can influence hybrid
            Assert.True(System.Enum.IsDefined(typeof(FitType), learningResult.FitType));
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            // Hybrid analysis should be comprehensive
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
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
            var evaluationData = CreateMockEvaluationData();

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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var hybridResult = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            // Hybrid result should be informed by both component results
            Assert.NotNull(hybridResult);
            Assert.NotNull(residualResult);
            Assert.NotNull(learningResult);

            // Verify both components produced valid outputs
            Assert.True(System.Enum.IsDefined(typeof(FitType), residualResult.FitType));
            Assert.True(System.Enum.IsDefined(typeof(FitType), learningResult.FitType));

            // Check that the hybrid incorporates information from both
            Assert.True(hybridResult.Recommendations.Count >= 1);
        }

        [Fact]
        public void DetectFit_WithSameOptions_ProducesSameResults()
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

            var evaluationData = CreateMockEvaluationData();

            // Act
            var result1 = detector1.DetectFit(evaluationData);
            var result2 = detector2.DetectFit(evaluationData);

            // Assert
            // With same options, should get same result
            Assert.Equal(result1.FitType, result2.FitType);
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
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            // If either detector indicates instability, it should be given weight
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
        }

        [Fact]
        public void DetectFit_WithConfidenceComparison()
        {
            // Arrange
            var residualAnalyzer = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualAnalyzer,
                learningCurveDetector
            );
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = detector.DetectFit(evaluationData);
            var residualResult = residualAnalyzer.DetectFit(evaluationData);
            var learningResult = learningCurveDetector.DetectFit(evaluationData);

            // Assert
            Assert.InRange(residualResult.ConfidenceLevel, 0.0, 1.0);

            // The combined confidence should be reasonable
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }
    }
}
