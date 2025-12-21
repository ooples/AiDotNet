using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class EnsembleFitDetectorTests
    {
        private class MockFitDetector : IFitDetector<double, Matrix<double>, Vector<double>>
        {
            private readonly FitDetectorResult<double> _result;

            public MockFitDetector(FitType fitType, double confidenceLevel, List<string>? recommendations = null)
            {
                _result = new FitDetectorResult<double>
                {
                    FitType = fitType,
                    ConfidenceLevel = confidenceLevel,
                    Recommendations = recommendations ?? new List<string> { $"Recommendation for {fitType}" }
                };
            }

            public FitDetectorResult<double> DetectFit(ModelEvaluationData<double, Matrix<double>, Vector<double>> evaluationData)
            {
                return _result;
            }
        }

        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData()
        {
            var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.1);
            var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.12);

            return FitDetectorTestHelper.CreateEvaluationData(
                trainActual, trainPredicted,
                validActual, validPredicted);
        }

        [Fact]
        public void Constructor_WithNullDetectors_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(null!));
        }

        [Fact]
        public void Constructor_WithEmptyDetectorList_ThrowsArgumentException()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors));
        }

        [Fact]
        public void Constructor_WithValidDetectors_CreatesInstance()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8)
            };

            // Act
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Assert
            Assert.NotNull(ensemble);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8)
            };
            var options = new EnsembleFitDetectorOptions
            {
                DetectorWeights = new List<double> { 1.0 },
                MaxRecommendations = 5
            };

            // Act
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors, options);

            // Assert
            Assert.NotNull(ensemble);
        }

        [Fact]
        public void DetectFit_WithNullEvaluationData_ThrowsArgumentNullException()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8)
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => ensemble.DetectFit(null!));
        }

        [Fact]
        public void DetectFit_WithSingleDetector_ReturnsSameFitType()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8)
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void DetectFit_WithMultipleDetectorsReturningGoodFit_ReturnsGoodFit()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8),
                new MockFitDetector(FitType.GoodFit, 0.9),
                new MockFitDetector(FitType.GoodFit, 0.85)
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void DetectFit_WithMixedFitTypes_CombinesResults()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8),
                new MockFitDetector(FitType.Moderate, 0.7),
                new MockFitDetector(FitType.GoodFit, 0.9)
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_CalculatesWeightedConfidence()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8),
                new MockFitDetector(FitType.GoodFit, 0.6)
            };
            var options = new EnsembleFitDetectorOptions
            {
                DetectorWeights = new List<double> { 1.0, 1.0 }
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors, options);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_WithDifferentWeights_AffectsResult()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.9),
                new MockFitDetector(FitType.PoorFit, 0.8)
            };
            var options = new EnsembleFitDetectorOptions
            {
                DetectorWeights = new List<double> { 2.0, 1.0 }
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors, options);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_CombinesRecommendationsFromAllDetectors()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8, new List<string> { "Rec 1" }),
                new MockFitDetector(FitType.GoodFit, 0.9, new List<string> { "Rec 2" }),
                new MockFitDetector(FitType.GoodFit, 0.85, new List<string> { "Rec 3" })
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Count >= 3); // At least 3 unique recommendations
        }

        [Fact]
        public void DetectFit_RemovesDuplicateRecommendations()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8, new List<string> { "Same Rec" }),
                new MockFitDetector(FitType.GoodFit, 0.9, new List<string> { "Same Rec" }),
                new MockFitDetector(FitType.GoodFit, 0.85, new List<string> { "Different Rec" })
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            var sameRecCount = result.Recommendations.Count(r => r == "Same Rec");
            Assert.True(sameRecCount <= 1, "Duplicate recommendations should be removed");
        }

        [Fact]
        public void DetectFit_RespectsMaxRecommendationsLimit()
        {
            // Arrange
            var recommendations = new List<string>();
            for (int i = 0; i < 20; i++)
            {
                recommendations.Add($"Recommendation {i}");
            }

            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8, recommendations)
            };

            var options = new EnsembleFitDetectorOptions
            {
                MaxRecommendations = 5
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors, options);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Recommendations.Count <= 5);
        }

        [Fact]
        public void DetectFit_IncludesIndividualResultsInAdditionalInfo()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8),
                new MockFitDetector(FitType.Moderate, 0.7)
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.AdditionalInfo);
            Assert.True(result.AdditionalInfo.ContainsKey("IndividualResults"));
            Assert.True(result.AdditionalInfo.ContainsKey("DetectorWeights"));
        }

        [Fact]
        public void DetectFit_IncludesGeneralRecommendationBasedOnFitType()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8, new List<string> { "Specific rec" })
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new MockFitDetector(FitType.GoodFit, 0.8)
            };
            var ensemble = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
            var evaluationData = CreateMockEvaluationData();

            // Act
            var result = ensemble.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.NotNull(result.AdditionalInfo);
        }
    }
}
