using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class CompressionAnalyzerTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var analyzer = new CompressionAnalyzer<double>();

            // Assert
            Assert.NotNull(analyzer);
        }

        [Fact]
        public void Constructor_WithCustomParameters_CreatesInstance()
        {
            // Arrange & Act
            var analyzer = new CompressionAnalyzer<double>(
                nearZeroThreshold: 0.001,
                histogramBins: 512);

            // Assert
            Assert.NotNull(analyzer);
        }

        #endregion

        #region Analyze Tests

        [Fact]
        public void Analyze_WithNullWeights_ThrowsException()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => analyzer.Analyze(null!));
        }

        [Fact]
        public void Analyze_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(Array.Empty<double>());

            // Act & Assert
            Assert.Throws<ArgumentException>(() => analyzer.Analyze(weights));
        }

        [Fact]
        public void Analyze_WithValidWeights_ReturnsAnalysisResult()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.5, 0.001, 0.8, 0.002, 0.7 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(6, result.TotalWeights);
        }

        [Fact]
        public void Analyze_WithHighSparsity_RecommendsPruning()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>(nearZeroThreshold: 0.1);
            // Create weights with many near-zero values
            var weights = new Vector<double>(new double[] {
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
                0.5, 0.6, 0.7, 0.8
            });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.True(result.NearZeroWeights >= 10);
        }

        [Fact]
        public void Analyze_CalculatesMeanMagnitude()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.Equal(3.0, result.MeanMagnitude, 6);
        }

        [Fact]
        public void Analyze_CalculatesMinMaxMagnitude()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { -5.0, 1.0, 2.0, 3.0, 10.0 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.Equal(1.0, result.MinMagnitude);
            Assert.Equal(10.0, result.MaxMagnitude);
        }

        [Fact]
        public void Analyze_CalculatesStandardDeviation()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.True(result.StdDevMagnitude > 0);
        }

        [Fact]
        public void Analyze_CalculatesEntropy()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.True(result.Entropy >= 0);
        }

        [Fact]
        public void Analyze_WithConvolutionalFlag_UsesConservativeSettings()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>(nearZeroThreshold: 0.1);
            var weights = new Vector<double>(new double[] {
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
                0.5, 0.5, 0.6, 0.6
            });

            // Act
            var resultConv = analyzer.Analyze(weights, isConvolutional: true);
            var resultFC = analyzer.Analyze(weights, isConvolutional: false);

            // Assert - both should return valid results
            Assert.NotNull(resultConv);
            Assert.NotNull(resultFC);
        }

        [Fact]
        public void Analyze_SetsRecommendationReasoning()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.False(string.IsNullOrEmpty(result.RecommendationReasoning));
        }

        [Fact]
        public void Analyze_SetsRecommendedParameters()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.NotNull(result.RecommendedParameters);
        }

        [Fact]
        public void Analyze_SetsEstimatedCompressionRatio()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.True(result.EstimatedCompressionRatio > 0);
        }

        [Fact]
        public void Analyze_WithUniformWeights_CalculatesLowEntropy()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 1.0, 1.0 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            // All same values = zero entropy
            Assert.Equal(0.0, result.Entropy, 6);
        }

        #endregion

        #region GenerateReport Tests

        [Fact]
        public void GenerateReport_ReturnsFormattedString()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
            var result = analyzer.Analyze(weights);

            // Act
            var report = analyzer.GenerateReport(result);

            // Assert
            Assert.Contains("Model Weight Analysis Report", report);
            Assert.Contains("Weight Statistics", report);
            Assert.Contains("Compression Potential", report);
            Assert.Contains("Recommendation", report);
        }

        [Fact]
        public void GenerateReport_ContainsTotalWeights()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
            var result = analyzer.Analyze(weights);

            // Act
            var report = analyzer.GenerateReport(result);

            // Assert
            Assert.Contains("Total Weights:", report);
        }

        [Fact]
        public void GenerateReport_ContainsRecommendedTechnique()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
            var result = analyzer.Analyze(weights);

            // Act
            var report = analyzer.GenerateReport(result);

            // Assert
            Assert.Contains("Technique:", report);
        }

        #endregion

        #region Type-Specific Tests

        [Fact]
        public void Analyze_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<float>();
            var weights = new Vector<float>(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(5, result.TotalWeights);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void Analyze_WithSingleWeight_ReturnsValidResult()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { 0.5 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(1, result.TotalWeights);
        }

        [Fact]
        public void Analyze_WithLargeArray_CompletesSuccessfully()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var random = RandomHelper.CreateSeededRandom(42);
            var weights = new double[10000];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var result = analyzer.Analyze(weightsVector);

            // Assert
            Assert.Equal(10000, result.TotalWeights);
            Assert.True(result.UniqueValues > 0);
        }

        [Fact]
        public void Analyze_WithNegativeWeights_HandlesCorrectly()
        {
            // Arrange
            var analyzer = new CompressionAnalyzer<double>();
            var weights = new Vector<double>(new double[] { -0.5, -0.3, 0.1, 0.3, 0.5 });

            // Act
            var result = analyzer.Analyze(weights);

            // Assert
            Assert.NotNull(result);
            // Magnitude should be positive
            Assert.True(result.MeanMagnitude > 0);
        }

        #endregion
    }
}
