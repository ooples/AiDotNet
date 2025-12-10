using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Metrics
{
    public class SimilarityMetricTests
    {
        #region Cosine Similarity Tests

        [Fact]
        public void CosineSimilarity_WithIdenticalVectors_ReturnsOne()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void CosineSimilarity_WithOrthogonalVectors_ReturnsZero()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 0.0, 1.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CosineSimilarity_WithOppositeVectors_ReturnsNegativeOne()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(-1.0, result, 10);
        }

        [Fact]
        public void CosineSimilarity_HigherIsBetter_ReturnsTrue()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.True(metric.HigherIsBetter);
        }

        [Fact]
        public void CosineSimilarity_WithScaledVectors_ReturnsSameValue()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert - cosine similarity is scale-invariant
            Assert.Equal(1.0, result, 10);
        }

        #endregion

        #region Euclidean Distance Tests

        [Fact]
        public void EuclideanDistance_WithIdenticalVectors_ReturnsZero()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void EuclideanDistance_WithDifferentVectors_ReturnsCorrectDistance()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 0.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 3.0, 4.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert - distance should be 5 (3-4-5 triangle)
            Assert.Equal(5.0, result, 10);
        }

        [Fact]
        public void EuclideanDistance_HigherIsBetter_ReturnsFalse()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();

            // Act & Assert
            Assert.False(metric.HigherIsBetter);
        }

        [Fact]
        public void EuclideanDistance_IsSymmetric()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

            // Act
            var d1 = metric.Calculate(v1, v2);
            var d2 = metric.Calculate(v2, v1);

            // Assert
            Assert.Equal(d1, d2, 10);
        }

        [Fact]
        public void EuclideanDistance_WithHighDimensionalVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var v2 = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        #endregion

        #region Manhattan Distance Tests

        [Fact]
        public void ManhattanDistance_WithIdenticalVectors_ReturnsZero()
        {
            // Arrange
            var metric = new ManhattanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void ManhattanDistance_WithDifferentVectors_ReturnsCorrectDistance()
        {
            // Arrange
            var metric = new ManhattanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 0.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 3.0, 4.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert - Manhattan distance is 3 + 4 = 7
            Assert.Equal(7.0, result, 10);
        }

        [Fact]
        public void ManhattanDistance_HigherIsBetter_ReturnsFalse()
        {
            // Arrange
            var metric = new ManhattanDistanceMetric<double>();

            // Act & Assert
            Assert.False(metric.HigherIsBetter);
        }

        [Fact]
        public void ManhattanDistance_IsSymmetric()
        {
            // Arrange
            var metric = new ManhattanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

            // Act
            var d1 = metric.Calculate(v1, v2);
            var d2 = metric.Calculate(v2, v1);

            // Assert
            Assert.Equal(d1, d2, 10);
        }

        [Fact]
        public void ManhattanDistance_WithNegativeValues_WorksCorrectly()
        {
            // Arrange
            var metric = new ManhattanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert - |(-1-1)| + |(-2-2)| + |(-3-3)| = 2 + 4 + 6 = 12
            Assert.Equal(12.0, result, 10);
        }

        #endregion

        #region Dot Product Tests

        [Fact]
        public void DotProduct_WithIdenticalUnitVectors_ReturnsOne()
        {
            // Arrange
            var metric = new DotProductMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void DotProduct_WithOrthogonalVectors_ReturnsZero()
        {
            // Arrange
            var metric = new DotProductMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 0.0, 1.0, 0.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void DotProduct_HigherIsBetter_ReturnsTrue()
        {
            // Arrange
            var metric = new DotProductMetric<double>();

            // Act & Assert
            Assert.True(metric.HigherIsBetter);
        }

        [Fact]
        public void DotProduct_WithRegularVectors_ReturnsCorrectValue()
        {
            // Arrange
            var metric = new DotProductMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert - 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            Assert.Equal(32.0, result, 10);
        }

        [Fact]
        public void DotProduct_IsSymmetric()
        {
            // Arrange
            var metric = new DotProductMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

            // Act
            var d1 = metric.Calculate(v1, v2);
            var d2 = metric.Calculate(v2, v1);

            // Assert
            Assert.Equal(d1, d2, 10);
        }

        #endregion

        #region Jaccard Similarity Tests

        [Fact]
        public void JaccardSimilarity_WithIdenticalVectors_ReturnsOne()
        {
            // Arrange
            var metric = new JaccardSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void JaccardSimilarity_WithDisjointVectors_ReturnsZero()
        {
            // Arrange
            var metric = new JaccardSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 0.0, 1.0, 1.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void JaccardSimilarity_HigherIsBetter_ReturnsTrue()
        {
            // Arrange
            var metric = new JaccardSimilarityMetric<double>();

            // Act & Assert
            Assert.True(metric.HigherIsBetter);
        }

        [Fact]
        public void JaccardSimilarity_WithPartialOverlap_ReturnsCorrectValue()
        {
            // Arrange
            var metric = new JaccardSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 1.0, 0.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 0.0, 1.0 });

            // Act
            var result = metric.Calculate(v1, v2);

            // Assert
            // Intersection: min(1,1) + min(1,0) + min(0,1) = 1 + 0 + 0 = 1
            // Union: max(1,1) + max(1,0) + max(0,1) = 1 + 1 + 1 = 3
            // Jaccard = 1/3 = 0.333...
            Assert.Equal(0.333333333, result, 5);
        }

        [Fact]
        public void JaccardSimilarity_IsSymmetric()
        {
            // Arrange
            var metric = new JaccardSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 2.0, 3.0, 4.0 });

            // Act
            var d1 = metric.Calculate(v1, v2);
            var d2 = metric.Calculate(v2, v1);

            // Assert
            Assert.Equal(d1, d2, 10);
        }

        #endregion

        #region Edge Cases and Numerical Stability

        [Fact]
        public void Metrics_WithSingleElementVectors_WorkCorrectly()
        {
            // Arrange
            var cosine = new CosineSimilarityMetric<double>();
            var euclidean = new EuclideanDistanceMetric<double>();
            var v1 = new Vector<double>(new double[] { 5.0 });
            var v2 = new Vector<double>(new double[] { 5.0 });

            // Act & Assert
            Assert.Equal(1.0, cosine.Calculate(v1, v2), 10);
            Assert.Equal(0.0, euclidean.Calculate(v1, v2), 10);
        }

        [Fact]
        public void Metrics_WithFloatType_WorkCorrectly()
        {
            // Arrange
            var cosine = new CosineSimilarityMetric<float>();
            var euclidean = new EuclideanDistanceMetric<float>();
            var v1 = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });
            var v2 = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act & Assert
            Assert.Equal(1.0f, cosine.Calculate(v1, v2), 5);
            Assert.Equal(0.0f, euclidean.Calculate(v1, v2), 5);
        }

        [Fact]
        public void Metrics_WithVerySmallValues_MaintainNumericalStability()
        {
            // Arrange
            var cosine = new CosineSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1e-10, 2e-10, 3e-10 });
            var v2 = new Vector<double>(new double[] { 1e-10, 2e-10, 3e-10 });

            // Act
            var result = cosine.Calculate(v1, v2);

            // Assert - should still be 1.0 for identical vectors
            Assert.Equal(1.0, result, 8);
        }

        [Fact]
        public void Metrics_WithLargeValues_MaintainNumericalStability()
        {
            // Arrange
            var cosine = new CosineSimilarityMetric<double>();
            var v1 = new Vector<double>(new double[] { 1e10, 2e10, 3e10 });
            var v2 = new Vector<double>(new double[] { 1e10, 2e10, 3e10 });

            // Act
            var result = cosine.Calculate(v1, v2);

            // Assert
            Assert.Equal(1.0, result, 8);
        }

        #endregion
    }
}
