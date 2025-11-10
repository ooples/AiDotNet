using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.LinearAlgebra
{
    /// <summary>
    /// Integration tests for Vector operations with mathematically verified results.
    /// </summary>
    public class VectorIntegrationTests
    {
        [Fact]
        public void VectorDotProduct_ProducesCorrectResult()
        {
            // Arrange
            // v1 = [1, 2, 3], v2 = [4, 5, 6]
            // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 2.0; v1[2] = 3.0;

            var v2 = new Vector<double>(3);
            v2[0] = 4.0; v2[1] = 5.0; v2[2] = 6.0;

            // Act
            var dotProduct = v1.DotProduct(v2);

            // Assert
            Assert.Equal(32.0, dotProduct, precision: 10);
        }

        [Fact]
        public void VectorMagnitude_ProducesCorrectResult()
        {
            // Arrange
            // v = [3, 4]
            // ||v|| = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
            var v = new Vector<double>(2);
            v[0] = 3.0; v[1] = 4.0;

            // Act
            var magnitude = v.Magnitude();

            // Assert
            Assert.Equal(5.0, magnitude, precision: 10);
        }

        [Fact]
        public void VectorNormalize_ProducesUnitVector()
        {
            // Arrange
            var v = new Vector<double>(3);
            v[0] = 3.0; v[1] = 4.0; v[2] = 0.0;

            // Act
            var normalized = v.Normalize();

            // Assert - Magnitude should be 1
            var magnitude = normalized.Magnitude();
            Assert.Equal(1.0, magnitude, precision: 10);

            // Check values: original magnitude was 5, so normalized = [3/5, 4/5, 0]
            Assert.Equal(0.6, normalized[0], precision: 10);
            Assert.Equal(0.8, normalized[1], precision: 10);
            Assert.Equal(0.0, normalized[2], precision: 10);
        }

        [Fact]
        public void VectorAddition_ProducesCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 2.0; v1[2] = 3.0;

            var v2 = new Vector<double>(3);
            v2[0] = 4.0; v2[1] = 5.0; v2[2] = 6.0;

            // Act
            var result = v1 + v2;

            // Assert
            Assert.Equal(5.0, result[0], precision: 10);
            Assert.Equal(7.0, result[1], precision: 10);
            Assert.Equal(9.0, result[2], precision: 10);
        }

        [Fact]
        public void VectorSubtraction_ProducesCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 10.0; v1[1] = 20.0; v1[2] = 30.0;

            var v2 = new Vector<double>(3);
            v2[0] = 1.0; v2[1] = 2.0; v2[2] = 3.0;

            // Act
            var result = v1 - v2;

            // Assert
            Assert.Equal(9.0, result[0], precision: 10);
            Assert.Equal(18.0, result[1], precision: 10);
            Assert.Equal(27.0, result[2], precision: 10);
        }

        [Fact]
        public void VectorScalarMultiplication_ProducesCorrectResult()
        {
            // Arrange
            var v = new Vector<double>(3);
            v[0] = 1.0; v[1] = 2.0; v[2] = 3.0;
            double scalar = 2.5;

            // Act
            var result = v * scalar;

            // Assert
            Assert.Equal(2.5, result[0], precision: 10);
            Assert.Equal(5.0, result[1], precision: 10);
            Assert.Equal(7.5, result[2], precision: 10);
        }

        [Fact]
        public void VectorCrossProduct_3D_ProducesCorrectResult()
        {
            // Arrange
            // v1 = [1, 0, 0], v2 = [0, 1, 0]
            // v1 × v2 = [0, 0, 1] (right-hand rule)
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 0.0; v1[2] = 0.0;

            var v2 = new Vector<double>(3);
            v2[0] = 0.0; v2[1] = 1.0; v2[2] = 0.0;

            // Act
            var cross = v1.CrossProduct(v2);

            // Assert
            Assert.Equal(0.0, cross[0], precision: 10);
            Assert.Equal(0.0, cross[1], precision: 10);
            Assert.Equal(1.0, cross[2], precision: 10);
        }

        [Fact]
        public void VectorCrossProduct_GeneralCase_ProducesCorrectResult()
        {
            // Arrange
            // v1 = [2, 3, 4], v2 = [5, 6, 7]
            // v1 × v2 = [3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5] = [21-24, 20-14, 12-15] = [-3, 6, -3]
            var v1 = new Vector<double>(3);
            v1[0] = 2.0; v1[1] = 3.0; v1[2] = 4.0;

            var v2 = new Vector<double>(3);
            v2[0] = 5.0; v2[1] = 6.0; v2[2] = 7.0;

            // Act
            var cross = v1.CrossProduct(v2);

            // Assert
            Assert.Equal(-3.0, cross[0], precision: 10);
            Assert.Equal(6.0, cross[1], precision: 10);
            Assert.Equal(-3.0, cross[2], precision: 10);
        }

        [Fact]
        public void VectorDistance_Euclidean_ProducesCorrectResult()
        {
            // Arrange
            // v1 = [0, 0], v2 = [3, 4]
            // distance = sqrt((3-0)^2 + (4-0)^2) = sqrt(9 + 16) = 5
            var v1 = new Vector<double>(2);
            v1[0] = 0.0; v1[1] = 0.0;

            var v2 = new Vector<double>(2);
            v2[0] = 3.0; v2[1] = 4.0;

            // Act
            var distance = v1.EuclideanDistance(v2);

            // Assert
            Assert.Equal(5.0, distance, precision: 10);
        }

        [Fact]
        public void VectorCosineSimilarity_ProducesCorrectResult()
        {
            // Arrange
            // v1 = [1, 0], v2 = [1, 0] (same direction)
            // cos(θ) = 1
            var v1 = new Vector<double>(2);
            v1[0] = 1.0; v1[1] = 0.0;

            var v2 = new Vector<double>(2);
            v2[0] = 1.0; v2[1] = 0.0;

            // Act
            var similarity = v1.CosineSimilarity(v2);

            // Assert
            Assert.Equal(1.0, similarity, precision: 10);
        }

        [Fact]
        public void VectorCosineSimilarity_OrthogonalVectors_ReturnsZero()
        {
            // Arrange
            // v1 = [1, 0], v2 = [0, 1] (perpendicular)
            // cos(90°) = 0
            var v1 = new Vector<double>(2);
            v1[0] = 1.0; v1[1] = 0.0;

            var v2 = new Vector<double>(2);
            v2[0] = 0.0; v2[1] = 1.0;

            // Act
            var similarity = v1.CosineSimilarity(v2);

            // Assert
            Assert.Equal(0.0, similarity, precision: 10);
        }

        [Fact]
        public void VectorElementWiseMultiplication_ProducesCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 2.0; v1[1] = 3.0; v1[2] = 4.0;

            var v2 = new Vector<double>(3);
            v2[0] = 5.0; v2[1] = 6.0; v2[2] = 7.0;

            // Act
            var result = v1.ElementWiseMultiply(v2);

            // Assert
            Assert.Equal(10.0, result[0], precision: 10);
            Assert.Equal(18.0, result[1], precision: 10);
            Assert.Equal(28.0, result[2], precision: 10);
        }

        [Fact]
        public void VectorSum_ProducesCorrectResult()
        {
            // Arrange
            var v = new Vector<double>(5);
            v[0] = 1.0; v[1] = 2.0; v[2] = 3.0; v[3] = 4.0; v[4] = 5.0;

            // Act
            var sum = v.Sum();

            // Assert
            Assert.Equal(15.0, sum, precision: 10);
        }

        [Fact]
        public void VectorMean_ProducesCorrectResult()
        {
            // Arrange
            var v = new Vector<double>(5);
            v[0] = 2.0; v[1] = 4.0; v[2] = 6.0; v[3] = 8.0; v[4] = 10.0;

            // Act
            var mean = v.Mean();

            // Assert
            Assert.Equal(6.0, mean, precision: 10);
        }

        [Fact]
        public void Vector_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var v1 = new Vector<float>(3);
            v1[0] = 1.0f; v1[1] = 2.0f; v1[2] = 3.0f;

            var v2 = new Vector<float>(3);
            v2[0] = 4.0f; v2[1] = 5.0f; v2[2] = 6.0f;

            // Act
            var dotProduct = v1.DotProduct(v2);

            // Assert
            Assert.Equal(32.0f, dotProduct, precision: 6);
        }
    }
}
