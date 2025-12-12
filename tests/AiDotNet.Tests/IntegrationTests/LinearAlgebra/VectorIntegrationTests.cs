using AiDotNet.LinearAlgebra;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.LinearAlgebra
{
    /// <summary>
    /// Integration tests for Vector operations with mathematically verified results.
    /// </summary>
    public class VectorIntegrationTests
    {
        #region Basic Operations Tests (Existing)

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

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithLength_CreatesVectorWithCorrectSize()
        {
            // Arrange & Act
            var v = new Vector<double>(5);

            // Assert
            Assert.Equal(5, v.Length);
            Assert.Equal(0.0, v[0]);
        }

        [Fact]
        public void Constructor_WithEnumerable_CreatesVectorWithValues()
        {
            // Arrange
            var values = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

            // Act
            var v = new Vector<double>(values);

            // Assert
            Assert.Equal(5, v.Length);
            Assert.Equal(1.0, v[0]);
            Assert.Equal(3.0, v[2]);
            Assert.Equal(5.0, v[4]);
        }

        [Fact]
        public void Constructor_WithInvalidLength_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Vector<double>(0));
            Assert.Throws<ArgumentException>(() => new Vector<double>(-1));
        }

        #endregion

        #region Indexer Tests

        [Fact]
        public void Indexer_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(3);

            // Act
            v[0] = 1.5;
            v[1] = 2.5;
            v[2] = 3.5;

            // Assert
            Assert.Equal(1.5, v[0]);
            Assert.Equal(2.5, v[1]);
            Assert.Equal(3.5, v[2]);
        }

        [Fact]
        public void Indexer_OutOfBounds_ThrowsException()
        {
            // Arrange
            var v = new Vector<double>(3);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => v[-1]);
            Assert.Throws<ArgumentOutOfRangeException>(() => v[3]);
            Assert.Throws<ArgumentOutOfRangeException>(() => v[-1] = 1.0);
            Assert.Throws<ArgumentOutOfRangeException>(() => v[3] = 1.0);
        }

        #endregion

        #region LINQ-Style Operations

        [Fact]
        public void Where_FiltersElementsCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act - keep only elements > 2
            var result = v.Where(x => x > 2.0);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(3.0, result[0]);
            Assert.Equal(4.0, result[1]);
            Assert.Equal(5.0, result[2]);
        }

        [Fact]
        public void Select_TransformsElementsCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act - square each element
            var result = v.Select(x => x * x);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(4.0, result[1]);
            Assert.Equal(9.0, result[2]);
        }

        [Fact]
        public void Take_ReturnsFirstNElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = new Vector<double>(v.Take(3));

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
        }

        [Fact]
        public void Skip_SkipsFirstNElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = new Vector<double>(v.Skip(2));

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(3.0, result[0]);
            Assert.Equal(4.0, result[1]);
            Assert.Equal(5.0, result[2]);
        }

        #endregion

        #region Range and Subvector Operations

        [Fact]
        public void GetSubVector_ExtractsCorrectElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var sub = v.GetSubVector(1, 3);

            // Assert
            Assert.Equal(3, sub.Length);
            Assert.Equal(2.0, sub[0]);
            Assert.Equal(3.0, sub[1]);
            Assert.Equal(4.0, sub[2]);
        }

        [Fact]
        public void Subvector_ExtractsCorrectElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var sub = v.Subvector(2, 2);

            // Assert
            Assert.Equal(2, sub.Length);
            Assert.Equal(3.0, sub[0]);
            Assert.Equal(4.0, sub[1]);
        }

        [Fact]
        public void GetRange_ExtractsCorrectElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var range = v.GetRange(1, 3);

            // Assert
            Assert.Equal(3, range.Length);
            Assert.Equal(2.0, range[0]);
            Assert.Equal(3.0, range[1]);
            Assert.Equal(4.0, range[2]);
        }

        [Fact]
        public void GetSegment_ExtractsCorrectElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var segment = v.GetSegment(1, 3);

            // Assert
            Assert.Equal(3, segment.Length);
            Assert.Equal(2.0, segment[0]);
            Assert.Equal(3.0, segment[1]);
            Assert.Equal(4.0, segment[2]);
        }

        [Fact]
        public void Slice_ExtractsCorrectElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var slice = v.Slice(1, 3);

            // Assert
            Assert.Equal(3, slice.Length);
            Assert.Equal(2.0, slice[0]);
            Assert.Equal(3.0, slice[1]);
            Assert.Equal(4.0, slice[2]);
        }

        #endregion

        #region Statistical Operations

        [Fact]
        public void Min_ReturnsSmallestValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 1.0, 4.0, 1.5, 9.0 });

            // Act
            var min = v.Min();

            // Assert
            Assert.Equal(1.0, min);
        }

        [Fact]
        public void Max_ReturnsLargestValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 1.0, 4.0, 1.5, 9.0 });

            // Act
            var max = v.Max();

            // Assert
            Assert.Equal(9.0, max);
        }

        [Fact]
        public void Variance_CalculatesCorrectly()
        {
            // Arrange
            // v = [2, 4, 6, 8]
            // mean = 5
            // variance = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = (9 + 1 + 1 + 9) / 4 = 5
            var v = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0 });

            // Act
            var variance = v.Variance();

            // Assert
            Assert.Equal(5.0, variance, precision: 10);
        }

        [Fact]
        public void StandardDeviation_CalculatesCorrectly()
        {
            // Arrange
            // v = [2, 4, 6, 8]
            // Using sample std dev (n-1)
            // mean = 5, variance = 20/3, std = sqrt(20/3) ≈ 2.58199
            var v = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0 });

            // Act
            var stdDev = v.StandardDeviation();

            // Assert
            Assert.Equal(2.58199, stdDev, precision: 5);
        }

        [Fact]
        public void Average_CalculatesCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0, 10.0 });

            // Act
            var avg = v.Average();

            // Assert
            Assert.Equal(6.0, avg, precision: 10);
        }

        [Fact]
        public void Median_OddLength_ReturnsMiddleValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 1.0, 5.0, 2.0, 4.0 });

            // Act
            var median = v.Median();

            // Assert - sorted: [1, 2, 3, 4, 5], median = 3
            Assert.Equal(3.0, median, precision: 10);
        }

        [Fact]
        public void Median_EvenLength_ReturnsAverageOfMiddleTwo()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var median = v.Median();

            // Assert - median = (2 + 3) / 2 = 2.5
            Assert.Equal(2.5, median, precision: 10);
        }

        #endregion

        #region Distance Metrics

        [Fact]
        public void ManhattanDistance_CalculatesCorrectly()
        {
            // Arrange
            // v1 = [1, 2, 3], v2 = [4, 6, 5]
            // Manhattan distance = |1-4| + |2-6| + |3-5| = 3 + 4 + 2 = 9
            var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new[] { 4.0, 6.0, 5.0 });

            // Act
            var distance = StatisticsHelper.ManhattanDistance(v1, v2);

            // Assert
            Assert.Equal(9.0, distance, precision: 10);
        }

        [Fact]
        public void HammingDistance_CalculatesCorrectly()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var v2 = new Vector<double>(new[] { 1.0, 3.0, 3.0, 5.0 });

            // Act - positions where values differ: indices 1 and 3
            var distance = StatisticsHelper.HammingDistance(v1, v2);

            // Assert
            Assert.Equal(2.0, distance, precision: 10);
        }

        #endregion

        #region Norm Operations

        [Fact]
        public void Norm_CalculatesL2Norm()
        {
            // Arrange
            // v = [3, 4]
            // L2 norm = sqrt(3^2 + 4^2) = 5
            var v = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var norm = v.Norm();

            // Assert
            Assert.Equal(5.0, norm, precision: 10);
        }

        [Fact]
        public void L2Norm_CalculatesCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 2.0 });

            // Act
            // L2 norm = sqrt(1 + 4 + 4) = 3
            var norm = v.L2Norm();

            // Assert
            Assert.Equal(3.0, norm, precision: 10);
        }

        #endregion

        #region Element-wise Operations

        [Fact]
        public void ElementwiseDivide_ProducesCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
            var v2 = new Vector<double>(new[] { 2.0, 4.0, 5.0 });

            // Act
            var result = v1.ElementwiseDivide(v2);

            // Assert
            Assert.Equal(5.0, result[0], precision: 10);
            Assert.Equal(5.0, result[1], precision: 10);
            Assert.Equal(6.0, result[2], precision: 10);
        }

        [Fact]
        public void PointwiseDivide_ProducesCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 12.0, 18.0, 24.0 });
            var v2 = new Vector<double>(new[] { 3.0, 6.0, 8.0 });

            // Act
            var result = v1.PointwiseDivide(v2);

            // Assert
            Assert.Equal(4.0, result[0], precision: 10);
            Assert.Equal(3.0, result[1], precision: 10);
            Assert.Equal(3.0, result[2], precision: 10);
        }

        [Fact]
        public void PointwiseExp_AppliesExponentialCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

            // Act
            var result = v.PointwiseExp();

            // Assert
            Assert.Equal(1.0, result[0], precision: 10);
            Assert.Equal(Math.E, result[1], precision: 10);
            Assert.Equal(Math.E * Math.E, result[2], precision: 10);
        }

        [Fact]
        public void PointwiseLog_AppliesNaturalLogCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, Math.E, Math.E * Math.E });

            // Act
            var result = v.PointwiseLog();

            // Assert
            Assert.Equal(0.0, result[0], precision: 10);
            Assert.Equal(1.0, result[1], precision: 10);
            Assert.Equal(2.0, result[2], precision: 10);
        }

        [Fact]
        public void PointwiseAbs_ReturnsAbsoluteValues()
        {
            // Arrange
            var v = new Vector<double>(new[] { -3.0, 0.0, 5.0, -7.0 });

            // Act
            var result = v.PointwiseAbs();

            // Assert
            Assert.Equal(3.0, result[0], precision: 10);
            Assert.Equal(0.0, result[1], precision: 10);
            Assert.Equal(5.0, result[2], precision: 10);
            Assert.Equal(7.0, result[3], precision: 10);
        }

        [Fact]
        public void PointwiseSqrt_CalculatesSquareRoots()
        {
            // Arrange
            var v = new Vector<double>(new[] { 4.0, 9.0, 16.0, 25.0 });

            // Act
            var result = v.PointwiseSqrt();

            // Assert
            Assert.Equal(2.0, result[0], precision: 10);
            Assert.Equal(3.0, result[1], precision: 10);
            Assert.Equal(4.0, result[2], precision: 10);
            Assert.Equal(5.0, result[3], precision: 10);
        }

        [Fact]
        public void PointwiseSign_ReturnsSignValues()
        {
            // Arrange
            var v = new Vector<double>(new[] { -5.0, 0.0, 3.0, -2.0 });

            // Act
            var result = v.PointwiseSign();

            // Assert
            Assert.Equal(-1.0, result[0], precision: 10);
            Assert.Equal(0.0, result[1], precision: 10);
            Assert.Equal(1.0, result[2], precision: 10);
            Assert.Equal(-1.0, result[3], precision: 10);
        }

        #endregion

        #region Static Factory Methods

        [Fact]
        public void Empty_CreatesEmptyVector()
        {
            // Act
            var v = Vector<double>.Empty();

            // Assert
            Assert.Equal(0, v.Length);
        }

        [Fact]
        public void Zeros_CreatesVectorOfZeros()
        {
            // Act
            var v = new Vector<double>(5).Zeros(5);

            // Assert
            Assert.Equal(5, v.Length);
            Assert.Equal(0.0, v[0]);
            Assert.Equal(0.0, v[4]);
        }

        [Fact]
        public void Ones_CreatesVectorOfOnes()
        {
            // Act
            var v = new Vector<double>(5).Ones(5);

            // Assert
            Assert.Equal(5, v.Length);
            Assert.Equal(1.0, v[0]);
            Assert.Equal(1.0, v[4]);
        }

        [Fact]
        public void CreateDefault_CreatesVectorWithDefaultValue()
        {
            // Act
            var v = Vector<double>.CreateDefault(4, 7.5);

            // Assert
            Assert.Equal(4, v.Length);
            Assert.Equal(7.5, v[0]);
            Assert.Equal(7.5, v[3]);
        }

        [Fact]
        public void Range_CreatesSequentialVector()
        {
            // Act - start at 5, create 4 elements
            var v = Vector<double>.Range(5, 4);

            // Assert
            Assert.Equal(4, v.Length);
            Assert.Equal(5.0, v[0]);
            Assert.Equal(6.0, v[1]);
            Assert.Equal(7.0, v[2]);
            Assert.Equal(8.0, v[3]);
        }

        [Fact]
        public void CreateRandom_CreatesRandomVector()
        {
            // Act
            var v = Vector<double>.CreateRandom(10);

            // Assert
            Assert.Equal(10, v.Length);
            // Check all values are between 0 and 1
            for (int i = 0; i < v.Length; i++)
            {
                Assert.True(v[i] >= 0.0 && v[i] <= 1.0);
            }
        }

        [Fact]
        public void CreateRandom_WithMinMax_CreatesRandomVectorInRange()
        {
            // Act
            var v = Vector<double>.CreateRandom(10, -5.0, 5.0);

            // Assert
            Assert.Equal(10, v.Length);
            // Check all values are between -5 and 5
            for (int i = 0; i < v.Length; i++)
            {
                Assert.True(v[i] >= -5.0 && v[i] <= 5.0);
            }
        }

        [Fact]
        public void CreateStandardBasis_CreatesCorrectVector()
        {
            // Act - create basis vector with 1 at index 2
            var v = Vector<double>.CreateStandardBasis(5, 2);

            // Assert
            Assert.Equal(5, v.Length);
            Assert.Equal(0.0, v[0]);
            Assert.Equal(0.0, v[1]);
            Assert.Equal(1.0, v[2]);
            Assert.Equal(0.0, v[3]);
            Assert.Equal(0.0, v[4]);
        }

        [Fact]
        public void FromArray_CreatesVectorFromArray()
        {
            // Arrange
            var array = new[] { 1.0, 2.0, 3.0, 4.0 };

            // Act
            var v = Vector<double>.FromArray(array);

            // Assert
            Assert.Equal(4, v.Length);
            Assert.Equal(1.0, v[0]);
            Assert.Equal(4.0, v[3]);
        }

        [Fact]
        public void FromList_CreatesVectorFromList()
        {
            // Arrange
            var list = new List<double> { 1.0, 2.0, 3.0, 4.0 };

            // Act
            var v = Vector<double>.FromList(list);

            // Assert
            Assert.Equal(4, v.Length);
            Assert.Equal(1.0, v[0]);
            Assert.Equal(4.0, v[3]);
        }

        [Fact]
        public void FromEnumerable_CreatesVectorFromEnumerable()
        {
            // Arrange
            var enumerable = Enumerable.Range(1, 5).Select(x => (double)x);

            // Act
            var v = Vector<double>.FromEnumerable(enumerable);

            // Assert
            Assert.Equal(5, v.Length);
            Assert.Equal(1.0, v[0]);
            Assert.Equal(5.0, v[4]);
        }

        #endregion

        #region Vector Operations

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var v2 = v1.Clone();
            v2[0] = 99.0;

            // Assert
            Assert.Equal(1.0, v1[0]); // Original unchanged
            Assert.Equal(99.0, v2[0]); // Clone modified
        }

        [Fact]
        public void Divide_DividesByScalar()
        {
            // Arrange
            var v = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

            // Act
            var result = v.Divide(10.0);

            // Assert
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
        }

        [Fact]
        public void Transform_WithFunction_TransformsElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var result = v.Transform(x => x * 2 + 1);

            // Assert
            Assert.Equal(3.0, result[0]);
            Assert.Equal(5.0, result[1]);
            Assert.Equal(7.0, result[2]);
        }

        [Fact]
        public void Transform_WithFunctionAndIndex_UsesIndex()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var result = v.Transform((x, i) => x * i);

            // Assert
            Assert.Equal(0.0, result[0]); // 1 * 0
            Assert.Equal(2.0, result[1]); // 2 * 1
            Assert.Equal(6.0, result[2]); // 3 * 2
        }

        [Fact]
        public void IndexOfMax_ReturnsCorrectIndex()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 7.0, 2.0, 9.0, 5.0 });

            // Act
            var index = v.IndexOfMax();

            // Assert
            Assert.Equal(3, index); // 9.0 is at index 3
        }

        [Fact]
        public void MaxIndex_ReturnsCorrectIndex()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 7.0, 2.0, 9.0, 5.0 });

            // Act
            var index = v.MaxIndex();

            // Assert
            Assert.Equal(3, index);
        }

        [Fact]
        public void MinIndex_ReturnsCorrectIndex()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 7.0, 2.0, 9.0, 5.0 });

            // Act
            var index = v.MinIndex();

            // Assert
            Assert.Equal(2, index); // 2.0 is at index 2
        }

        [Fact]
        public void OuterProduct_CreatesCorrectMatrix()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 1.0, 2.0 });
            var v2 = new Vector<double>(new[] { 3.0, 4.0, 5.0 });

            // Act
            var matrix = v1.OuterProduct(v2);

            // Assert - 2x3 matrix
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(3.0, matrix[0, 0]); // 1*3
            Assert.Equal(4.0, matrix[0, 1]); // 1*4
            Assert.Equal(5.0, matrix[0, 2]); // 1*5
            Assert.Equal(6.0, matrix[1, 0]); // 2*3
            Assert.Equal(8.0, matrix[1, 1]); // 2*4
            Assert.Equal(10.0, matrix[1, 2]); // 2*5
        }

        [Fact]
        public void RemoveAt_RemovesElementAtIndex()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = v.RemoveAt(2);

            // Assert
            Assert.Equal(4, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(4.0, result[2]); // was at index 3
            Assert.Equal(5.0, result[3]); // was at index 4
        }

        [Fact]
        public void NonZeroIndices_ReturnsCorrectIndices()
        {
            // Arrange
            var v = new Vector<double>(new[] { 0.0, 5.0, 0.0, 3.0, 0.0 });

            // Act
            var indices = v.NonZeroIndices().ToList();

            // Assert
            Assert.Equal(2, indices.Count);
            Assert.Equal(1, indices[0]);
            Assert.Equal(3, indices[1]);
        }

        [Fact]
        public void NonZeroCount_ReturnsCorrectCount()
        {
            // Arrange
            var v = new Vector<double>(new[] { 0.0, 5.0, 0.0, 3.0, 0.0, 1.0 });

            // Act
            var count = v.NonZeroCount();

            // Assert
            Assert.Equal(3, count);
        }

        [Fact]
        public void Fill_SetsAllElementsToValue()
        {
            // Arrange
            var v = new Vector<double>(5);

            // Act
            v.Fill(7.5);

            // Assert
            for (int i = 0; i < v.Length; i++)
            {
                Assert.Equal(7.5, v[i]);
            }
        }

        [Fact]
        public void Concatenate_WithParams_CombinesVectors()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 1.0, 2.0 });
            var v2 = new Vector<double>(new[] { 3.0, 4.0 });
            var v3 = new Vector<double>(new[] { 5.0, 6.0 });

            // Act
            var result = Vector<double>.Concatenate(v1, v2, v3);

            // Assert
            Assert.Equal(6, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(4.0, result[3]);
            Assert.Equal(6.0, result[5]);
        }

        [Fact]
        public void Concatenate_WithList_CombinesVectors()
        {
            // Arrange
            var vectors = new List<Vector<double>>
            {
                new Vector<double>(new[] { 1.0, 2.0 }),
                new Vector<double>(new[] { 3.0, 4.0 }),
                new Vector<double>(new[] { 5.0, 6.0 })
            };

            // Act
            var result = Vector<double>.Concatenate(vectors);

            // Assert
            Assert.Equal(6, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(6.0, result[5]);
        }

        [Fact]
        public void Transpose_CreatesRowMatrix()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = v.Transpose();

            // Assert
            Assert.Equal(1, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(2.0, matrix[0, 1]);
            Assert.Equal(3.0, matrix[0, 2]);
        }

        [Fact]
        public void AppendAsMatrix_CreatesMatrixWithConstantColumn()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = v.AppendAsMatrix(5.0);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(5.0, matrix[0, 1]);
            Assert.Equal(2.0, matrix[1, 0]);
            Assert.Equal(5.0, matrix[1, 1]);
        }

        [Fact]
        public void GetElements_ExtractsElementsAtIndices()
        {
            // Arrange
            var v = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var indices = new[] { 0, 2, 4 };

            // Act
            var result = v.GetElements(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(10.0, result[0]);
            Assert.Equal(30.0, result[1]);
            Assert.Equal(50.0, result[2]);
        }

        [Fact]
        public void BinarySearch_FindsExistingValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 });

            // Act
            var index = v.BinarySearch(5.0);

            // Assert
            Assert.Equal(2, index);
        }

        [Fact]
        public void BinarySearch_ReturnsNegativeForMissingValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 });

            // Act
            var index = v.BinarySearch(4.0);

            // Assert
            Assert.True(index < 0); // Not found
        }

        [Fact]
        public void IndexOf_FindsFirstOccurrence()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 2.0, 5.0 });

            // Act
            var index = v.IndexOf(2.0);

            // Assert
            Assert.Equal(1, index); // First occurrence at index 1
        }

        [Fact]
        public void IndexOf_ReturnsNegativeOneWhenNotFound()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var index = v.IndexOf(99.0);

            // Assert
            Assert.Equal(-1, index);
        }

        [Fact]
        public void SetValue_CreatesNewVectorWithChangedValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var result = v.SetValue(1, 99.0);

            // Assert
            Assert.Equal(1.0, v[1]); // Original unchanged
            Assert.Equal(99.0, result[1]); // New vector has change
        }

        [Fact]
        public void ToArray_CreatesIndependentArray()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var array = v.ToArray();
            array[0] = 99.0;

            // Assert
            Assert.Equal(1.0, v[0]); // Vector unchanged
            Assert.Equal(99.0, array[0]); // Array modified
        }

        #endregion

        #region Operator Tests

        [Fact]
        public void OperatorPlus_VectorPlusScalar_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var result = v + 10.0;

            // Assert
            Assert.Equal(11.0, result[0]);
            Assert.Equal(12.0, result[1]);
            Assert.Equal(13.0, result[2]);
        }

        [Fact]
        public void OperatorMinus_VectorMinusScalar_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

            // Act
            var result = v - 5.0;

            // Assert
            Assert.Equal(5.0, result[0]);
            Assert.Equal(15.0, result[1]);
            Assert.Equal(25.0, result[2]);
        }

        [Fact]
        public void OperatorMultiply_ScalarTimesVector_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var result = 3.0 * v;

            // Assert
            Assert.Equal(3.0, result[0]);
            Assert.Equal(6.0, result[1]);
            Assert.Equal(9.0, result[2]);
        }

        [Fact]
        public void OperatorDivide_VectorDividedByScalar_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

            // Act
            var result = v / 10.0;

            // Assert
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
        }

        [Fact]
        public void ImplicitOperator_ConvertsVectorToArray()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            double[] array = v;

            // Assert
            Assert.Equal(3, array.Length);
            Assert.Equal(1.0, array[0]);
            Assert.Equal(3.0, array[2]);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void EmptyVector_HasZeroLength()
        {
            // Act
            var v = Vector<double>.Empty();

            // Assert
            Assert.Equal(0, v.Length);
            Assert.True(v.IsEmpty);
        }

        [Fact]
        public void SingleElementVector_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 42.0 });

            // Assert
            Assert.Equal(1, v.Length);
            Assert.Equal(42.0, v[0]);
            Assert.Equal(42.0, v.Sum());
            Assert.Equal(42.0, v.Mean());
        }

        [Fact]
        public void LargeVector_WorksCorrectly()
        {
            // Arrange - Create vector with 10000 elements
            var values = Enumerable.Range(1, 10000).Select(x => (double)x).ToArray();
            var v = new Vector<double>(values);

            // Act & Assert
            Assert.Equal(10000, v.Length);
            Assert.Equal(1.0, v[0]);
            Assert.Equal(10000.0, v[9999]);

            // Sum = n(n+1)/2 = 10000*10001/2 = 50005000
            Assert.Equal(50005000.0, v.Sum(), precision: 10);
        }

        [Fact]
        public void VectorWithAllZeros_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(5).Zeros(5);

            // Assert
            Assert.Equal(0.0, v.Sum());
            Assert.Equal(0.0, v.Mean());
            Assert.Equal(0, v.NonZeroCount());
        }

        [Fact]
        public void VectorWithNegativeValues_WorksCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { -1.0, -2.0, -3.0 });

            // Act & Assert
            Assert.Equal(-6.0, v.Sum());
            Assert.Equal(-2.0, v.Mean());
            Assert.Equal(-1.0, v.Max());
            Assert.Equal(-3.0, v.Min());
        }

        [Fact]
        public void OrthogonalVectors_DotProductIsZero()
        {
            // Arrange - Vectors at right angles
            var v1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
            var v2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act
            var dot = v1.DotProduct(v2);

            // Assert
            Assert.Equal(0.0, dot, precision: 10);
        }

        [Fact]
        public void ParallelVectors_CosineSimilarityIsOne()
        {
            // Arrange - Vectors in same direction (one is scaled version of other)
            var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

            // Act
            var similarity = v1.CosineSimilarity(v2);

            // Assert
            Assert.Equal(1.0, similarity, precision: 10);
        }

        [Fact]
        public void UnitVector_HasMagnitudeOne()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

            // Act
            var magnitude = v.Magnitude();

            // Assert
            Assert.Equal(1.0, magnitude, precision: 10);
        }

        #endregion

        #region Different Numeric Types

        [Fact]
        public void Vector_WithIntType_WorksCorrectly()
        {
            // Arrange
            var v1 = new Vector<int>(new[] { 1, 2, 3 });
            var v2 = new Vector<int>(new[] { 4, 5, 6 });

            // Act
            var dotProduct = v1.DotProduct(v2);
            var sum = v1.Sum();

            // Assert
            Assert.Equal(32, dotProduct);
            Assert.Equal(6, sum);
        }

        [Fact]
        public void Vector_WithDecimalType_WorksCorrectly()
        {
            // Arrange
            var v1 = new Vector<decimal>(new[] { 1.5m, 2.5m, 3.5m });
            var v2 = new Vector<decimal>(new[] { 2.0m, 3.0m, 4.0m });

            // Act
            var result = v1 + v2;

            // Assert
            Assert.Equal(3.5m, result[0]);
            Assert.Equal(5.5m, result[1]);
            Assert.Equal(7.5m, result[2]);
        }

        #endregion

        #region Extension Methods

        [Fact]
        public void Argsort_ReturnsCorrectIndices()
        {
            // Arrange
            var v = new Vector<double>(new[] { 3.0, 1.0, 4.0, 1.5, 9.0 });

            // Act
            var indices = v.Argsort();

            // Assert - sorted order: [1, 1.5, 3, 4, 9] at indices [1, 3, 0, 2, 4]
            Assert.Equal(1, indices[0]); // smallest at index 1
            Assert.Equal(3, indices[1]); // second smallest at index 3
            Assert.Equal(0, indices[2]); // third at index 0
            Assert.Equal(2, indices[3]); // fourth at index 2
            Assert.Equal(4, indices[4]); // largest at index 4
        }

        [Fact]
        public void Repeat_RepeatsVectorCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0 });

            // Act
            var result = v.Repeat(3);

            // Assert
            Assert.Equal(6, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(1.0, result[2]);
            Assert.Equal(2.0, result[3]);
            Assert.Equal(1.0, result[4]);
            Assert.Equal(2.0, result[5]);
        }

        [Fact]
        public void AbsoluteMaximum_ReturnsLargestAbsoluteValue()
        {
            // Arrange
            var v = new Vector<double>(new[] { -10.0, 5.0, -3.0, 7.0 });

            // Act
            var absMax = v.AbsoluteMaximum();

            // Assert
            Assert.Equal(10.0, absMax); // |-10| = 10 is largest
        }

        [Fact]
        public void Maximum_WithScalar_ReturnsMaximums()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 5.0, 3.0 });

            // Act
            var result = v.Maximum(3.0);

            // Assert
            Assert.Equal(3.0, result[0]); // max(1, 3) = 3
            Assert.Equal(5.0, result[1]); // max(5, 3) = 5
            Assert.Equal(3.0, result[2]); // max(3, 3) = 3
        }

        [Fact]
        public void ToDiagonalMatrix_CreatesCorrectMatrix()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = v.ToDiagonalMatrix();

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(0.0, matrix[0, 1]);
            Assert.Equal(2.0, matrix[1, 1]);
            Assert.Equal(0.0, matrix[1, 2]);
            Assert.Equal(3.0, matrix[2, 2]);
        }

        [Fact]
        public void ToColumnMatrix_CreatesCorrectMatrix()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = v.ToColumnMatrix();

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(1, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(2.0, matrix[1, 0]);
            Assert.Equal(3.0, matrix[2, 0]);
        }

        [Fact]
        public void Extract_ExtractsFirstNElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = v.Extract(3);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
        }

        [Fact]
        public void Reshape_CreatesMatrixWithCorrectDimensions()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Act
            var matrix = v.Reshape(2, 3);

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(2.0, matrix[0, 1]);
            Assert.Equal(3.0, matrix[0, 2]);
            Assert.Equal(4.0, matrix[1, 0]);
            Assert.Equal(5.0, matrix[1, 1]);
            Assert.Equal(6.0, matrix[1, 2]);
        }

        [Fact]
        public void SubVector_WithIndices_ExtractsCorrectElements()
        {
            // Arrange
            var v = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var indices = new[] { 0, 2, 4 };

            // Act
            var result = v.Subvector(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(10.0, result[0]);
            Assert.Equal(30.0, result[1]);
            Assert.Equal(50.0, result[2]);
        }

        #endregion

        #region Serialization Tests

        [Fact]
        public void Serialize_Deserialize_PreservesVector()
        {
            // Arrange
            var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var serialized = original.Serialize();
            var deserialized = Vector<double>.Deserialize(serialized);

            // Assert
            Assert.Equal(original.Length, deserialized.Length);
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], deserialized[i]);
            }
        }

        #endregion

        #region ToString Test

        [Fact]
        public void ToString_FormatsCorrectly()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var str = v.ToString();

            // Assert
            Assert.Equal("[1, 2, 3]", str);
        }

        #endregion

        #region Additional Similarity Tests

        [Fact]
        public void JaccardSimilarity_CalculatesCorrectly()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
            var v2 = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0 });

            // Act - Intersection: 2, Union: 4, Jaccard = 2/4 = 0.5
            var similarity = StatisticsHelper.JaccardSimilarity(v1, v2);

            // Assert
            Assert.True(similarity >= 0.0 && similarity <= 1.0);
        }

        #endregion

        #region Normalize Edge Cases

        [Fact]
        public void Normalize_ZeroVector_ThrowsException()
        {
            // Arrange
            var v = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => v.Normalize());
        }

        #endregion

        #region GetEnumerator Tests

        [Fact]
        public void GetEnumerator_AllowsForeachIteration()
        {
            // Arrange
            var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var sum = 0.0;

            // Act
            foreach (var value in v)
            {
                sum += value;
            }

            // Assert
            Assert.Equal(6.0, sum);
        }

        #endregion

        #region PointwiseMultiplyInPlace Tests

        [Fact]
        public void PointwiseMultiplyInPlace_ModifiesOriginalVector()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var v2 = new Vector<double>(new[] { 5.0, 6.0, 7.0 });

            // Act
            v1.PointwiseMultiplyInPlace(v2);

            // Assert
            Assert.Equal(10.0, v1[0]);
            Assert.Equal(18.0, v1[1]);
            Assert.Equal(28.0, v1[2]);
        }

        #endregion

        #region StandardDeviation Edge Case

        [Fact]
        public void StandardDeviation_SingleElement_ReturnsZero()
        {
            // Arrange
            var v = new Vector<double>(new[] { 5.0 });

            // Act
            var stdDev = v.StandardDeviation();

            // Assert
            Assert.Equal(0.0, stdDev);
        }

        #endregion

        #region CreateRandom Edge Cases

        [Fact]
        public void CreateRandom_InvalidRange_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => Vector<double>.CreateRandom(5, 10.0, 5.0));
        }

        #endregion
    }
}
