using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LinearAlgebra
{
    public class VectorTests
    {
        [Fact]
        public void Constructor_WithLength_InitializesCorrectLength()
        {
            // Arrange & Act
            var vector = new Vector<double>(5);

            // Assert
            Assert.Equal(5, vector.Length);
            Assert.All(vector, item => Assert.Equal(0.0, item));
        }

        [Fact]
        public void Constructor_WithZeroLength_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Vector<double>(0));
        }

        [Fact]
        public void Constructor_WithNegativeLength_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Vector<double>(-1));
        }

        [Fact]
        public void Constructor_WithValues_InitializesCorrectly()
        {
            // Arrange
            var values = new double[] { 1.0, 2.0, 3.0, 4.0 };

            // Act
            var vector = new Vector<double>(values);

            // Assert
            Assert.Equal(4, vector.Length);
            Assert.Equal(1.0, vector[0]);
            Assert.Equal(2.0, vector[1]);
            Assert.Equal(3.0, vector[2]);
            Assert.Equal(4.0, vector[3]);
        }

        [Fact]
        public void Indexer_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(3);

            // Act
            vector[0] = 10.0;
            vector[1] = 20.0;
            vector[2] = 30.0;

            // Assert
            Assert.Equal(10.0, vector[0]);
            Assert.Equal(20.0, vector[1]);
            Assert.Equal(30.0, vector[2]);
        }

        [Fact]
        public void Add_TwoVectors_ReturnsCorrectSum()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

            // Act
            var result = v1.Add(v2);

            // Assert
            Assert.Equal(5.0, result[0]);
            Assert.Equal(7.0, result[1]);
            Assert.Equal(9.0, result[2]);
        }

        [Fact]
        public void Add_DifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => v1.Add(v2));
        }

        [Fact]
        public void Subtract_TwoVectors_ReturnsCorrectDifference()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var v2 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = v1.Subtract(v2);

            // Assert
            Assert.Equal(9.0, result[0]);
            Assert.Equal(18.0, result[1]);
            Assert.Equal(27.0, result[2]);
        }

        [Fact]
        public void Subtract_DifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => v1.Subtract(v2));
        }

        [Fact]
        public void Multiply_ByScalar_ReturnsCorrectResult()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = vector.Multiply(3.0);

            // Assert
            Assert.Equal(6.0, result[0]);
            Assert.Equal(12.0, result[1]);
            Assert.Equal(18.0, result[2]);
        }

        [Fact]
        public void DotProduct_TwoVectors_ReturnsCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

            // Act
            var result = v1.DotProduct(v2);

            // Assert
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            Assert.Equal(32.0, result);
        }

        [Fact]
        public void DotProduct_DifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new double[] { 4.0, 5.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => v1.DotProduct(v2));
        }

        [Fact]
        public void ElementwiseDivide_TwoVectors_ReturnsCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var v2 = new Vector<double>(new double[] { 2.0, 4.0, 5.0 });

            // Act
            var result = v1.ElementwiseDivide(v2);

            // Assert
            Assert.Equal(5.0, result[0]);
            Assert.Equal(5.0, result[1]);
            Assert.Equal(6.0, result[2]);
        }

        [Fact]
        public void ElementwiseDivide_DifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var v2 = new Vector<double>(new double[] { 2.0, 4.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => v1.ElementwiseDivide(v2));
        }

        [Fact]
        public void Magnitude_ReturnsCorrectResult()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 3.0, 4.0 });

            // Act
            var result = vector.Magnitude();

            // Assert
            // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0
            Assert.Equal(5.0, result, 5);
        }

        [Fact]
        public void Normalize_ReturnsUnitVector()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 3.0, 4.0 });

            // Act
            var result = vector.Normalize();

            // Assert
            Assert.Equal(0.6, result[0], 5);
            Assert.Equal(0.8, result[1], 5);
            Assert.Equal(1.0, result.Magnitude(), 5);
        }

        [Fact]
        public void Mean_ReturnsCorrectAverage()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0 });

            // Act
            var result = vector.Mean();

            // Assert
            // (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
            Assert.Equal(5.0, result);
        }

        [Fact]
        public void Variance_ReturnsCorrectValue()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0 });

            // Act
            var result = vector.Variance();

            // Assert
            // Mean = 5.0
            // Variance = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4
            //          = (9 + 1 + 1 + 9) / 4 = 20 / 4 = 5.0
            Assert.Equal(5.0, result, 5);
        }

        [Fact]
        public void Sum_ReturnsCorrectTotal()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = vector.Sum();

            // Assert
            Assert.Equal(15.0, result);
        }

        [Fact]
        public void GetEnumerator_AllowsIteration()
        {
            // Arrange
            var values = new double[] { 1.0, 2.0, 3.0 };
            var vector = new Vector<double>(values);

            // Act
            var result = vector.ToArray();

            // Assert
            Assert.Equal(values, result);
        }

        [Fact]
        public void Foreach_AllowsIteration()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var sum = 0.0;

            // Act
            foreach (var value in vector)
            {
                sum += value;
            }

            // Assert
            Assert.Equal(6.0, sum);
        }

        [Fact]
        public void Clone_CreatesDeepCopy()
        {
            // Arrange
            var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var clone = original.Clone();
            clone[0] = 999.0;

            // Assert
            Assert.Equal(1.0, original[0]);
            Assert.Equal(999.0, clone[0]);
            Assert.Equal(original.Length, clone.Length);
        }

        [Fact]
        public void Max_ReturnsLargestElement()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 3.0, 7.0, 2.0, 9.0, 1.0 });

            // Act
            var result = vector.Max();

            // Assert
            Assert.Equal(9.0, result);
        }

        [Fact]
        public void Min_ReturnsSmallestElement()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 3.0, 7.0, 2.0, 9.0, 1.0 });

            // Act
            var result = vector.Min();

            // Assert
            Assert.Equal(1.0, result);
        }

        [Fact]
        public void ToArray_ReturnsCorrectArray()
        {
            // Arrange
            var values = new double[] { 1.0, 2.0, 3.0 };
            var vector = new Vector<double>(values);

            // Act
            var result = vector.ToArray();

            // Assert
            Assert.Equal(values, result);
        }

        [Fact]
        public void Concatenate_TwoVectors_ReturnsCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 1.0, 2.0 });
            var v2 = new Vector<double>(new double[] { 3.0, 4.0, 5.0 });

            // Act
            var result = v1.Concatenate(v2);

            // Assert
            Assert.Equal(5, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
            Assert.Equal(4.0, result[3]);
            Assert.Equal(5.0, result[4]);
        }

        [Fact]
        public void Slice_ExtractsSubVector()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = vector.Slice(1, 3);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(2.0, result[0]);
            Assert.Equal(3.0, result[1]);
            Assert.Equal(4.0, result[2]);
        }

        [Fact]
        public void ElementwiseMultiply_TwoVectors_ReturnsCorrectResult()
        {
            // Arrange
            var v1 = new Vector<double>(new double[] { 2.0, 3.0, 4.0 });
            var v2 = new Vector<double>(new double[] { 5.0, 6.0, 7.0 });

            // Act
            var result = v1.ElementwiseMultiply(v2);

            // Assert
            Assert.Equal(10.0, result[0]);
            Assert.Equal(18.0, result[1]);
            Assert.Equal(28.0, result[2]);
        }

        [Fact]
        public void Apply_AppliesFunctionToEachElement()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = vector.Apply(x => x * 2.0);

            // Assert
            Assert.Equal(2.0, result[0]);
            Assert.Equal(4.0, result[1]);
            Assert.Equal(6.0, result[2]);
        }

        [Fact]
        public void IntVector_Constructor_WorksCorrectly()
        {
            // Arrange & Act
            var vector = new Vector<int>(new int[] { 1, 2, 3, 4 });

            // Assert
            Assert.Equal(4, vector.Length);
            Assert.Equal(1, vector[0]);
            Assert.Equal(2, vector[1]);
            Assert.Equal(3, vector[2]);
            Assert.Equal(4, vector[3]);
        }

        [Fact]
        public void IntVector_Add_WorksCorrectly()
        {
            // Arrange
            var v1 = new Vector<int>(new int[] { 1, 2, 3 });
            var v2 = new Vector<int>(new int[] { 4, 5, 6 });

            // Act
            var result = v1.Add(v2);

            // Assert
            Assert.Equal(5, result[0]);
            Assert.Equal(7, result[1]);
            Assert.Equal(9, result[2]);
        }

        [Fact]
        public void FloatVector_Constructor_WorksCorrectly()
        {
            // Arrange & Act
            var vector = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Assert
            Assert.Equal(3, vector.Length);
            Assert.Equal(1.0f, vector[0]);
            Assert.Equal(2.0f, vector[1]);
            Assert.Equal(3.0f, vector[2]);
        }

        [Fact]
        public void FloatVector_Multiply_WorksCorrectly()
        {
            // Arrange
            var vector = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });

            // Act
            var result = vector.Multiply(3.0f);

            // Assert
            Assert.Equal(6.0f, result[0]);
            Assert.Equal(12.0f, result[1]);
            Assert.Equal(18.0f, result[2]);
        }
    }
}
