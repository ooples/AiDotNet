using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LinearAlgebra
{
    public class TensorTests
    {
        [Fact]
        public void Constructor_WithDimensions_InitializesCorrectly()
        {
            // Arrange & Act
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });

            // Assert
            Assert.Equal(3, tensor.Rank);
            Assert.Equal(new int[] { 2, 3, 4 }, tensor.Shape);
            Assert.Equal(24, tensor.TotalSize);
        }

        [Fact]
        public void Constructor_WithEmptyDimensions_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Tensor<double>(new int[] { }));
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Tensor<double>(new int[] { 2, 0, 3 }));
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Tensor<double>(new int[] { 2, -1, 3 }));
        }

        [Fact]
        public void Constructor_WithData_InitializesCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });

            // Act
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(new int[] { 2, 3 }, tensor.Shape);
            Assert.Equal(6, tensor.TotalSize);
        }

        [Fact]
        public void Constructor_WithMatrix_InitializesCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            // Act
            var tensor = new Tensor<double>(new int[] { 2, 3 }, matrix);

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(new int[] { 2, 3 }, tensor.Shape);
            Assert.Equal(6, tensor.TotalSize);
        }

        [Fact]
        public void Constructor_WithIncompatibleMatrix_ThrowsArgumentException()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Tensor<double>(new int[] { 2, 3 }, matrix));
        }

        [Fact]
        public void Indexer_1D_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 5 });

            // Act
            tensor[new int[] { 0 }] = 10.0;
            tensor[new int[] { 2 }] = 20.0;
            tensor[new int[] { 4 }] = 30.0;

            // Assert
            Assert.Equal(10.0, tensor[new int[] { 0 }]);
            Assert.Equal(20.0, tensor[new int[] { 2 }]);
            Assert.Equal(30.0, tensor[new int[] { 4 }]);
        }

        [Fact]
        public void Indexer_2D_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });

            // Act
            tensor[new int[] { 0, 0 }] = 1.0;
            tensor[new int[] { 1, 2 }] = 2.0;
            tensor[new int[] { 2, 3 }] = 3.0;

            // Assert
            Assert.Equal(1.0, tensor[new int[] { 0, 0 }]);
            Assert.Equal(2.0, tensor[new int[] { 1, 2 }]);
            Assert.Equal(3.0, tensor[new int[] { 2, 3 }]);
        }

        [Fact]
        public void Indexer_3D_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });

            // Act
            tensor[new int[] { 0, 0, 0 }] = 100.0;
            tensor[new int[] { 1, 2, 3 }] = 200.0;

            // Assert
            Assert.Equal(100.0, tensor[new int[] { 0, 0, 0 }]);
            Assert.Equal(200.0, tensor[new int[] { 1, 2, 3 }]);
        }

        [Fact]
        public void Indexer_OutOfBounds_ThrowsIndexOutOfRangeException()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });

            // Act & Assert
            Assert.Throws<IndexOutOfRangeException>(() => tensor[new int[] { 3, 0 }]);
            Assert.Throws<IndexOutOfRangeException>(() => tensor[new int[] { 0, 5 }]);
        }

        [Fact]
        public void Add_TwoTensors_ReturnsCorrectSum()
        {
            // Arrange
            var data1 = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var data2 = new Vector<double>(new double[] { 5, 6, 7, 8 });
            var t1 = new Tensor<double>(new int[] { 2, 2 }, data1);
            var t2 = new Tensor<double>(new int[] { 2, 2 }, data2);

            // Act
            var result = t1.Add(t2);

            // Assert
            Assert.Equal(6.0, result[new int[] { 0, 0 }]);
            Assert.Equal(8.0, result[new int[] { 0, 1 }]);
            Assert.Equal(10.0, result[new int[] { 1, 0 }]);
            Assert.Equal(12.0, result[new int[] { 1, 1 }]);
        }

        [Fact]
        public void Add_DifferentShapes_ThrowsArgumentException()
        {
            // Arrange
            var t1 = new Tensor<double>(new int[] { 2, 3 });
            var t2 = new Tensor<double>(new int[] { 3, 2 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => t1.Add(t2));
        }

        [Fact]
        public void Subtract_TwoTensors_ReturnsCorrectDifference()
        {
            // Arrange
            var data1 = new Vector<double>(new double[] { 10, 20, 30, 40 });
            var data2 = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var t1 = new Tensor<double>(new int[] { 2, 2 }, data1);
            var t2 = new Tensor<double>(new int[] { 2, 2 }, data2);

            // Act
            var result = t1.Subtract(t2);

            // Assert
            Assert.Equal(9.0, result[new int[] { 0, 0 }]);
            Assert.Equal(18.0, result[new int[] { 0, 1 }]);
            Assert.Equal(27.0, result[new int[] { 1, 0 }]);
            Assert.Equal(36.0, result[new int[] { 1, 1 }]);
        }

        [Fact]
        public void Multiply_ByScalar_ReturnsCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 2, 4, 6, 8 });
            var tensor = new Tensor<double>(new int[] { 2, 2 }, data);

            // Act
            var result = tensor.Multiply(3.0);

            // Assert
            Assert.Equal(6.0, result[new int[] { 0, 0 }]);
            Assert.Equal(12.0, result[new int[] { 0, 1 }]);
            Assert.Equal(18.0, result[new int[] { 1, 0 }]);
            Assert.Equal(24.0, result[new int[] { 1, 1 }]);
        }

        [Fact]
        public void ElementwiseMultiply_TwoTensors_ReturnsCorrectResult()
        {
            // Arrange
            var data1 = new Vector<double>(new double[] { 2, 3, 4, 5 });
            var data2 = new Vector<double>(new double[] { 6, 7, 8, 9 });
            var t1 = new Tensor<double>(new int[] { 2, 2 }, data1);
            var t2 = new Tensor<double>(new int[] { 2, 2 }, data2);

            // Act
            var result = t1.ElementwiseMultiply(t2);

            // Assert
            Assert.Equal(12.0, result[new int[] { 0, 0 }]);
            Assert.Equal(21.0, result[new int[] { 0, 1 }]);
            Assert.Equal(32.0, result[new int[] { 1, 0 }]);
            Assert.Equal(45.0, result[new int[] { 1, 1 }]);
        }

        [Fact]
        public void Reshape_ValidDimensions_ReturnsCorrectShape()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var reshaped = tensor.Reshape(new int[] { 3, 2 });

            // Assert
            Assert.Equal(new int[] { 3, 2 }, reshaped.Shape);
            Assert.Equal(6, reshaped.TotalSize);
            Assert.Equal(1.0, reshaped[new int[] { 0, 0 }]);
            Assert.Equal(2.0, reshaped[new int[] { 0, 1 }]);
            Assert.Equal(3.0, reshaped[new int[] { 1, 0 }]);
        }

        [Fact]
        public void Reshape_IncompatibleSize_ThrowsArgumentException()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => tensor.Reshape(new int[] { 2, 4 }));
        }

        [Fact]
        public void Sum_ReturnsCorrectTotal()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var result = tensor.Sum();

            // Assert
            Assert.Equal(21.0, result);
        }

        [Fact]
        public void Mean_ReturnsCorrectAverage()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 2, 4, 6, 8 });
            var tensor = new Tensor<double>(new int[] { 2, 2 }, data);

            // Act
            var result = tensor.Mean();

            // Assert
            Assert.Equal(5.0, result);
        }

        [Fact]
        public void Clone_CreatesDeepCopy()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var original = new Tensor<double>(new int[] { 2, 2 }, data);

            // Act
            var clone = original.Clone();
            clone[new int[] { 0, 0 }] = 999.0;

            // Assert
            Assert.Equal(1.0, original[new int[] { 0, 0 }]);
            Assert.Equal(999.0, clone[new int[] { 0, 0 }]);
            Assert.Equal(original.Shape, clone.Shape);
        }

        [Fact]
        public void Transpose_2DTensor_ReturnsCorrectTranspose()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var transposed = tensor.Transpose(new int[] { 1, 0 });

            // Assert
            Assert.Equal(new int[] { 3, 2 }, transposed.Shape);
            Assert.Equal(1.0, transposed[new int[] { 0, 0 }]);
            Assert.Equal(4.0, transposed[new int[] { 0, 1 }]);
            Assert.Equal(2.0, transposed[new int[] { 1, 0 }]);
            Assert.Equal(5.0, transposed[new int[] { 1, 1 }]);
        }

        [Fact]
        public void Max_ReturnsLargestElement()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 3, 7, 2, 9, 1, 5 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var result = tensor.Max();

            // Assert
            Assert.Equal(9.0, result);
        }

        [Fact]
        public void Min_ReturnsSmallestElement()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 3, 7, 2, 9, 1, 5 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var result = tensor.Min();

            // Assert
            Assert.Equal(1.0, result);
        }

        [Fact]
        public void GetSlice_ExtractsSubTensor()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var tensor = new Tensor<double>(new int[] { 2, 4 }, data);

            // Act
            var slice = tensor.GetSlice(new int[] { 0, 1 }, new int[] { 1, 3 });

            // Assert
            Assert.Equal(new int[] { 1, 2 }, slice.Shape);
            Assert.Equal(2.0, slice[new int[] { 0, 0 }]);
            Assert.Equal(3.0, slice[new int[] { 0, 1 }]);
        }

        [Fact]
        public void Apply_AppliesFunctionToEachElement()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var tensor = new Tensor<double>(new int[] { 2, 2 }, data);

            // Act
            var result = tensor.Apply(x => x * 2.0);

            // Assert
            Assert.Equal(2.0, result[new int[] { 0, 0 }]);
            Assert.Equal(4.0, result[new int[] { 0, 1 }]);
            Assert.Equal(6.0, result[new int[] { 1, 0 }]);
            Assert.Equal(8.0, result[new int[] { 1, 1 }]);
        }

        [Fact]
        public void GetEnumerator_AllowsIteration()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var tensor = new Tensor<double>(new int[] { 2, 2 }, data);

            // Act
            var values = tensor.ToList();

            // Assert
            Assert.Equal(4, values.Count);
            Assert.Contains(1.0, values);
            Assert.Contains(2.0, values);
            Assert.Contains(3.0, values);
            Assert.Contains(4.0, values);
        }

        [Fact]
        public void IntTensor_Constructor_WorksCorrectly()
        {
            // Arrange & Act
            var data = new Vector<int>(new int[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<int>(new int[] { 2, 3 }, data);

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(new int[] { 2, 3 }, tensor.Shape);
            Assert.Equal(6, tensor.TotalSize);
        }

        [Fact]
        public void FloatTensor_Constructor_WorksCorrectly()
        {
            // Arrange & Act
            var data = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var tensor = new Tensor<float>(new int[] { 2, 2 }, data);

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(new int[] { 2, 2 }, tensor.Shape);
            Assert.Equal(4, tensor.TotalSize);
        }

        [Fact]
        public void Flatten_ReturnsVectorWithAllElements()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var flattened = tensor.Flatten();

            // Assert
            Assert.Equal(6, flattened.Length);
            Assert.Equal(1.0, flattened[0]);
            Assert.Equal(2.0, flattened[1]);
            Assert.Equal(6.0, flattened[5]);
        }

        [Fact]
        public void ToMatrix_2DTensor_ConvertsCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Act
            var matrix = tensor.ToMatrix();

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(6.0, matrix[1, 2]);
        }

        [Fact]
        public void Squeeze_RemovesSingletonDimensions()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var tensor = new Tensor<double>(new int[] { 1, 4, 1 }, data);

            // Act
            var squeezed = tensor.Squeeze();

            // Assert
            Assert.Equal(new int[] { 4 }, squeezed.Shape);
            Assert.Equal(1, squeezed.Rank);
        }

        [Fact]
        public void Unsqueeze_AddsDimension()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4 });
            var tensor = new Tensor<double>(new int[] { 4 }, data);

            // Act
            var unsqueezed = tensor.Unsqueeze(0);

            // Assert
            Assert.Equal(new int[] { 1, 4 }, unsqueezed.Shape);
            Assert.Equal(2, unsqueezed.Rank);
        }

        [Fact]
        public void Broadcast_ExpandsDimensions()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3 });
            var tensor = new Tensor<double>(new int[] { 3 }, data);

            // Act
            var broadcasted = tensor.Broadcast(new int[] { 2, 3 });

            // Assert
            Assert.Equal(new int[] { 2, 3 }, broadcasted.Shape);
            Assert.Equal(1.0, broadcasted[new int[] { 0, 0 }]);
            Assert.Equal(1.0, broadcasted[new int[] { 1, 0 }]);
            Assert.Equal(3.0, broadcasted[new int[] { 0, 2 }]);
            Assert.Equal(3.0, broadcasted[new int[] { 1, 2 }]);
        }
    }
}
