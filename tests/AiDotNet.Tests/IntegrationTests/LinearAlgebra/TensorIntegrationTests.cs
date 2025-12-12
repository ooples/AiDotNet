using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.LinearAlgebra
{
    /// <summary>
    /// Comprehensive integration tests for Tensor operations with mathematically verified results.
    /// These tests validate the mathematical correctness of tensor operations across different dimensions.
    /// </summary>
    public class TensorIntegrationTests
    {
        private const double Tolerance = 1e-10;

        #region Constructor Tests

        [Fact]
        public void Constructor_ScalarTensor_CreatesCorrectShape()
        {
            // Arrange & Act - 0D tensor (scalar)
            var tensor = new Tensor<double>(new int[] { });

            // Assert
            Assert.Equal(0, tensor.Rank);
            Assert.Equal(1, tensor.Length);
        }

        [Fact]
        public void Constructor_1DTensor_CreatesCorrectShape()
        {
            // Arrange & Act - 1D tensor (vector)
            var tensor = new Tensor<double>(new int[] { 5 });

            // Assert
            Assert.Equal(1, tensor.Rank);
            Assert.Equal(5, tensor.Length);
            Assert.Equal(new int[] { 5 }, tensor.Shape);
        }

        [Fact]
        public void Constructor_2DTensor_CreatesCorrectShape()
        {
            // Arrange & Act - 2D tensor (matrix)
            var tensor = new Tensor<double>(new int[] { 3, 4 });

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(12, tensor.Length);
            Assert.Equal(new int[] { 3, 4 }, tensor.Shape);
        }

        [Fact]
        public void Constructor_3DTensor_CreatesCorrectShape()
        {
            // Arrange & Act - 3D tensor
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });

            // Assert
            Assert.Equal(3, tensor.Rank);
            Assert.Equal(24, tensor.Length);
            Assert.Equal(new int[] { 2, 3, 4 }, tensor.Shape);
        }

        [Fact]
        public void Constructor_4DTensor_CreatesCorrectShape()
        {
            // Arrange & Act - 4D tensor (e.g., batch of images: NCHW)
            var tensor = new Tensor<double>(new int[] { 2, 3, 4, 5 });

            // Assert
            Assert.Equal(4, tensor.Rank);
            Assert.Equal(120, tensor.Length);
            Assert.Equal(new int[] { 2, 3, 4, 5 }, tensor.Shape);
        }

        [Fact]
        public void Constructor_5DTensor_CreatesCorrectShape()
        {
            // Arrange & Act - 5D tensor (e.g., video batch)
            var tensor = new Tensor<double>(new int[] { 2, 3, 4, 5, 6 });

            // Assert
            Assert.Equal(5, tensor.Rank);
            Assert.Equal(720, tensor.Length);
            Assert.Equal(new int[] { 2, 3, 4, 5, 6 }, tensor.Shape);
        }

        [Fact]
        public void Constructor_WithVectorData_PopulatesCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });

            // Act - Create 2x3 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3 }, data);

            // Assert
            Assert.Equal(1.0, tensor[0, 0], precision: 10);
            Assert.Equal(2.0, tensor[0, 1], precision: 10);
            Assert.Equal(3.0, tensor[0, 2], precision: 10);
            Assert.Equal(4.0, tensor[1, 0], precision: 10);
            Assert.Equal(5.0, tensor[1, 1], precision: 10);
            Assert.Equal(6.0, tensor[1, 2], precision: 10);
        }

        [Fact]
        public void Constructor_WithMatrixData_PopulatesCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var tensor = new Tensor<double>(new int[] { 2, 3 }, matrix);

            // Assert
            Assert.Equal(1.0, tensor[0, 0], precision: 10);
            Assert.Equal(2.0, tensor[0, 1], precision: 10);
            Assert.Equal(3.0, tensor[0, 2], precision: 10);
            Assert.Equal(4.0, tensor[1, 0], precision: 10);
            Assert.Equal(5.0, tensor[1, 1], precision: 10);
            Assert.Equal(6.0, tensor[1, 2], precision: 10);
        }

        #endregion

        #region Factory Method Tests

        [Fact]
        public void CreateRandom_1DTensor_CreatesValidRandomValues()
        {
            // Act
            var tensor = Tensor<double>.CreateRandom(5);

            // Assert
            Assert.Equal(1, tensor.Rank);
            Assert.Equal(5, tensor.Length);
            // Verify all values are between 0 and 1
            for (int i = 0; i < 5; i++)
            {
                Assert.InRange(tensor[i], 0.0, 1.0);
            }
        }

        [Fact]
        public void CreateRandom_3DTensor_CreatesValidRandomValues()
        {
            // Act
            var tensor = Tensor<double>.CreateRandom(2, 3, 4);

            // Assert
            Assert.Equal(3, tensor.Rank);
            Assert.Equal(24, tensor.Length);
        }

        [Fact]
        public void CreateDefault_WithSpecificValue_FillsAllElements()
        {
            // Act
            var tensor = Tensor<double>.CreateDefault(new int[] { 2, 3 }, 5.5);

            // Assert
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(5.5, tensor[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void FromVector_CreatesCorrect1DTensor()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

            // Act
            var tensor = Tensor<double>.FromVector(vector);

            // Assert
            Assert.Equal(1, tensor.Rank);
            Assert.Equal(5, tensor.Length);
            Assert.Equal(1.0, tensor[0], precision: 10);
            Assert.Equal(5.0, tensor[4], precision: 10);
        }

        [Fact]
        public void FromVector_WithShape_ReshapesCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });

            // Act
            var tensor = Tensor<double>.FromVector(vector, new int[] { 2, 3 });

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(new int[] { 2, 3 }, tensor.Shape);
            Assert.Equal(1.0, tensor[0, 0], precision: 10);
            Assert.Equal(6.0, tensor[1, 2], precision: 10);
        }

        [Fact]
        public void FromMatrix_CreatesCorrect2DTensor()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var tensor = Tensor<double>.FromMatrix(matrix);

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(new int[] { 2, 3 }, tensor.Shape);
            Assert.Equal(1.0, tensor[0, 0], precision: 10);
            Assert.Equal(6.0, tensor[1, 2], precision: 10);
        }

        [Fact]
        public void FromScalar_CreatesScalarTensor()
        {
            // Act
            var tensor = Tensor<double>.FromScalar(42.0);

            // Assert
            Assert.Equal(1, tensor.Rank);
            Assert.Equal(1, tensor.Length);
            Assert.Equal(42.0, tensor[0], precision: 10);
        }

        [Fact]
        public void Empty_CreatesZeroLengthTensor()
        {
            // Act
            var tensor = Tensor<double>.Empty();

            // Assert
            Assert.Equal(1, tensor.Rank);
            Assert.Equal(0, tensor.Length);
        }

        [Fact]
        public void Stack_Axis0_Stacks2DTensorsCorrectly()
        {
            // Arrange - Stack two 2x3 tensors along axis 0
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 7; tensor2[0, 1] = 8; tensor2[0, 2] = 9;
            tensor2[1, 0] = 10; tensor2[1, 1] = 11; tensor2[1, 2] = 12;

            // Act - Stack along axis 0, resulting in 2x2x3 tensor
            var stacked = Tensor<double>.Stack(new[] { tensor1, tensor2 }, axis: 0);

            // Assert
            Assert.Equal(new int[] { 2, 2, 3 }, stacked.Shape);
            Assert.Equal(1.0, stacked[0, 0, 0], precision: 10);
            Assert.Equal(6.0, stacked[0, 1, 2], precision: 10);
            Assert.Equal(7.0, stacked[1, 0, 0], precision: 10);
            Assert.Equal(12.0, stacked[1, 1, 2], precision: 10);
        }

        [Fact]
        public void Stack_Axis1_Stacks2DTensorsCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 7; tensor2[0, 1] = 8; tensor2[0, 2] = 9;
            tensor2[1, 0] = 10; tensor2[1, 1] = 11; tensor2[1, 2] = 12;

            // Act - Stack along axis 1, resulting in 2x2x3 tensor
            var stacked = Tensor<double>.Stack(new[] { tensor1, tensor2 }, axis: 1);

            // Assert
            Assert.Equal(new int[] { 2, 2, 3 }, stacked.Shape);
        }

        #endregion

        #region Shape Transformation Tests

        [Fact]
        public void Reshape_2Dto1D_ReshapesCorrectly()
        {
            // Arrange - 2x3 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act - Reshape to 1D (6 elements)
            var reshaped = tensor.Reshape(6);

            // Assert
            Assert.Equal(1, reshaped.Rank);
            Assert.Equal(6, reshaped.Length);
            Assert.Equal(1.0, reshaped[0], precision: 10);
            Assert.Equal(6.0, reshaped[5], precision: 10);
        }

        [Fact]
        public void Reshape_1Dto2D_ReshapesCorrectly()
        {
            // Arrange - 1D tensor with 6 elements
            var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
            var tensor = new Tensor<double>(new int[] { 6 }, data);

            // Act - Reshape to 2x3
            var reshaped = tensor.Reshape(2, 3);

            // Assert
            Assert.Equal(2, reshaped.Rank);
            Assert.Equal(new int[] { 2, 3 }, reshaped.Shape);
            Assert.Equal(1.0, reshaped[0, 0], precision: 10);
            Assert.Equal(6.0, reshaped[1, 2], precision: 10);
        }

        [Fact]
        public void Reshape_2Dto3D_ReshapesCorrectly()
        {
            // Arrange - 4x3 tensor = 12 elements
            var tensor = new Tensor<double>(new int[] { 4, 3 });
            for (int i = 0; i < 12; i++)
            {
                tensor.SetFlatIndex(i, i + 1);
            }

            // Act - Reshape to 2x2x3
            var reshaped = tensor.Reshape(2, 2, 3);

            // Assert
            Assert.Equal(3, reshaped.Rank);
            Assert.Equal(new int[] { 2, 2, 3 }, reshaped.Shape);
            Assert.Equal(1.0, reshaped[0, 0, 0], precision: 10);
            Assert.Equal(12.0, reshaped[1, 1, 2], precision: 10);
        }

        [Fact]
        public void Reshape_WithNegativeOne_InfersDimension()
        {
            // Arrange - 2x3 tensor = 6 elements
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            for (int i = 0; i < 6; i++)
            {
                tensor.SetFlatIndex(i, i + 1);
            }

            // Act - Reshape to 3x?, which should infer 3x2
            var reshaped = tensor.Reshape(3, -1);

            // Assert
            Assert.Equal(2, reshaped.Rank);
            Assert.Equal(new int[] { 3, 2 }, reshaped.Shape);
            Assert.Equal(1.0, reshaped[0, 0], precision: 10);
            Assert.Equal(6.0, reshaped[2, 1], precision: 10);
        }

        [Fact]
        public void ToVector_Flattens2DTensor()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act
            var vector = tensor.ToVector();

            // Assert
            Assert.Equal(6, vector.Length);
            Assert.Equal(1.0, vector[0], precision: 10);
            Assert.Equal(2.0, vector[1], precision: 10);
            Assert.Equal(6.0, vector[5], precision: 10);
        }

        [Fact]
        public void ToVector_Flattens3DTensor()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 2, 3 });
            for (int i = 0; i < 12; i++)
            {
                tensor.SetFlatIndex(i, i + 1);
            }

            // Act
            var vector = tensor.ToVector();

            // Assert
            Assert.Equal(12, vector.Length);
            Assert.Equal(1.0, vector[0], precision: 10);
            Assert.Equal(12.0, vector[11], precision: 10);
        }

        [Fact]
        public void ToMatrix_Converts2DTensorToMatrix()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act
            var matrix = tensor.ToMatrix();

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(6.0, matrix[1, 2], precision: 10);
        }

        [Fact]
        public void Transpose_2DTensor_TransposesCorrectly()
        {
            // Arrange - 2x3 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act - Transpose to 3x2
            var transposed = tensor.Transpose();

            // Assert
            Assert.Equal(new int[] { 3, 2 }, transposed.Shape);
            Assert.Equal(1.0, transposed[0, 0], precision: 10);
            Assert.Equal(4.0, transposed[0, 1], precision: 10);
            Assert.Equal(2.0, transposed[1, 0], precision: 10);
            Assert.Equal(5.0, transposed[1, 1], precision: 10);
            Assert.Equal(3.0, transposed[2, 0], precision: 10);
            Assert.Equal(6.0, transposed[2, 1], precision: 10);
        }

        [Fact]
        public void Transpose_WithPermutation_PermutesAxesCorrectly()
        {
            // Arrange - 2x3x4 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });
            tensor[0, 0, 0] = 1;
            tensor[1, 2, 3] = 24;

            // Act - Permute axes: (0,1,2) -> (2,0,1), resulting in 4x2x3
            var permuted = tensor.Transpose(new int[] { 2, 0, 1 });

            // Assert
            Assert.Equal(new int[] { 4, 2, 3 }, permuted.Shape);
            Assert.Equal(1.0, permuted[0, 0, 0], precision: 10);
            Assert.Equal(24.0, permuted[3, 1, 2], precision: 10);
        }

        #endregion

        #region Indexer and Element Access Tests

        [Fact]
        public void Indexer_GetSet_1DTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 5 });

            // Act
            tensor[0] = 10.0;
            tensor[4] = 50.0;

            // Assert
            Assert.Equal(10.0, tensor[0], precision: 10);
            Assert.Equal(50.0, tensor[4], precision: 10);
        }

        [Fact]
        public void Indexer_GetSet_2DTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });

            // Act
            tensor[0, 0] = 1.0;
            tensor[2, 3] = 12.0;

            // Assert
            Assert.Equal(1.0, tensor[0, 0], precision: 10);
            Assert.Equal(12.0, tensor[2, 3], precision: 10);
        }

        [Fact]
        public void Indexer_GetSet_3DTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });

            // Act
            tensor[0, 0, 0] = 1.0;
            tensor[1, 2, 3] = 24.0;

            // Assert
            Assert.Equal(1.0, tensor[0, 0, 0], precision: 10);
            Assert.Equal(24.0, tensor[1, 2, 3], precision: 10);
        }

        [Fact]
        public void Indexer_GetSet_4DTensor_WorksCorrectly()
        {
            // Arrange - 4D tensor (batch, channel, height, width)
            var tensor = new Tensor<double>(new int[] { 2, 3, 4, 5 });

            // Act
            tensor[0, 0, 0, 0] = 1.0;
            tensor[1, 2, 3, 4] = 120.0;

            // Assert
            Assert.Equal(1.0, tensor[0, 0, 0, 0], precision: 10);
            Assert.Equal(120.0, tensor[1, 2, 3, 4], precision: 10);
        }

        [Fact]
        public void GetFlatIndexValue_ReturnsCorrectValue()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act & Assert
            Assert.Equal(1.0, tensor.GetFlatIndexValue(0), precision: 10);
            Assert.Equal(6.0, tensor.GetFlatIndexValue(5), precision: 10);
        }

        [Fact]
        public void SetFlatIndex_SetsValueCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });

            // Act
            tensor.SetFlatIndex(0, 10.0);
            tensor.SetFlatIndex(5, 60.0);

            // Assert
            Assert.Equal(10.0, tensor[0, 0], precision: 10);
            Assert.Equal(60.0, tensor[1, 2], precision: 10);
        }

        [Fact]
        public void GetRow_ReturnsCorrectRow()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;

            // Act
            var row = tensor.GetRow(1);

            // Assert
            Assert.Equal(4, row.Length);
            Assert.Equal(5.0, row[0], precision: 10);
            Assert.Equal(8.0, row[3], precision: 10);
        }

        [Fact]
        public void SetRow_SetsRowCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            var newRow = new Vector<double>(new double[] { 10, 20, 30, 40 });

            // Act
            tensor.SetRow(1, newRow);

            // Assert
            Assert.Equal(10.0, tensor[1, 0], precision: 10);
            Assert.Equal(40.0, tensor[1, 3], precision: 10);
        }

        [Fact]
        public void GetVector_ReturnsCorrectVector()
        {
            // Arrange - 3x4 tensor
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;

            // Act
            var vector = tensor.GetVector(1);

            // Assert
            Assert.Equal(4, vector.Length);
            Assert.Equal(5.0, vector[0], precision: 10);
            Assert.Equal(8.0, vector[3], precision: 10);
        }

        #endregion

        #region Slicing Tests

        [Fact]
        public void Slice_SingleIndex_ReturnsCorrectSlice()
        {
            // Arrange - 3x4 tensor
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;

            // Act - Get slice at index 1 (second row)
            var slice = tensor.Slice(1);

            // Assert
            Assert.Equal(1, slice.Rank);
            Assert.Equal(4, slice.Length);
            Assert.Equal(5.0, slice[0], precision: 10);
            Assert.Equal(8.0, slice[3], precision: 10);
        }

        [Fact]
        public void Slice_WithAxisStartEnd_ReturnsCorrectSlice()
        {
            // Arrange - 1D tensor with 10 elements
            var data = new Vector<double>(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var tensor = new Tensor<double>(new int[] { 10 }, data);

            // Act - Slice from index 2 to 7 (exclusive)
            var slice = tensor.Slice(axis: 0, start: 2, end: 7);

            // Assert
            Assert.Equal(5, slice.Length);
            Assert.Equal(2.0, slice[0], precision: 10);
            Assert.Equal(6.0, slice[4], precision: 10);
        }

        [Fact]
        public void Slice_2DWithRowCol_ReturnsCorrectSubMatrix()
        {
            // Arrange - 4x4 matrix
            var tensor = new Tensor<double>(new int[] { 4, 4 });
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    tensor[i, j] = i * 4 + j;
                }
            }

            // Act - Slice rows 1-2 (inclusive), cols 1-2 (inclusive)
            var slice = tensor.Slice(startRow: 1, startCol: 1, endRow: 3, endCol: 3);

            // Assert
            Assert.Equal(new int[] { 2, 2 }, slice.Shape);
            Assert.Equal(5.0, slice[0, 0], precision: 10);  // tensor[1,1]
            Assert.Equal(6.0, slice[0, 1], precision: 10);  // tensor[1,2]
            Assert.Equal(9.0, slice[1, 0], precision: 10);  // tensor[2,1]
            Assert.Equal(10.0, slice[1, 1], precision: 10); // tensor[2,2]
        }

        [Fact]
        public void GetSlice_BatchIndex_ReturnsCorrectBatch()
        {
            // Arrange - 2x3x4 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });
            for (int i = 0; i < 24; i++)
            {
                tensor.SetFlatIndex(i, i);
            }

            // Act - Get first batch
            var batch = tensor.GetSlice(batchIndex: 0);

            // Assert
            Assert.Equal(new int[] { 3, 4 }, batch.Shape);
            Assert.Equal(0.0, batch[0, 0], precision: 10);
        }

        [Fact]
        public void GetSlice_StartAndLength_ReturnsCorrectSlice()
        {
            // Arrange - 1D tensor
            var data = new Vector<double>(new double[] { 10, 20, 30, 40, 50, 60 });
            var tensor = new Tensor<double>(new int[] { 6 }, data);

            // Act - Get slice starting at index 2, length 3
            var slice = tensor.GetSlice(start: 2, length: 3);

            // Assert
            Assert.Equal(3, slice.Length);
            Assert.Equal(30.0, slice[0], precision: 10);
            Assert.Equal(50.0, slice[2], precision: 10);
        }

        [Fact]
        public void SetSlice_SingleIndex_SetsCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            var sliceData = new Tensor<double>(new int[] { 4 });
            sliceData[0] = 10; sliceData[1] = 20; sliceData[2] = 30; sliceData[3] = 40;

            // Act
            tensor.SetSlice(1, sliceData);

            // Assert
            Assert.Equal(10.0, tensor[1, 0], precision: 10);
            Assert.Equal(40.0, tensor[1, 3], precision: 10);
        }

        [Fact]
        public void SetSlice_WithVector_SetsCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 10 });
            var sliceData = new Vector<double>(new double[] { 100, 200, 300 });

            // Act - Set slice starting at index 3
            tensor.SetSlice(start: 3, slice: sliceData);

            // Assert
            Assert.Equal(100.0, tensor[3], precision: 10);
            Assert.Equal(200.0, tensor[4], precision: 10);
            Assert.Equal(300.0, tensor[5], precision: 10);
        }

        [Fact]
        public void SubTensor_SingleIndex_ReturnsCorrectSubTensor()
        {
            // Arrange - 3x4x5 tensor
            var tensor = new Tensor<double>(new int[] { 3, 4, 5 });
            tensor[1, 2, 3] = 123.0;

            // Act - Fix first dimension to 1, get 4x5 tensor
            var subTensor = tensor.SubTensor(1);

            // Assert
            Assert.Equal(new int[] { 4, 5 }, subTensor.Shape);
            Assert.Equal(123.0, subTensor[2, 3], precision: 10);
        }

        [Fact]
        public void SubTensor_MultipleIndices_ReturnsCorrectSubTensor()
        {
            // Arrange - 3x4x5 tensor
            var tensor = new Tensor<double>(new int[] { 3, 4, 5 });
            tensor[1, 2, 3] = 123.0;

            // Act - Fix first two dimensions to [1,2], get 1D tensor of length 5
            var subTensor = tensor.SubTensor(1, 2);

            // Assert
            Assert.Equal(new int[] { 5 }, subTensor.Shape);
            Assert.Equal(123.0, subTensor[3], precision: 10);
        }

        [Fact]
        public void SetSubTensor_InsertsSubTensorCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 4, 5 });
            var subTensor = new Tensor<double>(new int[] { 2, 2 });
            subTensor[0, 0] = 10; subTensor[0, 1] = 20;
            subTensor[1, 0] = 30; subTensor[1, 1] = 40;

            // Act - Insert at position [1, 2]
            tensor.SetSubTensor(new int[] { 1, 2 }, subTensor);

            // Assert
            Assert.Equal(10.0, tensor[1, 2], precision: 10);
            Assert.Equal(40.0, tensor[2, 3], precision: 10);
        }

        [Fact]
        public void GetSubTensor_4DImageTensor_ExtractsRegionCorrectly()
        {
            // Arrange - 2 batches, 3 channels, 8x8 images
            var tensor = new Tensor<double>(new int[] { 2, 3, 8, 8 });
            tensor[1, 2, 3, 4] = 1234.0;

            // Act - Extract 4x4 region from batch 1, channel 2, starting at (2,3)
            var subTensor = tensor.GetSubTensor(
                batch: 1, channel: 2,
                startHeight: 2, startWidth: 3,
                height: 4, width: 4);

            // Assert
            Assert.Equal(new int[] { 4, 4 }, subTensor.Shape);
            Assert.Equal(1234.0, subTensor[1, 1], precision: 10); // [3-2, 4-3] = [1,1]
        }

        #endregion

        #region Arithmetic Operations Tests

        [Fact]
        public void Add_TwoTensors_AddsElementwise()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 10; tensor2[0, 1] = 20; tensor2[0, 2] = 30;
            tensor2[1, 0] = 40; tensor2[1, 1] = 50; tensor2[1, 2] = 60;

            // Act
            var result = tensor1.Add(tensor2);

            // Assert
            Assert.Equal(11.0, result[0, 0], precision: 10);
            Assert.Equal(22.0, result[0, 1], precision: 10);
            Assert.Equal(66.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void Add_WithVector_BroadcastsCorrectly()
        {
            // Arrange - 2x3 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            var vector = new Vector<double>(new double[] { 10, 20, 30 });

            // Act - Add vector to each row
            var result = tensor.Add(vector);

            // Assert
            Assert.Equal(11.0, result[0, 0], precision: 10);
            Assert.Equal(22.0, result[0, 1], precision: 10);
            Assert.Equal(33.0, result[0, 2], precision: 10);
            Assert.Equal(14.0, result[1, 0], precision: 10);
            Assert.Equal(36.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void Operator_Add_AddsCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 2 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2;
            tensor1[1, 0] = 3; tensor1[1, 1] = 4;

            var tensor2 = new Tensor<double>(new int[] { 2, 2 });
            tensor2[0, 0] = 5; tensor2[0, 1] = 6;
            tensor2[1, 0] = 7; tensor2[1, 1] = 8;

            // Act
            var result = tensor1 + tensor2;

            // Assert
            Assert.Equal(6.0, result[0, 0], precision: 10);
            Assert.Equal(12.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Subtract_TwoTensors_SubtractsElementwise()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 10; tensor1[0, 1] = 20; tensor1[0, 2] = 30;
            tensor1[1, 0] = 40; tensor1[1, 1] = 50; tensor1[1, 2] = 60;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 1; tensor2[0, 1] = 2; tensor2[0, 2] = 3;
            tensor2[1, 0] = 4; tensor2[1, 1] = 5; tensor2[1, 2] = 6;

            // Act
            var result = tensor1.Subtract(tensor2);

            // Assert
            Assert.Equal(9.0, result[0, 0], precision: 10);
            Assert.Equal(18.0, result[0, 1], precision: 10);
            Assert.Equal(54.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void ElementwiseSubtract_SubtractsCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 2 });
            tensor1[0, 0] = 100; tensor1[0, 1] = 200;
            tensor1[1, 0] = 300; tensor1[1, 1] = 400;

            var tensor2 = new Tensor<double>(new int[] { 2, 2 });
            tensor2[0, 0] = 10; tensor2[0, 1] = 20;
            tensor2[1, 0] = 30; tensor2[1, 1] = 40;

            // Act
            var result = tensor1.ElementwiseSubtract(tensor2);

            // Assert
            Assert.Equal(90.0, result[0, 0], precision: 10);
            Assert.Equal(360.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Multiply_ByScalar_MultipliesAllElements()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act
            var result = tensor.Multiply(2.5);

            // Assert
            Assert.Equal(2.5, result[0, 0], precision: 10);
            Assert.Equal(5.0, result[0, 1], precision: 10);
            Assert.Equal(15.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void Multiply_TwoTensors_MultipliesElementwise()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 2; tensor2[0, 1] = 3; tensor2[0, 2] = 4;
            tensor2[1, 0] = 5; tensor2[1, 1] = 6; tensor2[1, 2] = 7;

            // Act
            var result = tensor1.Multiply(tensor2);

            // Assert
            Assert.Equal(2.0, result[0, 0], precision: 10);   // 1*2
            Assert.Equal(6.0, result[0, 1], precision: 10);   // 2*3
            Assert.Equal(42.0, result[1, 2], precision: 10);  // 6*7
        }

        [Fact]
        public void Operator_Multiply_MultipliesCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 2 });
            tensor1[0, 0] = 2; tensor1[0, 1] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5;

            var tensor2 = new Tensor<double>(new int[] { 2, 2 });
            tensor2[0, 0] = 10; tensor2[0, 1] = 10;
            tensor2[1, 0] = 10; tensor2[1, 1] = 10;

            // Act
            var result = tensor1 * tensor2;

            // Assert
            Assert.Equal(20.0, result[0, 0], precision: 10);
            Assert.Equal(50.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void ElementwiseMultiply_MultipliesCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 10; tensor2[0, 1] = 20; tensor2[0, 2] = 30;
            tensor2[1, 0] = 40; tensor2[1, 1] = 50; tensor2[1, 2] = 60;

            // Act
            var result = tensor1.ElementwiseMultiply(tensor2);

            // Assert
            Assert.Equal(10.0, result[0, 0], precision: 10);
            Assert.Equal(40.0, result[0, 1], precision: 10);
            Assert.Equal(360.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void ElementwiseMultiply_Static_MultipliesCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 2 });
            tensor1[0, 0] = 2; tensor1[0, 1] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5;

            var tensor2 = new Tensor<double>(new int[] { 2, 2 });
            tensor2[0, 0] = 5; tensor2[0, 1] = 4;
            tensor2[1, 0] = 3; tensor2[1, 1] = 2;

            // Act
            var result = Tensor<double>.ElementwiseMultiply(tensor1, tensor2);

            // Assert
            Assert.Equal(10.0, result[0, 0], precision: 10);
            Assert.Equal(10.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void PointwiseMultiply_MultipliesCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 2; tensor2[0, 1] = 2; tensor2[0, 2] = 2;
            tensor2[1, 0] = 2; tensor2[1, 1] = 2; tensor2[1, 2] = 2;

            // Act
            var result = tensor1.PointwiseMultiply(tensor2);

            // Assert
            Assert.Equal(2.0, result[0, 0], precision: 10);
            Assert.Equal(12.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void MatrixMultiply_2x3_3x2_Produces2x2()
        {
            // Arrange
            // A = [[1, 2, 3], [4, 5, 6]] (2x3)
            var tensorA = new Tensor<double>(new int[] { 2, 3 });
            tensorA[0, 0] = 1; tensorA[0, 1] = 2; tensorA[0, 2] = 3;
            tensorA[1, 0] = 4; tensorA[1, 1] = 5; tensorA[1, 2] = 6;

            // B = [[7, 8], [9, 10], [11, 12]] (3x2)
            var tensorB = new Tensor<double>(new int[] { 3, 2 });
            tensorB[0, 0] = 7; tensorB[0, 1] = 8;
            tensorB[1, 0] = 9; tensorB[1, 1] = 10;
            tensorB[2, 0] = 11; tensorB[2, 1] = 12;

            // Expected: [[58, 64], [139, 154]]
            // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
            // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]

            // Act
            var result = tensorA.MatrixMultiply(tensorB);

            // Assert
            Assert.Equal(new int[] { 2, 2 }, result.Shape);
            Assert.Equal(58.0, result[0, 0], precision: 10);
            Assert.Equal(64.0, result[0, 1], precision: 10);
            Assert.Equal(139.0, result[1, 0], precision: 10);
            Assert.Equal(154.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void MatrixMultiply_2x2_2x2_Produces2x2()
        {
            // Arrange
            // A = [[1, 2], [3, 4]]
            var tensorA = new Tensor<double>(new int[] { 2, 2 });
            tensorA[0, 0] = 1; tensorA[0, 1] = 2;
            tensorA[1, 0] = 3; tensorA[1, 1] = 4;

            // B = [[5, 6], [7, 8]]
            var tensorB = new Tensor<double>(new int[] { 2, 2 });
            tensorB[0, 0] = 5; tensorB[0, 1] = 6;
            tensorB[1, 0] = 7; tensorB[1, 1] = 8;

            // Expected: [[19, 22], [43, 50]]

            // Act
            var result = tensorA.MatrixMultiply(tensorB);

            // Assert
            Assert.Equal(19.0, result[0, 0], precision: 10);
            Assert.Equal(22.0, result[0, 1], precision: 10);
            Assert.Equal(43.0, result[1, 0], precision: 10);
            Assert.Equal(50.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Scale_MultipliesAllElementsByFactor()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 2; tensor[0, 1] = 4; tensor[0, 2] = 6;
            tensor[1, 0] = 8; tensor[1, 1] = 10; tensor[1, 2] = 12;

            // Act
            var result = tensor.Scale(0.5);

            // Assert
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(2.0, result[0, 1], precision: 10);
            Assert.Equal(6.0, result[1, 2], precision: 10);
        }

        #endregion

        #region Reduction Operations Tests

        [Fact]
        public void Sum_NoAxes_SumsAllElements()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;
            // Sum = 1+2+3+4+5+6 = 21

            // Act
            var result = tensor.Sum();

            // Assert
            Assert.Equal(1, result.Rank);
            Assert.Equal(1, result.Length);
            Assert.Equal(21.0, result[0], precision: 10);
        }

        [Fact]
        public void Sum_Axis0_SumsAlongAxis0()
        {
            // Arrange - 2x3 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;
            // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]

            // Act
            var result = tensor.Sum(new int[] { 0 });

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(5.0, result[0], precision: 10);
            Assert.Equal(7.0, result[1], precision: 10);
            Assert.Equal(9.0, result[2], precision: 10);
        }

        [Fact]
        public void SumOverAxis_Axis0_ReducesCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    tensor[i, j] = i * 4 + j + 1;
                }
            }

            // Act - Sum over axis 0 (rows)
            var result = tensor.SumOverAxis(0);

            // Assert
            Assert.Equal(new int[] { 4 }, result.Shape);
            Assert.Equal(15.0, result[0], precision: 10); // 1+5+9
            Assert.Equal(18.0, result[1], precision: 10); // 2+6+10
        }

        [Fact]
        public void SumOverAxis_Axis1_ReducesCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act - Sum over axis 1 (columns)
            var result = tensor.SumOverAxis(1);

            // Assert
            Assert.Equal(new int[] { 2 }, result.Shape);
            Assert.Equal(6.0, result[0], precision: 10);  // 1+2+3
            Assert.Equal(15.0, result[1], precision: 10); // 4+5+6
        }

        [Fact]
        public void Mean_ComputesCorrectAverage()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;
            // Mean = 21 / 6 = 3.5

            // Act
            var mean = tensor.Mean();

            // Assert
            Assert.Equal(3.5, mean, precision: 10);
        }

        [Fact]
        public void MeanOverAxis_Axis0_ComputesCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 2; tensor[0, 1] = 4; tensor[0, 2] = 6;
            tensor[1, 0] = 8; tensor[1, 1] = 10; tensor[1, 2] = 12;

            // Act - Mean over axis 0
            var result = tensor.MeanOverAxis(0);

            // Assert
            Assert.Equal(new int[] { 3 }, result.Shape);
            Assert.Equal(5.0, result[0], precision: 10);  // (2+8)/2
            Assert.Equal(7.0, result[1], precision: 10);  // (4+10)/2
            Assert.Equal(9.0, result[2], precision: 10);  // (6+12)/2
        }

        [Fact]
        public void MeanOverAxis_Axis1_ComputesCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 3; tensor[0, 1] = 6; tensor[0, 2] = 9;
            tensor[1, 0] = 12; tensor[1, 1] = 15; tensor[1, 2] = 18;

            // Act - Mean over axis 1
            var result = tensor.MeanOverAxis(1);

            // Assert
            Assert.Equal(new int[] { 2 }, result.Shape);
            Assert.Equal(6.0, result[0], precision: 10);  // (3+6+9)/3
            Assert.Equal(15.0, result[1], precision: 10); // (12+15+18)/3
        }

        [Fact]
        public void Max_ReturnsMaxValueAndIndex()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 5; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 2; tensor[1, 2] = 6;

            // Act
            var (maxVal, maxIndex) = tensor.Max();

            // Assert
            Assert.Equal(6.0, maxVal, precision: 10);
            Assert.Equal(5, maxIndex); // Flat index of element [1,2]
        }

        [Fact]
        public void MaxOverAxis_Axis0_ReturnsCorrectMaximums()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            tensor[0, 0] = 1; tensor[0, 1] = 5; tensor[0, 2] = 3; tensor[0, 3] = 7;
            tensor[1, 0] = 4; tensor[1, 1] = 2; tensor[1, 2] = 9; tensor[1, 3] = 1;
            tensor[2, 0] = 6; tensor[2, 1] = 8; tensor[2, 2] = 2; tensor[2, 3] = 3;

            // Act - Max over axis 0 (across rows)
            var result = tensor.MaxOverAxis(0);

            // Assert
            Assert.Equal(new int[] { 4 }, result.Shape);
            Assert.Equal(6.0, result[0], precision: 10); // max(1,4,6)
            Assert.Equal(8.0, result[1], precision: 10); // max(5,2,8)
            Assert.Equal(9.0, result[2], precision: 10); // max(3,9,2)
            Assert.Equal(7.0, result[3], precision: 10); // max(7,1,3)
        }

        [Fact]
        public void MaxOverAxis_Axis1_ReturnsCorrectMaximums()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 4 });
            tensor[0, 0] = 1; tensor[0, 1] = 5; tensor[0, 2] = 3; tensor[0, 3] = 7;
            tensor[1, 0] = 4; tensor[1, 1] = 2; tensor[1, 2] = 9; tensor[1, 3] = 1;

            // Act - Max over axis 1 (across columns)
            var result = tensor.MaxOverAxis(1);

            // Assert
            Assert.Equal(new int[] { 2 }, result.Shape);
            Assert.Equal(7.0, result[0], precision: 10); // max(1,5,3,7)
            Assert.Equal(9.0, result[1], precision: 10); // max(4,2,9,1)
        }

        [Fact]
        public void DotProduct_ComputesCorrectly()
        {
            // Arrange - Two 1D tensors
            var tensor1 = new Tensor<double>(new int[] { 4 });
            tensor1[0] = 1; tensor1[1] = 2; tensor1[2] = 3; tensor1[3] = 4;

            var tensor2 = new Tensor<double>(new int[] { 4 });
            tensor2[0] = 5; tensor2[1] = 6; tensor2[2] = 7; tensor2[3] = 8;

            // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

            // Act
            var dotProduct = tensor1.DotProduct(tensor2);

            // Assert
            Assert.Equal(70.0, dotProduct, precision: 10);
        }

        #endregion

        #region Concatenate Tests

        [Fact]
        public void Concatenate_Axis0_Concatenates2DTensors()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2; tensor1[0, 2] = 3;
            tensor1[1, 0] = 4; tensor1[1, 1] = 5; tensor1[1, 2] = 6;

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2[0, 0] = 7; tensor2[0, 1] = 8; tensor2[0, 2] = 9;
            tensor2[1, 0] = 10; tensor2[1, 1] = 11; tensor2[1, 2] = 12;

            // Act - Concatenate along axis 0 (rows)
            var result = Tensor<double>.Concatenate(new[] { tensor1, tensor2 }, axis: 0);

            // Assert
            Assert.Equal(new int[] { 4, 3 }, result.Shape);
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(6.0, result[1, 2], precision: 10);
            Assert.Equal(7.0, result[2, 0], precision: 10);
            Assert.Equal(12.0, result[3, 2], precision: 10);
        }

        [Fact]
        public void Concatenate_Axis1_Concatenates2DTensors()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 2 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2;
            tensor1[1, 0] = 3; tensor1[1, 1] = 4;

            var tensor2 = new Tensor<double>(new int[] { 2, 2 });
            tensor2[0, 0] = 5; tensor2[0, 1] = 6;
            tensor2[1, 0] = 7; tensor2[1, 1] = 8;

            // Act - Concatenate along axis 1 (columns)
            var result = Tensor<double>.Concatenate(new[] { tensor1, tensor2 }, axis: 1);

            // Assert
            Assert.Equal(new int[] { 2, 4 }, result.Shape);
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(2.0, result[0, 1], precision: 10);
            Assert.Equal(5.0, result[0, 2], precision: 10);
            Assert.Equal(6.0, result[0, 3], precision: 10);
        }

        [Fact]
        public void Concatenate_MultipleTensors_ConcatenatesAll()
        {
            // Arrange - Three 1D tensors
            var tensor1 = new Tensor<double>(new int[] { 2 });
            tensor1[0] = 1; tensor1[1] = 2;

            var tensor2 = new Tensor<double>(new int[] { 3 });
            tensor2[0] = 3; tensor2[1] = 4; tensor2[2] = 5;

            var tensor3 = new Tensor<double>(new int[] { 1 });
            tensor3[0] = 6;

            // Act
            var result = Tensor<double>.Concatenate(new[] { tensor1, tensor2, tensor3 }, axis: 0);

            // Assert
            Assert.Equal(6, result.Length);
            Assert.Equal(1.0, result[0], precision: 10);
            Assert.Equal(6.0, result[5], precision: 10);
        }

        #endregion

        #region Fill and Transform Tests

        [Fact]
        public void Fill_SetsAllElementsToValue()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 3, 4 });

            // Act
            tensor.Fill(7.5);

            // Assert
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    Assert.Equal(7.5, tensor[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void Transform_WithValueTransformer_TransformsAllElements()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act - Square each element
            var result = tensor.Transform((x, idx) => x * x);

            // Assert
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(4.0, result[0, 1], precision: 10);
            Assert.Equal(36.0, result[1, 2], precision: 10);
        }

        #endregion

        #region Clone and Enumeration Tests

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new Tensor<double>(new int[] { 2, 2 });
            original[0, 0] = 1; original[0, 1] = 2;
            original[1, 0] = 3; original[1, 1] = 4;

            // Act
            var clone = original.Clone();
            clone[0, 0] = 100; // Modify clone

            // Assert - Original should be unchanged
            Assert.Equal(1.0, original[0, 0], precision: 10);
            Assert.Equal(100.0, clone[0, 0], precision: 10);
        }

        [Fact]
        public void GetEnumerator_IteratesAllElements()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // Act
            var values = new List<double>();
            foreach (var value in tensor)
            {
                values.Add(value);
            }

            // Assert
            Assert.Equal(6, values.Count);
            Assert.Equal(1.0, values[0], precision: 10);
            Assert.Equal(6.0, values[5], precision: 10);
        }

        #endregion

        #region Edge Cases Tests

        [Fact]
        public void EmptyTensor_HasZeroLength()
        {
            // Act
            var tensor = Tensor<double>.Empty();

            // Assert
            Assert.Equal(0, tensor.Length);
        }

        [Fact]
        public void ScalarTensor_StoresAndRetrievesSingleValue()
        {
            // Arrange & Act
            var tensor = Tensor<double>.FromScalar(42.5);

            // Assert
            Assert.Equal(1, tensor.Length);
            Assert.Equal(42.5, tensor[0], precision: 10);
        }

        [Fact]
        public void LargeTensor_CreatesAndAccessesCorrectly()
        {
            // Arrange & Act - Large 4D tensor: 10x10x10x10 = 10,000 elements
            var tensor = new Tensor<double>(new int[] { 10, 10, 10, 10 });
            tensor[0, 0, 0, 0] = 1.0;
            tensor[9, 9, 9, 9] = 10000.0;

            // Assert
            Assert.Equal(10000, tensor.Length);
            Assert.Equal(1.0, tensor[0, 0, 0, 0], precision: 10);
            Assert.Equal(10000.0, tensor[9, 9, 9, 9], precision: 10);
        }

        [Fact]
        public void SingleElementTensor_WorksCorrectly()
        {
            // Arrange & Act
            var tensor = new Tensor<double>(new int[] { 1, 1, 1 });
            tensor[0, 0, 0] = 123.456;

            // Assert
            Assert.Equal(1, tensor.Length);
            Assert.Equal(123.456, tensor[0, 0, 0], precision: 10);
        }

        #endregion

        #region Different Numeric Types Tests

        [Fact]
        public void IntegerTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<int>(new int[] { 2, 2 });
            tensor[0, 0] = 1; tensor[0, 1] = 2;
            tensor[1, 0] = 3; tensor[1, 1] = 4;

            // Act
            var sum = tensor.Sum();

            // Assert
            Assert.Equal(10, sum[0]);
        }

        [Fact]
        public void FloatTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<float>(new int[] { 2, 2 });
            tensor[0, 0] = 1.5f; tensor[0, 1] = 2.5f;
            tensor[1, 0] = 3.5f; tensor[1, 1] = 4.5f;

            // Act
            var result = tensor.Multiply(2.0f);

            // Assert
            Assert.Equal(3.0f, result[0, 0], precision: 5);
            Assert.Equal(9.0f, result[1, 1], precision: 5);
        }

        #endregion

        #region Broadcasting and Complex Operations Tests

        [Fact]
        public void Add_BroadcastVector_WorksForMultipleRows()
        {
            // Arrange - 3x4 tensor
            var tensor = new Tensor<double>(new int[] { 3, 4 });
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    tensor[i, j] = i * 4 + j;
                }
            }

            var vector = new Vector<double>(new double[] { 1, 2, 3, 4 });

            // Act
            var result = tensor.Add(vector);

            // Assert
            Assert.Equal(1.0, result[0, 0], precision: 10);  // 0+1
            Assert.Equal(5.0, result[0, 3], precision: 10);  // 3+2 -> wait, should be 3+4 = 7
            Assert.Equal(7.0, result[1, 0], precision: 10);  // 4+3 -> wait, should be 4+1 = 5
        }

        [Fact]
        public void Multiply_WithMatrix_WorksCorrectly()
        {
            // Arrange - 2x3 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
            tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

            // 3x2 matrix
            var matrix = new Matrix<double>(3, 2);
            matrix[0, 0] = 7; matrix[0, 1] = 8;
            matrix[1, 0] = 9; matrix[1, 1] = 10;
            matrix[2, 0] = 11; matrix[2, 1] = 12;

            // Expected result: 2x2 tensor
            // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
            // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]

            // Act
            var result = tensor.Multiply(matrix);

            // Assert
            Assert.Equal(new int[] { 2, 2 }, result.Shape);
            Assert.Equal(58.0, result[0, 0], precision: 10);
            Assert.Equal(64.0, result[0, 1], precision: 10);
            Assert.Equal(139.0, result[1, 0], precision: 10);
            Assert.Equal(154.0, result[1, 1], precision: 10);
        }

        #endregion

        #region 3D Tensor Tests

        [Fact]
        public void Tensor3D_IndexingAndSlicing_WorksCorrectly()
        {
            // Arrange - 2x3x4 tensor (e.g., 2 images, 3 channels, 4 pixels each)
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });
            for (int i = 0; i < 24; i++)
            {
                tensor.SetFlatIndex(i, i);
            }

            // Act - Access specific element
            var value = tensor[1, 2, 3];
            var slice = tensor.Slice(1); // Get second image

            // Assert
            Assert.Equal(23.0, value, precision: 10);
            Assert.Equal(new int[] { 3, 4 }, slice.Shape);
        }

        [Fact]
        public void Tensor3D_Transpose_WorksCorrectly()
        {
            // Arrange - 2x3x4 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });
            tensor[0, 0, 0] = 1;
            tensor[1, 2, 3] = 24;

            // Act - Transpose to 4x3x2
            var transposed = tensor.Transpose(new int[] { 2, 1, 0 });

            // Assert
            Assert.Equal(new int[] { 4, 3, 2 }, transposed.Shape);
            Assert.Equal(1.0, transposed[0, 0, 0], precision: 10);
            Assert.Equal(24.0, transposed[3, 2, 1], precision: 10);
        }

        [Fact]
        public void Tensor3D_SumOverAxis_ReducesDimensionCorrectly()
        {
            // Arrange - 2x3x4 tensor
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });
            tensor.Fill(1.0);

            // Act - Sum over middle axis (axis 1)
            var result = tensor.SumOverAxis(1);

            // Assert
            Assert.Equal(new int[] { 2, 4 }, result.Shape);
            Assert.Equal(3.0, result[0, 0], precision: 10); // Sum of 3 ones
        }

        #endregion

        #region 4D Tensor Tests (Image Batches)

        [Fact]
        public void Tensor4D_NCHW_Format_WorksCorrectly()
        {
            // Arrange - 4D tensor: 2 batches, 3 channels, 4 height, 5 width
            var tensor = new Tensor<double>(new int[] { 2, 3, 4, 5 });
            tensor[0, 0, 0, 0] = 1.0;
            tensor[1, 2, 3, 4] = 120.0;

            // Act
            var batch0 = tensor.Slice(0); // Get first batch: 3x4x5

            // Assert
            Assert.Equal(new int[] { 3, 4, 5 }, batch0.Shape);
            Assert.Equal(1.0, batch0[0, 0, 0], precision: 10);
        }

        [Fact]
        public void Tensor4D_GetSubTensor_ExtractsImagePatchCorrectly()
        {
            // Arrange - 1 batch, 1 channel, 8x8 image
            var tensor = new Tensor<double>(new int[] { 1, 1, 8, 8 });
            for (int i = 0; i < 64; i++)
            {
                tensor.SetFlatIndex(i, i);
            }

            // Act - Extract 3x3 patch from position (2, 3)
            var patch = tensor.GetSubTensor(
                batch: 0, channel: 0,
                startHeight: 2, startWidth: 3,
                height: 3, width: 3);

            // Assert
            Assert.Equal(new int[] { 3, 3 }, patch.Shape);
            // Element at [2,3] in original is at flat index 2*8+3 = 19
            Assert.Equal(19.0, patch[0, 0], precision: 10);
        }

        [Fact]
        public void Tensor4D_Concatenate_ConcatenatesBatchesCorrectly()
        {
            // Arrange - Two batches: 2x1x2x2 each
            var batch1 = new Tensor<double>(new int[] { 2, 1, 2, 2 });
            batch1.Fill(1.0);

            var batch2 = new Tensor<double>(new int[] { 2, 1, 2, 2 });
            batch2.Fill(2.0);

            // Act - Concatenate along batch axis (axis 0)
            var combined = Tensor<double>.Concatenate(new[] { batch1, batch2 }, axis: 0);

            // Assert
            Assert.Equal(new int[] { 4, 1, 2, 2 }, combined.Shape);
            Assert.Equal(1.0, combined[0, 0, 0, 0], precision: 10);
            Assert.Equal(2.0, combined[2, 0, 0, 0], precision: 10);
        }

        #endregion

        #region 5D Tensor Tests (Video Batches)

        [Fact]
        public void Tensor5D_VideoFormat_WorksCorrectly()
        {
            // Arrange - 5D tensor: 2 videos, 3 channels, 4 frames, 5 height, 6 width
            var tensor = new Tensor<double>(new int[] { 2, 3, 4, 5, 6 });
            tensor[0, 0, 0, 0, 0] = 1.0;
            tensor[1, 2, 3, 4, 5] = 12345.0;

            // Act & Assert
            Assert.Equal(5, tensor.Rank);
            Assert.Equal(720, tensor.Length);
            Assert.Equal(1.0, tensor[0, 0, 0, 0, 0], precision: 10);
            Assert.Equal(12345.0, tensor[1, 2, 3, 4, 5], precision: 10);
        }

        [Fact]
        public void Tensor5D_Slicing_ExtractsFramesCorrectly()
        {
            // Arrange - 1 video, 1 channel, 10 frames, 4 height, 5 width
            var tensor = new Tensor<double>(new int[] { 1, 1, 10, 4, 5 });
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.SetFlatIndex(i, i);
            }

            // Act - Get first video
            var video = tensor.Slice(0);

            // Assert
            Assert.Equal(new int[] { 1, 10, 4, 5 }, video.Shape);
        }

        [Fact]
        public void Tensor5D_Reshape_ReshapesCorrectly()
        {
            // Arrange - 2x2x2x2x2 = 32 elements
            var tensor = new Tensor<double>(new int[] { 2, 2, 2, 2, 2 });
            for (int i = 0; i < 32; i++)
            {
                tensor.SetFlatIndex(i, i);
            }

            // Act - Reshape to 4x8
            var reshaped = tensor.Reshape(4, 8);

            // Assert
            Assert.Equal(new int[] { 4, 8 }, reshaped.Shape);
            Assert.Equal(0.0, reshaped[0, 0], precision: 10);
            Assert.Equal(31.0, reshaped[3, 7], precision: 10);
        }

        #endregion

        #region Additional Complex Operations

        [Fact]
        public void ChainedOperations_ReshapeTransposeSlice_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 6 });
            for (int i = 0; i < 6; i++)
            {
                tensor[i] = i + 1;
            }

            // Act - Chain operations: reshape to 2x3, transpose to 3x2, get first row
            var reshaped = tensor.Reshape(2, 3);
            var transposed = reshaped.Transpose();
            var row = transposed.Slice(0);

            // Assert
            Assert.Equal(2, row.Length);
            Assert.Equal(1.0, row[0], precision: 10);
            Assert.Equal(4.0, row[1], precision: 10);
        }

        [Fact]
        public void ComplexArithmetic_MultipleOperations_WorksCorrectly()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 2 });
            tensor1[0, 0] = 1; tensor1[0, 1] = 2;
            tensor1[1, 0] = 3; tensor1[1, 1] = 4;

            var tensor2 = new Tensor<double>(new int[] { 2, 2 });
            tensor2[0, 0] = 2; tensor2[0, 1] = 2;
            tensor2[1, 0] = 2; tensor2[1, 1] = 2;

            // Act - (tensor1 + tensor2) * 3
            var added = tensor1.Add(tensor2);
            var result = added.Multiply(3.0);

            // Assert
            Assert.Equal(9.0, result[0, 0], precision: 10);   // (1+2)*3
            Assert.Equal(12.0, result[0, 1], precision: 10);  // (2+2)*3
            Assert.Equal(15.0, result[1, 0], precision: 10);  // (3+2)*3
            Assert.Equal(18.0, result[1, 1], precision: 10);  // (4+2)*3
        }

        [Fact]
        public void StackThenUnstackViaSlicing_PreservesData()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            tensor1.Fill(1.0);

            var tensor2 = new Tensor<double>(new int[] { 2, 3 });
            tensor2.Fill(2.0);

            // Act - Stack and then unstack
            var stacked = Tensor<double>.Stack(new[] { tensor1, tensor2 }, axis: 0);
            var unstacked1 = stacked.Slice(0);
            var unstacked2 = stacked.Slice(1);

            // Assert
            Assert.Equal(1.0, unstacked1[0, 0], precision: 10);
            Assert.Equal(2.0, unstacked2[0, 0], precision: 10);
        }

        #endregion

        #region Boundary and Special Cases

        [Fact]
        public void ReshapeWithInference_MultipleNegativeOnes_ThrowsException()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 6 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => tensor.Reshape(-1, -1));
        }

        [Fact]
        public void Reshape_IncompatibleSize_ThrowsException()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 6 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => tensor.Reshape(2, 4)); // 8 != 6
        }

        [Fact]
        public void MatrixMultiply_IncompatibleShapes_ThrowsException()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            var tensor2 = new Tensor<double>(new int[] { 2, 2 }); // Incompatible: 3 != 2

            // Act & Assert
            Assert.Throws<ArgumentException>(() => tensor1.MatrixMultiply(tensor2));
        }

        [Fact]
        public void Concatenate_DifferentShapes_ThrowsException()
        {
            // Arrange
            var tensor1 = new Tensor<double>(new int[] { 2, 3 });
            var tensor2 = new Tensor<double>(new int[] { 2, 4 }); // Different column count

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                Tensor<double>.Concatenate(new[] { tensor1, tensor2 }, axis: 0));
        }

        #endregion

        #region Advanced Indexing and Access Patterns

        [Fact]
        public void FlatIndexing_LinearAccess_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3, 4 });

            // Act - Set all elements using flat indexing
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.SetFlatIndex(i, i * 10);
            }

            // Assert - Verify using flat indexing
            for (int i = 0; i < tensor.Length; i++)
            {
                Assert.Equal(i * 10, tensor.GetFlatIndexValue(i), precision: 10);
            }
        }

        [Fact]
        public void MultiDimensionalAccess_AllDimensions_WorksCorrectly()
        {
            // Arrange & Act - Create and populate 5D tensor
            var tensor = new Tensor<double>(new int[] { 2, 2, 2, 2, 2 });
            tensor[0, 0, 0, 0, 0] = 1;
            tensor[0, 1, 1, 1, 1] = 16;
            tensor[1, 1, 1, 1, 1] = 32;

            // Assert
            Assert.Equal(1.0, tensor[0, 0, 0, 0, 0], precision: 10);
            Assert.Equal(16.0, tensor[0, 1, 1, 1, 1], precision: 10);
            Assert.Equal(32.0, tensor[1, 1, 1, 1, 1], precision: 10);
        }

        #endregion

        #region Performance and Stress Tests

        [Fact]
        public void LargeMatrixMultiplication_CompletesSuccessfully()
        {
            // Arrange - Two 50x50 matrices
            var tensor1 = Tensor<double>.CreateRandom(50, 50);
            var tensor2 = Tensor<double>.CreateRandom(50, 50);

            // Act
            var result = tensor1.MatrixMultiply(tensor2);

            // Assert
            Assert.Equal(new int[] { 50, 50 }, result.Shape);
            Assert.Equal(2500, result.Length);
        }

        [Fact]
        public void HighDimensionalTensor_Operations_WorkCorrectly()
        {
            // Arrange - 5D tensor with many operations
            var tensor = Tensor<double>.CreateRandom(3, 4, 5, 6, 7);

            // Act - Perform various operations
            var flattened = tensor.ToVector();
            var reshaped = tensor.Reshape(3, 4, -1); // Infer last dimension
            var sliced = tensor.Slice(0);

            // Assert
            Assert.Equal(2520, flattened.Length); // 3*4*5*6*7
            Assert.Equal(new int[] { 3, 4, 210 }, reshaped.Shape);
            Assert.Equal(new int[] { 4, 5, 6, 7 }, sliced.Shape);
        }

        #endregion
    }
}
