using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class SerializationHelperTests
    {
        [Fact]
        public void SerializeNode_WithNullNode_WritesCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeNode(null, writer);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.False(reader.ReadBoolean());
        }

        [Fact]
        public void SerializeNode_WithLeafNode_WritesCorrectly()
        {
            // Arrange
            var node = new DecisionTreeNode<double>
            {
                IsLeaf = true,
                Prediction = 42.0
            };
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeNode(node, writer);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.True(reader.ReadBoolean()); // node exists
            Assert.True(reader.ReadBoolean()); // is leaf
            Assert.Equal(42.0, reader.ReadDouble());
        }

        [Fact]
        public void DeserializeNode_WithNullNode_ReturnsNull()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(false); // null node
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.DeserializeNode(reader);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void DeserializeNode_WithLeafNode_ReturnsCorrectNode()
        {
            // Arrange
            var original = new DecisionTreeNode<double>
            {
                IsLeaf = true,
                Prediction = 3.14
            };
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeNode(original, writer);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.DeserializeNode(reader);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IsLeaf);
            Assert.Equal(3.14, result.Prediction);
        }

        [Fact]
        public void SerializeDeserializeNode_WithComplexTree_PreservesStructure()
        {
            // Arrange
            var root = new DecisionTreeNode<double>
            {
                IsLeaf = false,
                FeatureIndex = 0,
                SplitValue = 5.0,
                Left = new DecisionTreeNode<double> { IsLeaf = true, Prediction = 1.0 },
                Right = new DecisionTreeNode<double> { IsLeaf = true, Prediction = 2.0 }
            };
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeNode(root, writer);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.DeserializeNode(reader);

            // Assert
            Assert.NotNull(result);
            Assert.False(result.IsLeaf);
            Assert.Equal(0, result.FeatureIndex);
            Assert.Equal(5.0, result.SplitValue);
            Assert.NotNull(result.Left);
            Assert.True(result.Left.IsLeaf);
            Assert.Equal(1.0, result.Left.Prediction);
            Assert.NotNull(result.Right);
            Assert.True(result.Right.IsLeaf);
            Assert.Equal(2.0, result.Right.Prediction);
        }

        [Fact]
        public void WriteValue_WithDouble_WritesCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.WriteValue(writer, 123.456);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(123.456, reader.ReadDouble());
        }

        [Fact]
        public void ReadValue_WithDouble_ReadsCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(789.012);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.ReadValue(reader);

            // Assert
            Assert.Equal(789.012, result);
        }

        [Fact]
        public void WriteValue_WithFloat_WritesCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<float>.WriteValue(writer, 12.34f);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(12.34f, reader.ReadSingle());
        }

        [Fact]
        public void ReadValue_WithFloat_ReadsCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(56.78f);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<float>.ReadValue(reader);

            // Assert
            Assert.Equal(56.78f, result);
        }

        [Fact]
        public void WriteValue_WithInt_WritesCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<int>.WriteValue(writer, 42);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(42, reader.ReadInt32());
        }

        [Fact]
        public void ReadValue_WithInt_ReadsCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(99);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<int>.ReadValue(reader);

            // Assert
            Assert.Equal(99, result);
        }

        [Fact]
        public void SerializeMatrix_WithSmallMatrix_SerializesCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0;
            matrix[0, 1] = 2.0;
            matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0;
            matrix[1, 1] = 5.0;
            matrix[1, 2] = 6.0;
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeMatrix(writer, matrix);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(2, reader.ReadInt32()); // rows
            Assert.Equal(3, reader.ReadInt32()); // columns
        }

        [Fact]
        public void DeserializeMatrix_WithValidData_ReturnsCorrectMatrix()
        {
            // Arrange
            var original = new Matrix<double>(2, 2);
            original[0, 0] = 1.0;
            original[0, 1] = 2.0;
            original[1, 0] = 3.0;
            original[1, 1] = 4.0;
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeMatrix(writer, original);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.DeserializeMatrix(reader, 2, 2);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns);
            Assert.Equal(1.0, result[0, 0]);
            Assert.Equal(2.0, result[0, 1]);
            Assert.Equal(3.0, result[1, 0]);
            Assert.Equal(4.0, result[1, 1]);
        }

        [Fact]
        public void DeserializeMatrix_WithWrongDimensions_ThrowsException()
        {
            // Arrange
            var original = new Matrix<double>(2, 2);
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeMatrix(writer, original);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                SerializationHelper<double>.DeserializeMatrix(reader, 3, 3));
        }

        [Fact]
        public void SerializeMatrix_ToByteArray_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0;
            matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0;
            matrix[1, 1] = 4.0;

            // Act
            var bytes = SerializationHelper<double>.SerializeMatrix(matrix);

            // Assert
            Assert.NotNull(bytes);
            Assert.NotEmpty(bytes);
        }

        [Fact]
        public void DeserializeMatrix_FromByteArray_ReturnsCorrectMatrix()
        {
            // Arrange
            var original = new Matrix<double>(3, 2);
            original[0, 0] = 1.0;
            original[0, 1] = 2.0;
            original[1, 0] = 3.0;
            original[1, 1] = 4.0;
            original[2, 0] = 5.0;
            original[2, 1] = 6.0;
            var bytes = SerializationHelper<double>.SerializeMatrix(original);

            // Act
            var result = SerializationHelper<double>.DeserializeMatrix(bytes);

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(2, result.Columns);
            Assert.Equal(1.0, result[0, 0]);
            Assert.Equal(2.0, result[0, 1]);
            Assert.Equal(3.0, result[1, 0]);
            Assert.Equal(4.0, result[1, 1]);
            Assert.Equal(5.0, result[2, 0]);
            Assert.Equal(6.0, result[2, 1]);
        }

        [Fact]
        public void SerializeVector_WithSmallVector_SerializesCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeVector(writer, vector);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(3, reader.ReadInt32()); // length
        }

        [Fact]
        public void DeserializeVector_WithValidData_ReturnsCorrectVector()
        {
            // Arrange
            var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeVector(writer, original);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.DeserializeVector(reader, 4);

            // Assert
            Assert.Equal(4, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
            Assert.Equal(4.0, result[3]);
        }

        [Fact]
        public void DeserializeVector_WithWrongLength_ThrowsException()
        {
            // Arrange
            var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeVector(writer, original);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                SerializationHelper<double>.DeserializeVector(reader, 5));
        }

        [Fact]
        public void SerializeVector_ToByteArray_WorksCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var bytes = SerializationHelper<double>.SerializeVector(vector);

            // Assert
            Assert.NotNull(bytes);
            Assert.NotEmpty(bytes);
        }

        [Fact]
        public void DeserializeVector_FromByteArray_ReturnsCorrectVector()
        {
            // Arrange
            var original = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
            var bytes = SerializationHelper<float>.SerializeVector(original);

            // Act
            var result = SerializationHelper<float>.DeserializeVector(bytes);

            // Assert
            Assert.Equal(5, result.Length);
            Assert.Equal(1.0f, result[0]);
            Assert.Equal(2.0f, result[1]);
            Assert.Equal(3.0f, result[2]);
            Assert.Equal(4.0f, result[3]);
            Assert.Equal(5.0f, result[4]);
        }

        [Fact]
        public void SerializeTensor_WithSmallTensor_SerializesCorrectly()
        {
            // Arrange - use 1D tensor for simple index access
            var tensor = new Tensor<double>(new int[] { 6 });
            tensor[0] = 1.0;
            tensor[1] = 2.0;
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeTensor(writer, tensor);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(1, reader.ReadInt32()); // rank (1D tensor)
        }

        [Fact]
        public void DeserializeTensor_WithValidData_ReturnsCorrectTensor()
        {
            // Arrange - use 1D tensor for simple index access
            var original = new Tensor<double>(new int[] { 4 });
            original[0] = 1.0;
            original[1] = 2.0;
            original[2] = 3.0;
            original[3] = 4.0;
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<double>.SerializeTensor(writer, original);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<double>.DeserializeTensor(reader);

            // Assert
            Assert.Equal(1, result.Rank);
            Assert.Equal(4, result.Shape[0]);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
            Assert.Equal(4.0, result[3]);
        }

        [Fact]
        public void SerializeInterface_WithNullInstance_WritesEmptyString()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeInterface<object>(writer, null);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(string.Empty, reader.ReadString());
        }

        [Fact]
        public void SerializeInterface_WithNonNullInstance_WritesTypeName()
        {
            // Arrange
            var instance = "test string";
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<double>.SerializeInterface<object>(writer, instance);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            var typeName = reader.ReadString();
            Assert.Contains("String", typeName);
        }

        [Fact]
        public void SerializeDeserialize_Matrix_RoundTrip_PreservesData()
        {
            // Arrange
            var original = new Matrix<int>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    original[i, j] = i * 3 + j;

            // Act
            var bytes = SerializationHelper<int>.SerializeMatrix(original);
            var result = SerializationHelper<int>.DeserializeMatrix(bytes);

            // Assert
            Assert.Equal(original.Rows, result.Rows);
            Assert.Equal(original.Columns, result.Columns);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(original[i, j], result[i, j]);
        }

        [Fact]
        public void SerializeDeserialize_Vector_RoundTrip_PreservesData()
        {
            // Arrange
            var original = new Vector<double>(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 });

            // Act
            var bytes = SerializationHelper<double>.SerializeVector(original);
            var result = SerializationHelper<double>.DeserializeVector(bytes);

            // Assert
            Assert.Equal(original.Length, result.Length);
            for (int i = 0; i < original.Length; i++)
                Assert.Equal(original[i], result[i]);
        }

        [Fact]
        public void SerializeDeserialize_Tensor_RoundTrip_PreservesData()
        {
            // Arrange - use 1D tensor for simple index access
            var original = new Tensor<float>(new int[] { 12 }); // 2*3*2 = 12 elements in 1D
            for (int i = 0; i < original.Length; i++)
                original[i] = i * 1.5f;
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            SerializationHelper<float>.SerializeTensor(writer, original);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<float>.DeserializeTensor(reader);

            // Assert
            Assert.Equal(original.Rank, result.Rank);
            Assert.Equal(original.Length, result.Length);
            for (int i = 0; i < original.Length; i++)
                Assert.Equal(original[i], result[i]);
        }

        [Fact]
        public void WriteValue_WithDecimal_WritesCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<decimal>.WriteValue(writer, 123.456m);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(123.456m, reader.ReadDecimal());
        }

        [Fact]
        public void ReadValue_WithDecimal_ReadsCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(789.012m);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<decimal>.ReadValue(reader);

            // Assert
            Assert.Equal(789.012m, result);
        }

        [Fact]
        public void WriteValue_WithLong_WritesCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Act
            SerializationHelper<long>.WriteValue(writer, 123456789L);

            // Assert
            ms.Position = 0;
            using var reader = new BinaryReader(ms);
            Assert.Equal(123456789L, reader.ReadInt64());
        }

        [Fact]
        public void ReadValue_WithLong_ReadsCorrectly()
        {
            // Arrange
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(987654321L);
            ms.Position = 0;
            using var reader = new BinaryReader(ms);

            // Act
            var result = SerializationHelper<long>.ReadValue(reader);

            // Assert
            Assert.Equal(987654321L, result);
        }
    }
}
