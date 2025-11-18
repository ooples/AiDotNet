using AiDotNet.LinearAlgebra;
using AiDotNet.Serialization;
using AiDotNet.Validation;
using AiDotNet.NeuralNetworks;
using AiDotNet.Enums;
using AiDotNet.Exceptions;
using Newtonsoft.Json;
using Xunit;
using System.IO;

namespace AiDotNetTests.IntegrationTests.SerializationAndValidation
{
    /// <summary>
    /// Comprehensive integration tests for Serialization and Validation utilities.
    /// Tests verify JSON serialization/deserialization and validation methods.
    /// </summary>
    public class SerializationAndValidationIntegrationTests
    {
        private const double Tolerance = 1e-10;

        #region MatrixJsonConverter Tests

        [Fact]
        public void MatrixJsonConverter_SmallMatrix_RoundTripSerializationPreservesValues()
        {
            // Arrange
            var original = new Matrix<double>(3, 3);
            original[0, 0] = 1.5; original[0, 1] = 2.3; original[0, 2] = 3.7;
            original[1, 0] = 4.1; original[1, 1] = 5.9; original[1, 2] = 6.2;
            original[2, 0] = 7.8; original[2, 1] = 8.4; original[2, 2] = 9.6;

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(original.Rows, deserialized.Rows);
            Assert.Equal(original.Columns, deserialized.Columns);
            for (int i = 0; i < original.Rows; i++)
            {
                for (int j = 0; j < original.Columns; j++)
                {
                    Assert.Equal(original[i, j], deserialized[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void MatrixJsonConverter_MediumMatrix100x100_RoundTripSucceeds()
        {
            // Arrange
            var original = new Matrix<double>(100, 100);
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    original[i, j] = i * 100 + j + 0.5;
                }
            }

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(100, deserialized.Rows);
            Assert.Equal(100, deserialized.Columns);
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.Equal(original[i, j], deserialized[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void MatrixJsonConverter_WithNaNValues_PreservesNaN()
        {
            // Arrange
            var original = new Matrix<double>(2, 2);
            original[0, 0] = double.NaN;
            original[0, 1] = 2.0;
            original[1, 0] = 3.0;
            original[1, 1] = double.NaN;

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.True(double.IsNaN(deserialized[0, 0]));
            Assert.Equal(2.0, deserialized[0, 1], precision: 10);
            Assert.Equal(3.0, deserialized[1, 0], precision: 10);
            Assert.True(double.IsNaN(deserialized[1, 1]));
        }

        [Fact]
        public void MatrixJsonConverter_WithInfinityValues_PreservesInfinity()
        {
            // Arrange
            var original = new Matrix<double>(2, 3);
            original[0, 0] = double.PositiveInfinity;
            original[0, 1] = 1.5;
            original[0, 2] = double.NegativeInfinity;
            original[1, 0] = 2.5;
            original[1, 1] = double.PositiveInfinity;
            original[1, 2] = 3.5;

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.True(double.IsPositiveInfinity(deserialized[0, 0]));
            Assert.Equal(1.5, deserialized[0, 1], precision: 10);
            Assert.True(double.IsNegativeInfinity(deserialized[0, 2]));
            Assert.Equal(2.5, deserialized[1, 0], precision: 10);
            Assert.True(double.IsPositiveInfinity(deserialized[1, 1]));
            Assert.Equal(3.5, deserialized[1, 2], precision: 10);
        }

        [Fact]
        public void MatrixJsonConverter_WithZeros_PreservesZeros()
        {
            // Arrange
            var original = new Matrix<double>(3, 3);
            // All elements default to zero

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(0.0, deserialized[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void MatrixJsonConverter_SingleElement_RoundTripSucceeds()
        {
            // Arrange
            var original = new Matrix<double>(1, 1);
            original[0, 0] = 42.0;

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(1, deserialized.Rows);
            Assert.Equal(1, deserialized.Columns);
            Assert.Equal(42.0, deserialized[0, 0], precision: 10);
        }

        [Fact]
        public void MatrixJsonConverter_FloatType_RoundTripSucceeds()
        {
            // Arrange
            var original = new Matrix<float>(2, 2);
            original[0, 0] = 1.5f; original[0, 1] = 2.5f;
            original[1, 0] = 3.5f; original[1, 1] = 4.5f;

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<float>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(1.5f, deserialized[0, 0], precision: 6);
            Assert.Equal(2.5f, deserialized[0, 1], precision: 6);
            Assert.Equal(3.5f, deserialized[1, 0], precision: 6);
            Assert.Equal(4.5f, deserialized[1, 1], precision: 6);
        }

        [Fact]
        public void MatrixJsonConverter_IntType_RoundTripSucceeds()
        {
            // Arrange
            var original = new Matrix<int>(2, 2);
            original[0, 0] = 1; original[0, 1] = 2;
            original[1, 0] = 3; original[1, 1] = 4;

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<int>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(1, deserialized[0, 0]);
            Assert.Equal(2, deserialized[0, 1]);
            Assert.Equal(3, deserialized[1, 0]);
            Assert.Equal(4, deserialized[1, 1]);
        }

        [Fact]
        public void MatrixJsonConverter_NullMatrix_SerializesToNull()
        {
            // Arrange
            Matrix<double>? original = null;
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.Equal("null", json);
            Assert.Null(deserialized);
        }

        [Fact]
        public void MatrixJsonConverter_JsonFormat_IsValidAndReadable()
        {
            // Arrange
            var original = new Matrix<double>(2, 2);
            original[0, 0] = 1.0; original[0, 1] = 2.0;
            original[1, 0] = 3.0; original[1, 1] = 4.0;

            var settings = new JsonSerializerSettings { Formatting = Formatting.Indented };
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);

            // Assert
            Assert.Contains("\"rows\"", json);
            Assert.Contains("\"columns\"", json);
            Assert.Contains("\"data\"", json);
            Assert.Contains("2", json); // rows value
        }

        #endregion

        #region VectorJsonConverter Tests

        [Fact]
        public void VectorJsonConverter_SmallVector_RoundTripSerializationPreservesValues()
        {
            // Arrange
            var original = new Vector<double>(new[] { 1.5, 2.3, 3.7, 4.1, 5.9 });
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(original.Length, deserialized.Length);
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], deserialized[i], precision: 10);
            }
        }

        [Fact]
        public void VectorJsonConverter_LargeVector1000Elements_RoundTripSucceeds()
        {
            // Arrange
            var values = new double[1000];
            for (int i = 0; i < 1000; i++)
            {
                values[i] = i * 0.5 + 1.0;
            }
            var original = new Vector<double>(values);
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(1000, deserialized.Length);
            for (int i = 0; i < 1000; i++)
            {
                Assert.Equal(original[i], deserialized[i], precision: 10);
            }
        }

        [Fact]
        public void VectorJsonConverter_WithNaNValues_PreservesNaN()
        {
            // Arrange
            var original = new Vector<double>(new[] { double.NaN, 2.0, 3.0, double.NaN, 5.0 });
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.True(double.IsNaN(deserialized[0]));
            Assert.Equal(2.0, deserialized[1], precision: 10);
            Assert.Equal(3.0, deserialized[2], precision: 10);
            Assert.True(double.IsNaN(deserialized[3]));
            Assert.Equal(5.0, deserialized[4], precision: 10);
        }

        [Fact]
        public void VectorJsonConverter_WithInfinityValues_PreservesInfinity()
        {
            // Arrange
            var original = new Vector<double>(new[] {
                double.PositiveInfinity,
                2.0,
                double.NegativeInfinity,
                4.0
            });
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.True(double.IsPositiveInfinity(deserialized[0]));
            Assert.Equal(2.0, deserialized[1], precision: 10);
            Assert.True(double.IsNegativeInfinity(deserialized[2]));
            Assert.Equal(4.0, deserialized[3], precision: 10);
        }

        [Fact]
        public void VectorJsonConverter_SingleElement_RoundTripSucceeds()
        {
            // Arrange
            var original = new Vector<double>(new[] { 42.0 });
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(1, deserialized.Length);
            Assert.Equal(42.0, deserialized[0], precision: 10);
        }

        [Fact]
        public void VectorJsonConverter_FloatType_RoundTripSucceeds()
        {
            // Arrange
            var original = new Vector<float>(new[] { 1.5f, 2.5f, 3.5f });
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<float>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(1.5f, deserialized[0], precision: 6);
            Assert.Equal(2.5f, deserialized[1], precision: 6);
            Assert.Equal(3.5f, deserialized[2], precision: 6);
        }

        [Fact]
        public void VectorJsonConverter_NullVector_SerializesToNull()
        {
            // Arrange
            Vector<double>? original = null;
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

            // Assert
            Assert.Equal("null", json);
            Assert.Null(deserialized);
        }

        [Fact]
        public void VectorJsonConverter_JsonFormat_IsValidAndReadable()
        {
            // Arrange
            var original = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var settings = new JsonSerializerSettings { Formatting = Formatting.Indented };
            settings.Converters.Add(new VectorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);

            // Assert
            Assert.Contains("\"length\"", json);
            Assert.Contains("\"data\"", json);
            Assert.Contains("3", json); // length value
        }

        #endregion

        #region TensorJsonConverter Tests

        [Fact]
        public void TensorJsonConverter_1DTensor_RoundTripSerializationPreservesValues()
        {
            // Arrange
            var original = new Tensor<double>(new[] { 5 });
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            original = new Tensor<double>(new[] { 5 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(original.Shape, deserialized.Shape);
            Assert.Equal(original.Length, deserialized.Length);
            var originalArray = original.ToArray();
            var deserializedArray = deserialized.ToArray();
            for (int i = 0; i < originalArray.Length; i++)
            {
                Assert.Equal(originalArray[i], deserializedArray[i], precision: 10);
            }
        }

        [Fact]
        public void TensorJsonConverter_2DTensor_RoundTripSucceeds()
        {
            // Arrange
            var original = new Tensor<double>(new[] { 3, 4 });
            var data = new Vector<double>(new double[] {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0
            });
            original = new Tensor<double>(new[] { 3, 4 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(new[] { 3, 4 }, deserialized.Shape);
            var originalArray = original.ToArray();
            var deserializedArray = deserialized.ToArray();
            for (int i = 0; i < originalArray.Length; i++)
            {
                Assert.Equal(originalArray[i], deserializedArray[i], precision: 10);
            }
        }

        [Fact]
        public void TensorJsonConverter_3DTensor_RoundTripSucceeds()
        {
            // Arrange - 2x3x4 tensor (24 elements)
            var values = new double[24];
            for (int i = 0; i < 24; i++)
            {
                values[i] = i + 1.0;
            }
            var data = new Vector<double>(values);
            var original = new Tensor<double>(new[] { 2, 3, 4 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(new[] { 2, 3, 4 }, deserialized.Shape);
            var originalArray = original.ToArray();
            var deserializedArray = deserialized.ToArray();
            for (int i = 0; i < originalArray.Length; i++)
            {
                Assert.Equal(originalArray[i], deserializedArray[i], precision: 10);
            }
        }

        [Fact]
        public void TensorJsonConverter_4DTensor_RoundTripSucceeds()
        {
            // Arrange - 2x2x2x2 tensor (16 elements)
            var values = new double[16];
            for (int i = 0; i < 16; i++)
            {
                values[i] = i * 0.5;
            }
            var data = new Vector<double>(values);
            var original = new Tensor<double>(new[] { 2, 2, 2, 2 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(new[] { 2, 2, 2, 2 }, deserialized.Shape);
            var originalArray = original.ToArray();
            var deserializedArray = deserialized.ToArray();
            for (int i = 0; i < originalArray.Length; i++)
            {
                Assert.Equal(originalArray[i], deserializedArray[i], precision: 10);
            }
        }

        [Fact]
        public void TensorJsonConverter_5DTensor_RoundTripSucceeds()
        {
            // Arrange - 2x2x2x2x2 tensor (32 elements)
            var values = new double[32];
            for (int i = 0; i < 32; i++)
            {
                values[i] = i + 0.1;
            }
            var data = new Vector<double>(values);
            var original = new Tensor<double>(new[] { 2, 2, 2, 2, 2 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(new[] { 2, 2, 2, 2, 2 }, deserialized.Shape);
            var originalArray = original.ToArray();
            var deserializedArray = deserialized.ToArray();
            for (int i = 0; i < originalArray.Length; i++)
            {
                Assert.Equal(originalArray[i], deserializedArray[i], precision: 10);
            }
        }

        [Fact]
        public void TensorJsonConverter_WithNaNValues_PreservesNaN()
        {
            // Arrange
            var data = new Vector<double>(new[] { double.NaN, 2.0, double.NaN, 4.0 });
            var original = new Tensor<double>(new[] { 2, 2 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            var array = deserialized.ToArray();
            Assert.True(double.IsNaN(array[0]));
            Assert.Equal(2.0, array[1], precision: 10);
            Assert.True(double.IsNaN(array[2]));
            Assert.Equal(4.0, array[3], precision: 10);
        }

        [Fact]
        public void TensorJsonConverter_WithInfinityValues_PreservesInfinity()
        {
            // Arrange
            var data = new Vector<double>(new[] {
                double.PositiveInfinity,
                2.0,
                double.NegativeInfinity,
                4.0
            });
            var original = new Tensor<double>(new[] { 2, 2 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            var array = deserialized.ToArray();
            Assert.True(double.IsPositiveInfinity(array[0]));
            Assert.Equal(2.0, array[1], precision: 10);
            Assert.True(double.IsNegativeInfinity(array[2]));
            Assert.Equal(4.0, array[3], precision: 10);
        }

        [Fact]
        public void TensorJsonConverter_SingleElement_RoundTripSucceeds()
        {
            // Arrange
            var data = new Vector<double>(new[] { 42.0 });
            var original = new Tensor<double>(new[] { 1 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(new[] { 1 }, deserialized.Shape);
            Assert.Equal(42.0, deserialized.ToArray()[0], precision: 10);
        }

        [Fact]
        public void TensorJsonConverter_FloatType_RoundTripSucceeds()
        {
            // Arrange
            var data = new Vector<float>(new[] { 1.5f, 2.5f, 3.5f, 4.5f });
            var original = new Tensor<float>(new[] { 2, 2 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<float>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            var array = deserialized.ToArray();
            Assert.Equal(1.5f, array[0], precision: 6);
            Assert.Equal(2.5f, array[1], precision: 6);
            Assert.Equal(3.5f, array[2], precision: 6);
            Assert.Equal(4.5f, array[3], precision: 6);
        }

        [Fact]
        public void TensorJsonConverter_NullTensor_SerializesToNull()
        {
            // Arrange
            Tensor<double>? original = null;
            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.Equal("null", json);
            Assert.Null(deserialized);
        }

        [Fact]
        public void TensorJsonConverter_JsonFormat_IsValidAndReadable()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var original = new Tensor<double>(new[] { 2, 2 }, data);

            var settings = new JsonSerializerSettings { Formatting = Formatting.Indented };
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);

            // Assert
            Assert.Contains("\"shape\"", json);
            Assert.Contains("\"data\"", json);
        }

        [Fact]
        public void TensorJsonConverter_LargeTensor100x100_RoundTripSucceeds()
        {
            // Arrange
            var values = new double[10000];
            for (int i = 0; i < 10000; i++)
            {
                values[i] = i * 0.01;
            }
            var data = new Vector<double>(values);
            var original = new Tensor<double>(new[] { 100, 100 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(new[] { 100, 100 }, deserialized.Shape);
            Assert.Equal(10000, deserialized.Length);
        }

        #endregion

        #region JsonConverterRegistry Tests

        [Fact]
        public void JsonConverterRegistry_RegisterAllConverters_RegistersConverters()
        {
            // Arrange & Act
            JsonConverterRegistry.RegisterAllConverters();
            var converters = JsonConverterRegistry.GetAllConverters();

            // Assert
            Assert.NotNull(converters);
            Assert.True(converters.Count >= 3); // At least Matrix, Vector, and Tensor converters
            Assert.Contains(converters, c => c is MatrixJsonConverter);
            Assert.Contains(converters, c => c is VectorJsonConverter);
            Assert.Contains(converters, c => c is TensorJsonConverter);
        }

        [Fact]
        public void JsonConverterRegistry_GetAllConverters_AutoInitializes()
        {
            // Arrange
            JsonConverterRegistry.ClearConverters();

            // Act
            var converters = JsonConverterRegistry.GetAllConverters();

            // Assert
            Assert.NotNull(converters);
            Assert.True(converters.Count > 0);
        }

        [Fact]
        public void JsonConverterRegistry_GetConvertersForType_ReturnsConverters()
        {
            // Arrange
            JsonConverterRegistry.RegisterAllConverters();

            // Act
            var converters = JsonConverterRegistry.GetConvertersForType<double>();

            // Assert
            Assert.NotNull(converters);
            Assert.True(converters.Count > 0);
        }

        [Fact]
        public void JsonConverterRegistry_RegisterCustomConverter_AddsConverter()
        {
            // Arrange
            JsonConverterRegistry.ClearConverters();
            var customConverter = new MatrixJsonConverter();

            // Act
            JsonConverterRegistry.RegisterConverter(customConverter);
            var converters = JsonConverterRegistry.GetAllConverters();

            // Assert
            Assert.Contains(customConverter, converters);
        }

        [Fact]
        public void JsonConverterRegistry_ClearConverters_RemovesAllConverters()
        {
            // Arrange
            JsonConverterRegistry.RegisterAllConverters();

            // Act
            JsonConverterRegistry.ClearConverters();
            var converters = JsonConverterRegistry.GetAllConverters();

            // Assert - After clear, GetAllConverters auto-initializes
            Assert.NotNull(converters);
            Assert.True(converters.Count > 0); // Auto-initialized
        }

        #endregion

        #region VectorValidator Tests

        [Fact]
        public void VectorValidator_ValidateLength_ValidLength_Succeeds()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert - No exception should be thrown
            VectorValidator.ValidateLength(vector, 3, "Test", "ValidateLength");
        }

        [Fact]
        public void VectorValidator_ValidateLength_InvalidLength_ThrowsException()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<VectorLengthMismatchException>(() =>
                VectorValidator.ValidateLength(vector, 5, "Test", "ValidateLength"));
        }

        [Fact]
        public void VectorValidator_ValidateLengthForShape_ValidShape_Succeeds()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Act & Assert - 2x3 shape = 6 elements
            VectorValidator.ValidateLengthForShape(vector, new[] { 2, 3 }, "Test", "ValidateShape");
        }

        [Fact]
        public void VectorValidator_ValidateLengthForShape_InvalidShape_ThrowsException()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act & Assert - 2x3 shape = 6 elements, but vector has 5
            Assert.Throws<VectorLengthMismatchException>(() =>
                VectorValidator.ValidateLengthForShape(vector, new[] { 2, 3 }, "Test", "ValidateShape"));
        }

        #endregion

        #region TensorValidator Tests

        [Fact]
        public void TensorValidator_ValidateShape_ValidShape_Succeeds()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
            var tensor = new Tensor<double>(new[] { 2, 3 }, data);

            // Act & Assert - No exception should be thrown
            TensorValidator.ValidateShape(tensor, new[] { 2, 3 }, "Test", "ValidateShape");
        }

        [Fact]
        public void TensorValidator_ValidateShape_InvalidShape_ThrowsException()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var tensor = new Tensor<double>(new[] { 2, 2 }, data);

            // Act & Assert
            Assert.Throws<TensorShapeMismatchException>(() =>
                TensorValidator.ValidateShape(tensor, new[] { 3, 3 }, "Test", "ValidateShape"));
        }

        [Fact]
        public void TensorValidator_ValidateForwardPassPerformed_ValidInput_Succeeds()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var tensor = new Tensor<double>(new[] { 2, 2 }, data);

            // Act & Assert - No exception should be thrown
            TensorValidator.ValidateForwardPassPerformed(tensor, "Test", "Layer", "Backward");
        }

        [Fact]
        public void TensorValidator_ValidateForwardPassPerformed_NullInput_ThrowsException()
        {
            // Arrange
            Tensor<double>? tensor = null;

            // Act & Assert
            Assert.Throws<ForwardPassRequiredException>(() =>
                TensorValidator.ValidateForwardPassPerformed(tensor, "Test", "Layer", "Backward"));
        }

        [Fact]
        public void TensorValidator_ValidateShapesMatch_MatchingShapes_Succeeds()
        {
            // Arrange
            var data1 = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var tensor1 = new Tensor<double>(new[] { 2, 2 }, data1);
            var data2 = new Vector<double>(new[] { 5.0, 6.0, 7.0, 8.0 });
            var tensor2 = new Tensor<double>(new[] { 2, 2 }, data2);

            // Act & Assert - No exception should be thrown
            TensorValidator.ValidateShapesMatch(tensor1, tensor2, "Test", "Add");
        }

        [Fact]
        public void TensorValidator_ValidateShapesMatch_DifferentShapes_ThrowsException()
        {
            // Arrange
            var data1 = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var tensor1 = new Tensor<double>(new[] { 2, 2 }, data1);
            var data2 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var tensor2 = new Tensor<double>(new[] { 3 }, data2);

            // Act & Assert
            Assert.Throws<TensorShapeMismatchException>(() =>
                TensorValidator.ValidateShapesMatch(tensor1, tensor2, "Test", "Add"));
        }

        [Fact]
        public void TensorValidator_ValidateRank_ValidRank_Succeeds()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
            var tensor = new Tensor<double>(new[] { 2, 3 }, data);

            // Act & Assert - No exception should be thrown (rank = 2)
            TensorValidator.ValidateRank(tensor, 2, "Test", "ValidateRank");
        }

        [Fact]
        public void TensorValidator_ValidateRank_InvalidRank_ThrowsException()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var tensor = new Tensor<double>(new[] { 3 }, data);

            // Act & Assert - Tensor has rank 1, expecting 2
            Assert.Throws<TensorRankException>(() =>
                TensorValidator.ValidateRank(tensor, 2, "Test", "ValidateRank"));
        }

        #endregion

        #region RegressionValidator Tests

        [Fact]
        public void RegressionValidator_ValidateFeatureCount_ValidCount_Succeeds()
        {
            // Arrange
            var x = new Matrix<double>(10, 5); // 10 samples, 5 features

            // Act & Assert - No exception should be thrown
            RegressionValidator.ValidateFeatureCount(x, 5, "Test", "Predict");
        }

        [Fact]
        public void RegressionValidator_ValidateFeatureCount_InvalidCount_ThrowsException()
        {
            // Arrange
            var x = new Matrix<double>(10, 5); // 10 samples, 5 features

            // Act & Assert
            Assert.Throws<InvalidInputDimensionException>(() =>
                RegressionValidator.ValidateFeatureCount(x, 3, "Test", "Predict"));
        }

        [Fact]
        public void RegressionValidator_ValidateInputOutputDimensions_ValidDimensions_Succeeds()
        {
            // Arrange
            var x = new Matrix<double>(10, 5); // 10 samples, 5 features
            var y = new Vector<double>(10); // 10 target values

            // Act & Assert - No exception should be thrown
            RegressionValidator.ValidateInputOutputDimensions(x, y, "Test", "Fit");
        }

        [Fact]
        public void RegressionValidator_ValidateInputOutputDimensions_MismatchedDimensions_ThrowsException()
        {
            // Arrange
            var x = new Matrix<double>(10, 5); // 10 samples, 5 features
            var y = new Vector<double>(8); // 8 target values (mismatch!)

            // Act & Assert
            Assert.Throws<InvalidInputDimensionException>(() =>
                RegressionValidator.ValidateInputOutputDimensions(x, y, "Test", "Fit"));
        }

        [Fact]
        public void RegressionValidator_ValidateDataValues_ValidData_Succeeds()
        {
            // Arrange
            var x = new Matrix<double>(3, 2);
            x[0, 0] = 1.0; x[0, 1] = 2.0;
            x[1, 0] = 3.0; x[1, 1] = 4.0;
            x[2, 0] = 5.0; x[2, 1] = 6.0;
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert - No exception should be thrown
            RegressionValidator.ValidateDataValues(x, y, "Test", "Fit");
        }

        [Fact]
        public void RegressionValidator_ValidateDataValues_MatrixWithNaN_ThrowsException()
        {
            // Arrange
            var x = new Matrix<double>(3, 2);
            x[0, 0] = 1.0; x[0, 1] = 2.0;
            x[1, 0] = double.NaN; x[1, 1] = 4.0;
            x[2, 0] = 5.0; x[2, 1] = 6.0;
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<InvalidDataValueException>(() =>
                RegressionValidator.ValidateDataValues(x, y, "Test", "Fit"));
        }

        [Fact]
        public void RegressionValidator_ValidateDataValues_VectorWithInfinity_ThrowsException()
        {
            // Arrange
            var x = new Matrix<double>(3, 2);
            x[0, 0] = 1.0; x[0, 1] = 2.0;
            x[1, 0] = 3.0; x[1, 1] = 4.0;
            x[2, 0] = 5.0; x[2, 1] = 6.0;
            var y = new Vector<double>(new[] { 1.0, double.PositiveInfinity, 3.0 });

            // Act & Assert
            Assert.Throws<InvalidDataValueException>(() =>
                RegressionValidator.ValidateDataValues(x, y, "Test", "Fit"));
        }

        #endregion

        #region ArchitectureValidator Tests

        [Fact]
        public void ArchitectureValidator_ValidateInputType_ValidType_Succeeds()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                inputSize: 10,
                outputSize: 5,
                hiddenLayerSizes: new[] { 20 }
            );

            // Act & Assert - No exception should be thrown
            ArchitectureValidator.ValidateInputType(architecture, InputType.OneDimensional, "FeedForward");
        }

        [Fact]
        public void ArchitectureValidator_ValidateInputType_InvalidType_ThrowsException()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                inputSize: 10,
                outputSize: 5,
                hiddenLayerSizes: new[] { 20 }
            );

            // Act & Assert
            Assert.Throws<InvalidInputTypeException>(() =>
                ArchitectureValidator.ValidateInputType(architecture, InputType.TwoDimensional, "CNN"));
        }

        #endregion

        #region SerializationValidator Tests

        [Fact]
        public void SerializationValidator_ValidateWriter_ValidWriter_Succeeds()
        {
            // Arrange
            using var stream = new MemoryStream();
            using var writer = new BinaryWriter(stream);

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateWriter(writer, "Test", "Write");
        }

        [Fact]
        public void SerializationValidator_ValidateWriter_NullWriter_ThrowsException()
        {
            // Arrange
            BinaryWriter? writer = null;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateWriter(writer, "Test", "Write"));
        }

        [Fact]
        public void SerializationValidator_ValidateReader_ValidReader_Succeeds()
        {
            // Arrange
            using var stream = new MemoryStream(new byte[] { 1, 2, 3 });
            using var reader = new BinaryReader(stream);

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateReader(reader, "Test", "Read");
        }

        [Fact]
        public void SerializationValidator_ValidateReader_NullReader_ThrowsException()
        {
            // Arrange
            BinaryReader? reader = null;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateReader(reader, "Test", "Read"));
        }

        [Fact]
        public void SerializationValidator_ValidateStream_ValidReadableStream_Succeeds()
        {
            // Arrange
            using var stream = new MemoryStream(new byte[] { 1, 2, 3 });

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateStream(stream, requireRead: true, requireWrite: false, "Test", "Read");
        }

        [Fact]
        public void SerializationValidator_ValidateStream_ValidWritableStream_Succeeds()
        {
            // Arrange
            using var stream = new MemoryStream();

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateStream(stream, requireRead: false, requireWrite: true, "Test", "Write");
        }

        [Fact]
        public void SerializationValidator_ValidateStream_NullStream_ThrowsException()
        {
            // Arrange
            Stream? stream = null;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateStream(stream, requireRead: true, "Test", "Read"));
        }

        [Fact]
        public void SerializationValidator_ValidateFilePath_ValidPath_Succeeds()
        {
            // Arrange
            string path = "/path/to/file.json";

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateFilePath(path, "Test", "Save");
        }

        [Fact]
        public void SerializationValidator_ValidateFilePath_NullPath_ThrowsException()
        {
            // Arrange
            string? path = null;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateFilePath(path, "Test", "Save"));
        }

        [Fact]
        public void SerializationValidator_ValidateFilePath_EmptyPath_ThrowsException()
        {
            // Arrange
            string path = "";

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateFilePath(path, "Test", "Save"));
        }

        [Fact]
        public void SerializationValidator_ValidateVersion_MatchingVersion_Succeeds()
        {
            // Arrange
            int actualVersion = 1;
            int expectedVersion = 1;

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateVersion(actualVersion, expectedVersion, "Test", "Load");
        }

        [Fact]
        public void SerializationValidator_ValidateVersion_MismatchedVersion_ThrowsException()
        {
            // Arrange
            int actualVersion = 2;
            int expectedVersion = 1;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateVersion(actualVersion, expectedVersion, "Test", "Load"));
        }

        [Fact]
        public void SerializationValidator_ValidateLayerTypeName_ValidName_Succeeds()
        {
            // Arrange
            string layerTypeName = "DenseLayer";

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateLayerTypeName(layerTypeName, "Test", "Deserialize");
        }

        [Fact]
        public void SerializationValidator_ValidateLayerTypeName_NullName_ThrowsException()
        {
            // Arrange
            string? layerTypeName = null;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateLayerTypeName(layerTypeName, "Test", "Deserialize"));
        }

        [Fact]
        public void SerializationValidator_ValidateLayerTypeName_EmptyName_ThrowsException()
        {
            // Arrange
            string layerTypeName = "";

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateLayerTypeName(layerTypeName, "Test", "Deserialize"));
        }

        [Fact]
        public void SerializationValidator_ValidateLayerTypeExists_ValidType_Succeeds()
        {
            // Arrange
            string layerTypeName = "System.String";
            Type? layerType = typeof(string);

            // Act & Assert - No exception should be thrown
            SerializationValidator.ValidateLayerTypeExists(layerTypeName, layerType, "Test", "Deserialize");
        }

        [Fact]
        public void SerializationValidator_ValidateLayerTypeExists_NullType_ThrowsException()
        {
            // Arrange
            string layerTypeName = "NonExistentLayer";
            Type? layerType = null;

            // Act & Assert
            Assert.Throws<SerializationException>(() =>
                SerializationValidator.ValidateLayerTypeExists(layerTypeName, layerType, "Test", "Deserialize"));
        }

        #endregion

        #region Integration Scenarios

        [Fact]
        public void IntegrationScenario_Serialize100x100Matrix_DeserializeAndVerifyEquality()
        {
            // Arrange - Create a 100x100 matrix with sequential values
            var original = new Matrix<double>(100, 100);
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    original[i, j] = i * 100.0 + j;
                }
            }

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new MatrixJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(100, deserialized.Rows);
            Assert.Equal(100, deserialized.Columns);

            // Verify all elements match
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.Equal(original[i, j], deserialized[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void IntegrationScenario_TensorWithNaN_SerializeAndVerifyPreservation()
        {
            // Arrange - Create a 3D tensor with some NaN values
            var values = new double[] {
                1.0, 2.0, double.NaN, 4.0,
                5.0, double.NaN, 7.0, 8.0,
                9.0, 10.0, 11.0, double.NaN
            };
            var data = new Vector<double>(values);
            var original = new Tensor<double>(new[] { 3, 2, 2 }, data);

            var settings = new JsonSerializerSettings();
            settings.Converters.Add(new TensorJsonConverter());

            // Act
            string json = JsonConvert.SerializeObject(original, settings);
            var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

            // Assert
            Assert.NotNull(deserialized);
            var deserializedArray = deserialized.ToArray();

            Assert.Equal(1.0, deserializedArray[0], precision: 10);
            Assert.Equal(2.0, deserializedArray[1], precision: 10);
            Assert.True(double.IsNaN(deserializedArray[2]));
            Assert.Equal(4.0, deserializedArray[3], precision: 10);
            Assert.Equal(5.0, deserializedArray[4], precision: 10);
            Assert.True(double.IsNaN(deserializedArray[5]));
            Assert.Equal(7.0, deserializedArray[6], precision: 10);
            Assert.Equal(8.0, deserializedArray[7], precision: 10);
            Assert.Equal(9.0, deserializedArray[8], precision: 10);
            Assert.Equal(10.0, deserializedArray[9], precision: 10);
            Assert.Equal(11.0, deserializedArray[10], precision: 10);
            Assert.True(double.IsNaN(deserializedArray[11]));
        }

        [Fact]
        public void IntegrationScenario_ValidateTensorShapeForNeuralNetwork_Succeeds()
        {
            // Arrange - Create a tensor representing a mini-batch of images (batch_size=32, height=28, width=28, channels=1)
            var values = new double[32 * 28 * 28 * 1];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = i * 0.001;
            }
            var data = new Vector<double>(values);
            var tensor = new Tensor<double>(new[] { 32, 28, 28, 1 }, data);

            // Act & Assert - Validate the tensor has the expected shape for neural network input
            TensorValidator.ValidateShape(tensor, new[] { 32, 28, 28, 1 }, "NeuralNetwork", "Forward");
            TensorValidator.ValidateRank(tensor, 4, "NeuralNetwork", "Forward");
        }

        [Fact]
        public void IntegrationScenario_ValidateRegressionOutputDimensions_Succeeds()
        {
            // Arrange - Training data with 100 samples and 5 features
            var xTrain = new Matrix<double>(100, 5);
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    xTrain[i, j] = i + j * 0.5;
                }
            }
            var yTrain = new Vector<double>(100);
            for (int i = 0; i < 100; i++)
            {
                yTrain[i] = i * 2.0;
            }

            // Act & Assert - Validate input/output dimensions match
            RegressionValidator.ValidateInputOutputDimensions(xTrain, yTrain, "LinearRegression", "Fit");

            // Test data should have same number of features
            var xTest = new Matrix<double>(20, 5);
            RegressionValidator.ValidateFeatureCount(xTest, 5, "LinearRegression", "Predict");
        }

        [Fact]
        public void IntegrationScenario_ValidateNeuralArchitecture_InputToHiddenToOutput()
        {
            // Arrange - Create a neural network architecture with proper layer sizes
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                inputSize: 784, // 28x28 flattened image
                outputSize: 10, // 10 digit classes
                hiddenLayerSizes: new[] { 128, 64 }
            );

            // Act & Assert - Validate the architecture has the correct input type
            ArchitectureValidator.ValidateInputType(architecture, InputType.OneDimensional, "FeedForwardNetwork");

            // Verify architecture properties
            Assert.Equal(784, architecture.InputSize);
            Assert.Equal(10, architecture.OutputSize);
            Assert.Equal(InputType.OneDimensional, architecture.InputType);
        }

        [Fact]
        public void IntegrationScenario_CompleteSerializationPipeline_MatrixVectorTensor()
        {
            // Arrange - Create all three data structures
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            var tensorData = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            var tensor = new Tensor<double>(new[] { 2, 2, 2 }, tensorData);

            // Register all converters
            JsonConverterRegistry.RegisterAllConverters();
            var converters = JsonConverterRegistry.GetAllConverters();
            var settings = new JsonSerializerSettings();
            foreach (var converter in converters)
            {
                settings.Converters.Add(converter);
            }

            // Act - Serialize all three
            string matrixJson = JsonConvert.SerializeObject(matrix, settings);
            string vectorJson = JsonConvert.SerializeObject(vector, settings);
            string tensorJson = JsonConvert.SerializeObject(tensor, settings);

            // Deserialize all three
            var deserializedMatrix = JsonConvert.DeserializeObject<Matrix<double>>(matrixJson, settings);
            var deserializedVector = JsonConvert.DeserializeObject<Vector<double>>(vectorJson, settings);
            var deserializedTensor = JsonConvert.DeserializeObject<Tensor<double>>(tensorJson, settings);

            // Assert - Verify all deserializations succeeded
            Assert.NotNull(deserializedMatrix);
            Assert.NotNull(deserializedVector);
            Assert.NotNull(deserializedTensor);

            // Verify matrix values
            Assert.Equal(5.0, deserializedMatrix[1, 1], precision: 10);

            // Verify vector values
            Assert.Equal(5, deserializedVector.Length);
            Assert.Equal(3.0, deserializedVector[2], precision: 10);

            // Verify tensor values
            Assert.Equal(new[] { 2, 2, 2 }, deserializedTensor.Shape);
        }

        #endregion
    }
}
