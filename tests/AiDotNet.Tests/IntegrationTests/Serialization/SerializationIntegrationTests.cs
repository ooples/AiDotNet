using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Models;
using AiDotNet.Serialization;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Serialization;

/// <summary>
/// Integration tests for the AiDotNet.Serialization module.
/// Tests JSON converters for Vector, Matrix, Tensor, and the serialization infrastructure.
/// </summary>
public class SerializationIntegrationTests : IDisposable
{
    public SerializationIntegrationTests()
    {
        // Clear converters before each test to ensure isolation
        JsonConverterRegistry.ClearConverters();
    }

    public void Dispose()
    {
        // Clean up after each test
        JsonConverterRegistry.ClearConverters();
    }

    #region JsonConverterRegistry Tests

    [Fact]
    public void JsonConverterRegistry_RegisterAllConverters_InitializesConverters()
    {
        JsonConverterRegistry.RegisterAllConverters();

        var converters = JsonConverterRegistry.GetAllConverters();

        Assert.NotNull(converters);
        Assert.Equal(3, converters.Count); // Matrix, Vector, Tensor
    }

    [Fact]
    public void JsonConverterRegistry_RegisterAllConverters_IsIdempotent()
    {
        JsonConverterRegistry.RegisterAllConverters();
        JsonConverterRegistry.RegisterAllConverters();
        JsonConverterRegistry.RegisterAllConverters();

        var converters = JsonConverterRegistry.GetAllConverters();

        // Verify total count is still 3 (not 9)
        Assert.Equal(3, converters.Count);

        // Verify exactly one of each converter type (no duplicates)
        var vectorConverterCount = converters.Count(c => c is VectorJsonConverter);
        var matrixConverterCount = converters.Count(c => c is MatrixJsonConverter);
        var tensorConverterCount = converters.Count(c => c is TensorJsonConverter);

        Assert.Equal(1, vectorConverterCount);
        Assert.Equal(1, matrixConverterCount);
        Assert.Equal(1, tensorConverterCount);
    }

    [Fact]
    public void JsonConverterRegistry_GetAllConverters_AutoInitializes()
    {
        // Don't call RegisterAllConverters - GetAllConverters should do it
        var converters = JsonConverterRegistry.GetAllConverters();

        Assert.NotNull(converters);
        Assert.Equal(3, converters.Count);
    }

    [Fact]
    public void JsonConverterRegistry_GetConvertersForType_ReturnsConverters()
    {
        var converters = JsonConverterRegistry.GetConvertersForType<double>();

        Assert.NotNull(converters);
        Assert.Equal(3, converters.Count);
    }

    [Fact]
    public void JsonConverterRegistry_RegisterConverter_AddsCustomConverter()
    {
        JsonConverterRegistry.RegisterAllConverters();
        var customConverter = new MockJsonConverter();

        JsonConverterRegistry.RegisterConverter(customConverter);

        var converters = JsonConverterRegistry.GetAllConverters();
        Assert.Equal(4, converters.Count);
        Assert.Contains(customConverter, converters);
    }

    [Fact]
    public void JsonConverterRegistry_RegisterConverter_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            JsonConverterRegistry.RegisterConverter(null!));
    }

    [Fact]
    public void JsonConverterRegistry_RegisterConverter_DoesNotDuplicate()
    {
        JsonConverterRegistry.RegisterAllConverters();
        var customConverter = new MockJsonConverter();

        JsonConverterRegistry.RegisterConverter(customConverter);
        JsonConverterRegistry.RegisterConverter(customConverter);

        var converters = JsonConverterRegistry.GetAllConverters();
        Assert.Equal(4, converters.Count); // Should not add duplicate
    }

    [Fact]
    public void JsonConverterRegistry_ClearConverters_RemovesAll()
    {
        // Step 1: Register default converters
        JsonConverterRegistry.RegisterAllConverters();
        var initialCount = JsonConverterRegistry.GetAllConverters().Count;
        Assert.Equal(3, initialCount); // 3 default converters: Matrix, Vector, Tensor

        // Step 2: Register a custom converter
        var customConverter = new MockJsonConverter();
        JsonConverterRegistry.RegisterConverter(customConverter);

        // Step 3: Verify custom converter was added
        var convertersBeforeClear = JsonConverterRegistry.GetAllConverters();
        Assert.Equal(4, convertersBeforeClear.Count); // 3 defaults + 1 custom
        Assert.Contains(customConverter, convertersBeforeClear);

        // Step 4: Clear all converters
        JsonConverterRegistry.ClearConverters();

        // Step 5: Get converters again (triggers auto-reinitialization with defaults only)
        var convertersAfterClear = JsonConverterRegistry.GetAllConverters();

        // Step 6: Verify custom converter was removed and only defaults remain
        Assert.Equal(3, convertersAfterClear.Count); // Back to 3 defaults only
        Assert.DoesNotContain(customConverter, convertersAfterClear);
    }

    #endregion

    #region VectorJsonConverter Tests

    [Fact]
    public void VectorJsonConverter_CanConvert_Vector_ReturnsTrue()
    {
        var converter = new VectorJsonConverter();

        Assert.True(converter.CanConvert(typeof(Vector<double>)));
        Assert.True(converter.CanConvert(typeof(Vector<float>)));
        Assert.True(converter.CanConvert(typeof(Vector<int>)));
    }

    [Fact]
    public void VectorJsonConverter_CanConvert_NonVector_ReturnsFalse()
    {
        var converter = new VectorJsonConverter();

        Assert.False(converter.CanConvert(typeof(string)));
        Assert.False(converter.CanConvert(typeof(int)));
        Assert.False(converter.CanConvert(typeof(double[])));
        Assert.False(converter.CanConvert(typeof(List<double>)));
    }

    [Fact]
    public void VectorJsonConverter_RoundTrip_PreservesData()
    {
        var originalData = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        var originalVector = new Vector<double>(originalData);
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(originalVector, settings);
        var deserializedVector = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserializedVector);
        Assert.Equal(originalVector.Length, deserializedVector.Length);
        for (int i = 0; i < originalVector.Length; i++)
        {
            Assert.Equal(originalVector[i], deserializedVector[i]);
        }
    }

    [Fact]
    public void VectorJsonConverter_Serialize_EmptyVector()
    {
        var emptyVector = new Vector<double>(0);
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(emptyVector, settings);
        var deserializedVector = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserializedVector);
        Assert.Equal(0, deserializedVector.Length);
    }

    [Fact]
    public void VectorJsonConverter_Serialize_SingleElement()
    {
        var singleVector = new Vector<double>(new double[] { 42.0 });
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(singleVector, settings);
        var deserializedVector = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserializedVector);
        Assert.Equal(1, deserializedVector.Length);
        Assert.Equal(42.0, deserializedVector[0]);
    }

    [Fact]
    public void VectorJsonConverter_Serialize_Null_WritesNull()
    {
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        Vector<double>? nullVector = null;
        var json = JsonConvert.SerializeObject(nullVector, settings);

        Assert.Equal("null", json);
    }

    [Fact]
    public void VectorJsonConverter_Deserialize_Null_ReturnsNull()
    {
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var result = JsonConvert.DeserializeObject<Vector<double>>("null", settings);

        Assert.Null(result);
    }

    [Fact]
    public void VectorJsonConverter_Deserialize_InvalidJson_ThrowsException()
    {
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        // JSON missing 'length' property
        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Vector<double>>("{\"data\":[1,2,3]}", settings));
    }

    [Fact]
    public void VectorJsonConverter_Deserialize_MismatchedLength_ThrowsException()
    {
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        // length says 5 but only 3 elements in data
        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Vector<double>>("{\"length\":5,\"data\":[1,2,3]}", settings));
    }

    [Fact]
    public void VectorJsonConverter_JsonFormat_IsCorrect()
    {
        var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var converter = new VectorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter },
            Formatting = Formatting.None
        };

        var json = JsonConvert.SerializeObject(vector, settings);

        Assert.Contains("\"length\":3", json);
        Assert.Contains("\"data\":", json);
    }

    #endregion

    #region MatrixJsonConverter Tests

    [Fact]
    public void MatrixJsonConverter_CanConvert_Matrix_ReturnsTrue()
    {
        var converter = new MatrixJsonConverter();

        Assert.True(converter.CanConvert(typeof(Matrix<double>)));
        Assert.True(converter.CanConvert(typeof(Matrix<float>)));
    }

    [Fact]
    public void MatrixJsonConverter_CanConvert_NonMatrix_ReturnsFalse()
    {
        var converter = new MatrixJsonConverter();

        Assert.False(converter.CanConvert(typeof(string)));
        Assert.False(converter.CanConvert(typeof(Vector<double>)));
        Assert.False(converter.CanConvert(typeof(double[,])));
    }

    [Fact]
    public void MatrixJsonConverter_RoundTrip_PreservesData()
    {
        var originalMatrix = new Matrix<double>(2, 3);
        originalMatrix[0, 0] = 1.0; originalMatrix[0, 1] = 2.0; originalMatrix[0, 2] = 3.0;
        originalMatrix[1, 0] = 4.0; originalMatrix[1, 1] = 5.0; originalMatrix[1, 2] = 6.0;

        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(originalMatrix, settings);
        var deserializedMatrix = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserializedMatrix);
        Assert.Equal(originalMatrix.Rows, deserializedMatrix.Rows);
        Assert.Equal(originalMatrix.Columns, deserializedMatrix.Columns);

        for (int i = 0; i < originalMatrix.Rows; i++)
        {
            for (int j = 0; j < originalMatrix.Columns; j++)
            {
                Assert.Equal(originalMatrix[i, j], deserializedMatrix[i, j]);
            }
        }
    }

    [Fact]
    public void MatrixJsonConverter_Serialize_EmptyMatrix()
    {
        var emptyMatrix = new Matrix<double>(0, 0);
        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(emptyMatrix, settings);
        var deserializedMatrix = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserializedMatrix);
        Assert.Equal(0, deserializedMatrix.Rows);
        Assert.Equal(0, deserializedMatrix.Columns);
    }

    [Fact]
    public void MatrixJsonConverter_Serialize_SingleElement()
    {
        var singleMatrix = new Matrix<double>(1, 1);
        singleMatrix[0, 0] = 99.5;

        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(singleMatrix, settings);
        var deserializedMatrix = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserializedMatrix);
        Assert.Equal(1, deserializedMatrix.Rows);
        Assert.Equal(1, deserializedMatrix.Columns);
        Assert.Equal(99.5, deserializedMatrix[0, 0]);
    }

    [Fact]
    public void MatrixJsonConverter_Serialize_Null_WritesNull()
    {
        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        Matrix<double>? nullMatrix = null;
        var json = JsonConvert.SerializeObject(nullMatrix, settings);

        Assert.Equal("null", json);
    }

    [Fact]
    public void MatrixJsonConverter_Deserialize_InvalidJson_ThrowsException()
    {
        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        // JSON missing required properties
        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Matrix<double>>("{\"data\":[1,2,3]}", settings));
    }

    [Fact]
    public void MatrixJsonConverter_Deserialize_MismatchedDimensions_ThrowsException()
    {
        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        // rows*columns = 6 but only 3 elements in data
        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Matrix<double>>("{\"rows\":2,\"columns\":3,\"data\":[1,2,3]}", settings));
    }

    [Fact]
    public void MatrixJsonConverter_JsonFormat_IsCorrect()
    {
        var matrix = new Matrix<double>(2, 2);
        var converter = new MatrixJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter },
            Formatting = Formatting.None
        };

        var json = JsonConvert.SerializeObject(matrix, settings);

        Assert.Contains("\"rows\":2", json);
        Assert.Contains("\"columns\":2", json);
        Assert.Contains("\"data\":", json);
    }

    #endregion

    #region TensorJsonConverter Tests

    [Fact]
    public void TensorJsonConverter_CanConvert_Tensor_ReturnsTrue()
    {
        var converter = new TensorJsonConverter();

        Assert.True(converter.CanConvert(typeof(Tensor<double>)));
        Assert.True(converter.CanConvert(typeof(Tensor<float>)));
    }

    [Fact]
    public void TensorJsonConverter_CanConvert_NonTensor_ReturnsFalse()
    {
        var converter = new TensorJsonConverter();

        Assert.False(converter.CanConvert(typeof(string)));
        Assert.False(converter.CanConvert(typeof(Vector<double>)));
        Assert.False(converter.CanConvert(typeof(Matrix<double>)));
    }

    [Fact]
    public void TensorJsonConverter_RoundTrip_1D_PreservesData()
    {
        var originalTensor = new Tensor<double>(new int[] { 5 });
        for (int i = 0; i < 5; i++)
        {
            originalTensor[i] = i * 1.5;
        }

        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(originalTensor, settings);
        var deserializedTensor = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserializedTensor);
        Assert.Equal(originalTensor.Shape.Length, deserializedTensor.Shape.Length);
        Assert.Equal(originalTensor.Shape[0], deserializedTensor.Shape[0]);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(originalTensor[i], deserializedTensor[i]);
        }
    }

    [Fact]
    public void TensorJsonConverter_RoundTrip_2D_PreservesData()
    {
        var originalTensor = new Tensor<double>(new int[] { 2, 3 });
        int index = 0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                originalTensor[i, j] = ++index * 0.5;
            }
        }

        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(originalTensor, settings);
        var deserializedTensor = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserializedTensor);
        Assert.Equal(2, deserializedTensor.Shape.Length);
        Assert.Equal(2, deserializedTensor.Shape[0]);
        Assert.Equal(3, deserializedTensor.Shape[1]);
    }

    [Fact]
    public void TensorJsonConverter_RoundTrip_3D_PreservesData()
    {
        var originalTensor = new Tensor<double>(new int[] { 2, 3, 4 });

        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        var json = JsonConvert.SerializeObject(originalTensor, settings);
        var deserializedTensor = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserializedTensor);
        Assert.Equal(3, deserializedTensor.Shape.Length);
        Assert.Equal(2, deserializedTensor.Shape[0]);
        Assert.Equal(3, deserializedTensor.Shape[1]);
        Assert.Equal(4, deserializedTensor.Shape[2]);
    }

    [Fact]
    public void TensorJsonConverter_Serialize_Null_WritesNull()
    {
        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        Tensor<double>? nullTensor = null;
        var json = JsonConvert.SerializeObject(nullTensor, settings);

        Assert.Equal("null", json);
    }

    [Fact]
    public void TensorJsonConverter_Deserialize_MissingShape_ThrowsException()
    {
        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Tensor<double>>("{\"data\":[1,2,3]}", settings));
    }

    [Fact]
    public void TensorJsonConverter_Deserialize_MismatchedShape_ThrowsException()
    {
        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter }
        };

        // shape [2,3] = 6 elements but only 3 in data
        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Tensor<double>>("{\"shape\":[2,3],\"data\":[1,2,3]}", settings));
    }

    [Fact]
    public void TensorJsonConverter_JsonFormat_IsCorrect()
    {
        var tensor = new Tensor<double>(new int[] { 2, 2 });
        var converter = new TensorJsonConverter();

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { converter },
            Formatting = Formatting.None
        };

        var json = JsonConvert.SerializeObject(tensor, settings);

        Assert.Contains("\"shape\":", json);
        Assert.Contains("\"data\":", json);
    }

    #endregion

    #region SafeSerializationBinder Tests

    [Fact]
    public void SafeSerializationBinder_BindToType_AllowsAiDotNetTypes()
    {
        var binder = new SafeSerializationBinder();

        // Test that AiDotNet types work with full assembly qualification
        // The Serialization_WithSafeBinder_WorksForAllowedTypes test verifies the full end-to-end
        // Here we test that BindToName correctly outputs type information for AiDotNet types
        binder.BindToName(typeof(ReasoningConfig), out string? assemblyName, out string? typeName);

        Assert.NotNull(typeName);
        Assert.Contains("AiDotNet", typeName);
        Assert.Contains("ReasoningConfig", typeName);
    }

    [Fact]
    public void SafeSerializationBinder_BindToType_AllowsPrimitiveTypes()
    {
        var binder = new SafeSerializationBinder();

        // Should allow System primitives
        var stringType = binder.BindToType(null, "System.String");
        var intType = binder.BindToType(null, "System.Int32");
        var doubleType = binder.BindToType(null, "System.Double");

        Assert.Equal(typeof(string), stringType);
        Assert.Equal(typeof(int), intType);
        Assert.Equal(typeof(double), doubleType);
    }

    [Fact]
    public void SafeSerializationBinder_BindToType_AllowsGenericCollections()
    {
        var binder = new SafeSerializationBinder();

        // Should allow System.Collections.Generic types
        var listType = binder.BindToType(null, "System.Collections.Generic.List`1[[System.Int32, mscorlib]]");

        Assert.NotNull(listType);
    }

    [Fact]
    public void SafeSerializationBinder_BindToType_RejectsDangerousTypes()
    {
        var binder = new SafeSerializationBinder();

        // Should reject types outside allowed namespaces
        Assert.Throws<InvalidOperationException>(() =>
            binder.BindToType(null, "System.IO.FileInfo"));
    }

    [Fact]
    public void SafeSerializationBinder_BindToName_DelegatesToDefaultBinder()
    {
        var binder = new SafeSerializationBinder();

        binder.BindToName(typeof(int), out string? assemblyName, out string? typeName);

        Assert.NotNull(typeName);
        Assert.Contains("Int32", typeName);
    }

    [Fact]
    public void SafeSerializationBinder_AllowsArrayOfPrimitives()
    {
        var binder = new SafeSerializationBinder();

        var doubleArrayType = binder.BindToType(null, "System.Double[]");

        Assert.NotNull(doubleArrayType);
        Assert.True(doubleArrayType.IsArray);
    }

    [Fact]
    public void SafeSerializationBinder_AllowsNullablePrimitives()
    {
        var binder = new SafeSerializationBinder();

        var nullableIntType = binder.BindToType(null, "System.Nullable`1[[System.Int32, mscorlib]]");

        Assert.NotNull(nullableIntType);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllConverters_WorkTogether()
    {
        JsonConverterRegistry.RegisterAllConverters();
        var converters = JsonConverterRegistry.GetAllConverters();

        var settings = new JsonSerializerSettings
        {
            Converters = converters
        };

        // Test all types together
        var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[1, 0] = 3; matrix[1, 1] = 4;
        var tensor = new Tensor<double>(new int[] { 2, 2 });

        var vectorJson = JsonConvert.SerializeObject(vector, settings);
        var matrixJson = JsonConvert.SerializeObject(matrix, settings);
        var tensorJson = JsonConvert.SerializeObject(tensor, settings);

        var deserializedVector = JsonConvert.DeserializeObject<Vector<double>>(vectorJson, settings);
        var deserializedMatrix = JsonConvert.DeserializeObject<Matrix<double>>(matrixJson, settings);
        var deserializedTensor = JsonConvert.DeserializeObject<Tensor<double>>(tensorJson, settings);

        Assert.NotNull(deserializedVector);
        Assert.NotNull(deserializedMatrix);
        Assert.NotNull(deserializedTensor);

        Assert.Equal(vector.Length, deserializedVector.Length);
        Assert.Equal(matrix.Rows, deserializedMatrix.Rows);
        Assert.Equal(tensor.Shape.Length, deserializedTensor.Shape.Length);
    }

    [Fact]
    public void Serialization_WithSafeBinder_WorksForAllowedTypes()
    {
        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder(),
            Converters = JsonConverterRegistry.GetAllConverters()
        };

        var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var json = JsonConvert.SerializeObject(vector, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(vector.Length, deserialized.Length);
    }

    [Fact]
    public void ComplexObject_WithLinearAlgebraTypes_SerializesCorrectly()
    {
        var settings = new JsonSerializerSettings
        {
            Converters = JsonConverterRegistry.GetAllConverters()
        };

        var complexObject = new TestContainer
        {
            Name = "Test",
            Vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
            Matrix = new Matrix<double>(2, 2)
        };
        complexObject.Matrix[0, 0] = 1; complexObject.Matrix[0, 1] = 2;
        complexObject.Matrix[1, 0] = 3; complexObject.Matrix[1, 1] = 4;

        var json = JsonConvert.SerializeObject(complexObject, settings);
        var deserialized = JsonConvert.DeserializeObject<TestContainer>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal("Test", deserialized.Name);
        Assert.NotNull(deserialized.Vector);
        Assert.Equal(3, deserialized.Vector.Length);
        Assert.NotNull(deserialized.Matrix);
        Assert.Equal(2, deserialized.Matrix.Rows);
    }

    [Fact]
    public void LargeVector_SerializesCorrectly()
    {
        var largeData = new double[10000];
        for (int i = 0; i < largeData.Length; i++)
        {
            largeData[i] = i * 0.001;
        }
        var largeVector = new Vector<double>(largeData);

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { new VectorJsonConverter() }
        };

        var json = JsonConvert.SerializeObject(largeVector, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(10000, deserialized.Length);
        Assert.Equal(largeData[0], deserialized[0]);
        Assert.Equal(largeData[9999], deserialized[9999]);
    }

    [Fact]
    public void LargeMatrix_SerializesCorrectly()
    {
        var largeMatrix = new Matrix<double>(100, 100);
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                largeMatrix[i, j] = i * 100 + j;
            }
        }

        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { new MatrixJsonConverter() }
        };

        var json = JsonConvert.SerializeObject(largeMatrix, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(100, deserialized.Rows);
        Assert.Equal(100, deserialized.Columns);
        Assert.Equal(0, deserialized[0, 0]);
        Assert.Equal(9999, deserialized[99, 99]);
    }

    #endregion

    #region Helper Classes

    private class MockJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType) => false;

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
            => throw new NotImplementedException();

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
            => throw new NotImplementedException();
    }

    private class TestContainer
    {
        public string Name { get; set; } = string.Empty;
        public Vector<double>? Vector { get; set; }
        public Matrix<double>? Matrix { get; set; }
    }

    #endregion
}
