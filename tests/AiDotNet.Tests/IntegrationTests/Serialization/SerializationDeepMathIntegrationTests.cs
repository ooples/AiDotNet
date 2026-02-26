using AiDotNet.Serialization;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Serialization;

/// <summary>
/// Deep math integration tests for JSON serialization roundtrips of Vector, Matrix, and Tensor.
/// Verifies exact data preservation, edge cases, precision retention, and error handling.
/// </summary>
public class SerializationDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    private static JsonSerializerSettings CreateSettings()
    {
        var settings = new JsonSerializerSettings();
        settings.Converters.Add(new VectorJsonConverter());
        settings.Converters.Add(new MatrixJsonConverter());
        settings.Converters.Add(new TensorJsonConverter());
        return settings;
    }

    // ============================
    // Vector Roundtrip Tests
    // ============================

    [Fact]
    public void Vector_Roundtrip_PreservesExactValues()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.5, -3.7, 0.0, 100.0 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(v.Length, deserialized.Length);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(v[i], deserialized[i], Tolerance);
        }
    }

    [Fact]
    public void Vector_Roundtrip_PreservesHighPrecision()
    {
        // Test that very precise floating point values survive the roundtrip
        var v = new Vector<double>(new double[]
        {
            Math.PI, Math.E, Math.Sqrt(2.0), 1.0 / 3.0, 1e-15, 1e+15
        });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(v[i], deserialized[i], Tolerance);
        }
    }

    [Fact]
    public void Vector_Roundtrip_NegativeValues()
    {
        var v = new Vector<double>(new double[] { -1e-10, -999.999, -0.0, double.MinValue / 2 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(v.Length, deserialized.Length);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(v[i], deserialized[i], Tolerance);
        }
    }

    [Fact]
    public void Vector_Roundtrip_SingleElement()
    {
        var v = new Vector<double>(new double[] { 42.0 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(1, deserialized.Length);
        Assert.Equal(42.0, deserialized[0], Tolerance);
    }

    [Fact]
    public void Vector_Roundtrip_AllZeros()
    {
        var v = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(0.0, deserialized[i], Tolerance);
        }
    }

    [Fact]
    public void Vector_Roundtrip_LargeVector()
    {
        // Test with 1000 elements to verify no truncation
        var data = new double[1000];
        for (int i = 0; i < data.Length; i++)
            data[i] = i * 0.001 - 0.5; // range [-0.5, 0.499]

        var v = new Vector<double>(data);
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(1000, deserialized.Length);
        for (int i = 0; i < 1000; i++)
        {
            Assert.Equal(v[i], deserialized[i], Tolerance);
        }
    }

    [Fact]
    public void Vector_JsonStructure_ContainsLengthAndData()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);

        Assert.Contains("\"length\":3", json);
        Assert.Contains("\"data\":", json);
    }

    [Fact]
    public void Vector_Deserialize_MissingLength_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"data\":[1.0,2.0]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Vector<double>>(invalidJson, settings));
    }

    [Fact]
    public void Vector_Deserialize_MissingData_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"length\":2}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Vector<double>>(invalidJson, settings));
    }

    [Fact]
    public void Vector_Deserialize_LengthMismatch_Throws()
    {
        var settings = CreateSettings();
        // length says 3 but only 2 elements
        var invalidJson = "{\"length\":3,\"data\":[1.0,2.0]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Vector<double>>(invalidJson, settings));
    }

    [Fact]
    public void Vector_Deserialize_NegativeLength_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"length\":-1,\"data\":[]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Vector<double>>(invalidJson, settings));
    }

    // ============================
    // Matrix Roundtrip Tests
    // ============================

    [Fact]
    public void Matrix_Roundtrip_PreservesExactValues()
    {
        var m = new Matrix<double>(3, 3);
        // Fill with known values: Hilbert-like matrix
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i, j] = 1.0 / (i + j + 1);

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(3, deserialized.Rows);
        Assert.Equal(3, deserialized.Columns);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(m[i, j], deserialized[i, j], Tolerance);
    }

    [Fact]
    public void Matrix_Roundtrip_Identity()
    {
        var m = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            m[i, i] = 1.0;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserialized);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(i == j ? 1.0 : 0.0, deserialized[i, j], Tolerance);
    }

    [Fact]
    public void Matrix_Roundtrip_NonSquare()
    {
        var m = new Matrix<double>(2, 5);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 5; j++)
                m[i, j] = i * 10.0 + j;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(2, deserialized.Rows);
        Assert.Equal(5, deserialized.Columns);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 5; j++)
                Assert.Equal(m[i, j], deserialized[i, j], Tolerance);
    }

    [Fact]
    public void Matrix_Roundtrip_SingleElement()
    {
        var m = new Matrix<double>(1, 1);
        m[0, 0] = -7.5;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(1, deserialized.Rows);
        Assert.Equal(1, deserialized.Columns);
        Assert.Equal(-7.5, deserialized[0, 0], Tolerance);
    }

    [Fact]
    public void Matrix_Roundtrip_SymmetricPreserved()
    {
        // Symmetric matrix: A = A^T
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 2.0; m[1, 1] = 5.0; m[1, 2] = 7.0;
        m[2, 0] = 3.0; m[2, 1] = 7.0; m[2, 2] = 9.0;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(deserialized);
        // Verify symmetry is preserved
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(deserialized[i, j], deserialized[j, i], Tolerance);
    }

    [Fact]
    public void Matrix_JsonStructure_ContainsRowsColumnsData()
    {
        var m = new Matrix<double>(2, 3);
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);

        Assert.Contains("\"rows\":2", json);
        Assert.Contains("\"columns\":3", json);
        Assert.Contains("\"data\":", json);
    }

    [Fact]
    public void Matrix_Deserialize_DimensionMismatch_Throws()
    {
        var settings = CreateSettings();
        // 2x2 = 4 elements but only 3 provided
        var invalidJson = "{\"rows\":2,\"columns\":2,\"data\":[1.0,2.0,3.0]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Matrix<double>>(invalidJson, settings));
    }

    [Fact]
    public void Matrix_Deserialize_MissingRows_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"columns\":2,\"data\":[1.0,2.0]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Matrix<double>>(invalidJson, settings));
    }

    [Fact]
    public void Matrix_Deserialize_NegativeDimension_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"rows\":-1,\"columns\":2,\"data\":[]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Matrix<double>>(invalidJson, settings));
    }

    // ============================
    // Tensor Roundtrip Tests
    // ============================

    [Fact]
    public void Tensor_Roundtrip_1D()
    {
        var t = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++)
            t[i] = i * 1.5;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(5, deserialized.Length);
        for (int i = 0; i < 5; i++)
            Assert.Equal(t[i], deserialized[i], Tolerance);
    }

    [Fact]
    public void Tensor_Roundtrip_2D()
    {
        var shape = new[] { 3, 4 };
        var t = new Tensor<double>(shape);
        for (int i = 0; i < 12; i++)
            t[i] = Math.Sin(i * 0.5);

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(12, deserialized.Length);
        for (int i = 0; i < 12; i++)
            Assert.Equal(t[i], deserialized[i], Tolerance);
    }

    [Fact]
    public void Tensor_Roundtrip_3D()
    {
        var shape = new[] { 2, 3, 4 };
        var t = new Tensor<double>(shape);
        for (int i = 0; i < 24; i++)
            t[i] = i * 0.1 - 1.2;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(24, deserialized.Length);
        for (int i = 0; i < 24; i++)
            Assert.Equal(t[i], deserialized[i], Tolerance);
    }

    [Fact]
    public void Tensor_Roundtrip_4D()
    {
        // Simulates a small batch of feature maps: [batch, channels, height, width]
        var shape = new[] { 2, 3, 2, 2 };
        var t = new Tensor<double>(shape);
        for (int i = 0; i < 24; i++)
            t[i] = Math.Cos(i);

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(24, deserialized.Length);
        for (int i = 0; i < 24; i++)
            Assert.Equal(t[i], deserialized[i], Tolerance);
    }

    [Fact]
    public void Tensor_Roundtrip_SingleElement()
    {
        var t = new Tensor<double>(new[] { 1 });
        t[0] = 99.99;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(1, deserialized.Length);
        Assert.Equal(99.99, deserialized[0], Tolerance);
    }

    [Fact]
    public void Tensor_Roundtrip_LargeData()
    {
        // 10x10x10 = 1000 elements
        var shape = new[] { 10, 10, 10 };
        var t = new Tensor<double>(shape);
        for (int i = 0; i < 1000; i++)
            t[i] = i * 0.001;

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(1000, deserialized.Length);
        for (int i = 0; i < 1000; i++)
            Assert.Equal(t[i], deserialized[i], Tolerance);
    }

    [Fact]
    public void Tensor_JsonStructure_ContainsShapeAndData()
    {
        var t = new Tensor<double>(new[] { 2, 3 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);

        Assert.Contains("\"shape\":", json);
        Assert.Contains("\"data\":", json);
    }

    [Fact]
    public void Tensor_Deserialize_ShapeMismatch_Throws()
    {
        var settings = CreateSettings();
        // Shape [2,3] = 6 elements but only 4 provided
        var invalidJson = "{\"shape\":[2,3],\"data\":[1.0,2.0,3.0,4.0]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Tensor<double>>(invalidJson, settings));
    }

    [Fact]
    public void Tensor_Deserialize_MissingShape_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"data\":[1.0,2.0]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Tensor<double>>(invalidJson, settings));
    }

    [Fact]
    public void Tensor_Deserialize_NegativeDimension_Throws()
    {
        var settings = CreateSettings();
        var invalidJson = "{\"shape\":[-1,3],\"data\":[]}";

        Assert.Throws<JsonSerializationException>(() =>
            JsonConvert.DeserializeObject<Tensor<double>>(invalidJson, settings));
    }

    // ============================
    // Cross-Type Consistency Tests
    // ============================

    [Fact]
    public void Vector_DoubleSerialization_IsIdempotent()
    {
        // Serialize -> Deserialize -> Serialize should produce identical JSON
        var v = new Vector<double>(new double[] { 1.0, Math.PI, -0.001 });
        var settings = CreateSettings();

        var json1 = JsonConvert.SerializeObject(v, settings);
        var roundtrip = JsonConvert.DeserializeObject<Vector<double>>(json1, settings);
        Assert.NotNull(roundtrip);
        var json2 = JsonConvert.SerializeObject(roundtrip, settings);

        Assert.Equal(json1, json2);
    }

    [Fact]
    public void Matrix_DoubleSerialization_IsIdempotent()
    {
        var m = new Matrix<double>(2, 2);
        m[0, 0] = 1.0; m[0, 1] = 2.0;
        m[1, 0] = 3.0; m[1, 1] = 4.0;
        var settings = CreateSettings();

        var json1 = JsonConvert.SerializeObject(m, settings);
        var roundtrip = JsonConvert.DeserializeObject<Matrix<double>>(json1, settings);
        Assert.NotNull(roundtrip);
        var json2 = JsonConvert.SerializeObject(roundtrip, settings);

        Assert.Equal(json1, json2);
    }

    [Fact]
    public void Tensor_DoubleSerialization_IsIdempotent()
    {
        var t = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++)
            t[i] = i + 0.5;
        var settings = CreateSettings();

        var json1 = JsonConvert.SerializeObject(t, settings);
        var roundtrip = JsonConvert.DeserializeObject<Tensor<double>>(json1, settings);
        Assert.NotNull(roundtrip);
        var json2 = JsonConvert.SerializeObject(roundtrip, settings);

        Assert.Equal(json1, json2);
    }

    // ============================
    // Mathematical Property Preservation Tests
    // ============================

    [Fact]
    public void Vector_Roundtrip_PreservesDotProduct()
    {
        // dot(v, w) before roundtrip == dot(v', w') after roundtrip
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var w = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

        var dotBefore = 0.0;
        for (int i = 0; i < 3; i++)
            dotBefore += v[i] * w[i];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        var settings = CreateSettings();

        var vJson = JsonConvert.SerializeObject(v, settings);
        var wJson = JsonConvert.SerializeObject(w, settings);
        var vRound = JsonConvert.DeserializeObject<Vector<double>>(vJson, settings);
        var wRound = JsonConvert.DeserializeObject<Vector<double>>(wJson, settings);

        Assert.NotNull(vRound);
        Assert.NotNull(wRound);

        var dotAfter = 0.0;
        for (int i = 0; i < 3; i++)
            dotAfter += vRound[i] * wRound[i];

        Assert.Equal(32.0, dotBefore, Tolerance);
        Assert.Equal(dotBefore, dotAfter, Tolerance);
    }

    [Fact]
    public void Matrix_Roundtrip_PreservesDeterminant2x2()
    {
        // det([[a,b],[c,d]]) = ad - bc
        var m = new Matrix<double>(2, 2);
        m[0, 0] = 3.0; m[0, 1] = 7.0;
        m[1, 0] = 1.0; m[1, 1] = -4.0;

        var detBefore = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]; // 3*(-4) - 7*1 = -19

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var mRound = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(mRound);

        var detAfter = mRound[0, 0] * mRound[1, 1] - mRound[0, 1] * mRound[1, 0];

        Assert.Equal(-19.0, detBefore, Tolerance);
        Assert.Equal(detBefore, detAfter, Tolerance);
    }

    [Fact]
    public void Matrix_Roundtrip_PreservesTrace()
    {
        // trace(A) = sum of diagonal elements
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 5.0; m[0, 1] = 1.0; m[0, 2] = 2.0;
        m[1, 0] = 3.0; m[1, 1] = -2.0; m[1, 2] = 4.0;
        m[2, 0] = 7.0; m[2, 1] = 8.0; m[2, 2] = 9.0;

        var traceBefore = m[0, 0] + m[1, 1] + m[2, 2]; // 5 + (-2) + 9 = 12

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(m, settings);
        var mRound = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.NotNull(mRound);

        var traceAfter = mRound[0, 0] + mRound[1, 1] + mRound[2, 2];

        Assert.Equal(12.0, traceBefore, Tolerance);
        Assert.Equal(traceBefore, traceAfter, Tolerance);
    }

    [Fact]
    public void Tensor_Roundtrip_PreservesSum()
    {
        var t = new Tensor<double>(new[] { 3, 4 });
        var expectedSum = 0.0;
        for (int i = 0; i < 12; i++)
        {
            t[i] = (i + 1) * 0.5; // 0.5, 1.0, 1.5, ..., 6.0
            expectedSum += t[i];
        }
        // Sum = 0.5 * (1+2+...+12) = 0.5 * 78 = 39.0

        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(t, settings);
        var tRound = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(tRound);

        var actualSum = 0.0;
        for (int i = 0; i < 12; i++)
            actualSum += tRound[i];

        Assert.Equal(39.0, expectedSum, Tolerance);
        Assert.Equal(expectedSum, actualSum, Tolerance);
    }

    // ============================
    // Null Handling Tests
    // ============================

    [Fact]
    public void Vector_SerializeNull_DeserializesNull()
    {
        var settings = CreateSettings();
        Vector<double>? v = null;

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.Null(deserialized);
    }

    [Fact]
    public void Matrix_SerializeNull_DeserializesNull()
    {
        var settings = CreateSettings();
        Matrix<double>? m = null;

        var json = JsonConvert.SerializeObject(m, settings);
        var deserialized = JsonConvert.DeserializeObject<Matrix<double>>(json, settings);

        Assert.Null(deserialized);
    }

    [Fact]
    public void Tensor_SerializeNull_DeserializesNull()
    {
        var settings = CreateSettings();
        Tensor<double>? t = null;

        var json = JsonConvert.SerializeObject(t, settings);
        var deserialized = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.Null(deserialized);
    }

    // ============================
    // CanConvert Tests
    // ============================

    [Fact]
    public void VectorConverter_CanConvert_VectorDouble_True()
    {
        var converter = new VectorJsonConverter();
        Assert.True(converter.CanConvert(typeof(Vector<double>)));
    }

    [Fact]
    public void VectorConverter_CanConvert_String_False()
    {
        var converter = new VectorJsonConverter();
        Assert.False(converter.CanConvert(typeof(string)));
    }

    [Fact]
    public void MatrixConverter_CanConvert_MatrixDouble_True()
    {
        var converter = new MatrixJsonConverter();
        Assert.True(converter.CanConvert(typeof(Matrix<double>)));
    }

    [Fact]
    public void MatrixConverter_CanConvert_Int_False()
    {
        var converter = new MatrixJsonConverter();
        Assert.False(converter.CanConvert(typeof(int)));
    }

    [Fact]
    public void TensorConverter_CanConvert_TensorDouble_True()
    {
        var converter = new TensorJsonConverter();
        Assert.True(converter.CanConvert(typeof(Tensor<double>)));
    }

    [Fact]
    public void TensorConverter_CanConvert_List_False()
    {
        var converter = new TensorJsonConverter();
        Assert.False(converter.CanConvert(typeof(List<double>)));
    }

    // ============================
    // Special Floating Point Values
    // ============================

    [Fact]
    public void Vector_Roundtrip_ZeroVector()
    {
        var v = new Vector<double>(10);
        // All zeros by default
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        Assert.Equal(10, deserialized.Length);
        for (int i = 0; i < 10; i++)
            Assert.Equal(0.0, deserialized[i], Tolerance);
    }

    [Fact]
    public void Vector_Roundtrip_AlternatingSign()
    {
        var v = new Vector<double>(new double[] { 1.0, -1.0, 1.0, -1.0, 1.0 });
        var settings = CreateSettings();

        var json = JsonConvert.SerializeObject(v, settings);
        var deserialized = JsonConvert.DeserializeObject<Vector<double>>(json, settings);

        Assert.NotNull(deserialized);
        for (int i = 0; i < 5; i++)
        {
            var expected = (i % 2 == 0) ? 1.0 : -1.0;
            Assert.Equal(expected, deserialized[i], Tolerance);
        }
    }

    // ============================
    // Roundtrip With Mixed Settings
    // ============================

    [Fact]
    public void AllConverters_RegisteredOnce_WorkTogether()
    {
        var settings = CreateSettings();

        var v = new Vector<double>(new double[] { 1.0, 2.0 });
        var m = new Matrix<double>(2, 2);
        m[0, 0] = 3.0; m[0, 1] = 4.0;
        m[1, 0] = 5.0; m[1, 1] = 6.0;
        var t = new Tensor<double>(new[] { 2 });
        t[0] = 7.0; t[1] = 8.0;

        var vJson = JsonConvert.SerializeObject(v, settings);
        var mJson = JsonConvert.SerializeObject(m, settings);
        var tJson = JsonConvert.SerializeObject(t, settings);

        var vRound = JsonConvert.DeserializeObject<Vector<double>>(vJson, settings);
        var mRound = JsonConvert.DeserializeObject<Matrix<double>>(mJson, settings);
        var tRound = JsonConvert.DeserializeObject<Tensor<double>>(tJson, settings);

        Assert.NotNull(vRound);
        Assert.NotNull(mRound);
        Assert.NotNull(tRound);
        Assert.Equal(1.0, vRound[0], Tolerance);
        Assert.Equal(3.0, mRound[0, 0], Tolerance);
        Assert.Equal(7.0, tRound[0], Tolerance);
    }
}
