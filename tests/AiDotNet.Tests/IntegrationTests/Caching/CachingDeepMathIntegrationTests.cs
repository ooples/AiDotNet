using AiDotNet.Caching;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Caching;

/// <summary>
/// Deep integration tests for DeterministicCacheKeyGenerator:
/// determinism, uniqueness, SHA-256 properties, input sensitivity,
/// boundary conditions, and collision resistance.
/// </summary>
public class CachingDeepMathIntegrationTests
{
    // ============================
    // Determinism Tests
    // ============================

    [Fact]
    public void GenerateKey_SameInputs_SameKey()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v, "test");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v, "test");

        Assert.Equal(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentCalls_SameResult()
    {
        // Create two separate but identical vectors
        var v1 = new Vector<double>(new double[] { 1.5, 2.5, 3.5 });
        var v2 = new Vector<double>(new double[] { 1.5, 2.5, 3.5 });

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v1, "descriptor");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v2, "descriptor");

        Assert.Equal(key1, key2);
    }

    [Fact]
    public void GenerateKey_MultipleCallsSameInstance_Consistent()
    {
        var v = new Vector<double>(new double[] { 0.0, 1.0 });
        var keys = new HashSet<string>();

        for (int i = 0; i < 100; i++)
            keys.Add(DeterministicCacheKeyGenerator.GenerateKey(v, "test"));

        // All 100 calls should produce the same key
        Assert.Single(keys);
    }

    // ============================
    // SHA-256 Format Tests
    // ============================

    [Fact]
    public void GenerateKey_Returns64CharHexString()
    {
        // SHA-256 produces 256 bits = 32 bytes = 64 hex chars
        var v = new Vector<double>(new double[] { 1.0 });

        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");

        Assert.Equal(64, key.Length);
    }

    [Fact]
    public void GenerateKey_LowercaseHex()
    {
        var v = new Vector<double>(new double[] { 1.0 });

        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");

        // Should only contain lowercase hex characters
        Assert.Matches("^[0-9a-f]{64}$", key);
    }

    // ============================
    // Input Sensitivity Tests
    // ============================

    [Fact]
    public void GenerateKey_DifferentVectors_DifferentKeys()
    {
        var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new double[] { 1.0, 2.0, 4.0 }); // Changed last element

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v1, "test");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v2, "test");

        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentDescriptors_DifferentKeys()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0 });

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v, "descriptorA");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v, "descriptorB");

        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentVectorLengths_DifferentKeys()
    {
        var v1 = new Vector<double>(new double[] { 1.0, 2.0 });
        var v2 = new Vector<double>(new double[] { 1.0, 2.0, 0.0 }); // Extra zero

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v1, "test");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v2, "test");

        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_SmallValueDifference_DifferentKeys()
    {
        // Even tiny value differences should produce different keys
        var v1 = new Vector<double>(new double[] { 1.0 });
        var v2 = new Vector<double>(new double[] { 1.0000000000000001 });

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v1, "test");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v2, "test");

        // May or may not differ due to double precision, but if values differ...
        if (v1[0] != v2[0])
        {
            Assert.NotEqual(key1, key2);
        }
    }

    [Fact]
    public void GenerateKey_ZeroVector_ValidKey()
    {
        var v = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");

        Assert.Equal(64, key.Length);
        Assert.Matches("^[0-9a-f]{64}$", key);
    }

    [Fact]
    public void GenerateKey_NegativeValues_ValidKey()
    {
        var v = new Vector<double>(new double[] { -1.0, -2.5, -100.0 });

        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");

        Assert.Equal(64, key.Length);
    }

    [Fact]
    public void GenerateKey_LargeValues_ValidKey()
    {
        var v = new Vector<double>(new double[] { 1e15, -1e15, 1e-15 });

        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");

        Assert.Equal(64, key.Length);
    }

    // ============================
    // Collision Resistance Tests
    // ============================

    [Fact]
    public void GenerateKey_ManyDistinctInputs_AllDistinctKeys()
    {
        var keys = new HashSet<string>();

        for (int i = 0; i < 100; i++)
        {
            var v = new Vector<double>(new double[] { i * 0.1 });
            var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");
            keys.Add(key);
        }

        // All 100 keys should be unique
        Assert.Equal(100, keys.Count);
    }

    [Fact]
    public void GenerateKey_DifferentLengthVectors_AllDifferent()
    {
        var keys = new HashSet<string>();

        for (int len = 1; len <= 10; len++)
        {
            var data = new double[len];
            for (int i = 0; i < len; i++)
                data[i] = 1.0;

            var v = new Vector<double>(data);
            var key = DeterministicCacheKeyGenerator.GenerateKey(v, "test");
            keys.Add(key);
        }

        // All 10 keys should be unique (different lengths)
        Assert.Equal(10, keys.Count);
    }

    // ============================
    // Descriptor Sensitivity Tests
    // ============================

    [Fact]
    public void GenerateKey_EmptyDescriptor_ValidKey()
    {
        var v = new Vector<double>(new double[] { 1.0 });

        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "");

        Assert.Equal(64, key.Length);
    }

    [Fact]
    public void GenerateKey_WhitespaceDescriptor_TrimmedSame()
    {
        var v = new Vector<double>(new double[] { 1.0 });

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(v, "test");
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(v, "  test  ");

        // Descriptors are trimmed, so these should match
        Assert.Equal(key1, key2);
    }

    // ============================
    // Null Input Tests
    // ============================

    [Fact]
    public void GenerateKey_NullVector_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            DeterministicCacheKeyGenerator.GenerateKey<double>(null!, "test"));
    }

    [Fact]
    public void GenerateKey_NullDescriptor_Throws()
    {
        var v = new Vector<double>(new double[] { 1.0 });
        Assert.Throws<ArgumentNullException>(() =>
            DeterministicCacheKeyGenerator.GenerateKey(v, null!));
    }

    // ============================
    // CreateInputDataDescriptor Tests
    // ============================

    [Fact]
    public void CreateDescriptor_MatrixInput_ContainsShape()
    {
        var xTrain = new Matrix<double>(10, 5);
        var yTrain = new Vector<double>(10);

        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain);

        Assert.Contains("Matrix(10,5)", descriptor);
        Assert.Contains("Vector(10)", descriptor);
    }

    [Fact]
    public void CreateDescriptor_WithValidation_IncludesValSection()
    {
        var xTrain = new Matrix<double>(10, 5);
        var yTrain = new Vector<double>(10);
        var xVal = new Matrix<double>(3, 5);
        var yVal = new Vector<double>(3);

        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain, xVal, yVal);

        Assert.Contains("train:", descriptor);
        Assert.Contains("val:", descriptor);
    }

    [Fact]
    public void CreateDescriptor_WithTest_IncludesTestSection()
    {
        var xTrain = new Matrix<double>(10, 5);
        var yTrain = new Vector<double>(10);
        var xTest = new Matrix<double>(2, 5);
        var yTest = new Vector<double>(2);

        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain, xTest: xTest, yTest: yTest);

        Assert.Contains("test:", descriptor);
    }

    [Fact]
    public void CreateDescriptor_SameInputs_SameDescriptor()
    {
        var xTrain1 = new Matrix<double>(10, 5);
        var yTrain1 = new Vector<double>(10);
        var xTrain2 = new Matrix<double>(10, 5);
        var yTrain2 = new Vector<double>(10);

        var desc1 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain1, yTrain1);
        var desc2 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain2, yTrain2);

        Assert.Equal(desc1, desc2);
    }

    [Fact]
    public void CreateDescriptor_DifferentShapes_DifferentDescriptors()
    {
        var xTrain1 = new Matrix<double>(10, 5);
        var yTrain1 = new Vector<double>(10);
        var xTrain2 = new Matrix<double>(10, 6); // Different columns
        var yTrain2 = new Vector<double>(10);

        var desc1 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain1, yTrain1);
        var desc2 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain2, yTrain2);

        Assert.NotEqual(desc1, desc2);
    }

    // ============================
    // End-to-End Cache Key Tests
    // ============================

    [Fact]
    public void FullCacheKeyPipeline_Deterministic()
    {
        // Simulate full pipeline: descriptor + key generation
        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);

        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain);

        var params1 = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        var params2 = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(params1, descriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(params2, descriptor);

        Assert.Equal(key1, key2);
    }

    [Fact]
    public void FullCacheKeyPipeline_ChangedParams_DifferentKey()
    {
        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);

        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain);

        var params1 = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        var params2 = new Vector<double>(new double[] { 0.1, 0.2, 0.4 }); // Changed

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(params1, descriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(params2, descriptor);

        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void FullCacheKeyPipeline_ChangedData_DifferentKey()
    {
        var modelParams = new Vector<double>(new double[] { 0.1, 0.2 });

        var xTrain1 = new Matrix<double>(100, 10);
        var yTrain1 = new Vector<double>(100);
        var desc1 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain1, yTrain1);

        var xTrain2 = new Matrix<double>(200, 10); // Different row count
        var yTrain2 = new Vector<double>(200);
        var desc2 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain2, yTrain2);

        var key1 = DeterministicCacheKeyGenerator.GenerateKey(modelParams, desc1);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey(modelParams, desc2);

        Assert.NotEqual(key1, key2);
    }

    // ============================
    // Large Vector Tests
    // ============================

    [Fact]
    public void GenerateKey_LargeVector_ValidKey()
    {
        var data = new double[1000];
        for (int i = 0; i < 1000; i++)
            data[i] = i * 0.001;

        var v = new Vector<double>(data);
        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "large");

        Assert.Equal(64, key.Length);
        Assert.Matches("^[0-9a-f]{64}$", key);
    }

    [Fact]
    public void GenerateKey_SingleElement_ValidKey()
    {
        var v = new Vector<double>(new double[] { 42.0 });
        var key = DeterministicCacheKeyGenerator.GenerateKey(v, "single");

        Assert.Equal(64, key.Length);
    }
}
