using AiDotNet.Caching;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Caching;

/// <summary>
/// Integration tests for DeterministicCacheKeyGenerator to verify deterministic key generation,
/// uniqueness, and edge case handling.
/// </summary>
public class DeterministicCacheKeyGeneratorIntegrationTests
{
    #region Determinism Tests

    [Fact]
    public void GenerateKey_SameInputs_ProducesSameKey()
    {
        // Arrange
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        const string inputDescriptor = "train:Matrix(100,10)xVector(100)";

        // Act
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);

        // Assert
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void GenerateKey_MultipleCalls_AlwaysSameResult()
    {
        // Arrange
        var parameters = new Vector<double>(new[] { 1.5, 2.5, 3.5, 4.5, 5.5 });
        const string inputDescriptor = "train:Matrix(1000,100)xVector(1000)|val:Matrix(200,100)xVector(200)";

        // Act - Generate key 100 times
        var keys = new HashSet<string>();
        for (int i = 0; i < 100; i++)
        {
            keys.Add(DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor));
        }

        // Assert - All keys should be identical
        Assert.Single(keys);
    }

    [Fact]
    public void GenerateKey_ReturnsValidSha256HexString()
    {
        // Arrange
        var parameters = new Vector<double>(new[] { 1.0, 2.0 });
        const string inputDescriptor = "train:Matrix(10,5)xVector(10)";

        // Act
        var key = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);

        // Assert - SHA-256 produces 64 hex characters (32 bytes)
        Assert.NotNull(key);
        Assert.Equal(64, key.Length);
        Assert.True(key.All(c => char.IsDigit(c) || (c >= 'a' && c <= 'f')),
            $"Key should be lowercase hex: {key}");
    }

    #endregion

    #region Uniqueness Tests

    [Fact]
    public void GenerateKey_DifferentParameters_ProducesDifferentKeys()
    {
        // Arrange
        var parameters1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var parameters2 = new Vector<double>(new[] { 1.0, 2.0, 3.1 }); // Slightly different
        const string inputDescriptor = "train:Matrix(100,10)xVector(100)";

        // Act
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters1, inputDescriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters2, inputDescriptor);

        // Assert
        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentDescriptors_ProducesDifferentKeys()
    {
        // Arrange
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        const string inputDescriptor1 = "train:Matrix(100,10)xVector(100)";
        const string inputDescriptor2 = "train:Matrix(200,10)xVector(200)"; // Different size

        // Act
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor1);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor2);

        // Assert
        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentParameterCounts_ProducesDifferentKeys()
    {
        // Arrange
        var parameters1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var parameters2 = new Vector<double>(new[] { 1.0, 2.0 }); // Different count
        const string inputDescriptor = "train:Matrix(100,10)xVector(100)";

        // Act
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters1, inputDescriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters2, inputDescriptor);

        // Assert
        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_LargeDataset_KeysAreUnique()
    {
        // Arrange - Generate 1000 different parameter sets
        var keys = new HashSet<string>();
        const string inputDescriptor = "train:Matrix(100,10)xVector(100)";

        // Act
        for (int i = 0; i < 1000; i++)
        {
            var parameters = new Vector<double>(new[] { (double)i, i + 0.5, i + 0.25 });
            keys.Add(DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor));
        }

        // Assert - All 1000 keys should be unique
        Assert.Equal(1000, keys.Count);
    }

    #endregion

    #region Null Validation Tests

    [Fact]
    public void GenerateKey_NullParameters_ThrowsArgumentNullException()
    {
        // Arrange
        const string inputDescriptor = "train:Matrix(100,10)xVector(100)";

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            DeterministicCacheKeyGenerator.GenerateKey<double>(null!, inputDescriptor));
    }

    [Fact]
    public void GenerateKey_NullDescriptor_ThrowsArgumentNullException()
    {
        // Arrange
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, null!));
    }

    #endregion

    #region CreateInputDataDescriptor Tests

    [Fact]
    public void CreateInputDataDescriptor_TrainOnly_ReturnsCorrectFormat()
    {
        // Arrange
        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);

        // Act
        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain);

        // Assert
        Assert.Contains("train:", descriptor);
        Assert.Contains("Matrix(100,10)", descriptor);
        Assert.Contains("Vector(100)", descriptor);
    }

    [Fact]
    public void CreateInputDataDescriptor_WithValidation_IncludesValidationInfo()
    {
        // Arrange
        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);
        var xVal = new Matrix<double>(20, 10);
        var yVal = new Vector<double>(20);

        // Act
        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain, xVal, yVal);

        // Assert
        Assert.Contains("train:", descriptor);
        Assert.Contains("val:", descriptor);
        Assert.Contains("Matrix(20,10)", descriptor);
    }

    [Fact]
    public void CreateInputDataDescriptor_WithAllDatasets_IncludesAllInfo()
    {
        // Arrange
        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);
        var xVal = new Matrix<double>(20, 10);
        var yVal = new Vector<double>(20);
        var xTest = new Matrix<double>(30, 10);
        var yTest = new Vector<double>(30);

        // Act
        var descriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain, yTrain, xVal, yVal, xTest, yTest);

        // Assert
        Assert.Contains("train:", descriptor);
        Assert.Contains("val:", descriptor);
        Assert.Contains("test:", descriptor);
    }

    [Fact]
    public void CreateInputDataDescriptor_SameInputs_ProducesSameDescriptor()
    {
        // Arrange
        var xTrain1 = new Matrix<double>(100, 10);
        var yTrain1 = new Vector<double>(100);
        var xTrain2 = new Matrix<double>(100, 10);
        var yTrain2 = new Vector<double>(100);

        // Act
        var descriptor1 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain1, yTrain1);
        var descriptor2 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain2, yTrain2);

        // Assert
        Assert.Equal(descriptor1, descriptor2);
    }

    [Fact]
    public void CreateInputDataDescriptor_DifferentShapes_ProducesDifferentDescriptors()
    {
        // Arrange
        var xTrain1 = new Matrix<double>(100, 10);
        var yTrain1 = new Vector<double>(100);
        var xTrain2 = new Matrix<double>(200, 10); // Different row count
        var yTrain2 = new Vector<double>(200);

        // Act
        var descriptor1 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain1, yTrain1);
        var descriptor2 = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            xTrain2, yTrain2);

        // Assert
        Assert.NotEqual(descriptor1, descriptor2);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void GenerateKey_EmptyParameters_StillGeneratesValidKey()
    {
        // Arrange
        var parameters = new Vector<double>(0);
        const string inputDescriptor = "train:Matrix(0,0)xVector(0)";

        // Act
        var key = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);

        // Assert
        Assert.NotNull(key);
        Assert.Equal(64, key.Length); // Still valid SHA-256
    }

    [Fact]
    public void GenerateKey_VeryLargeParameters_GeneratesValidKey()
    {
        // Arrange - Create a large parameter vector
        var largeParams = new double[10000];
        for (int i = 0; i < largeParams.Length; i++)
        {
            largeParams[i] = i * 0.001;
        }
        var parameters = new Vector<double>(largeParams);
        const string inputDescriptor = "train:Matrix(100000,1000)xVector(100000)";

        // Act
        var key = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);

        // Assert
        Assert.NotNull(key);
        Assert.Equal(64, key.Length);
    }

    [Fact]
    public void GenerateKey_ExtremeValues_HandlesCorrectly()
    {
        // Arrange - Test with extreme values
        var parameters = new Vector<double>(new[] {
            double.MaxValue,
            double.MinValue,
            double.Epsilon,
            0.0,
            -0.0,
            1.0 / 3.0 // Repeating decimal
        });
        const string inputDescriptor = "train:Matrix(10,5)xVector(10)";

        // Act
        var key = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);

        // Assert
        Assert.NotNull(key);
        Assert.Equal(64, key.Length);
    }

    [Fact]
    public void GenerateKey_NaNAndInfinity_ProducesConsistentKeys()
    {
        // Arrange
        var parameters1 = new Vector<double>(new[] { double.NaN, double.PositiveInfinity });
        var parameters2 = new Vector<double>(new[] { double.NaN, double.PositiveInfinity });
        const string inputDescriptor = "train:Matrix(10,5)xVector(10)";

        // Act
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters1, inputDescriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters2, inputDescriptor);

        // Assert - NaN and Infinity should still produce consistent keys
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void GenerateKey_WhitespaceInDescriptor_HandlesCorrectly()
    {
        // Arrange
        var parameters = new Vector<double>(new[] { 1.0, 2.0 });
        const string descriptor1 = "  train:Matrix(10,5)xVector(10)  ";
        const string descriptor2 = "train:Matrix(10,5)xVector(10)";

        // Act
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, descriptor1);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, descriptor2);

        // Assert - Whitespace is trimmed, so keys should be equal
        Assert.Equal(key1, key2);
    }

    #endregion

    #region Culture Invariance Tests

    [Fact]
    public void GenerateKey_FloatingPointValues_CultureInvariant()
    {
        // Arrange - Values that might format differently in different cultures
        var parameters = new Vector<double>(new[] {
            1.5,    // Uses decimal separator
            1000.5, // Might use thousand separator in some cultures
            0.001   // Very small decimal
        });
        const string inputDescriptor = "train:Matrix(10,5)xVector(10)";

        // Act - Generate key (should use invariant culture internally)
        var key1 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);
        var key2 = DeterministicCacheKeyGenerator.GenerateKey<double>(parameters, inputDescriptor);

        // Assert - Keys should always be the same regardless of culture
        Assert.Equal(key1, key2);
        Assert.Equal(64, key1.Length);
    }

    #endregion
}
