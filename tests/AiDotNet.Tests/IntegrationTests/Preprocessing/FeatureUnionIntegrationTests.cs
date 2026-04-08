using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Integration tests for FeatureUnion.
/// Tests parallel transformations and feature concatenation.
/// </summary>
public class FeatureUnionIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(double[,] data)
    {
        return new Matrix<double>(data);
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_CreatesEmptyUnion()
    {
        // Act
        var union = new FeatureUnion<double>();

        // Assert
        Assert.False(union.IsFitted);
        Assert.Empty(union.GetTransformerNames());
    }

    #endregion

    #region Add Tests

    [Fact]
    public void Add_WithName_AddsTransformerWithName()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler = new StandardScaler<double>();

        // Act
        union.Add("my_scaler", scaler);

        // Assert
        var names = union.GetTransformerNames();
        Assert.Single(names);
        Assert.Equal("my_scaler", names[0]);
    }

    [Fact]
    public void Add_WithoutName_AddsWithAutomaticName()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler = new StandardScaler<double>();

        // Act
        union.Add(scaler);

        // Assert
        var names = union.GetTransformerNames();
        Assert.Equal("transformer_0", names[0]);
    }

    [Fact]
    public void Add_NullTransformer_ThrowsArgumentNullException()
    {
        // Arrange
        var union = new FeatureUnion<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => union.Add("test", null!));
    }

    [Fact]
    public void Add_EmptyName_ThrowsArgumentException()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => union.Add("", scaler));
    }

    [Fact]
    public void Add_WhitespaceName_ThrowsArgumentException()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => union.Add("   ", scaler));
    }

    [Fact]
    public void Add_ReturnsThisForChaining()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler = new StandardScaler<double>();

        // Act
        var result = union.Add(scaler);

        // Assert
        Assert.Same(union, result);
    }

    [Fact]
    public void Add_MultipleTransformers_ChainsCalls()
    {
        // Arrange
        var union = new FeatureUnion<double>();

        // Act
        union.Add("scaler1", new StandardScaler<double>())
             .Add("scaler2", new MinMaxScaler<double>());

        // Assert
        var names = union.GetTransformerNames();
        Assert.Equal(2, names.Length);
    }

    #endregion

    #region Fit Tests

    [Fact]
    public void Fit_NoTransformers_ThrowsInvalidOperationException()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => union.Fit(data));
    }

    [Fact]
    public void Fit_SetsFittedToTrue()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        union.Fit(data);

        // Assert
        Assert.True(union.IsFitted);
    }

    [Fact]
    public void Fit_FitsAllTransformers()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler1 = new StandardScaler<double>();
        var scaler2 = new MinMaxScaler<double>();
        union.Add(scaler1);
        union.Add(scaler2);
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        union.Fit(data);

        // Assert
        Assert.True(scaler1.IsFitted);
        Assert.True(scaler2.IsFitted);
    }

    #endregion

    #region Transform Tests

    [Fact]
    public void Transform_SingleTransformer_ReturnsTransformedOutput()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = union.FitTransform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Check standardization (mean ~0)
        double col0Mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(col0Mean) < 1e-10);
    }

    [Fact]
    public void Transform_MultipleTransformers_ConcatenatesOutputs()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>()); // Outputs 2 columns
        union.Add(new MinMaxScaler<double>());   // Outputs 2 columns
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = union.FitTransform(data);

        // Assert - Should have 4 columns (2 + 2)
        Assert.Equal(3, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    [Fact]
    public void Transform_AllTransformersReceiveSameInput()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add("standard", new StandardScaler<double>());
        union.Add("minmax", new MinMaxScaler<double>());

        // Input with known values
        var data = CreateTestMatrix(new double[,] { { 0 }, { 5 }, { 10 } });

        // Act
        var result = union.FitTransform(data);

        // Assert - Both transformers should have processed the same data
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Column 0: StandardScaler (mean ~0)
        double stdMean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(stdMean) < 1e-10, $"StandardScaler mean should be ~0, got {stdMean}");

        // Column 1: MinMaxScaler (range [0, 1])
        Assert.True(Math.Abs(result[0, 1] - 0.0) < 1e-10, "MinMaxScaler min should be 0");
        Assert.True(Math.Abs(result[2, 1] - 1.0) < 1e-10, "MinMaxScaler max should be 1");
    }

    [Fact]
    public void Transform_WithoutFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => union.Transform(data));
    }

    #endregion

    #region GetTransformer Tests

    [Fact]
    public void GetTransformer_ExistingName_ReturnsTransformer()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        var scaler = new StandardScaler<double>();
        union.Add("my_scaler", scaler);

        // Act
        var retrieved = union.GetTransformer("my_scaler");

        // Assert
        Assert.Same(scaler, retrieved);
    }

    [Fact]
    public void GetTransformer_NonExistingName_ReturnsNull()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add("scaler", new StandardScaler<double>());

        // Act
        var retrieved = union.GetTransformer("nonexistent");

        // Assert
        Assert.Null(retrieved);
    }

    #endregion

    #region GetTransformerOutputWidths Tests

    [Fact]
    public void GetTransformerOutputWidths_BeforeFit_ReturnsEmptyDictionary()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add("scaler", new StandardScaler<double>());

        // Act
        var widths = union.GetTransformerOutputWidths();

        // Assert
        Assert.Empty(widths);
    }

    [Fact]
    public void GetTransformerOutputWidths_AfterFit_ReturnsCorrectWidths()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add("scaler1", new StandardScaler<double>()); // 2 columns in, 2 out
        union.Add("scaler2", new MinMaxScaler<double>());   // 2 columns in, 2 out
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act
        union.Fit(data);
        var widths = union.GetTransformerOutputWidths();

        // Assert
        Assert.Equal(2, widths.Count);
        Assert.Equal(2, widths["scaler1"]);
        Assert.Equal(2, widths["scaler2"]);
    }

    #endregion

    #region GetFeatureNamesOut Tests

    [Fact]
    public void GetFeatureNamesOut_PrefixesWithTransformerName()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add("standard", new StandardScaler<double>());
        union.Add("minmax", new MinMaxScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        union.Fit(data);

        var inputNames = new[] { "col_a", "col_b" };

        // Act
        var outputNames = union.GetFeatureNamesOut(inputNames);

        // Assert
        Assert.Equal(4, outputNames.Length);
        Assert.Equal("standard__col_a", outputNames[0]);
        Assert.Equal("standard__col_b", outputNames[1]);
        Assert.Equal("minmax__col_a", outputNames[2]);
        Assert.Equal("minmax__col_b", outputNames[3]);
    }

    [Fact]
    public void GetFeatureNamesOut_BeforeFit_ReturnsEmptyArray()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>());

        // Act
        var outputNames = union.GetFeatureNamesOut(new[] { "col_0" });

        // Assert
        Assert.Empty(outputNames);
    }

    #endregion

    #region SupportsInverseTransform Tests

    [Fact]
    public void SupportsInverseTransform_ReturnsFalse()
    {
        // Arrange
        var union = new FeatureUnion<double>();

        // Assert
        Assert.False(union.SupportsInverseTransform);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void FitTransform_SingleColumnInput_WorksCorrectly()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>());
        union.Add(new MinMaxScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1 }, { 3 }, { 5 } });

        // Act
        var result = union.FitTransform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns); // 1 + 1
    }

    [Fact]
    public void FitTransform_SingleRowInput_WorksCorrectly()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new MinMaxScaler<double>()); // Use MinMax since StandardScaler may have issues with single row
        var data = CreateTestMatrix(new double[,] { { 1, 2, 3 } });

        // Act
        var result = union.FitTransform(data);

        // Assert
        Assert.Equal(1, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void FitTransform_ManyTransformers_ConcatenatesAll()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        for (int i = 0; i < 10; i++)
        {
            union.Add($"scaler_{i}", new StandardScaler<double>());
        }

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = union.FitTransform(data);

        // Assert - Should have 20 columns (10 * 2)
        Assert.Equal(3, result.Rows);
        Assert.Equal(20, result.Columns);
    }

    [Fact]
    public void Transform_DifferentRowCount_WorksCorrectly()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new StandardScaler<double>());

        var trainData = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var testData = CreateTestMatrix(new double[,] { { 3, 4 } }); // Single row

        // Act
        union.Fit(trainData);
        var result = union.Transform(testData);

        // Assert
        Assert.Equal(1, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void FitTransform_PreservesRowOrder()
    {
        // Arrange
        var union = new FeatureUnion<double>();
        union.Add(new MinMaxScaler<double>());

        // Data where row order matters
        var data = CreateTestMatrix(new double[,] { { 0, 0 }, { 5, 5 }, { 10, 10 } });

        // Act
        var result = union.FitTransform(data);

        // Assert - First row should be all 0s, last row should be all 1s
        Assert.True(Math.Abs(result[0, 0] - 0.0) < 1e-10);
        Assert.True(Math.Abs(result[0, 1] - 0.0) < 1e-10);
        Assert.True(Math.Abs(result[2, 0] - 1.0) < 1e-10);
        Assert.True(Math.Abs(result[2, 1] - 1.0) < 1e-10);
    }

    #endregion
}
