using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Integration tests for ColumnTransformer.
/// Tests column-wise transformations, remainder handling, and mixed column types.
/// </summary>
public class ColumnTransformerIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(double[,] data)
    {
        return new Matrix<double>(data);
    }

    private static void AssertMatrixEqual(Matrix<double> expected, Matrix<double> actual, double tolerance = Tolerance)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);

        for (int i = 0; i < expected.Rows; i++)
        {
            for (int j = 0; j < expected.Columns; j++)
            {
                Assert.True(
                    Math.Abs(expected[i, j] - actual[i, j]) < tolerance,
                    $"Mismatch at [{i},{j}]: expected {expected[i, j]}, actual {actual[i, j]}");
            }
        }
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultRemainder_IsDrop()
    {
        // Act
        var transformer = new ColumnTransformer<double>();

        // Assert
        Assert.Equal(ColumnTransformerRemainder.Drop, transformer.Remainder);
        Assert.False(transformer.IsFitted);
    }

    [Fact]
    public void Constructor_WithPassthrough_SetsRemainder()
    {
        // Act
        var transformer = new ColumnTransformer<double>(ColumnTransformerRemainder.Passthrough);

        // Assert
        Assert.Equal(ColumnTransformerRemainder.Passthrough, transformer.Remainder);
    }

    #endregion

    #region Add Tests

    [Fact]
    public void Add_WithValidParameters_AddsTransformer()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();

        // Act
        transformer.Add("scaler", scaler, new[] { 0, 1 });

        // Assert
        var names = transformer.GetTransformerNames();
        Assert.Contains("scaler", names);
    }

    [Fact]
    public void Add_WithoutName_AddsWithAutomaticName()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();

        // Act
        transformer.Add(scaler, new[] { 0 });

        // Assert
        var names = transformer.GetTransformerNames();
        Assert.Equal("transformer_0", names[0]);
    }

    [Fact]
    public void Add_NullTransformer_ThrowsArgumentNullException()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transformer.Add("test", null!, new[] { 0 }));
    }

    [Fact]
    public void Add_NullColumns_ThrowsArgumentException()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transformer.Add("test", scaler, null!));
    }

    [Fact]
    public void Add_EmptyColumns_ThrowsArgumentException()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transformer.Add("test", scaler, Array.Empty<int>()));
    }

    [Fact]
    public void Add_EmptyName_ThrowsArgumentException()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transformer.Add("", scaler, new[] { 0 }));
    }

    [Fact]
    public void Add_ReturnsThisForChaining()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();

        // Act
        var result = transformer.Add(scaler, new[] { 0 });

        // Assert
        Assert.Same(transformer, result);
    }

    #endregion

    #region Fit Tests

    [Fact]
    public void Fit_InvalidColumnIndex_ThrowsArgumentException()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 5 }); // Column 5 doesn't exist
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } }); // Only 2 columns

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transformer.Fit(data));
    }

    [Fact]
    public void Fit_NegativeColumnIndex_ThrowsArgumentException()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { -1 }); // Invalid index
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transformer.Fit(data));
    }

    [Fact]
    public void Fit_SetsFittedToTrue()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0 });
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        transformer.Fit(data);

        // Assert
        Assert.True(transformer.IsFitted);
    }

    #endregion

    #region Transform Tests

    [Fact]
    public void Transform_SingleColumn_TransformsOnlySpecifiedColumn()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0 });
        var data = CreateTestMatrix(new double[,] { { 1, 100 }, { 3, 200 }, { 5, 300 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert - Only column 0 should be transformed, column 1 is dropped (default remainder=Drop)
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns); // Only 1 column output

        // Check column 0 is standardized (mean ~0)
        double mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(mean) < 1e-10);
    }

    [Fact]
    public void Transform_MultipleTransformers_AppliesEachToSpecifiedColumns()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add("standard", new StandardScaler<double>(), new[] { 0 });
        transformer.Add("minmax", new MinMaxScaler<double>(), new[] { 1 });

        var data = CreateTestMatrix(new double[,] { { 1, 0 }, { 3, 5 }, { 5, 10 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Column 0: StandardScaler (mean ~0)
        double col0Mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(col0Mean) < 1e-10);

        // Column 1: MinMaxScaler (range [0, 1])
        Assert.True(Math.Abs(result[0, 1] - 0.0) < 1e-10); // min maps to 0
        Assert.True(Math.Abs(result[2, 1] - 1.0) < 1e-10); // max maps to 1
    }

    [Fact]
    public void Transform_WithPassthrough_IncludesUnspecifiedColumns()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>(ColumnTransformerRemainder.Passthrough);
        transformer.Add(new StandardScaler<double>(), new[] { 0 });
        var data = CreateTestMatrix(new double[,] { { 1, 100, 1000 }, { 3, 200, 2000 }, { 5, 300, 3000 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert - Should have 3 columns (1 transformed + 2 passthrough)
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);

        // Column 0: StandardScaler (mean ~0)
        double col0Mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(col0Mean) < 1e-10, $"Column 0 mean should be ~0, got {col0Mean}");

        // Columns 1 and 2: Passthrough (original values)
        Assert.Equal(100, result[0, 1]);
        Assert.Equal(200, result[1, 1]);
        Assert.Equal(1000, result[0, 2]);
        Assert.Equal(2000, result[1, 2]);
    }

    [Fact]
    public void Transform_WithDrop_ExcludesUnspecifiedColumns()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>(ColumnTransformerRemainder.Drop);
        transformer.Add(new StandardScaler<double>(), new[] { 0 });
        var data = CreateTestMatrix(new double[,] { { 1, 100, 1000 }, { 3, 200, 2000 }, { 5, 300, 3000 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert - Should have only 1 column (transformed)
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void Transform_OverlappingColumns_ProcessesBothTransformers()
    {
        // Arrange - Note: Overlapping columns may have undefined behavior
        // This test documents the current behavior
        var transformer = new ColumnTransformer<double>();
        transformer.Add("first", new StandardScaler<double>(), new[] { 0, 1 });
        transformer.Add("second", new MinMaxScaler<double>(), new[] { 0 }); // Column 0 again

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert - Both transformers produce output
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns); // 2 from first + 1 from second
    }

    #endregion

    #region GetTransformer Tests

    [Fact]
    public void GetTransformer_ExistingName_ReturnsTransformer()
    {
        // Arrange
        var columnTransformer = new ColumnTransformer<double>();
        var scaler = new StandardScaler<double>();
        columnTransformer.Add("my_scaler", scaler, new[] { 0 });

        // Act
        var retrieved = columnTransformer.GetTransformer("my_scaler");

        // Assert
        Assert.Same(scaler, retrieved);
    }

    [Fact]
    public void GetTransformer_NonExistingName_ReturnsNull()
    {
        // Arrange
        var columnTransformer = new ColumnTransformer<double>();
        columnTransformer.Add("scaler", new StandardScaler<double>(), new[] { 0 });

        // Act
        var retrieved = columnTransformer.GetTransformer("nonexistent");

        // Assert
        Assert.Null(retrieved);
    }

    #endregion

    #region GetFeatureNamesOut Tests

    [Fact]
    public void GetFeatureNamesOut_WithInputNames_ReturnsTransformedNames()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add("scaler", new StandardScaler<double>(), new[] { 0, 1 });

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        transformer.Fit(data);

        var inputNames = new[] { "feature_a", "feature_b" };

        // Act
        var outputNames = transformer.GetFeatureNamesOut(inputNames);

        // Assert
        Assert.Equal(2, outputNames.Length);
        Assert.Equal("feature_a", outputNames[0]);
        Assert.Equal("feature_b", outputNames[1]);
    }

    [Fact]
    public void GetFeatureNamesOut_WithPassthrough_IncludesRemainderNames()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>(ColumnTransformerRemainder.Passthrough);
        transformer.Add("scaler", new StandardScaler<double>(), new[] { 0 });

        var data = CreateTestMatrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        transformer.Fit(data);

        var inputNames = new[] { "col_0", "col_1", "col_2" };

        // Act
        var outputNames = transformer.GetFeatureNamesOut(inputNames);

        // Assert - Should have 3 names (1 transformed + 2 passthrough)
        Assert.Equal(3, outputNames.Length);
        Assert.Equal("col_0", outputNames[0]); // Transformed
        Assert.Contains("col_1", outputNames); // Passthrough
        Assert.Contains("col_2", outputNames); // Passthrough
    }

    [Fact]
    public void GetFeatureNamesOut_BeforeFit_ReturnsEmptyArray()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0 });

        // Act
        var outputNames = transformer.GetFeatureNamesOut(new[] { "col_0", "col_1" });

        // Assert
        Assert.Empty(outputNames);
    }

    #endregion

    #region SupportsInverseTransform Tests

    [Fact]
    public void SupportsInverseTransform_ReturnsFalse()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();

        // Assert
        Assert.False(transformer.SupportsInverseTransform);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void FitTransform_AllColumnsSpecified_ProcessesAllColumns()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0, 1, 2 });

        var data = CreateTestMatrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void FitTransform_NonContiguousColumns_WorksCorrectly()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0, 2 }); // Skip column 1

        var data = CreateTestMatrix(new double[,] { { 1, 100, 2 }, { 3, 200, 4 }, { 5, 300, 6 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert - Should have 2 columns (columns 0 and 2 transformed)
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void FitTransform_ReversedColumnOrder_MaintainsOrder()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 2, 0 }); // Reversed order

        var data = CreateTestMatrix(new double[,] { { 1, 100, 10 }, { 2, 200, 20 }, { 3, 300, 30 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void Transform_NewDataWithDifferentRows_WorksCorrectly()
    {
        // Arrange
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0 });

        var trainData = CreateTestMatrix(new double[,] { { 1 }, { 3 }, { 5 } });
        var testData = CreateTestMatrix(new double[,] { { 3 } }); // Different number of rows

        // Act
        transformer.Fit(trainData);
        var result = transformer.Transform(testData);

        // Assert
        Assert.Equal(1, result.Rows);
        Assert.Equal(1, result.Columns);

        // Value 3 is the mean, so should transform to ~0
        Assert.True(Math.Abs(result[0, 0]) < 1e-10);
    }

    [Fact]
    public void FitTransform_DuplicateColumnInSingleTransformer_ProcessesCorrectly()
    {
        // Arrange - Duplicate column index
        var transformer = new ColumnTransformer<double>();
        transformer.Add(new StandardScaler<double>(), new[] { 0, 0 }); // Column 0 twice

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = transformer.FitTransform(data);

        // Assert - This may have unexpected behavior, document what happens
        // Expect 2 columns output (duplicate processing)
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    #endregion
}
