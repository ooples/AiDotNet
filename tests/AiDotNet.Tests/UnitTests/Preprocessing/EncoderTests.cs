using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Unit tests for encoder classes: OneHotEncoder, LabelEncoder, and OrdinalEncoder.
/// </summary>
public class EncoderTests
{
    private const double Tolerance = 1e-10;

    #region OneHotEncoder Tests

    [Fact]
    public void OneHotEncoder_BasicEncoding_Works()
    {
        // Arrange - Values 1, 2, 3 should become one-hot vectors
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 1.0 }
        });

        var encoder = new OneHotEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(4, result.Rows);
        Assert.Equal(3, result.Columns); // 3 unique values

        // Row 0: value 1 -> [1, 0, 0]
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[0, 1], Tolerance);
        Assert.Equal(0.0, result[0, 2], Tolerance);

        // Row 1: value 2 -> [0, 1, 0]
        Assert.Equal(0.0, result[1, 0], Tolerance);
        Assert.Equal(1.0, result[1, 1], Tolerance);
        Assert.Equal(0.0, result[1, 2], Tolerance);

        // Row 2: value 3 -> [0, 0, 1]
        Assert.Equal(0.0, result[2, 0], Tolerance);
        Assert.Equal(0.0, result[2, 1], Tolerance);
        Assert.Equal(1.0, result[2, 2], Tolerance);

        // Row 3: value 1 -> [1, 0, 0]
        Assert.Equal(1.0, result[3, 0], Tolerance);
        Assert.Equal(0.0, result[3, 1], Tolerance);
        Assert.Equal(0.0, result[3, 2], Tolerance);
    }

    [Fact]
    public void OneHotEncoder_DropFirst_AvoidsMulticollinearity()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 }
        });

        var encoder = new OneHotEncoder<double>(dropFirst: true);

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - Only 2 columns (3 categories - 1)
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Row 0: value 1 (first category) -> [0, 0] (all zeros when dropped)
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[0, 1], Tolerance);

        // Row 1: value 2 -> [1, 0]
        Assert.Equal(1.0, result[1, 0], Tolerance);
        Assert.Equal(0.0, result[1, 1], Tolerance);

        // Row 2: value 3 -> [0, 1]
        Assert.Equal(0.0, result[2, 0], Tolerance);
        Assert.Equal(1.0, result[2, 1], Tolerance);
    }

    [Fact]
    public void OneHotEncoder_UnknownCategory_ErrorHandling()
    {
        // Arrange
        var trainData = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 }
        });
        var testData = new Matrix<double>(new double[,]
        {
            { 3.0 } // Unknown category
        });

        var encoder = new OneHotEncoder<double>(handleUnknown: OneHotUnknownHandling.Error);

        // Act
        encoder.Fit(trainData);

        // Assert
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    [Fact]
    public void OneHotEncoder_UnknownCategory_IgnoreHandling()
    {
        // Arrange
        var trainData = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 }
        });
        var testData = new Matrix<double>(new double[,]
        {
            { 3.0 } // Unknown category
        });

        var encoder = new OneHotEncoder<double>(handleUnknown: OneHotUnknownHandling.Ignore);

        // Act
        encoder.Fit(trainData);
        var result = encoder.Transform(testData);

        // Assert - All zeros for unknown category
        Assert.Equal(1, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[0, 1], Tolerance);
    }

    [Fact]
    public void OneHotEncoder_InverseTransform_ReturnsOriginalValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 1.0 }
        });

        var encoder = new OneHotEncoder<double>();

        // Act
        encoder.Fit(data);
        var encoded = encoder.Transform(data);
        var decoded = encoder.InverseTransform(encoded);

        // Assert
        Assert.Equal(1.0, decoded[0, 0], Tolerance);
        Assert.Equal(2.0, decoded[1, 0], Tolerance);
        Assert.Equal(3.0, decoded[2, 0], Tolerance);
        Assert.Equal(1.0, decoded[3, 0], Tolerance);
    }

    [Fact]
    public void OneHotEncoder_MultipleColumns_EncodesAllSelected()
    {
        // Arrange - Two categorical columns
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { 1.0, 10.0 }
        });

        var encoder = new OneHotEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - 2 categories in col0 + 2 categories in col1 = 4 columns
        Assert.Equal(3, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    [Fact]
    public void OneHotEncoder_SpecificColumns_OnlyEncodesSelected()
    {
        // Arrange - Only encode column 0
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 100.0 },
            { 2.0, 200.0 }
        });

        var encoder = new OneHotEncoder<double>(columnIndices: new[] { 0 });

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - 2 one-hot columns + 1 pass-through column = 3 columns
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
        // Last column should be pass-through values
        Assert.Equal(100.0, result[0, 2], Tolerance);
        Assert.Equal(200.0, result[1, 2], Tolerance);
    }

    [Fact]
    public void OneHotEncoder_Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0 } });
        var encoder = new OneHotEncoder<double>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => encoder.Transform(data));
    }

    [Fact]
    public void OneHotEncoder_GetFeatureNamesOut_GeneratesCorrectNames()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 }
        });

        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);

        // Act
        var names = encoder.GetFeatureNamesOut(new[] { "color" });

        // Assert
        Assert.Equal(3, names.Length);
        Assert.Equal("color_1", names[0]);
        Assert.Equal("color_2", names[1]);
        Assert.Equal("color_3", names[2]);
    }

    [Fact]
    public void OneHotEncoder_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var encoder = new OneHotEncoder<double>();

        // Assert
        Assert.True(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void OneHotEncoder_Properties_ReturnCorrectValues()
    {
        // Arrange
        var encoder = new OneHotEncoder<double>(dropFirst: true, handleUnknown: OneHotUnknownHandling.Ignore);

        // Assert
        Assert.True(encoder.DropFirst);
        Assert.Equal(OneHotUnknownHandling.Ignore, encoder.HandleUnknown);
    }

    #endregion

    #region LabelEncoder Tests

    [Fact]
    public void LabelEncoder_BasicEncoding_Works()
    {
        // Arrange - Values get encoded as 0, 1, 2
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 },
            { 20.0 },
            { 10.0 }
        });

        var encoder = new LabelEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - Sorted order: 10=0, 20=1, 30=2
        Assert.Equal(2.0, result[0, 0], Tolerance); // 30 -> 2
        Assert.Equal(0.0, result[1, 0], Tolerance); // 10 -> 0
        Assert.Equal(1.0, result[2, 0], Tolerance); // 20 -> 1
        Assert.Equal(0.0, result[3, 0], Tolerance); // 10 -> 0
    }

    [Fact]
    public void LabelEncoder_InverseTransform_ReturnsOriginalValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 },
            { 20.0 }
        });

        var encoder = new LabelEncoder<double>();

        // Act
        encoder.Fit(data);
        var encoded = encoder.Transform(data);
        var decoded = encoder.InverseTransform(encoded);

        // Assert
        Assert.Equal(30.0, decoded[0, 0], Tolerance);
        Assert.Equal(10.0, decoded[1, 0], Tolerance);
        Assert.Equal(20.0, decoded[2, 0], Tolerance);
    }

    [Fact]
    public void LabelEncoder_UnknownValue_ReturnsMinusOne()
    {
        // Arrange
        var trainData = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 }
        });
        var testData = new Matrix<double>(new double[,]
        {
            { 3.0 } // Unknown
        });

        var encoder = new LabelEncoder<double>();

        // Act
        encoder.Fit(trainData);
        var result = encoder.Transform(testData);

        // Assert - Unknown gets -1
        Assert.Equal(-1.0, result[0, 0], Tolerance);
    }

    [Fact]
    public void LabelEncoder_MultipleColumns_EncodesAll()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 100.0 },
            { 2.0, 200.0 }
        });

        var encoder = new LabelEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - Both columns encoded independently
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(0.0, result[0, 0], Tolerance); // 1 -> 0
        Assert.Equal(1.0, result[1, 0], Tolerance); // 2 -> 1
        Assert.Equal(0.0, result[0, 1], Tolerance); // 100 -> 0
        Assert.Equal(1.0, result[1, 1], Tolerance); // 200 -> 1
    }

    [Fact]
    public void LabelEncoder_SpecificColumns_OnlyEncodesSelected()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0, 100.0 },
            { 10.0, 200.0 }
        });

        var encoder = new LabelEncoder<double>(columnIndices: new[] { 0 });

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance); // 30 -> 1 (encoded)
        Assert.Equal(0.0, result[1, 0], Tolerance); // 10 -> 0 (encoded)
        Assert.Equal(100.0, result[0, 1], Tolerance); // Unchanged
        Assert.Equal(200.0, result[1, 1], Tolerance); // Unchanged
    }

    [Fact]
    public void LabelEncoder_Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0 } });
        var encoder = new LabelEncoder<double>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => encoder.Transform(data));
    }

    [Fact]
    public void LabelEncoder_NClasses_ReturnsCorrectCounts()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 100.0 },
            { 2.0, 200.0 },
            { 3.0, 100.0 }
        });

        var encoder = new LabelEncoder<double>();

        // Act
        encoder.Fit(data);

        // Assert
        Assert.NotNull(encoder.NClasses);
        Assert.Equal(3, encoder.NClasses[0]); // 3 unique in column 0
        Assert.Equal(2, encoder.NClasses[1]); // 2 unique in column 1
    }

    [Fact]
    public void LabelEncoder_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var encoder = new LabelEncoder<double>();

        // Assert
        Assert.True(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void LabelEncoder_GetFeatureNamesOut_ReturnsInputNames()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);

        // Act
        var names = encoder.GetFeatureNamesOut(new[] { "a", "b" });

        // Assert
        Assert.Equal(new[] { "a", "b" }, names);
    }

    #endregion

    #region OrdinalEncoder Tests

    [Fact]
    public void OrdinalEncoder_BasicEncoding_Works()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 },
            { 20.0 }
        });

        var encoder = new OrdinalEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - Sorted order: 10=0, 20=1, 30=2
        Assert.Equal(2.0, result[0, 0], Tolerance); // 30 -> 2
        Assert.Equal(0.0, result[1, 0], Tolerance); // 10 -> 0
        Assert.Equal(1.0, result[2, 0], Tolerance); // 20 -> 1
    }

    [Fact]
    public void OrdinalEncoder_CustomCategories_UsesProvidedOrder()
    {
        // Arrange - Custom order: [20, 10, 30] -> indices 0, 1, 2
        var categories = new List<double[]> { new[] { 20.0, 10.0, 30.0 } };
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 },
            { 20.0 }
        });

        var encoder = new OrdinalEncoder<double>(categories: categories);

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert - Custom order: 20=0, 10=1, 30=2
        Assert.Equal(2.0, result[0, 0], Tolerance); // 30 -> 2
        Assert.Equal(1.0, result[1, 0], Tolerance); // 10 -> 1
        Assert.Equal(0.0, result[2, 0], Tolerance); // 20 -> 0
    }

    [Fact]
    public void OrdinalEncoder_UnknownValue_ErrorHandling()
    {
        // Arrange
        var trainData = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 }
        });
        var testData = new Matrix<double>(new double[,]
        {
            { 3.0 } // Unknown
        });

        var encoder = new OrdinalEncoder<double>(handleUnknown: UnknownValueHandling.Error);

        // Act
        encoder.Fit(trainData);

        // Assert
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    [Fact]
    public void OrdinalEncoder_UnknownValue_UseEncodedValue()
    {
        // Arrange
        var trainData = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 }
        });
        var testData = new Matrix<double>(new double[,]
        {
            { 3.0 } // Unknown
        });

        var encoder = new OrdinalEncoder<double>(
            handleUnknown: UnknownValueHandling.UseEncodedValue,
            unknownValue: -999);

        // Act
        encoder.Fit(trainData);
        var result = encoder.Transform(testData);

        // Assert
        Assert.Equal(-999.0, result[0, 0], Tolerance);
    }

    [Fact]
    public void OrdinalEncoder_InverseTransform_ReturnsOriginalValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 },
            { 20.0 }
        });

        var encoder = new OrdinalEncoder<double>();

        // Act
        encoder.Fit(data);
        var encoded = encoder.Transform(data);
        var decoded = encoder.InverseTransform(encoded);

        // Assert
        Assert.Equal(30.0, decoded[0, 0], Tolerance);
        Assert.Equal(10.0, decoded[1, 0], Tolerance);
        Assert.Equal(20.0, decoded[2, 0], Tolerance);
    }

    [Fact]
    public void OrdinalEncoder_MultipleColumns_EncodesAll()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 100.0 },
            { 2.0, 200.0 }
        });

        var encoder = new OrdinalEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void OrdinalEncoder_SpecificColumns_OnlyEncodesSelected()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0, 100.0 },
            { 10.0, 200.0 }
        });

        var encoder = new OrdinalEncoder<double>(columnIndices: new[] { 0 });

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance); // 30 -> 1 (encoded)
        Assert.Equal(0.0, result[1, 0], Tolerance); // 10 -> 0 (encoded)
        Assert.Equal(100.0, result[0, 1], Tolerance); // Unchanged
        Assert.Equal(200.0, result[1, 1], Tolerance); // Unchanged
    }

    [Fact]
    public void OrdinalEncoder_Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0 } });
        var encoder = new OrdinalEncoder<double>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => encoder.Transform(data));
    }

    [Fact]
    public void OrdinalEncoder_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var encoder = new OrdinalEncoder<double>();

        // Assert
        Assert.True(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void OrdinalEncoder_Properties_ReturnCorrectValues()
    {
        // Arrange
        var encoder = new OrdinalEncoder<double>(
            handleUnknown: UnknownValueHandling.UseEncodedValue,
            unknownValue: -42);

        // Assert
        Assert.Equal(UnknownValueHandling.UseEncodedValue, encoder.HandleUnknown);
        Assert.Equal(-42, encoder.UnknownValue);
    }

    [Fact]
    public void OrdinalEncoder_GetFeatureNamesOut_ReturnsInputNames()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var encoder = new OrdinalEncoder<double>();
        encoder.Fit(data);

        // Act
        var names = encoder.GetFeatureNamesOut(new[] { "x", "y" });

        // Assert
        Assert.Equal(new[] { "x", "y" }, names);
    }

    #endregion

    #region FitTransform Tests

    [Fact]
    public void OneHotEncoder_FitTransform_WorksCorrectly()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 }
        });

        var encoder = new OneHotEncoder<double>();

        // Act
        var result = encoder.FitTransform(data);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void LabelEncoder_FitTransform_WorksCorrectly()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 }
        });

        var encoder = new LabelEncoder<double>();

        // Act
        var result = encoder.FitTransform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance); // 30 -> 1
        Assert.Equal(0.0, result[1, 0], Tolerance); // 10 -> 0
    }

    [Fact]
    public void OrdinalEncoder_FitTransform_WorksCorrectly()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 30.0 },
            { 10.0 }
        });

        var encoder = new OrdinalEncoder<double>();

        // Act
        var result = encoder.FitTransform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance); // 30 -> 1
        Assert.Equal(0.0, result[1, 0], Tolerance); // 10 -> 0
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void OneHotEncoder_SingleCategory_Works()
    {
        // Arrange - Only one unique value
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 1.0 },
            { 1.0 }
        });

        var encoder = new OneHotEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[1, 0], Tolerance);
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    [Fact]
    public void LabelEncoder_SingleCategory_ReturnsZeros()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 5.0 },
            { 5.0 }
        });

        var encoder = new LabelEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[1, 0], Tolerance);
    }

    [Fact]
    public void OneHotEncoder_LargeNumberOfCategories_Works()
    {
        // Arrange - 10 unique values
        var data = new Matrix<double>(10, 1);
        for (int i = 0; i < 10; i++)
        {
            data[i, 0] = i + 1.0;
        }

        var encoder = new OneHotEncoder<double>();

        // Act
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Assert
        Assert.Equal(10, result.Rows);
        Assert.Equal(10, result.Columns);
        Assert.Equal(10, encoder.NOutputFeatures);
    }

    [Fact]
    public void OneHotEncoder_Categories_ReturnsLearnedCategories()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 3.0 },
            { 1.0 },
            { 2.0 }
        });

        var encoder = new OneHotEncoder<double>();

        // Act
        encoder.Fit(data);

        // Assert
        Assert.NotNull(encoder.Categories);
        Assert.Single(encoder.Categories);
        Assert.Equal(new[] { 1.0, 2.0, 3.0 }, encoder.Categories[0]); // Sorted
    }

    #endregion
}
