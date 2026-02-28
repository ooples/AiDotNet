using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep math-correctness integration tests for preprocessing encoders
/// (LabelEncoder, OneHotEncoder) and imputers (SimpleImputer).
/// Each test hand-computes expected values and verifies code matches.
/// </summary>
public class EncodersAndImputersDeepMathIntegrationTests
{
    private const double Tol = 1e-8;

    private static Matrix<double> MakeMatrix(double[,] data) => new(data);

    // ========================================================================
    // LabelEncoder - Category to Integer Mapping
    // ========================================================================

    [Fact]
    public void LabelEncoder_SortedMapping_ThreeCategories()
    {
        // Values: [30, 10, 20, 10, 30]
        // Sorted unique: [10, 20, 30] => mapping: 10->0, 20->1, 30->2
        // Result: [2, 0, 1, 0, 2]
        var data = MakeMatrix(new double[,] { { 30 }, { 10 }, { 20 }, { 10 }, { 30 } });

        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2.0, result[0, 0], Tol); // 30 -> 2
        Assert.Equal(0.0, result[1, 0], Tol); // 10 -> 0
        Assert.Equal(1.0, result[2, 0], Tol); // 20 -> 1
        Assert.Equal(0.0, result[3, 0], Tol); // 10 -> 0
        Assert.Equal(2.0, result[4, 0], Tol); // 30 -> 2
    }

    [Fact]
    public void LabelEncoder_NClasses_IsCorrect()
    {
        var data = MakeMatrix(new double[,] { { 30 }, { 10 }, { 20 }, { 10 }, { 30 } });
        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);

        Assert.NotNull(encoder.NClasses);
        Assert.Equal(3, encoder.NClasses[0]); // 3 unique values: 10, 20, 30
    }

    [Fact]
    public void LabelEncoder_InverseTransform_RecoverOriginal()
    {
        var data = MakeMatrix(new double[,] { { 30 }, { 10 }, { 20 } });
        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);

        var encoded = encoder.Transform(data);
        var decoded = encoder.InverseTransform(encoded);

        Assert.Equal(30.0, decoded[0, 0], Tol);
        Assert.Equal(10.0, decoded[1, 0], Tol);
        Assert.Equal(20.0, decoded[2, 0], Tol);
    }

    [Fact]
    public void LabelEncoder_SingleCategory_MapsToZero()
    {
        var data = MakeMatrix(new double[,] { { 5 }, { 5 }, { 5 } });
        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[1, 0], Tol);
        Assert.Equal(0.0, result[2, 0], Tol);
    }

    [Fact]
    public void LabelEncoder_UnknownValue_ReturnsMinusOne()
    {
        var trainData = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });
        var encoder = new LabelEncoder<double>();
        encoder.Fit(trainData);

        // Transform with value not seen during fit
        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(-1.0, result[0, 0], Tol);
    }

    [Fact]
    public void LabelEncoder_ConsecutiveIntegers_PreserveOrder()
    {
        // Values already sorted: [1, 2, 3] => 1->0, 2->1, 3->2
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(1.0, result[1, 0], Tol);
        Assert.Equal(2.0, result[2, 0], Tol);
    }

    [Fact]
    public void LabelEncoder_MultipleColumns_IndependentMapping()
    {
        // Col 0: [10, 20] => 10->0, 20->1
        // Col 1: [100, 200, 300] repeated => 100->0, 200->1, 300->2
        var data = MakeMatrix(new double[,] {
            { 10, 300 },
            { 20, 100 },
            { 10, 200 }
        });

        var encoder = new LabelEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Col 0
        Assert.Equal(0.0, result[0, 0], Tol); // 10 -> 0
        Assert.Equal(1.0, result[1, 0], Tol); // 20 -> 1
        Assert.Equal(0.0, result[2, 0], Tol); // 10 -> 0

        // Col 1
        Assert.Equal(2.0, result[0, 1], Tol); // 300 -> 2
        Assert.Equal(0.0, result[1, 1], Tol); // 100 -> 0
        Assert.Equal(1.0, result[2, 1], Tol); // 200 -> 1
    }

    [Fact]
    public void LabelEncoder_SpecificColumns_OnlyEncodesSelected()
    {
        // Only encode column 0, leave column 1 as-is
        var data = MakeMatrix(new double[,] {
            { 30, 999 },
            { 10, 888 }
        });

        var encoder = new LabelEncoder<double>(columnIndices: new[] { 0 });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol);   // 30 -> 1 (encoded)
        Assert.Equal(999.0, result[0, 1], Tol);  // 999 unchanged
        Assert.Equal(0.0, result[1, 0], Tol);    // 10 -> 0 (encoded)
        Assert.Equal(888.0, result[1, 1], Tol);  // 888 unchanged
    }

    // ========================================================================
    // OneHotEncoder - Binary Indicator Columns
    // ========================================================================

    [Fact]
    public void OneHot_ThreeCategories_CorrectOutput()
    {
        // Values: [10, 20, 30] => categories sorted: [10, 20, 30]
        // 10 -> [1, 0, 0]
        // 20 -> [0, 1, 0]
        // 30 -> [0, 0, 1]
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Columns);

        // Row 0: 10 -> [1, 0, 0]
        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);
        Assert.Equal(0.0, result[0, 2], Tol);

        // Row 1: 20 -> [0, 1, 0]
        Assert.Equal(0.0, result[1, 0], Tol);
        Assert.Equal(1.0, result[1, 1], Tol);
        Assert.Equal(0.0, result[1, 2], Tol);

        // Row 2: 30 -> [0, 0, 1]
        Assert.Equal(0.0, result[2, 0], Tol);
        Assert.Equal(0.0, result[2, 1], Tol);
        Assert.Equal(1.0, result[2, 2], Tol);
    }

    [Fact]
    public void OneHot_DropFirst_ReducesColumns()
    {
        // Values: [10, 20, 30] with dropFirst
        // 10 -> [0, 0] (first dropped)
        // 20 -> [1, 0]
        // 30 -> [0, 1]
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder = new OneHotEncoder<double>(dropFirst: true);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2, result.Columns); // 3 categories - 1 = 2

        // Row 0: 10 -> [0, 0]
        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);

        // Row 1: 20 -> [1, 0]
        Assert.Equal(1.0, result[1, 0], Tol);
        Assert.Equal(0.0, result[1, 1], Tol);

        // Row 2: 30 -> [0, 1]
        Assert.Equal(0.0, result[2, 0], Tol);
        Assert.Equal(1.0, result[2, 1], Tol);
    }

    [Fact]
    public void OneHot_NOutputFeatures_IsCorrect()
    {
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);
        Assert.Equal(3, encoder.NOutputFeatures);

        var encoderDrop = new OneHotEncoder<double>(dropFirst: true);
        encoderDrop.Fit(data);
        Assert.Equal(2, encoderDrop.NOutputFeatures);
    }

    [Fact]
    public void OneHot_Categories_AreSorted()
    {
        var data = MakeMatrix(new double[,] { { 30 }, { 10 }, { 20 } });
        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);

        Assert.NotNull(encoder.Categories);
        Assert.Equal(3, encoder.Categories[0].Length);
        Assert.Equal(10.0, encoder.Categories[0][0], Tol);
        Assert.Equal(20.0, encoder.Categories[0][1], Tol);
        Assert.Equal(30.0, encoder.Categories[0][2], Tol);
    }

    [Fact]
    public void OneHot_EachRowSumsToOne()
    {
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 }, { 10 } });
        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int row = 0; row < result.Rows; row++)
        {
            double rowSum = 0;
            for (int col = 0; col < result.Columns; col++)
            {
                rowSum += result[row, col];
            }
            Assert.Equal(1.0, rowSum, Tol);
        }
    }

    [Fact]
    public void OneHot_UnknownHandlingIgnore_AllZeros()
    {
        var trainData = MakeMatrix(new double[,] { { 10 }, { 20 } });
        var encoder = new OneHotEncoder<double>(handleUnknown: OneHotUnknownHandling.Ignore);
        encoder.Fit(trainData);

        // Transform with unknown value
        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        // Unknown should be all zeros
        for (int col = 0; col < result.Columns; col++)
        {
            Assert.Equal(0.0, result[0, col], Tol);
        }
    }

    [Fact]
    public void OneHot_InverseTransform_RecoverOriginal()
    {
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });
        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);

        var encoded = encoder.Transform(data);
        var decoded = encoder.InverseTransform(encoded);

        Assert.Equal(10.0, decoded[0, 0], Tol);
        Assert.Equal(20.0, decoded[1, 0], Tol);
        Assert.Equal(30.0, decoded[2, 0], Tol);
    }

    [Fact]
    public void OneHot_TwoColumns_CorrectExpansion()
    {
        // Col 0: 2 categories [1, 2] => 2 one-hot columns
        // Col 1: 3 categories [10, 20, 30] => 3 one-hot columns
        // Total output: 5 columns
        var data = MakeMatrix(new double[,] {
            { 1, 10 },
            { 2, 20 },
            { 1, 30 }
        });

        var encoder = new OneHotEncoder<double>();
        encoder.Fit(data);

        Assert.Equal(5, encoder.NOutputFeatures);

        var result = encoder.Transform(data);
        Assert.Equal(5, result.Columns);
    }

    // ========================================================================
    // SimpleImputer - Mean Strategy
    // ========================================================================

    [Fact]
    public void Imputer_Mean_HandComputed()
    {
        // Col 0: [1, NaN, 3, NaN, 5] => valid=[1,3,5], mean=3.0
        // After imputation: [1, 3, 3, 3, 5]
        var data = MakeMatrix(new double[,] {
            { 1 }, { double.NaN }, { 3 }, { double.NaN }, { 5 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        imputer.Fit(data);

        Assert.NotNull(imputer.Statistics);
        Assert.Equal(3.0, imputer.Statistics[0], Tol);

        var result = imputer.Transform(data);
        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(3.0, result[1, 0], Tol);
        Assert.Equal(3.0, result[2, 0], Tol);
        Assert.Equal(3.0, result[3, 0], Tol);
        Assert.Equal(5.0, result[4, 0], Tol);
    }

    [Fact]
    public void Imputer_Mean_MultipleColumns()
    {
        // Col 0: [1, NaN, 5] => mean=3.0
        // Col 1: [10, 20, NaN] => mean=15.0
        var data = MakeMatrix(new double[,] {
            { 1, 10 },
            { double.NaN, 20 },
            { 5, double.NaN }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        imputer.Fit(data);

        Assert.NotNull(imputer.Statistics);
        Assert.Equal(3.0, imputer.Statistics[0], Tol);
        Assert.Equal(15.0, imputer.Statistics[1], Tol);

        var result = imputer.Transform(data);
        Assert.Equal(3.0, result[1, 0], Tol);  // NaN replaced with mean
        Assert.Equal(15.0, result[2, 1], Tol); // NaN replaced with mean
    }

    [Fact]
    public void Imputer_Mean_NoMissingValues_Unchanged()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        imputer.Fit(data);
        var result = imputer.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(2.0, result[1, 0], Tol);
        Assert.Equal(3.0, result[2, 0], Tol);
    }

    // ========================================================================
    // SimpleImputer - Median Strategy
    // ========================================================================

    [Fact]
    public void Imputer_Median_OddCount()
    {
        // Col 0: [1, NaN, 3, NaN, 5] => valid=[1,3,5], median=3
        var data = MakeMatrix(new double[,] {
            { 1 }, { double.NaN }, { 3 }, { double.NaN }, { 5 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);
        imputer.Fit(data);

        Assert.NotNull(imputer.Statistics);
        Assert.Equal(3.0, imputer.Statistics[0], Tol);
    }

    [Fact]
    public void Imputer_Median_EvenCount()
    {
        // Col 0: [1, NaN, 4, NaN, 2, 3] => valid=[1,4,2,3], sorted=[1,2,3,4], median=(2+3)/2=2.5
        var data = MakeMatrix(new double[,] {
            { 1 }, { double.NaN }, { 4 }, { double.NaN }, { 2 }, { 3 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);
        imputer.Fit(data);

        Assert.NotNull(imputer.Statistics);
        Assert.Equal(2.5, imputer.Statistics[0], Tol);
    }

    // ========================================================================
    // SimpleImputer - Constant Strategy
    // ========================================================================

    [Fact]
    public void Imputer_Constant_ReplacesWithSpecifiedValue()
    {
        var data = MakeMatrix(new double[,] {
            { 1 }, { double.NaN }, { 3 }
        });

        var imputer = new SimpleImputer<double>(
            ImputationStrategy.Constant, fillValue: -999.0);
        imputer.Fit(data);
        var result = imputer.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(-999.0, result[1, 0], Tol);
        Assert.Equal(3.0, result[2, 0], Tol);
    }

    // ========================================================================
    // SimpleImputer - MostFrequent Strategy
    // ========================================================================

    [Fact]
    public void Imputer_MostFrequent_ReplacesWithMode()
    {
        // Col 0: [1, 2, 2, NaN, 3] => valid=[1,2,2,3], mode=2 (most frequent)
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 2 }, { double.NaN }, { 3 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.MostFrequent);
        imputer.Fit(data);

        Assert.NotNull(imputer.Statistics);
        Assert.Equal(2.0, imputer.Statistics[0], Tol);

        var result = imputer.Transform(data);
        Assert.Equal(2.0, result[3, 0], Tol); // NaN replaced with mode
    }

    // ========================================================================
    // SimpleImputer - Edge Cases
    // ========================================================================

    [Fact]
    public void Imputer_StrategyProperty_ReturnsCorrectly()
    {
        var mean = new SimpleImputer<double>(ImputationStrategy.Mean);
        var median = new SimpleImputer<double>(ImputationStrategy.Median);
        var constant = new SimpleImputer<double>(ImputationStrategy.Constant);
        var mostFreq = new SimpleImputer<double>(ImputationStrategy.MostFrequent);

        Assert.Equal(ImputationStrategy.Mean, mean.Strategy);
        Assert.Equal(ImputationStrategy.Median, median.Strategy);
        Assert.Equal(ImputationStrategy.Constant, constant.Strategy);
        Assert.Equal(ImputationStrategy.MostFrequent, mostFreq.Strategy);
    }

    [Fact]
    public void Imputer_Mean_ExcludesNaN_FromComputation()
    {
        // If mean were computed with NaN, it would be NaN
        // Mean of [2, NaN, 4] should be (2+4)/2 = 3, not NaN
        var data = MakeMatrix(new double[,] {
            { 2 }, { double.NaN }, { 4 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        imputer.Fit(data);

        Assert.NotNull(imputer.Statistics);
        Assert.False(double.IsNaN(imputer.Statistics[0]));
        Assert.Equal(3.0, imputer.Statistics[0], Tol);
    }

    // ========================================================================
    // Cross-Component: Encoder + Imputer Pipeline
    // ========================================================================

    [Fact]
    public void LabelEncoder_FitTransform_EquivalentToSeparate()
    {
        var data = MakeMatrix(new double[,] { { 30 }, { 10 }, { 20 } });

        var encoder1 = new LabelEncoder<double>();
        encoder1.Fit(data);
        var result1 = encoder1.Transform(data);

        var encoder2 = new LabelEncoder<double>();
        var result2 = encoder2.FitTransform(data);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(result1[i, 0], result2[i, 0], Tol);
        }
    }

    [Fact]
    public void OneHot_FitTransform_EquivalentToSeparate()
    {
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder1 = new OneHotEncoder<double>();
        encoder1.Fit(data);
        var result1 = encoder1.Transform(data);

        var encoder2 = new OneHotEncoder<double>();
        var result2 = encoder2.FitTransform(data);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j], Tol);
            }
        }
    }
}
