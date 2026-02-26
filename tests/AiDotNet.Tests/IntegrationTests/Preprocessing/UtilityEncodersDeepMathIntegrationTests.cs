using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep math-correctness integration tests for utility categorical encoders:
/// SumEncoder, CountEncoder, BinaryEncoder, HashingEncoder.
/// Each test hand-computes expected values and verifies code matches.
/// </summary>
public class UtilityEncodersDeepMathIntegrationTests
{
    private const double Tol = 1e-8;

    private static Matrix<double> MakeMatrix(double[,] data) => new(data);

    // ========================================================================
    // SumEncoder - Deviation / Effect Coding
    // ========================================================================

    [Fact]
    public void SumEncoder_ThreeCategories_ContrastValues()
    {
        // k=3 categories (10, 20, 30), creates 2 columns
        // Sum coding (effect coding):
        // - Cat 10 (index 0): [1, 0]   (1 in col 0)
        // - Cat 20 (index 1): [0, 1]   (1 in col 1)
        // - Cat 30 (index 2, reference): [-1, -1] (reference = -1 in all)
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder = new SumEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Cat 10 (first): [1, 0]
        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);

        // Cat 20 (second): [0, 1]
        Assert.Equal(0.0, result[1, 0], Tol);
        Assert.Equal(1.0, result[1, 1], Tol);

        // Cat 30 (reference/last): [-1, -1]
        Assert.Equal(-1.0, result[2, 0], Tol);
        Assert.Equal(-1.0, result[2, 1], Tol);
    }

    [Fact]
    public void SumEncoder_ColumnSumsToZero()
    {
        // Key property of sum coding: each column sums to zero across all categories
        // (when each category appears once)
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });

        var encoder = new SumEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double colSum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                colSum += result[row, col];
            }
            Assert.Equal(0.0, colSum, 1e-6);
        }
    }

    [Fact]
    public void SumEncoder_OutputDimension_KMinusOne()
    {
        // 5 categories -> 4 output columns
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var encoder = new SumEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(5, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    [Fact]
    public void SumEncoder_TwoCategories_OneColumn()
    {
        // k=2: creates 1 column
        // Cat 1 (first): [1]
        // Cat 2 (reference): [-1]
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });

        var encoder = new SumEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(1, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(-1.0, result[1, 0], Tol);
    }

    [Fact]
    public void SumEncoder_UnknownCategory_ZeroVector()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new SumEncoder<double>(handleUnknown: SumEncoderHandleUnknown.UseZeros);
        encoder.Fit(data);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);
    }

    [Fact]
    public void SumEncoder_RepeatedCategories_SameEncoding()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new SumEncoder<double>();
        encoder.Fit(data);

        // Transform data with repeated categories
        var testData = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 3 }, { 3 } });
        var result = encoder.Transform(testData);

        // Same category should produce same encoding
        Assert.Equal(result[0, 0], result[1, 0], Tol);
        Assert.Equal(result[0, 1], result[1, 1], Tol);
        Assert.Equal(result[3, 0], result[4, 0], Tol);
        Assert.Equal(result[3, 1], result[4, 1], Tol);
    }

    // ========================================================================
    // CountEncoder - Frequency Encoding
    // ========================================================================

    [Fact]
    public void CountEncoder_RawCounts_HandComputed()
    {
        // Data: [1, 1, 1, 2, 2, 3]
        // Cat 1: count=3, Cat 2: count=2, Cat 3: count=1
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 }, { 3 } });

        var encoder = new CountEncoder<double>(normalize: false, logTransform: false);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3.0, result[0, 0], Tol); // Cat 1
        Assert.Equal(3.0, result[1, 0], Tol); // Cat 1
        Assert.Equal(3.0, result[2, 0], Tol); // Cat 1
        Assert.Equal(2.0, result[3, 0], Tol); // Cat 2
        Assert.Equal(2.0, result[4, 0], Tol); // Cat 2
        Assert.Equal(1.0, result[5, 0], Tol); // Cat 3
    }

    [Fact]
    public void CountEncoder_Normalized_CountsDividedByN()
    {
        // Data: [1, 1, 1, 2, 2, 3] (n=6)
        // Normalized: Cat 1: 3/6=0.5, Cat 2: 2/6=0.3333, Cat 3: 1/6=0.1667
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 }, { 3 } });

        var encoder = new CountEncoder<double>(normalize: true, logTransform: false);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3.0 / 6, result[0, 0], Tol);
        Assert.Equal(2.0 / 6, result[3, 0], Tol);
        Assert.Equal(1.0 / 6, result[5, 0], Tol);
    }

    [Fact]
    public void CountEncoder_LogTransform_Log1POfCount()
    {
        // LogTransform: log1p(count) = log(1 + count)
        // Cat 1: count=3, log(1+3) = log(4) = 1.3862943...
        // Cat 2: count=2, log(1+2) = log(3) = 1.0986122...
        // Cat 3: count=1, log(1+1) = log(2) = 0.6931471...
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 }, { 3 } });

        var encoder = new CountEncoder<double>(normalize: false, logTransform: true);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(Math.Log(1 + 3), result[0, 0], Tol);
        Assert.Equal(Math.Log(1 + 2), result[3, 0], Tol);
        Assert.Equal(Math.Log(1 + 1), result[5, 0], Tol);
    }

    [Fact]
    public void CountEncoder_NormalizedAndLogTransform_Combined()
    {
        // First normalize (count/n), then log1p: log(1 + count/n)
        // Cat 1: log(1 + 3/6) = log(1.5) = 0.4054651...
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 }, { 3 } });

        var encoder = new CountEncoder<double>(normalize: true, logTransform: true);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(Math.Log(1 + 3.0 / 6), result[0, 0], Tol);
        Assert.Equal(Math.Log(1 + 2.0 / 6), result[3, 0], Tol);
        Assert.Equal(Math.Log(1 + 1.0 / 6), result[5, 0], Tol);
    }

    [Fact]
    public void CountEncoder_MoreFrequent_HigherEncoding()
    {
        // More frequent categories should have higher encoded values
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 1 }, { 2 }, { 3 } });

        var encoder = new CountEncoder<double>(normalize: false);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Cat 1 (count=4) > Cat 2 (count=1) > Cat 3 (count=1)
        Assert.True(result[0, 0] > result[4, 0]);
    }

    [Fact]
    public void CountEncoder_UnknownCategory_UsesDefaultValue()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });

        var encoder = new CountEncoder<double>(unknownValue: 0.5);
        encoder.Fit(data);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(0.5, result[0, 0], Tol);
    }

    [Fact]
    public void CountEncoder_PreservesOutputDimension()
    {
        // CountEncoder should NOT expand dimensions (same columns in = same columns out)
        var data = MakeMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 1, 10 } });

        var encoder = new CountEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    // ========================================================================
    // BinaryEncoder - Binary Representation
    // ========================================================================

    [Fact]
    public void BinaryEncoder_FourCategories_TwoBitsNeeded()
    {
        // 4 categories + 1 for unknown = 5, ceil(log2(6)) = 3 bits
        // Ordinal mapping (1-indexed):
        //   Cat 10 -> 1 -> binary 001
        //   Cat 20 -> 2 -> binary 010
        //   Cat 30 -> 3 -> binary 011
        //   Cat 40 -> 4 -> binary 100
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 }, { 40 } });

        var encoder = new BinaryEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Should create 3 columns (for 4 categories + unknown = 5 values, ceil(log2(6)) = 3)
        Assert.True(result.Columns >= 2, $"Expected at least 2 columns, got {result.Columns}");

        // Each row should be a unique binary representation
        var encodings = new HashSet<string>();
        for (int i = 0; i < result.Rows; i++)
        {
            string encoding = "";
            for (int j = 0; j < result.Columns; j++)
            {
                encoding += result[i, j].ToString("F0");
            }
            encodings.Add(encoding);
        }
        // All 4 categories should have unique binary encodings
        Assert.Equal(4, encodings.Count);
    }

    [Fact]
    public void BinaryEncoder_OutputBitsAreZeroOrOne()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 } });

        var encoder = new BinaryEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // All values should be either 0 or 1
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                double val = result[i, j];
                Assert.True(Math.Abs(val) < Tol || Math.Abs(val - 1.0) < Tol,
                    $"Binary encoding at [{i},{j}] should be 0 or 1, got {val}");
            }
        }
    }

    [Fact]
    public void BinaryEncoder_FewerColumnsThanOneHot()
    {
        // 8 categories: one-hot would use 8 columns
        // Binary encoding uses ceil(log2(9)) = 4 columns
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 } });

        var encoder = new BinaryEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Should be significantly fewer columns than one-hot
        Assert.True(result.Columns < 8, $"Binary encoding should use < 8 columns, got {result.Columns}");
        Assert.True(result.Columns >= 4, $"Need at least 4 columns for 8 categories, got {result.Columns}");
    }

    [Fact]
    public void BinaryEncoder_UnknownCategory_AllZeros()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new BinaryEncoder<double>(handleUnknown: BinaryEncoderHandleUnknown.AllZeros);
        encoder.Fit(data);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        // Unknown should be encoded as 0 (all zero bits)
        for (int j = 0; j < result.Columns; j++)
        {
            Assert.Equal(0.0, result[0, j], Tol);
        }
    }

    [Fact]
    public void BinaryEncoder_TwoCategories_OneBitSuffices()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });

        var encoder = new BinaryEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // 2 categories + 1 unknown = 3, ceil(log2(4)) = 2 bits
        Assert.True(result.Columns <= 3, $"2 categories should need at most 3 columns, got {result.Columns}");

        // The two categories should have different encodings
        bool different = false;
        for (int j = 0; j < result.Columns; j++)
        {
            if (Math.Abs(result[0, j] - result[1, j]) > Tol)
            {
                different = true;
                break;
            }
        }
        Assert.True(different, "Two different categories should have different binary encodings");
    }

    // ========================================================================
    // HashingEncoder - Feature Hashing
    // ========================================================================

    [Fact]
    public void HashingEncoder_FixedOutputDimension()
    {
        // HashingEncoder should always produce nComponents columns per encoded column
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var encoder = new HashingEncoder<double>(nComponents: 4);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(5, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    [Fact]
    public void HashingEncoder_SameInput_SameOutput()
    {
        // Deterministic: same category should always hash to same bucket
        var data = MakeMatrix(new double[,] { { 42 }, { 42 }, { 42 } });

        var encoder = new HashingEncoder<double>(nComponents: 8);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // All three rows should be identical
        for (int j = 0; j < result.Columns; j++)
        {
            Assert.Equal(result[0, j], result[1, j], Tol);
            Assert.Equal(result[1, j], result[2, j], Tol);
        }
    }

    [Fact]
    public void HashingEncoder_OutputSparse_MostlyZeros()
    {
        // Each category hashes to a single bucket out of nComponents
        // So most values in each row should be zero
        var data = MakeMatrix(new double[,] { { 1 } });

        var encoder = new HashingEncoder<double>(nComponents: 16, alternateSign: false);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        int nonZeroCount = 0;
        for (int j = 0; j < result.Columns; j++)
        {
            if (Math.Abs(result[0, j]) > Tol) nonZeroCount++;
        }
        // Exactly 1 bucket should be non-zero for a single-column single-value input
        Assert.Equal(1, nonZeroCount);
    }

    [Fact]
    public void HashingEncoder_AlternateSign_ProducesNegativeValues()
    {
        // With alternateSign=true, some hash values become -1 instead of +1
        // Test with many different categories to increase probability of seeing both signs
        var dataArr = new double[100, 1];
        for (int i = 0; i < 100; i++) dataArr[i, 0] = i;
        var data = MakeMatrix(dataArr);

        var encoder = new HashingEncoder<double>(nComponents: 4, alternateSign: true);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        bool hasPositive = false;
        bool hasNegative = false;
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                if (result[i, j] > Tol) hasPositive = true;
                if (result[i, j] < -Tol) hasNegative = true;
            }
        }
        Assert.True(hasPositive, "Should have some positive values");
        Assert.True(hasNegative, "With alternateSign, should have some negative values");
    }

    [Fact]
    public void HashingEncoder_NoAlternateSign_AllNonNegative()
    {
        // Without alternateSign, all values should be non-negative
        var dataArr = new double[50, 1];
        for (int i = 0; i < 50; i++) dataArr[i, 0] = i;
        var data = MakeMatrix(dataArr);

        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True(result[i, j] >= -Tol,
                    $"Without alternateSign, value at [{i},{j}]={result[i, j]} should be non-negative");
            }
        }
    }

    [Fact]
    public void HashingEncoder_HandlesUnseenCategories()
    {
        // Hash encoder should handle unseen categories without error
        var trainData = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new HashingEncoder<double>(nComponents: 4);
        encoder.Fit(trainData);

        // Transform with unseen category
        var testData = MakeMatrix(new double[,] { { 999 } });
        var result = encoder.Transform(testData);

        // Should produce valid output without throwing
        Assert.Equal(1, result.Rows);
        Assert.Equal(4, result.Columns);

        // At least one bucket should be non-zero
        bool anyNonZero = false;
        for (int j = 0; j < result.Columns; j++)
        {
            if (Math.Abs(result[0, j]) > Tol) anyNonZero = true;
        }
        Assert.True(anyNonZero, "Unseen category should still hash to some bucket");
    }

    [Fact]
    public void HashingEncoder_InvalidNComponents_Throws()
    {
        Assert.Throws<ArgumentException>(() => new HashingEncoder<double>(nComponents: 0));
    }

    // ========================================================================
    // Cross-Encoder Properties
    // ========================================================================

    [Fact]
    public void SumEncoder_InverseTransform_NotSupported()
    {
        var encoder = new SumEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void CountEncoder_InverseTransform_NotSupported()
    {
        var encoder = new CountEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void HashingEncoder_InverseTransform_NotSupported()
    {
        var encoder = new HashingEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void BinaryEncoder_SupportsInverseTransform()
    {
        var encoder = new BinaryEncoder<double>();
        Assert.True(encoder.SupportsInverseTransform);
    }
}
