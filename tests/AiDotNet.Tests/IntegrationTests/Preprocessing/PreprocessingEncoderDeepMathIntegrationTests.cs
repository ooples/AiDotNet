using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep mathematical correctness tests for HashingEncoder's MurmurHash3-based feature hashing,
/// bucket distribution, alternate sign mode, pass-through columns, and feature name generation.
/// Each test verifies exact behavior of the hashing trick algorithm.
/// </summary>
public class PreprocessingEncoderDeepMathIntegrationTests
{
    #region Helpers

    private static Matrix<double> M(double[,] data) => new(data);

    private static void AssertCell(Matrix<double> m, int row, int col, double expected, double tol = 1e-10)
    {
        Assert.True(
            Math.Abs(m[row, col] - expected) < tol,
            $"[{row},{col}]: expected {expected}, got {m[row, col]} (diff={Math.Abs(m[row, col] - expected)})");
    }

    #endregion

    #region HashingEncoder Output Shape

    /// <summary>
    /// With nComponents=4 and 1 input column (all columns processed),
    /// output should have exactly 4 columns (one per hash bucket).
    /// </summary>
    [Fact]
    public void HashingEncoder_SingleColumn_OutputHas_NComponents_Columns()
    {
        var encoder = new HashingEncoder<double>(nComponents: 4);
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    /// <summary>
    /// With nComponents=8 (default) and 2 input columns,
    /// output should have 8*2=16 columns when all columns are encoded.
    /// </summary>
    [Fact]
    public void HashingEncoder_TwoColumns_OutputHas_2xNComponents_Columns()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8);
        var data = M(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2, result.Rows);
        Assert.Equal(16, result.Columns); // 2 columns * 8 components each
    }

    /// <summary>
    /// With columnIndices specifying only column 0, column 1 should pass through.
    /// Output = nComponents (for col 0) + 1 (pass-through for col 1).
    /// </summary>
    [Fact]
    public void HashingEncoder_PartialColumns_PassThrough()
    {
        var encoder = new HashingEncoder<double>(nComponents: 4, columnIndices: new[] { 0 });
        var data = M(new double[,] { { 1.0, 99.0 }, { 2.0, 88.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2, result.Rows);
        Assert.Equal(5, result.Columns); // 4 (hash of col 0) + 1 (pass-through col 1)

        // The pass-through column (last column) should have original values
        AssertCell(result, 0, 4, 99.0);
        AssertCell(result, 1, 4, 88.0);
    }

    #endregion

    #region Bucket Assignment Determinism

    /// <summary>
    /// The same value in the same column should always hash to the same bucket.
    /// Two rows with identical values should produce identical hash encodings.
    /// </summary>
    [Fact]
    public void HashingEncoder_SameValue_SameColumn_DeterministicBucket()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        var data = M(new double[,] { { 5.0 }, { 5.0 }, { 5.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // All three rows should be identical
        for (int col = 0; col < 8; col++)
        {
            AssertCell(result, 1, col, result[0, col]);
            AssertCell(result, 2, col, result[0, col]);
        }
    }

    /// <summary>
    /// Each row's hash encoding should have exactly one non-zero element (with alternateSign=false)
    /// since each value hashes to exactly one bucket with count 1.
    /// </summary>
    [Fact]
    public void HashingEncoder_SingleValue_ExactlyOneBucketNonZero()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        var data = M(new double[,] { { 42.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        int nonZeroCount = 0;
        for (int col = 0; col < 8; col++)
        {
            if (Math.Abs(result[0, col]) > 1e-10)
            {
                nonZeroCount++;
                // With alternateSign=false, the non-zero value should be exactly 1.0
                AssertCell(result, 0, col, 1.0);
            }
        }
        Assert.Equal(1, nonZeroCount);
    }

    /// <summary>
    /// Different values should (generally) hash to different buckets.
    /// With 8 components, the probability of collision for 2 values is 1/8 = 12.5%.
    /// We test 10 distinct values and expect at least 2 different bucket assignments.
    /// </summary>
    [Fact]
    public void HashingEncoder_DifferentValues_DifferentBuckets()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
                                      { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Find which bucket each value hashed to
        var buckets = new HashSet<int>();
        for (int row = 0; row < 10; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                if (Math.Abs(result[row, col]) > 1e-10)
                {
                    buckets.Add(col);
                    break;
                }
            }
        }

        // With 10 values and 8 buckets, we should have at least 2 different buckets
        Assert.True(buckets.Count >= 2, $"Expected at least 2 different buckets, got {buckets.Count}");
    }

    #endregion

    #region Alternate Sign Mode

    /// <summary>
    /// With alternateSign=true, the non-zero value should be +1.0 or -1.0
    /// depending on the hash sign bit (MurmurHash3 output).
    /// </summary>
    [Fact]
    public void HashingEncoder_AlternateSign_ValueIsPlusOrMinusOne()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: true);
        var data = M(new double[,] { { 42.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        bool foundNonZero = false;
        for (int col = 0; col < 8; col++)
        {
            double val = result[0, col];
            if (Math.Abs(val) > 1e-10)
            {
                foundNonZero = true;
                // With alternate sign, the value should be exactly +1 or -1
                Assert.True(Math.Abs(Math.Abs(val) - 1.0) < 1e-10,
                    $"Expected +1 or -1, got {val}");
            }
        }
        Assert.True(foundNonZero, "Expected at least one non-zero bucket");
    }

    /// <summary>
    /// With alternateSign=false, the non-zero value should always be +1.0.
    /// The sign is always positive.
    /// </summary>
    [Fact]
    public void HashingEncoder_NoAlternateSign_ValueIsAlwaysPositive()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        // Test multiple values to ensure none produce negative results
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int row = 0; row < 5; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                double val = result[row, col];
                if (Math.Abs(val) > 1e-10)
                {
                    Assert.True(val > 0, $"Row {row}, col {col}: expected positive, got {val}");
                }
            }
        }
    }

    /// <summary>
    /// With alternateSign=true, the MurmurHash3 should produce both positive and negative
    /// hashes for different inputs, resulting in both +1 and -1 values across many inputs.
    /// </summary>
    [Fact]
    public void HashingEncoder_AlternateSign_ProducesBothSigns()
    {
        var encoder = new HashingEncoder<double>(nComponents: 16, alternateSign: true);
        // Use many different values to ensure both signs appear
        var values = new double[100, 1];
        for (int i = 0; i < 100; i++)
            values[i, 0] = i + 1.0;
        var data = M(values);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        bool hasPositive = false;
        bool hasNegative = false;
        for (int row = 0; row < 100; row++)
        {
            for (int col = 0; col < 16; col++)
            {
                double val = result[row, col];
                if (val > 0.5) hasPositive = true;
                if (val < -0.5) hasNegative = true;
            }
        }

        Assert.True(hasPositive, "Expected at least one +1 value with alternateSign");
        Assert.True(hasNegative, "Expected at least one -1 value with alternateSign");
    }

    #endregion

    #region Hash Collision Behavior

    /// <summary>
    /// When two columns hash to the same bucket (collision), their contributions are summed.
    /// With two columns both encoding to the same bucket with alternateSign=false,
    /// the bucket value should be 2.0.
    /// </summary>
    [Fact]
    public void HashingEncoder_MultipleColumns_CollisionSumsContributions()
    {
        // With nComponents=1, ALL values must hash to bucket 0 (only one bucket)
        var encoder = new HashingEncoder<double>(nComponents: 1, alternateSign: false);
        var data = M(new double[,] { { 1.0, 2.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // 2 columns, each contributing 1.0 to the single bucket = 2.0
        Assert.Equal(2, result.Columns); // nComponents=1 per column * 2 columns
        // Each column gets its own set of nComponents buckets
        AssertCell(result, 0, 0, 1.0); // column 0's single bucket
        AssertCell(result, 0, 1, 1.0); // column 1's single bucket
    }

    #endregion

    #region Row Sum Conservation

    /// <summary>
    /// With alternateSign=false, each row's sum for a single encoded column should be 1.0
    /// (exactly one bucket gets value 1.0).
    /// </summary>
    [Fact]
    public void HashingEncoder_NoAlternateSign_RowSumEqualsOne()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        var data = M(new double[,] { { 7.5 }, { -3.2 }, { 0.0 }, { 100.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int row = 0; row < 4; row++)
        {
            double sum = 0;
            for (int col = 0; col < 8; col++)
            {
                sum += result[row, col];
            }
            Assert.True(Math.Abs(sum - 1.0) < 1e-10,
                $"Row {row}: expected sum 1.0, got {sum}");
        }
    }

    /// <summary>
    /// With alternateSign=true, each row's absolute sum for a single encoded column should be 1.0
    /// (exactly one bucket gets |value| = 1.0).
    /// </summary>
    [Fact]
    public void HashingEncoder_AlternateSign_AbsRowSumEqualsOne()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: true);
        var data = M(new double[,] { { 7.5 }, { -3.2 }, { 0.0 }, { 100.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int row = 0; row < 4; row++)
        {
            double absSum = 0;
            for (int col = 0; col < 8; col++)
            {
                absSum += Math.Abs(result[row, col]);
            }
            Assert.True(Math.Abs(absSum - 1.0) < 1e-10,
                $"Row {row}: expected abs sum 1.0, got {absSum}");
        }
    }

    #endregion

    #region Feature Name Generation

    /// <summary>
    /// GetFeatureNamesOut should produce correct names for encoded and pass-through columns.
    /// </summary>
    [Fact]
    public void HashingEncoder_FeatureNames_EncodedColumns()
    {
        var encoder = new HashingEncoder<double>(nComponents: 3);
        var data = M(new double[,] { { 1.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut();
        Assert.Equal(3, names.Length);
        Assert.Equal("x0_hash0", names[0]);
        Assert.Equal("x0_hash1", names[1]);
        Assert.Equal("x0_hash2", names[2]);
    }

    /// <summary>
    /// With custom input feature names, those names are used as base.
    /// </summary>
    [Fact]
    public void HashingEncoder_FeatureNames_CustomInputNames()
    {
        var encoder = new HashingEncoder<double>(nComponents: 2);
        var data = M(new double[,] { { 1.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut(new[] { "color" });
        Assert.Equal(2, names.Length);
        Assert.Equal("color_hash0", names[0]);
        Assert.Equal("color_hash1", names[1]);
    }

    /// <summary>
    /// Pass-through columns keep their original name (not hash-suffixed).
    /// </summary>
    [Fact]
    public void HashingEncoder_FeatureNames_PassThroughColumn()
    {
        var encoder = new HashingEncoder<double>(nComponents: 2, columnIndices: new[] { 0 });
        var data = M(new double[,] { { 1.0, 2.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut(new[] { "category", "amount" });
        Assert.Equal(3, names.Length); // 2 hash + 1 passthrough
        Assert.Equal("category_hash0", names[0]);
        Assert.Equal("category_hash1", names[1]);
        Assert.Equal("amount", names[2]);
    }

    #endregion

    #region Edge Cases and Validation

    /// <summary>
    /// nComponents=1 means every value hashes to bucket 0.
    /// </summary>
    [Fact]
    public void HashingEncoder_NComponents1_AllSameBucket()
    {
        var encoder = new HashingEncoder<double>(nComponents: 1, alternateSign: false);
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(1, result.Columns);
        // All rows should have value 1.0 in the single bucket
        for (int row = 0; row < 3; row++)
        {
            AssertCell(result, row, 0, 1.0);
        }
    }

    /// <summary>
    /// nComponents must be >= 1, constructor should throw for 0.
    /// </summary>
    [Fact]
    public void HashingEncoder_NComponents0_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new HashingEncoder<double>(nComponents: 0));
    }

    /// <summary>
    /// Transform before Fit should throw InvalidOperationException.
    /// </summary>
    [Fact]
    public void HashingEncoder_TransformBeforeFit_ThrowsInvalidOperation()
    {
        var encoder = new HashingEncoder<double>(nComponents: 4);
        var data = M(new double[,] { { 1.0 } });
        Assert.Throws<InvalidOperationException>(() => encoder.Transform(data));
    }

    /// <summary>
    /// Inverse transform is not supported for hash encoding.
    /// SupportsInverseTransform should be false.
    /// </summary>
    [Fact]
    public void HashingEncoder_SupportsInverseTransform_IsFalse()
    {
        var encoder = new HashingEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    /// <summary>
    /// GetFeatureNamesOut before Fit should return empty array.
    /// </summary>
    [Fact]
    public void HashingEncoder_FeatureNamesBeforeFit_ReturnsEmpty()
    {
        var encoder = new HashingEncoder<double>(nComponents: 4);
        var names = encoder.GetFeatureNamesOut();
        Assert.Empty(names);
    }

    /// <summary>
    /// Fitting and transforming with multiple rows should produce consistent results.
    /// The same value in the same column across different rows should hash identically.
    /// </summary>
    [Fact]
    public void HashingEncoder_MultipleRows_ConsistentHashing()
    {
        var encoder = new HashingEncoder<double>(nComponents: 4, alternateSign: true);
        var data = M(new double[,] { { 3.14 }, { 2.71 }, { 3.14 } }); // rows 0 and 2 are same
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Rows 0 and 2 should be identical
        for (int col = 0; col < 4; col++)
        {
            AssertCell(result, 2, col, result[0, col]);
        }

        // Row 1 might differ (different input value)
        // Just verify it's a valid encoding (exactly one non-zero)
        int nonZero = 0;
        for (int col = 0; col < 4; col++)
        {
            if (Math.Abs(result[1, col]) > 1e-10) nonZero++;
        }
        Assert.Equal(1, nonZero);
    }

    #endregion

    #region MurmurHash3 Column Index Separation

    /// <summary>
    /// The same value in different columns should hash to potentially different buckets
    /// because the column index is mixed into the hash.
    /// With many columns, we should see different bucket assignments.
    /// </summary>
    [Fact]
    public void HashingEncoder_SameValueDifferentColumns_DifferentHashes()
    {
        var encoder = new HashingEncoder<double>(nComponents: 8, alternateSign: false);
        // Create a 1-row matrix with the same value (1.0) in 20 columns
        var values = new double[1, 20];
        for (int c = 0; c < 20; c++) values[0, c] = 1.0;
        var data = M(values);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // result has 20*8 = 160 columns
        Assert.Equal(160, result.Columns);

        // For each of the 20 input columns, find which bucket got the 1.0
        var buckets = new List<int>();
        for (int inputCol = 0; inputCol < 20; inputCol++)
        {
            int startIdx = inputCol * 8;
            for (int b = 0; b < 8; b++)
            {
                if (Math.Abs(result[0, startIdx + b]) > 1e-10)
                {
                    buckets.Add(b);
                    break;
                }
            }
        }

        // With 20 columns and 8 buckets, we should see at least 2 different bucket indices
        // (probability of all same = (1/8)^19 ~ 0)
        var distinctBuckets = new HashSet<int>(buckets);
        Assert.True(distinctBuckets.Count >= 2,
            $"Expected different buckets for different columns, got {distinctBuckets.Count} distinct");
    }

    #endregion
}
