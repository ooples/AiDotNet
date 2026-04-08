using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep mathematical correctness tests for contrast coding encoders: HelmertEncoder,
/// BackwardDifferenceEncoder, and SumEncoder. Each test verifies exact hand-computed
/// contrast matrix entries against the statistical theory of contrast coding schemes.
/// </summary>
public class ContrastEncoderDeepMathIntegrationTests
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

    #region Helmert Encoder - Standard

    /// <summary>
    /// Standard Helmert with k=3 categories {1,2,3}.
    /// Contrast matrix (k x k-1):
    /// Col 0: level 0 vs mean(1,2) → [2/3, -1/3, -1/3]
    /// Col 1: level 1 vs mean(2)   → [0,    1/2,  -1/2]
    ///
    /// For category 1.0 (index 0): [2/3, 0]
    /// For category 2.0 (index 1): [-1/3, 1/2]
    /// For category 3.0 (index 2): [-1/3, -1/2]
    /// </summary>
    [Fact]
    public void HelmertEncoder_3Categories_CorrectContrastMatrix()
    {
        var encoder = new HelmertEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns); // k-1 = 2

        // Category 1.0 (index 0): [2/3, 0]
        AssertCell(result, 0, 0, 2.0 / 3.0);
        AssertCell(result, 0, 1, 0.0);

        // Category 2.0 (index 1): [-1/3, 1/2]
        AssertCell(result, 1, 0, -1.0 / 3.0);
        AssertCell(result, 1, 1, 1.0 / 2.0);

        // Category 3.0 (index 2): [-1/3, -1/2]
        AssertCell(result, 2, 0, -1.0 / 3.0);
        AssertCell(result, 2, 1, -1.0 / 2.0);
    }

    /// <summary>
    /// Standard Helmert with k=4 categories {1,2,3,4}.
    /// Col 0: level 0 vs mean(1,2,3) → [3/4, -1/4, -1/4, -1/4]
    /// Col 1: level 1 vs mean(2,3)   → [0,    2/3,  -1/3, -1/3]
    /// Col 2: level 2 vs mean(3)     → [0,    0,     1/2,  -1/2]
    /// </summary>
    [Fact]
    public void HelmertEncoder_4Categories_CorrectContrastMatrix()
    {
        var encoder = new HelmertEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(4, result.Rows);
        Assert.Equal(3, result.Columns);

        // Category 1.0 (index 0): [3/4, 0, 0]
        AssertCell(result, 0, 0, 3.0 / 4.0);
        AssertCell(result, 0, 1, 0.0);
        AssertCell(result, 0, 2, 0.0);

        // Category 2.0 (index 1): [-1/4, 2/3, 0]
        AssertCell(result, 1, 0, -1.0 / 4.0);
        AssertCell(result, 1, 1, 2.0 / 3.0);
        AssertCell(result, 1, 2, 0.0);

        // Category 3.0 (index 2): [-1/4, -1/3, 1/2]
        AssertCell(result, 2, 0, -1.0 / 4.0);
        AssertCell(result, 2, 1, -1.0 / 3.0);
        AssertCell(result, 2, 2, 1.0 / 2.0);

        // Category 4.0 (index 3): [-1/4, -1/3, -1/2]
        AssertCell(result, 3, 0, -1.0 / 4.0);
        AssertCell(result, 3, 1, -1.0 / 3.0);
        AssertCell(result, 3, 2, -1.0 / 2.0);
    }

    /// <summary>
    /// Standard Helmert column sum should be zero (orthogonality to intercept).
    /// For each contrast column, sum of all category weights = 0.
    /// Col 0: 2/3 + (-1/3) + (-1/3) = 0
    /// Col 1: 0 + 1/2 + (-1/2) = 0
    /// </summary>
    [Fact]
    public void HelmertEncoder_3Categories_ColumnSumsAreZero()
    {
        var encoder = new HelmertEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double sum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                sum += result[row, col];
            }
            Assert.True(Math.Abs(sum) < 1e-10, $"Column {col} sum: expected 0, got {sum}");
        }
    }

    /// <summary>
    /// Helmert with k=2 categories: simplest case.
    /// Col 0: level 0 vs level 1 → [1/2, -1/2]
    /// </summary>
    [Fact]
    public void HelmertEncoder_2Categories_SimpleContrast()
    {
        var encoder = new HelmertEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2, result.Rows);
        Assert.Equal(1, result.Columns);

        AssertCell(result, 0, 0, 1.0 / 2.0);
        AssertCell(result, 1, 0, -1.0 / 2.0);
    }

    #endregion

    #region Helmert Encoder - Reversed

    /// <summary>
    /// Reversed Helmert with k=3: compares each level to mean of PREVIOUS levels.
    /// Col 0: level 1 vs level 0      → [-1/2, 1/2, 0]
    /// Col 1: level 2 vs mean(0,1)    → [-1/3, -1/3, 2/3]
    /// </summary>
    [Fact]
    public void HelmertEncoder_Reversed_3Categories_CorrectContrastMatrix()
    {
        var encoder = new HelmertEncoder<double>(reversed: true);
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Category 1.0 (index 0): [-1/2, -1/3]
        AssertCell(result, 0, 0, -1.0 / 2.0);
        AssertCell(result, 0, 1, -1.0 / 3.0);

        // Category 2.0 (index 1): [1/2, -1/3]
        AssertCell(result, 1, 0, 1.0 / 2.0);
        AssertCell(result, 1, 1, -1.0 / 3.0);

        // Category 3.0 (index 2): [0, 2/3]
        AssertCell(result, 2, 0, 0.0);
        AssertCell(result, 2, 1, 2.0 / 3.0);
    }

    /// <summary>
    /// Reversed Helmert column sums should also be zero.
    /// </summary>
    [Fact]
    public void HelmertEncoder_Reversed_ColumnSumsAreZero()
    {
        var encoder = new HelmertEncoder<double>(reversed: true);
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double sum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                sum += result[row, col];
            }
            Assert.True(Math.Abs(sum) < 1e-10, $"Column {col} sum: expected 0, got {sum}");
        }
    }

    /// <summary>
    /// Unknown categories with HelmertHandleUnknown.UseZeros should give all zeros.
    /// </summary>
    [Fact]
    public void HelmertEncoder_UnknownCategory_UseZeros_AllZeros()
    {
        var encoder = new HelmertEncoder<double>(handleUnknown: HelmertHandleUnknown.UseZeros);
        var fitData = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(fitData);

        // Transform with unknown category 99.0
        var testData = M(new double[,] { { 99.0 } });
        var result = encoder.Transform(testData);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 0, 1, 0.0);
    }

    /// <summary>
    /// Unknown categories with HelmertHandleUnknown.Error should throw.
    /// </summary>
    [Fact]
    public void HelmertEncoder_UnknownCategory_Error_Throws()
    {
        var encoder = new HelmertEncoder<double>(handleUnknown: HelmertHandleUnknown.Error);
        var fitData = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(fitData);

        var testData = M(new double[,] { { 99.0 } });
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    #endregion

    #region Backward Difference Encoder

    /// <summary>
    /// Backward difference with k=3 categories {1,2,3}.
    /// Contrast matrix:
    /// Col 0 (level 1 vs level 0): rows <= 0 get -(3-0-1)/3 = -2/3, rows > 0 get (0+1)/3 = 1/3
    /// Col 1 (level 2 vs level 1): rows <= 1 get -(3-1-1)/3 = -1/3, rows > 1 get (1+1)/3 = 2/3
    ///
    /// Index 0: [-2/3, -1/3]
    /// Index 1: [1/3,  -1/3]
    /// Index 2: [1/3,   2/3]
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_3Categories_CorrectContrastMatrix()
    {
        var encoder = new BackwardDifferenceEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Index 0: [-2/3, -1/3]
        AssertCell(result, 0, 0, -2.0 / 3.0);
        AssertCell(result, 0, 1, -1.0 / 3.0);

        // Index 1: [1/3, -1/3]
        AssertCell(result, 1, 0, 1.0 / 3.0);
        AssertCell(result, 1, 1, -1.0 / 3.0);

        // Index 2: [1/3, 2/3]
        AssertCell(result, 2, 0, 1.0 / 3.0);
        AssertCell(result, 2, 1, 2.0 / 3.0);
    }

    /// <summary>
    /// Backward difference with k=4 categories.
    /// Col 0: rows <= 0 get -(4-0-1)/4 = -3/4, rows > 0 get 1/4
    /// Col 1: rows <= 1 get -(4-1-1)/4 = -2/4 = -1/2, rows > 1 get 2/4 = 1/2
    /// Col 2: rows <= 2 get -(4-2-1)/4 = -1/4, rows > 2 get 3/4
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_4Categories_CorrectContrastMatrix()
    {
        var encoder = new BackwardDifferenceEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(4, result.Rows);
        Assert.Equal(3, result.Columns);

        // Index 0: [-3/4, -1/2, -1/4]
        AssertCell(result, 0, 0, -3.0 / 4.0);
        AssertCell(result, 0, 1, -1.0 / 2.0);
        AssertCell(result, 0, 2, -1.0 / 4.0);

        // Index 1: [1/4, -1/2, -1/4]
        AssertCell(result, 1, 0, 1.0 / 4.0);
        AssertCell(result, 1, 1, -1.0 / 2.0);
        AssertCell(result, 1, 2, -1.0 / 4.0);

        // Index 2: [1/4, 1/2, -1/4]
        AssertCell(result, 2, 0, 1.0 / 4.0);
        AssertCell(result, 2, 1, 1.0 / 2.0);
        AssertCell(result, 2, 2, -1.0 / 4.0);

        // Index 3: [1/4, 1/2, 3/4]
        AssertCell(result, 3, 0, 1.0 / 4.0);
        AssertCell(result, 3, 1, 1.0 / 2.0);
        AssertCell(result, 3, 2, 3.0 / 4.0);
    }

    /// <summary>
    /// Backward difference column sums should equal zero (orthogonal to intercept).
    /// Col 0: -2/3 + 1/3 + 1/3 = 0
    /// Col 1: -1/3 + (-1/3) + 2/3 = 0
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_3Categories_ColumnSumsAreZero()
    {
        var encoder = new BackwardDifferenceEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double sum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                sum += result[row, col];
            }
            Assert.True(Math.Abs(sum) < 1e-10, $"Column {col} sum: expected 0, got {sum}");
        }
    }

    /// <summary>
    /// The key property of backward difference coding: the difference between consecutive
    /// rows in any column equals 1/k.
    /// For k=3: difference between row i+1 and row i in column c:
    /// Row 0 to 1 in col 0: 1/3 - (-2/3) = 1 ... actually the difference is 1.0 at the step
    /// and 0 elsewhere. Let me recalculate.
    /// Actually the step at col c happens between row c and row c+1:
    /// Col 0: row0=-2/3, row1=1/3, diff=1.0; row1=1/3, row2=1/3, diff=0
    /// Col 1: row0=-1/3, row1=-1/3, diff=0; row1=-1/3, row2=2/3, diff=1.0
    /// So the difference is exactly 1.0 at the step boundary and 0 elsewhere.
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_StepProperty_DiffEquals1AtBoundary()
    {
        var encoder = new BackwardDifferenceEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Col 0: step between row 0 and row 1 should be 1.0
        double diff01_c0 = result[1, 0] - result[0, 0]; // 1/3 - (-2/3) = 1.0
        Assert.True(Math.Abs(diff01_c0 - 1.0) < 1e-10, $"Step at col 0: expected 1.0, got {diff01_c0}");

        // Col 0: no step between row 1 and row 2
        double diff12_c0 = result[2, 0] - result[1, 0]; // 1/3 - 1/3 = 0
        Assert.True(Math.Abs(diff12_c0) < 1e-10, $"No step at col 0: expected 0, got {diff12_c0}");

        // Col 1: no step between row 0 and row 1
        double diff01_c1 = result[1, 1] - result[0, 1]; // -1/3 - (-1/3) = 0
        Assert.True(Math.Abs(diff01_c1) < 1e-10, $"No step at col 1: expected 0, got {diff01_c1}");

        // Col 1: step between row 1 and row 2 should be 1.0
        double diff12_c1 = result[2, 1] - result[1, 1]; // 2/3 - (-1/3) = 1.0
        Assert.True(Math.Abs(diff12_c1 - 1.0) < 1e-10, $"Step at col 1: expected 1.0, got {diff12_c1}");
    }

    /// <summary>
    /// Backward difference with k=2: simplest case.
    /// Col 0: row 0 gets -(2-0-1)/2 = -1/2, row 1 gets (0+1)/2 = 1/2
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_2Categories_SimpleContrast()
    {
        var encoder = new BackwardDifferenceEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2, result.Rows);
        Assert.Equal(1, result.Columns);

        AssertCell(result, 0, 0, -1.0 / 2.0);
        AssertCell(result, 1, 0, 1.0 / 2.0);
    }

    /// <summary>
    /// Unknown category with BackwardDifferenceHandleUnknown.UseZeros gives all zeros.
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_UnknownCategory_UseZeros_AllZeros()
    {
        var encoder = new BackwardDifferenceEncoder<double>(handleUnknown: BackwardDifferenceHandleUnknown.UseZeros);
        var fitData = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(fitData);

        var testData = M(new double[,] { { 99.0 } });
        var result = encoder.Transform(testData);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 0, 1, 0.0);
    }

    #endregion

    #region Sum Encoder

    /// <summary>
    /// Sum encoding with k=3 categories {1,2,3}. Categories sorted: [1,2,3].
    /// Last category (3.0, index 2) is reference → [-1, -1] in all columns.
    /// Category 1.0 (index 0): [1, 0]
    /// Category 2.0 (index 1): [0, 1]
    /// Category 3.0 (index 2): [-1, -1]
    /// </summary>
    [Fact]
    public void SumEncoder_3Categories_CorrectEncoding()
    {
        var encoder = new SumEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Category 1.0 (index 0): [1, 0]
        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 0, 1, 0.0);

        // Category 2.0 (index 1): [0, 1]
        AssertCell(result, 1, 0, 0.0);
        AssertCell(result, 1, 1, 1.0);

        // Category 3.0 (index 2, reference): [-1, -1]
        AssertCell(result, 2, 0, -1.0);
        AssertCell(result, 2, 1, -1.0);
    }

    /// <summary>
    /// Sum encoding column sums should equal k-2 (not zero).
    /// Col 0: 1 + 0 + (-1) = 0
    /// Col 1: 0 + 1 + (-1) = 0
    /// Actually column sums ARE zero for sum coding. Let me verify:
    /// k=3: each column has one +1, (k-2) zeros, and one -1 → sum = 1 + 0 - 1 = 0
    /// But for k=4: col 0 = [1, 0, 0, -1] → sum = 0; col 1 = [0, 1, 0, -1] → sum = 0
    /// So column sums are always zero.
    /// </summary>
    [Fact]
    public void SumEncoder_ColumnSumsAreZero()
    {
        var encoder = new SumEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double sum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                sum += result[row, col];
            }
            Assert.True(Math.Abs(sum) < 1e-10, $"Column {col} sum: expected 0, got {sum}");
        }
    }

    /// <summary>
    /// Sum encoding with k=4 categories {1,2,3,4}. Reference is category 4 (last).
    /// Category 1 (idx 0): [1, 0, 0]
    /// Category 2 (idx 1): [0, 1, 0]
    /// Category 3 (idx 2): [0, 0, 1]
    /// Category 4 (idx 3): [-1,-1,-1]
    /// </summary>
    [Fact]
    public void SumEncoder_4Categories_CorrectEncoding()
    {
        var encoder = new SumEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(4, result.Rows);
        Assert.Equal(3, result.Columns);

        // Category 1 (index 0): [1, 0, 0]
        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 0, 1, 0.0);
        AssertCell(result, 0, 2, 0.0);

        // Category 2 (index 1): [0, 1, 0]
        AssertCell(result, 1, 0, 0.0);
        AssertCell(result, 1, 1, 1.0);
        AssertCell(result, 1, 2, 0.0);

        // Category 3 (index 2): [0, 0, 1]
        AssertCell(result, 2, 0, 0.0);
        AssertCell(result, 2, 1, 0.0);
        AssertCell(result, 2, 2, 1.0);

        // Category 4 (index 3, reference): [-1, -1, -1]
        AssertCell(result, 3, 0, -1.0);
        AssertCell(result, 3, 1, -1.0);
        AssertCell(result, 3, 2, -1.0);
    }

    /// <summary>
    /// Sum encoding with k=2 (simplest case).
    /// Category 1 (index 0): [1]
    /// Category 2 (index 1, reference): [-1]
    /// </summary>
    [Fact]
    public void SumEncoder_2Categories_SimpleContrast()
    {
        var encoder = new SumEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(2, result.Rows);
        Assert.Equal(1, result.Columns);

        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 1, 0, -1.0);
    }

    /// <summary>
    /// Row sums of Sum encoding: reference row sums to -(k-1), other rows sum to 1.
    /// k=3: row 0 = [1, 0] → sum=1; row 1 = [0, 1] → sum=1; row 2 = [-1, -1] → sum=-2
    /// </summary>
    [Fact]
    public void SumEncoder_3Categories_RowSums()
    {
        var encoder = new SumEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Non-reference rows sum to 1
        double sum0 = result[0, 0] + result[0, 1];
        Assert.True(Math.Abs(sum0 - 1.0) < 1e-10, $"Row 0 sum: expected 1, got {sum0}");

        double sum1 = result[1, 0] + result[1, 1];
        Assert.True(Math.Abs(sum1 - 1.0) < 1e-10, $"Row 1 sum: expected 1, got {sum1}");

        // Reference row sums to -(k-1) = -2
        double sum2 = result[2, 0] + result[2, 1];
        Assert.True(Math.Abs(sum2 - (-2.0)) < 1e-10, $"Row 2 sum: expected -2, got {sum2}");
    }

    /// <summary>
    /// Unknown category with SumEncoderHandleUnknown.UseZeros gives all zeros.
    /// </summary>
    [Fact]
    public void SumEncoder_UnknownCategory_UseZeros_AllZeros()
    {
        var encoder = new SumEncoder<double>(handleUnknown: SumEncoderHandleUnknown.UseZeros);
        var fitData = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(fitData);

        var testData = M(new double[,] { { 99.0 } });
        var result = encoder.Transform(testData);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 0, 1, 0.0);
    }

    /// <summary>
    /// Unknown category with SumEncoderHandleUnknown.Error should throw.
    /// </summary>
    [Fact]
    public void SumEncoder_UnknownCategory_Error_Throws()
    {
        var encoder = new SumEncoder<double>(handleUnknown: SumEncoderHandleUnknown.Error);
        var fitData = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(fitData);

        var testData = M(new double[,] { { 99.0 } });
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    #endregion

    #region Feature Name Generation

    /// <summary>
    /// Helmert feature names follow {base}_helmert_{index} pattern.
    /// Reversed Helmert uses {base}_rev_helmert_{index}.
    /// </summary>
    [Fact]
    public void HelmertEncoder_FeatureNames_CorrectPattern()
    {
        var encoder = new HelmertEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut(new[] { "grade" });
        Assert.Equal(2, names.Length);
        Assert.Equal("grade_helmert_0", names[0]);
        Assert.Equal("grade_helmert_1", names[1]);
    }

    /// <summary>
    /// Reversed Helmert feature names use rev_helmert suffix.
    /// </summary>
    [Fact]
    public void HelmertEncoder_Reversed_FeatureNames_CorrectPattern()
    {
        var encoder = new HelmertEncoder<double>(reversed: true);
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut(new[] { "grade" });
        Assert.Equal(2, names.Length);
        Assert.Equal("grade_rev_helmert_0", names[0]);
        Assert.Equal("grade_rev_helmert_1", names[1]);
    }

    /// <summary>
    /// BackwardDifference feature names follow {base}_diff_{n+1}_vs_{n} pattern.
    /// </summary>
    [Fact]
    public void BackwardDifferenceEncoder_FeatureNames_CorrectPattern()
    {
        var encoder = new BackwardDifferenceEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut(new[] { "edu" });
        Assert.Equal(2, names.Length);
        Assert.Equal("edu_diff_1_vs_0", names[0]);
        Assert.Equal("edu_diff_2_vs_1", names[1]);
    }

    /// <summary>
    /// Sum encoder feature names follow {base}_sum_{index} pattern.
    /// </summary>
    [Fact]
    public void SumEncoder_FeatureNames_CorrectPattern()
    {
        var encoder = new SumEncoder<double>();
        var data = M(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        encoder.Fit(data);

        var names = encoder.GetFeatureNamesOut(new[] { "region" });
        Assert.Equal(2, names.Length);
        Assert.Equal("region_sum_0", names[0]);
        Assert.Equal("region_sum_1", names[1]);
    }

    #endregion

    #region Multi-Column and Pass-Through

    /// <summary>
    /// When only some columns are encoded, pass-through columns keep original values.
    /// </summary>
    [Fact]
    public void SumEncoder_PartialColumns_PassThrough()
    {
        var encoder = new SumEncoder<double>(columnIndices: new[] { 0 });
        var data = M(new double[,] { { 1.0, 42.0 }, { 2.0, 88.0 }, { 3.0, 77.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // 2 sum-encoded columns for col 0 + 1 pass-through for col 1 = 3 columns
        Assert.Equal(3, result.Columns);

        // Pass-through column (last)
        AssertCell(result, 0, 2, 42.0);
        AssertCell(result, 1, 2, 88.0);
        AssertCell(result, 2, 2, 77.0);
    }

    /// <summary>
    /// Categories are sorted numerically, so data order doesn't affect encoding.
    /// Data presented as [3, 1, 2] should still sort to [1, 2, 3] internally.
    /// </summary>
    [Fact]
    public void HelmertEncoder_CategoriesSorted_RegardlessOfInputOrder()
    {
        var encoder = new HelmertEncoder<double>();
        var data = M(new double[,] { { 3.0 }, { 1.0 }, { 2.0 } });
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Row 0 has value 3.0 (index 2 in sorted [1,2,3])
        // Helmert index 2: [-1/3, -1/2]
        AssertCell(result, 0, 0, -1.0 / 3.0);
        AssertCell(result, 0, 1, -1.0 / 2.0);

        // Row 1 has value 1.0 (index 0 in sorted [1,2,3])
        // Helmert index 0: [2/3, 0]
        AssertCell(result, 1, 0, 2.0 / 3.0);
        AssertCell(result, 1, 1, 0.0);

        // Row 2 has value 2.0 (index 1 in sorted [1,2,3])
        // Helmert index 1: [-1/3, 1/2]
        AssertCell(result, 2, 0, -1.0 / 3.0);
        AssertCell(result, 2, 1, 1.0 / 2.0);
    }

    #endregion
}
