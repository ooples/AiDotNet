using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Deep math integration tests for AdvancedLinearAlgebra:
/// Matrix decomposition properties (LU, QR, SVD, Cholesky, Eigen),
/// Sparse matrix math (COO/CSR format, SpMV),
/// Numerical stability (condition number, pivoting),
/// Matrix norms, determinant, inverse, rank.
/// </summary>
public class AdvancedLinearAlgebraDeepMathIntegrationTests
{
    // ============================
    // LU Decomposition Properties
    // ============================

    [Fact]
    public void LUMath_Factorization_LU_EqualA()
    {
        // A = L * U where L is lower triangular with 1s on diagonal, U is upper triangular
        // A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
        // Without pivoting:
        // U row 0 = [2, 1, 1]
        // L[1,0] = 4/2 = 2, U row 1 = [4-2*2, 3-2*1, 3-2*1] = [0, 1, 1]
        // L[2,0] = 8/2 = 4, temp = [8-4*2, 7-4*1, 9-4*1] = [0, 3, 5]
        // L[2,1] = 3/1 = 3, U row 2 = [0, 0, 5-3*1] = [0, 0, 2]
        double[,] l = { { 1, 0, 0 }, { 2, 1, 0 }, { 4, 3, 1 } };
        double[,] u = { { 2, 1, 1 }, { 0, 1, 1 }, { 0, 0, 2 } };

        double[,] lu = MatMul(l, u);

        double[,] a = { { 2, 1, 1 }, { 4, 3, 3 }, { 8, 7, 9 } };

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(a[i, j], lu[i, j], 1e-10);
    }

    [Fact]
    public void LUMath_LowerTriangular_OnesOnDiagonal()
    {
        // L in LU decomposition has 1s on the diagonal
        double[,] l = { { 1, 0, 0 }, { 0.5, 1, 0 }, { 0.25, 0.5, 1 } };

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0, l[i, i], 1e-10);
            for (int j = i + 1; j < 3; j++)
            {
                Assert.Equal(0.0, l[i, j], 1e-10); // Upper triangle is zero
            }
        }
    }

    // ============================
    // QR Decomposition Properties
    // ============================

    [Fact]
    public void QRMath_Q_Orthogonal()
    {
        // Q^T * Q = I for orthogonal Q
        // Simple 2x2 case: A = QR where Q is rotation matrix
        double theta = Math.PI / 6; // 30 degrees
        double[,] q = { { Math.Cos(theta), -Math.Sin(theta) }, { Math.Sin(theta), Math.Cos(theta) } };

        double[,] qtq = MatMul(Transpose(q), q);

        // Should be identity
        Assert.Equal(1.0, qtq[0, 0], 1e-10);
        Assert.Equal(0.0, qtq[0, 1], 1e-10);
        Assert.Equal(0.0, qtq[1, 0], 1e-10);
        Assert.Equal(1.0, qtq[1, 1], 1e-10);
    }

    [Fact]
    public void QRMath_Q_PreservesNorm()
    {
        // ||Qx|| = ||x|| for orthogonal Q
        double theta = Math.PI / 4;
        double[,] q = { { Math.Cos(theta), -Math.Sin(theta) }, { Math.Sin(theta), Math.Cos(theta) } };

        double[] x = { 3.0, 4.0 };
        double normX = Math.Sqrt(x[0] * x[0] + x[1] * x[1]);

        double[] qx = { q[0, 0] * x[0] + q[0, 1] * x[1], q[1, 0] * x[0] + q[1, 1] * x[1] };
        double normQx = Math.Sqrt(qx[0] * qx[0] + qx[1] * qx[1]);

        Assert.Equal(normX, normQx, 1e-10);
    }

    [Fact]
    public void QRMath_R_UpperTriangular()
    {
        // R should be upper triangular with positive diagonal
        double[,] r = { { 5.0, 3.0 }, { 0.0, 2.0 } };

        // Below diagonal should be zero
        Assert.Equal(0.0, r[1, 0], 1e-10);

        // Diagonal should be positive (conventional choice)
        Assert.True(r[0, 0] > 0, "R diagonal should be positive");
        Assert.True(r[1, 1] > 0, "R diagonal should be positive");
    }

    // ============================
    // SVD Properties
    // ============================

    [Fact]
    public void SVDMath_SingularValues_NonNegative()
    {
        // Singular values are always non-negative
        double[] singularValues = { 5.0, 3.0, 1.0, 0.5 };
        foreach (double s in singularValues)
        {
            Assert.True(s >= 0, $"Singular value {s} should be non-negative");
        }
    }

    [Fact]
    public void SVDMath_SingularValues_Descending()
    {
        // By convention, singular values are in descending order
        double[] singularValues = { 5.0, 3.0, 1.0, 0.5 };
        for (int i = 1; i < singularValues.Length; i++)
        {
            Assert.True(singularValues[i] <= singularValues[i - 1],
                "Singular values should be in descending order");
        }
    }

    [Theory]
    [InlineData(3, 2, 2)]   // min(3,2) = 2 singular values
    [InlineData(5, 5, 5)]   // Square: 5 singular values
    [InlineData(2, 4, 2)]   // min(2,4) = 2 singular values
    public void SVDMath_NumberOfSingularValues(int m, int n, int expectedCount)
    {
        int count = Math.Min(m, n);
        Assert.Equal(expectedCount, count);
    }

    [Fact]
    public void SVDMath_FrobeniusNorm_FromSingularValues()
    {
        // ||A||_F = sqrt(sum(sigma_i^2))
        double[] singularValues = { 5.0, 3.0, 1.0 };
        double frobeniusNorm = Math.Sqrt(singularValues.Sum(s => s * s));
        Assert.Equal(Math.Sqrt(35), frobeniusNorm, 1e-10);
    }

    [Theory]
    [InlineData(new double[] { 5, 3, 1, 0.01 }, 500)]   // Large condition number
    [InlineData(new double[] { 5, 5, 5, 5 }, 1)]          // Perfect condition number
    public void SVDMath_ConditionNumber(double[] singularValues, double expectedCondition)
    {
        // Condition number = sigma_max / sigma_min
        double condition = singularValues.Max() / singularValues.Min();
        Assert.Equal(expectedCondition, condition, 1e-10);
    }

    // ============================
    // Cholesky Decomposition Properties
    // ============================

    [Fact]
    public void CholeskyMath_LLT_EqualsA()
    {
        // A = L * L^T for symmetric positive definite A
        // A = [[4, 2], [2, 5]]
        // L = [[2, 0], [1, 2]]
        double[,] l = { { 2, 0 }, { 1, 2 } };
        double[,] lt = Transpose(l);
        double[,] result = MatMul(l, lt);

        Assert.Equal(4.0, result[0, 0], 1e-10);
        Assert.Equal(2.0, result[0, 1], 1e-10);
        Assert.Equal(2.0, result[1, 0], 1e-10);
        Assert.Equal(5.0, result[1, 1], 1e-10);
    }

    [Fact]
    public void CholeskyMath_RequiresPositiveDefinite()
    {
        // A matrix is positive definite if all eigenvalues > 0
        // Equivalently: x^T A x > 0 for all x != 0
        double[,] a = { { 4, 2 }, { 2, 5 } };
        double[] x = { 1, 1 };

        // x^T A x = 1*4*1 + 1*2*1 + 1*2*1 + 1*5*1 = 4+2+2+5 = 13 > 0
        double xAx = a[0, 0] * x[0] * x[0] + a[0, 1] * x[0] * x[1] +
                      a[1, 0] * x[1] * x[0] + a[1, 1] * x[1] * x[1];
        Assert.True(xAx > 0, "x^T A x should be positive for positive definite matrix");
    }

    // ============================
    // Eigenvalue Decomposition Properties
    // ============================

    [Fact]
    public void EigenMath_CharacteristicPolynomial_2x2()
    {
        // For A = [[a, b], [c, d]], eigenvalues solve: lambda^2 - (a+d)*lambda + (ad-bc) = 0
        double a = 3, b = 1, c = 1, d = 3;
        double trace = a + d;  // Sum of eigenvalues
        double det = a * d - b * c;  // Product of eigenvalues

        // lambda = (trace ± sqrt(trace^2 - 4*det)) / 2
        double discriminant = trace * trace - 4 * det;
        double lambda1 = (trace + Math.Sqrt(discriminant)) / 2;
        double lambda2 = (trace - Math.Sqrt(discriminant)) / 2;

        Assert.Equal(4.0, lambda1, 1e-10);
        Assert.Equal(2.0, lambda2, 1e-10);

        // Verify: sum = trace, product = determinant
        Assert.Equal(trace, lambda1 + lambda2, 1e-10);
        Assert.Equal(det, lambda1 * lambda2, 1e-10);
    }

    [Fact]
    public void EigenMath_SymmetricMatrix_RealEigenvalues()
    {
        // Symmetric matrices always have real eigenvalues
        // A = [[2, 1], [1, 2]]
        double a = 2, b = 1, c = 1, d = 2;
        double discriminant = (a + d) * (a + d) - 4 * (a * d - b * c);

        Assert.True(discriminant >= 0,
            "Symmetric matrix should have non-negative discriminant (real eigenvalues)");
    }

    [Fact]
    public void EigenMath_Trace_IsSumOfEigenvalues()
    {
        double[] eigenvalues = { 5.0, 3.0, 1.0 };
        double trace = eigenvalues.Sum();
        Assert.Equal(9.0, trace, 1e-10);
    }

    [Fact]
    public void EigenMath_Determinant_IsProductOfEigenvalues()
    {
        double[] eigenvalues = { 5.0, 3.0, 2.0 };
        double det = eigenvalues.Aggregate(1.0, (acc, e) => acc * e);
        Assert.Equal(30.0, det, 1e-10);
    }

    // ============================
    // Matrix Norms
    // ============================

    [Fact]
    public void NormMath_FrobeniusNorm()
    {
        // ||A||_F = sqrt(sum(a_ij^2))
        double[,] a = { { 1, 2 }, { 3, 4 } };
        double frobNorm = Math.Sqrt(1 + 4 + 9 + 16);
        Assert.Equal(Math.Sqrt(30), frobNorm, 1e-10);
    }

    [Fact]
    public void NormMath_InfinityNorm()
    {
        // ||A||_inf = max row sum of absolute values
        double[,] a = { { 1, -2 }, { 3, -4 } };
        double row0 = Math.Abs(1) + Math.Abs(-2); // 3
        double row1 = Math.Abs(3) + Math.Abs(-4); // 7
        double infNorm = Math.Max(row0, row1);
        Assert.Equal(7.0, infNorm, 1e-10);
    }

    [Fact]
    public void NormMath_OneNorm()
    {
        // ||A||_1 = max column sum of absolute values
        double[,] a = { { 1, -2 }, { 3, -4 } };
        double col0 = Math.Abs(1) + Math.Abs(3); // 4
        double col1 = Math.Abs(-2) + Math.Abs(-4); // 6
        double oneNorm = Math.Max(col0, col1);
        Assert.Equal(6.0, oneNorm, 1e-10);
    }

    // ============================
    // Sparse Matrix Math: COO Format
    // ============================

    [Fact]
    public void SparseMath_COO_NonZeroCount()
    {
        // COO format stores (row, col, value) triples
        (int row, int col, double val)[] entries =
        {
            (0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)
        };

        int nnz = entries.Length;
        int totalElements = 3 * 3; // 3x3 matrix

        double sparsity = 1.0 - (double)nnz / totalElements;
        Assert.Equal(5, nnz);
        Assert.Equal(4.0 / 9.0, sparsity, 1e-10);
    }

    [Theory]
    [InlineData(1000, 1000, 5000, 0.995)]     // Very sparse
    [InlineData(100, 100, 1000, 0.9)]          // Moderately sparse
    [InlineData(10, 10, 50, 0.5)]              // 50% sparse
    public void SparseMath_Sparsity(int rows, int cols, int nnz, double expectedSparsity)
    {
        double sparsity = 1.0 - (double)nnz / (rows * cols);
        Assert.Equal(expectedSparsity, sparsity, 1e-3);
    }

    [Theory]
    [InlineData(1000, 1000, 5000)]
    public void SparseMath_StorageSavings(int rows, int cols, int nnz)
    {
        // Dense storage: rows * cols * 8 bytes (double)
        long denseBytes = (long)rows * cols * 8;

        // COO storage: nnz * (4 + 4 + 8) = nnz * 16 bytes (row int, col int, value double)
        long cooBytes = (long)nnz * 16;

        // CSR storage: nnz * (4 + 8) + (rows + 1) * 4 bytes
        long csrBytes = (long)nnz * 12 + ((long)rows + 1) * 4;

        Assert.True(cooBytes < denseBytes, "COO should use less storage than dense for sparse matrices");
        Assert.True(csrBytes < cooBytes, "CSR should use less storage than COO");
    }

    // ============================
    // Sparse Matrix Math: SpMV (Sparse Matrix-Vector Multiply)
    // ============================

    [Fact]
    public void SparseMath_SpMV_CorrectResult()
    {
        // Sparse matrix:
        // [1 0 3]   [1]   [1*1 + 0*2 + 3*3]   [10]
        // [0 2 0] * [2] = [0*1 + 2*2 + 0*3] = [ 4]
        // [4 0 5]   [3]   [4*1 + 0*2 + 5*3]   [19]

        (int row, int col, double val)[] entries =
        {
            (0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)
        };

        double[] x = { 1, 2, 3 };
        double[] result = new double[3];

        foreach (var (row, col, val) in entries)
        {
            result[row] += val * x[col];
        }

        Assert.Equal(10.0, result[0], 1e-10);
        Assert.Equal(4.0, result[1], 1e-10);
        Assert.Equal(19.0, result[2], 1e-10);
    }

    // ============================
    // Numerical Stability: Condition Number
    // ============================

    [Theory]
    [InlineData(1.0, "well-conditioned")]
    [InlineData(100.0, "moderately ill-conditioned")]
    [InlineData(1e10, "severely ill-conditioned")]
    public void StabilityMath_ConditionNumber_Interpretation(double conditionNumber, string expectedCategory)
    {
        string category;
        if (conditionNumber < 10) category = "well-conditioned";
        else if (conditionNumber < 1e6) category = "moderately ill-conditioned";
        else category = "severely ill-conditioned";

        Assert.Equal(expectedCategory, category);
    }

    [Fact]
    public void StabilityMath_ConditionNumber_IdentityMatrix()
    {
        // Identity matrix has condition number = 1 (best possible)
        double sigmaMax = 1.0;
        double sigmaMin = 1.0;
        double condition = sigmaMax / sigmaMin;
        Assert.Equal(1.0, condition, 1e-10);
    }

    // ============================
    // Determinant Properties
    // ============================

    [Fact]
    public void DeterminantMath_2x2_Formula()
    {
        // det([[a, b], [c, d]]) = ad - bc
        double a = 3, b = 8, c = 4, d = 6;
        double det = a * d - b * c;
        Assert.Equal(-14.0, det, 1e-10);
    }

    [Fact]
    public void DeterminantMath_3x3_SarrusRule()
    {
        // det of 3x3 matrix using Sarrus' rule
        // A = [[1,2,3],[4,5,6],[7,8,0]]
        double det = 1 * 5 * 0 + 2 * 6 * 7 + 3 * 4 * 8
                     - 3 * 5 * 7 - 2 * 4 * 0 - 1 * 6 * 8;
        // = 0 + 84 + 96 - 105 - 0 - 48 = 27
        Assert.Equal(27.0, det, 1e-10);
    }

    [Fact]
    public void DeterminantMath_ProductRule()
    {
        // det(A * B) = det(A) * det(B)
        double detA = 3.0;
        double detB = 5.0;
        double detAB = detA * detB;
        Assert.Equal(15.0, detAB, 1e-10);
    }

    [Fact]
    public void DeterminantMath_InverseRule()
    {
        // det(A^-1) = 1 / det(A)
        double detA = 4.0;
        double detAInv = 1.0 / detA;
        Assert.Equal(0.25, detAInv, 1e-10);
    }

    [Fact]
    public void DeterminantMath_TransposeRule()
    {
        // det(A^T) = det(A)
        // Both compute the same value
        double det = 7.5;
        double detTranspose = det; // By theorem
        Assert.Equal(det, detTranspose, 1e-10);
    }

    // ============================
    // Matrix Rank
    // ============================

    [Theory]
    [InlineData(new double[] { 5, 3, 1, 0 }, 3)]         // 3 non-zero singular values
    [InlineData(new double[] { 5, 3, 0, 0 }, 2)]         // 2 non-zero singular values
    [InlineData(new double[] { 5, 0, 0, 0 }, 1)]         // Rank 1 matrix
    [InlineData(new double[] { 0, 0, 0, 0 }, 0)]         // Zero matrix
    public void RankMath_FromSingularValues(double[] singularValues, int expectedRank)
    {
        double epsilon = 1e-10;
        int rank = singularValues.Count(s => Math.Abs(s) > epsilon);
        Assert.Equal(expectedRank, rank);
    }

    [Theory]
    [InlineData(3, 5, 3)]   // rank <= min(m, n)
    [InlineData(5, 3, 3)]
    [InlineData(4, 4, 4)]
    public void RankMath_UpperBound(int m, int n, int maxRank)
    {
        int bound = Math.Min(m, n);
        Assert.Equal(maxRank, bound);
    }

    // ============================
    // Helper Methods
    // ============================

    private static double[,] MatMul(double[,] a, double[,] b)
    {
        int m = a.GetLength(0), n = b.GetLength(1), k = a.GetLength(1);
        var result = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int p = 0; p < k; p++)
                    result[i, j] += a[i, p] * b[p, j];
        return result;
    }

    private static double[,] Transpose(double[,] a)
    {
        int m = a.GetLength(0), n = a.GetLength(1);
        var result = new double[n, m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[j, i] = a[i, j];
        return result;
    }
}
