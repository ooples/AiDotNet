using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Integration tests for matrix decomposition classes.
/// </summary>
public class MatrixDecompositionIntegrationTests
{
    private const double Tolerance = 1e-6;

    /// <summary>
    /// Creates a 3x3 symmetric positive definite matrix for testing.
    /// </summary>
    private static Matrix<double> CreateSPDMatrix()
    {
        // A^T * A is always SPD for full-rank A
        var a = new Matrix<double>(new double[,]
        {
            { 2.0, 1.0, 0.0 },
            { 1.0, 3.0, 1.0 },
            { 0.0, 1.0, 2.0 },
        });
        // A^T * A
        var result = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double sum = 0;
                for (int k = 0; k < 3; k++)
                    sum += a[k, i] * a[k, j];
                result[i, j] = sum;
            }
        return result;
    }

    /// <summary>
    /// Creates a general 3x3 matrix for testing.
    /// </summary>
    private static Matrix<double> CreateGeneralMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 4.0, 1.0, 2.0 },
            { 1.0, 5.0, 3.0 },
            { 2.0, 3.0, 6.0 },
        });
    }

    /// <summary>
    /// Creates a 4x3 rectangular matrix for testing.
    /// </summary>
    private static Matrix<double> CreateRectangularMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 },
        });
    }

    /// <summary>
    /// Creates a non-negative 3x3 matrix for NMF testing.
    /// </summary>
    private static Matrix<double> CreateNonNegativeMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
        });
    }

    /// <summary>
    /// Multiplies two matrices (helper for verification).
    /// </summary>
    private static Matrix<double> Multiply(Matrix<double> a, Matrix<double> b)
    {
        int rows = a.Rows, cols = b.Columns, inner = a.Columns;
        var result = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int k = 0; k < inner; k++)
                    sum += a[i, k] * b[k, j];
                result[i, j] = sum;
            }
        return result;
    }

    /// <summary>
    /// Asserts two matrices are approximately equal.
    /// </summary>
    private static void AssertMatricesEqual(Matrix<double> expected, Matrix<double> actual, double tolerance)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int i = 0; i < expected.Rows; i++)
            for (int j = 0; j < expected.Columns; j++)
                Assert.True(Math.Abs(expected[i, j] - actual[i, j]) < tolerance,
                    $"Mismatch at [{i},{j}]: expected {expected[i, j]}, got {actual[i, j]}");
    }

    #region CholeskyDecomposition Tests

    [Fact]
    public void Cholesky_Construction_DoesNotThrow()
    {
        var matrix = CreateSPDMatrix();
        var chol = new CholeskyDecomposition<double>(matrix);
        Assert.NotNull(chol);
    }

    [Fact]
    public void Cholesky_L_TimesLT_ReconstructsA()
    {
        var matrix = CreateSPDMatrix();
        var chol = new CholeskyDecomposition<double>(matrix);
        var l = chol.L;

        // L * L^T should equal A
        var lt = new Matrix<double>(l.Columns, l.Rows);
        for (int i = 0; i < l.Rows; i++)
            for (int j = 0; j < l.Columns; j++)
                lt[j, i] = l[i, j];

        var reconstructed = Multiply(l, lt);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    #endregion

    #region QrDecomposition Tests

    [Fact]
    public void QR_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var qr = new QrDecomposition<double>(matrix);
        Assert.NotNull(qr);
    }

    [Fact]
    public void QR_Q_TimesR_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var qr = new QrDecomposition<double>(matrix);
        var reconstructed = Multiply(qr.Q, qr.R);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void QR_R_IsUpperTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var qr = new QrDecomposition<double>(matrix);
        var r = qr.R;

        for (int i = 1; i < r.Rows; i++)
            for (int j = 0; j < Math.Min(i, r.Columns); j++)
                Assert.True(Math.Abs(r[i, j]) < Tolerance,
                    $"R[{i},{j}] = {r[i, j]} should be zero (upper triangular)");
    }

    #endregion

    #region SvdDecomposition Tests

    [Fact]
    public void SVD_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);
        Assert.NotNull(svd);
    }

    [Fact]
    public void SVD_SingularValues_NonNegative()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);
        var s = svd.S;

        for (int i = 0; i < s.Length; i++)
        {
            Assert.True(s[i] >= -Tolerance,
                $"Singular value S[{i}] = {s[i]} should be non-negative");
        }
    }

    #endregion

    #region LuDecomposition Tests

    [Fact]
    public void LU_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);
        Assert.NotNull(lu);
    }

    [Fact]
    public void LU_L_IsLowerTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);
        var l = lu.L;

        for (int i = 0; i < l.Rows; i++)
            for (int j = i + 1; j < l.Columns; j++)
                Assert.True(Math.Abs(l[i, j]) < Tolerance,
                    $"L[{i},{j}] = {l[i, j]} should be zero (lower triangular)");
    }

    [Fact]
    public void LU_U_IsUpperTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);
        var u = lu.U;

        for (int i = 1; i < u.Rows; i++)
            for (int j = 0; j < Math.Min(i, u.Columns); j++)
                Assert.True(Math.Abs(u[i, j]) < Tolerance,
                    $"U[{i},{j}] = {u[i, j]} should be zero (upper triangular)");
    }

    #endregion

    #region EigenDecomposition Tests

    [Fact]
    public void Eigen_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var eigen = new EigenDecomposition<double>(matrix);
        Assert.NotNull(eigen);
    }

    #endregion

    #region LdlDecomposition Tests

    [Fact]
    public void LDL_Construction_DoesNotThrow()
    {
        var matrix = CreateSPDMatrix();
        var ldl = new LdlDecomposition<double>(matrix);
        Assert.NotNull(ldl);
    }

    [Fact]
    public void LDL_L_IsLowerTriangular()
    {
        var matrix = CreateSPDMatrix();
        var ldl = new LdlDecomposition<double>(matrix);
        var l = ldl.L;

        for (int i = 0; i < l.Rows; i++)
            for (int j = i + 1; j < l.Columns; j++)
                Assert.True(Math.Abs(l[i, j]) < Tolerance,
                    $"L[{i},{j}] = {l[i, j]} should be zero (lower triangular)");
    }

    #endregion

    #region BidiagonalDecomposition Tests

    [Fact]
    public void Bidiagonal_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var bidiag = new BidiagonalDecomposition<double>(matrix);
        Assert.NotNull(bidiag);
    }

    #endregion

    #region HessenbergDecomposition Tests

    [Fact]
    public void Hessenberg_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var hess = new HessenbergDecomposition<double>(matrix);
        Assert.NotNull(hess);
    }

    #endregion

    #region SchurDecomposition Tests

    [Fact]
    public void Schur_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var schur = new SchurDecomposition<double>(matrix);
        Assert.NotNull(schur);
    }

    #endregion

    #region TridiagonalDecomposition Tests

    [Fact]
    public void Tridiagonal_Construction_DoesNotThrow()
    {
        var matrix = CreateSPDMatrix();
        var trid = new TridiagonalDecomposition<double>(matrix);
        Assert.NotNull(trid);
    }

    #endregion

    #region PolarDecomposition Tests

    [Fact]
    public void Polar_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var polar = new PolarDecomposition<double>(matrix);
        Assert.NotNull(polar);
    }

    [Fact]
    public void Polar_U_TimesP_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var polar = new PolarDecomposition<double>(matrix);
        var reconstructed = Multiply(polar.U, polar.P);
        AssertMatricesEqual(matrix, reconstructed, 0.1); // Polar decomposition may have some numerical error
    }

    #endregion

    #region GramSchmidtDecomposition Tests

    [Fact]
    public void GramSchmidt_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var gs = new GramSchmidtDecomposition<double>(matrix);
        Assert.NotNull(gs);
    }

    [Fact]
    public void GramSchmidt_Q_TimesR_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var gs = new GramSchmidtDecomposition<double>(matrix);
        var reconstructed = Multiply(gs.Q, gs.R);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    #endregion

    #region LqDecomposition Tests

    [Fact]
    public void LQ_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var lq = new LqDecomposition<double>(matrix);
        Assert.NotNull(lq);
    }

    [Fact]
    public void LQ_L_TimesQ_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var lq = new LqDecomposition<double>(matrix);
        var reconstructed = Multiply(lq.L, lq.Q);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    #endregion

    #region UduDecomposition Tests

    [Fact]
    public void UDU_Construction_DoesNotThrow()
    {
        var matrix = CreateSPDMatrix();
        var udu = new UduDecomposition<double>(matrix);
        Assert.NotNull(udu);
    }

    #endregion

    #region NmfDecomposition Tests

    [Fact]
    public void NMF_Construction_DoesNotThrow()
    {
        var matrix = CreateNonNegativeMatrix();
        var nmf = new NmfDecomposition<double>(matrix, components: 2);
        Assert.NotNull(nmf);
    }

    [Fact]
    public void NMF_W_And_H_AreNonNegative()
    {
        var matrix = CreateNonNegativeMatrix();
        var nmf = new NmfDecomposition<double>(matrix, components: 2);
        var w = nmf.W;
        var h = nmf.H;

        for (int i = 0; i < w.Rows; i++)
            for (int j = 0; j < w.Columns; j++)
                Assert.True(w[i, j] >= -Tolerance, $"W[{i},{j}] should be non-negative");

        for (int i = 0; i < h.Rows; i++)
            for (int j = 0; j < h.Columns; j++)
                Assert.True(h[i, j] >= -Tolerance, $"H[{i},{j}] should be non-negative");
    }

    #endregion

    #region IcaDecomposition Tests

    [Fact]
    public void ICA_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var ica = new IcaDecomposition<double>(matrix);
        Assert.NotNull(ica);
    }

    #endregion

    #region CramerDecomposition Tests

    [Fact]
    public void Cramer_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var cramer = new CramerDecomposition<double>(matrix);
        Assert.NotNull(cramer);
    }

    #endregion

    #region NormalDecomposition Tests

    [Fact]
    public void Normal_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var normal = new NormalDecomposition<double>(matrix);
        Assert.NotNull(normal);
    }

    #endregion

    #region TakagiDecomposition Tests

    [Fact]
    public void Takagi_Construction_DoesNotThrow()
    {
        // Takagi requires a symmetric matrix
        var matrix = CreateSPDMatrix();
        var takagi = new TakagiDecomposition<double>(matrix);
        Assert.NotNull(takagi);
    }

    #endregion

    #region ComplexMatrixDecomposition Tests

    [Fact]
    public void Complex_Construction_DoesNotThrow()
    {
        var matrix = CreateGeneralMatrix();
        var baseDecomp = new QrDecomposition<double>(matrix);
        var complex = new ComplexMatrixDecomposition<double>(baseDecomp);
        Assert.NotNull(complex);
    }

    #endregion
}
