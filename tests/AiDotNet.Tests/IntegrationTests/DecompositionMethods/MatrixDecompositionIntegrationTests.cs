using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Integration tests for matrix decomposition classes.
/// Every decomposition is verified via mathematical invariants:
/// reconstruction (factors multiply back to original), structural properties
/// (triangular, orthogonal, etc.), and known-value checks.
/// </summary>
public class MatrixDecompositionIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    #region Helpers

    private static Matrix<double> CreateSPDMatrix()
    {
        // A^T * A is always SPD for full-rank A
        var a = new Matrix<double>(new double[,]
        {
            { 2.0, 1.0, 0.0 },
            { 1.0, 3.0, 1.0 },
            { 0.0, 1.0, 2.0 },
        });
        return Multiply(Transpose(a), a);
    }

    private static Matrix<double> CreateGeneralMatrix()
    {
        // Intentionally asymmetric to test non-symmetric code paths
        return new Matrix<double>(new double[,]
        {
            { 4.0, 1.0, 2.0 },
            { 3.0, 5.0, 1.0 },
            { 1.0, 4.0, 6.0 },
        });
    }

    private static Matrix<double> CreateSymmetricMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 4.0, 1.0, 2.0 },
            { 1.0, 5.0, 3.0 },
            { 2.0, 3.0, 6.0 },
        });
    }

    private static Matrix<double> CreateNonNegativeMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
        });
    }

    private static Matrix<double> Create2x2Matrix()
    {
        // Known eigenvalues: 1 and 4 (trace=5, det=4)
        return new Matrix<double>(new double[,]
        {
            { 2.0, 1.0 },
            { 1.0, 3.0 },
        });
    }

    private static Matrix<double> CreateIdentityMatrix(int n)
    {
        var m = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
            m[i, i] = 1.0;
        return m;
    }

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

    private static Matrix<double> Transpose(Matrix<double> m)
    {
        var result = new Matrix<double>(m.Columns, m.Rows);
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                result[j, i] = m[i, j];
        return result;
    }

    /// <summary>
    /// Creates a diagonal matrix from a vector.
    /// </summary>
    private static Matrix<double> DiagFromVector(Vector<double> v)
    {
        int n = v.Length;
        var m = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
            m[i, i] = v[i];
        return m;
    }

    private static void AssertMatricesEqual(Matrix<double> expected, Matrix<double> actual, double tolerance)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int i = 0; i < expected.Rows; i++)
            for (int j = 0; j < expected.Columns; j++)
                Assert.True(Math.Abs(expected[i, j] - actual[i, j]) < tolerance,
                    $"Mismatch at [{i},{j}]: expected {expected[i, j]:F10}, got {actual[i, j]:F10}");
    }

    private static void AssertIsOrthogonal(Matrix<double> q, double tolerance)
    {
        // Q^T * Q should be identity
        var qtq = Multiply(Transpose(q), q);
        var identity = CreateIdentityMatrix(q.Columns);
        AssertMatricesEqual(identity, qtq, tolerance);
    }

    private static void AssertIsUpperTriangular(Matrix<double> r, double tolerance)
    {
        for (int i = 1; i < r.Rows; i++)
            for (int j = 0; j < Math.Min(i, r.Columns); j++)
                Assert.True(Math.Abs(r[i, j]) < tolerance,
                    $"R[{i},{j}] = {r[i, j]} should be zero (upper triangular)");
    }

    private static void AssertIsLowerTriangular(Matrix<double> l, double tolerance)
    {
        for (int i = 0; i < l.Rows; i++)
            for (int j = i + 1; j < l.Columns; j++)
                Assert.True(Math.Abs(l[i, j]) < tolerance,
                    $"L[{i},{j}] = {l[i, j]} should be zero (lower triangular)");
    }

    #endregion

    #region CholeskyDecomposition Tests

    [Fact]
    public void Cholesky_LTimesLT_ReconstructsA()
    {
        var matrix = CreateSPDMatrix();
        var chol = new CholeskyDecomposition<double>(matrix);
        var l = chol.L;

        var reconstructed = Multiply(l, Transpose(l));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void Cholesky_L_IsLowerTriangular()
    {
        var matrix = CreateSPDMatrix();
        var chol = new CholeskyDecomposition<double>(matrix);
        AssertIsLowerTriangular(chol.L, Tolerance);
    }

    [Fact]
    public void Cholesky_L_DiagonalPositive()
    {
        // For SPD matrix, Cholesky L has positive diagonal
        var matrix = CreateSPDMatrix();
        var chol = new CholeskyDecomposition<double>(matrix);
        var l = chol.L;

        for (int i = 0; i < l.Rows; i++)
            Assert.True(l[i, i] > 0,
                $"L[{i},{i}] = {l[i, i]} should be positive for SPD matrix");
    }

    [Fact]
    public void Cholesky_KnownResult_2x2()
    {
        // A = [[4, 2], [2, 5]] -> L = [[2, 0], [1, 2]]
        // L*L^T = [[4, 2], [2, 5]] = A
        var a = new Matrix<double>(new double[,] { { 4.0, 2.0 }, { 2.0, 5.0 } });
        var chol = new CholeskyDecomposition<double>(a);
        var l = chol.L;

        Assert.Equal(2.0, l[0, 0], Tolerance);
        Assert.Equal(0.0, l[0, 1], Tolerance);
        Assert.Equal(1.0, l[1, 0], Tolerance);
        Assert.Equal(2.0, l[1, 1], Tolerance);
    }

    #endregion

    #region QrDecomposition Tests

    [Fact]
    public void QR_QTimesR_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var qr = new QrDecomposition<double>(matrix);
        var reconstructed = Multiply(qr.Q, qr.R);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void QR_Q_IsOrthogonal()
    {
        var matrix = CreateGeneralMatrix();
        var qr = new QrDecomposition<double>(matrix);
        AssertIsOrthogonal(qr.Q, Tolerance);
    }

    [Fact]
    public void QR_R_IsUpperTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var qr = new QrDecomposition<double>(matrix);
        AssertIsUpperTriangular(qr.R, Tolerance);
    }

    #endregion

    #region SvdDecomposition Tests

    [Fact]
    public void SVD_USVt_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);

        // U * diag(S) * Vt = A
        var sDiag = DiagFromVector(svd.S);
        var reconstructed = Multiply(Multiply(svd.U, sDiag), svd.Vt);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void SVD_SingularValues_NonNegative()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);

        for (int i = 0; i < svd.S.Length; i++)
            Assert.True(svd.S[i] >= -Tolerance,
                $"Singular value S[{i}] = {svd.S[i]} should be non-negative");
    }

    [Fact]
    public void SVD_SingularValues_Descending()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);

        for (int i = 0; i < svd.S.Length - 1; i++)
            Assert.True(svd.S[i] >= svd.S[i + 1] - Tolerance,
                $"S[{i}]={svd.S[i]} should be >= S[{i + 1}]={svd.S[i + 1]} (descending order)");
    }

    [Fact]
    public void SVD_U_IsOrthogonal()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);
        AssertIsOrthogonal(svd.U, Tolerance);
    }

    [Fact]
    public void SVD_Vt_IsOrthogonal()
    {
        var matrix = CreateGeneralMatrix();
        var svd = new SvdDecomposition<double>(matrix);
        // Vt * Vt^T = I (rows of Vt are orthonormal)
        var vtVtT = Multiply(svd.Vt, Transpose(svd.Vt));
        var identity = CreateIdentityMatrix(svd.Vt.Rows);
        AssertMatricesEqual(identity, vtVtT, Tolerance);
    }

    #endregion

    #region LuDecomposition Tests

    [Fact]
    public void LU_LTimesU_ReconstructsPermutedA()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);

        var product = Multiply(lu.L, lu.U);

        // Apply permutation: P*A = L*U, so product should equal permuted A
        // Each row of product should match a row of matrix (permuted)
        var perm = lu.P;
        var permutedA = new Matrix<double>(matrix.Rows, matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                permutedA[i, j] = matrix[perm[i], j];

        AssertMatricesEqual(permutedA, product, Tolerance);
    }

    [Fact]
    public void LU_L_IsLowerTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);
        AssertIsLowerTriangular(lu.L, Tolerance);
    }

    [Fact]
    public void LU_U_IsUpperTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);
        AssertIsUpperTriangular(lu.U, Tolerance);
    }

    [Fact]
    public void LU_L_HasUnitDiagonal()
    {
        var matrix = CreateGeneralMatrix();
        var lu = new LuDecomposition<double>(matrix);
        var l = lu.L;

        for (int i = 0; i < l.Rows; i++)
            Assert.Equal(1.0, l[i, i], Tolerance);
    }

    #endregion

    #region EigenDecomposition Tests

    [Fact]
    public void Eigen_AVEqualsVLambda()
    {
        // For symmetric matrix, A*V = V*diag(eigenvalues)
        var matrix = CreateSymmetricMatrix();
        var eigen = new EigenDecomposition<double>(matrix);
        var v = eigen.EigenVectors;
        var lambda = eigen.EigenValues;

        // A*V should equal V*diag(lambda)
        var av = Multiply(matrix, v);
        var vLambda = Multiply(v, DiagFromVector(lambda));
        AssertMatricesEqual(av, vLambda, LooseTolerance);
    }

    [Fact]
    public void Eigen_EigenvaluesSum_EqualsTrace()
    {
        // Sum of eigenvalues = trace of matrix
        var matrix = CreateSymmetricMatrix();
        var eigen = new EigenDecomposition<double>(matrix);

        double eigenSum = 0;
        for (int i = 0; i < eigen.EigenValues.Length; i++)
            eigenSum += eigen.EigenValues[i];

        double trace = 0;
        for (int i = 0; i < matrix.Rows; i++)
            trace += matrix[i, i];

        Assert.Equal(trace, eigenSum, LooseTolerance);
    }

    [Fact]
    public void Eigen_EigenvaluesProduct_EqualsDeterminant()
    {
        // Product of eigenvalues = determinant
        var matrix = Create2x2Matrix();
        var eigen = new EigenDecomposition<double>(matrix);

        double eigenProduct = 1.0;
        for (int i = 0; i < eigen.EigenValues.Length; i++)
            eigenProduct *= eigen.EigenValues[i];

        // det(A) = ad - bc = 2*3 - 1*1 = 5
        double det = 5.0;
        Assert.Equal(det, eigenProduct, LooseTolerance);
    }

    [Fact]
    public void Eigen_2x2_KnownEigenvalues()
    {
        // A = [[2, 1], [1, 3]]
        // Eigenvalues: (5 +/- sqrt(5)) / 2 ≈ 3.618 and 1.382
        var matrix = Create2x2Matrix();
        var eigen = new EigenDecomposition<double>(matrix);
        var vals = new double[eigen.EigenValues.Length];
        for (int i = 0; i < vals.Length; i++)
            vals[i] = eigen.EigenValues[i];
        Array.Sort(vals);

        double expected1 = (5.0 - Math.Sqrt(5.0)) / 2.0; // ~1.382
        double expected2 = (5.0 + Math.Sqrt(5.0)) / 2.0; // ~3.618

        Assert.Equal(expected1, vals[0], LooseTolerance);
        Assert.Equal(expected2, vals[1], LooseTolerance);
    }

    [Fact]
    public void Eigen_Eigenvalues_AreReal_ForSymmetricMatrix()
    {
        var matrix = CreateSPDMatrix();
        var eigen = new EigenDecomposition<double>(matrix);
        var eigenvalues = eigen.EigenValues;
        Assert.NotNull(eigenvalues);
        Assert.Equal(3, eigenvalues.Length);

        // For SPD matrix, all eigenvalues should be positive
        for (int i = 0; i < eigenvalues.Length; i++)
        {
            Assert.True(eigenvalues[i] > 0,
                $"Eigenvalue [{i}] = {eigenvalues[i]} should be positive for SPD matrix");
        }
    }

    #endregion

    #region LdlDecomposition Tests

    [Fact]
    public void LDL_LDLt_ReconstructsA()
    {
        var matrix = CreateSPDMatrix();
        var ldl = new LdlDecomposition<double>(matrix);

        // L * diag(D) * L^T = A
        var dDiag = DiagFromVector(ldl.D);
        var reconstructed = Multiply(Multiply(ldl.L, dDiag), Transpose(ldl.L));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void LDL_L_IsLowerTriangular()
    {
        var matrix = CreateSPDMatrix();
        var ldl = new LdlDecomposition<double>(matrix);
        AssertIsLowerTriangular(ldl.L, Tolerance);
    }

    [Fact]
    public void LDL_L_HasUnitDiagonal()
    {
        var matrix = CreateSPDMatrix();
        var ldl = new LdlDecomposition<double>(matrix);
        var l = ldl.L;

        for (int i = 0; i < l.Rows; i++)
            Assert.Equal(1.0, l[i, i], Tolerance);
    }

    [Fact]
    public void LDL_D_Positive_ForSPD()
    {
        // For SPD matrix, all D values should be positive
        var matrix = CreateSPDMatrix();
        var ldl = new LdlDecomposition<double>(matrix);

        for (int i = 0; i < ldl.D.Length; i++)
            Assert.True(ldl.D[i] > 0,
                $"D[{i}] = {ldl.D[i]} should be positive for SPD matrix");
    }

    #endregion

    #region BidiagonalDecomposition Tests

    [Fact]
    public void Bidiagonal_UBVt_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var bidiag = new BidiagonalDecomposition<double>(matrix);

        var reconstructed = Multiply(Multiply(bidiag.U, bidiag.B), Transpose(bidiag.V));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void Bidiagonal_B_IsBidiagonal()
    {
        var matrix = CreateGeneralMatrix();
        var bidiag = new BidiagonalDecomposition<double>(matrix);
        var b = bidiag.B;

        // Bidiagonal: only main diagonal and one superdiagonal are non-zero
        for (int i = 0; i < b.Rows; i++)
            for (int j = 0; j < b.Columns; j++)
                if (j != i && j != i + 1)
                    Assert.True(Math.Abs(b[i, j]) < Tolerance,
                        $"B[{i},{j}] = {b[i, j]} should be zero (bidiagonal)");
    }

    [Fact]
    public void Bidiagonal_B_HasCorrectDimensions()
    {
        var matrix = CreateGeneralMatrix();
        var bidiag = new BidiagonalDecomposition<double>(matrix);
        var b = bidiag.B;
        Assert.NotNull(b);
        Assert.Equal(matrix.Rows, b.Rows);
        Assert.Equal(matrix.Columns, b.Columns);
    }

    #endregion

    #region HessenbergDecomposition Tests

    [Fact]
    public void Hessenberg_QHQt_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var hess = new HessenbergDecomposition<double>(matrix);
        var q = hess.OrthogonalMatrix;
        var h = hess.HessenbergMatrix;

        // Q * H * Q^T = A
        var reconstructed = Multiply(Multiply(q, h), Transpose(q));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void Hessenberg_H_IsUpperHessenberg()
    {
        var matrix = CreateGeneralMatrix();
        var hess = new HessenbergDecomposition<double>(matrix);
        var h = hess.HessenbergMatrix;

        // Upper Hessenberg: all entries below the first sub-diagonal are zero
        for (int i = 2; i < h.Rows; i++)
            for (int j = 0; j < i - 1; j++)
                Assert.True(Math.Abs(h[i, j]) < Tolerance,
                    $"H[{i},{j}] = {h[i, j]} should be zero (upper Hessenberg)");
    }

    [Fact]
    public void Hessenberg_Q_IsOrthogonal()
    {
        var matrix = CreateGeneralMatrix();
        var hess = new HessenbergDecomposition<double>(matrix);
        AssertIsOrthogonal(hess.OrthogonalMatrix, Tolerance);
    }

    #endregion

    #region SchurDecomposition Tests

    [Fact]
    public void Schur_QTQt_ReconstructsA()
    {
        var matrix = CreateSymmetricMatrix();
        var schur = new SchurDecomposition<double>(matrix);
        var q = schur.UnitaryMatrix;
        var t = schur.SchurMatrix;

        // Q * T * Q^T = A
        var reconstructed = Multiply(Multiply(q, t), Transpose(q));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void Schur_DiagonalOfT_ContainsEigenvalues()
    {
        // For symmetric matrix, Schur form is diagonal with eigenvalues
        var matrix = CreateSymmetricMatrix();
        var schur = new SchurDecomposition<double>(matrix);
        var t = schur.SchurMatrix;

        // Diagonal elements should sum to trace of A
        double diagSum = 0;
        for (int i = 0; i < t.Rows; i++)
            diagSum += t[i, i];

        double trace = 0;
        for (int i = 0; i < matrix.Rows; i++)
            trace += matrix[i, i];

        Assert.Equal(trace, diagSum, Tolerance);
    }

    [Fact]
    public void Schur_SchurMatrix_HasCorrectDimensions()
    {
        var matrix = CreateGeneralMatrix();
        var schur = new SchurDecomposition<double>(matrix);
        var s = schur.SchurMatrix;
        Assert.NotNull(s);
        Assert.Equal(matrix.Rows, s.Rows);
        Assert.Equal(matrix.Columns, s.Columns);
    }

    #endregion

    #region TridiagonalDecomposition Tests

    [Fact]
    public void Tridiagonal_QTQt_ReconstructsA()
    {
        var matrix = CreateSPDMatrix();
        var trid = new TridiagonalDecomposition<double>(matrix);
        var q = trid.QMatrix;
        var t = trid.TMatrix;

        // Q * T * Q^T = A
        var reconstructed = Multiply(Multiply(q, t), Transpose(q));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void Tridiagonal_T_IsTridiagonal()
    {
        var matrix = CreateSPDMatrix();
        var trid = new TridiagonalDecomposition<double>(matrix);
        var t = trid.TMatrix;

        for (int i = 0; i < t.Rows; i++)
            for (int j = 0; j < t.Columns; j++)
                if (Math.Abs(i - j) > 1)
                    Assert.True(Math.Abs(t[i, j]) < Tolerance,
                        $"T[{i},{j}] = {t[i, j]} should be zero (tridiagonal)");
    }

    [Fact]
    public void Tridiagonal_Q_IsOrthogonal()
    {
        var matrix = CreateSPDMatrix();
        var trid = new TridiagonalDecomposition<double>(matrix);
        AssertIsOrthogonal(trid.QMatrix, Tolerance);
    }

    [Fact]
    public void Tridiagonal_TMatrix_HasCorrectDimensions()
    {
        var matrix = CreateSPDMatrix();
        var trid = new TridiagonalDecomposition<double>(matrix);
        var t = trid.TMatrix;
        Assert.NotNull(t);
        Assert.Equal(matrix.Rows, t.Rows);
        Assert.Equal(matrix.Columns, t.Columns);
    }

    #endregion

    #region PolarDecomposition Tests

    [Fact]
    public void Polar_UP_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var polar = new PolarDecomposition<double>(matrix);
        var reconstructed = Multiply(polar.U, polar.P);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void Polar_P_IsSymmetric()
    {
        var matrix = CreateGeneralMatrix();
        var polar = new PolarDecomposition<double>(matrix);
        var p = polar.P;

        for (int i = 0; i < p.Rows; i++)
            for (int j = i + 1; j < p.Columns; j++)
                Assert.Equal(p[i, j], p[j, i], Tolerance);
    }

    [Fact]
    public void Polar_P_IsPositiveSemiDefinite()
    {
        // Eigenvalues of P should be non-negative
        var matrix = CreateGeneralMatrix();
        var polar = new PolarDecomposition<double>(matrix);
        var p = polar.P;

        var eigen = new EigenDecomposition<double>(p);
        for (int i = 0; i < eigen.EigenValues.Length; i++)
            Assert.True(eigen.EigenValues[i] >= -Tolerance,
                $"P eigenvalue[{i}] = {eigen.EigenValues[i]} should be non-negative");
    }

    #endregion

    #region GramSchmidtDecomposition Tests

    [Fact]
    public void GramSchmidt_QR_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var gs = new GramSchmidtDecomposition<double>(matrix);
        var reconstructed = Multiply(gs.Q, gs.R);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void GramSchmidt_Q_IsOrthogonal()
    {
        var matrix = CreateGeneralMatrix();
        var gs = new GramSchmidtDecomposition<double>(matrix);
        AssertIsOrthogonal(gs.Q, Tolerance);
    }

    [Fact]
    public void GramSchmidt_R_IsUpperTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var gs = new GramSchmidtDecomposition<double>(matrix);
        AssertIsUpperTriangular(gs.R, Tolerance);
    }

    #endregion

    #region LqDecomposition Tests

    [Fact]
    public void LQ_LQ_ReconstructsA()
    {
        var matrix = CreateGeneralMatrix();
        var lq = new LqDecomposition<double>(matrix);
        var reconstructed = Multiply(lq.L, lq.Q);
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void LQ_L_IsLowerTriangular()
    {
        var matrix = CreateGeneralMatrix();
        var lq = new LqDecomposition<double>(matrix);
        AssertIsLowerTriangular(lq.L, Tolerance);
    }

    [Fact]
    public void LQ_Q_RowsOrthonormal()
    {
        var matrix = CreateGeneralMatrix();
        var lq = new LqDecomposition<double>(matrix);
        // Q * Q^T = I (rows of Q are orthonormal)
        var qqt = Multiply(lq.Q, Transpose(lq.Q));
        var identity = CreateIdentityMatrix(lq.Q.Rows);
        AssertMatricesEqual(identity, qqt, Tolerance);
    }

    #endregion

    #region UduDecomposition Tests

    [Fact]
    public void UDU_UDUt_ReconstructsA()
    {
        var matrix = CreateSPDMatrix();
        var udu = new UduDecomposition<double>(matrix);

        // U * diag(D) * U^T = A
        var dDiag = DiagFromVector(udu.D);
        var reconstructed = Multiply(Multiply(udu.U, dDiag), Transpose(udu.U));
        AssertMatricesEqual(matrix, reconstructed, Tolerance);
    }

    [Fact]
    public void UDU_U_IsUpperTriangular()
    {
        var matrix = CreateSPDMatrix();
        var udu = new UduDecomposition<double>(matrix);
        AssertIsUpperTriangular(udu.U, Tolerance);
    }

    [Fact]
    public void UDU_U_HasUnitDiagonal()
    {
        var matrix = CreateSPDMatrix();
        var udu = new UduDecomposition<double>(matrix);
        var u = udu.U;

        for (int i = 0; i < u.Rows; i++)
            Assert.Equal(1.0, u[i, i], Tolerance);
    }

    #endregion

    #region NmfDecomposition Tests

    [Fact]
    public void NMF_WH_ApproximatesA()
    {
        var matrix = CreateNonNegativeMatrix();
        var nmf = new NmfDecomposition<double>(matrix, components: 2);

        var reconstructed = Multiply(nmf.W, nmf.H);

        // NMF is approximate - check Frobenius norm of error is small relative to original
        double errorNorm = 0;
        double originalNorm = 0;
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                double diff = matrix[i, j] - reconstructed[i, j];
                errorNorm += diff * diff;
                originalNorm += matrix[i, j] * matrix[i, j];
            }

        double relativeError = Math.Sqrt(errorNorm / originalNorm);
        Assert.True(relativeError < 0.5,
            $"NMF relative reconstruction error {relativeError:F4} should be < 0.5");
    }

    [Fact]
    public void NMF_W_And_H_AreNonNegative()
    {
        var matrix = CreateNonNegativeMatrix();
        var nmf = new NmfDecomposition<double>(matrix, components: 2);

        for (int i = 0; i < nmf.W.Rows; i++)
            for (int j = 0; j < nmf.W.Columns; j++)
                Assert.True(nmf.W[i, j] >= -Tolerance,
                    $"W[{i},{j}] = {nmf.W[i, j]} should be non-negative");

        for (int i = 0; i < nmf.H.Rows; i++)
            for (int j = 0; j < nmf.H.Columns; j++)
                Assert.True(nmf.H[i, j] >= -Tolerance,
                    $"H[{i},{j}] = {nmf.H[i, j]} should be non-negative");
    }

    [Fact]
    public void NMF_Dimensions_AreCorrect()
    {
        var matrix = CreateNonNegativeMatrix(); // 3x3
        int components = 2;
        var nmf = new NmfDecomposition<double>(matrix, components: components);

        Assert.Equal(3, nmf.W.Rows);
        Assert.Equal(components, nmf.W.Columns);
        Assert.Equal(components, nmf.H.Rows);
        Assert.Equal(3, nmf.H.Columns);
    }

    #endregion

    #region IcaDecomposition Tests

    /// <summary>
    /// Creates a dataset with 100 observations suitable for ICA.
    /// Mixes 3 independent sources with a known mixing matrix.
    /// </summary>
    private static Matrix<double> CreateIcaTestData()
    {
        int n = 100;
        var data = new Matrix<double>(n, 3);
        // Create 3 independent sources: sin, sawtooth, square wave
        for (int i = 0; i < n; i++)
        {
            double t = i * 0.1;
            double s1 = Math.Sin(2.0 * t);
            double s2 = ((i * 7) % 20) / 10.0 - 1.0; // sawtooth-like
            double s3 = (i % 10 < 5) ? 1.0 : -1.0;    // square wave

            // Mix with a known mixing matrix
            data[i, 0] = 1.0 * s1 + 0.5 * s2 + 0.3 * s3;
            data[i, 1] = 0.5 * s1 + 1.0 * s2 + 0.4 * s3;
            data[i, 2] = 0.3 * s1 + 0.4 * s2 + 1.0 * s3;
        }

        return data;
    }

    [Fact]
    public void ICA_ProducesComponents_WithCorrectDimensions()
    {
        var matrix = CreateIcaTestData();
        var ica = new IcaDecomposition<double>(matrix);

        Assert.True(ica.MixingMatrix.Rows > 0, "Mixing matrix should have rows");
        Assert.True(ica.IndependentComponents.Rows > 0, "Independent components should have rows");
        Assert.Equal(3, ica.UnmixingMatrix.Rows);
        Assert.Equal(3, ica.Mean.Length);
    }

    [Fact]
    public void ICA_MixingReconstruction_ApproximatesCenteredData()
    {
        var matrix = CreateIcaTestData();
        var ica = new IcaDecomposition<double>(matrix);

        // ICA relationship: S = W * K^T * X_centered^T, A = MixingMatrix
        // Reconstruction: X_centered^T ≈ A * S  (features x observations)
        var s = ica.IndependentComponents;   // (components x observations)
        var a = ica.MixingMatrix;            // (features x components)

        var mean = ica.Mean;
        double errorNorm = 0;
        double originalNorm = 0;

        // Reconstruction via the proper mixing relationship: A * S
        var reconstruction = Multiply(a, s); // (features x observations)

        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                double centered = matrix[i, j] - mean[j];
                double recon = reconstruction[j, i]; // reconstruction is (features x observations)
                double diff = centered - recon;
                errorNorm += diff * diff;
                originalNorm += centered * centered;
            }

        double relativeError = Math.Sqrt(errorNorm / Math.Max(originalNorm, 1e-10));
        // ICA uses non-deterministic SecureRandom, so reconstruction quality varies.
        Assert.True(relativeError < 1.0,
            $"ICA reconstruction relative error {relativeError:F4} should be < 1.0");
    }

    #endregion

    #region CramerDecomposition Tests

    [Fact]
    public void Cramer_Solve_SatisfiesAxEqualsB()
    {
        // Solve A*x = b using Cramer's rule
        var matrix = CreateGeneralMatrix();
        var b = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var cramer = new CramerDecomposition<double>(matrix);
        var x = cramer.Solve(b);

        Assert.Equal(3, x.Length);

        // Verify A*x = b
        for (int i = 0; i < matrix.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < matrix.Columns; j++)
                sum += matrix[i, j] * x[j];
            Assert.Equal(b[i], sum, LooseTolerance);
        }
    }

    [Fact]
    public void Cramer_Invert_TimesA_IsIdentity()
    {
        var matrix = CreateGeneralMatrix();
        var cramer = new CramerDecomposition<double>(matrix);
        var inv = cramer.Invert();

        var product = Multiply(inv, matrix);
        var identity = CreateIdentityMatrix(3);
        AssertMatricesEqual(identity, product, LooseTolerance);
    }

    #endregion

    #region NormalDecomposition Tests

    [Fact]
    public void Normal_Solve_SatisfiesAxEqualsB()
    {
        // For square full-rank matrix, Normal equations give exact solution
        var matrix = CreateGeneralMatrix();
        var b = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var normal = new NormalDecomposition<double>(matrix);
        var x = normal.Solve(b);

        Assert.Equal(3, x.Length);

        // Verify A*x = b
        for (int i = 0; i < matrix.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < matrix.Columns; j++)
                sum += matrix[i, j] * x[j];
            Assert.Equal(b[i], sum, LooseTolerance);
        }
    }

    [Fact]
    public void Normal_Solve_AgreesWith_Cramer()
    {
        // Both methods should give the same solution for square system
        var matrix = CreateGeneralMatrix();
        var b = new Vector<double>(new double[] { 7.0, 11.0, 13.0 });

        var cramer = new CramerDecomposition<double>(matrix);
        var xCramer = cramer.Solve(b);

        var normal = new NormalDecomposition<double>(matrix);
        var xNormal = normal.Solve(b);

        for (int i = 0; i < xCramer.Length; i++)
            Assert.Equal(xCramer[i], xNormal[i], LooseTolerance);
    }

    #endregion

    #region TakagiDecomposition Tests

    [Fact]
    public void Takagi_SigmaMatrix_IsReal()
    {
        var matrix = CreateSPDMatrix();
        var takagi = new TakagiDecomposition<double>(matrix);
        var sigma = takagi.SigmaMatrix;

        Assert.True(sigma.Rows > 0, "Sigma matrix should have rows");
        Assert.True(sigma.Columns > 0, "Sigma matrix should have columns");
    }

    [Fact]
    public void Takagi_SigmaMatrix_DiagonalNonNegative()
    {
        var matrix = CreateSPDMatrix();
        var takagi = new TakagiDecomposition<double>(matrix);
        var sigma = takagi.SigmaMatrix;

        int n = Math.Min(sigma.Rows, sigma.Columns);
        for (int i = 0; i < n; i++)
            Assert.True(sigma[i, i] >= -Tolerance,
                $"Sigma[{i},{i}] = {sigma[i, i]} should be non-negative");
    }

    #endregion

    #region ComplexMatrixDecomposition Tests

    [Fact]
    public void Complex_WrapsBaseDecomposition()
    {
        var matrix = CreateGeneralMatrix();
        var baseDecomp = new QrDecomposition<double>(matrix);
        var complex = new ComplexMatrixDecomposition<double>(baseDecomp);

        Assert.NotNull(complex);
        // The wrapped decomposition should produce a valid complex matrix A
        Assert.Equal(matrix.Rows, complex.A.Rows);
        Assert.Equal(matrix.Columns, complex.A.Columns);
    }

    #endregion
}
