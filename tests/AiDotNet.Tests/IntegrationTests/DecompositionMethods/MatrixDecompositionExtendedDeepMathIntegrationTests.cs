using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Deep math-correctness integration tests for additional matrix decompositions:
/// Bidiagonal, LQ, Cramer, and cross-decomposition solve consistency.
/// </summary>
public class MatrixDecompositionExtendedDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    private static Matrix<double> Create3x3() => new(new double[,] {
        { 2, -1, 0 }, { -1, 2, -1 }, { 0, -1, 2 }
    });
    private static Matrix<double> CreateSPD3x3() => new(new double[,] {
        { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 }
    });
    private static Matrix<double> Create2x2() => new(new double[,] { { 4, 3 }, { 6, 3 } });
    private static Matrix<double> CreateIdentity3() => Matrix<double>.CreateIdentity(3);

    #region Bidiagonal Decomposition

    [Fact]
    public void Bidiagonal_A_Equals_UBVt()
    {
        var A = Create3x3();
        var bidiag = new BidiagonalDecomposition<double>(A);

        var UBVt = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());
        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.Equal(A[i, j], UBVt[i, j], LooseTolerance);
    }

    [Fact]
    public void Bidiagonal_U_IsOrthogonal()
    {
        var A = Create3x3();
        var bidiag = new BidiagonalDecomposition<double>(A);
        var UtU = bidiag.U.Transpose().Multiply(bidiag.U);

        for (int i = 0; i < UtU.Rows; i++)
            for (int j = 0; j < UtU.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, UtU[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Bidiagonal_V_IsOrthogonal()
    {
        var A = Create3x3();
        var bidiag = new BidiagonalDecomposition<double>(A);
        var VtV = bidiag.V.Transpose().Multiply(bidiag.V);

        for (int i = 0; i < VtV.Rows; i++)
            for (int j = 0; j < VtV.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, VtV[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Bidiagonal_B_IsBidiagonal()
    {
        // B should have non-zero only on main diagonal and superdiagonal
        var A = CreateSPD3x3();
        var bidiag = new BidiagonalDecomposition<double>(A);

        for (int i = 0; i < bidiag.B.Rows; i++)
            for (int j = 0; j < bidiag.B.Columns; j++)
                if (j != i && j != i + 1)
                    Assert.Equal(0.0, bidiag.B[i, j], LooseTolerance);
    }

    [Fact]
    public void Bidiagonal_Identity_B_IsIdentity()
    {
        // Identity matrix: U = I, B = I, V = I (or signed versions)
        var I = CreateIdentity3();
        var bidiag = new BidiagonalDecomposition<double>(I);

        // A = U*B*V^T should still equal I
        var UBVt = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, UBVt[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Bidiagonal_SPD_Factorization()
    {
        var A = CreateSPD3x3();
        var bidiag = new BidiagonalDecomposition<double>(A);

        var UBVt = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());
        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.Equal(A[i, j], UBVt[i, j], LooseTolerance);
    }

    #endregion

    #region LQ Decomposition

    [Fact]
    public void LQ_A_Equals_LQ()
    {
        var A = Create3x3();
        var lq = new LqDecomposition<double>(A);

        var LQ = lq.L.Multiply(lq.Q);
        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.Equal(A[i, j], LQ[i, j], LooseTolerance);
    }

    [Fact]
    public void LQ_L_IsLowerTriangular()
    {
        var A = Create3x3();
        var lq = new LqDecomposition<double>(A);

        for (int i = 0; i < lq.L.Rows; i++)
            for (int j = i + 1; j < lq.L.Columns; j++)
                Assert.Equal(0.0, lq.L[i, j], LooseTolerance);
    }

    [Fact]
    public void LQ_Q_IsOrthogonal()
    {
        var A = Create3x3();
        var lq = new LqDecomposition<double>(A);
        var QQt = lq.Q.Multiply(lq.Q.Transpose());

        for (int i = 0; i < QQt.Rows; i++)
            for (int j = 0; j < QQt.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QQt[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void LQ_Q_RowsHaveUnitNorm()
    {
        var A = Create3x3();
        var lq = new LqDecomposition<double>(A);

        for (int i = 0; i < lq.Q.Rows; i++)
        {
            double norm = 0;
            for (int j = 0; j < lq.Q.Columns; j++)
                norm += lq.Q[i, j] * lq.Q[i, j];
            Assert.Equal(1.0, Math.Sqrt(norm), LooseTolerance);
        }
    }

    [Fact]
    public void LQ_Identity_Factorization()
    {
        var I = CreateIdentity3();
        var lq = new LqDecomposition<double>(I);

        var LQ = lq.L.Multiply(lq.Q);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, LQ[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void LQ_SPD_Factorization()
    {
        var A = CreateSPD3x3();
        var lq = new LqDecomposition<double>(A);

        var LQ = lq.L.Multiply(lq.Q);
        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.Equal(A[i, j], LQ[i, j], LooseTolerance);
    }

    [Fact]
    public void LQ_Transpose_Equals_QR_Of_Transpose()
    {
        // LQ(A) is related to QR(A^T): if A = LQ, then A^T = Q^T * L^T = QR
        var A = CreateSPD3x3();
        var lq = new LqDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A.Transpose());

        // LQ: A = L*Q, so A^T = Q^T * L^T
        // QR: A^T = Q2 * R2
        // Both factorize A^T, so L^T should equal R2 (up to sign)
        // Verify: reconstructed A^T matches
        var At_from_lq = lq.Q.Transpose().Multiply(lq.L.Transpose());
        var At_from_qr = qr.Q.Multiply(qr.R);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(At_from_lq[i, j], At_from_qr[i, j], LooseTolerance);
    }

    #endregion

    #region Cramer Decomposition (Solve via determinants)

    [Fact]
    public void Cramer_Solve_2x2_HandCalculated()
    {
        // A = [[4,3],[6,3]], b = [10,12]
        // det(A) = 4*3 - 3*6 = 12 - 18 = -6
        // x = det([[10,3],[12,3]]) / det(A) = (30-36)/-6 = -6/-6 = 1
        // y = det([[4,10],[6,12]]) / det(A) = (48-60)/-6 = -12/-6 = 2
        var A = Create2x2();
        var b = new Vector<double>(new double[] { 10, 12 });
        var cramer = new CramerDecomposition<double>(A);
        var x = cramer.Solve(b);

        Assert.Equal(1.0, x[0], LooseTolerance);
        Assert.Equal(2.0, x[1], LooseTolerance);
    }

    [Fact]
    public void Cramer_Solve_Matches_LU()
    {
        var A = Create3x3();
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        var cramer = new CramerDecomposition<double>(A);
        var lu = new LuDecomposition<double>(A);

        var xCramer = cramer.Solve(b);
        var xLU = lu.Solve(b);

        for (int i = 0; i < xCramer.Length; i++)
            Assert.Equal(xLU[i], xCramer[i], LooseTolerance);
    }

    [Fact]
    public void Cramer_Solve_VerifyAxEqualsB()
    {
        var A = CreateSPD3x3();
        var b = new Vector<double>(new double[] { 7, 10, 10 });
        var cramer = new CramerDecomposition<double>(A);
        var x = cramer.Solve(b);

        var Ax = A.Multiply(x);
        for (int i = 0; i < b.Length; i++)
            Assert.Equal(b[i], Ax[i], LooseTolerance);
    }

    [Fact]
    public void Cramer_Solve_Identity_ReturnsB()
    {
        var I = CreateIdentity3();
        var b = new Vector<double>(new double[] { 5, 10, 15 });
        var cramer = new CramerDecomposition<double>(I);
        var x = cramer.Solve(b);

        for (int i = 0; i < b.Length; i++)
            Assert.Equal(b[i], x[i], LooseTolerance);
    }

    [Fact]
    public void Cramer_Invert_TimesA_Equals_Identity()
    {
        var A = Create2x2();
        var cramer = new CramerDecomposition<double>(A);
        var Ainv = cramer.Invert();
        var AinvA = Ainv.Multiply(A);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, AinvA[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Cramer_Invert_2x2_HandCalculated()
    {
        // A = [[4,3],[6,3]], det = -6
        // A^-1 = (1/det) * [[3,-3],[-6,4]] = [[-0.5, 0.5],[1, -2/3]]
        var A = Create2x2();
        var cramer = new CramerDecomposition<double>(A);
        var Ainv = cramer.Invert();

        Assert.Equal(-0.5, Ainv[0, 0], LooseTolerance);
        Assert.Equal(0.5, Ainv[0, 1], LooseTolerance);
        Assert.Equal(1.0, Ainv[1, 0], LooseTolerance);
        Assert.Equal(-2.0 / 3.0, Ainv[1, 1], LooseTolerance);
    }

    #endregion

    #region Cross-Decomposition Solve Consistency

    [Fact]
    public void AllSolvers_ConsistentResults_3x3()
    {
        var A = CreateSPD3x3();
        var b = new Vector<double>(new double[] { 7, 10, 10 });

        var lu = new LuDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A);
        var chol = new CholeskyDecomposition<double>(A);
        var ldl = new LdlDecomposition<double>(A);
        var cramer = new CramerDecomposition<double>(A);
        var hess = new HessenbergDecomposition<double>(A);
        var schur = new SchurDecomposition<double>(A);

        var xLU = lu.Solve(b);
        var xQR = qr.Solve(b);
        var xChol = chol.Solve(b);
        var xLDL = ldl.Solve(b);
        var xCramer = cramer.Solve(b);
        var xHess = hess.Solve(b);
        var xSchur = schur.Solve(b);

        // All should agree with LU as reference
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(xLU[i], xQR[i], LooseTolerance);
            Assert.Equal(xLU[i], xChol[i], LooseTolerance);
            Assert.Equal(xLU[i], xLDL[i], LooseTolerance);
            Assert.Equal(xLU[i], xCramer[i], LooseTolerance);
            Assert.Equal(xLU[i], xHess[i], LooseTolerance);
            Assert.Equal(xLU[i], xSchur[i], LooseTolerance);
        }
    }

    [Fact]
    public void AllInverters_ConsistentResults_3x3()
    {
        var A = CreateSPD3x3();

        var lu = new LuDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A);
        var cramer = new CramerDecomposition<double>(A);

        var invLU = lu.Invert();
        var invQR = qr.Invert();
        var invCramer = cramer.Invert();

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(invLU[i, j], invQR[i, j], LooseTolerance);
                Assert.Equal(invLU[i, j], invCramer[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Bidiagonal_And_SVD_SameMatrixIdentity()
    {
        // Both Bidiagonal and SVD decompose A into three factors
        // Verify both reconstruct the same matrix
        var A = CreateSPD3x3();
        var bidiag = new BidiagonalDecomposition<double>(A);
        var svd = new SvdDecomposition<double>(A);

        var fromBidiag = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());

        int m = A.Rows, n = A.Columns;
        var Smat = new Matrix<double>(m, n);
        int minDim = Math.Min(m, n);
        for (int i = 0; i < minDim && i < svd.S.Length; i++)
            Smat[i, i] = svd.S[i];
        var fromSVD = svd.U.Multiply(Smat).Multiply(svd.Vt);

        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
            {
                Assert.Equal(A[i, j], fromBidiag[i, j], LooseTolerance);
                Assert.Equal(A[i, j], fromSVD[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void LQ_And_QR_SolveConsistently()
    {
        var A = Create3x3();
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        var lq = new LqDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A);

        var xLQ = lq.Solve(b);
        var xQR = qr.Solve(b);

        for (int i = 0; i < xLQ.Length; i++)
            Assert.Equal(xQR[i], xLQ[i], LooseTolerance);
    }

    #endregion

    #region Hessenberg Algorithm Variants

    [Fact]
    public void Hessenberg_Givens_Factorization()
    {
        var A = Create3x3();
        var hess = new HessenbergDecomposition<double>(A, HessenbergAlgorithmType.Givens);

        var QHQt = hess.OrthogonalMatrix.Multiply(hess.HessenbergMatrix).Multiply(hess.OrthogonalMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QHQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Hessenberg_Givens_H_IsUpperHessenberg()
    {
        var A = CreateSPD3x3();
        var hess = new HessenbergDecomposition<double>(A, HessenbergAlgorithmType.Givens);

        for (int i = 2; i < hess.HessenbergMatrix.Rows; i++)
            for (int j = 0; j < i - 1; j++)
                Assert.Equal(0.0, hess.HessenbergMatrix[i, j], LooseTolerance);
    }

    [Fact]
    public void Hessenberg_Lanczos_Factorization()
    {
        var A = Create3x3();
        var hess = new HessenbergDecomposition<double>(A, HessenbergAlgorithmType.Lanczos);

        var QHQt = hess.OrthogonalMatrix.Multiply(hess.HessenbergMatrix).Multiply(hess.OrthogonalMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QHQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Hessenberg_AllAlgorithms_PreserveTrace()
    {
        // Hessenberg reduction preserves trace (similar matrices have same trace)
        var A = CreateSPD3x3();
        double traceA = 0;
        for (int i = 0; i < A.Rows; i++) traceA += A[i, i];

        var algorithms = new[]
        {
            HessenbergAlgorithmType.Householder,
            HessenbergAlgorithmType.Givens,
            HessenbergAlgorithmType.Lanczos,
        };

        foreach (var algo in algorithms)
        {
            var hess = new HessenbergDecomposition<double>(A, algo);
            double traceH = 0;
            for (int i = 0; i < hess.HessenbergMatrix.Rows; i++)
                traceH += hess.HessenbergMatrix[i, i];

            Assert.Equal(traceA, traceH, LooseTolerance);
        }
    }

    #endregion

    #region Schur Algorithm Variants

    [Fact]
    public void Schur_QR_Algorithm_Factorization()
    {
        var A = Create3x3();
        var schur = new SchurDecomposition<double>(A, SchurAlgorithmType.QR);

        var QSQt = schur.UnitaryMatrix.Multiply(schur.SchurMatrix).Multiply(schur.UnitaryMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QSQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Schur_QR_DiagonalContainsEigenvalues()
    {
        var A = Create3x3();
        var schur = new SchurDecomposition<double>(A, SchurAlgorithmType.QR);
        var eigen = new EigenDecomposition<double>(A);

        var schurDiag = new double[3];
        for (int i = 0; i < 3; i++) schurDiag[i] = schur.SchurMatrix[i, i];
        Array.Sort(schurDiag);

        var eigenvals = new double[eigen.EigenValues.Length];
        for (int i = 0; i < eigenvals.Length; i++) eigenvals[i] = eigen.EigenValues[i];
        Array.Sort(eigenvals);

        for (int i = 0; i < 3; i++)
            Assert.Equal(eigenvals[i], schurDiag[i], LooseTolerance);
    }

    [Fact]
    public void Schur_AllAlgorithms_PreserveTrace()
    {
        var A = Create3x3();
        double traceA = 0;
        for (int i = 0; i < A.Rows; i++) traceA += A[i, i];

        var algorithms = new[]
        {
            SchurAlgorithmType.Francis,
            SchurAlgorithmType.QR,
        };

        foreach (var algo in algorithms)
        {
            var schur = new SchurDecomposition<double>(A, algo);
            double traceS = 0;
            for (int i = 0; i < schur.SchurMatrix.Rows; i++)
                traceS += schur.SchurMatrix[i, i];

            Assert.Equal(traceA, traceS, LooseTolerance);
        }
    }

    #endregion

    #region Tridiagonal Algorithm Variants

    [Fact]
    public void Tridiagonal_Givens_Factorization()
    {
        var A = CreateSPD3x3();
        var tri = new TridiagonalDecomposition<double>(A, TridiagonalAlgorithmType.Givens);

        var QTQt = tri.QMatrix.Multiply(tri.TMatrix).Multiply(tri.QMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QTQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Tridiagonal_Lanczos_Factorization()
    {
        var A = CreateSPD3x3();
        var tri = new TridiagonalDecomposition<double>(A, TridiagonalAlgorithmType.Lanczos);

        var QTQt = tri.QMatrix.Multiply(tri.TMatrix).Multiply(tri.QMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QTQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Tridiagonal_AllAlgorithms_PreserveTrace()
    {
        var A = CreateSPD3x3();
        double traceA = 0;
        for (int i = 0; i < A.Rows; i++) traceA += A[i, i];

        var algorithms = new[]
        {
            TridiagonalAlgorithmType.Householder,
            TridiagonalAlgorithmType.Givens,
            TridiagonalAlgorithmType.Lanczos,
        };

        foreach (var algo in algorithms)
        {
            var tri = new TridiagonalDecomposition<double>(A, algo);
            double traceT = 0;
            for (int i = 0; i < tri.TMatrix.Rows; i++)
                traceT += tri.TMatrix[i, i];

            Assert.Equal(traceA, traceT, LooseTolerance);
        }
    }

    [Fact]
    public void Tridiagonal_Symmetric_T_IsSymmetric()
    {
        // For symmetric input, tridiagonal T should also be symmetric
        var A = CreateSPD3x3();
        var tri = new TridiagonalDecomposition<double>(A);

        for (int i = 0; i < tri.TMatrix.Rows; i++)
            for (int j = 0; j < tri.TMatrix.Columns; j++)
                if (Math.Abs(i - j) <= 1)
                    Assert.Equal(tri.TMatrix[i, j], tri.TMatrix[j, i], LooseTolerance);
    }

    #endregion

    #region GramSchmidt Algorithm Variants

    [Fact]
    public void GramSchmidt_Modified_A_Equals_QR()
    {
        var A = Create3x3();
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        var QR = gs.Q.Multiply(gs.R);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QR[i, j], LooseTolerance);
    }

    [Fact]
    public void GramSchmidt_Modified_Q_IsOrthogonal()
    {
        var A = CreateSPD3x3();
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);
        var QtQ = gs.Q.Transpose().Multiply(gs.Q);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QtQ[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void GramSchmidt_Classical_Vs_Modified_SameResult()
    {
        var A = CreateSPD3x3();
        var gsClassical = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Classical);
        var gsModified = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        // Both should reconstruct A correctly
        var QR_classical = gsClassical.Q.Multiply(gsClassical.R);
        var QR_modified = gsModified.Q.Multiply(gsModified.R);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(A[i, j], QR_classical[i, j], LooseTolerance);
                Assert.Equal(A[i, j], QR_modified[i, j], LooseTolerance);
            }
    }

    #endregion
}
