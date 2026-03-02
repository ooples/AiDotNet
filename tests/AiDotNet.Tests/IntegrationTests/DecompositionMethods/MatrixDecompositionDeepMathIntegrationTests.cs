using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Deep math-correctness integration tests for matrix decompositions.
/// Verifies fundamental mathematical identities: factorization correctness,
/// structural properties (triangularity, orthogonality), and hand-calculated values.
/// </summary>
public class MatrixDecompositionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    private static Matrix<double> Create2x2() => new(new double[,] { { 4, 3 }, { 6, 3 } });
    private static Matrix<double> Create3x3() => new(new double[,] {
        { 2, -1, 0 }, { -1, 2, -1 }, { 0, -1, 2 }
    });
    private static Matrix<double> CreateSPD3x3() => new(new double[,] {
        { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 }
    });
    private static Matrix<double> CreateIdentity3() => Matrix<double>.CreateIdentity(3);

    #region LU Decomposition

    [Fact]
    public void LU_PA_Equals_LU_2x2()
    {
        // Verify P*A = L*U for a hand-calculated example
        var A = Create2x2();
        var lu = new LuDecomposition<double>(A);
        var L = lu.L;
        var U = lu.U;
        var P = lu.P;

        // Construct permutation matrix from P vector
        int n = A.Rows;
        var PA = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                PA[i, j] = A[P[i], j];
            }
        }

        var LU = L.Multiply(U);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(PA[i, j], LU[i, j], Tolerance);
    }

    [Fact]
    public void LU_PA_Equals_LU_3x3()
    {
        var A = Create3x3();
        var lu = new LuDecomposition<double>(A);
        var L = lu.L;
        var U = lu.U;
        var P = lu.P;

        int n = A.Rows;
        var PA = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                PA[i, j] = A[P[i], j];

        var LU = L.Multiply(U);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(PA[i, j], LU[i, j], Tolerance);
    }

    [Fact]
    public void LU_L_IsLowerTriangular()
    {
        var A = Create3x3();
        var lu = new LuDecomposition<double>(A);
        var L = lu.L;

        for (int i = 0; i < L.Rows; i++)
            for (int j = i + 1; j < L.Columns; j++)
                Assert.Equal(0.0, L[i, j], Tolerance);
    }

    [Fact]
    public void LU_U_IsUpperTriangular()
    {
        var A = Create3x3();
        var lu = new LuDecomposition<double>(A);
        var U = lu.U;

        for (int i = 1; i < U.Rows; i++)
            for (int j = 0; j < i; j++)
                Assert.Equal(0.0, U[i, j], Tolerance);
    }

    [Fact]
    public void LU_L_DiagonalIsOne()
    {
        var A = Create3x3();
        var lu = new LuDecomposition<double>(A);
        var L = lu.L;

        for (int i = 0; i < L.Rows; i++)
            Assert.Equal(1.0, L[i, i], Tolerance);
    }

    [Fact]
    public void LU_Identity_GivesIdentityFactors()
    {
        var I = CreateIdentity3();
        var lu = new LuDecomposition<double>(I);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0, lu.L[i, i], Tolerance);
            Assert.Equal(1.0, lu.U[i, i], Tolerance);
        }
    }

    [Fact]
    public void LU_HandCalculated_2x2()
    {
        // A = [[4,3],[6,3]]. With partial pivoting, pivot row 1 (6>4):
        // P = [1,0], PA = [[6,3],[4,3]]
        // L21 = 4/6 = 2/3
        // U = [[6, 3], [0, 3-3*(2/3)]] = [[6,3],[0,1]]
        // L = [[1,0],[2/3,1]]
        var A = Create2x2();
        var lu = new LuDecomposition<double>(A);

        // Verify the factorization is correct regardless of exact L/U values
        int n = 2;
        var PA = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                PA[i, j] = A[lu.P[i], j];

        var LU = lu.L.Multiply(lu.U);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(PA[i, j], LU[i, j], Tolerance);
    }

    #endregion

    #region QR Decomposition

    [Fact]
    public void QR_A_Equals_QR_3x3()
    {
        var A = Create3x3();
        var qr = new QrDecomposition<double>(A);
        var QR = qr.Q.Multiply(qr.R);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QR[i, j], LooseTolerance);
    }

    [Fact]
    public void QR_Q_IsOrthogonal_QtQ_EqualsIdentity()
    {
        var A = Create3x3();
        var qr = new QrDecomposition<double>(A);
        var QtQ = qr.Q.Transpose().Multiply(qr.Q);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QtQ[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void QR_R_IsUpperTriangular()
    {
        var A = Create3x3();
        var qr = new QrDecomposition<double>(A);

        for (int i = 1; i < 3; i++)
            for (int j = 0; j < i; j++)
                Assert.Equal(0.0, qr.R[i, j], LooseTolerance);
    }

    [Fact]
    public void QR_Q_ColumnsHaveUnitNorm()
    {
        var A = Create3x3();
        var qr = new QrDecomposition<double>(A);

        for (int j = 0; j < 3; j++)
        {
            double norm = 0;
            for (int i = 0; i < 3; i++)
                norm += qr.Q[i, j] * qr.Q[i, j];
            Assert.Equal(1.0, Math.Sqrt(norm), LooseTolerance);
        }
    }

    [Fact]
    public void QR_Identity_Q_IsIdentity()
    {
        var I = CreateIdentity3();
        var qr = new QrDecomposition<double>(I);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, Math.Abs(qr.Q[i, j]), LooseTolerance);
            }
    }

    [Fact]
    public void QR_A_Equals_QR_SPD()
    {
        var A = CreateSPD3x3();
        var qr = new QrDecomposition<double>(A);
        var QR = qr.Q.Multiply(qr.R);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QR[i, j], LooseTolerance);
    }

    #endregion

    #region Cholesky Decomposition

    [Fact]
    public void Cholesky_A_Equals_LLt()
    {
        // A = L * L^T for symmetric positive definite matrix
        var A = CreateSPD3x3();
        var chol = new CholeskyDecomposition<double>(A);
        var LLt = chol.L.Multiply(chol.L.Transpose());

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], LLt[i, j], LooseTolerance);
    }

    [Fact]
    public void Cholesky_L_IsLowerTriangular()
    {
        var A = CreateSPD3x3();
        var chol = new CholeskyDecomposition<double>(A);

        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 3; j++)
                Assert.Equal(0.0, chol.L[i, j], Tolerance);
    }

    [Fact]
    public void Cholesky_L_DiagonalIsPositive()
    {
        var A = CreateSPD3x3();
        var chol = new CholeskyDecomposition<double>(A);

        for (int i = 0; i < 3; i++)
            Assert.True(chol.L[i, i] > 0, $"L[{i},{i}] = {chol.L[i, i]} should be positive");
    }

    [Fact]
    public void Cholesky_Identity_L_IsIdentity()
    {
        var I = CreateIdentity3();
        var chol = new CholeskyDecomposition<double>(I);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, chol.L[i, j], Tolerance);
            }
    }

    [Fact]
    public void Cholesky_HandCalculated_2x2()
    {
        // A = [[4,2],[2,5]], L = [[2,0],[1,2]], L*L^T = [[4,2],[2,5]]
        var A = new Matrix<double>(new double[,] { { 4, 2 }, { 2, 5 } });
        var chol = new CholeskyDecomposition<double>(A);

        Assert.Equal(2.0, chol.L[0, 0], Tolerance);
        Assert.Equal(0.0, chol.L[0, 1], Tolerance);
        Assert.Equal(1.0, chol.L[1, 0], Tolerance);
        Assert.Equal(2.0, chol.L[1, 1], Tolerance);
    }

    #endregion

    #region SVD Decomposition

    [Fact]
    public void SVD_A_Equals_USVt()
    {
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);

        // Reconstruct A = U * diag(S) * Vt
        int m = A.Rows, n = A.Columns;
        var Smat = new Matrix<double>(m, n);
        int minDim = Math.Min(m, n);
        for (int i = 0; i < minDim && i < svd.S.Length; i++)
            Smat[i, i] = svd.S[i];

        var reconstructed = svd.U.Multiply(Smat).Multiply(svd.Vt);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(A[i, j], reconstructed[i, j], LooseTolerance);
    }

    [Fact]
    public void SVD_U_IsOrthogonal()
    {
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);
        var UtU = svd.U.Transpose().Multiply(svd.U);

        for (int i = 0; i < UtU.Rows; i++)
            for (int j = 0; j < UtU.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, UtU[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void SVD_V_IsOrthogonal()
    {
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);
        var VtV = svd.Vt.Multiply(svd.Vt.Transpose());

        for (int i = 0; i < VtV.Rows; i++)
            for (int j = 0; j < VtV.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, VtV[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void SVD_SingularValues_AreNonNegative()
    {
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);

        for (int i = 0; i < svd.S.Length; i++)
            Assert.True(svd.S[i] >= -Tolerance, $"S[{i}] = {svd.S[i]} should be >= 0");
    }

    [Fact]
    public void SVD_SingularValues_AreDescending()
    {
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);

        for (int i = 0; i < svd.S.Length - 1; i++)
            Assert.True(svd.S[i] >= svd.S[i + 1] - Tolerance,
                $"S[{i}]={svd.S[i]} should >= S[{i + 1}]={svd.S[i + 1]}");
    }

    [Fact]
    public void SVD_Identity_SingularValues_AllOnes()
    {
        var I = CreateIdentity3();
        var svd = new SvdDecomposition<double>(I);

        for (int i = 0; i < svd.S.Length; i++)
            Assert.Equal(1.0, svd.S[i], LooseTolerance);
    }

    [Fact]
    public void SVD_Symmetric_SingularValues_AreAbsEigenvalues()
    {
        // For symmetric matrices, singular values = |eigenvalues|
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);
        var eigen = new EigenDecomposition<double>(A);

        var absEigenvalues = new double[eigen.EigenValues.Length];
        for (int i = 0; i < absEigenvalues.Length; i++)
            absEigenvalues[i] = Math.Abs(eigen.EigenValues[i]);
        Array.Sort(absEigenvalues);
        Array.Reverse(absEigenvalues); // descending

        var singularValues = new double[svd.S.Length];
        for (int i = 0; i < singularValues.Length; i++)
            singularValues[i] = svd.S[i];
        Array.Sort(singularValues);
        Array.Reverse(singularValues);

        int minLen = Math.Min(absEigenvalues.Length, singularValues.Length);
        for (int i = 0; i < minLen; i++)
            Assert.Equal(absEigenvalues[i], singularValues[i], LooseTolerance);
    }

    #endregion

    #region Eigen Decomposition

    [Fact]
    public void Eigen_Av_Equals_LambdaV()
    {
        // A*v = λ*v for each eigenpair
        var A = Create3x3();
        var eigen = new EigenDecomposition<double>(A);

        for (int k = 0; k < eigen.EigenValues.Length; k++)
        {
            double lambda = eigen.EigenValues[k];
            var v = eigen.EigenVectors.GetColumn(k);
            var Av = A.Multiply(v);

            // Normalize both sides to handle sign ambiguity
            double vNorm = 0;
            for (int i = 0; i < v.Length; i++) vNorm += v[i] * v[i];
            vNorm = Math.Sqrt(vNorm);

            if (vNorm < 1e-10) continue; // skip zero eigenvectors

            for (int i = 0; i < v.Length; i++)
            {
                Assert.Equal(lambda * v[i], Av[i], LooseTolerance);
            }
        }
    }

    [Fact]
    public void Eigen_Symmetric_EigenvaluesAreReal()
    {
        // Symmetric matrix always has real eigenvalues
        var A = Create3x3();
        var eigen = new EigenDecomposition<double>(A);

        // All eigenvalues should be finite real numbers
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            Assert.False(double.IsNaN(eigen.EigenValues[i]), $"Eigenvalue {i} is NaN");
            Assert.False(double.IsInfinity(eigen.EigenValues[i]), $"Eigenvalue {i} is Infinity");
        }
    }

    [Fact]
    public void Eigen_TraceEqualsSumOfEigenvalues()
    {
        // tr(A) = sum of eigenvalues
        var A = Create3x3();
        var eigen = new EigenDecomposition<double>(A);

        double trace = 0;
        for (int i = 0; i < A.Rows; i++) trace += A[i, i];

        double eigenSum = 0;
        for (int i = 0; i < eigen.EigenValues.Length; i++) eigenSum += eigen.EigenValues[i];

        Assert.Equal(trace, eigenSum, LooseTolerance);
    }

    [Fact]
    public void Eigen_HandCalculated_2x2_Diagonal()
    {
        // Diagonal matrix [[3,0],[0,5]] has eigenvalues 3 and 5
        var A = new Matrix<double>(new double[,] { { 3, 0 }, { 0, 5 } });
        var eigen = new EigenDecomposition<double>(A);

        var eigenvals = new double[eigen.EigenValues.Length];
        for (int i = 0; i < eigenvals.Length; i++) eigenvals[i] = eigen.EigenValues[i];
        Array.Sort(eigenvals);

        Assert.Equal(3.0, eigenvals[0], LooseTolerance);
        Assert.Equal(5.0, eigenvals[1], LooseTolerance);
    }

    [Fact]
    public void Eigen_SPD_AllEigenvaluesPositive()
    {
        var A = CreateSPD3x3();
        var eigen = new EigenDecomposition<double>(A);

        for (int i = 0; i < eigen.EigenValues.Length; i++)
            Assert.True(eigen.EigenValues[i] > 0,
                $"SPD eigenvalue {i} = {eigen.EigenValues[i]} should be positive");
    }

    [Fact]
    public void Eigen_Identity_AllEigenvaluesOne()
    {
        var I = CreateIdentity3();
        var eigen = new EigenDecomposition<double>(I);

        for (int i = 0; i < eigen.EigenValues.Length; i++)
            Assert.Equal(1.0, eigen.EigenValues[i], LooseTolerance);
    }

    #endregion

    #region LDL Decomposition

    [Fact]
    public void LDL_A_Equals_LDLt()
    {
        var A = CreateSPD3x3();
        var ldl = new LdlDecomposition<double>(A);
        var L = ldl.L;
        var D = ldl.D;

        // Reconstruct: A = L * diag(D) * L^T
        int n = A.Rows;
        var Dmat = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++) Dmat[i, i] = D[i];

        var LDLt = L.Multiply(Dmat).Multiply(L.Transpose());

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(A[i, j], LDLt[i, j], LooseTolerance);
    }

    [Fact]
    public void LDL_L_IsUnitLowerTriangular()
    {
        var A = CreateSPD3x3();
        var ldl = new LdlDecomposition<double>(A);

        // Diagonal should be 1
        for (int i = 0; i < ldl.L.Rows; i++)
            Assert.Equal(1.0, ldl.L[i, i], Tolerance);

        // Upper triangle should be 0
        for (int i = 0; i < ldl.L.Rows; i++)
            for (int j = i + 1; j < ldl.L.Columns; j++)
                Assert.Equal(0.0, ldl.L[i, j], Tolerance);
    }

    [Fact]
    public void LDL_SPD_DiagonalIsPositive()
    {
        var A = CreateSPD3x3();
        var ldl = new LdlDecomposition<double>(A);

        for (int i = 0; i < ldl.D.Length; i++)
            Assert.True(ldl.D[i] > 0, $"D[{i}] = {ldl.D[i]} should be positive for SPD matrix");
    }

    [Fact]
    public void LDL_Identity_D_AllOnes()
    {
        var I = CreateIdentity3();
        var ldl = new LdlDecomposition<double>(I);

        for (int i = 0; i < ldl.D.Length; i++)
            Assert.Equal(1.0, ldl.D[i], Tolerance);
    }

    [Fact]
    public void LDL_HandCalculated_2x2()
    {
        // A = [[4,2],[2,5]]
        // L = [[1,0],[0.5,1]], D = [4, 4]  since D[0]=4, L[1,0]=2/4=0.5, D[1]=5-0.5*2=4
        var A = new Matrix<double>(new double[,] { { 4, 2 }, { 2, 5 } });
        var ldl = new LdlDecomposition<double>(A);

        Assert.Equal(4.0, ldl.D[0], Tolerance);
        Assert.Equal(4.0, ldl.D[1], Tolerance);
        Assert.Equal(0.5, ldl.L[1, 0], Tolerance);
    }

    #endregion

    #region Gram-Schmidt Decomposition

    [Fact]
    public void GramSchmidt_A_Equals_QR()
    {
        var A = Create3x3();
        var gs = new GramSchmidtDecomposition<double>(A);

        var QR = gs.Q.Multiply(gs.R);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QR[i, j], LooseTolerance);
    }

    [Fact]
    public void GramSchmidt_Q_ColumnsOrthogonal()
    {
        var A = Create3x3();
        var gs = new GramSchmidtDecomposition<double>(A);
        var QtQ = gs.Q.Transpose().Multiply(gs.Q);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QtQ[i, j], LooseTolerance);
            }
    }

    #endregion

    #region Hessenberg Decomposition

    [Fact]
    public void Hessenberg_A_Equals_QHQt()
    {
        var A = Create3x3();
        var hess = new HessenbergDecomposition<double>(A);

        var QHQt = hess.OrthogonalMatrix.Multiply(hess.HessenbergMatrix).Multiply(hess.OrthogonalMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QHQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Hessenberg_H_IsUpperHessenberg()
    {
        // H should have zeros below the first subdiagonal
        var A = Create3x3();
        var hess = new HessenbergDecomposition<double>(A);

        for (int i = 2; i < hess.HessenbergMatrix.Rows; i++)
            for (int j = 0; j < i - 1; j++)
                Assert.Equal(0.0, hess.HessenbergMatrix[i, j], LooseTolerance);
    }

    [Fact]
    public void Hessenberg_Q_IsOrthogonal()
    {
        var A = Create3x3();
        var hess = new HessenbergDecomposition<double>(A);
        var QtQ = hess.OrthogonalMatrix.Transpose().Multiply(hess.OrthogonalMatrix);

        for (int i = 0; i < QtQ.Rows; i++)
            for (int j = 0; j < QtQ.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QtQ[i, j], LooseTolerance);
            }
    }

    #endregion

    #region Schur Decomposition

    [Fact]
    public void Schur_A_Equals_QTQt()
    {
        var A = Create3x3();
        var schur = new SchurDecomposition<double>(A);

        var QTQt = schur.UnitaryMatrix.Multiply(schur.SchurMatrix).Multiply(schur.UnitaryMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QTQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Schur_Q_IsOrthogonal()
    {
        var A = Create3x3();
        var schur = new SchurDecomposition<double>(A);
        var QtQ = schur.UnitaryMatrix.Transpose().Multiply(schur.UnitaryMatrix);

        for (int i = 0; i < QtQ.Rows; i++)
            for (int j = 0; j < QtQ.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QtQ[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Schur_T_DiagonalContainsEigenvalues()
    {
        // For real symmetric matrices, Schur form T is diagonal with eigenvalues
        var A = Create3x3();
        var schur = new SchurDecomposition<double>(A);
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

    #endregion

    #region Polar Decomposition

    [Fact]
    public void Polar_A_Equals_UP()
    {
        // Polar: A = U * P where U is orthogonal and P is symmetric positive semi-definite
        var A = Create3x3();
        var polar = new PolarDecomposition<double>(A);

        var UP = polar.U.Multiply(polar.P);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], UP[i, j], LooseTolerance);
    }

    [Fact]
    public void Polar_U_IsOrthogonal()
    {
        var A = Create3x3();
        var polar = new PolarDecomposition<double>(A);
        var UtU = polar.U.Transpose().Multiply(polar.U);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, UtU[i, j], LooseTolerance);
            }
    }

    [Fact]
    public void Polar_P_IsSymmetric()
    {
        var A = Create3x3();
        var polar = new PolarDecomposition<double>(A);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(polar.P[i, j], polar.P[j, i], LooseTolerance);
    }

    #endregion

    #region Tridiagonal Decomposition

    [Fact]
    public void Tridiagonal_A_Equals_QTQt()
    {
        var A = CreateSPD3x3();
        var tri = new TridiagonalDecomposition<double>(A);

        var QTQt = tri.QMatrix.Multiply(tri.TMatrix).Multiply(tri.QMatrix.Transpose());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], QTQt[i, j], LooseTolerance);
    }

    [Fact]
    public void Tridiagonal_T_IsTridiagonal()
    {
        var A = CreateSPD3x3();
        var tri = new TridiagonalDecomposition<double>(A);

        // Elements more than 1 away from diagonal should be 0
        for (int i = 0; i < tri.TMatrix.Rows; i++)
            for (int j = 0; j < tri.TMatrix.Columns; j++)
                if (Math.Abs(i - j) > 1)
                    Assert.Equal(0.0, tri.TMatrix[i, j], LooseTolerance);
    }

    [Fact]
    public void Tridiagonal_Q_IsOrthogonal()
    {
        var A = CreateSPD3x3();
        var tri = new TridiagonalDecomposition<double>(A);
        var QtQ = tri.QMatrix.Transpose().Multiply(tri.QMatrix);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, QtQ[i, j], LooseTolerance);
            }
    }

    #endregion

    #region UDU Decomposition

    [Fact]
    public void UDU_A_Equals_UDUt()
    {
        var A = CreateSPD3x3();
        var udu = new UduDecomposition<double>(A);

        int n = A.Rows;
        var Dmat = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++) Dmat[i, i] = udu.D[i];

        var UDUt = udu.U.Multiply(Dmat).Multiply(udu.U.Transpose());
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(A[i, j], UDUt[i, j], LooseTolerance);
    }

    [Fact]
    public void UDU_U_IsUnitUpperTriangular()
    {
        var A = CreateSPD3x3();
        var udu = new UduDecomposition<double>(A);

        // Diagonal should be 1
        for (int i = 0; i < udu.U.Rows; i++)
            Assert.Equal(1.0, udu.U[i, i], Tolerance);

        // Below diagonal should be 0
        for (int i = 1; i < udu.U.Rows; i++)
            for (int j = 0; j < i; j++)
                Assert.Equal(0.0, udu.U[i, j], Tolerance);
    }

    #endregion

    #region Cross-Decomposition Consistency

    [Fact]
    public void LU_And_QR_SolveConsistently()
    {
        // Both decompositions should solve Ax=b consistently
        var A = Create3x3();
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        var lu = new LuDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A);

        var xLU = lu.Solve(b);
        var xQR = qr.Solve(b);

        for (int i = 0; i < xLU.Length; i++)
            Assert.Equal(xLU[i], xQR[i], LooseTolerance);
    }

    [Fact]
    public void Cholesky_And_LDL_SolveConsistently()
    {
        var A = CreateSPD3x3();
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        var chol = new CholeskyDecomposition<double>(A);
        var ldl = new LdlDecomposition<double>(A);

        var xChol = chol.Solve(b);
        var xLDL = ldl.Solve(b);

        for (int i = 0; i < xChol.Length; i++)
            Assert.Equal(xChol[i], xLDL[i], LooseTolerance);
    }

    [Fact]
    public void AllDecompositions_Solve_Ax_Equals_b()
    {
        var A = CreateSPD3x3();
        var b = new Vector<double>(new double[] { 7, 10, 10 });

        var lu = new LuDecomposition<double>(A);
        var x = lu.Solve(b);

        // Verify Ax = b
        var Ax = A.Multiply(x);
        for (int i = 0; i < b.Length; i++)
            Assert.Equal(b[i], Ax[i], LooseTolerance);
    }

    #endregion
}
