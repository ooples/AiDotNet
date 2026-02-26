using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Deep math-correctness integration tests for NMF decomposition, Normal decomposition,
/// and comprehensive Solve/Invert tests across all decomposition types.
/// </summary>
public class NMFAndNormalDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;
    private const double NmfTolerance = 0.5; // NMF is approximate

    private static Matrix<double> CreateSPD3x3() => new(new double[,] {
        { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 }
    });
    private static Matrix<double> Create3x3() => new(new double[,] {
        { 2, -1, 0 }, { -1, 2, -1 }, { 0, -1, 2 }
    });

    #region NMF Decomposition

    [Fact]
    public void NMF_WH_ApproximatesA()
    {
        // NMF: V ~= W * H for non-negative matrix
        // Use a rank-2 non-negative matrix (V = trueW * trueH) so 2-component NMF
        // should achieve near-perfect reconstruction
        // trueW = [[3,1],[2,1],[1,2],[0,3],[1,1]], trueH = [[1,2,0,1],[1,0,2,1]]
        var V = new Matrix<double>(new double[,] {
            { 4, 6, 2, 4 },
            { 3, 4, 2, 3 },
            { 3, 2, 4, 3 },
            { 3, 0, 6, 3 },
            { 2, 2, 2, 2 }
        });

        var nmf = new NmfDecomposition<double>(V, 2, 200, 1e-6);
        var WH = nmf.W.Multiply(nmf.H);

        // Rank-2 NMF on a rank-2 matrix should reconstruct closely
        double totalError = 0;
        for (int i = 0; i < V.Rows; i++)
            for (int j = 0; j < V.Columns; j++)
                totalError += Math.Abs(V[i, j] - WH[i, j]);

        double avgError = totalError / (V.Rows * V.Columns);
        Assert.True(avgError < NmfTolerance, $"Average NMF reconstruction error {avgError} should be < {NmfTolerance}");
    }

    [Fact]
    public void NMF_W_IsNonNegative()
    {
        var V = new Matrix<double>(new double[,] {
            { 5, 3, 1 }, { 4, 2, 1 }, { 1, 1, 5 }
        });

        var nmf = new NmfDecomposition<double>(V, 2);

        for (int i = 0; i < nmf.W.Rows; i++)
            for (int j = 0; j < nmf.W.Columns; j++)
                Assert.True(nmf.W[i, j] >= -Tolerance,
                    $"W[{i},{j}] = {nmf.W[i, j]} should be non-negative");
    }

    [Fact]
    public void NMF_H_IsNonNegative()
    {
        var V = new Matrix<double>(new double[,] {
            { 5, 3, 1 }, { 4, 2, 1 }, { 1, 1, 5 }
        });

        var nmf = new NmfDecomposition<double>(V, 2);

        for (int i = 0; i < nmf.H.Rows; i++)
            for (int j = 0; j < nmf.H.Columns; j++)
                Assert.True(nmf.H[i, j] >= -Tolerance,
                    $"H[{i},{j}] = {nmf.H[i, j]} should be non-negative");
    }

    [Fact]
    public void NMF_Dimensions_AreCorrect()
    {
        // V is m x n, W is m x k, H is k x n
        int m = 5, n = 4, k = 2;
        var V = new Matrix<double>(new double[,] {
            { 5, 3, 0, 1 },
            { 4, 0, 0, 1 },
            { 1, 1, 0, 5 },
            { 1, 0, 0, 4 },
            { 0, 1, 5, 4 }
        });

        var nmf = new NmfDecomposition<double>(V, k);

        Assert.Equal(m, nmf.W.Rows);
        Assert.Equal(k, nmf.W.Columns);
        Assert.Equal(k, nmf.H.Rows);
        Assert.Equal(n, nmf.H.Columns);
    }

    [Fact]
    public void NMF_MoreComponents_BetterApproximation()
    {
        var V = new Matrix<double>(new double[,] {
            { 5, 3, 0, 1 },
            { 4, 0, 0, 1 },
            { 1, 1, 0, 5 },
            { 1, 0, 0, 4 },
            { 0, 1, 5, 4 }
        });

        var nmf1 = new NmfDecomposition<double>(V, 1, 200, 1e-6);
        var nmf3 = new NmfDecomposition<double>(V, 3, 200, 1e-6);

        var WH1 = nmf1.W.Multiply(nmf1.H);
        var WH3 = nmf3.W.Multiply(nmf3.H);

        double error1 = 0, error3 = 0;
        for (int i = 0; i < V.Rows; i++)
            for (int j = 0; j < V.Columns; j++)
            {
                error1 += (V[i, j] - WH1[i, j]) * (V[i, j] - WH1[i, j]);
                error3 += (V[i, j] - WH3[i, j]) * (V[i, j] - WH3[i, j]);
            }

        Assert.True(error3 <= error1 + LooseTolerance,
            $"3-component error ({error3}) should be <= 1-component error ({error1})");
    }

    [Fact]
    public void NMF_IdentityLikeMatrix_ReconstructsWell()
    {
        // All-positive scaled identity should be reconstructed well
        var V = new Matrix<double>(new double[,] {
            { 5, 0, 0 }, { 0, 3, 0 }, { 0, 0, 7 }
        });

        var nmf = new NmfDecomposition<double>(V, 3, 200, 1e-6);
        var WH = nmf.W.Multiply(nmf.H);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(V[i, j], WH[i, j], NmfTolerance);
    }

    [Fact]
    public void NMF_NegativeMatrix_ThrowsException()
    {
        var V = new Matrix<double>(new double[,] {
            { 1, -1 }, { 2, 3 }
        });

        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(V, 1));
    }

    #endregion

    #region Normal Decomposition

    [Fact]
    public void Normal_Solve_AxEqualsB_SquareMatrix()
    {
        // Normal equations solve (A^T A)x = A^T b via Cholesky
        var A = CreateSPD3x3();
        var b = new Vector<double>(new double[] { 7, 10, 10 });
        var normal = new NormalDecomposition<double>(A);
        var x = normal.Solve(b);

        var Ax = A.Multiply(x);
        for (int i = 0; i < b.Length; i++)
            Assert.Equal(b[i], Ax[i], LooseTolerance);
    }

    [Fact]
    public void Normal_Solve_Consistent_WithLU()
    {
        var A = CreateSPD3x3();
        var b = new Vector<double>(new double[] { 7, 10, 10 });

        var normal = new NormalDecomposition<double>(A);
        var lu = new LuDecomposition<double>(A);

        var xNormal = normal.Solve(b);
        var xLU = lu.Solve(b);

        for (int i = 0; i < xNormal.Length; i++)
            Assert.Equal(xLU[i], xNormal[i], LooseTolerance);
    }

    [Fact]
    public void Normal_Solve_Identity_ReturnsB()
    {
        var I = Matrix<double>.CreateIdentity(3);
        var b = new Vector<double>(new double[] { 5, 10, 15 });
        var normal = new NormalDecomposition<double>(I);
        var x = normal.Solve(b);

        for (int i = 0; i < b.Length; i++)
            Assert.Equal(b[i], x[i], LooseTolerance);
    }

    [Fact]
    public void Normal_Invert_TimesA_Equals_Identity()
    {
        var A = CreateSPD3x3();
        var normal = new NormalDecomposition<double>(A);
        var Ainv = normal.Invert();
        var AinvA = Ainv.Multiply(A);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, AinvA[i, j], LooseTolerance);
            }
    }

    #endregion

    #region Deep Solve Tests - Hand-Calculated Solutions

    [Fact]
    public void Solve_HandCalculated_2x2_AllMethods()
    {
        // A = [[2,1],[1,3]], b = [5,10]
        // Solution: x1 = (5*3 - 1*10)/(2*3-1*1) = 5/5 = 1
        //           x2 = (2*10 - 5*1)/(2*3-1*1) = 15/5 = 3
        var A = new Matrix<double>(new double[,] { { 2, 1 }, { 1, 3 } });
        var b = new Vector<double>(new double[] { 5, 10 });

        var lu = new LuDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A);
        var chol = new CholeskyDecomposition<double>(A);
        var cramer = new CramerDecomposition<double>(A);

        var xLU = lu.Solve(b);
        var xQR = qr.Solve(b);
        var xChol = chol.Solve(b);
        var xCramer = cramer.Solve(b);

        Assert.Equal(1.0, xLU[0], LooseTolerance);
        Assert.Equal(3.0, xLU[1], LooseTolerance);
        Assert.Equal(1.0, xQR[0], LooseTolerance);
        Assert.Equal(3.0, xQR[1], LooseTolerance);
        Assert.Equal(1.0, xChol[0], LooseTolerance);
        Assert.Equal(3.0, xChol[1], LooseTolerance);
        Assert.Equal(1.0, xCramer[0], LooseTolerance);
        Assert.Equal(3.0, xCramer[1], LooseTolerance);
    }

    [Fact]
    public void Invert_HandCalculated_2x2_AllMethods()
    {
        // A = [[2,1],[1,3]], det = 5
        // A^-1 = (1/5)*[[3,-1],[-1,2]] = [[0.6,-0.2],[-0.2,0.4]]
        var A = new Matrix<double>(new double[,] { { 2, 1 }, { 1, 3 } });

        var lu = new LuDecomposition<double>(A);
        var qr = new QrDecomposition<double>(A);
        var cramer = new CramerDecomposition<double>(A);

        var invLU = lu.Invert();
        var invQR = qr.Invert();
        var invCramer = cramer.Invert();

        double[,] expected = { { 0.6, -0.2 }, { -0.2, 0.4 } };

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(expected[i, j], invLU[i, j], LooseTolerance);
                Assert.Equal(expected[i, j], invQR[i, j], LooseTolerance);
                Assert.Equal(expected[i, j], invCramer[i, j], LooseTolerance);
            }
    }

    #endregion

    #region Decomposition Properties on Larger Matrix

    [Fact]
    public void LU_4x4_PA_Equals_LU()
    {
        var A = new Matrix<double>(new double[,] {
            { 2, 1, 1, 0 },
            { 4, 3, 3, 1 },
            { 8, 7, 9, 5 },
            { 6, 7, 9, 8 }
        });
        var lu = new LuDecomposition<double>(A);

        int n = A.Rows;
        var PA = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                PA[i, j] = A[lu.P[i], j];

        var LU = lu.L.Multiply(lu.U);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(PA[i, j], LU[i, j], LooseTolerance);
    }

    [Fact]
    public void SVD_4x4_A_Equals_USVt()
    {
        var A = new Matrix<double>(new double[,] {
            { 1, 0, 0, 0 },
            { 0, 2, 0, 0 },
            { 0, 0, 3, 0 },
            { 0, 0, 0, 4 }
        });
        var svd = new SvdDecomposition<double>(A);

        int m = A.Rows, n = A.Columns;
        var Smat = new Matrix<double>(m, n);
        for (int i = 0; i < Math.Min(m, n) && i < svd.S.Length; i++)
            Smat[i, i] = svd.S[i];

        var reconstructed = svd.U.Multiply(Smat).Multiply(svd.Vt);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(A[i, j], reconstructed[i, j], LooseTolerance);

        // Diagonal matrix should have singular values 4, 3, 2, 1 (descending)
        Assert.Equal(4.0, svd.S[0], LooseTolerance);
        Assert.Equal(3.0, svd.S[1], LooseTolerance);
        Assert.Equal(2.0, svd.S[2], LooseTolerance);
        Assert.Equal(1.0, svd.S[3], LooseTolerance);
    }

    [Fact]
    public void Eigen_4x4_Diagonal_ExactEigenvalues()
    {
        var A = new Matrix<double>(new double[,] {
            { 1, 0, 0, 0 },
            { 0, 2, 0, 0 },
            { 0, 0, 3, 0 },
            { 0, 0, 0, 4 }
        });
        var eigen = new EigenDecomposition<double>(A);

        var eigenvals = new double[eigen.EigenValues.Length];
        for (int i = 0; i < eigenvals.Length; i++)
            eigenvals[i] = eigen.EigenValues[i];
        Array.Sort(eigenvals);

        Assert.Equal(1.0, eigenvals[0], LooseTolerance);
        Assert.Equal(2.0, eigenvals[1], LooseTolerance);
        Assert.Equal(3.0, eigenvals[2], LooseTolerance);
        Assert.Equal(4.0, eigenvals[3], LooseTolerance);
    }

    [Fact]
    public void Cholesky_4x4_SPD_LLt_Equals_A()
    {
        // 4x4 SPD matrix
        var A = new Matrix<double>(new double[,] {
            { 10, 2, 3, 1 },
            { 2, 8, 1, 4 },
            { 3, 1, 9, 2 },
            { 1, 4, 2, 7 }
        });
        var chol = new CholeskyDecomposition<double>(A);
        var LLt = chol.L.Multiply(chol.L.Transpose());

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(A[i, j], LLt[i, j], LooseTolerance);
    }

    #endregion

    #region Determinant-Like Properties

    [Fact]
    public void Eigen_ProductOfEigenvalues_EqualsDetViaLU()
    {
        // det(A) = product of eigenvalues
        var A = Create3x3();
        var eigen = new EigenDecomposition<double>(A);

        double eigProduct = 1.0;
        for (int i = 0; i < eigen.EigenValues.Length; i++)
            eigProduct *= eigen.EigenValues[i];

        // Compute det via LU
        var lu = new LuDecomposition<double>(A);
        double detLU = 1.0;
        for (int i = 0; i < lu.U.Rows; i++)
            detLU *= lu.U[i, i];
        // Account for permutation sign
        int swaps = 0;
        for (int i = 0; i < lu.P.Length; i++)
            if (lu.P[i] != i) swaps++;
        if (swaps % 2 == 1) detLU = -detLU;

        Assert.Equal(eigProduct, detLU, LooseTolerance);
    }

    [Fact]
    public void SVD_ProductOfSingularValues_EqualsAbsDet()
    {
        // |det(A)| = product of singular values
        var A = Create3x3();
        var svd = new SvdDecomposition<double>(A);

        double svProduct = 1.0;
        for (int i = 0; i < svd.S.Length; i++)
            svProduct *= svd.S[i];

        // Compute |det| via eigenvalues
        var eigen = new EigenDecomposition<double>(A);
        double absDet = 1.0;
        for (int i = 0; i < eigen.EigenValues.Length; i++)
            absDet *= eigen.EigenValues[i];
        absDet = Math.Abs(absDet);

        Assert.Equal(absDet, svProduct, LooseTolerance);
    }

    #endregion

    #region Condition Number

    [Fact]
    public void SVD_ConditionNumber_RatioOfExtremes()
    {
        // Condition number = max(S) / min(S)
        var A = CreateSPD3x3();
        var svd = new SvdDecomposition<double>(A);

        double maxS = svd.S[0];
        double minS = svd.S[svd.S.Length - 1];
        double conditionNumber = maxS / minS;

        // For our SPD matrix, condition number should be finite and > 1
        Assert.True(conditionNumber >= 1.0, $"Condition number {conditionNumber} should be >= 1");
        Assert.True(!double.IsInfinity(conditionNumber), "Condition number should be finite for non-singular matrix");
    }

    [Fact]
    public void SVD_Identity_ConditionNumber_IsOne()
    {
        var I = Matrix<double>.CreateIdentity(3);
        var svd = new SvdDecomposition<double>(I);

        double conditionNumber = svd.S[0] / svd.S[svd.S.Length - 1];
        Assert.Equal(1.0, conditionNumber, LooseTolerance);
    }

    #endregion
}
