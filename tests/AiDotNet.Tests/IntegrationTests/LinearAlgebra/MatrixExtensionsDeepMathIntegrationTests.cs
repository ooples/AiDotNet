using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Deep mathematical integration tests for MatrixExtensions.
/// Tests verify mathematical properties and identities that expose implementation bugs.
/// </summary>
public class MatrixExtensionsDeepMathIntegrationTests
{
    private const double Tol = 1e-6;
    private const double LooseTol = 1e-3; // For iterative methods

    // ─── Kronecker Product ──────────────────────────────────────────────

    [Fact]
    public void KroneckerProduct_TraceProperty_TrAkronB_Equals_TrA_Times_TrB()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1 },
            { 3, 4 }
        });
        var B = new Matrix<double>(new double[,]
        {
            { 5, 6 },
            { 7, 8 }
        });

        var AkB = A.KroneckerProduct(B);

        double trA = A[0, 0] + A[1, 1]; // 2 + 4 = 6
        double trB = B[0, 0] + B[1, 1]; // 5 + 8 = 13
        double trAkB = 0;
        for (int i = 0; i < AkB.Rows; i++) trAkB += AkB[i, i];

        Assert.Equal(trA * trB, trAkB, Tol);
    }

    [Fact]
    public void KroneckerProduct_DeterminantProperty_DetAkronB()
    {
        // det(A ⊗ B) = det(A)^q * det(B)^p where A is p×p, B is q×q
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1 },
            { 3, 4 }
        });
        var B = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 0, 3 }
        });

        double detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]; // 8 - 3 = 5
        double detB = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]; // 3 - 0 = 3

        var AkB = A.KroneckerProduct(B);
        double detAkB = MatrixHelper<double>.CalculateDeterminantRecursive(AkB);

        // p=2, q=2: det(A)^2 * det(B)^2 = 25 * 9 = 225
        double expected = Math.Pow(detA, 2) * Math.Pow(detB, 2);
        Assert.Equal(expected, detAkB, Tol);
    }

    [Fact]
    public void KroneckerProduct_MixedProductProperty()
    {
        // (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)
        var A = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var B = new Matrix<double>(new double[,] { { 5, 6 }, { 7, 8 } });
        var C = new Matrix<double>(new double[,] { { 2, 0 }, { 1, 3 } });
        var D = new Matrix<double>(new double[,] { { 1, 1 }, { 0, 2 } });

        var lhs = A.KroneckerProduct(B).Multiply(C.KroneckerProduct(D));
        var rhs = A.Multiply(C).KroneckerProduct(B.Multiply(D));

        for (int i = 0; i < lhs.Rows; i++)
            for (int j = 0; j < lhs.Columns; j++)
                Assert.Equal(rhs[i, j], lhs[i, j], Tol);
    }

    [Fact]
    public void KroneckerProduct_IdentityKronecker_IsBlockDiagonal()
    {
        // I_2 ⊗ A = block diagonal [A, 0; 0, A]
        var I2 = new Matrix<double>(new double[,] { { 1, 0 }, { 0, 1 } });
        var A = new Matrix<double>(new double[,] { { 3, 1 }, { 2, 5 } });

        var result = I2.KroneckerProduct(A);

        // Top-left block = A
        Assert.Equal(A[0, 0], result[0, 0], Tol);
        Assert.Equal(A[0, 1], result[0, 1], Tol);
        Assert.Equal(A[1, 0], result[1, 0], Tol);
        Assert.Equal(A[1, 1], result[1, 1], Tol);

        // Off-diagonal blocks = 0
        Assert.Equal(0.0, result[0, 2], Tol);
        Assert.Equal(0.0, result[0, 3], Tol);
        Assert.Equal(0.0, result[1, 2], Tol);
        Assert.Equal(0.0, result[1, 3], Tol);

        // Bottom-right block = A
        Assert.Equal(A[0, 0], result[2, 2], Tol);
        Assert.Equal(A[0, 1], result[2, 3], Tol);
        Assert.Equal(A[1, 0], result[3, 2], Tol);
        Assert.Equal(A[1, 1], result[3, 3], Tol);
    }

    [Fact]
    public void KroneckerProduct_Dimensions_mpByNq()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        }); // 2x3
        var B = new Matrix<double>(new double[,]
        {
            { 7, 8 },
            { 9, 10 },
            { 11, 12 }
        }); // 3x2

        var result = A.KroneckerProduct(B);

        Assert.Equal(6, result.Rows);    // 2*3
        Assert.Equal(6, result.Columns); // 3*2
    }

    // ─── Matrix Exponential ─────────────────────────────────────────────

    [Fact]
    public void MatrixExponential_ZeroMatrix_ReturnsIdentity()
    {
        var zero = new Matrix<double>(new double[,] { { 0, 0 }, { 0, 0 } });
        var expZero = zero.MatrixExponential();

        Assert.Equal(1.0, expZero[0, 0], Tol);
        Assert.Equal(0.0, expZero[0, 1], Tol);
        Assert.Equal(0.0, expZero[1, 0], Tol);
        Assert.Equal(1.0, expZero[1, 1], Tol);
    }

    [Fact]
    public void MatrixExponential_DiagonalMatrix_ExponentiatesDiagonal()
    {
        // exp(diag(a,b)) = diag(exp(a), exp(b))
        var D = new Matrix<double>(new double[,] { { 1, 0 }, { 0, 2 } });
        var expD = D.MatrixExponential(order: 15);

        Assert.Equal(Math.Exp(1), expD[0, 0], LooseTol);
        Assert.Equal(0.0, expD[0, 1], LooseTol);
        Assert.Equal(0.0, expD[1, 0], LooseTol);
        Assert.Equal(Math.Exp(2), expD[1, 1], LooseTol);
    }

    [Fact]
    public void MatrixExponential_DeterminantProperty_DetExpA_Equals_ExpTrA()
    {
        // det(exp(A)) = exp(tr(A)) — Jacobi's formula
        var A = new Matrix<double>(new double[,]
        {
            { 0.1, 0.2 },
            { 0.3, 0.4 }
        });

        var expA = A.MatrixExponential(order: 15);
        double detExpA = MatrixHelper<double>.CalculateDeterminantRecursive(expA);
        double trA = A[0, 0] + A[1, 1]; // 0.5
        double expTrA = Math.Exp(trA);

        Assert.Equal(expTrA, detExpA, LooseTol);
    }

    [Fact]
    public void MatrixExponential_ScalarMultiple_OfIdentity()
    {
        // exp(c*I) = exp(c)*I
        double c = 0.5;
        var cI = new Matrix<double>(new double[,] { { c, 0 }, { 0, c } });
        var expCI = cI.MatrixExponential(order: 12);

        double expC = Math.Exp(c);
        Assert.Equal(expC, expCI[0, 0], LooseTol);
        Assert.Equal(0.0, expCI[0, 1], LooseTol);
        Assert.Equal(0.0, expCI[1, 0], LooseTol);
        Assert.Equal(expC, expCI[1, 1], LooseTol);
    }

    [Fact]
    public void MatrixExponential_NilpotentMatrix_ExactResult()
    {
        // For nilpotent N (N^2=0): exp(N) = I + N
        var N = new Matrix<double>(new double[,]
        {
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 0, 0, 0 }
        });
        // N^2 = [[0,0,1],[0,0,0],[0,0,0]], N^3 = 0
        // exp(N) = I + N + N^2/2 = [[1,1,0.5],[0,1,1],[0,0,1]]
        var expN = N.MatrixExponential(order: 5);

        Assert.Equal(1.0, expN[0, 0], Tol);
        Assert.Equal(1.0, expN[0, 1], Tol);
        Assert.Equal(0.5, expN[0, 2], Tol);
        Assert.Equal(0.0, expN[1, 0], Tol);
        Assert.Equal(1.0, expN[1, 1], Tol);
        Assert.Equal(1.0, expN[1, 2], Tol);
        Assert.Equal(0.0, expN[2, 0], Tol);
        Assert.Equal(0.0, expN[2, 1], Tol);
        Assert.Equal(1.0, expN[2, 2], Tol);
    }

    // ─── Matrix Power ───────────────────────────────────────────────────

    [Fact]
    public void MatrixPower_ZeroPower_ReturnsIdentity()
    {
        var A = new Matrix<double>(new double[,] { { 3, 1 }, { 2, 5 } });
        var A0 = A.MatrixPower(0);

        Assert.Equal(1.0, A0[0, 0], Tol);
        Assert.Equal(0.0, A0[0, 1], Tol);
        Assert.Equal(0.0, A0[1, 0], Tol);
        Assert.Equal(1.0, A0[1, 1], Tol);
    }

    [Fact]
    public void MatrixPower_FirstPower_ReturnsOriginal()
    {
        var A = new Matrix<double>(new double[,] { { 3, 1 }, { 2, 5 } });
        var A1 = A.MatrixPower(1);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(A[i, j], A1[i, j], Tol);
    }

    [Fact]
    public void MatrixPower_AdditiveProperty_AmPlusN_Equals_Am_Times_An()
    {
        // A^(m+n) = A^m * A^n
        var A = new Matrix<double>(new double[,] { { 1, 2 }, { 0, 3 } });

        var A3 = A.MatrixPower(3);
        var A5 = A.MatrixPower(5);
        var A8 = A.MatrixPower(8);
        var A3xA5 = A3.Multiply(A5);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(A8[i, j], A3xA5[i, j], Tol);
    }

    [Fact]
    public void MatrixPower_DeterminantProperty_DetAk_Equals_DetA_k()
    {
        // det(A^k) = det(A)^k
        var A = new Matrix<double>(new double[,] { { 2, 1 }, { 1, 3 } });

        double detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]; // 5
        var A4 = A.MatrixPower(4);
        double detA4 = MatrixHelper<double>.CalculateDeterminantRecursive(A4);

        Assert.Equal(Math.Pow(detA, 4), detA4, Tol);
    }

    [Fact]
    public void MatrixPower_HandComputed_2x2_Squared()
    {
        // A = [[1,2],[3,4]], A^2 = [[7,10],[15,22]]
        var A = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var A2 = A.MatrixPower(2);

        Assert.Equal(7.0, A2[0, 0], Tol);
        Assert.Equal(10.0, A2[0, 1], Tol);
        Assert.Equal(15.0, A2[1, 0], Tol);
        Assert.Equal(22.0, A2[1, 1], Tol);
    }

    [Fact]
    public void MatrixPower_IdempotentMatrix_PowerEqualsOriginal()
    {
        // For a projection matrix P (P^2=P), P^k = P for all k >= 1
        // P = 1/2 * [[1,1],[1,1]] is idempotent
        var P = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5 },
            { 0.5, 0.5 }
        });

        var P5 = P.MatrixPower(5);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(P[i, j], P5[i, j], Tol);
    }

    // ─── Inverse Methods ────────────────────────────────────────────────

    [Fact]
    public void InverseGaussianJordan_TimesOriginal_IsIdentity()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 4, 7 },
            { 2, 6 }
        });

        var Ainv = A.InverseGaussianJordanElimination();
        var product = A.Multiply(Ainv);

        Assert.Equal(1.0, product[0, 0], Tol);
        Assert.Equal(0.0, product[0, 1], Tol);
        Assert.Equal(0.0, product[1, 0], Tol);
        Assert.Equal(1.0, product[1, 1], Tol);
    }

    [Fact]
    public void InverseGaussianJordan_3x3_HandComputed()
    {
        // A = [[1,2,3],[0,1,4],[5,6,0]]
        // det(A) = 1(0-24)-2(0-20)+3(0-5) = -24+40-15 = 1
        // A^-1 can be hand-computed
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 0, 1, 4 },
            { 5, 6, 0 }
        });

        var Ainv = A.InverseGaussianJordanElimination();
        var product = A.Multiply(Ainv);

        // Verify A * A^(-1) = I
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = i == j ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < Tol,
                    $"(A * A^-1)[{i},{j}] = {product[i, j]}, expected {expected}");
            }
    }

    [Fact]
    public void InverseNewton_TimesOriginal_IsIdentity()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1 },
            { 1, 3 }
        });

        var Ainv = A.InverseNewton(maxIterations: 200);
        var product = A.Multiply(Ainv);

        Assert.Equal(1.0, product[0, 0], LooseTol);
        Assert.Equal(0.0, product[0, 1], LooseTol);
        Assert.Equal(0.0, product[1, 0], LooseTol);
        Assert.Equal(1.0, product[1, 1], LooseTol);
    }

    [Fact]
    public void InverseStrassen_2x2_TimesOriginal_IsIdentity()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 3, 1 },
            { 2, 5 }
        });

        var Ainv = A.InverseStrassen();
        var product = A.Multiply(Ainv);

        Assert.Equal(1.0, product[0, 0], Tol);
        Assert.Equal(0.0, product[0, 1], Tol);
        Assert.Equal(0.0, product[1, 0], Tol);
        Assert.Equal(1.0, product[1, 1], Tol);
    }

    [Fact]
    public void InverseDispatcher_AllMethods_ProduceSameResult()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1 },
            { 1, 3 }
        });

        var gaussJordan = A.Inverse(InverseType.GaussianJordan);
        var newton = A.Inverse(InverseType.Newton, maxIterations: 200);
        var strassen = A.Inverse(InverseType.Strassen);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(gaussJordan[i, j], newton[i, j], LooseTol);
                Assert.Equal(gaussJordan[i, j], strassen[i, j], Tol);
            }
    }

    [Fact]
    public void InverseOfInverse_ReturnsOriginal()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1 },
            { 1, 3 }
        });

        var Ainv = A.InverseGaussianJordanElimination();
        var AinvInv = Ainv.InverseGaussianJordanElimination();

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(A[i, j], AinvInv[i, j], Tol);
    }

    // ─── Triangular Matrix Inversion ────────────────────────────────────

    [Fact]
    public void InvertUpperTriangular_TimesOriginal_IsIdentity()
    {
        var U = new Matrix<double>(new double[,]
        {
            { 2, 3, 1 },
            { 0, 4, 5 },
            { 0, 0, 6 }
        });

        var Uinv = U.InvertUpperTriangularMatrix();
        var product = U.Multiply(Uinv);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = i == j ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < Tol,
                    $"(U * U^-1)[{i},{j}] = {product[i, j]}, expected {expected}");
            }
    }

    [Fact]
    public void InvertUpperTriangular_ResultIsUpperTriangular()
    {
        var U = new Matrix<double>(new double[,]
        {
            { 3, 1, 2 },
            { 0, 5, 4 },
            { 0, 0, 7 }
        });

        var Uinv = U.InvertUpperTriangularMatrix();

        // Below diagonal should be zero
        Assert.Equal(0.0, Uinv[1, 0], Tol);
        Assert.Equal(0.0, Uinv[2, 0], Tol);
        Assert.Equal(0.0, Uinv[2, 1], Tol);
    }

    [Fact]
    public void InvertLowerTriangular_TimesOriginal_IsIdentity()
    {
        var L = new Matrix<double>(new double[,]
        {
            { 2, 0, 0 },
            { 3, 4, 0 },
            { 1, 5, 6 }
        });

        var Linv = L.InvertLowerTriangularMatrix();
        var product = L.Multiply(Linv);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double expected = i == j ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < Tol,
                    $"(L * L^-1)[{i},{j}] = {product[i, j]}, expected {expected}");
            }
    }

    [Fact]
    public void InvertLowerTriangular_ResultIsLowerTriangular()
    {
        var L = new Matrix<double>(new double[,]
        {
            { 3, 0, 0 },
            { 1, 5, 0 },
            { 2, 4, 7 }
        });

        var Linv = L.InvertLowerTriangularMatrix();

        // Above diagonal should be zero
        Assert.Equal(0.0, Linv[0, 1], Tol);
        Assert.Equal(0.0, Linv[0, 2], Tol);
        Assert.Equal(0.0, Linv[1, 2], Tol);
    }

    [Fact]
    public void InvertDiagonalMatrix_HandComputed()
    {
        var D = new Matrix<double>(new double[,]
        {
            { 2, 0, 0 },
            { 0, 5, 0 },
            { 0, 0, 4 }
        });

        var Dinv = D.InvertDiagonalMatrix();

        Assert.Equal(0.5, Dinv[0, 0], Tol);
        Assert.Equal(0.2, Dinv[1, 1], Tol);
        Assert.Equal(0.25, Dinv[2, 2], Tol);
    }

    // ─── Forward Substitution ───────────────────────────────────────────

    [Fact]
    public void ForwardSubstitution_HandComputed()
    {
        // Lx = b where L = [[2,0,0],[1,3,0],[4,5,6]], b = [4,7,38]
        // x1 = 4/2 = 2
        // x2 = (7-1*2)/3 = 5/3
        // x3 = (38-4*2-5*(5/3))/6 = (38-8-25/3)/6 = (114/3-24/3-25/3)/6 = 65/(3*6) = 65/18
        var L = new Matrix<double>(new double[,]
        {
            { 2, 0, 0 },
            { 1, 3, 0 },
            { 4, 5, 6 }
        });
        var b = new Vector<double>(new double[] { 4, 7, 38 });

        var x = L.ForwardSubstitution(b);

        // Verify Lx = b
        for (int i = 0; i < 3; i++)
        {
            double sum = 0;
            for (int j = 0; j < 3; j++) sum += L[i, j] * x[j];
            Assert.Equal(b[i], sum, Tol);
        }
    }

    [Fact]
    public void ForwardSubstitution_IdentityMatrix_ReturnsSameVector()
    {
        var I = new Matrix<double>(new double[,]
        {
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 }
        });
        var b = new Vector<double>(new double[] { 3, 7, 11 });

        var x = I.ForwardSubstitution(b);

        Assert.Equal(3.0, x[0], Tol);
        Assert.Equal(7.0, x[1], Tol);
        Assert.Equal(11.0, x[2], Tol);
    }

    // ─── Matrix Type Checks ─────────────────────────────────────────────

    [Fact]
    public void IsSymmetricMatrix_SymmetricInput_ReturnsTrue()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 2, 5, 6 },
            { 3, 6, 9 }
        });

        Assert.True(A.IsSymmetricMatrix());
    }

    [Fact]
    public void IsSymmetricMatrix_AsymmetricInput_ReturnsFalse()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        Assert.False(A.IsSymmetricMatrix());
    }

    [Fact]
    public void IsSkewSymmetricMatrix_SkewInput_ReturnsTrue()
    {
        // A = -A^T, diag = 0
        var A = new Matrix<double>(new double[,]
        {
            { 0, 2, -3 },
            { -2, 0, 5 },
            { 3, -5, 0 }
        });

        Assert.True(A.IsSkewSymmetricMatrix());
    }

    [Fact]
    public void IsPermutationMatrix_ValidPermutation_ReturnsTrue()
    {
        var P = new Matrix<double>(new double[,]
        {
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 1, 0, 0 }
        });

        Assert.True(P.IsPermutationMatrix());
    }

    [Fact]
    public void IsPermutationMatrix_NotPermutation_ReturnsFalse()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 0, 0 }
        });

        Assert.False(A.IsPermutationMatrix());
    }

    [Fact]
    public void IsToeplitzMatrix_ValidToeplitz_ReturnsTrue()
    {
        // Toeplitz: constant diagonals
        var T = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 1, 2 },
            { 5, 4, 1 }
        });

        Assert.True(T.IsToeplitzMatrix());
    }

    [Fact]
    public void IsHankelMatrix_ValidHankel_ReturnsTrue()
    {
        // Hankel: constant anti-diagonals
        var H = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 2, 3, 4 },
            { 3, 4, 5 }
        });

        Assert.True(H.IsHankelMatrix());
    }

    [Fact]
    public void IsStochasticMatrix_RowStochastic_ReturnsTrue()
    {
        // Row stochastic: non-negative, each row sums to 1
        var S = new Matrix<double>(new double[,]
        {
            { 0.3, 0.7 },
            { 0.5, 0.5 }
        });

        Assert.True(S.IsStochasticMatrix());
    }

    [Fact]
    public void IsDoublyStochasticMatrix_Valid_ReturnsTrue()
    {
        // Both rows and columns sum to 1
        var DS = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5 },
            { 0.5, 0.5 }
        });

        Assert.True(DS.IsDoublyStochasticMatrix());
    }

    [Fact]
    public void IsIdempotentMatrix_ProjectionMatrix_ReturnsTrue()
    {
        // P^2 = P
        var P = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5 },
            { 0.5, 0.5 }
        });

        Assert.True(P.IsIdempotentMatrix());
    }

    [Fact]
    public void IsInvolutoryMatrix_ValidInvolution_ReturnsTrue()
    {
        // A^2 = I
        var A = new Matrix<double>(new double[,]
        {
            { 1, 0 },
            { 0, -1 }
        });

        Assert.True(A.IsInvolutoryMatrix());
    }

    [Fact]
    public void IsPositiveDefiniteMatrix_SPDMatrix_ReturnsTrue()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 4, 2 },
            { 2, 5 }
        });

        Assert.True(A.IsPositiveDefiniteMatrix());
    }

    [Fact]
    public void IsPositiveDefiniteMatrix_IndefiniteMatrix_ReturnsFalse()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 3 },
            { 3, 1 }
        });
        // eigenvalues: 4 and -2, so NOT positive definite

        Assert.False(A.IsPositiveDefiniteMatrix());
    }

    // ─── Flatten and Reshape ────────────────────────────────────────────

    [Fact]
    public void Flatten_ThenReshape_ReturnsOriginal()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var flat = A.Flatten();
        Assert.Equal(6, flat.Length);

        // Reshape back
        var B = new Matrix<double>(2, 3);
        int idx = 0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                B[i, j] = flat[idx++];

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], B[i, j], Tol);
    }

    [Fact]
    public void Reshape_PreservesElementOrder()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        }); // 2x3

        var B = A.Reshape(3, 2); // 3x2

        // Row-major order: 1,2,3,4,5,6
        Assert.Equal(1.0, B[0, 0], Tol);
        Assert.Equal(2.0, B[0, 1], Tol);
        Assert.Equal(3.0, B[1, 0], Tol);
        Assert.Equal(4.0, B[1, 1], Tol);
        Assert.Equal(5.0, B[2, 0], Tol);
        Assert.Equal(6.0, B[2, 1], Tol);
    }

    // ─── Cross-Property Identities ──────────────────────────────────────

    [Fact]
    public void CrossProperty_InverseTranspose_Equals_TransposeInverse()
    {
        // (A^-1)^T = (A^T)^-1
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1 },
            { 1, 3 }
        });

        var AinvT = A.InverseGaussianJordanElimination().Transpose();
        var ATinv = A.Transpose().InverseGaussianJordanElimination();

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(AinvT[i, j], ATinv[i, j], Tol);
    }

    [Fact]
    public void CrossProperty_ProductInverse_Equals_ReversedInverseProduct()
    {
        // (AB)^-1 = B^-1 * A^-1
        var A = new Matrix<double>(new double[,] { { 2, 1 }, { 1, 3 } });
        var B = new Matrix<double>(new double[,] { { 4, 2 }, { 1, 5 } });

        var AB = A.Multiply(B);
        var ABinv = AB.InverseGaussianJordanElimination();
        var BinvAinv = B.InverseGaussianJordanElimination().Multiply(
            A.InverseGaussianJordanElimination());

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(ABinv[i, j], BinvAinv[i, j], Tol);
    }

    [Fact]
    public void CrossProperty_KroneckerInverse_Equals_InverseKronecker()
    {
        // (A ⊗ B)^-1 = A^-1 ⊗ B^-1
        var A = new Matrix<double>(new double[,] { { 2, 1 }, { 1, 3 } });
        var B = new Matrix<double>(new double[,] { { 4, 1 }, { 2, 5 } });

        var AkB = A.KroneckerProduct(B);
        var AkBinv = AkB.InverseGaussianJordanElimination();
        var AinvkBinv = A.InverseGaussianJordanElimination()
            .KroneckerProduct(B.InverseGaussianJordanElimination());

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(AkBinv[i, j], AinvkBinv[i, j], Tol);
    }

    [Fact]
    public void CrossProperty_PowerAndDet_AreConsistent()
    {
        // det(A^3) = det(A)^3
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 0 },
            { 0, 3, 1 },
            { 2, 0, 4 }
        });

        double detA = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        var A3 = A.MatrixPower(3);
        double detA3 = MatrixHelper<double>.CalculateDeterminantRecursive(A3);

        Assert.Equal(Math.Pow(detA, 3), detA3, Tol);
    }

    [Fact]
    public void CrossProperty_ExpAndInverse_ExpNegA_Equals_InvExpA()
    {
        // exp(-A) = (exp(A))^-1 for small A
        var A = new Matrix<double>(new double[,]
        {
            { 0.1, 0.05 },
            { 0.05, 0.2 }
        });

        var negA = A.Negate();
        var expNegA = negA.MatrixExponential(order: 15);
        var expAinv = A.MatrixExponential(order: 15).InverseGaussianJordanElimination();

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(expAinv[i, j], expNegA[i, j], LooseTol);
    }

    [Fact]
    public void Negate_TwiceReturnsOriginal()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, -2, 3 },
            { -4, 5, -6 }
        });

        var doubleNeg = A.Negate().Negate();

        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.Equal(A[i, j], doubleNeg[i, j], Tol);
    }

    [Fact]
    public void PointwiseMultiply_WithIdentityVector_PreservesRows()
    {
        // Pointwise multiply each row by [1,1,...] should preserve the matrix
        var A = new Matrix<double>(new double[,]
        {
            { 2, 3 },
            { 4, 5 }
        });
        var ones = new Vector<double>(new double[] { 1, 1 });

        var result = A.PointwiseMultiply(ones);

        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.Equal(A[i, j], result[i, j], Tol);
    }
}
