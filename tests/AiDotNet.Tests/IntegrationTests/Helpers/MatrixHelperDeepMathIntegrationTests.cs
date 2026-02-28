using AiDotNet.Extensions;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Deep mathematical integration tests for MatrixHelper&lt;T&gt;.
/// Tests verify mathematical properties and identities that expose implementation bugs.
/// </summary>
public class MatrixHelperDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    // ─── Determinant Mathematical Properties ────────────────────────────

    [Fact]
    public void Determinant_Multiplicativity_DetAB_Equals_DetA_Times_DetB()
    {
        // det(AB) = det(A) * det(B)
        var A = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var B = new Matrix<double>(new double[,] { { 5, 6 }, { 7, 8 } });
        var AB = A.Multiply(B);

        double detA = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        double detB = MatrixHelper<double>.CalculateDeterminantRecursive(B);
        double detAB = MatrixHelper<double>.CalculateDeterminantRecursive(AB);

        // detA = 1*4-2*3 = -2, detB = 5*8-6*7 = -2, detAB should be 4
        Assert.Equal(-2.0, detA, Tol);
        Assert.Equal(-2.0, detB, Tol);
        Assert.Equal(detA * detB, detAB, Tol);
    }

    [Fact]
    public void Determinant_TransposeInvariance_DetAT_Equals_DetA()
    {
        var A = new Matrix<double>(new double[,] { { 2, 3, 1 }, { 1, 4, 5 }, { 6, 2, 3 } });
        var AT = A.Transpose();

        double detA = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        double detAT = MatrixHelper<double>.CalculateDeterminantRecursive(AT);

        Assert.Equal(detA, detAT, Tol);
    }

    [Fact]
    public void Determinant_ScalingProperty_Det_kA_Equals_kn_DetA()
    {
        // det(kA) = k^n * det(A) for n x n matrix
        var A = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        double k = 3.0;
        int n = 2;

        // Create kA manually
        var kA = new Matrix<double>(new double[,] { { k * 1, k * 2 }, { k * 3, k * 4 } });

        double detA = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        double detkA = MatrixHelper<double>.CalculateDeterminantRecursive(kA);

        Assert.Equal(Math.Pow(k, n) * detA, detkA, Tol);
    }

    [Fact]
    public void Determinant_TriangularMatrix_IsProductOfDiagonal()
    {
        var upper = new Matrix<double>(new double[,]
        {
            { 2, 5, 7 },
            { 0, 3, 8 },
            { 0, 0, 4 }
        });

        double det = MatrixHelper<double>.CalculateDeterminantRecursive(upper);

        // Product of diagonal: 2 * 3 * 4 = 24
        Assert.Equal(24.0, det, Tol);
    }

    [Fact]
    public void Determinant_RowSwap_NegatesDeterminant()
    {
        var A = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        // Swap rows: [[3,4],[1,2]]
        var B = new Matrix<double>(new double[,] { { 3, 4 }, { 1, 2 } });

        double detA = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        double detB = MatrixHelper<double>.CalculateDeterminantRecursive(B);

        Assert.Equal(-detA, detB, Tol);
    }

    [Fact]
    public void Determinant_OrthogonalMatrix_HasAbsValueOne()
    {
        // Rotation matrix is orthogonal with det = 1
        double theta = Math.PI / 4; // 45 degrees
        var R = new Matrix<double>(new double[,]
        {
            { Math.Cos(theta), -Math.Sin(theta) },
            { Math.Sin(theta),  Math.Cos(theta) }
        });

        double det = MatrixHelper<double>.CalculateDeterminantRecursive(R);

        Assert.Equal(1.0, Math.Abs(det), Tol);
    }

    [Fact]
    public void Determinant_4x4_HandComputed()
    {
        // Matrix with known determinant
        var A = new Matrix<double>(new double[,]
        {
            { 1, 0, 2, -1 },
            { 3, 0, 0,  5 },
            { 2, 1, 4, -3 },
            { 1, 0, 5,  0 }
        });
        // Expand along column 1 (lots of zeros): only i=2 contributes
        // det = 1 * C_21 where C_21 is cofactor with sign (-1)^(2+1)
        // Submatrix removing row 2, col 1: [[1,2,-1],[3,0,5],[1,5,0]]
        // det(sub) = 1*(0-25) - 2*(0-5) + (-1)*(15-0) = -25 + 10 - 15 = -30
        // Cofactor sign (-1)^3 = -1, so contribution = -1 * (-30) = 30
        double det = MatrixHelper<double>.CalculateDeterminantRecursive(A);

        Assert.Equal(30.0, det, Tol);
    }

    // ─── Givens Rotation ────────────────────────────────────────────────

    [Fact]
    public void GivensRotation_CSS_SumToOne()
    {
        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(3.0, 4.0);

        Assert.Equal(1.0, c * c + s * s, Tol);
    }

    [Fact]
    public void GivensRotation_AppliedToVector_ZeroesOneElement()
    {
        // Standard Givens: for (a,b), compute c,s so that applying rotation zeroes one element
        double a = 3.0, b = 4.0;
        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(a, b);

        // One of these should be zero (depending on convention):
        // Convention 1 (zero b): c*a + s*b = r, -s*a + c*b = 0
        // Convention 2 (zero a): c*a - s*b = 0, s*a + c*b = r
        double result1 = -s * a + c * b; // Standard rotation zeroes this
        double result2 = c * a - s * b;  // Alternative zeroes this

        // At least one should be zero
        bool zeroesAnElement = Math.Abs(result1) < Tol || Math.Abs(result2) < Tol;
        Assert.True(zeroesAnElement,
            $"Givens(3,4) gives c={c}, s={s}. " +
            $"-s*a+c*b={result1}, c*a-s*b={result2}. " +
            "Neither is zero - c and s don't produce a valid Givens rotation.");
    }

    [Fact]
    public void GivensRotation_PreservesNorm()
    {
        // For any valid rotation [c,s;-s,c], ||R*v|| = ||v||
        double a = 3.0, b = 4.0;
        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(a, b);

        double originalNorm = Math.Sqrt(a * a + b * b); // 5

        // Apply [c,s;-s,c] (standard rotation)
        double new1 = c * a + s * b;
        double new2 = -s * a + c * b;
        double rotatedNorm = Math.Sqrt(new1 * new1 + new2 * new2);

        Assert.Equal(originalNorm, rotatedNorm, Tol);
    }

    [Fact]
    public void ApplyGivensRotation_PreservesRowNormSum()
    {
        // An orthogonal row operation preserves the Frobenius norm of the affected rows
        var H = new Matrix<double>(new double[,]
        {
            { 3, 1, 4 },
            { 1, 5, 9 },
            { 2, 6, 5 }
        });

        // Compute norms of rows 0 and 1 before rotation
        double normBefore = 0;
        for (int k = 0; k < 3; k++)
        {
            normBefore += H[0, k] * H[0, k] + H[1, k] * H[1, k];
        }

        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(H[0, 0], H[1, 0]);
        MatrixHelper<double>.ApplyGivensRotation(H, c, s, 0, 1, 0, 3);

        // Compute norms of rows 0 and 1 after rotation
        double normAfter = 0;
        for (int k = 0; k < 3; k++)
        {
            normAfter += H[0, k] * H[0, k] + H[1, k] * H[1, k];
        }

        // Orthogonal transformation preserves Frobenius norm of affected rows
        Assert.Equal(normBefore, normAfter, Tol);
    }

    [Fact]
    public void ApplyGivensRotation_TransformationIsOrthogonal()
    {
        // The 2x2 transformation matrix [c,s;row2] should have determinant ±1
        // and satisfy R^T*R = I
        double a = 5.0, b = 12.0; // 5-12-13 triangle
        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(a, b);

        // Apply to identity-like rows to extract the transformation
        var M = new Matrix<double>(new double[,]
        {
            { 1, 0 },
            { 0, 1 }
        });
        MatrixHelper<double>.ApplyGivensRotation(M, c, s, 0, 1, 0, 2);

        // M now contains the rotation matrix. Check det = ±1
        double det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0];
        Assert.True(Math.Abs(Math.Abs(det) - 1.0) < Tol,
            $"Givens transformation determinant = {det}, expected ±1. " +
            $"Matrix = [[{M[0, 0]},{M[0, 1]}],[{M[1, 0]},{M[1, 1]}]]");
    }

    // ─── Householder Vector and Transformation ──────────────────────────

    [Fact]
    public void HouseholderVector_IsUnitNorm()
    {
        var x = new Vector<double>(new double[] { 3, 1, 4, 1, 5 });
        var v = MatrixHelper<double>.CreateHouseholderVector(x);

        double norm = 0;
        for (int i = 0; i < v.Length; i++) norm += v[i] * v[i];
        norm = Math.Sqrt(norm);

        Assert.Equal(1.0, norm, Tol);
    }

    [Fact]
    public void HouseholderVector_ReflectionZeroesSubdiagonal()
    {
        // After applying P = I - 2*v*v^T to x, result should be [±||x||, 0, 0, ...]
        var x = new Vector<double>(new double[] { 3, 4 });
        var v = MatrixHelper<double>.CreateHouseholderVector(x);

        // Compute Px = x - 2*v*(v^T*x)
        double vtx = 0;
        for (int i = 0; i < x.Length; i++) vtx += v[i] * x[i];

        var px = new double[x.Length];
        for (int i = 0; i < x.Length; i++) px[i] = x[i] - 2 * v[i] * vtx;

        // The second element should be zero
        Assert.True(Math.Abs(px[1]) < Tol,
            $"Householder reflection didn't zero subdiagonal: Px = [{px[0]}, {px[1]}]");

        // First element should be ±||x||
        double xNorm = Math.Sqrt(9 + 16); // 5
        Assert.Equal(xNorm, Math.Abs(px[0]), Tol);
    }

    [Fact]
    public void HouseholderVector_ReflectionPreservesNorm()
    {
        var x = new Vector<double>(new double[] { 2, 6, 3 });
        var v = MatrixHelper<double>.CreateHouseholderVector(x);

        // Compute Px
        double vtx = 0;
        for (int i = 0; i < x.Length; i++) vtx += v[i] * x[i];
        var px = new double[x.Length];
        for (int i = 0; i < x.Length; i++) px[i] = x[i] - 2 * v[i] * vtx;

        double xNorm = Math.Sqrt(4 + 36 + 9); // sqrt(49) = 7
        double pxNorm = Math.Sqrt(px[0] * px[0] + px[1] * px[1] + px[2] * px[2]);

        Assert.Equal(xNorm, pxNorm, Tol);
    }

    [Fact]
    public void HouseholderVector_NearZeroInput_ReturnsZeroVector()
    {
        var x = new Vector<double>(new double[] { 1e-16, 1e-17, 1e-18 });
        var v = MatrixHelper<double>.CreateHouseholderVector(x);

        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(0.0, v[i], 1e-12);
        }
    }

    // ─── Power Iteration ────────────────────────────────────────────────

    [Fact]
    public void PowerIteration_DiagonalMatrix_ReturnsDominantEigenvalue()
    {
        // Diagonal matrix: eigenvalues are 5, 2, 3
        var A = new Matrix<double>(new double[,]
        {
            { 5, 0, 0 },
            { 0, 2, 0 },
            { 0, 0, 3 }
        });

        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(A, 200, 1e-12);

        // Dominant eigenvalue should be 5
        // Need to compute actual eigenvalue from Rayleigh quotient: v^T * A * v
        var Av = A.Multiply(eigenvector);
        double rayleigh = eigenvector.DotProduct(Av);

        Assert.Equal(5.0, rayleigh, 0.1);
    }

    [Fact]
    public void PowerIteration_Symmetric2x2_ReturnsCorrectEigenvalue()
    {
        // [[4,1],[1,3]]: eigenvalues are (7±sqrt(5))/2 ≈ 4.618, 2.382
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1 },
            { 1, 3 }
        });

        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(A, 200, 1e-12);

        double expectedDominant = (7 + Math.Sqrt(5)) / 2; // ≈ 4.618

        // The returned eigenvalue should approximate the dominant eigenvalue
        // Compute actual Rayleigh quotient as the code's eigenvalue estimate might be wrong
        var Av = A.Multiply(eigenvector);
        double rayleigh = eigenvector.DotProduct(Av);

        Assert.Equal(expectedDominant, rayleigh, 0.05);
    }

    [Fact]
    public void PowerIteration_ScaledIdentity_ReturnsScaleFactor()
    {
        // 3*I has all eigenvalues = 3
        var A = new Matrix<double>(new double[,]
        {
            { 3, 0 },
            { 0, 3 }
        });

        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(A, 200, 1e-12);

        // Must compute Rayleigh quotient since eigenvalue field might be buggy
        var Av = A.Multiply(eigenvector);
        double rayleigh = eigenvector.DotProduct(Av);

        Assert.Equal(3.0, rayleigh, 0.1);
    }

    [Fact]
    public void PowerIteration_EigenvectorSatisfies_Av_Equals_LambdaV()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1 },
            { 1, 3 }
        });

        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(A, 200, 1e-12);

        // A*v should be parallel to v (i.e., A*v = lambda*v)
        var Av = A.Multiply(eigenvector);
        double lambda = eigenvector.DotProduct(Av); // Rayleigh quotient

        // Check: Av - lambda*v ≈ 0
        double residual = 0;
        for (int i = 0; i < eigenvector.Length; i++)
        {
            double diff = Av[i] - lambda * eigenvector[i];
            residual += diff * diff;
        }
        residual = Math.Sqrt(residual);

        Assert.True(residual < 0.01,
            $"Eigenvector residual ||Av - lambda*v|| = {residual}, expected near 0");
    }

    [Fact]
    public void PowerIteration_ReturnedEigenvalue_MatchesRayleighQuotient()
    {
        // This test specifically checks if the returned eigenvalue field is correct
        var A = new Matrix<double>(new double[,]
        {
            { 6, 2 },
            { 2, 3 }
        });
        // Eigenvalues: (9±sqrt(25))/2 = (9±5)/2 = 7, 2

        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(A, 200, 1e-12);

        // Compute Rayleigh quotient manually
        var Av = A.Multiply(eigenvector);
        double rayleigh = eigenvector.DotProduct(Av);

        // The returned eigenvalue should match the Rayleigh quotient
        Assert.Equal(rayleigh, eigenvalue, 0.1);
    }

    // ─── Gram-Schmidt Orthogonalization ─────────────────────────────────

    [Fact]
    public void GramSchmidt_ProducesOrthonormalColumns()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 1, 1, 0 },
            { 1, 0, 1 },
            { 0, 1, 1 }
        });

        var Q = MatrixHelper<double>.OrthogonalizeColumns(A);

        // Q^T * Q should be identity (3x3)
        var QTQ = Q.Transpose().Multiply(Q);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = i == j ? 1.0 : 0.0;
                Assert.True(Math.Abs(QTQ[i, j] - expected) < Tol,
                    $"Q^T*Q[{i},{j}] = {QTQ[i, j]}, expected {expected}");
            }
        }
    }

    [Fact]
    public void GramSchmidt_NearlyParallelVectors_StillOrthogonalizes()
    {
        // Two nearly parallel columns
        var A = new Matrix<double>(new double[,]
        {
            { 1.000, 1.001 },
            { 1.000, 1.000 },
            { 1.000, 0.999 }
        });

        var Q = MatrixHelper<double>.OrthogonalizeColumns(A);

        var col0 = Q.GetColumn(0);
        var col1 = Q.GetColumn(1);
        double dot = col0.DotProduct(col1);

        Assert.True(Math.Abs(dot) < 1e-4,
            $"Nearly-parallel columns not orthogonalized: dot = {dot}");
    }

    [Fact]
    public void GramSchmidt_AlreadyOrthonormal_PreservesColumns()
    {
        // Standard basis vectors are already orthonormal
        var I = Matrix<double>.CreateIdentity(3);
        var Q = MatrixHelper<double>.OrthogonalizeColumns(I);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = i == j ? 1.0 : 0.0;
                // Signs might flip, so check absolute value for off-diagonal
                if (i == j)
                    Assert.True(Math.Abs(Math.Abs(Q[i, j]) - 1.0) < Tol);
                else
                    Assert.True(Math.Abs(Q[i, j]) < Tol);
            }
        }
    }

    // ─── Tridiagonal Solver ─────────────────────────────────────────────

    [Fact]
    public void TridiagonalSolve_KnownSolution_VerifiesResidual()
    {
        // System: [2,-1,0; -1,2,-1; 0,-1,2] * x = [1; 0; 1]
        // Known solution: x = [1, 1, 1] (verify: [2-1, -1+2-1, -1+2] = [1, 0, 1] ✓)
        var lower = new Vector<double>(new double[] { 0, -1, -1 });
        var diag = new Vector<double>(new double[] { 2, 2, 2 });
        var upper = new Vector<double>(new double[] { -1, -1, 0 });
        var rhs = new Vector<double>(new double[] { 1, 0, 1 });
        var solution = new Vector<double>(3);

        MatrixHelper<double>.TridiagonalSolve(lower, diag, upper, solution, rhs);

        // Verify x = [1, 1, 1]
        Assert.Equal(1.0, solution[0], Tol);
        Assert.Equal(1.0, solution[1], Tol);
        Assert.Equal(1.0, solution[2], Tol);
    }

    [Fact]
    public void TridiagonalSolve_VerifyAxEqualsB()
    {
        // Arbitrary tridiagonal system
        var lower = new Vector<double>(new double[] { 0, 2, -1, 3 });
        var diag = new Vector<double>(new double[] { 4, 5, 6, 7 });
        var upper = new Vector<double>(new double[] { 1, -2, 3, 0 });
        var rhs = new Vector<double>(new double[] { 7, 4, 13, 10 });
        var solution = new Vector<double>(4);

        MatrixHelper<double>.TridiagonalSolve(lower, diag, upper, solution, rhs);

        // Reconstruct Ax and verify = b
        // Row 0: diag[0]*x[0] + upper[0]*x[1] = rhs[0]
        double r0 = diag[0] * solution[0] + upper[0] * solution[1];
        Assert.Equal(rhs[0], r0, Tol);

        // Row 1: lower[1]*x[0] + diag[1]*x[1] + upper[1]*x[2]
        double r1 = lower[1] * solution[0] + diag[1] * solution[1] + upper[1] * solution[2];
        Assert.Equal(rhs[1], r1, Tol);

        // Row 2: lower[2]*x[1] + diag[2]*x[2] + upper[2]*x[3]
        double r2 = lower[2] * solution[1] + diag[2] * solution[2] + upper[2] * solution[3];
        Assert.Equal(rhs[2], r2, Tol);

        // Row 3: lower[3]*x[2] + diag[3]*x[3]
        double r3 = lower[3] * solution[2] + diag[3] * solution[3];
        Assert.Equal(rhs[3], r3, Tol);
    }

    [Fact]
    public void TridiagonalSolve_DiagonalSystem_ReturnsDivision()
    {
        // Pure diagonal (no sub/super): Dx = b => x = b/d
        var lower = new Vector<double>(new double[] { 0, 0, 0 });
        var diag = new Vector<double>(new double[] { 2, 4, 5 });
        var upper = new Vector<double>(new double[] { 0, 0, 0 });
        var rhs = new Vector<double>(new double[] { 6, 12, 15 });
        var solution = new Vector<double>(3);

        MatrixHelper<double>.TridiagonalSolve(lower, diag, upper, solution, rhs);

        Assert.Equal(3.0, solution[0], Tol); // 6/2
        Assert.Equal(3.0, solution[1], Tol); // 12/4
        Assert.Equal(3.0, solution[2], Tol); // 15/5
    }

    // ─── Hat Matrix Properties ──────────────────────────────────────────

    [Fact]
    public void HatMatrix_Idempotent_HH_Equals_H()
    {
        var X = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 1, 2 },
            { 1, 3 },
            { 1, 4 },
            { 1, 5 }
        });

        var H = MatrixHelper<double>.CalculateHatMatrix(X);
        var HH = H.Multiply(H);

        for (int i = 0; i < H.Rows; i++)
            for (int j = 0; j < H.Columns; j++)
                Assert.True(Math.Abs(H[i, j] - HH[i, j]) < Tol,
                    $"H not idempotent at [{i},{j}]: H={H[i, j]}, H^2={HH[i, j]}");
    }

    [Fact]
    public void HatMatrix_TraceEqualsNumberOfPredictors()
    {
        // Use columns that are linearly independent: 1, x, x^2
        var X = new Matrix<double>(new double[,]
        {
            { 1, 1, 1 },
            { 1, 2, 4 },
            { 1, 3, 9 },
            { 1, 4, 16 },
            { 1, 5, 25 }
        });

        var H = MatrixHelper<double>.CalculateHatMatrix(X);

        double trace = 0;
        for (int i = 0; i < H.Rows; i++) trace += H[i, i];

        // tr(H) = rank(X) = 3 (columns 1, x, x^2 are independent)
        Assert.Equal(3.0, trace, Tol);
    }

    [Fact]
    public void HatMatrix_ResidualMakerAnnihilatesX()
    {
        // (I - H) * X = 0 for any design matrix X
        var X = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 1, 2 },
            { 1, 3 }
        });

        var H = MatrixHelper<double>.CalculateHatMatrix(X);
        int n = X.Rows;

        // Compute (I-H)*X
        var I = Matrix<double>.CreateIdentity(n);
        var IminusH = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                IminusH[i, j] = I[i, j] - H[i, j];

        var residual = IminusH.Multiply(X);

        for (int i = 0; i < residual.Rows; i++)
            for (int j = 0; j < residual.Columns; j++)
                Assert.True(Math.Abs(residual[i, j]) < Tol,
                    $"(I-H)*X not zero at [{i},{j}]: {residual[i, j]}");
    }

    [Fact]
    public void HatMatrix_LeverageValuesInRange()
    {
        // For proper design matrix, diagonal elements h_ii in [1/n, 1]
        var X = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 1, 2 },
            { 1, 3 },
            { 1, 4 }
        });

        var H = MatrixHelper<double>.CalculateHatMatrix(X);

        for (int i = 0; i < H.Rows; i++)
        {
            Assert.True(H[i, i] >= -Tol && H[i, i] <= 1.0 + Tol,
                $"Leverage h_{i}{i} = {H[i, i]} out of [0,1]");
        }
    }

    // ─── Hessenberg Reduction Properties ────────────────────────────────

    [Fact]
    public void HessenbergReduction_PreservesTrace()
    {
        // Similarity transformation preserves trace (sum of eigenvalues)
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1, 2, 3 },
            { 1, 3, 1, 2 },
            { 2, 1, 2, 1 },
            { 3, 2, 1, 1 }
        });

        double traceOriginal = 0;
        for (int i = 0; i < 4; i++) traceOriginal += A[i, i]; // 4+3+2+1 = 10

        var H = MatrixHelper<double>.ReduceToHessenbergFormat(A);

        double traceH = 0;
        for (int i = 0; i < 4; i++) traceH += H[i, i];

        Assert.Equal(traceOriginal, traceH, Tol);
    }

    [Fact]
    public void HessenbergReduction_PreservesDeterminant()
    {
        // Similarity transformation preserves determinant (product of eigenvalues)
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, 0 },
            { 1, 3, 1 },
            { 0, 1, 4 }
        });

        double detOriginal = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        var H = MatrixHelper<double>.ReduceToHessenbergFormat(A);
        double detH = MatrixHelper<double>.CalculateDeterminantRecursive(H);

        Assert.Equal(detOriginal, detH, 0.01);
    }

    [Fact]
    public void HessenbergReduction_ResultIsHessenberg()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 5, 1, 3, 2 },
            { 1, 4, 2, 1 },
            { 3, 2, 3, 1 },
            { 2, 1, 1, 6 }
        });

        var H = MatrixHelper<double>.ReduceToHessenbergFormat(A);

        Assert.True(MatrixHelper<double>.IsUpperHessenberg(H, Tol));
    }

    // ─── Outer Product Properties ───────────────────────────────────────

    [Fact]
    public void OuterProduct_RankOne_DeterminantZero()
    {
        // Rank-1 matrix has determinant 0 (for n >= 2)
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });

        var M = MatrixHelper<double>.OuterProduct(v1, v2);
        double det = MatrixHelper<double>.CalculateDeterminantRecursive(M);

        Assert.True(Math.Abs(det) < Tol,
            $"Rank-1 outer product should have det=0, got {det}");
    }

    [Fact]
    public void OuterProduct_Trace_EqualsInnerProduct()
    {
        // tr(v1 * v2^T) = v1 . v2 (when same length)
        var v1 = new Vector<double>(new double[] { 2, 3, 5 });
        var v2 = new Vector<double>(new double[] { 7, 11, 13 });

        var M = MatrixHelper<double>.OuterProduct(v1, v2);
        double trace = 0;
        for (int i = 0; i < 3; i++) trace += M[i, i];

        double innerProduct = v1.DotProduct(v2); // 14 + 33 + 65 = 112

        Assert.Equal(innerProduct, trace, Tol);
    }

    [Fact]
    public void OuterProduct_Symmetric_WhenSameVector()
    {
        // v * v^T is symmetric
        var v = new Vector<double>(new double[] { 2, 3, 5 });
        var M = MatrixHelper<double>.OuterProduct(v, v);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(M[i, j], M[j, i], Tol);
    }

    // ─── Hypotenuse Numerical Stability ─────────────────────────────────

    [Fact]
    public void Hypotenuse_LargeValues_DoesNotOverflow()
    {
        // Direct sqrt(a^2+b^2) would overflow for large values
        double a = 1e150;
        double b = 1e150;

        double result = MatrixHelper<double>.Hypotenuse(a, b);

        double expected = a * Math.Sqrt(2); // sqrt(2) * 1e150
        Assert.True(!double.IsInfinity(result), "Hypotenuse overflowed");
        Assert.Equal(expected, result, expected * 1e-10);
    }

    [Fact]
    public void Hypotenuse_SmallValues_DoesNotUnderflow()
    {
        double a = 1e-200;
        double b = 1e-200;

        double result = MatrixHelper<double>.Hypotenuse(a, b);

        double expected = a * Math.Sqrt(2);
        Assert.True(result > 0, "Hypotenuse underflowed to zero");
        Assert.Equal(expected, result, expected * 1e-6);
    }

    [Fact]
    public void Hypotenuse_MixedMagnitudes_AccurateResult()
    {
        // Very different magnitudes: sqrt(1e-300^2 + 1^2) ≈ 1.0
        double a = 1e-300;
        double b = 1.0;

        double result = MatrixHelper<double>.Hypotenuse(a, b);

        Assert.Equal(1.0, result, 1e-10);
    }

    [Fact]
    public void Hypotenuse_ArrayVersion_MatchesPythagorean()
    {
        // sqrt(3^2 + 4^2 + 12^2) = sqrt(9+16+144) = sqrt(169) = 13
        var values = new double[] { 3.0, 4.0, 12.0 };
        double result = MatrixHelper<double>.Hypotenuse(values);

        Assert.Equal(13.0, result, Tol);
    }

    // ─── Spectral Norm ──────────────────────────────────────────────────

    [Fact]
    public void SpectralNorm_DiagonalMatrix_ReturnsMaxAbsDiagonal()
    {
        var A = new Matrix<double>(new double[,]
        {
            { -3, 0 },
            {  0, 2 }
        });

        double norm = MatrixHelper<double>.SpectralNorm(A);

        // Spectral norm of diagonal = max |diagonal element| = 3
        Assert.True(Math.Abs(norm - 3.0) < 0.5,
            $"Spectral norm of diag(-3,2) = {norm}, expected ~3.0");
    }

    [Fact]
    public void SpectralNorm_OrthogonalMatrix_ReturnsOne()
    {
        double theta = Math.PI / 3;
        var R = new Matrix<double>(new double[,]
        {
            { Math.Cos(theta), -Math.Sin(theta) },
            { Math.Sin(theta),  Math.Cos(theta) }
        });

        double norm = MatrixHelper<double>.SpectralNorm(R);

        Assert.True(Math.Abs(norm - 1.0) < 0.2,
            $"Spectral norm of rotation = {norm}, expected ~1.0");
    }

    // ─── Cross-Property Identity Tests ──────────────────────────────────

    [Fact]
    public void Identity_DetInverse_IsReciprocal()
    {
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1 },
            { 1, 3 }
        });

        double detA = MatrixHelper<double>.CalculateDeterminantRecursive(A);
        var Ainv = A.Inverse();
        double detAinv = MatrixHelper<double>.CalculateDeterminantRecursive(Ainv);

        // det(A^-1) = 1/det(A)
        Assert.Equal(1.0 / detA, detAinv, Tol);
    }

    [Fact]
    public void Identity_HatMatrix_Symmetric()
    {
        var X = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 1, 4 },
            { 1, 6 },
            { 1, 8 }
        });

        var H = MatrixHelper<double>.CalculateHatMatrix(X);

        for (int i = 0; i < H.Rows; i++)
            for (int j = i + 1; j < H.Columns; j++)
                Assert.Equal(H[i, j], H[j, i], Tol);
    }

    [Fact]
    public void Identity_GramSchmidt_SpanPreserved()
    {
        // After orthogonalization, Q*Q^T*A should equal A
        // (since Q spans the same column space as A)
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });

        var Q = MatrixHelper<double>.OrthogonalizeColumns(A);

        // Q*Q^T is the projection onto the column space of A
        var P = Q.Multiply(Q.Transpose());

        // P*A should equal A (since A is in its own column space)
        var PA = P.Multiply(A);
        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Columns; j++)
                Assert.True(Math.Abs(PA[i, j] - A[i, j]) < Tol,
                    $"Column space not preserved at [{i},{j}]: PA={PA[i, j]}, A={A[i, j]}");
    }

    [Fact]
    public void BandDiagonalMultiply_TridiagonalCase_MatchesHandComputed()
    {
        // Full tridiagonal matrix: A = [2, -1, 0; -1, 2, -1; 0, -1, 2]
        // BandDiagonalMultiply uses banded storage: row i stores band elements.
        // For leftSide=1, rightSide=1 (bandwidth=3):
        //   Row 0 (k=-1): j=1 maps to col 0, j=2 maps to col 1 => [*, A00, A01] = [0, 2, -1]
        //   Row 1 (k= 0): j=0..2 maps to col 0..2 => [A10, A11, A12] = [-1, 2, -1]
        //   Row 2 (k= 1): j=0 maps to col 1, j=1 maps to col 2 => [A21, A22, *] = [-1, 2, 0]
        var bandMatrix = new Matrix<double>(new double[,]
        {
            { 0, 2, -1 },
            { -1, 2, -1 },
            { -1, 2, 0 }
        });

        var x = new Vector<double>(new double[] { 1, 2, 3 });
        var result = new Vector<double>(3);

        MatrixHelper<double>.BandDiagonalMultiply(1, 1, bandMatrix, result, x);

        // Expected: A*x = [2*1-1*2, -1*1+2*2-1*3, -1*2+2*3] = [0, 0, 4]
        Assert.Equal(0.0, result[0], Tol);
        Assert.Equal(0.0, result[1], Tol);
        Assert.Equal(4.0, result[2], Tol);
    }

    [Fact]
    public void ExtractDiagonal_RecoversDiagonalMatrix()
    {
        var D = new Matrix<double>(new double[,]
        {
            { 7, 0, 0 },
            { 0, 11, 0 },
            { 0, 0, 13 }
        });

        var diag = MatrixHelper<double>.ExtractDiagonal(D);

        Assert.Equal(7.0, diag[0], Tol);
        Assert.Equal(11.0, diag[1], Tol);
        Assert.Equal(13.0, diag[2], Tol);
    }
}
