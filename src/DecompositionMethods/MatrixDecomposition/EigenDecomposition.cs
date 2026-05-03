using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs eigenvalue decomposition of a matrix, breaking it down into its eigenvalues and eigenvectors.
/// </summary>
/// <typeparam name="T">The numeric data type used in calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Eigenvalue decomposition is a way to factorize a square matrix into a set of eigenvectors and eigenvalues.
/// This is useful in many applications including principal component analysis, vibration analysis,
/// and solving systems of differential equations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Eigenvalue decomposition finds special directions (eigenvectors) in which a matrix
/// acts like simple scaling. When you multiply the matrix by an eigenvector, you get the same vector back
/// but scaled by a number (the eigenvalue). Think of it like finding the "natural directions" of a transformation.
/// </para>
/// <para>
/// Real-world applications:
/// - Principal Component Analysis (PCA) for data dimensionality reduction
/// - Vibration analysis in mechanical engineering
/// - Google's PageRank algorithm
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class EigenDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the eigenvectors of the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// Eigenvectors are special vectors that, when multiplied by the original matrix,
    /// result in a vector that points in the same direction but may be scaled.
    /// Each eigenvector corresponds to an eigenvalue in the EigenValues property.
    /// </remarks>
    public Matrix<T> EigenVectors { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the eigenvalues of the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// Eigenvalues are special scalars that, when the original matrix multiplies its corresponding eigenvector,
    /// the result is the same as scaling the eigenvector by this value.
    /// Each eigenvalue corresponds to an eigenvector in the EigenVectors property.
    /// </remarks>
    public Vector<T> EigenValues { get; private set; } = new Vector<T>(0);

    private readonly EigenAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new eigenvalue decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for eigenvalue decomposition. Defaults to QR algorithm.</param>
    /// <remarks>
    /// Different algorithms have different trade-offs in terms of speed, accuracy, and applicability:
    /// - QR: Generally robust and accurate for most matrices.
    /// - PowerIteration: Simpler but may be slower for finding all eigenvalues.
    /// - Jacobi: Works well for symmetric matrices.
    /// </remarks>
    public EigenDecomposition(Matrix<T> matrix, EigenAlgorithmType algorithm = EigenAlgorithmType.QR)
        : base(matrix)
    {
        ValidateMatrix(matrix, requireSquare: true);
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the eigenvalue decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (EigenValues, EigenVectors) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate eigenvalue decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for eigenvalue decomposition.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors of the matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeDecomposition(Matrix<T> matrix, EigenAlgorithmType algorithm)
    {
        return algorithm switch
        {
            EigenAlgorithmType.QR => ComputeEigenQR(matrix),
            EigenAlgorithmType.PowerIteration => ComputeEigenPowerIteration(matrix),
            EigenAlgorithmType.Jacobi => ComputeEigenJacobi(matrix),
            _ => throw new ArgumentException("Unsupported eigenvalue decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the Power Iteration method.
    /// </summary>
    /// <remarks>
    /// The Power Iteration method works by repeatedly multiplying a vector by the matrix
    /// and normalizing the result. This process converges to the dominant eigenvector
    /// (the one with the largest eigenvalue in absolute terms).
    /// 
    /// This implementation uses deflation to find multiple eigenvalues and eigenvectors.
    /// After finding each eigenvector, we modify the matrix to remove the influence of
    /// that eigenvector, allowing us to find the next most dominant one.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenPowerIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> eigenValues = new(n);
        Matrix<T> eigenVectors = Matrix<T>.CreateIdentity(n);

        Matrix<T> deflated = matrix.Clone();
        int maxIterations = 2000;
        T tolerance = NumOps.FromDouble(1e-14);

        for (int i = 0; i < n; i++)
        {
            // Start with a vector that has all ones - more stable than random
            Vector<T> v = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                v[j] = NumOps.One;
            }
            // Normalize the initial vector
            T initNorm = v.Norm();
            if (NumOps.GreaterThan(initNorm, NumOps.FromDouble(1e-14)))
            {
                v = v.Divide(initNorm);
            }

            T eigenValue = NumOps.Zero;
            bool converged = false;

            // Phase 1: Standard power iteration
            for (int iter = 0; iter < maxIterations && !converged; iter++)
            {
                // Orthogonalize against previously found eigenvectors
                for (int j = 0; j < i; j++)
                {
                    Vector<T> prevV = eigenVectors.GetColumn(j);
                    T projection = v.DotProduct(prevV);
                    v = v.Subtract(prevV.Multiply(projection));
                }
                T vNorm = v.Norm();
                if (NumOps.LessThan(vNorm, NumOps.FromDouble(1e-14)))
                {
                    // Vector became zero, reinitialize
                    for (int j = 0; j < n; j++) v[j] = NumOps.FromDouble(j + 1);
                    v = v.Divide(v.Norm());
                    continue;
                }
                v = v.Divide(vNorm);

                // Power iteration step: v = A*v / ||A*v||
                Vector<T> w = deflated.Multiply(v);
                T norm = w.Norm();

                // Check for zero norm (indicates zero eigenvalue or numerical issues)
                if (NumOps.LessThan(norm, NumOps.FromDouble(1e-14)))
                {
                    eigenValues[i] = NumOps.Zero;
                    converged = true;
                    break;
                }

                Vector<T> vNew = w.Divide(norm);

                // Compute Rayleigh quotient: λ = v^T * A * v
                Vector<T> Av = deflated.Multiply(vNew);
                T newEigenValue = vNew.DotProduct(Av);

                // Check convergence: ||Av - λv|| should be small
                Vector<T> residual = Av.Subtract(vNew.Multiply(newEigenValue));
                T residualNorm = residual.Norm();

                if (NumOps.LessThan(residualNorm, tolerance))
                {
                    v = vNew;
                    eigenValue = newEigenValue;
                    eigenValues[i] = eigenValue;
                    converged = true;
                    break;
                }

                v = vNew;
                eigenValue = newEigenValue;
                eigenValues[i] = eigenValue;
            }

            // Phase 2: Rayleigh quotient iteration for refinement (if not converged)
            if (!converged)
            {
                for (int iter = 0; iter < 200; iter++)
                {
                    Vector<T> Av = deflated.Multiply(v);
                    eigenValue = v.DotProduct(Av);

                    // Solve (A - λI)w = v using shifted matrix
                    // For better convergence, we use the residual direction
                    Vector<T> residual = Av.Subtract(v.Multiply(eigenValue));
                    T residualNorm = residual.Norm();

                    if (NumOps.LessThan(residualNorm, tolerance))
                    {
                        eigenValues[i] = eigenValue;
                        break;
                    }

                    // Update v in the direction of the residual (gradient descent on Rayleigh quotient)
                    // v_new = (A*v) / ||A*v||
                    T avNorm = Av.Norm();
                    if (NumOps.GreaterThan(avNorm, NumOps.FromDouble(1e-14)))
                    {
                        v = Av.Divide(avNorm);
                    }
                    eigenValues[i] = eigenValue;
                }
            }

            eigenVectors.SetColumn(i, v);

            // Deflate: remove the found eigenvalue/eigenvector from the matrix
            // A_new = A - λ * v * v^T
            deflated = deflated.Subtract(MatrixHelper<T>.OuterProduct(v, v).Multiply(eigenValues[i]));
        }

        return (eigenValues, eigenVectors);
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the QR algorithm with Wilkinson shift.
    /// </summary>
    /// <remarks>
    /// The QR algorithm is an iterative method that works by repeatedly factoring the matrix
    /// into a product Q*R (where Q is orthogonal and R is upper triangular), and then
    /// recombining as R*Q. This process eventually converges to a matrix where the eigenvalues
    /// appear on the diagonal.
    ///
    /// This implementation uses Wilkinson shift for faster convergence on symmetric matrices.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> A = matrix.Clone();
        Matrix<T> Q = Matrix<T>.CreateIdentity(n);

        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 200;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Check if already diagonal (for symmetric) or upper triangular
            bool converged = true;
            for (int i = 1; i < n && converged; i++)
            {
                for (int j = 0; j < i && converged; j++)
                {
                    if (NumOps.GreaterThan(NumOps.Abs(A[i, j]), tolerance))
                    {
                        converged = false;
                    }
                }
            }
            if (converged) break;

            // Wilkinson shift: compute shift from bottom 2x2 submatrix
            T shift = NumOps.Zero;
            if (n >= 2)
            {
                T a = A[n - 2, n - 2];
                T b = A[n - 2, n - 1];
                T c = A[n - 1, n - 2];
                T d = A[n - 1, n - 1];

                // Compute eigenvalue of 2x2 block closest to d
                T delta = NumOps.Divide(NumOps.Subtract(a, d), NumOps.FromDouble(2.0));
                T signDelta = NumOps.GreaterThanOrEquals(delta, NumOps.Zero) ? NumOps.One : NumOps.FromDouble(-1.0);
                T bc = NumOps.Multiply(b, c);
                T sqrtTerm = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(delta, delta), bc));

                if (NumOps.GreaterThan(NumOps.Abs(sqrtTerm), NumOps.FromDouble(1e-14)))
                {
                    shift = NumOps.Subtract(d, NumOps.Divide(NumOps.Multiply(signDelta, bc),
                        NumOps.Add(NumOps.Abs(delta), sqrtTerm)));
                }
                else
                {
                    shift = d;
                }
            }

            // Shifted QR step: A - shift*I = Q*R, then A = R*Q + shift*I
            Matrix<T> shiftedA = A.Clone();
            for (int i = 0; i < n; i++)
            {
                shiftedA[i, i] = NumOps.Subtract(shiftedA[i, i], shift);
            }

            var qrDecomp = new QrDecomposition<T>(shiftedA);
            (var q, var r) = (qrDecomp.Q, qrDecomp.R);

            A = r.Multiply(q);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = NumOps.Add(A[i, i], shift);
            }

            Q = Q.Multiply(q);
        }

        Vector<T> eigenValues = MatrixHelper<T>.ExtractDiagonal(A);
        return (eigenValues, Q);
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the Jacobi method.
    /// </summary>
    /// <remarks>
    /// The Jacobi method is an iterative algorithm particularly effective for symmetric matrices.
    /// It works by performing a series of rotations (Jacobi rotations) that gradually make the
    /// off-diagonal elements of the matrix smaller, eventually converging to a diagonal matrix
    /// where the diagonal elements are the eigenvalues.
    /// 
    /// In simple terms, this method systematically eliminates the largest off-diagonal element
    /// in each iteration until all off-diagonal elements are close to zero.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors of the matrix.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenJacobi(Matrix<T> matrix)
    {
        // Cyclic-Jacobi eigensolver for symmetric matrices, transcribed from
        // the canonical algorithm in Numerical Recipes §11.1 (Jacobi
        // transformations) and matching LAPACK's DSYEVJ. Replaces the prior
        // single-pivot O(n^4) implementation that:
        //   1. Hard-coded the iteration count at 100, leaving n>~20 grossly
        //      under-converged (issue #1230 — at n=256 only 0.06% of
        //      off-diagonal pairs were ever touched).
        //   2. Materialised a full identity rotation matrix per rotation and
        //      ran J^T · A · J via two general matrix multiplies, costing
        //      O(n^3) per rotation × O(n^2) rotations = O(n^5) total. At
        //      n=256 that's ~10 minutes per call.
        //
        // Sweep-based cyclic Jacobi:
        //   - Each sweep walks every (p, q) pair with p<q in lexicographic
        //     order applying an in-place 2-axis rotation. Each rotation
        //     touches only the p-th and q-th rows/columns of A and the p-th
        //     and q-th columns of V — O(n) per rotation.
        //   - n*(n-1)/2 rotations per sweep ⇒ O(n^3) per sweep.
        //   - Convergence is quadratic; ~6–10 sweeps suffice for double-
        //     precision residuals (per Numerical Recipes §11.1), so total
        //     work is O(n^3) — the standard bound for symmetric eigensolvers.
        //
        // First-three-sweeps threshold trick (LAPACK pattern): only rotate
        // pairs whose magnitude exceeds 0.2 * sum_off / n^2. This skips
        // already-small entries that would otherwise burn rotations on noise.
        // After sweep 4 the threshold drops to 0 (rotate every pair); after
        // sweep 4 we also forcibly zero pairs whose magnitude is below
        // 100 * eps * (|A_pp| + |A_qq|) instead of rotating, mirroring
        // LAPACK's "tiny rotation" guard against numerical drift.

        int n = matrix.Rows;

        // Symmetry precondition: cyclic Jacobi assumes a symmetric input
        // (the in-place rotation mirrors A[p,q] into A[q,p] without
        // separately validating they were equal to start with). For a
        // non-symmetric matrix the routine silently solves a different
        // problem. Surface this as ArgumentException at the API
        // boundary so callers get an actionable failure instead of
        // wrong eigenpairs.
        //
        // Tolerance is scaled by ‖A‖_∞ (max-abs entry) so the check
        // works regardless of the matrix's overall magnitude — a fixed
        // 1e-10 absolute threshold would falsely throw on matrices
        // whose entries are larger than ~1e10 even when symmetric to
        // floating-point precision, and would let through asymmetry on
        // matrices with entries near machine epsilon. Floor at 1e-12
        // so all-zero matrices don't divide by zero.
        T maxAbs = NumOps.Zero;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                T abs = NumOps.Abs(matrix[i, j]);
                if (NumOps.GreaterThan(abs, maxAbs)) maxAbs = abs;
            }
        }
        T relScale = NumOps.GreaterThan(maxAbs, NumOps.FromDouble(1.0))
            ? maxAbs
            : NumOps.One;
        T symmetryTol = NumOps.Multiply(NumOps.FromDouble(1e-10), relScale);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = i + 1; j < matrix.Columns; j++)
            {
                T diff = NumOps.Abs(NumOps.Subtract(matrix[i, j], matrix[j, i]));
                if (NumOps.GreaterThan(diff, symmetryTol))
                {
                    throw new ArgumentException(
                        "Jacobi eigen decomposition requires a symmetric matrix; " +
                        $"|A[{i},{j}] - A[{j},{i}]| = {diff} exceeds tolerance {symmetryTol} " +
                        $"(scaled by ‖A‖_∞ = {maxAbs}).",
                        nameof(matrix));
                }
            }
        }

        Matrix<T> A = matrix.Clone();
        Matrix<T> V = Matrix<T>.CreateIdentity(n);
        if (n <= 1)
        {
            return (MatrixHelper<T>.ExtractDiagonal(A), V);
        }

        T zero = NumOps.Zero;
        T one = NumOps.One;
        T half = NumOps.FromDouble(0.5);
        T tiny = NumOps.FromDouble(1e-30);
        T machineEps = NumOps.FromDouble(1e-15);

        // Convergence: terminate once ‖off-diag(A)‖_F^2 ≤ (tol·‖A‖_F)^2.
        T frobNormSq = SumSquaredEntries(A);
        T frobTolSq = NumOps.Multiply(frobNormSq, NumOps.FromDouble(1e-24)); // (1e-12)^2
        T frobFloorSq = NumOps.FromDouble(1e-30);
        T convergenceThresholdSq = NumOps.GreaterThan(frobTolSq, frobFloorSq) ? frobTolSq : frobFloorSq;

        // Sweep cap: 50 is the standard engineering ceiling — well-behaved
        // matrices converge in 6–10 sweeps and pathological cases that
        // wouldn't converge in 50 sweeps wouldn't converge at all under
        // double precision. (LAPACK DSYEVJ caps at 30; we use 50 for safety
        // headroom on extreme inputs.)
        const int maxSweeps = 50;
        bool converged = false;

        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            // Off-diagonal mass for the threshold trick AND the convergence
            // check. Track BOTH the linear (sum of |apq| over p<q) and
            // squared (sum of apq² over all i≠j) forms — the linear sum
            // drives the LAPACK-style early-sweep threshold heuristic
            // (compare per-element |apq| against threshold = 0.2 * sumOff /
            // n²), the squared form drives the Frobenius-norm convergence
            // check. The previous code mixed the two by scaling
            // offDiagSumSq linearly which gave a threshold whose
            // dimensionality didn't match either |apq| or |apq|² — the
            // wrong set of pairs got rotated.
            T offDiagSumSq = SumSquaredOffDiagonal(A);
            if (NumOps.LessThan(offDiagSumSq, convergenceThresholdSq))
            {
                converged = true;
                break;
            }
            T offDiagAbsSum = SumAbsUpperOffDiagonal(A);

            // LAPACK DSYEVJ-style early-sweep threshold: rotate only pairs
            // whose magnitude exceeds 0.2 * sumOff / n² during the first
            // three sweeps (saves rotations on entries that haven't built
            // up enough relative magnitude yet). Threshold and comparand
            // are both LINEAR (|apq| vs threshold) — consistent units.
            // After sweep 3 the threshold drops to zero (rotate every
            // pair above the absolute tiny floor).
            T threshold;
            if (sweep < 3)
            {
                T scale = NumOps.FromDouble(0.2 / ((double)n * n));
                threshold = NumOps.Multiply(scale, offDiagAbsSum);
            }
            else
            {
                threshold = zero;
            }

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    T apq = A[p, q];
                    T absApq = NumOps.Abs(apq);

                    // After sweep 4, zero out entries that are negligible
                    // relative to the diagonals — saves a rotation that
                    // would only inject noise.
                    if (sweep > 3)
                    {
                        T diagScale = NumOps.Add(NumOps.Abs(A[p, p]), NumOps.Abs(A[q, q]));
                        T tinyAllowed = NumOps.Multiply(NumOps.FromDouble(100), NumOps.Multiply(machineEps, diagScale));
                        if (NumOps.LessThan(absApq, tinyAllowed))
                        {
                            A[p, q] = zero;
                            A[q, p] = zero;
                            continue;
                        }
                    }

                    // Skip below-threshold entries during the early sweeps.
                    // Linear |apq| against linear threshold — units match.
                    if (sweep < 3 && NumOps.LessThan(absApq, threshold))
                    {
                        continue;
                    }
                    if (NumOps.LessThan(absApq, tiny))
                    {
                        continue;
                    }

                    // Compute rotation parameters c=cos(θ), s=sin(θ) such that
                    // the resulting 2×2 block is diagonal:
                    //   [c -s][app apq][c  s]   [app' 0  ]
                    //   [s  c][apq aqq][-s c] = [0   aqq']
                    // Using the stable form from NR §11.1 to avoid
                    // catastrophic cancellation in tan(θ).
                    T app = A[p, p];
                    T aqq = A[q, q];
                    T diff = NumOps.Subtract(aqq, app);
                    T t;
                    if (NumOps.LessThan(NumOps.Abs(diff), NumOps.Multiply(machineEps, NumOps.Abs(apq))))
                    {
                        // diff ≈ 0: 45-degree rotation.
                        t = NumOps.LessThan(apq, zero) ? NumOps.Negate(one) : one;
                    }
                    else
                    {
                        T theta = NumOps.Multiply(half, NumOps.Divide(diff, apq));
                        T thetaSq = NumOps.Multiply(theta, theta);
                        T denom = NumOps.Add(NumOps.Abs(theta), NumOps.Sqrt(NumOps.Add(one, thetaSq)));
                        t = NumOps.Divide(one, denom);
                        if (NumOps.LessThan(theta, zero)) t = NumOps.Negate(t);
                    }
                    T c = NumOps.Divide(one, NumOps.Sqrt(NumOps.Add(one, NumOps.Multiply(t, t))));
                    T s = NumOps.Multiply(t, c);

                    // In-place 2-axis rotation update.
                    //
                    // Diagonals: app' = app - t·apq, aqq' = aqq + t·apq.
                    A[p, p] = NumOps.Subtract(app, NumOps.Multiply(t, apq));
                    A[q, q] = NumOps.Add(aqq, NumOps.Multiply(t, apq));
                    A[p, q] = zero;
                    A[q, p] = zero;

                    // For each i ≠ p,q rotate the (i,p) and (i,q) entries.
                    // Symmetric form keeps both upper- and lower-triangle
                    // entries in lockstep so the matrix stays symmetric and
                    // future SumSquaredOffDiagonal reads see the right values.
                    for (int i = 0; i < n; i++)
                    {
                        if (i == p || i == q) continue;
                        T aip = A[i, p];
                        T aiq = A[i, q];
                        A[i, p] = NumOps.Subtract(NumOps.Multiply(c, aip), NumOps.Multiply(s, aiq));
                        A[p, i] = A[i, p];
                        A[i, q] = NumOps.Add(NumOps.Multiply(s, aip), NumOps.Multiply(c, aiq));
                        A[q, i] = A[i, q];
                    }

                    // Accumulate rotation into eigenvector matrix:
                    //   V[:, p] = c*V[:, p] - s*V[:, q]
                    //   V[:, q] = s*V[:, p] + c*V[:, q]
                    for (int i = 0; i < n; i++)
                    {
                        T vip = V[i, p];
                        T viq = V[i, q];
                        V[i, p] = NumOps.Subtract(NumOps.Multiply(c, vip), NumOps.Multiply(s, viq));
                        V[i, q] = NumOps.Add(NumOps.Multiply(s, vip), NumOps.Multiply(c, viq));
                    }
                }
            }
        }

        // Convergence guarantee: if the loop exhausts maxSweeps without
        // hitting the convergence threshold, surface that as
        // InvalidOperationException so the caller doesn't silently consume
        // potentially-wrong eigenpairs. Matches LAPACK's
        // 'INFO > 0 means did not converge' contract.
        if (!converged)
        {
            T finalOffDiag = SumSquaredOffDiagonal(A);
            if (NumOps.GreaterThanOrEquals(finalOffDiag, convergenceThresholdSq))
            {
                throw new InvalidOperationException(
                    $"Jacobi eigen decomposition did not converge within {maxSweeps} sweeps. " +
                    $"Final off-diagonal mass {finalOffDiag} >= convergence threshold " +
                    $"{convergenceThresholdSq}. The input matrix may be too ill-conditioned " +
                    "for the cyclic Jacobi algorithm; consider PowerIteration with deflation " +
                    "or a Schur-based eigensolver.");
            }
        }

        Vector<T> eigenValues = MatrixHelper<T>.ExtractDiagonal(A);
        return (eigenValues, V);
    }

    /// <summary>
    /// Sum of squared entries — used for the Frobenius-norm-based convergence
    /// threshold in the sweep-based Jacobi loop.
    /// </summary>
    private T SumSquaredEntries(Matrix<T> m)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                sum = NumOps.Add(sum, NumOps.Multiply(m[i, j], m[i, j]));
        return sum;
    }

    /// <summary>
    /// Sum of squared off-diagonal entries (i ≠ j). Used for the cyclic-
    /// Jacobi convergence check: when this drops below tol² · ‖A‖_F², the
    /// matrix is effectively diagonal and the eigenvalues are A's diagonal.
    /// </summary>
    private T SumSquaredOffDiagonal(Matrix<T> m)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                if (i == j) continue;
                sum = NumOps.Add(sum, NumOps.Multiply(m[i, j], m[i, j]));
            }
        }
        return sum;
    }

    /// <summary>
    /// Sum of <c>|a[i,j]|</c> over the strict upper triangle (i &lt; j).
    /// Drives the LAPACK-style early-sweep threshold in cyclic Jacobi:
    /// rotate only pairs whose magnitude exceeds <c>0.2 * sumAbs / n²</c>.
    /// Linear (not squared) so the threshold and per-element comparand
    /// have matching units.
    /// </summary>
    private T SumAbsUpperOffDiagonal(Matrix<T> m)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < m.Rows - 1; i++)
        {
            for (int j = i + 1; j < m.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Abs(m[i, j]));
            }
        }
        return sum;
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using the eigenvalue decomposition.
    /// </summary>
    /// <remarks>
    /// This method uses the eigenvalue decomposition to solve the system of equations.
    /// It works by transforming the problem into the eigenvector basis, where the system
    /// becomes diagonal and easy to solve, then transforming back to the original basis.
    ///
    /// The solution is computed as: x = V * D⁻¹ * V^T * b
    /// where V is the matrix of eigenvectors, D is a diagonal matrix of eigenvalues,
    /// and V^T is the transpose of V.
    /// </remarks>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    public override Vector<T> Solve(Vector<T> b)
    {
        Matrix<T> D = Matrix<T>.CreateDiagonal(EigenValues);
        return EigenVectors.Multiply(D.InvertDiagonalMatrix()).Multiply(EigenVectors.Transpose()).Multiply(b);
    }

    /// <summary>
    /// Computes the inverse of the original matrix using the eigenvalue decomposition.
    /// </summary>
    /// <remarks>
    /// This method uses the eigenvalue decomposition to compute the inverse of the matrix.
    /// The inverse is calculated as: A⁻¹ = V * D⁻¹ * V^T
    /// where V is the matrix of eigenvectors, D is a diagonal matrix of eigenvalues,
    /// and V^T is the transpose of V.
    ///
    /// This approach can be more numerically stable than directly inverting the matrix,
    /// especially for matrices that are nearly singular (close to having no inverse).
    /// </remarks>
    /// <returns>The inverse of the original matrix.</returns>
    public override Matrix<T> Invert()
    {
        Matrix<T> D = Matrix<T>.CreateDiagonal(EigenValues);
        return EigenVectors.Multiply(D.InvertDiagonalMatrix()).Multiply(EigenVectors.Transpose());
    }
}
