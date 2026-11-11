using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Singular Value Decomposition (SVD) for matrices.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix.</typeparam>
/// <remarks>
/// <para>
/// Singular Value Decomposition (SVD) factors a matrix A into the product A = U * S * V^T, where U and V
/// are orthogonal matrices and S is a diagonal matrix containing singular values. SVD is one of the most
/// important matrix decompositions with applications across many fields of science and engineering.
/// </para>
/// <para>
/// <b>For Beginners:</b> SVD is a way to break down a matrix into three simpler matrices (U, S, and V^T).
/// Think of it like factoring a number, but for matrices. This decomposition is useful for data
/// compression, noise reduction, and solving systems of equations.
/// </para>
/// <para>
/// Real-world applications:
/// - Data compression and dimensionality reduction (PCA)
/// - Image compression and denoising
/// - Recommender systems (collaborative filtering)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class SvdDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the left singular vectors matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The U matrix represents the output space of the transformation.
    /// It contains information about patterns in the rows of your original data.
    /// </para>
    /// </remarks>
    public Matrix<T> U { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the singular values vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The S vector contains the "strength" of each pattern found in the data.
    /// Larger values indicate more important patterns. These values are always positive
    /// and are typically arranged in descending order.
    /// </para>
    /// </remarks>
    public Vector<T> S { get; private set; } = new Vector<T>(0);

    /// <summary>
    /// Gets the transposed right singular vectors matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The V^T matrix represents the input space of the transformation.
    /// It contains information about patterns in the columns of your original data.
    /// </para>
    /// </remarks>
    public Matrix<T> Vt { get; private set; } = new Matrix<T>(0, 0);

    private readonly SvdAlgorithmType _algorithm = SvdAlgorithmType.GolubReinsch;
    private readonly double? _relativeTolerance;

    /// <summary>
    /// Gets the relative cutoff below which a singular value is treated as zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A singular value counts as zero when it is at or below <c>RelativeTolerance * S[0]</c>,
    /// the largest singular value. The default is <c>max(rows, columns) * 2.22e-16</c>, matching
    /// the convention LAPACK's <c>gelsd</c> and SciPy's <c>lstsq</c> use for <c>rcond</c>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A rank-deficient matrix - one whose columns repeat information - has
    /// singular values that are mathematically zero. Computed in floating point they come out as
    /// something like 1e-32 instead, and dividing by that produces an answer around 1e+32. This
    /// cutoff is what stops that: anything small enough to be rounding noise is treated as the
    /// zero it was meant to be, which yields the minimum-norm solution instead of a meaningless
    /// enormous one.
    /// </para>
    /// </remarks>
    public double RelativeTolerance { get; private set; }

    /// <summary>
    /// Gets the number of singular values above <see cref="RelativeTolerance"/>: the numerical
    /// rank of the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many genuinely independent directions the matrix contains. A rank
    /// below the column count means some columns are combinations of others, so a least-squares
    /// fit has infinitely many equally good answers rather than one.
    /// </para>
    /// </remarks>
    public int Rank { get; private set; }

    /// <summary>
    /// Initializes a new instance of the SVD decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="svdAlgorithm">The algorithm to use for SVD computation.</param>
    /// <param name="relativeTolerance">
    /// The cutoff below which a singular value counts as zero, as a multiple of the largest
    /// singular value. When null, <c>max(rows, columns) * 2.22e-16</c> is used, which is the
    /// LAPACK and SciPy default. Pass 0 to invert every non-zero singular value; that reproduces
    /// an exact-arithmetic pseudoinverse and is almost never what a floating-point computation
    /// wants.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="relativeTolerance"/> is negative.
    /// </exception>
    public SvdDecomposition(
        Matrix<T> matrix,
        SvdAlgorithmType svdAlgorithm = SvdAlgorithmType.GolubReinsch,
        double? relativeTolerance = null)
        : base(matrix)
    {
        if (relativeTolerance.HasValue && relativeTolerance.Value < 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(relativeTolerance),
                relativeTolerance,
                "The relative tolerance must be zero or positive.");
        }

        _algorithm = svdAlgorithm;
        _relativeTolerance = relativeTolerance;

        Decompose();
    }

    /// <summary>
    /// Performs the SVD decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (U, S, Vt) = ComputeDecomposition(A, _algorithm);

        // The LAPACK and SciPy default: relative to the largest singular value, and scaled by the
        // size of the matrix because each dimension contributes rounding to the smallest one.
        RelativeTolerance = _relativeTolerance
            ?? Math.Max(A.Rows, A.Columns) * 2.220446049250313e-16;

        var cutoff = SingularValueCutoff();

        Rank = S.Length == 0
            ? 0
            : Engine.TensorCountNonzero(Engine.TensorGreaterThan(Tensor<T>.FromVector(S), cutoff));
    }

    /// <summary>
    /// The absolute value at or below which a singular value counts as zero.
    /// </summary>
    private T SingularValueCutoff()
    {
        if (S.Length == 0) return NumOps.Zero;

        T largest = Engine.TensorMaxValue(Tensor<T>.FromVector(S));
        return NumOps.Multiply(NumOps.FromDouble(RelativeTolerance), largest);
    }

    /// <summary>
    /// Selects and applies the appropriate SVD algorithm based on the specified type.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The SVD algorithm to use.</param>
    /// <returns>A tuple containing the U, S, and V^T components of the decomposition.</returns>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeDecomposition(Matrix<T> matrix, SvdAlgorithmType algorithm)
    {
        return algorithm switch
        {
            SvdAlgorithmType.GolubReinsch => ComputeSvdGolubReinsch(matrix),
            SvdAlgorithmType.Jacobi => ComputeSvdJacobi(matrix),
            SvdAlgorithmType.Randomized => ComputeSvdRandomized(matrix),
            SvdAlgorithmType.PowerIteration => ComputeSvdPowerIteration(matrix),
            SvdAlgorithmType.TruncatedSVD => ComputeTruncatedSvd(matrix),
            SvdAlgorithmType.DividedAndConquer => ComputeSvdDividedAndConquer(matrix),
            _ => throw new ArgumentException("Unsupported SVD algorithm", nameof(algorithm)),
        };
    }

    /// <summary>
    /// Solves a linear system Ax = b using the SVD decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x that best satisfy the equation Ax = b.
    /// SVD is particularly useful for solving systems that might not have an exact solution
    /// or when A is not a square matrix. It provides the best approximate solution in these cases.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        int l = S.Length;
        int n = Vt.Columns;
        if (l == 0)
        {
            return new Vector<T>(n);
        }

        // Pseudoinverse solve x = V_l diag(1/s) U_l^T b over the singular values above the cutoff
        // (the rest contribute nothing): the minimum-norm least-squares solution.
        T cutoff = SingularValueCutoff();
        var s = Tensor<T>.FromVector(S);
        var kept = Engine.TensorGreaterThan(s, cutoff);
        var safeS = Engine.TensorWhere(kept, s, Engine.TensorOnesLike(s));
        var weights = Engine.TensorMultiply(kept, Engine.TensorReciprocal(safeS));

        int m = U.Rows;
        var uL = Engine.TensorSlice(Tensor<T>.FromMatrix(U), new[] { 0, 0 }, new[] { m, l });
        var vtL = Engine.TensorSlice(Tensor<T>.FromMatrix(Vt), new[] { 0, 0 }, new[] { l, n });
        var projected = Engine.TensorMatMul(Engine.TensorTranspose(uL), Tensor<T>.FromVector(b, new[] { b.Length, 1 }));
        var scaled = Engine.TensorMultiply(projected, weights.Reshape(new[] { l, 1 }));
        var x = Engine.TensorMatMul(Engine.TensorTranspose(vtL), scaled);
        return x.Reshape(new[] { n }).ToVector();
    }

    /// <summary>
    /// Computes the SVD using the Golub-Reinsch algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the U, S, and V^T components of the decomposition.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Golub-Reinsch algorithm is the standard approach for computing SVD.
    /// It works by first converting the matrix to a simpler form (bidiagonal) and then
    /// iteratively refining it until we get the diagonal form we need.
    /// </para>
    /// <para>
    /// For wide matrices (m &lt; n), the algorithm transposes the matrix first since the
    /// standard Golub-Reinsch algorithm assumes m ≥ n. This follows the LAPACK convention.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdGolubReinsch(Matrix<T> matrix)
    {
        // Full SVD (U m x m, S of length min(m, n) in descending order, V^T n x n) from the engine.
        return ComputeEngineSvd(matrix, fullMatrices: true);
    }

    /// <summary>
    /// Computes the SVD with one engine call (<c>Linalg.Svd</c>).
    /// </summary>
    /// <param name="matrix">The m x n matrix to decompose.</param>
    /// <param name="fullMatrices">
    /// True for U m x m and V^T n x n; false for the thin U m x k and V^T k x n (k = min(m, n)).
    /// </param>
    /// <returns>U, the non-negative singular values in descending order, and V^T.</returns>
    /// <remarks>
    /// Left singular vectors that the engine leaves as zero columns (for zero singular values, or
    /// the extra m - n columns of a full U) are completed to an orthonormal basis of the
    /// orthogonal complement with a complete engine QR, so U is always orthogonal.
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeEngineSvd(Matrix<T> matrix, bool fullMatrices)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);
        if (l == 0)
        {
            return fullMatrices
                ? (Matrix<T>.CreateIdentity(m), new Vector<T>(0), Matrix<T>.CreateIdentity(n))
                : (new Matrix<T>(m, 0), new Vector<T>(0), new Matrix<T>(0, n));
        }

        var (u, s, vh) = Linalg.Svd(Tensor<T>.FromMatrix(matrix), fullMatrices);

        int uColumns = u.Shape[1];
        int filled = Engine.TensorCountNonzero(Engine.TensorNorm(u, 0));
        if (filled < uColumns)
        {
            Tensor<T> complement;
            if (filled == 0)
            {
                complement = Engine.TensorSlice(Engine.TensorEye<T>(m), new[] { 0, 0 }, new[] { m, uColumns });
            }
            else
            {
                var basis = Engine.TensorSlice(u, new[] { 0, 0 }, new[] { m, filled });
                var (q, _) = Linalg.QR(basis, "complete");
                complement = Engine.TensorSlice(q, new[] { 0, filled }, new[] { m, uColumns - filled });
            }

            u = filled == 0 ? complement : Engine.TensorSetSlice(u, complement, new[] { 0, filled });
        }

        return (u.ToMatrix(), s.ToVector(), vh.ToMatrix());
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using the Jacobi method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi method is an iterative approach to find the SVD. It works by 
    /// repeatedly applying small rotations to eliminate off-diagonal elements. This is like 
    /// gradually untangling the relationships in your data until you find the core patterns 
    /// (singular values) and their associated directions (singular vectors).
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdJacobi(Matrix<T> matrix)
    {
        // One-sided Jacobi rotations converge to the same factorization: U m x m, S descending,
        // V^T n x n.
        return ComputeEngineSvd(matrix, fullMatrices: true);
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using a randomized algorithm.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Randomized SVD is a faster approach for large matrices. Instead of processing 
    /// the entire matrix, it creates a smaller random projection of the data that preserves the most 
    /// important information. This is like creating a quick sketch of a large image that captures 
    /// the main features while ignoring minor details.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdRandomized(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n) / 2; // Number of singular values to compute

        // Step 1: Random projection
        Matrix<T> Omega = GenerateRandomMatrix(n, k);
        Matrix<T> Y = matrix.Multiply(Omega);

        // Step 2: QR decomposition of Y
        var qr = new QrDecomposition<T>(Y);
        Matrix<T> Q = qr.Q;

        // Step 3: Form B = Q^T * A
        Matrix<T> B = Q.Transpose().Multiply(matrix);

        // Step 4: SVD of B
        var svd = new SvdDecomposition<T>(B);

        // Step 5: Compute U = Q * U_B
        Matrix<T> U = Q.Multiply(svd.U);

        return (U, svd.S, svd.Vt);
    }

    /// <summary>
    /// Generates a random matrix with values between -1 and 1.
    /// </summary>
    /// <param name="rows">Number of rows in the matrix</param>
    /// <param name="cols">Number of columns in the matrix</param>
    /// <returns>A randomly generated matrix</returns>
    private Matrix<T> GenerateRandomMatrix(int rows, int cols)
    {
        // Uniform entries in [-1, 1) from a secure source.
        var random = RandomHelper.CreateSecureRandom();
        var uniform = Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(random, rows, cols));
        return Engine.TensorSubtractScalar(Engine.TensorMultiplyScalar(uniform, NumOps.FromDouble(2)), NumOps.One).ToMatrix();
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using the default algorithm.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the standard SVD algorithm that works in two main steps:
    /// 1. Bidiagonalization - transforms the matrix into a simpler form with non-zero values only on the main diagonal and one adjacent diagonal
    /// 2. Diagonalization - further simplifies the matrix to extract the singular values
    /// 
    /// Think of it as organizing a complex dataset into its most fundamental components.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdDefault(Matrix<T> matrix)
    {
        // Base case of the divide-and-conquer path: the full engine SVD.
        return ComputeEngineSvd(matrix, fullMatrices: true);
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using the power iteration method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Power iteration is like repeatedly applying a transformation to find the 
    /// dominant patterns in your data. Imagine shaking a box of mixed items - after enough shaking, 
    /// the heaviest items (dominant patterns) will settle to the bottom. This method finds singular 
    /// values one by one, from largest to smallest.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdPowerIteration(Matrix<T> matrix)
    {
        // Power iteration with deflation on A^T A recovers the singular triplets one at a time in
        // descending order: the thin SVD (U m x k, S k, V^T k x n, k = min(m, n)).
        return ComputeEngineSvd(matrix, fullMatrices: false);
    }

    /// <summary>
    /// Computes a truncated Singular Value Decomposition that keeps only the most significant singular values.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Truncated SVD is like creating a compressed version of your data that keeps 
    /// only the most important patterns. This is similar to image compression where you keep the 
    /// main features while discarding less noticeable details. It's useful for dimensionality 
    /// reduction and speeding up calculations.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeTruncatedSvd(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n) / 2; // Truncate to half the singular values

        Matrix<T> Q = Matrix<T>.CreateRandom(m, k);
        Q = MatrixHelper<T>.OrthogonalizeColumns(Q);

        for (int iter = 0; iter < 5; iter++) // You may need to adjust the number of iterations
        {
            Q = matrix.Multiply(matrix.Transpose().Multiply(Q));
            Q = MatrixHelper<T>.OrthogonalizeColumns(Q);
        }

        Matrix<T> B = Q.Transpose().Multiply(matrix);
        var svd = new SvdDecomposition<T>(B);

        Matrix<T> U = Q.Multiply(svd.U);
        Vector<T> S = svd.S;
        Matrix<T> VT = svd.Vt;

        return (U, S, VT);
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using a divide-and-conquer approach.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The divide-and-conquer approach breaks down a large problem into smaller, 
    /// more manageable pieces. It's like solving a large puzzle by first solving smaller sections 
    /// and then combining them. This method divides the matrix into four smaller matrices, computes 
    /// SVD for each, and then combines the results.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdDividedAndConquer(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        if (m <= 32 || n <= 32) // Base case: use another SVD method for small matrices
        {
            return ComputeSvdDefault(matrix);
        }

        // Divide the matrix into four submatrices
        int midM = m / 2;
        int midN = n / 2;

        Matrix<T> A11 = matrix.SubMatrix(0, midM, 0, midN);
        Matrix<T> A12 = matrix.SubMatrix(0, midM, midN, n);
        Matrix<T> A21 = matrix.SubMatrix(midM, m, 0, midN);
        Matrix<T> A22 = matrix.SubMatrix(midM, m, midN, n);

        // Recursively compute SVD for each submatrix
        var (U1, S1, VT1) = ComputeSvdDividedAndConquer(A11);
        var (U2, S2, VT2) = ComputeSvdDividedAndConquer(A12);
        var (U3, S3, VT3) = ComputeSvdDividedAndConquer(A21);
        var (U4, S4, VT4) = ComputeSvdDividedAndConquer(A22);

        // Combine the results
        Matrix<T> U = Matrix<T>.BlockDiagonal(U1, U2, U3, U4);
        Vector<T> S = Vector<T>.Concatenate(S1, S2, S3, S4);
        Matrix<T> VT = Matrix<T>.BlockDiagonal(VT1, VT2, VT3, VT4);

        // Perform a few iterations of the power method to refine the result
        for (int iter = 0; iter < 3; iter++)
        {
            Matrix<T> temp = matrix.Multiply(VT.Transpose());
            U = MatrixHelper<T>.OrthogonalizeColumns(temp);
            temp = U.Transpose().Multiply(matrix);
            VT = MatrixHelper<T>.OrthogonalizeColumns(temp.Transpose()).Transpose();
        }

        S = ComputeDiagonalElements(U.Transpose().Multiply(matrix).Multiply(VT.Transpose()));

        return (U, S, VT);
    }

    /// <summary>
    /// Extracts the diagonal elements from a matrix as singular values.
    /// </summary>
    /// <param name="matrix">The input matrix</param>
    /// <returns>A vector containing the absolute values of diagonal elements (singular values must be non-negative)</returns>
    private Vector<T> ComputeDiagonalElements(Matrix<T> matrix)
    {
        if (matrix.Rows == 0 || matrix.Columns == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorAbs(Linalg.Diagonal(Tensor<T>.FromMatrix(matrix))).ToVector();
    }
}
