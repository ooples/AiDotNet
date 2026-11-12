namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for matrix operations used in AI and machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for matrix elements.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A matrix is a rectangular array of numbers arranged in rows and columns.
/// Matrices are fundamental in machine learning for representing data, transformations, and 
/// mathematical operations.
/// </para>
/// </remarks>
public static class MatrixHelper<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Calculates the determinant of a matrix using a recursive algorithm.
    /// </summary>
    /// <param name="matrix">The matrix whose determinant is to be calculated.</param>
    /// <returns>The determinant value of the matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The determinant is a special number calculated from a square matrix.
    /// It tells us important information about the matrix, such as whether it has an inverse.
    /// If the determinant is zero, the matrix doesn't have an inverse.
    /// </para>
    /// <para>
    /// This method uses a recursive approach, breaking down the calculation into smaller parts
    /// by creating submatrices.
    /// </para>
    /// </remarks>
    public static T CalculateDeterminantRecursive(Matrix<T> matrix)
    {
        var rows = matrix.Rows;

        if (rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square.");
        }

        if (rows == 1)
        {
            return matrix[0, 0];
        }

        if (rows == 0)
        {
            return _numOps.Zero;
        }

        // Laplace expansion along row 0: sum_i (-1)^i * a[0, i] * det(minor(0, i)).
        T plus = _numOps.One;
        T minus = _numOps.FromDouble(-1);
        var signs = new Vector<T>(Enumerable.Range(0, rows).Select(i => i % 2 == 0 ? plus : minus));
        var minors = new Vector<T>(Enumerable.Range(0, rows)
            .Select(i => CalculateDeterminantRecursive(CreateSubMatrix(matrix, 0, i))));

        var engine = AiDotNetEngine.Current;
        return engine.DotProduct(engine.Multiply(signs, matrix.GetRow(0)), minors);
    }

    /// <summary>
    /// Creates a submatrix by excluding a specified row and column from the original matrix.
    /// </summary>
    /// <param name="matrix">The original matrix.</param>
    /// <param name="excludeRowIndex">The index of the row to exclude.</param>
    /// <param name="excludeColumnIndex">The index of the column to exclude.</param>
    /// <returns>A new matrix with the specified row and column removed.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A submatrix is created by removing a specific row and column from a matrix.
    /// This is commonly used in determinant calculations and other matrix operations.
    /// </para>
    /// </remarks>
    private static Matrix<T> CreateSubMatrix(Matrix<T> matrix, int excludeRowIndex, int excludeColumnIndex)
    {
        return matrix.RemoveRow(excludeRowIndex).RemoveColumn(excludeColumnIndex);
    }

    /// <summary>
    /// Reduces a matrix to Hessenberg form, which is useful for eigenvalue calculations.
    /// </summary>
    /// <param name="matrix">The matrix to reduce.</param>
    /// <returns>The matrix in Hessenberg form.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hessenberg matrix is almost triangular - it has zeros below the first 
    /// subdiagonal. Converting a matrix to Hessenberg form is often a first step in calculating 
    /// eigenvalues, which are important values that help us understand the behavior of linear 
    /// transformations in machine learning algorithms.
    /// </para>
    /// <para>
    /// This method uses Householder transformations to efficiently reduce the matrix.
    /// </para>
    /// </remarks>
    public static Matrix<T> ReduceToHessenbergFormat(Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var result = Matrix<T>.CreateMatrix<T>(rows, rows);

        for (int k = 0; k < rows - 2; k++)
        {
            var xVector = matrix.GetColumnSegment(k, k + 1, rows - k - 1);

            var hVector = CreateHouseholderVector(xVector);
            matrix = ApplyHouseholderTransformation(matrix, hVector, k);
        }

        return matrix;
    }

    /// <summary>
    /// Extracts the diagonal elements of a matrix into a vector.
    /// </summary>
    /// <param name="matrix">The matrix from which to extract the diagonal.</param>
    /// <returns>A vector containing the diagonal elements of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The diagonal of a matrix consists of the elements where the row index equals 
    /// the column index (top-left to bottom-right). In many AI algorithms, the diagonal elements 
    /// have special significance, such as representing variances in covariance matrices.
    /// </para>
    /// </remarks>
    public static Vector<T> ExtractDiagonal(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (matrix.Columns < n)
        {
            throw new ArgumentOutOfRangeException(nameof(matrix),
                $"The diagonal of a {n}x{matrix.Columns} matrix has no entry ({n - 1}, {n - 1}).");
        }

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AiDotNetEngine.Current.TensorDiagonal(Tensor<T>.FromMatrix(matrix)).ToVector();
    }

    /// <summary>
    /// Computes the outer product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>A matrix representing the outer product of the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The outer product of two vectors results in a matrix. If you have a vector 
    /// of size n and another of size m, their outer product is an n×m matrix where each element 
    /// is the product of the corresponding elements from each vector. This operation is used in 
    /// various machine learning algorithms, including neural networks for weight updates.
    /// </para>
    /// </remarks>
    public static Matrix<T> OuterProduct(Vector<T> v1, Vector<T> v2)
    {
        int n = v1.Length;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AiDotNetEngine.Current.TensorOuter(Tensor<T>.FromVector(v1), Tensor<T>.FromVector(v2.GetSubVector(0, n)))
            .ToMatrix();
    }

    /// <summary>
    /// Calculates the hypotenuse of a right triangle given the lengths of the other two sides.
    /// </summary>
    /// <param name="x">The length of one side of the right triangle.</param>
    /// <param name="y">The length of the other side of the right triangle.</param>
    /// <returns>The length of the hypotenuse.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The hypotenuse is the longest side of a right triangle, opposite to the right angle.
    /// This method calculates it using a numerically stable algorithm that avoids overflow or underflow
    /// issues that can occur with a direct application of the Pythagorean theorem (a² + b² = c²).
    /// </para>
    /// <para>
    /// This function is useful in many AI algorithms, particularly when calculating distances or norms.
    /// </para>
    /// </remarks>
    public static T Hypotenuse(T x, T y)
    {
        T _xabs = _numOps.Abs(x), _yabs = _numOps.Abs(y), _min, _max;

        if (_numOps.LessThan(_xabs, _yabs))
        {
            _min = _xabs; _max = _yabs;
        }
        else
        {
            _min = _yabs; _max = _xabs;
        }

        if (_numOps.Equals(_min, _numOps.Zero))
        {
            return _max;
        }

        T _u = _numOps.Divide(_min, _max);

        return _numOps.Multiply(_max, _numOps.Sqrt(_numOps.Add(_numOps.One, _numOps.Multiply(_u, _u))));
    }

    /// <summary>
    /// Calculates the Euclidean norm (magnitude) of a vector of values.
    /// </summary>
    /// <param name="values">The values to calculate the norm for.</param>
    /// <returns>The Euclidean norm of the values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Euclidean norm is a way to measure the "length" or "magnitude" of a vector.
    /// It's calculated as the square root of the sum of the squares of all values. In a 2D space,
    /// this is equivalent to finding the hypotenuse of a right triangle using the Pythagorean theorem.
    /// </para>
    /// <para>
    /// In machine learning, norms are often used to measure the size of vectors, such as weight vectors
    /// in neural networks or for regularization techniques.
    /// </para>
    /// </remarks>
    public static T Hypotenuse(params T[] values)
    {
        T _sum = _numOps.Zero;
        if (values.Length > 0)
        {
            var vector = Vector<T>.FromSpan(values);
            _sum = AiDotNetEngine.Current.DotProduct(vector, vector);
        }

        return _numOps.Sqrt(_sum);
    }

    /// <summary>
    /// Determines if a matrix is in upper Hessenberg form within a specified tolerance.
    /// </summary>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="tolerance">The numerical tolerance for considering a value as zero.</param>
    /// <returns>True if the matrix is in upper Hessenberg form, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper Hessenberg matrix is almost triangular - it has zeros below the first 
    /// subdiagonal. This is an intermediate form used in many eigenvalue algorithms, making computations 
    /// more efficient than working with a full matrix.
    /// </para>
    /// </remarks>
    public static bool IsUpperHessenberg(Matrix<T> matrix, T tolerance)
    {
        if (matrix.Rows < 3)
        {
            return true;
        }

        // Entries below the first subdiagonal (column - row <= -2) must not exceed the tolerance.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var exceeds = engine.TensorGreaterThan(engine.TensorAbs(Tensor<T>.FromMatrix(matrix)), tolerance);
        return engine.TensorCountNonzero(engine.TensorTril(exceeds, -2)) == 0;
    }

    /// <summary>
    /// Orthogonalizes the columns of a matrix using the Gram-Schmidt process.
    /// </summary>
    /// <param name="matrix">The matrix whose columns will be orthogonalized.</param>
    /// <returns>A matrix with orthogonal columns.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Orthogonalization means making vectors perpendicular to each other. 
    /// The Gram-Schmidt process takes a set of vectors and creates a new set where each vector 
    /// is perpendicular (orthogonal) to all previous vectors. This is important in many machine 
    /// learning algorithms that need independent features or basis vectors.
    /// </para>
    /// </remarks>
    public static Matrix<T> OrthogonalizeColumns(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);

        for (int j = 0; j < n; j++)
        {
            Vector<T> v = matrix.GetColumn(j);
            for (int i = 0; i < j; i++)
            {
                Vector<T> qi = Q.GetColumn(i);
                T r = _numOps.Divide(v.DotProduct(qi), qi.DotProduct(qi));
                v = v.Subtract(qi.Multiply(r));
            }
            T norm = v.Norm();
            if (!_numOps.Equals(norm, _numOps.Zero))
            {
                v = v.Divide(norm);
            }
            Q.SetColumn(j, v);
        }

        return Q;
    }

    /// <summary>
    /// Computes the cosine and sine components of a Givens rotation.
    /// </summary>
    /// <param name="a">The first element used to compute the rotation.</param>
    /// <param name="b">The second element used to compute the rotation.</param>
    /// <returns>A tuple containing the cosine and sine values of the rotation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Givens rotation is a way to zero out specific elements in a matrix.
    /// It's like rotating a 2D coordinate system to make one component become zero.
    /// This is useful in many numerical algorithms to simplify matrices step by step.
    /// </para>
    /// </remarks>
    public static (T c, T s) ComputeGivensRotation(T a, T b)
    {
        if (_numOps.Equals(b, _numOps.Zero))
        {
            return (_numOps.One, _numOps.Zero);
        }

        var t = _numOps.Divide(a, b);
        var u = _numOps.Sqrt(_numOps.Add(_numOps.One, _numOps.Multiply(t, t)));
        var c = _numOps.Divide(_numOps.One, u);
        var s = _numOps.Multiply(c, t);

        return (c, s);
    }

    /// <summary>
    /// Applies a Givens rotation to specific rows of a matrix.
    /// </summary>
    /// <param name="H">The matrix to which the rotation will be applied.</param>
    /// <param name="c">The cosine component of the Givens rotation.</param>
    /// <param name="s">The sine component of the Givens rotation.</param>
    /// <param name="i">The index of the first row to be rotated.</param>
    /// <param name="j">The index of the second row to be rotated.</param>
    /// <param name="kStart">The starting column index for the rotation.</param>
    /// <param name="kEnd">The ending column index for the rotation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method applies a rotation to two rows of a matrix. It's like
    /// mixing two rows together in specific proportions (determined by c and s) to create
    /// new rows. This is commonly used to zero out specific elements in numerical algorithms.
    /// </para>
    /// </remarks>
    public static void ApplyGivensRotation(Matrix<T> H, T c, T s, int i, int j, int kStart, int kEnd)
    {
        int length = kEnd - kStart;
        if (length <= 0)
        {
            return;
        }

        // Rows i and j over columns [kStart, kEnd): (c*ri + s*rj, -s*ri + c*rj)
        var engine = AiDotNetEngine.Current;
        var rowI = H.GetRowSegment(i, kStart, length);
        var rowJ = H.GetRowSegment(j, kStart, length);
        var newI = engine.Add(engine.Multiply(rowI, c), engine.Multiply(rowJ, s));
        var newJ = engine.Add(engine.Multiply(rowI, _numOps.Negate(s)), engine.Multiply(rowJ, c));
        newI.AsSpan().CopyTo(H.GetRowSpan(i).Slice(kStart, length));
        newJ.AsSpan().CopyTo(H.GetRowSpan(j).Slice(kStart, length));
    }

    /// <summary>
    /// Applies a Householder transformation to a matrix.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <param name="vector">The Householder vector defining the reflection.</param>
    /// <param name="k">The starting row and column for the transformation.</param>
    /// <returns>The transformed matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Householder transformation is a way to reflect vectors across a plane.
    /// In matrix operations, it's used to introduce zeros in specific parts of a matrix.
    /// This is a key step in many algorithms that decompose matrices into simpler forms.
    /// </para>
    /// </remarks>
    public static Matrix<T> ApplyHouseholderTransformation(Matrix<T> matrix, Vector<T> vector, int k)
    {
        var rows = matrix.Rows;
        int span = rows - k - 1;
        if (span <= 0 || k >= rows)
        {
            return matrix;
        }

        var engine = AiDotNetEngine.Current;
        var v = vector.GetSubVector(0, span);
        var twoV = engine.Multiply(v, _numOps.FromDouble(2));

        // Apply Householder reflector from the left: H * A
        // This modifies columns k to n-1 (must include column k to zero out elements below subdiagonal!)
        // Block B = A[k+1.., k..]: B -= (2v) (v^T B)
        var left = matrix.GetSubMatrix(k + 1, k, span, rows - k);
        var leftSums = engine.MatrixVectorMultiply(left.Transpose(), v);
        matrix.SetSubMatrix(k + 1, k, engine.MatrixSubtract(left, OuterProduct(twoV, leftSums, span, rows - k)));

        // Apply Householder reflector from the right: A * H
        // This modifies all rows to preserve similarity transformation
        // Block C = A[.., k+1..]: C -= (C v) (2v)^T
        var right = matrix.GetSubMatrix(0, k + 1, rows, span);
        var rightSums = engine.MatrixVectorMultiply(right, v);
        matrix.SetSubMatrix(0, k + 1, engine.MatrixSubtract(right, OuterProduct(rightSums, twoV, rows, span)));

        return matrix;
    }

    /// <summary>The [rows, columns] outer product a * b^T of two vectors.</summary>
    private static Matrix<T> OuterProduct(Vector<T> a, Vector<T> b, int rows, int columns)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AiDotNetEngine.Current.TensorOuter(Tensor<T>.FromVector(a), Tensor<T>.FromVector(b))
            .Reshape(new[] { rows, columns }).ToMatrix();
    }

    /// <summary>
    /// Creates a Householder vector from a given vector.
    /// </summary>
    /// <param name="xVector">The input vector.</param>
    /// <returns>A Householder vector that can be used for reflection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Householder vector defines a reflection plane that, when applied to the
    /// original vector, zeros out all but the first component. This is useful in many matrix
    /// decomposition algorithms to systematically simplify matrices.
    /// </para>
    /// </remarks>
    public static Vector<T> CreateHouseholderVector(Vector<T> xVector)
    {
        var engine = AiDotNetEngine.Current;
        T norm = xVector.Length == 0 ? _numOps.Zero : _numOps.Sqrt(engine.DotProduct(xVector, xVector));

        // If the input vector is near-zero, return a zero vector (identity transformation)
        // This works because: P = I - 2*v*v^T, and if v = 0, then P = I
        if (_numOps.LessThan(norm, _numOps.FromDouble(1e-14)))
        {
            return new Vector<T>(xVector.Length);
        }

        // Use OPPOSITE sign for numerical stability (avoid cancellation)
        T alpha = _numOps.LessThan(xVector[0], _numOps.Zero)
            ? norm
            : _numOps.Negate(norm);

        var result = xVector.Clone();
        result[0] = _numOps.Subtract(xVector[0], alpha);

        T vNorm = _numOps.Sqrt(engine.DotProduct(result, result));

        // Handle the case where vNorm is near-zero (return zero vector for identity)
        if (_numOps.LessThan(vNorm, _numOps.FromDouble(1e-14)))
        {
            return new Vector<T>(xVector.Length);
        }

        return engine.Divide(result, vNorm);
    }

    /// <summary>
    /// Implements the power iteration algorithm to find the dominant eigenvalue and eigenvector of a matrix.
    /// </summary>
    /// <param name="aMatrix">The input matrix for which to find the dominant eigenvalue and eigenvector.</param>
    /// <param name="maxIterations">The maximum number of iterations to perform.</param>
    /// <param name="tolerance">The convergence tolerance.</param>
    /// <returns>A tuple containing the dominant eigenvalue and its corresponding eigenvector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An eigenvalue and eigenvector are special values and vectors associated with a matrix.
    /// When you multiply a matrix by its eigenvector, you get the same vector scaled by the eigenvalue.
    /// The power iteration method repeatedly multiplies the matrix by a vector and normalizes it until
    /// it converges to the eigenvector with the largest eigenvalue (the dominant one). This is useful
    /// in many AI algorithms like PageRank, PCA, and recommendation systems.
    /// </para>
    /// </remarks>
    public static (T, Vector<T>) PowerIteration(Matrix<T> aMatrix, int maxIterations, T tolerance)
    {
        var rows = aMatrix.Rows;
        var bVector = new Vector<T>(rows);
        var b2Vector = new Vector<T>(rows);
        T eigenvalue = _numOps.Zero;

        // Initial guess for the eigenvector
        bVector.Fill(_numOps.One);
        var engine = AiDotNetEngine.Current;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Multiply A by the vector b
            b2Vector = engine.MatrixVectorMultiply(aMatrix, bVector);

            // Estimate the eigenvalue using Rayleigh quotient: b^T * Ab / (b^T * b)
            // Since bVector is normalized (b^T * b = 1), this simplifies to b^T * Ab
            T newEigenvalue = engine.DotProduct(bVector, b2Vector);

            // Normalize the vector
            T norm = _numOps.Sqrt(engine.DotProduct(b2Vector, b2Vector));
            b2Vector = engine.Divide(b2Vector, norm);

            // Check for convergence
            if (_numOps.LessThan(_numOps.Abs(_numOps.Subtract(newEigenvalue, eigenvalue)), tolerance))
            {
                break;
            }
            eigenvalue = newEigenvalue;
            bVector = b2Vector;
        }

        return (eigenvalue, b2Vector);
    }

    /// <summary>
    /// Calculates the spectral norm of a matrix, which is the largest singular value.
    /// </summary>
    /// <param name="matrix">The matrix for which to calculate the spectral norm.</param>
    /// <returns>The spectral norm of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The spectral norm measures the maximum "stretching" that a matrix can cause
    /// when applied to a vector. It's the largest singular value of the matrix, which indicates
    /// how much the matrix can amplify a vector in any direction. In machine learning, this helps
    /// understand the stability of algorithms and the conditioning of data.
    /// </para>
    /// </remarks>
    public static T SpectralNorm(Matrix<T> matrix)
    {
        // Use the power iteration method to estimate the spectral norm
        int maxIterations = 100;
        T tolerance = _numOps.FromDouble(1e-10);

        Vector<T> v = Vector<T>.CreateRandom(matrix.Columns);
        v = v.Divide(v.Norm());

        for (int i = 0; i < maxIterations; i++)
        {
            Vector<T> w = matrix.Transpose().Multiply(matrix.Multiply(v));
            T lambda = v.DotProduct(w);
            Vector<T> vNew = w.Divide(w.Norm());

            if (_numOps.LessThan(vNew.Subtract(v).Norm(), tolerance))
            {
                break;
            }

            v = vNew;
        }

        return _numOps.Sqrt(v.DotProduct(matrix.Transpose().Multiply(matrix.Multiply(v))));
    }

    /// <summary>
    /// Determines if a matrix is invertible (non-singular).
    /// </summary>
    /// <param name="matrix">The matrix to check for invertibility.</param>
    /// <returns>True if the matrix is invertible, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An invertible matrix is one that has an inverse - another matrix that, when
    /// multiplied with the original, gives the identity matrix. For a matrix to be invertible, it must
    /// be square (same number of rows and columns) and have a non-zero determinant. In machine learning,
    /// invertible matrices are important for solving linear systems and in algorithms like linear regression.
    /// </para>
    /// </remarks>
    public static bool IsInvertible(Matrix<T> matrix)
    {
        // Check if the matrix is square
        if (matrix.Rows != matrix.Columns)
        {
            return false;
        }

        // Check if the determinant is zero (or very close to zero)
        T determinant = matrix.Determinant();
        T tolerance = _numOps.FromDouble(1e-10);

        return _numOps.GreaterThan(_numOps.Abs(determinant), tolerance);
    }

    /// <summary>
    /// Inverts a matrix using a provided matrix decomposition.
    /// </summary>
    /// <param name="decomposition">The matrix decomposition to use for inversion.</param>
    /// <returns>The inverse of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix inversion is like finding the reciprocal of a number, but for matrices.
    /// This method uses a decomposition (a way of breaking down a matrix into simpler parts) to
    /// efficiently compute the inverse. Matrix inversion is used in many machine learning algorithms,
    /// especially in linear regression and when solving systems of linear equations.
    /// </para>
    /// </remarks>
    public static Matrix<T> InvertUsingDecomposition(IMatrixDecomposition<T> decomposition)
    {
        int n = decomposition.A.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // Column j of the inverse solves A x = e_j.
        return Matrix<T>.FromColumnVectors(Enumerable.Range(0, n)
            .Select(j => decomposition.Solve(Vector<T>.CreateStandardBasis(n, j)).GetSubVector(0, n)));
    }

    /// <summary>
    /// Solves a tridiagonal system of linear equations.
    /// </summary>
    /// <param name="vector1">The subdiagonal elements (below the main diagonal).</param>
    /// <param name="vector2">The main diagonal elements.</param>
    /// <param name="vector3">The superdiagonal elements (above the main diagonal).</param>
    /// <param name="solutionVector">The vector where the solution will be stored.</param>
    /// <param name="actualVector">The right-hand side vector of the system.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tridiagonal matrix is a special type of matrix where non-zero elements
    /// are only on the main diagonal and the diagonals directly above and below it. This method
    /// efficiently solves equations of the form Ax = b, where A is a tridiagonal matrix.
    /// Tridiagonal systems appear in many numerical methods for differential equations and
    /// spline interpolation used in machine learning.
    /// </para>
    /// </remarks>
    public static void TridiagonalSolve(Vector<T> vector1, Vector<T> vector2, Vector<T> vector3,
    Vector<T> solutionVector, Vector<T> actualVector)
    {
        var size = vector1.Length;
        T bet;
        var gamVector = new Vector<T>(size);

        if (_numOps.Equals(vector2[0], _numOps.Zero))
        {
            throw new InvalidOperationException("Not a tridiagonal matrix!");
        }

        bet = vector2[0];
        solutionVector[0] = _numOps.Divide(actualVector[0], bet);
        for (int i = 1; i < size; i++)
        {
            gamVector[i] = _numOps.Divide(vector3[i - 1], bet);
            bet = _numOps.Subtract(vector2[i], _numOps.Multiply(vector1[i], gamVector[i]));

            if (_numOps.Equals(bet, _numOps.Zero))
            {
                throw new InvalidOperationException("Not a tridiagonal matrix!");
            }

            solutionVector[i] = _numOps.Divide(
                _numOps.Subtract(actualVector[i], _numOps.Multiply(vector1[i], solutionVector[i - 1])),
                bet
            );
        }

        for (int i = size - 2; i >= 0; i--)
        {
            solutionVector[i] = _numOps.Subtract(
                solutionVector[i],
                _numOps.Multiply(gamVector[i + 1], solutionVector[i + 1])
            );
        }
    }

    /// <summary>
    /// Multiplies a band diagonal matrix by a vector.
    /// </summary>
    /// <param name="leftSide">The number of subdiagonals (bands below the main diagonal).</param>
    /// <param name="rightSide">The number of superdiagonals (bands above the main diagonal).</param>
    /// <param name="matrix">The band diagonal matrix stored in compact form.</param>
    /// <param name="solutionVector">The vector where the result will be stored.</param>
    /// <param name="actualVector">The vector to multiply with the matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A band diagonal matrix is a matrix where non-zero elements are concentrated
    /// around the main diagonal within a certain "band". This method efficiently multiplies such
    /// a matrix with a vector without processing all the zero elements outside the band.
    /// Band matrices often arise when discretizing differential equations and in image processing
    /// algorithms used in machine learning.
    /// </para>
    /// </remarks>
    public static void BandDiagonalMultiply(int leftSide, int rightSide, Matrix<T> matrix, Vector<T> solutionVector, Vector<T> actualVector)
    {
        var size = matrix.Rows;
        int bandWidth = leftSide + rightSide + 1;
        if (size == 0 || bandWidth <= 0)
        {
            return;
        }

        // Row i of the compact band matrix multiplies x[i - leftSide + j] for band column j; entries that
        // fall outside x are skipped, which is the same as reading zero padding on both sides of x.
        var engine = AiDotNetEngine.Current;
        var padded = Vector<T>.Concatenate(new Vector<T>(leftSide), actualVector.GetSubVector(0, size), new Vector<T>(rightSide));
        var windows = Matrix<T>.FromColumnVectors(Enumerable.Range(0, bandWidth).Select(j => padded.GetSubVector(j, size)));
        var band = matrix.GetSubMatrix(0, 0, size, bandWidth);

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var products = engine.TensorMultiply(Tensor<T>.FromMatrix(band), Tensor<T>.FromMatrix(windows));
        engine.ReduceSum(products, new[] { 1 }, false).AsSpan().CopyTo(solutionVector.AsWritableSpan());
    }

    /// <summary>
    /// Calculates the Hat Matrix (also known as the projection matrix) used in regression analysis.
    /// </summary>
    /// <param name="features">The feature matrix (design matrix) containing the independent variables.</param>
    /// <returns>The Hat Matrix that projects the dependent variable onto the fitted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hat Matrix is an important concept in regression analysis. It "puts a hat" on 
    /// your data, transforming your actual observed values into predicted values. Mathematically, it's 
    /// calculated as H = X(X'X)^(-1)X', where X is your feature matrix, X' is its transpose, and ^(-1) 
    /// means matrix inverse.
    /// </para>
    /// <para>
    /// The Hat Matrix has several important properties:
    /// - It's used to calculate fitted values in regression: y = Hy
    /// - The diagonal elements (H_ii) tell you how much influence each data point has on the model
    /// - These diagonal values are used to identify outliers and high-leverage points
    /// - In machine learning, understanding the Hat Matrix helps with model diagnostics and improving prediction accuracy
    /// </para>
    /// </remarks>
    public static Matrix<T> CalculateHatMatrix(Matrix<T> features)
    {
        var transposeFeatures = features.Transpose();
        var inverseMatrix = transposeFeatures.Multiply(features).Inverse();

        return features.Multiply(inverseMatrix.Multiply(transposeFeatures));
    }
}
