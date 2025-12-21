namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for matrix operations, making it easier to work with matrices in AI applications.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A matrix is a rectangular array of numbers arranged in rows and columns.
/// These extension methods add useful functionality to matrices, like adding columns or performing
/// mathematical operations that are commonly needed in AI and machine learning algorithms.
/// </remarks>
public static class MatrixExtensions
{
    /// <summary>
    /// Adds a constant value as the first column of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric data type of the matrix elements.</typeparam>
    /// <param name="matrix">The original matrix to modify.</param>
    /// <param name="value">The constant value to add as the first column.</param>
    /// <returns>A new matrix with the constant column added as the first column.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method is often used in linear regression to add a "bias" term.
    /// It creates a new matrix with an extra column at the beginning, where every value in that
    /// column is the same (usually 1). This helps the AI model learn an offset or baseline value.
    /// </remarks>
    public static Matrix<T> AddConstantColumn<T>(this Matrix<T> matrix, T value)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var newMatrix = new Matrix<T>(matrix.Rows, matrix.Columns + 1);

        // Vectorized: Copy each row using SIMD operations
        for (int i = 0; i < matrix.Rows; i++)
        {
            newMatrix[i, 0] = value;
            var sourceRow = matrix.GetRowReadOnlySpan(i);
            var destRow = newMatrix.GetRowSpan(i);
            numOps.Copy(sourceRow, destRow.Slice(1, matrix.Columns));
        }

        return newMatrix;
    }

    /// <summary>
    /// Converts a matrix to a vector by flattening its elements in row-major order.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to convert to a vector.</param>
    /// <returns>A vector containing all elements of the matrix in row-major order.</returns>
    /// <remarks>
    /// <para>
    /// This method flattens a matrix into a vector by concatenating all rows in order.
    /// The resulting vector has a length equal to rows * columns of the original matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a grid of numbers and writing them out in a single line,
    /// going from left to right, top to bottom. For example, a 2x3 matrix:
    /// [1, 2, 3]
    /// [4, 5, 6]
    /// becomes the vector: [1, 2, 3, 4, 5, 6]
    /// </para>
    /// </remarks>
    public static Vector<T> ToVector<T>(this Matrix<T> matrix)
    {
        if (matrix == null)
            throw new ArgumentNullException(nameof(matrix));

        int size = matrix.Rows * matrix.Columns;
        var result = new Vector<T>(size);

        // Vectorized: Copy entire matrix data at once
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Copy(matrix.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Adds a vector to each row of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric data type of the matrix and vector elements.</typeparam>
    /// <param name="matrix">The matrix to which the vector will be added.</param>
    /// <param name="vector">The vector to add to each row of the matrix.</param>
    /// <returns>A new matrix where each row is the sum of the original row and the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector length doesn't match the matrix column count.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This is like adding the same set of values to every row in your matrix.
    /// For example, if you have a matrix of features for different data points, and you want to
    /// adjust all features by the same amount, you would use this method.
    /// </remarks>
    public static Matrix<T> AddVectorToEachRow<T>(this Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Columns != vector.Length)
            throw new ArgumentException("Vector length must match matrix column count");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        // Vectorized: Add vector to each row using SIMD operations
        for (int i = 0; i < matrix.Rows; i++)
        {
            var sourceRow = matrix.GetRowReadOnlySpan(i);
            var destRow = result.GetRowSpan(i);
            numOps.Add(sourceRow, vector.AsSpan(), destRow);
        }

        return result;
    }

    /// <summary>
    /// Calculates the sum of each column in the matrix.
    /// </summary>
    /// <typeparam name="T">The numeric data type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix whose columns will be summed.</param>
    /// <returns>A vector containing the sum of each column.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method adds up all the values in each column of your matrix.
    /// For example, if your matrix represents multiple data points with features in columns,
    /// this would give you the total of each feature across all data points.
    /// </remarks>
    public static Vector<T> SumColumns<T>(this Matrix<T> matrix)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            T sum = ops.Zero;
            for (int i = 0; i < matrix.Rows; i++)
            {
                sum = ops.Add(sum, matrix[i, j]);
            }

            result[j] = sum;
        }

        return result;
    }

    /// <summary>
    /// Extracts a specific column from the matrix as a vector.
    /// </summary>
    /// <typeparam name="T">The numeric data type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix from which to extract the column.</param>
    /// <param name="columnIndex">The index of the column to extract.</param>
    /// <returns>A vector containing the values from the specified column.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method pulls out a single column from your matrix and returns it as a vector.
    /// This is useful when you need to work with just one feature or dimension from your dataset.
    /// </remarks>
    public static Vector<T> GetColumn<T>(this Matrix<T> matrix, int columnIndex)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var column = new Vector<T>(matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            column[i] = matrix[i, columnIndex];
        }

        return column;
    }

    /// <summary>
    /// Performs backward substitution to solve a system of linear equations represented by an upper triangular matrix.
    /// </summary>
    /// <typeparam name="T">The numeric data type of the matrix and vector elements.</typeparam>
    /// <param name="aMatrix">The upper triangular coefficient matrix.</param>
    /// <param name="bVector">The right-hand side vector of the equation system.</param>
    /// <returns>The solution vector x where Ax = b.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Backward substitution is a method to solve equations when your matrix is in a special form
    /// called "upper triangular" (where all values below the diagonal are zero).
    /// 
    /// This is often used as the final step in solving systems of linear equations, which is a common
    /// task in many AI algorithms like linear regression. The method starts from the bottom row and works upward,
    /// solving for one variable at a time.
    /// </remarks>
    public static Vector<T> BackwardSubstitution<T>(this Matrix<T> aMatrix, Vector<T> bVector)
    {
        int n = aMatrix.Rows;
        var xVector = new Vector<T>(n);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = n - 1; i >= 0; --i)
        {
            xVector[i] = bVector[i];
            for (int j = i + 1; j < n; ++j)
            {
                xVector[i] = ops.Subtract(xVector[i], ops.Multiply(aMatrix[i, j], xVector[j]));
            }
            xVector[i] = ops.Divide(xVector[i], aMatrix[i, i]);
        }

        return xVector;
    }

    /// <summary>
    /// Identifies the types of a matrix based on its properties.
    /// </summary>
    /// <typeparam name="T">The numeric data type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="matrixDecomposition">The decomposition method to use for certain matrix type checks.</param>
    /// <param name="tolerance">The numerical tolerance for floating-point comparisons (default: 1e-10).</param>
    /// <param name="subDiagonalThreshold">The threshold for sub-diagonal elements (default: 1).</param>
    /// <param name="superDiagonalThreshold">The threshold for super-diagonal elements (default: 1).</param>
    /// <param name="sparsityThreshold">The threshold ratio for determining if a matrix is sparse (default: 0.5).</param>
    /// <param name="denseThreshold">The threshold ratio for determining if a matrix is dense (default: 0.5).</param>
    /// <param name="blockRows">The number of rows in each block for block matrix detection (default: 2).</param>
    /// <param name="blockCols">The number of columns in each block for block matrix detection (default: 2).</param>
    /// <returns>An enumeration of matrix types that apply to the given matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method examines a matrix and tells you what special properties it has.
    /// 
    /// Matrices can have many different characteristics (like being symmetric, triangular, etc.)
    /// that make them behave in special ways. Knowing these properties can help you choose the right
    /// algorithms to work with your data efficiently.
    /// 
    /// For example, if your matrix is "sparse" (mostly zeros), you can use special techniques that
    /// save memory and computation time.
    /// </remarks>
    public static IEnumerable<MatrixType> GetMatrixTypes<T>(this Matrix<T> matrix, IMatrixDecomposition<T> matrixDecomposition,
    T? tolerance = default, int subDiagonalThreshold = 1, int superDiagonalThreshold = 1, T? sparsityThreshold = default, T? denseThreshold = default,
    int blockRows = 2, int blockCols = 2)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        tolerance ??= ops.FromDouble(1e-10);
        sparsityThreshold ??= ops.FromDouble(0.5);
        denseThreshold ??= ops.FromDouble(0.5);

        var complexMatrix = matrix.ToComplexMatrix();
        foreach (var matrixType in EnumHelper.GetEnumValues<MatrixType>())
        {
            switch (matrixType)
            {
                case MatrixType.Square:
                    if (matrix.IsSquareMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Rectangular:
                    if (matrix.IsRectangularMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Diagonal:
                    if (matrix.IsDiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Identity:
                    if (matrix.IsIdentityMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Symmetric:
                    if (matrix.IsSymmetricMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.SkewSymmetric:
                    if (matrix.IsSkewSymmetricMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.UpperTriangular:
                    if (matrix.IsUpperTriangularMatrix(tolerance))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.LowerTriangular:
                    if (matrix.IsLowerTriangularMatrix(tolerance))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Zero:
                    if (matrix.IsZeroMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Scalar:
                    if (matrix.IsScalarMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.UpperBidiagonal:
                    if (matrix.IsUpperBidiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.LowerBidiagonal:
                    if (matrix.IsLowerBidiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Tridiagonal:
                    if (matrix.IsTridiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Band:
                    if (matrix.IsBandMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Hermitian:
                    if (complexMatrix.IsHermitianMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.SkewHermitian:
                    if (complexMatrix.IsSkewHermitianMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Orthogonal:
                    if (matrix.IsOrthogonalMatrix(matrixDecomposition))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Unitary:
                    var complexDecomposition = matrixDecomposition.ToComplexDecomposition();
                    if (complexMatrix.IsUnitaryMatrix(complexDecomposition))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Singular:
                    if (matrix.IsSingularMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.NonSingular:
                    if (matrix.IsNonSingularMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.PositiveDefinite:
                    if (matrix.IsPositiveDefiniteMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.PositiveSemiDefinite:
                    if (matrix.IsPositiveSemiDefiniteMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.OrthogonalProjection:
                    if (matrix.IsOrthogonalProjectionMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Idempotent:
                    if (matrix.IsIdempotentMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Involutory:
                    if (matrix.IsInvolutoryMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Stochastic:
                    if (matrix.IsStochasticMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.DoublyStochastic:
                    if (matrix.IsDoublyStochasticMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Permutation:
                    if (matrix.IsPermutationMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Adjacency:
                    if (matrix.IsAdjacencyMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Incidence:
                    if (matrix.IsIncidenceMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Laplacian:
                    if (matrix.IsLaplacianMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Toeplitz:
                    if (matrix.IsToeplitzMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Hankel:
                    if (matrix.IsHankelMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Circulant:
                    if (matrix.IsCirculantMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Block:
                    if (matrix.IsBlockMatrix(blockRows, blockCols))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Sparse:
                    if (matrix.IsSparseMatrix(sparsityThreshold))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Dense:
                    if (matrix.IsDenseMatrix(denseThreshold))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Partitioned:
                    if (matrix.IsPartitionedMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Companion:
                    if (matrix.IsCompanionMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Vandermonde:
                    if (matrix.IsVandermondeMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Hilbert:
                    if (matrix.IsHilbertMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Cauchy:
                    if (matrix.IsCauchyMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                default:
                    yield return matrixType;
                    break;
            }
        }
    }

    /// <summary>
    /// Determines if a matrix can be divided into consistent blocks of a specified size.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="blockRows">The number of rows in each block.</param>
    /// <param name="blockCols">The number of columns in each block.</param>
    /// <returns>True if the matrix can be divided into consistent blocks, otherwise false.</returns>
    /// <remarks>
    /// <para>For Beginners: A block matrix is a matrix that can be divided into smaller matrices (blocks) 
    /// where each block has the same properties. Think of it like dividing a large grid into smaller, 
    /// identical sub-grids. This method checks if your matrix can be neatly divided this way.</para>
    /// </remarks>
    public static bool IsBlockMatrix<T>(this Matrix<T> matrix, int blockRows, int blockCols)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if the matrix dimensions are compatible with the block size
        if (rows % blockRows != 0 || cols % blockCols != 0)
        {
            return false;
        }

        int numBlockRows = rows / blockRows;
        int numBlockCols = cols / blockCols;

        // Check if each block is consistent
        for (int blockRow = 0; blockRow < numBlockRows; blockRow++)
        {
            for (int blockCol = 0; blockCol < numBlockCols; blockCol++)
            {
                var block = matrix.GetBlock(blockRow * blockRows, blockCol * blockCols, blockRows, blockCols);
                if (!block.IsConsistentBlock())
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Extracts a block (sub-matrix) from a matrix starting at the specified position.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="startRow">The starting row index (0-based).</param>
    /// <param name="startCol">The starting column index (0-based).</param>
    /// <param name="blockRows">The number of rows in the block.</param>
    /// <param name="blockCols">The number of columns in the block.</param>
    /// <returns>A new matrix containing the extracted block.</returns>
    /// <remarks>
    /// <para>For Beginners: This method lets you take a smaller piece (or "block") out of a larger matrix.
    /// It's like cutting out a rectangular section from a grid. You specify where to start cutting (startRow, startCol)
    /// and how big a piece to cut (blockRows, blockCols).</para>
    /// </remarks>
    public static Matrix<T> GetBlock<T>(this Matrix<T> matrix, int startRow, int startCol, int blockRows, int blockCols)
    {
        var block = new Matrix<T>(blockRows, blockCols);
        for (int i = 0; i < blockRows; i++)
        {
            for (int j = 0; j < blockCols; j++)
            {
                block[i, j] = matrix[startRow + i, startCol + j];
            }
        }

        return block;
    }

    /// <summary>
    /// Checks if all elements in a matrix block are identical.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="block">The matrix block to check.</param>
    /// <returns>True if all elements in the block are identical, otherwise false.</returns>
    /// <remarks>
    /// <para>For Beginners: This method checks if all numbers in a matrix are exactly the same.
    /// For example, if every position in your grid contains the number 5, this would return true.
    /// If there's even one different number, it returns false.</para>
    /// </remarks>
    public static bool IsConsistentBlock<T>(this Matrix<T> block)
    {
        var rows = block.Rows;
        var cols = block.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Implement your own logic to check consistency within the block
        // For simplicity, let's assume the block is consistent if all elements are the same
        T firstElement = block[0, 0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(block[i, j], firstElement))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is upper triangular (all elements below the main diagonal are zero).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="tolerance">Optional tolerance value for floating-point comparisons. Default is 1e-10.</param>
    /// <returns>True if the matrix is upper triangular, otherwise false.</returns>
    /// <remarks>
    /// <para>For Beginners: An upper triangular matrix has all its non-zero values either on or above the main diagonal
    /// (the diagonal line from top-left to bottom-right). Everything below this diagonal is zero.
    /// This is useful in many mathematical operations because it simplifies calculations.</para>
    /// <para>The tolerance parameter helps when working with decimal numbers that might have tiny rounding errors.</para>
    /// </remarks>
    public static bool IsUpperTriangularMatrix<T>(this Matrix<T> matrix, T? tolerance = default)
    {
        var ops = MathHelper.GetNumericOperations<T>();

        // If tolerance is not provided (i.e., it's default(T)), use a small value based on the type
        if (tolerance?.Equals(default(T)) ?? true)
        {
            tolerance = ops.FromDouble(1e-10); // Use a small value as default tolerance
        }

        for (int i = 1; i < matrix.Rows; i++)
        {
            for (int j = 0; j < i; j++)
            {
                // If any element below the main diagonal is non-zero (greater than tolerance), the matrix is not upper triangular.
                if (ops.GreaterThan(ops.Abs(matrix[i, j]), tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is sparse (contains mostly zero elements).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="sparsityThreshold">Optional threshold that determines how many zeros make a matrix sparse. Default is 0.5 (50%).</param>
    /// <returns>True if the matrix is sparse, otherwise false.</returns>
    /// <remarks>
    /// <para>For Beginners: A sparse matrix is one that has mostly zeros. This is important because
    /// sparse matrices can be stored and processed more efficiently using special techniques.
    /// By default, this method considers a matrix sparse if at least 50% of its elements are zero,
    /// but you can adjust this threshold.</para>
    /// </remarks>
    public static bool IsSparseMatrix<T>(this Matrix<T> matrix, T? sparsityThreshold = default)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        sparsityThreshold ??= NumOps.FromDouble(0.5);
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        int totalElements = rows * cols;
        int zeroCount = 0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (NumOps.Equals(matrix[i, j], NumOps.Zero))
                {
                    zeroCount++;
                }
            }
        }

        T sparsityRatio = NumOps.Divide(NumOps.FromDouble(zeroCount), NumOps.FromDouble(totalElements));
        return NumOps.GreaterThanOrEquals(sparsityRatio, sparsityThreshold);
    }

    /// <summary>
    /// Determines if a matrix is dense (contains mostly non-zero elements).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="denseThreshold">Optional threshold that determines how many non-zeros make a matrix dense. Default is 0.5 (50%).</param>
    /// <returns>True if the matrix is dense, otherwise false.</returns>
    /// <remarks>
    /// <para>For Beginners: A dense matrix is the opposite of a sparse matrix - it has mostly non-zero values.
    /// Dense matrices typically require different processing techniques than sparse ones.
    /// By default, this method considers a matrix dense if at least 50% of its elements are non-zero,
    /// but you can adjust this threshold.</para>
    /// </remarks>
    public static bool IsDenseMatrix<T>(this Matrix<T> matrix, T? denseThreshold = default)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        denseThreshold ??= NumOps.FromDouble(0.5);
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        int totalElements = rows * cols;
        int nonZeroCount = 0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!NumOps.Equals(matrix[i, j], NumOps.Zero))
                {
                    nonZeroCount++;
                }
            }
        }

        T densityRatio = NumOps.Divide(NumOps.FromDouble(nonZeroCount), NumOps.FromDouble(totalElements));
        return NumOps.GreaterThanOrEquals(densityRatio, denseThreshold);
    }

    /// <summary>
    /// Determines if a matrix is lower triangular (all elements above the main diagonal are zero).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="tolerance">Optional tolerance value for floating-point comparisons. Default is 1e-10.</param>
    /// <returns>True if the matrix is lower triangular, otherwise false.</returns>
    /// <remarks>
    /// <para>For Beginners: A lower triangular matrix has all its non-zero values either on or below the main diagonal
    /// (the diagonal line from top-left to bottom-right). Everything above this diagonal is zero.
    /// Like upper triangular matrices, lower triangular matrices simplify many mathematical operations.</para>
    /// <para>The tolerance parameter helps when working with decimal numbers that might have tiny rounding errors.</para>
    /// </remarks>
    public static bool IsLowerTriangularMatrix<T>(this Matrix<T> matrix, T? tolerance = default)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        tolerance ??= NumOps.FromDouble(1e-10);
        var rows = matrix.Rows;

        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                // If any element above the main diagonal is non-zero, the matrix is not lower triangular.
                if (NumOps.GreaterThan(NumOps.Abs(matrix[i, j]), tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is square (has the same number of rows and columns).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix has the same number of rows and columns; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A square matrix is simply a matrix with the same number of rows and columns.
    /// For example, a 3×3 matrix is square, while a 2×3 matrix is not.
    /// </para>
    /// </remarks>
    public static bool IsSquareMatrix<T>(this Matrix<T> matrix)
    {
        return matrix.Rows == matrix.Columns;
    }

    /// <summary>
    /// Determines if a matrix is rectangular (has a different number of rows and columns).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix has a different number of rows and columns; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A rectangular matrix has a different number of rows and columns.
    /// For example, a 2×3 matrix (2 rows, 3 columns) is rectangular.
    /// </para>
    /// </remarks>
    public static bool IsRectangularMatrix<T>(this Matrix<T> matrix)
    {
        return matrix.Rows != matrix.Columns;
    }

    /// <summary>
    /// Determines if a matrix is symmetric (equal to its transpose).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is symmetric; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A symmetric matrix is like a mirror image across its diagonal.
    /// For any position (i,j) in the matrix, the value is the same as at position (j,i).
    /// For example, if the value at row 2, column 3 is 5, then the value at row 3, column 2 must also be 5.
    /// </para>
    /// </remarks>
    public static bool IsSymmetricMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[j, i]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is diagonal (all non-diagonal elements are zero).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is diagonal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix has values only along its main diagonal (from top-left to bottom-right),
    /// and zeros everywhere else. For example:
    /// [5 0 0]
    /// [0 2 0]
    /// [0 0 9]
    /// This is a diagonal matrix because only the diagonal positions have non-zero values.
    /// </para>
    /// </remarks>
    public static bool IsDiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A diagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i != j && !ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is an identity matrix (diagonal elements are 1, all others are 0).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is an identity matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An identity matrix is a special diagonal matrix where all the diagonal values are 1.
    /// It's like the number "1" in multiplication - when you multiply any matrix by the identity matrix,
    /// you get the original matrix back unchanged. For example:
    /// [1 0 0]
    /// [0 1 0]
    /// [0 0 1]
    /// </para>
    /// </remarks>
    public static bool IsIdentityMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // An identity matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j)
                {
                    if (!ops.Equals(matrix[i, j], ops.One))
                    {
                        return false; // Diagonal elements must be 1
                    }
                }
                else
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Non-diagonal elements must be 0
                    }
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is skew-symmetric (equal to the negative of its transpose).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is skew-symmetric; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A skew-symmetric matrix is one where the value at position (i,j) is the negative
    /// of the value at position (j,i). For example, if the value at row 2, column 3 is 5, then the value at
    /// row 3, column 2 must be -5. Also, all diagonal elements must be zero because a number cannot be the
    /// negative of itself (unless it's zero).
    /// </para>
    /// </remarks>
    public static bool IsSkewSymmetricMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A skew-symmetric matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], ops.Negate(matrix[j, i])))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a scalar matrix (diagonal elements are equal, all others are 0).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a scalar matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A scalar matrix is a special type of diagonal matrix where all the diagonal values
    /// are the same. For example:
    /// [3 0 0]
    /// [0 3 0]
    /// [0 0 3]
    /// This is a scalar matrix because all diagonal elements have the same value (3) and all other elements are zero.
    /// </para>
    /// </remarks>
    public static bool IsScalarMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A scalar matrix must be square
        }

        T diagonalValue = matrix[0, 0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j)
                {
                    if (!ops.Equals(matrix[i, j], diagonalValue))
                    {
                        return false; // All diagonal elements must be equal
                    }
                }
                else
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // All off-diagonal elements must be zero
                    }
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is upper bidiagonal (non-zero elements only on main diagonal and first superdiagonal).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is upper bidiagonal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper bidiagonal matrix has non-zero values only on the main diagonal and the diagonal
    /// immediately above it (called the superdiagonal). All other elements must be zero. For example:
    /// [4 7 0]
    /// [0 2 5]
    /// [0 0 9]
    /// This is an upper bidiagonal matrix because values appear only on the main diagonal and the diagonal above it.
    /// </para>
    /// </remarks>
    public static bool IsUpperBidiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // An upper bidiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i > j || j > i + 1)
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Elements below the main diagonal or above the first superdiagonal must be zero
                    }
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a lower bidiagonal matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is lower bidiagonal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A lower bidiagonal matrix is a square matrix where all elements are zero except 
    /// for the main diagonal and the diagonal immediately below it (the first subdiagonal).
    /// For example:
    /// [a 0 0]
    /// [b c 0]
    /// [0 d e]
    /// </para>
    /// </remarks>
    public static bool IsLowerBidiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A lower bidiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i < j || i > j + 1)
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Elements above the main diagonal or below the first subdiagonal must be zero
                    }
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a tridiagonal matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is tridiagonal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tridiagonal matrix is a square matrix where all elements are zero except 
    /// for the main diagonal and the diagonals immediately above and below it.
    /// For example:
    /// [a b 0]
    /// [c d e]
    /// [0 f g]
    /// </para>
    /// </remarks>
    public static bool IsTridiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A tridiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(i - j) > 1 && !ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false; // Elements outside the three diagonals must be zero
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a band matrix with specified sub-diagonal and super-diagonal thresholds.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="subDiagonalThreshold">The number of sub-diagonals to consider (default is 1).</param>
    /// <param name="superDiagonalThreshold">The number of super-diagonals to consider (default is 1).</param>
    /// <returns>True if the matrix is a band matrix with the specified thresholds; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A band matrix is a square matrix where all elements are zero except for 
    /// those on the main diagonal and a specific number of diagonals above and below it.
    /// The parameters let you specify how many diagonals above and below the main diagonal can have non-zero values.
    /// </para>
    /// </remarks>
    public static bool IsBandMatrix<T>(this Matrix<T> matrix, int subDiagonalThreshold = 1, int superDiagonalThreshold = 1)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A band matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(i - j) > subDiagonalThreshold || Math.Abs(i - j) > superDiagonalThreshold)
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Elements outside the specified bands must be zero
                    }
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a complex matrix is Hermitian (equal to its conjugate transpose).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The complex matrix to check.</param>
    /// <returns>True if the matrix is Hermitian; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hermitian matrix is a complex square matrix that equals its own conjugate transpose.
    /// The conjugate transpose means you flip the matrix along its main diagonal and take the complex conjugate 
    /// of each element (change the sign of the imaginary part). Hermitian matrices are the complex equivalent 
    /// of symmetric matrices in real numbers.
    /// </para>
    /// </remarks>
    public static bool IsHermitianMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A Hermitian matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != matrix[j, i].Conjugate())
                {
                    return false; // Check if element is equal to its conjugate transpose
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a complex matrix is skew-Hermitian (equal to the negative of its conjugate transpose).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The complex matrix to check.</param>
    /// <returns>True if the matrix is skew-Hermitian; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A skew-Hermitian matrix is a complex square matrix that equals the negative of its 
    /// conjugate transpose. This means when you flip the matrix along its main diagonal, take the complex conjugate 
    /// of each element, and then negate all values, you get back the original matrix. The diagonal elements of a 
    /// skew-Hermitian matrix must be purely imaginary or zero.
    /// </para>
    /// </remarks>
    public static bool IsSkewHermitianMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A skew-Hermitian matrix must be square
        }

        var ops = MathHelper.GetNumericOperations<Complex<T>>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], ops.Negate(matrix[j, i].Conjugate())))
                {
                    return false; // Check if element is equal to the negation of its conjugate transpose
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is orthogonal (its transpose equals its inverse).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="matrixDecomposition">The matrix decomposition to use for calculating the inverse.</param>
    /// <returns>True if the matrix is orthogonal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An orthogonal matrix is a square matrix whose transpose equals its inverse.
    /// This means that when you multiply an orthogonal matrix by its transpose, you get the identity matrix.
    /// Orthogonal matrices preserve lengths and angles when used for transformations, making them useful
    /// in computer graphics, physics, and data analysis.
    /// </para>
    /// </remarks>
    public static bool IsOrthogonalMatrix<T>(this Matrix<T> matrix, IMatrixDecomposition<T> matrixDecomposition)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // An orthogonal matrix must be square
        }

        var transpose = matrix.Transpose();
        var inverse = matrixDecomposition.Invert();

        var ops = MathHelper.GetNumericOperations<T>();

        // Check if transpose is equal to inverse
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (ops.GreaterThan(ops.Abs(ops.Subtract(transpose[i, j], inverse[i, j])), ops.FromDouble(0.0001)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Creates a new complex matrix with the specified dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The source matrix (used for type inference).</param>
    /// <param name="rows">The number of rows for the new matrix.</param>
    /// <param name="cols">The number of columns for the new matrix.</param>
    /// <returns>A new complex matrix with the specified dimensions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method creates a new empty complex matrix with the dimensions you specify.
    /// It's useful when you need to create a result matrix for operations on complex matrices.
    /// </para>
    /// </remarks>
    public static Matrix<Complex<T>> CreateComplexMatrix<T>(this Matrix<Complex<T>> matrix, int rows, int cols)
    {
        return new Matrix<Complex<T>>(rows, cols);
    }

    /// <summary>
    /// Computes the conjugate transpose (also known as Hermitian transpose) of a complex matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the real and imaginary parts of the complex numbers.</typeparam>
    /// <param name="matrix">The complex matrix to transpose and conjugate.</param>
    /// <returns>A new matrix that is the conjugate transpose of the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The conjugate transpose of a complex matrix is created by:
    /// 1. Flipping the matrix over its diagonal (exchanging rows and columns)
    /// 2. Taking the complex conjugate of each element (reversing the sign of the imaginary part)
    /// 
    /// For example, if you have a matrix:
    /// [a+bi  c+di]
    /// [e+fi  g+hi]
    /// 
    /// The conjugate transpose would be:
    /// [a-bi  e-fi]
    /// [c-di  g-hi]
    /// 
    /// This operation is important in quantum computing and signal processing applications.
    /// </para>
    /// </remarks>
    public static Matrix<Complex<T>> ConjugateTranspose<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var result = matrix.CreateComplexMatrix(cols, rows);

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[i, j] = matrix[j, i].Conjugate();
            }
        }

        return result;
    }

    /// <summary>
    /// Determines if a matrix is singular (non-invertible).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is singular; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A singular matrix is a square matrix that doesn't have an inverse.
    /// This happens when the determinant of the matrix is zero. In practical terms, singular
    /// matrices represent transformations that collapse dimensions (like projecting 3D onto a plane),
    /// making it impossible to reverse the transformation.
    /// 
    /// Singular matrices cause problems in many algorithms because they can't be inverted,
    /// which is why it's important to check for this condition.
    /// </para>
    /// </remarks>
    public static bool IsSingularMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A singular matrix must be square
        }

        // Calculate the determinant
        T determinant = matrix.GetDeterminant();

        // If determinant is zero, the matrix is singular
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.LessThan(ops.Abs(determinant), ops.FromDouble(1e-10));
    }

    /// <summary>
    /// Determines if a matrix is non-singular (invertible).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is non-singular; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A non-singular matrix is a square matrix that has an inverse.
    /// This happens when the determinant of the matrix is not zero. Non-singular matrices
    /// represent transformations that can be reversed, like rotating or scaling in ways
    /// that preserve all dimensions.
    /// 
    /// Having a non-singular matrix is often a requirement for solving systems of equations
    /// and many other mathematical operations in AI and machine learning.
    /// </para>
    /// </remarks>
    public static bool IsNonSingularMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A singular matrix must be square
        }

        // Calculate the determinant
        T determinant = matrix.GetDeterminant();

        // If determinant is not zero, the matrix is non-singular
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.GreaterThanOrEquals(ops.Abs(determinant), ops.FromDouble(1e-10));
    }

    /// <summary>
    /// Determines if a matrix is positive definite.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="tolerance">Optional tolerance value for numerical stability. Default is 1e-10.</param>
    /// <returns>True if the matrix is positive definite; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A positive definite matrix is a special type of symmetric matrix where:
    /// 1. The matrix must be symmetric (equal to its transpose)
    /// 2. All eigenvalues of the matrix are positive
    /// 
    /// In simpler terms, when you multiply this matrix by any non-zero vector, the result points
    /// in a direction that makes a positive angle with the original vector.
    /// 
    /// Positive definite matrices are important in machine learning for:
    /// - Covariance matrices in statistics
    /// - Kernel methods like Support Vector Machines
    /// - Optimization problems where we need to ensure a unique minimum exists
    /// 
    /// This method uses Cholesky decomposition to check for positive definiteness, which is
    /// more efficient than calculating eigenvalues directly.
    /// </para>
    /// </remarks>
    public static bool IsPositiveDefiniteMatrix<T>(this Matrix<T> matrix, T? tolerance = default)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive definite matrices must be symmetric
        }

        var ops = MathHelper.GetNumericOperations<T>();

        // Set default tolerance if not provided
        tolerance ??= ops.FromDouble(1e-10);

        try
        {
            // Attempt Cholesky decomposition
            var choleskyDecomposition = new CholeskyDecomposition<T>(matrix);

            // Get the lower triangular matrix L
            var L = choleskyDecomposition.L;

            // Check if all diagonal elements of L are positive
            for (int i = 0; i < L.Rows; i++)
            {
                if (ops.LessThanOrEquals(L[i, i], tolerance))
                {
                    return false;
                }
            }

            return true;
        }
        catch (Exception)
        {
            // If Cholesky decomposition fails, the matrix is not positive definite
            return false;
        }
    }

    /// <summary>
    /// Determines if a matrix is idempotent (equal to its own square).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is idempotent; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An idempotent matrix is a matrix that, when multiplied by itself,
    /// gives the same matrix: A² = A.
    /// 
    /// This property is important in:
    /// - Projection matrices in linear algebra
    /// - Statistical operations like hat matrices in regression
    /// - Machine learning algorithms that involve projections onto subspaces
    /// 
    /// Examples of idempotent matrices include identity matrices and projection matrices.
    /// </para>
    /// </remarks>
    public static bool IsIdempotentMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var product = matrix.Multiply(matrix);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if (ops.GreaterThan(ops.Abs(ops.Subtract(matrix[i, j], product[i, j])), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is stochastic (each row sums to 1 and all elements are non-negative).
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is stochastic; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A stochastic matrix (also called a probability matrix or Markov matrix)
    /// has two key properties:
    /// 1. All elements are non-negative (= 0)
    /// 2. The sum of each row equals 1
    /// 
    /// These matrices are used to represent transition probabilities in Markov chains, where:
    /// - Each row represents a current state
    /// - Each column represents a possible next state
    /// - Each element represents the probability of transitioning from one state to another
    /// 
    /// Stochastic matrices are fundamental in:
    /// - Markov processes
    /// - PageRank algorithm (used by Google)
    /// - Natural language processing
    /// - Reinforcement learning
    /// </para>
    /// </remarks>
    public static bool IsStochasticMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            T rowSum = ops.Zero;
            for (int j = 0; j < cols; j++)
            {
                if (ops.LessThan(matrix[i, j], ops.Zero))
                {
                    return false; // All elements must be non-negative
                }
                rowSum = ops.Add(rowSum, matrix[i, j]);
            }
            if (ops.GreaterThan(ops.Abs(ops.Subtract(rowSum, ops.One)), ops.FromDouble(1e-10)))
            {
                return false; // Each row must sum to 1
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a doubly stochastic matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is doubly stochastic; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A doubly stochastic matrix is a square matrix where:
    /// 1. All elements are non-negative (greater than or equal to zero)
    /// 2. The sum of elements in each row equals 1
    /// 3. The sum of elements in each column equals 1
    /// 
    /// These matrices are commonly used in probability theory and Markov chains to represent 
    /// transition probabilities between states.
    /// </para>
    /// </remarks>
    public static bool IsDoublyStochasticMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if all elements are non-negative and rows sum to 1
        for (int i = 0; i < rows; i++)
        {
            T rowSum = ops.Zero;
            for (int j = 0; j < cols; j++)
            {
                if (ops.LessThan(matrix[i, j], ops.Zero))
                {
                    return false; // All elements must be non-negative
                }
                rowSum = ops.Add(rowSum, matrix[i, j]);
            }
            if (ops.GreaterThan(ops.Abs(ops.Subtract(rowSum, ops.One)), ops.FromDouble(1e-10)))
            {
                return false; // Each row must sum to 1
            }
        }

        // Check if columns sum to 1
        for (int j = 0; j < cols; j++)
        {
            T colSum = ops.Zero;
            for (int i = 0; i < rows; i++)
            {
                colSum = ops.Add(colSum, matrix[i, j]);
            }
            if (ops.GreaterThan(ops.Abs(ops.Subtract(colSum, ops.One)), ops.FromDouble(1e-10)))
            {
                return false; // Each column must sum to 1
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is an adjacency matrix representing a graph.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is an adjacency matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An adjacency matrix represents connections between nodes in a graph:
    /// - The matrix must be square (same number of rows and columns)
    /// - Each element must be either 0 or 1
    /// - A value of 1 at position [i,j] means there is a connection from node i to node j
    /// - A value of 0 means there is no connection
    /// 
    /// This is a fundamental way to represent relationships between objects in computer science.
    /// </para>
    /// </remarks>
    public static bool IsAdjacencyMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (rows != cols)
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Check if the element is either 0 or 1
                if (!ops.Equals(matrix[i, j], ops.Zero) && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a circulant matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is circulant; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A circulant matrix is a special square matrix where:
    /// 1. Each row is a circular shift of the row above it
    /// 2. The first row defines the entire matrix
    /// 3. Each subsequent row shifts the elements one position to the right
    /// 
    /// For example, if the first row is [a, b, c], the circulant matrix would be:
    /// [a, b, c]
    /// [c, a, b]
    /// [b, c, a]
    /// 
    /// These matrices have special properties that make them useful in signal processing and solving certain types of equations.
    /// </para>
    /// </remarks>
    public static bool IsCirculantMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if each row is a cyclic shift of the previous row
        for (int i = 1; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[0, (j + i) % cols]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix can be considered a partitioned matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix can be partitioned; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A partitioned matrix is a matrix that can be divided into smaller submatrices.
    /// This method checks if the number of rows is divisible by the square root of the number of columns,
    /// which is one way to determine if a matrix can be neatly partitioned into equal-sized blocks.
    /// 
    /// Partitioned matrices are useful in block matrix operations and can simplify complex matrix calculations.
    /// </para>
    /// </remarks>
    public static bool IsPartitionedMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        if (rows == 0 || cols == 0)
        {
            return false;
        }

        return rows % (int)Math.Sqrt(cols) == 0;
    }

    /// <summary>
    /// Determines if a matrix is a Vandermonde matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a Vandermonde matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Vandermonde matrix is a special matrix where:
    /// 1. The first column can contain any values
    /// 2. Each subsequent column is formed by raising the corresponding element in the first column to a power
    /// 
    /// For example, if the first column is [x1, x2, x3], the Vandermonde matrix would be:
    /// [x1°, x1¹, x1², ...]
    /// [x2°, x2¹, x2², ...]
    /// [x3°, x3¹, x3², ...]
    /// 
    /// These matrices are important in polynomial interpolation and solving systems of linear equations.
    /// </para>
    /// </remarks>
    public static bool IsVandermondeMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix has at least 2 rows and if each column forms a geometric progression
        for (int j = 1; j < cols; j++)
        {
            for (int i = 1; i < rows; i++)
            {
                // Check if the current element is equal to the previous element multiplied by x_i
                T expectedValue = ops.Multiply(matrix[i - 1, j], matrix[i, 0]);
                if (ops.GreaterThan(ops.Abs(ops.Subtract(matrix[i, j], expectedValue)), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a Cauchy matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a Cauchy matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Cauchy matrix is formed from two sequences of numbers (x1, x2, ...) and (y1, y2, ...).
    /// Each element at position [i,j] equals 1/(x_i - y_j).
    /// 
    /// For this implementation:
    /// - The x values are taken from the first column of the matrix
    /// - The y values are taken from the first row of the matrix
    /// - The method checks if each element follows the Cauchy formula
    /// 
    /// Cauchy matrices have applications in interpolation problems and numerical analysis.
    /// </para>
    /// </remarks>
    public static bool IsCauchyMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if each element satisfies the Cauchy matrix definition
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                T element = matrix[i, j];
                T x = matrix[i, 0];
                T y = matrix[0, j];

                // Avoid division by zero
                if (ops.LessThan(ops.Abs(ops.Subtract(x, y)), ops.FromDouble(1e-10)))
                {
                    return false;
                }

                T expectedValue = ops.Divide(ops.One, ops.Subtract(x, y));
                if (ops.GreaterThan(ops.Abs(ops.Subtract(element, expectedValue)), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a Hilbert matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a Hilbert matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hilbert matrix is a special square matrix where each element at position (i,j) 
    /// is defined as 1/(i+j+1). These matrices are important in numerical analysis but are known to be 
    /// difficult to work with because they become increasingly ill-conditioned as their size grows.
    /// </para>
    /// </remarks>
    public static bool IsHilbertMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if the elements satisfy the Hilbert matrix definition
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                T expectedValue = ops.Divide(ops.One, ops.FromDouble(i + j + 1)); // Hilbert matrix definition
                if (ops.GreaterThan(ops.Abs(ops.Subtract(matrix[i, j], expectedValue)), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a companion matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a companion matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A companion matrix is a special square matrix that helps solve polynomial equations.
    /// It has a specific structure where:
    /// 1. The first row contains coefficients of a polynomial
    /// 2. The subdiagonal (the diagonal just below the main diagonal) contains all 1's
    /// 3. All other elements are 0
    /// 
    /// Companion matrices are useful for finding roots of polynomials and in control theory.
    /// </para>
    /// </remarks>
    public static bool IsCompanionMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if the first row contains the coefficients of a polynomial in reverse order
        for (int j = 0; j < cols - 1; j++)
        {
            if (!ops.Equals(matrix[0, j], ops.Zero) && !ops.Equals(matrix[0, j], ops.One))
            {
                return false;
            }
        }
        if (!ops.Equals(matrix[0, cols - 1], ops.One))
        {
            return false;
        }

        // Check if each subdiagonal contains a 1
        for (int i = 1; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j + 1 && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }
                if (i != j + 1 && !ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines whether a matrix is invertible.
    /// </summary>
    /// <typeparam name="T">The numeric type used for matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is invertible; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if a matrix is invertible by verifying that it is square and
    /// calculating its determinant. If the determinant is zero (or very close to zero for floating-point types),
    /// the matrix is not invertible. A matrix is invertible if and only if its determinant is non-zero.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if you can find the inverse of a matrix.
    /// 
    /// A matrix is invertible when:
    /// - It is square (same number of rows and columns)
    /// - Its determinant is not zero
    /// 
    /// The inverse of a matrix is like the reciprocal of a number (1/x).
    /// Not all matrices have inverses, just like division by zero is not allowed.
    /// 
    /// This method is important for solving systems of linear equations
    /// and is used in many machine learning algorithms.
    /// </para>
    /// </remarks>
    public static bool IsInvertible<T>(this Matrix<T> matrix)
    {
        // Get numeric operations for the type T
        var numOps = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (matrix.Rows != matrix.Columns)
        {
            return false;
        }

        try
        {
            // For small matrices (2x2 or 3x3), calculating the determinant directly is efficient
            if (matrix.Rows <= 3)
            {
                T determinant = matrix.Determinant();

                // Check if determinant is zero or very close to zero
                if (numOps.Equals(determinant, numOps.Zero) ||
                    MathHelper.AlmostEqual(determinant, numOps.Zero))
                {
                    return false;
                }

                return true;
            }

            // For larger matrices, another approach is to try an LU decomposition
            // If successful, the matrix is invertible
            var decomposition = new LuDecomposition<T>(matrix);
            var (l, u, p) = (decomposition.L, decomposition.U, decomposition.P);

            // Check the diagonal elements of U matrix
            // If any diagonal element is zero, the matrix is not invertible
            for (int i = 0; i < u.Rows; i++)
            {
                if (numOps.Equals(u[i, i], numOps.Zero) ||
                    MathHelper.AlmostEqual(u[i, i], numOps.Zero))
                {
                    return false;
                }
            }

            return true;
        }
        catch (Exception)
        {
            // If any calculation fails (e.g., due to numerical instability),
            // be conservative and assume the matrix is not invertible
            return false;
        }
    }

    /// <summary>
    /// Determines if a matrix is a Hankel matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a Hankel matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hankel matrix is a matrix where each skew-diagonal (running from bottom-left to top-right) 
    /// contains the same value. In other words, the value at position (i,j) depends only on the sum i+j.
    /// 
    /// Hankel matrices appear in signal processing, control theory, and when solving certain types of differential equations.
    /// </para>
    /// </remarks>
    public static bool IsHankelMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if each element is the same as the one diagonally above and to the right of it
        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = 0; j < cols - 1; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[i + 1, j + 1]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a Toeplitz matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a Toeplitz matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Toeplitz matrix is a matrix where each descending diagonal from left to right 
    /// contains the same value. In other words, the value at position (i,j) depends only on the difference i-j.
    /// 
    /// Toeplitz matrices are common in signal processing and solving differential equations. They have special 
    /// properties that make certain calculations more efficient.
    /// </para>
    /// </remarks>
    public static bool IsToeplitzMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if each element is the same as the one diagonally above it
        for (int i = 1; i < rows; i++)
        {
            for (int j = 1; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[i - 1, j - 1]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a Laplacian matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a Laplacian matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Laplacian matrix represents a graph and has special properties:
    /// 1. It's symmetric (mirror image across the diagonal)
    /// 2. Off-diagonal elements are non-positive (zero or negative)
    /// 3. Each row and column sums to zero
    /// 4. Each diagonal element equals the sum of the absolute values of the off-diagonal elements in its row
    /// 
    /// Laplacian matrices are used in graph theory, network analysis, and spectral clustering algorithms.
    /// </para>
    /// </remarks>
    public static bool IsLaplacianMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            T rowSum = ops.Zero;
            T colSum = ops.Zero;

            for (int j = 0; j < rows; j++)
            {
                // Check for symmetry
                if (!ops.Equals(matrix[i, j], matrix[j, i]))
                {
                    return false;
                }

                // Check if off-diagonal elements are non-positive
                if (i != j && ops.GreaterThan(matrix[i, j], ops.Zero))
                {
                    return false;
                }

                // Sum the row and column elements
                rowSum = ops.Add(rowSum, matrix[i, j]);
                colSum = ops.Add(colSum, matrix[j, i]);
            }

            // Check if the sum of each row and column is zero
            if (ops.GreaterThan(ops.Abs(rowSum), ops.FromDouble(1e-10)) ||
                ops.GreaterThan(ops.Abs(colSum), ops.FromDouble(1e-10)))
            {
                return false;
            }

            // Check if diagonal elements are non-negative and equal to the sum of absolute values of the off-diagonal elements
            T offDiagonalSum = ops.Zero;
            for (int j = 0; j < rows; j++)
            {
                if (i != j)
                {
                    offDiagonalSum = ops.Add(offDiagonalSum, ops.Abs(matrix[i, j]));
                }
            }
            if (!ops.Equals(matrix[i, i], offDiagonalSum))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is an incidence matrix for an undirected graph.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is an incidence matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An incidence matrix represents the relationship between vertices (rows) and edges (columns) 
    /// in a graph. For an undirected graph:
    /// 1. Each element is either 0 or 1
    /// 2. Each column has exactly two 1's (representing the two vertices connected by that edge)
    /// 3. All other elements are 0
    /// 
    /// Incidence matrices are used in graph theory and network analysis to represent connections between nodes.
    /// </para>
    /// </remarks>
    public static bool IsIncidenceMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows; // number of vertices
        var cols = matrix.Columns; // number of edges
        var ops = MathHelper.GetNumericOperations<T>();

        for (int j = 0; j < cols; j++)
        {
            int countOnes = 0;

            for (int i = 0; i < rows; i++)
            {
                // Check if the element is either 0 or 1
                if (!ops.Equals(matrix[i, j], ops.Zero) && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }

                // Count the number of 1's in the current column
                if (ops.Equals(matrix[i, j], ops.One))
                {
                    countOnes++;
                }
            }

            // Each column must have exactly two 1's for an undirected graph
            if (countOnes != 2)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is a permutation matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is a permutation matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A permutation matrix is a special square matrix that has exactly one entry of 1 in each row and each column, 
    /// with all other entries being 0. These matrices are used to represent rearrangements (permutations) of elements in a vector.
    /// </para>
    /// </remarks>
    public static bool IsPermutationMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            int rowCount = 0;
            int colCount = 0;
            for (int j = 0; j < cols; j++)
            {
                // Check if the element is either 0 or 1
                if (!ops.Equals(matrix[i, j], ops.Zero) && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }

                // Count the number of 1's in the current row
                if (ops.Equals(matrix[i, j], ops.One))
                {
                    rowCount++;
                }

                // Count the number of 1's in the current column
                if (ops.Equals(matrix[j, i], ops.One))
                {
                    colCount++;
                }
            }

            // Each row and column must have exactly one 1
            if (rowCount != 1 || colCount != 1)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is an involutory matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is an involutory matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An involutory matrix is a matrix that, when multiplied by itself, gives the identity matrix.
    /// In other words, it's its own inverse (A² = I). These matrices are useful in various applications including cryptography
    /// and computer graphics.
    /// </para>
    /// </remarks>
    public static bool IsInvolutoryMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }
        var product = matrix.Multiply(matrix);

        return product.IsIdentityMatrix();
    }

    /// <summary>
    /// Determines if a matrix is an orthogonal projection matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is an orthogonal projection matrix; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An orthogonal projection matrix is a matrix that projects vectors onto a subspace.
    /// It has two key properties: it's symmetric (equal to its transpose) and idempotent (multiplying it by itself 
    /// gives the same matrix). In simpler terms, it's used to "flatten" data onto a lower-dimensional space while 
    /// preserving as much information as possible.
    /// </para>
    /// </remarks>
    public static bool IsOrthogonalProjectionMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false;
        }

        if (!matrix.IsIdempotentMatrix())
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Determines if a matrix is positive semi-definite.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if the matrix is positive semi-definite; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A positive semi-definite matrix is a symmetric matrix where all eigenvalues are non-negative.
    /// These matrices are important in machine learning, statistics, and optimization problems. They represent covariance 
    /// matrices, kernel matrices in kernel methods, and Hessian matrices in certain optimization problems. A key property 
    /// is that for any vector x, x^T*A*x = 0, which means these matrices preserve or increase vector lengths in certain directions.
    /// </para>
    /// </remarks>
    public static bool IsPositiveSemiDefiniteMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive semi-definite matrices must be symmetric
        }

        var eigenvalues = matrix.Eigenvalues();
        var ops = MathHelper.GetNumericOperations<T>();
        // Check if all eigenvalues are non-negative
        foreach (var eigenvalue in eigenvalues)
        {
            if (ops.LessThan(eigenvalue, ops.Zero))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Calculates the eigenvalues of a matrix using the QR algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix for which to calculate eigenvalues.</param>
    /// <returns>A vector containing the eigenvalues of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Eigenvalues are special scalars associated with a matrix. When the matrix is multiplied by certain vectors 
    /// (called eigenvectors), the result is the same as multiplying those vectors by the eigenvalue. Eigenvalues are crucial in many 
    /// applications including principal component analysis, vibration analysis, and quantum mechanics. This method uses the QR algorithm, 
    /// which is an iterative approach to find eigenvalues of a matrix.
    /// </para>
    /// </remarks>
    public static Vector<T> Eigenvalues<T>(this Matrix<T> matrix)
    {
        // QR algorithm for finding eigenvalues of a symmetric matrix
        var rows = matrix.Rows;
        var a = new Matrix<T>(rows, rows);

        // Copy the matrix data manually
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                a[i, j] = matrix[i, j];
            }
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var eigenvalues = new Vector<T>(rows);

        for (int k = rows - 1; k > 0; k--)
        {
            while (ops.GreaterThan(ops.Abs(a[k, k - 1]), ops.FromDouble(1e-10))) // Replace ops.Epsilon with a small value
            {
                T mu = a[k, k];
                for (int i = 0; i <= k; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        a[i, j] = ops.Subtract(a[i, j], ops.Multiply(ops.Multiply(mu, a[k, i]), a[k, j]));
                    }
                }

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < rows; j++)
                    {
                        a[i, j] = ops.Subtract(a[i, j], ops.Multiply(ops.Multiply(mu, a[i, k]), a[j, k]));
                    }
                }
            }

            eigenvalues[k] = a[k, k];
            for (int i = 0; i <= k; i++)
            {
                a[k, i] = a[i, k] = ops.Zero;
            }
        }
        eigenvalues[0] = a[0, 0];

        return eigenvalues;
    }

    /// <summary>
    /// Calculates the determinant of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix for which to calculate the determinant.</param>
    /// <returns>The determinant of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The determinant is a special number calculated from a square matrix. It tells you important 
    /// information about the matrix, such as whether it has an inverse (when determinant is not zero). Geometrically, 
    /// the determinant represents how much the matrix scales volumes. This method uses a recursive approach called 
    /// cofactor expansion to calculate the determinant.
    /// </para>
    /// </remarks>
    public static T GetDeterminant<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Base case: for 1x1 matrix, determinant is the single element
        if (rows == 1)
        {
            return matrix[0, 0];
        }

        T det = ops.Zero;
        // Recursive case: compute the determinant using cofactor expansion
        for (int j = 0; j < rows; j++)
        {
            // Calculate the cofactor of matrix[0, j]
            var submatrix = Matrix<T>.CreateMatrix<T>(rows - 1, rows - 1);
            for (int i = 1; i < rows; i++)
            {
                for (int k = 0; k < rows; k++)
                {
                    if (k < j)
                    {
                        submatrix[i - 1, k] = matrix[i, k];
                    }
                    else if (k > j)
                    {
                        submatrix[i - 1, k - 1] = matrix[i, k];
                    }
                }
            }

            // Add the cofactor to the determinant
            T sign = ops.FromDouble(Math.Pow(-1, j));
            T cofactor = ops.Multiply(sign, matrix[0, j]);
            T subDet = submatrix.GetDeterminant();
            det = ops.Add(det, ops.Multiply(cofactor, subDet));
        }

        return det;
    }

    /// <summary>
    /// Inverts an upper triangular matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The upper triangular matrix to invert.</param>
    /// <returns>The inverted upper triangular matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper triangular matrix is a square matrix where all elements below the main diagonal are zero.
    /// Inverting a matrix means finding another matrix that, when multiplied with the original matrix, gives the identity matrix.
    /// This method provides a specialized, efficient way to invert upper triangular matrices.
    /// </para>
    /// </remarks>
    public static Matrix<T> InvertUpperTriangularMatrix<T>(this Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Create the inverse matrix
        var inverse = Matrix<T>.CreateMatrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i <= j)
                {
                    inverse[i, j] = ops.Divide(ops.One, matrix[i, j]);
                }
                else
                {
                    inverse[i, j] = ops.Zero;
                }
            }
        }

        return inverse;
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using forward substitution, where A is a lower triangular matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix and vector elements.</typeparam>
    /// <param name="aMatrix">The lower triangular coefficient matrix A.</param>
    /// <param name="bVector">The right-hand side vector b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forward substitution is a method for solving a system of linear equations where the coefficient matrix
    /// is lower triangular (all elements above the main diagonal are zero). The method works by solving for each variable in sequence,
    /// starting from the first equation and using previously computed values.
    /// </para>
    /// </remarks>
    public static Vector<T> ForwardSubstitution<T>(this Matrix<T> aMatrix, Vector<T> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var x = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            x[i] = bVector[i];
            for (int j = 0; j < i; j++)
            {
                x[i] = ops.Subtract(x[i], ops.Multiply(aMatrix[i, j], x[j]));
            }

            x[i] = ops.Divide(x[i], aMatrix[i, i]);
        }

        return x;
    }

    /// <summary>
    /// Converts a real-valued matrix to a complex-valued matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The real-valued matrix to convert.</param>
    /// <returns>A complex-valued matrix where each element has the original value as its real part and zero as its imaginary part.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A complex number has two parts: a real part and an imaginary part. This method takes a matrix of real numbers
    /// and creates a new matrix where each element is a complex number with the original value as the real part and zero as the imaginary part.
    /// This is useful when you need to perform operations that require complex numbers.
    /// </para>
    /// </remarks>
    public static Matrix<Complex<T>> ToComplexMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var complexMatrix = Matrix<Complex<T>>.CreateMatrix<Complex<T>>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                complexMatrix[i, j] = new Complex<T>(matrix[i, j], ops.Zero);
            }
        }

        return complexMatrix;
    }

    /// <summary>
    /// Converts a real-valued vector to a complex-valued vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The real-valued vector to convert.</param>
    /// <returns>A complex-valued vector where each element has the original value as its real part and zero as its imaginary part.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Similar to converting a matrix to complex form, this method takes a vector of real numbers
    /// and creates a new vector where each element is a complex number with the original value as the real part and zero as the imaginary part.
    /// This is useful when you need to perform operations that require complex numbers.
    /// </para>
    /// </remarks>
    public static Vector<Complex<T>> ToComplexVector<T>(this Vector<T> vector)
    {
        var count = vector.Length;
        var ops = MathHelper.GetNumericOperations<T>();
        var complexVector = new Vector<Complex<T>>(count);

        for (int i = 0; i < count; i++)
        {
            complexVector[i] = new Complex<T>(vector[i], ops.Zero);
        }

        return complexVector;
    }

    /// <summary>
    /// Extracts the real part of a complex-valued matrix to create a real-valued matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The complex-valued matrix.</param>
    /// <returns>A real-valued matrix containing only the real parts of the complex matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a matrix of complex numbers and creates a new matrix containing only the real parts
    /// of those complex numbers. The imaginary parts are discarded. This is useful when you've performed calculations with complex numbers
    /// but only need the real results.
    /// </para>
    /// </remarks>
    public static Matrix<T> ToRealMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        var rows = matrix.Rows;
        var realMatrix = new Matrix<T>(rows, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                realMatrix[i, j] = matrix[i, j].Real;
            }
        }

        return realMatrix;
    }

    /// <summary>
    /// Inverts a lower triangular matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The lower triangular matrix to invert.</param>
    /// <returns>The inverted lower triangular matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A lower triangular matrix is a square matrix where all elements above the main diagonal are zero.
    /// Inverting a matrix means finding another matrix that, when multiplied with the original matrix, gives the identity matrix.
    /// This method provides a specialized, efficient way to invert lower triangular matrices.
    /// </para>
    /// </remarks>
    public static Matrix<T> InvertLowerTriangularMatrix<T>(this Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var invL = Matrix<T>.CreateMatrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            invL[i, i] = ops.Divide(ops.One, matrix[i, i]);
            for (int j = 0; j < i; j++)
            {
                T sum = ops.Zero;
                for (int k = j; k < i; k++)
                {
                    sum = ops.Add(sum, ops.Multiply(matrix[i, k], invL[k, j]));
                }
                invL[i, j] = ops.Negate(ops.Divide(sum, matrix[i, i]));
            }
        }

        return invL;
    }

    /// <summary>
    /// Inverts a matrix using the Gaussian-Jordan elimination method.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to invert.</param>
    /// <returns>The inverted matrix.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the matrix is singular and cannot be inverted.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix inversion is like finding the reciprocal of a number. For example, the reciprocal of 2 is 1/2.
    /// Similarly, the inverse of a matrix A is another matrix that, when multiplied with A, gives the identity matrix (similar to how 2 × 1/2 = 1).
    /// The Gaussian-Jordan elimination is a step-by-step process to find this inverse by transforming the original matrix into the identity matrix.
    /// </para>
    /// </remarks>
    public static Matrix<T> InverseGaussianJordanElimination<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var augmentedMatrix = new Matrix<T>(rows, 2 * rows);

        // Copy matrix into the left half of the augmented matrix
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                augmentedMatrix[i, j] = matrix[i, j];
            }
        }

        // Add identity matrix to the right half of the augmented matrix
        for (int i = 0; i < rows; i++)
        {
            augmentedMatrix[i, i + rows] = ops.One;
        }

        // Perform Gaussian elimination with partial pivoting
        for (int i = 0; i < rows; i++)
        {
            // Find pivot row
            int maxRowIndex = i;
            T maxValue = ops.Abs(augmentedMatrix[i, i]);
            for (int k = i + 1; k < rows; k++)
            {
                T absValue = ops.Abs(augmentedMatrix[k, i]);
                if (ops.GreaterThan(absValue, maxValue))
                {
                    maxRowIndex = k;
                    maxValue = absValue;
                }
            }

            // Check for singularity
            if (ops.Equals(maxValue, ops.Zero))
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
            }

            // Swap current row with pivot row
            if (maxRowIndex != i)
            {
                for (int j = 0; j < 2 * rows; j++)
                {
                    (augmentedMatrix[maxRowIndex, j], augmentedMatrix[i, j]) = (augmentedMatrix[i, j], augmentedMatrix[maxRowIndex, j]);
                }
            }

            // Make diagonal element 1
            T pivot = augmentedMatrix[i, i];
            for (int j = 0; j < 2 * rows; j++)
            {
                augmentedMatrix[i, j] = ops.Divide(augmentedMatrix[i, j], pivot);
            }

            // Make other elements in the column zero
            for (int k = 0; k < rows; k++)
            {
                if (k != i)
                {
                    T factor = augmentedMatrix[k, i];
                    for (int j = 0; j < 2 * rows; j++)
                    {
                        augmentedMatrix[k, j] = ops.Subtract(augmentedMatrix[k, j], ops.Multiply(factor, augmentedMatrix[i, j]));
                    }
                }
            }
        }

        // Extract the right half of the augmented matrix (the inverse)
        var inverseMatrix = new Matrix<T>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                inverseMatrix[i, j] = augmentedMatrix[i, j + rows];
            }
        }

        return inverseMatrix;
    }

    /// <summary>
    /// Extracts a submatrix of specified dimensions from the top-left corner of the original matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="rows">The number of rows to extract.</param>
    /// <param name="columns">The number of columns to extract.</param>
    /// <returns>A new matrix containing the extracted elements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a smaller matrix from a larger one by taking only the specified number of rows and columns
    /// from the top-left corner. Think of it like cropping a photo to focus on just one part of the image.
    /// </para>
    /// </remarks>
    public static Matrix<T> Extract<T>(this Matrix<T> matrix, int rows, int columns)
    {
        var result = Matrix<T>.CreateMatrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the nullity of a matrix, which is the dimension of its null space.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="threshold">Optional threshold value for determining when a value is considered zero. If not provided, a default threshold is calculated.</param>
    /// <returns>The nullity of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The nullity of a matrix tells you how many independent ways there are to get zero when multiplying the matrix by a vector.
    /// In practical terms, it helps identify how much "redundant" information is in your data. A higher nullity means more redundancy or dependency in your data.
    /// </para>
    /// </remarks>
    public static int GetNullity<T>(this Matrix<T> matrix, T? threshold = default)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns);
        var _epsilon = ops.FromDouble(1e-10); // Small number instead of Epsilon
        var _threshold = threshold != null && ops.GreaterThanOrEquals(threshold, ops.Zero)
            ? threshold
            : ops.Multiply(
                ops.FromDouble(0.5 * Math.Sqrt(rows + columns + 1)),
                ops.Multiply(weightsVector[0], _epsilon)
            );
        int nullity = 0;

        for (int i = 0; i < columns; i++)
        {
            if (ops.LessThanOrEquals(weightsVector[i], _threshold))
            {
                nullity++;
            }
        }

        return nullity;
    }

    /// <summary>
    /// Computes the null space (kernel) of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="threshold">The threshold value for determining when a value is considered zero.</param>
    /// <returns>A matrix whose columns form a basis for the null space of the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The null space of a matrix is the set of all vectors that, when multiplied by the matrix, give zero.
    /// These vectors represent "hidden patterns" or "invisible dimensions" in your data. Finding the null space helps identify
    /// what information your data cannot capture or distinguish between.
    /// </para>
    /// </remarks>
    public static Matrix<T> Nullspace<T>(this Matrix<T> matrix, T threshold)
    {
        int rows = matrix.Rows, columns = matrix.Columns, nullIndex = 0;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns);
        var _epsilon = ops.FromDouble(1e-10); // Small number instead of Epsilon
        var _threshold = ops.GreaterThanOrEquals(threshold, ops.Zero)
            ? threshold
            : ops.Multiply(
                ops.FromDouble(0.5 * Math.Sqrt(rows + columns + 1)),
                ops.Multiply(weightsVector[0], _epsilon)
            );
        var nullspaceMatrix = Matrix<T>.CreateMatrix<T>(columns, matrix.GetNullity(_threshold));
        var _vMatrix = matrix.Clone();

        for (int i = 0; i < columns; i++)
        {
            if (ops.LessThanOrEquals(weightsVector[i], _threshold))
            {
                for (int j = 0; j < columns; j++)
                {
                    nullspaceMatrix[j, nullIndex] = _vMatrix[j, i];
                }
                nullIndex++;
            }
        }

        return nullspaceMatrix;
    }

    /// <summary>
    /// Computes the range (column space) of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="threshold">The threshold value for determining when a value is considered zero.</param>
    /// <returns>A matrix whose columns form a basis for the range of the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The range of a matrix represents all possible outputs you can get when multiplying the matrix by any vector.
    /// It shows what kinds of transformations or changes the matrix can produce. Understanding the range helps identify what patterns
    /// or variations your data can represent.
    /// </para>
    /// </remarks>
    public static Matrix<T> GetRange<T>(this Matrix<T> matrix, T threshold)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rows = matrix.Rows, columns = matrix.Columns, rank = 0;
        var weightsVector = new Vector<T>(columns);
        var rangeMatrix = new Matrix<T>(rows, matrix.GetRank(threshold));
        var _uMatrix = matrix.Clone();

        for (int i = 0; i < columns; i++)
        {
            if (ops.GreaterThan(weightsVector[i], threshold))
            {
                for (int j = 0; j < rows; j++)
                {
                    rangeMatrix[j, rank] = _uMatrix[j, i];
                }
                rank++;
            }
        }

        return rangeMatrix;
    }

    /// <summary>
    /// Sets a submatrix within a larger matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The target matrix to modify.</param>
    /// <param name="startRow">The starting row index where the submatrix will be placed.</param>
    /// <param name="startCol">The starting column index where the submatrix will be placed.</param>
    /// <param name="submatrix">The smaller matrix to insert into the target matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Think of this like pasting a small picture into a specific location of a larger picture.
    /// The startRow and startCol parameters tell the method where to begin placing the smaller matrix.
    /// </para>
    /// </remarks>
    public static void SetSubmatrix<T>(this Matrix<T> matrix, int startRow, int startCol, Matrix<T> submatrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Vectorized: Copy each row using SIMD operations
        for (int i = 0; i < submatrix.Rows; i++)
        {
            var sourceRow = submatrix.GetRowReadOnlySpan(i);
            var destRow = matrix.GetRowSpan(startRow + i);
            numOps.Copy(sourceRow, destRow.Slice(startCol, submatrix.Columns));
        }
    }

    /// <summary>
    /// Determines if a matrix contains only zero values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <returns>True if all elements in the matrix are zero; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks if every single value in the matrix is zero.
    /// It's like checking if a grid of numbers contains only zeros.
    /// </para>
    /// </remarks>
    public static bool IsZeroMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Calculates the Frobenius norm of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to calculate the norm for.</param>
    /// <returns>The Frobenius norm value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Frobenius norm is a way to measure the "size" of a matrix.
    /// It's calculated by squaring each element, adding them all up, and then taking the square root of that sum.
    /// Think of it like finding the length of a vector, but for a matrix.
    /// </para>
    /// </remarks>
    public static T FrobeniusNorm<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Vectorized: Compute sum of squares using SIMD dot product (x · x = sum of x² terms)
        var span = matrix.AsSpan();
        T sumOfSquares = numOps.Dot(span, span);
        return numOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Creates a new matrix with all elements negated.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to negate.</param>
    /// <returns>A new matrix with all elements negated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a new matrix where each value is the negative of the original.
    /// For example, if an element in the original matrix is 5, it will be -5 in the new matrix.
    /// </para>
    /// </remarks>
    public static Matrix<T> Negate<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        // Vectorized: Negate all elements at once using SIMD operations
        numOps.Negate(matrix.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Inverts a matrix using Newton's iterative method.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="A">The matrix to invert.</param>
    /// <param name="maxIterations">The maximum number of iterations to perform.</param>
    /// <param name="tolerance">The convergence tolerance. If null, a default value is used.</param>
    /// <returns>The inverted matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square.</exception>
    /// <exception cref="InvalidOperationException">Thrown when Newton's method does not converge.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix inversion is like finding the reciprocal of a number (e.g., 1/x).
    /// Newton's method is an iterative approach that gradually improves an initial guess for the inverse.
    /// This method keeps refining the approximation until it's accurate enough (determined by the tolerance)
    /// or until it reaches the maximum number of iterations.
    /// </para>
    /// </remarks>
    public static Matrix<T> InverseNewton<T>(this Matrix<T> A, int maxIterations = 100, T? tolerance = default)
    {
        if (A.Rows != A.Columns)
            throw new ArgumentException("Matrix must be square.");

        int n = A.Rows;
        var X = A.Transpose();
        var I = Matrix<T>.CreateIdentity(n);
        var NumOps = MathHelper.GetNumericOperations<T>();
        tolerance ??= NumOps.FromDouble(1e-10);

        for (int k = 0; k < maxIterations; k++)
        {
            var R = I.Subtract(A.Multiply(X));
            if (NumOps.LessThan(R.FrobeniusNorm(), tolerance))
            {
                return X;
            }

            X = X.Add(X.Multiply(R));
        }

        throw new InvalidOperationException("Newton's method did not converge.");
    }

    /// <summary>
    /// Inverts a matrix using Strassen's algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="A">The matrix to invert.</param>
    /// <returns>The inverted matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square or its size is not a power of 2.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Strassen's algorithm is a divide-and-conquer approach to matrix operations.
    /// It works by breaking down the matrix into smaller submatrices, solving the smaller problems,
    /// and then combining the results. This method requires that the matrix size be a power of 2
    /// (like 2, 4, 8, 16, etc.). It's often more efficient for large matrices than standard methods.
    /// </para>
    /// </remarks>
    public static Matrix<T> InverseStrassen<T>(this Matrix<T> A)
    {
        if (A.Rows != A.Columns)
            throw new ArgumentException("Matrix must be square.");

        int n = A.Rows;
        var NumOps = MathHelper.GetNumericOperations<T>();

        if (n == 1)
        {
            return new Matrix<T>(1, 1) { [0, 0] = NumOps.Divide(NumOps.One, A[0, 0]) };
        }

        if (n % 2 != 0)
        {
            throw new ArgumentException("Matrix size must be a power of 2 for Strassen's algorithm.");
        }

        int m = n / 2;

        var A11 = A.Submatrix(0, 0, m, m);
        var A12 = A.Submatrix(0, m, m, m);
        var A21 = A.Submatrix(m, 0, m, m);
        var A22 = A.Submatrix(m, m, m, m);

        var A11_inv = InverseStrassen(A11);
        var S = A22.Subtract(A21.Multiply(A11_inv).Multiply(A12));
        var S_inv = InverseStrassen(S);
        var P = A11_inv.Add(A11_inv.Multiply(A12).Multiply(S_inv).Multiply(A21).Multiply(A11_inv));
        var Q = A11_inv.Multiply(A12).Multiply(S_inv).Negate();
        var R = S_inv.Multiply(A21).Multiply(A11_inv).Negate();

        var result = new Matrix<T>(n, n);
        result.SetSubmatrix(0, 0, P);
        result.SetSubmatrix(0, m, Q);
        result.SetSubmatrix(m, 0, R);
        result.SetSubmatrix(m, m, S_inv);

        return result;
    }

    /// <summary>
    /// Inverts a matrix using the specified algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to invert.</param>
    /// <param name="inverseType">The algorithm to use for matrix inversion.</param>
    /// <param name="maxIterations">The maximum number of iterations for iterative methods.</param>
    /// <param name="tolerance">The convergence tolerance for iterative methods.</param>
    /// <returns>The inverted matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when an invalid inverse type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is a convenient way to invert a matrix using different algorithms.
    /// You can choose between:
    /// - GaussianJordan: A direct method that works for any square matrix.
    /// - Newton: An iterative method that gradually improves the approximation.
    /// - Strassen: A divide-and-conquer method for matrices with dimensions that are powers of 2.
    /// Each method has its advantages in different situations.
    /// </para>
    /// </remarks>
    public static Matrix<T> Inverse<T>(this Matrix<T> matrix, InverseType inverseType = InverseType.GaussianJordan, int maxIterations = 100, T? tolerance = default)
    {
        return inverseType switch
        {
            InverseType.Strassen => InverseStrassen(matrix),
            InverseType.Newton => InverseNewton(matrix, maxIterations, tolerance),
            InverseType.GaussianJordan => InverseGaussianJordanElimination(matrix),
            _ => throw new ArgumentException("Invalid inverse type", nameof(inverseType)),
        };
    }

    /// <summary>
    /// Transposes a matrix by swapping its rows and columns.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to transpose.</param>
    /// <returns>A new matrix that is the transpose of the original matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the matrix has zero rows or columns.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transposing a matrix means converting its rows into columns and columns into rows.
    /// For example, if you have a matrix with 3 rows and 2 columns, the transposed matrix will have 2 rows and 3 columns.
    /// It's like flipping the matrix along its diagonal.
    /// </para>
    /// </remarks>
    public static Matrix<T> Transpose<T>(this Matrix<T> matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        var rows = matrix.Rows;
        var columns = matrix.Columns;
        if (rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        if (columns == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one column of values", nameof(matrix));
        }

        var newMatrix = Matrix<T>.CreateMatrix<T>(columns, rows);
        for (int i = 0; i < columns; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                newMatrix[i, j] = matrix[j, i];
            }
        }

        return newMatrix;
    }

    /// <summary>
    /// Calculates the rank of a matrix based on a given threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to calculate the rank for.</param>
    /// <param name="threshold">The threshold value for determining linearly independent rows/columns.</param>
    /// <returns>The rank of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The rank of a matrix tells you how many linearly independent rows or columns it has.
    /// Think of it as the number of dimensions the matrix can represent. The threshold parameter helps determine
    /// when values are considered significant enough to contribute to the rank.
    /// </para>
    /// </remarks>
    public static int GetRank<T>(this Matrix<T> matrix, T threshold)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns);
        var epsilon = ops.FromDouble(1e-10); // Small number instead of Epsilon
        var thresh = ops.GreaterThanOrEquals(threshold, ops.Zero)
            ? threshold
            : ops.Multiply(ops.FromDouble(0.5 * Math.Sqrt(rows + columns + 1)), ops.Multiply(weightsVector[0], epsilon));
        int rank = 0;

        for (int i = 0; i < columns; i++)
        {
            if (ops.GreaterThan(weightsVector[i], thresh))
            {
                rank++;
            }
        }

        return rank;
    }

    /// <summary>
    /// Swaps two rows in a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to modify.</param>
    /// <param name="row1Index">The index of the first row to swap.</param>
    /// <param name="row2Index">The index of the second row to swap.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method exchanges the positions of two rows in a matrix.
    /// It's like swapping two rows in a spreadsheet - all values in those rows change places with each other.
    /// </para>
    /// </remarks>
    public static void SwapRows<T>(this Matrix<T> matrix, int row1Index, int row2Index)
    {
        var rows = matrix.Rows;
        for (int i = 0; i < rows; i++)
        {
            (matrix[row2Index, i], matrix[row1Index, i]) = (matrix[row1Index, i], matrix[row2Index, i]);
        }
    }

    /// <summary>
    /// Inverts a diagonal matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The diagonal matrix to invert.</param>
    /// <returns>The inverted diagonal matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix has non-zero values only along its main diagonal (top-left to bottom-right).
    /// Inverting a diagonal matrix is simple - just replace each diagonal element with its reciprocal (1/value).
    /// This method assumes the input is already a diagonal matrix.
    /// </para>
    /// </remarks>
    public static Matrix<T> InvertDiagonalMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var invertedMatrix = Matrix<T>.CreateMatrix<T>(rows, rows);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < rows; i++)
        {
            invertedMatrix[i, i] = ops.Divide(ops.One, matrix[i, i]);
        }

        return invertedMatrix;
    }

    /// <summary>
    /// Inverts a unitary matrix by taking its transpose.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The unitary matrix to invert.</param>
    /// <returns>The inverted unitary matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A unitary matrix is a special type of matrix where its inverse equals its conjugate transpose.
    /// This property makes unitary matrices very useful in many AI and mathematical applications because they preserve
    /// the length of vectors and the angles between them. For real-valued matrices, unitary matrices are called orthogonal matrices.
    /// </para>
    /// </remarks>
    public static Matrix<Complex<T>> InvertUnitaryMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        return matrix.Transpose();
    }

    /// <summary>
    /// Determines if a matrix is unitary by checking if its conjugate transpose equals its inverse.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to check.</param>
    /// <param name="matrixDecomposition">The decomposition method used to calculate the inverse.</param>
    /// <returns>True if the matrix is unitary; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A unitary matrix has the special property that its inverse equals its conjugate transpose.
    /// This method checks if a matrix is unitary by comparing these two values. Unitary matrices are important in quantum
    /// computing and many AI algorithms because they preserve the "length" of vectors they operate on.
    /// </para>
    /// </remarks>
    public static bool IsUnitaryMatrix<T>(this Matrix<Complex<T>> matrix, IMatrixDecomposition<Complex<T>> matrixDecomposition)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A unitary matrix must be square
        }

        var conjugateTranspose = matrix.ConjugateTranspose();
        var inverse = matrixDecomposition.Invert();

        // Check if conjugate transpose is equal to inverse
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (conjugateTranspose[i, j] != inverse[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Calculates the determinant of a square matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to calculate the determinant for.</param>
    /// <returns>The determinant value.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square or has zero rows/columns.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The determinant is a special number calculated from a square matrix that tells you
    /// important information about the matrix. If the determinant is zero, the matrix is "singular" (has no inverse).
    /// The determinant also tells you how much the matrix scales areas or volumes when used as a transformation.
    /// This method uses a recursive approach to calculate the determinant.
    /// </para>
    /// </remarks>
    public static T Determinant<T>(this Matrix<T> matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        var columns = matrix.Columns;
        var rows = matrix.Rows;
        if (rows <= 0)
        {
            throw new ArgumentException($"{nameof(rows)} needs to be an integer larger than 0");
        }

        if (columns <= 0)
        {
            throw new ArgumentException($"{nameof(columns)} needs to be an integer larger than 0");
        }

        if (rows != columns)
        {
            throw new ArgumentException($"You need to have a square matrix to calculate the determinant value so the length of {nameof(rows)} {nameof(columns)} must be equal");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        if (rows == 2)
        {
            return ops.Subtract(
                ops.Multiply(matrix[0, 0], matrix[1, 1]),
                ops.Multiply(matrix[0, 1], matrix[1, 0])
            );
        }
        else
        {
            T determinant = ops.Zero;
            for (int i = 0; i < rows; i++)
            {
                var tempMatrix = new Matrix<T>(rows - 1, rows - 1);
                for (int j = 0; j < rows - 1; j++)
                {
                    for (int k = 0; k < rows - 1; k++)
                    {
                        tempMatrix[j, k] = matrix[j < i ? j : j + 1, k];
                    }
                }

                T subDeterminant = tempMatrix.Determinant();
                T sign = ops.FromDouble(i % 2 == 0 ? 1 : -1);
                T product = ops.Multiply(ops.Multiply(sign, matrix[0, i]), subDeterminant);
                determinant = ops.Add(determinant, product);
            }

            return determinant;
        }
    }

    /// <summary>
    /// Retrieves a specific row from the matrix as an array.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to extract the row from.</param>
    /// <param name="rowIndex">The zero-based index of the row to retrieve.</param>
    /// <returns>An array containing all elements in the specified row.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method extracts a single horizontal row of values from your matrix.
    /// Think of it like selecting an entire row from a spreadsheet. The rowIndex parameter specifies
    /// which row you want (starting from 0 for the first row).
    /// </para>
    /// </remarks>
    public static T[] GetRow<T>(this Matrix<T> matrix, int rowIndex)
    {
        return [.. Enumerable.Range(0, matrix.Columns).Select(x => matrix[rowIndex, x])];
    }

    /// <summary>
    /// Extracts a portion of a column from the matrix as a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to extract the column portion from.</param>
    /// <param name="columnIndex">The zero-based index of the column to extract from.</param>
    /// <param name="startRow">The zero-based index of the first row to include.</param>
    /// <param name="length">The number of elements to extract from the column.</param>
    /// <returns>A vector containing the specified portion of the column.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when columnIndex is outside the matrix bounds, startRow is outside the matrix bounds,
    /// or the requested length would extend beyond the matrix bounds.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method extracts a section of a vertical column from your matrix.
    /// You specify which column you want with columnIndex, where to start in that column with startRow,
    /// and how many values to extract with length. The result is a vector (a one-dimensional array)
    /// containing just those values.
    /// </para>
    /// </remarks>
    public static Vector<T> GetSubColumn<T>(this Matrix<T> matrix, int columnIndex, int startRow, int length)
    {
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        if (startRow < 0 || startRow >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(startRow));
        if (length < 0 || startRow + length > matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(length));

        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = matrix[startRow + i, columnIndex];
        }

        return result;
    }

    /// <summary>
    /// Calculates the logarithm of the determinant of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to calculate the log determinant for.</param>
    /// <returns>The logarithm of the determinant value.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The determinant is a special number calculated from a square matrix.
    /// For very large or very small determinant values, calculating the logarithm of the determinant
    /// helps avoid numerical overflow or underflow issues. This method uses LU decomposition
    /// (a way of factoring matrices) to calculate the log determinant more efficiently and accurately.
    /// </para>
    /// <para>
    /// This is particularly useful in statistical applications like calculating multivariate normal
    /// distribution probabilities.
    /// </para>
    /// </remarks>
    public static T LogDeterminant<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square to calculate determinant.");
        }

        // Use LU decomposition to calculate log determinant
        var lu = new LuDecomposition<T>(matrix);
        var U = lu.U;

        T logDet = numOps.Zero;
        for (int i = 0; i < U.Rows; i++)
        {
            logDet = numOps.Add(logDet, numOps.Log(numOps.Abs(U[i, i])));
        }

        return logDet;
    }

    /// <summary>
    /// Performs element-by-element multiplication of two matrices of the same dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The first matrix for multiplication.</param>
    /// <param name="other">The second matrix for multiplication.</param>
    /// <returns>A new matrix where each element is the product of the corresponding elements in the input matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrices have different dimensions.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is different from regular matrix multiplication. In pointwise multiplication
    /// (also called Hadamard product), each element in the result is calculated by multiplying the corresponding
    /// elements at the same position in both input matrices. Both matrices must have exactly the same number
    /// of rows and columns.
    /// </para>
    /// <para>
    /// For example, if matrix[1,2] = 5 and other[1,2] = 3, then the result[1,2] will be 15.
    /// </para>
    /// </remarks>
    public static Matrix<T> PointwiseMultiply<T>(this Matrix<T> matrix, Matrix<T> other)
    {
        if (matrix.Rows != other.Rows || matrix.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for pointwise multiplication.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        // Vectorized: Multiply all elements at once using SIMD operations
        numOps.Multiply(matrix.AsSpan(), other.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Multiplies each row of a matrix by the corresponding element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix whose rows will be scaled.</param>
    /// <param name="vector">The vector containing scaling factors for each row.</param>
    /// <returns>A new matrix with each row scaled by the corresponding vector element.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of rows in the matrix doesn't match the vector length.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method scales each row of your matrix by a corresponding value from the vector.
    /// For example, if your vector is [2, 3, 4] and your matrix has 3 rows, then the first row will be multiplied by 2,
    /// the second row by 3, and the third row by 4. This is useful for applying different weights to each row of data.
    /// </para>
    /// </remarks>
    public static Matrix<T> PointwiseMultiply<T>(this Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Rows != vector.Length)
        {
            throw new ArgumentException("The number of rows in the matrix must match the length of the vector.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(matrix.Rows, matrix.Columns);

        // Vectorized: Scale each row by corresponding vector element using SIMD operations
        for (int i = 0; i < matrix.Rows; i++)
        {
            var sourceRow = matrix.GetRowReadOnlySpan(i);
            var destRow = result.GetRowSpan(i);
            numOps.MultiplyScalar(sourceRow, vector[i], destRow);
        }

        return result;
    }

    /// <summary>
    /// Adds a new column to the right side of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to add a column to.</param>
    /// <param name="column">The vector to add as a new column.</param>
    /// <returns>A new matrix with the additional column.</returns>
    /// <exception cref="ArgumentException">Thrown when the column length doesn't match the matrix row count.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a new matrix that includes all the data from your original matrix,
    /// plus an additional column at the right side. The new column's values come from the vector you provide.
    /// The vector must have the same number of elements as the matrix has rows.
    /// </para>
    /// <para>
    /// This is useful when you need to augment your data with additional features or when constructing
    /// special matrices for certain algorithms.
    /// </para>
    /// </remarks>
    public static Matrix<T> AddColumn<T>(this Matrix<T> matrix, Vector<T> column)
    {
        if (matrix.Rows != column.Length)
        {
            throw new ArgumentException("Column length must match matrix row count.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> newMatrix = new Matrix<T>(matrix.Rows, matrix.Columns + 1);

        // Vectorized: Copy each row using SIMD operations, then set last column
        for (int i = 0; i < matrix.Rows; i++)
        {
            var sourceRow = matrix.GetRowReadOnlySpan(i);
            var destRow = newMatrix.GetRowSpan(i);
            numOps.Copy(sourceRow, destRow.Slice(0, matrix.Columns));
            newMatrix[i, matrix.Columns] = column[i];
        }

        return newMatrix;
    }

    /// <summary>
    /// Extracts a submatrix from the original matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The source matrix to extract from.</param>
    /// <param name="startRow">The zero-based index of the first row to include.</param>
    /// <param name="startCol">The zero-based index of the first column to include.</param>
    /// <param name="numRows">The number of rows to extract.</param>
    /// <param name="numCols">The number of columns to extract.</param>
    /// <returns>A new matrix containing the specified portion of the original matrix.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the specified submatrix dimensions extend beyond the bounds of the original matrix.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you extract a smaller matrix from within a larger one.
    /// Think of it like cropping a rectangular section from a spreadsheet. You specify where to start
    /// (startRow, startCol) and how many rows and columns to include (numRows, numCols).
    /// </para>
    /// <para>
    /// For example, if you have a 5x5 matrix and call Submatrix(1, 2, 2, 2), you'll get a 2x2 matrix
    /// containing the elements from rows 1-2 and columns 2-3 of the original matrix.
    /// </para>
    /// </remarks>
    public static Matrix<T> Submatrix<T>(this Matrix<T> matrix, int startRow, int startCol, int numRows, int numCols)
    {
        if (startRow < 0 || startCol < 0 || startRow + numRows > matrix.Rows || startCol + numCols > matrix.Columns)
        {
            throw new ArgumentOutOfRangeException("Invalid submatrix dimensions");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> submatrix = new Matrix<T>(numRows, numCols);

        // Vectorized: Copy row segments using SIMD operations
        for (int i = 0; i < numRows; i++)
        {
            var sourceRow = matrix.GetRowReadOnlySpan(startRow + i);
            var destRow = submatrix.GetRowSpan(i);
            numOps.Copy(sourceRow.Slice(startCol, numCols), destRow);
        }

        return submatrix;
    }

    /// <summary>
    /// Creates a new matrix containing only the specified columns from the original matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The source matrix to extract columns from.</param>
    /// <param name="columnIndices">A collection of zero-based indices of columns to extract.</param>
    /// <returns>A new matrix containing only the specified columns.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to create a new matrix that includes only specific
    /// columns from your original matrix. For example, if you have a dataset where each column represents
    /// a different feature, you can use this method to select only the features you're interested in.
    /// </para>
    /// <para>
    /// The order of columns in the result will match the order of indices in the columnIndices collection.
    /// </para>
    /// </remarks>
    public static Matrix<T> GetColumns<T>(this Matrix<T> matrix, IEnumerable<int> columnIndices)
    {
        return new Matrix<T>(GetColumnVectors(matrix, [.. columnIndices]));
    }

    /// <summary>
    /// Gets specific column vectors from a matrix based on the specified indices.
    /// </summary>
    /// <param name="matrix">The matrix from which to extract columns.</param>
    /// <param name="indices">The indices of the columns to extract.</param>
    /// <returns>An array of vectors representing the selected columns.</returns>
    /// <remarks>
    /// <para>
    /// This extension method allows for efficient extraction of specific columns from a matrix.
    /// It creates a new array of vectors where each vector represents one of the requested columns.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you pull out specific columns from your data matrix.
    /// 
    /// Imagine your data as a grid of numbers:
    /// - Each column might represent a feature like age, height, or temperature
    /// - This method lets you select only the columns you're interested in
    /// - The result is a collection of vectors, where each vector is one column
    /// 
    /// For example, if you only want columns 0, 3, and 5 from a matrix with 10 columns,
    /// you would pass [0, 3, 5] as the indices.
    /// </para>
    /// </remarks>
    public static Vector<T>[] GetColumnVectors<T>(this Matrix<T> matrix, int[] indices)
    {
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        Vector<T>[] columns = new Vector<T>[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= matrix.Columns)
            {
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Column index {indices[i]} is out of range for matrix with {matrix.Columns} columns");
            }
            columns[i] = matrix.GetColumn(indices[i]);
        }

        return columns;
    }

    /// <summary>
    /// Finds the maximum value in the matrix after applying a transformation function to each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to search for the maximum value.</param>
    /// <param name="selector">A function to transform each element before comparison.</param>
    /// <returns>The maximum value after applying the transformation function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps you find the largest value in your matrix, but with a twist.
    /// Before comparing values, it applies a function (the selector) to each element. For example, if you
    /// want to find the element with the largest absolute value, you could use a selector that calculates
    /// the absolute value of each element.
    /// </para>
    /// <para>
    /// The selector function takes an element of type T and returns a transformed value of type T.
    /// </para>
    /// </remarks>
    public static T Max<T>(this Matrix<T> matrix, Func<T, T> selector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T max = selector(matrix[0, 0]);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                T value = selector(matrix[i, j]);
                if (numOps.GreaterThan(value, max))
                {
                    max = value;
                }
            }
        }

        return max;
    }

    /// <summary>
    /// Extracts a range of consecutive rows from the matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The source matrix to extract rows from.</param>
    /// <param name="startRow">The zero-based index of the first row to include.</param>
    /// <param name="rowCount">The number of consecutive rows to extract.</param>
    /// <returns>A new matrix containing the specified rows.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you extract a sequence of rows from your matrix.
    /// For example, if you have a dataset where each row represents a different observation,
    /// you can use this method to select a specific range of observations (like rows 10-20).
    /// </para>
    /// <para>
    /// The resulting matrix will have the same number of columns as the original matrix.
    /// </para>
    /// </remarks>
    public static Matrix<T> GetRowRange<T>(this Matrix<T> matrix, int startRow, int rowCount)
    {
        Matrix<T> result = new(rowCount, matrix.Columns);
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[startRow + i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// For each row in the matrix, finds the index of the column containing the maximum value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <returns>A vector where each element is the index of the maximum value in the corresponding row.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines each row of your matrix and identifies which column
    /// contains the largest value. It returns a vector where each element corresponds to a row in your
    /// original matrix, and the value is the column index (position) of the maximum value in that row.
    /// </para>
    /// <para>
    /// This is particularly useful in machine learning for finding the predicted class in classification
    /// problems, where each row might represent probabilities for different classes.
    /// </para>
    /// </remarks>
    public static Vector<T> RowWiseArgmax<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        Vector<T> result = new(matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T max = matrix[i, 0];
            int maxIndex = 0;
            for (int j = 1; j < matrix.Columns; j++)
            {
                if (numOps.GreaterThan(matrix[i, j], max))
                {
                    max = matrix[i, j];
                    maxIndex = j;
                }
            }
            result[i] = numOps.FromDouble(maxIndex);
        }

        return result;
    }

    /// <summary>
    /// Computes the Kronecker product of two matrices.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="a">The first matrix.</param>
    /// <param name="b">The second matrix.</param>
    /// <returns>The Kronecker product of the two matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Kronecker product is a special way of combining two matrices that results
    /// in a much larger matrix. If matrix A is m×n and matrix B is p×q, their Kronecker product will be
    /// a matrix of size (m×p)×(n×q).
    /// </para>
    /// <para>
    /// Think of it as replacing each element of matrix A with a scaled copy of matrix B, where the scaling
    /// factor is the value of the element in A. This operation is useful in various fields including
    /// quantum computing, image processing, and when working with certain types of mathematical models.
    /// </para>
    /// </remarks>
    public static Matrix<T> KroneckerProduct<T>(this Matrix<T> a, Matrix<T> b)
    {
        int m = a.Rows;
        int n = a.Columns;
        int p = b.Rows;
        int q = b.Columns;

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(m * p, n * q);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    for (int l = 0; l < q; l++)
                    {
                        result[i * p + k, j * q + l] = numOps.Multiply(a[i, j], b[k, l]);
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts a two-dimensional matrix into a one-dimensional vector by placing all elements in a single row.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to flatten.</param>
    /// <returns>A vector containing all elements of the matrix in row-major order.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Flattening a matrix means converting it from a 2D structure (rows and columns) 
    /// into a 1D structure (a single line of values). This method takes all the values from your matrix 
    /// and puts them into a vector (a one-dimensional array), reading from left to right, top to bottom.
    /// </para>
    /// <para>
    /// For example, if you have a 2×3 matrix:
    /// [1, 2, 3]
    /// [4, 5, 6]
    /// The flattened vector would be: [1, 2, 3, 4, 5, 6]
    /// </para>
    /// <para>
    /// This is commonly used in machine learning when you need to feed a 2D structure (like an image) 
    /// into an algorithm that only accepts 1D inputs.
    /// </para>
    /// </remarks>
    public static Vector<T> Flatten<T>(this Matrix<T> matrix)
    {
        var vector = new Vector<T>(matrix.Rows * matrix.Columns);
        int index = 0;

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                vector[index++] = matrix[i, j];
            }
        }

        return vector;
    }

    /// <summary>
    /// Reorganizes the elements of a matrix into a new matrix with different dimensions while preserving all data.
    /// </summary>
    /// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
    /// <param name="matrix">The matrix to reshape.</param>
    /// <param name="newRows">The number of rows in the reshaped matrix.</param>
    /// <param name="newColumns">The number of columns in the reshaped matrix.</param>
    /// <returns>A new matrix with the specified dimensions containing all elements from the original matrix.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the total number of elements in the new shape doesn't match the original matrix.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reshaping a matrix means changing its dimensions (rows and columns) while keeping 
    /// all the same values. It's like rearranging the same set of numbers into a different grid pattern.
    /// </para>
    /// <para>
    /// For example, if you have a 2×3 matrix (2 rows, 3 columns):
    /// [1, 2, 3]
    /// [4, 5, 6]
    /// 
    /// You could reshape it to a 3×2 matrix (3 rows, 2 columns):
    /// [1, 2]
    /// [3, 4]
    /// [5, 6]
    /// </para>
    /// <para>
    /// The total number of elements must stay the same (in this example, 6 elements). The method reads the original 
    /// matrix row by row and fills the new matrix in the same way.
    /// </para>
    /// <para>
    /// This is useful in data processing and machine learning when you need to transform data between different 
    /// formats, such as converting between image representations or preparing data for specific algorithms.
    /// </para>
    /// </remarks>
    public static Matrix<T> Reshape<T>(this Matrix<T> matrix, int newRows, int newColumns)
    {
        if (matrix.Rows * matrix.Columns != newRows * newColumns)
        {
            throw new ArgumentException("The total number of elements must remain the same after reshaping.");
        }

        var reshaped = new Matrix<T>(newRows, newColumns);
        int index = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                int newRow = index / newColumns;
                int newCol = index % newColumns;
                reshaped[newRow, newCol] = matrix[i, j];
                index++;
            }
        }

        return reshaped;
    }

    /// <summary>
    /// Creates a submatrix from the given matrix using the specified indices.
    /// </summary>
    /// <typeparam name="T">The type of elements in the matrix.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="indices">The indices of rows to include in the submatrix.</param>
    /// <returns>A new matrix containing only the specified rows.</returns>
    public static Matrix<T> Submatrix<T>(this Matrix<T> matrix, int[] indices)
    {
        var result = new Matrix<T>(indices.Length, matrix.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[indices[i], j];
            }
        }

        return result;
    }
}
