using AiDotNet.Tensors.Helpers;
namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a mathematical matrix of elements of type T, providing various matrix operations.
/// </summary>
/// <typeparam name="T">The numeric type of the matrix elements (e.g., double, float, int).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A matrix is like a table or grid of numbers arranged in rows and columns.
/// You can perform various mathematical operations on matrices such as addition, subtraction, and multiplication.
/// Matrices are commonly used in AI for storing and manipulating data, like representing weights in neural networks.</para>
/// </remarks>
public class Matrix<T> : MatrixBase<T>, IEnumerable<T>
{
    /// <summary>
    /// Initializes a new matrix with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an empty matrix with the specified size.
    /// For example, Matrix(3, 4) creates a matrix with 3 rows and 4 columns.</para>
    /// </remarks>
    public Matrix(int rows, int columns) : base(rows, columns)
    {
    }

    /// <summary>
    /// Initializes a new matrix from a collection of collections, where each inner collection represents a row.
    /// </summary>
    /// <param name="values">A collection of collections, where each inner collection represents a row of the matrix.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This lets you create a matrix from lists of values.
    /// Each inner list becomes one row in the matrix.</para>
    /// </remarks>
    public Matrix(IEnumerable<IEnumerable<T>> values) : base(values)
    {
    }

    /// <summary>
    /// Initializes a new matrix from a 2D array.
    /// </summary>
    /// <param name="data">A 2D array containing the matrix data.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a matrix directly from a 2D array (a grid of values).</para>
    /// </remarks>
    public Matrix(T[,] data) : base(data)
    {
    }

    /// <summary>
    /// Creates a new instance of the matrix with the specified dimensions.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A new matrix instance.</returns>
    protected override MatrixBase<T> CreateInstance(int rows, int cols)
    {
        return new Matrix<T>(rows, cols);
    }

    /// <summary>
    /// Creates a new matrix with the specified dimensions.
    /// </summary>
    /// <typeparam name="T2">Unused type parameter (maintained for compatibility).</typeparam>
    /// <param name="rows">The number of rows.</param>
    /// <param name="columns">The number of columns.</param>
    /// <returns>A new matrix with the specified dimensions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a helper method to create a new matrix with a specific size.</para>
    /// </remarks>
    public static Matrix<T> CreateMatrix<T2>(int rows, int columns)
    {
        return new Matrix<T>(rows, columns);
    }

    /// <summary>
    /// Creates an identity matrix of the specified size.
    /// </summary>
    /// <typeparam name="T2">Unused type parameter (maintained for compatibility).</typeparam>
    /// <param name="size">The size of the square identity matrix.</param>
    /// <returns>An identity matrix of the specified size.</returns>
    /// <exception cref="ArgumentException">Thrown when size is less than 1.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> An identity matrix is a special square matrix where all elements are 0 except
    /// for the main diagonal (top-left to bottom-right), which contains 1s. It's similar to the number 1 in
    /// multiplication - multiplying any matrix by an identity matrix gives you the original matrix.</para>
    /// <para><b>Performance:</b> Uses vectorized Fill operation to initialize with zeros, then sets diagonal elements (5-10x faster than loops).</para>
    /// </remarks>
    public static Matrix<T> CreateIdentityMatrix(int size)
    {
        if (size < 1)
        {
            throw new ArgumentException($"{nameof(size)} has to be a minimum of 1", nameof(size));
        }

        var identityMatrix = new Matrix<T>(size, size);

        // Use vectorized Fill to initialize entire matrix with zeros
        _numOps.Fill(identityMatrix.AsWritableSpan(), _numOps.Zero);

        // Set diagonal elements to one
        for (int i = 0; i < size; i++)
        {
            identityMatrix[i, i] = _numOps.One;
        }

        return identityMatrix;
    }

    /// <summary>
    /// Gets a column from the matrix as a Vector.
    /// </summary>
    /// <param name="col">The zero-based index of the column to retrieve.</param>
    /// <returns>A Vector containing the values from the specified column.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This extracts a single column from the matrix as a vector.
    /// For example, in a 3ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½3 matrix, getting column 1 would give you the middle column as a vector.</para>
    /// </remarks>
    public new Vector<T> GetColumn(int col)
    {
        return base.GetColumn(col);
    }

    /// <summary>
    /// Creates a deep copy of this matrix.
    /// </summary>
    /// <returns>A new matrix that is a copy of this matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an exact duplicate of the matrix that can be modified
    /// independently without affecting the original.</para>
    /// </remarks>
    public new Matrix<T> Clone()
    {
        return (Matrix<T>)base.Clone();
    }

    /// <summary>
    /// Adds another matrix to this matrix.
    /// </summary>
    /// <param name="other">The matrix to add to this matrix.</param>
    /// <returns>A new matrix that is the sum of this matrix and the other matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds corresponding elements of two matrices together.
    /// For example, the element at row 1, column 2 in the result will be the sum of the elements
    /// at row 1, column 2 in both input matrices.</para>
    /// </remarks>
    public new Matrix<T> Add(MatrixBase<T> other)
    {
        return (Matrix<T>)base.Add(other);
    }

    /// <summary>
    /// Subtracts another matrix from this matrix.
    /// </summary>
    /// <param name="other">The matrix to subtract from this matrix.</param>
    /// <returns>A new matrix that is the difference of this matrix and the other matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This subtracts corresponding elements of the second matrix from the first.
    /// For example, the element at row 1, column 2 in the result will be the element at row 1, column 2 in the
    /// first matrix minus the element at row 1, column 2 in the second matrix.</para>
    /// </remarks>
    public new Matrix<T> Subtract(MatrixBase<T> other)
    {
        return (Matrix<T>)base.Subtract(other);
    }

    /// <summary>
    /// Multiplies this matrix by another matrix.
    /// </summary>
    /// <param name="other">The matrix to multiply with this matrix.</param>
    /// <returns>A new matrix that is the product of this matrix and the other matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Matrix multiplication is different from regular multiplication.
    /// Each element in the result is calculated by taking a row from the first matrix and a column from the second matrix,
    /// multiplying corresponding elements, and summing them up. This operation is fundamental in many AI algorithms,
    /// especially in neural networks where it's used to apply weights to inputs.</para>
    /// </remarks>
    public new Matrix<T> Multiply(MatrixBase<T> other)
    {
        return (Matrix<T>)base.Multiply(other);
    }

    /// <summary>
    /// Multiplies this matrix by a vector.
    /// </summary>
    /// <param name="vector">The vector to multiply with this matrix.</param>
    /// <returns>A new vector that is the product of this matrix and the vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This applies the matrix transformation to a vector.
    /// It's commonly used in AI to transform data or apply learned weights to input features.
    /// The result is calculated by taking each row of the matrix, multiplying it element-wise with the vector,
    /// and summing the products.</para>
    /// </remarks>
    public new Vector<T> Multiply(Vector<T> vector)
    {
        return (Vector<T>)base.Multiply(vector);
    }

    /// <summary>
    /// Multiplies this matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply with this matrix.</param>
    /// <returns>A new matrix where each element is multiplied by the scalar value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies every element in the matrix by the same number (scalar).
    /// For example, multiplying a matrix by 2 would double every value in the matrix.</para>
    /// </remarks>
    public new Matrix<T> Multiply(T scalar)
    {
        return (Matrix<T>)base.Multiply(scalar);
    }

    /// <summary>
    /// Transposes this matrix (swaps rows and columns).
    /// </summary>
    /// <returns>A new matrix that is the transpose of this matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transposing a matrix means flipping it over its diagonal - rows become columns
    /// and columns become rows. For example, the element at row 2, column 3 in the original matrix
    /// will be at row 3, column 2 in the transposed matrix.</para>
    /// </remarks>
    public new Matrix<T> Transpose()
    {
        return (Matrix<T>)base.Transpose();
    }

    /// <summary>
    /// Adds two matrices together.
    /// </summary>
    /// <param name="left">The first matrix.</param>
    /// <param name="right">The second matrix.</param>
    /// <returns>A new matrix that is the sum of the two matrices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator allows you to use the + symbol to add matrices together,
    /// just like you would with regular numbers.</para>
    /// </remarks>
    public static Matrix<T> operator +(Matrix<T> left, Matrix<T> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Subtracts the right matrix from the left matrix.
    /// </summary>
    /// <param name="left">The matrix to subtract from.</param>
    /// <param name="right">The matrix to subtract.</param>
    /// <returns>A new matrix that is the difference of the two matrices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator allows you to use the - symbol to subtract one matrix from another,
    /// similar to how you would subtract regular numbers.</para>
    /// </remarks>
    public static Matrix<T> operator -(Matrix<T> left, Matrix<T> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Multiplies two matrices together.
    /// </summary>
    /// <param name="left">The left matrix in the multiplication.</param>
    /// <param name="right">The right matrix in the multiplication.</param>
    /// <returns>A new matrix that is the result of multiplying the left and right matrices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Matrix multiplication combines two matrices to create a new one. 
    /// Unlike regular multiplication, the order matters (A*B is not the same as B*A). 
    /// For this operation to work, the number of columns in the left matrix must equal 
    /// the number of rows in the right matrix.</para>
    /// </remarks>
    public static Matrix<T> operator *(Matrix<T> left, Matrix<T> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    /// <param name="matrix">The matrix to multiply.</param>
    /// <param name="vector">The vector to multiply by.</param>
    /// <returns>A new vector that is the result of the multiplication.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operation transforms a vector using a matrix.
    /// It's commonly used in AI to apply transformations to data points.
    /// The number of columns in the matrix must equal the length of the vector.</para>
    /// </remarks>
    public static Vector<T> operator *(Matrix<T> matrix, Vector<T> vector)
    {
        return matrix.Multiply(vector);
    }

    /// <summary>
    /// Multiplies each element of a matrix by a scalar value.
    /// </summary>
    /// <param name="matrix">The matrix to multiply.</param>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new matrix with each element multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This simply multiplies every number in the matrix by the same value.
    /// For example, multiplying a matrix by 2 doubles every value in the matrix.</para>
    /// </remarks>
    public static Matrix<T> operator *(Matrix<T> matrix, T scalar)
    {
        return matrix.Multiply(scalar);
    }

    /// <summary>
    /// Divides each element of a matrix by a scalar value.
    /// </summary>
    /// <param name="matrix">The matrix to divide.</param>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new matrix with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This divides every number in the matrix by the same value.
    /// For example, dividing a matrix by 2 halves every value in the matrix.</para>
    /// </remarks>
    public static Matrix<T> operator /(Matrix<T> matrix, T scalar)
    {
        return matrix.Divide(scalar);
    }

    /// <summary>
    /// Divides each element of the left matrix by the corresponding element in the right matrix.
    /// </summary>
    /// <param name="left">The left matrix (numerator).</param>
    /// <param name="right">The right matrix (denominator).</param>
    /// <returns>A new matrix with the results of element-wise division.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This performs division between corresponding elements in two matrices.
    /// Both matrices must have the same dimensions (same number of rows and columns).</para>
    /// </remarks>
    public static Matrix<T> operator /(Matrix<T> left, Matrix<T> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Creates a matrix from a single vector (as a row).
    /// </summary>
    /// <param name="vector">The vector to convert to a matrix.</param>
    /// <returns>A matrix with a single row containing the vector's elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a one-dimensional array of numbers (vector) 
    /// into a matrix with just one row. It's useful when you need to apply matrix operations 
    /// to vector data.</para>
    /// </remarks>
    public static Matrix<T> CreateFromVector(Vector<T> vector)
    {
        return new Matrix<T>([vector.AsEnumerable()]);
    }

    /// <summary>
    /// Creates a matrix filled with ones.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A matrix of the specified size filled with ones.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a matrix where every element has the value 1.
    /// Such matrices are often used as starting points in various algorithms or for masking operations.</para>
    /// </remarks>
    public new Matrix<T> Ones(int rows, int cols)
    {
        return (Matrix<T>)base.Ones(rows, cols);
    }

    /// <summary>
    /// Creates a matrix filled with zeros.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A matrix of the specified size filled with zeros.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a matrix where every element has the value 0.
    /// Zero matrices are commonly used as initial values before accumulating results.</para>
    /// </remarks>
    public new Matrix<T> Zeros(int rows, int cols)
    {
        return (Matrix<T>)base.Zeros(rows, cols);
    }

    /// <summary>
    /// Creates a new matrix filled with ones.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A new matrix of the specified size filled with ones.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This static method creates a new matrix where every element has the value 1.
    /// It's a convenient way to create a ones matrix without first creating an empty matrix.</para>
    /// </remarks>
    public static Matrix<T> CreateOnes(int rows, int cols)
    {
        var matrix = new Matrix<T>(rows, cols);
        return matrix.Ones(rows, cols);
    }

    /// <summary>
    /// Creates a new matrix filled with zeros.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A new matrix of the specified size filled with zeros.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This static method creates a new matrix where every element has the value 0.
    /// It's a convenient way to create a zeros matrix without first creating an empty matrix.</para>
    /// </remarks>
    public static Matrix<T> CreateZeros(int rows, int cols)
    {
        var matrix = new Matrix<T>(rows, cols);
        return matrix.Zeros(rows, cols);
    }

    /// <summary>
    /// Creates a diagonal matrix from a vector.
    /// </summary>
    /// <param name="diagonal">The vector containing values for the diagonal.</param>
    /// <returns>A square matrix with the vector values on the diagonal and zeros elsewhere.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A diagonal matrix has values only along its main diagonal (top-left to bottom-right),
    /// with zeros everywhere else. This method creates such a matrix using the values from the provided vector.
    /// Diagonal matrices have special properties that make certain calculations simpler.</para>
    /// <para><b>Performance:</b> Uses vectorized Fill operation to initialize with zeros, then sets diagonal elements (5-10x faster than loops).</para>
    /// </remarks>
    public static Matrix<T> CreateDiagonal(Vector<T> diagonal)
    {
        var matrix = new Matrix<T>(diagonal.Length, diagonal.Length);

        // Use vectorized Fill to initialize entire matrix with zeros
        _numOps.Fill(matrix.AsWritableSpan(), _numOps.Zero);

        // Set diagonal elements from the vector
        for (int i = 0; i < diagonal.Length; i++)
        {
            matrix[i, i] = diagonal[i];
        }

        return matrix;
    }

    /// <summary>
    /// Creates an identity matrix of the specified size.
    /// </summary>
    /// <param name="size">The size of the square matrix.</param>
    /// <returns>An identity matrix of the specified size.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An identity matrix is a special diagonal matrix with 1s on the diagonal and 0s elsewhere.
    /// It works like the number 1 in multiplication - multiplying any matrix by the identity matrix gives you the original matrix.
    /// It's often used in linear algebra and machine learning algorithms.</para>
    /// <para><b>Performance:</b> Uses vectorized Fill operation to initialize with zeros, then sets diagonal elements (5-10x faster than loops).</para>
    /// </remarks>
    public static Matrix<T> CreateIdentity(int size)
    {
        var identity = new Matrix<T>(size, size);

        // Use vectorized Fill to initialize entire matrix with zeros
        _numOps.Fill(identity.AsWritableSpan(), _numOps.Zero);

        // Set diagonal elements to one
        for (int i = 0; i < size; i++)
        {
            identity[i, i] = _numOps.One;
        }

        return identity;
    }

    /// <summary>
    /// Creates a matrix filled with random values between 0 and 1.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="columns">The number of columns.</param>
    /// <returns>A matrix of the specified size filled with random values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a matrix where each element is a random number between 0 and 1.
    /// Random matrices are often used as starting points in machine learning algorithms, especially for initializing
    /// weights in neural networks.</para>
    /// </remarks>
    public static Matrix<T> CreateRandom(int rows, int columns)
    {
        Matrix<T> matrix = new(rows, columns);
        var random = RandomHelper.CreateSecureRandom();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = _numOps.FromDouble(random.NextDouble());
            }
        }

        return matrix;
    }

    /// <summary>
    /// Creates a matrix filled with a specified default value.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="columns">The number of columns.</param>
    /// <param name="defaultValue">The value to fill the matrix with.</param>
    /// <returns>A matrix of the specified size filled with the default value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a matrix where every element has the same specified value.
    /// It's useful when you need a matrix with a specific starting value other than 0 or 1.</para>
    /// <para><b>Performance:</b> Uses vectorized Fill operation for SIMD acceleration (5-15x faster with AVX2).</para>
    /// </remarks>
    public static Matrix<T> CreateDefault(int rows, int columns, T defaultValue)
    {
        var matrix = new Matrix<T>(rows, columns);
        _numOps.Fill(matrix.AsWritableSpan(), defaultValue);
        return matrix;
    }

    /// <summary>
    /// Creates a matrix with random values within a specified range.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <param name="min">The minimum value for random elements (default is -1.0).</param>
    /// <param name="max">The maximum value for random elements (default is 1.0).</param>
    /// <returns>A new matrix filled with random values.</returns>
    /// <exception cref="ArgumentException">Thrown when minimum value is greater than or equal to maximum value.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix filled with random numbers. 
    /// Each number will be between the min and max values you specify. Random matrices are often 
    /// used as starting points in machine learning algorithms.</para>
    /// </remarks>
    public static Matrix<T> CreateRandom(int rows, int columns, double min = -1.0, double max = 1.0)
    {
        if (min >= max)
            throw new ArgumentException("Minimum value must be less than maximum value");

        var random = RandomHelper.CreateSecureRandom();
        var matrix = new Matrix<T>(rows, columns);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                // Generate random value between min and max
                double randomValue = random.NextDouble() * (max - min) + min;
                matrix[i, j] = _numOps.FromDouble(randomValue);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Creates a block diagonal matrix from multiple matrices.
    /// </summary>
    /// <param name="matrices">The matrices to place on the diagonal.</param>
    /// <returns>A new block diagonal matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when matrices is null.</exception>
    /// <exception cref="ArgumentException">Thrown when matrices is empty or contains null elements.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> A block diagonal matrix is a special matrix where smaller matrices are placed
    /// along the diagonal, with zeros everywhere else. It's like placing each input matrix in its own
    /// section of a larger matrix, with no overlap between them.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row for SIMD acceleration when copying blocks.</para>
    /// </remarks>
    public static Matrix<T> BlockDiagonal(params Matrix<T>[] matrices)
    {
        if (matrices == null)
            throw new ArgumentNullException(nameof(matrices));

        if (matrices.Length == 0)
            throw new ArgumentException("Matrices array cannot be empty", nameof(matrices));

        for (int i = 0; i < matrices.Length; i++)
        {
            if (matrices[i] == null)
                throw new ArgumentException($"Matrix at index {i} is null", nameof(matrices));
        }

        int totalRows = matrices.Sum(m => m.Rows);
        int totalCols = matrices.Sum(m => m.Columns);
        Matrix<T> result = new(totalRows, totalCols);

        int rowOffset = 0;
        int colOffset = 0;
        foreach (var matrix in matrices)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                var sourceRowSpan = matrix.GetRowReadOnlySpan(i);
                var destRow = result.GetRowSpan(rowOffset + i);
                var destSlice = destRow.Slice(colOffset, matrix.Columns);
                _numOps.Copy(sourceRowSpan, destSlice);
            }
            rowOffset += matrix.Rows;
            colOffset += matrix.Columns;
        }

        return result;
    }

    /// <summary>
    /// Creates an empty matrix with zero rows and columns.
    /// </summary>
    /// <returns>A new empty matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An empty matrix has no elements at all. It's useful as a starting point
    /// when you need to build a matrix from scratch or represent the absence of data.</para>
    /// </remarks>
    public new static Matrix<T> Empty()
    {
        return new Matrix<T>(0, 0);
    }

    /// <summary>
    /// Creates a matrix from a collection of column vectors.
    /// </summary>
    /// <param name="vectors">The collection of vectors to use as columns.</param>
    /// <returns>A new matrix with the given vectors as columns.</returns>
    /// <exception cref="ArgumentNullException">Thrown when vectors is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the vector list is empty or vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a collection of vectors and arranges them side by side
    /// to form a matrix. Each vector becomes one column in the resulting matrix. All vectors must have
    /// the same length to create a valid matrix.</para>
    /// </remarks>
    public static Matrix<T> FromColumnVectors(IEnumerable<IEnumerable<T>> vectors)
    {
        if (vectors == null)
            throw new ArgumentNullException(nameof(vectors));

        var vectorList = new List<List<T>>();
        foreach (var vector in vectors)
        {
            if (vector == null)
                throw new ArgumentException("Vector collection contains null elements", nameof(vectors));
            vectorList.Add(vector.ToList());
        }

        if (vectorList.Count == 0)
            throw new ArgumentException("Vector list cannot be empty", nameof(vectors));

        int rows = vectorList[0].Count;
        if (vectorList.Any(v => v.Count != rows))
            throw new ArgumentException("All vectors must have the same length", nameof(vectors));

        var matrix = new Matrix<T>(rows, vectorList.Count);

        for (int j = 0; j < vectorList.Count; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                matrix[i, j] = vectorList[j][i];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Subtracts another matrix from this matrix.
    /// </summary>
    /// <param name="other">The matrix to subtract.</param>
    /// <returns>A new matrix that is the result of the subtraction.</returns>
    /// <exception cref="ArgumentException">Thrown when matrices have different dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operation subtracts each element of the second matrix from the 
    /// corresponding element in this matrix. Both matrices must have the same number of rows and columns.</para>
    /// </remarks>
    public Matrix<T> Subtract(Matrix<T> other)
    {
        if (this.Rows != other.Rows || this.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for subtraction.");
        }

        Matrix<T> result = new(Rows, Columns);
        // Use vectorized Subtract operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Subtract(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Gets a segment of a column as a vector.
    /// </summary>
    /// <param name="columnIndex">The index of the column.</param>
    /// <param name="startRow">The starting row index.</param>
    /// <param name="length">The number of elements to include.</param>
    /// <returns>A vector containing the specified segment of the column.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a portion of a column from the matrix.
    /// It's like taking a slice from a specific column, starting at a particular row and
    /// continuing for a specified number of elements.</para>
    /// </remarks>
    public Vector<T> GetColumnSegment(int columnIndex, int startRow, int length)
    {
        var result = new Vector<T>(length);
        var destSpan = result.AsWritableSpan();
        for (int i = 0; i < length; i++)
        {
            destSpan[i] = _data[(startRow + i) * _cols + columnIndex];
        }
        return result;
    }

    /// <summary>
    /// Gets a segment of a row as a vector.
    /// </summary>
    /// <param name="rowIndex">The index of the row.</param>
    /// <param name="startColumn">The starting column index.</param>
    /// <param name="length">The number of elements to include.</param>
    /// <returns>A vector containing the specified segment of the row.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a portion of a row from the matrix.
    /// It's like taking a slice from a specific row, starting at a particular column and
    /// continuing for a specified number of elements.</para>
    /// <para><b>Performance:</b> Uses SIMD-accelerated Copy since row segments are contiguous.</para>
    /// </remarks>
    public Vector<T> GetRowSegment(int rowIndex, int startColumn, int length)
    {
        var result = new Vector<T>(length);
        var sourceSpan = new ReadOnlySpan<T>(_data, rowIndex * _cols + startColumn, length);
        _numOps.Copy(sourceSpan, result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Extracts a sub-matrix from this matrix.
    /// </summary>
    /// <param name="startRow">The starting row index.</param>
    /// <param name="startColumn">The starting column index.</param>
    /// <param name="rowCount">The number of rows to include.</param>
    /// <param name="columnCount">The number of columns to include.</param>
    /// <returns>A new matrix that is a subset of this matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a smaller matrix from within the larger one.
    /// Think of it like cutting out a rectangular section from the original matrix, starting at the
    /// specified row and column, and including the specified number of rows and columns.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row when extracting full-width submatrices.</para>
    /// </remarks>
    public Matrix<T> GetSubMatrix(int startRow, int startColumn, int rowCount, int columnCount)
    {
        Matrix<T> subMatrix = new(rowCount, columnCount);

        if (startColumn == 0 && columnCount == Columns)
        {
            for (int i = 0; i < rowCount; i++)
            {
                _numOps.Copy(GetRowReadOnlySpan(startRow + i), subMatrix.GetRowSpan(i));
            }
        }
        else
        {
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    subMatrix[i, j] = this[startRow + i, startColumn + j];
                }
            }
        }

        return subMatrix;
    }

    /// <summary>
    /// Converts the matrix to a column vector by stacking columns.
    /// </summary>
    /// <returns>A vector containing all elements of the matrix, arranged column by column.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes all the values in the matrix and puts them into a single vector.
    /// It goes down each column one by one, taking all values from the first column, then the second column, and so on.</para>
    /// </remarks>
    public Vector<T> ToColumnVector()
    {
        Vector<T> result = new(Rows * Columns);
        int index = 0;
        for (int j = 0; j < Columns; j++)
        {
            for (int i = 0; i < Rows; i++)
            {
                result[index++] = this[i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Converts the matrix to a row vector by stacking rows.
    /// </summary>
    /// <returns>A vector containing all elements of the matrix, arranged row by row.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes all the values in the matrix and puts them into a single vector.
    /// It goes across each row one by one, taking all values from the first row, then the second row, and so on.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation since matrix data is stored in row-major order (contiguous).</para>
    /// </remarks>
    public Vector<T> ToRowVector()
    {
        Vector<T> result = new(Rows * Columns);
        _numOps.Copy(new ReadOnlySpan<T>(_data), result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Adds a tensor to this matrix.
    /// </summary>
    /// <param name="tensor">The tensor to add to this matrix.</param>
    /// <returns>A new matrix containing the sum of this matrix and the tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor dimensions don't match matrix dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method combines two mathematical objects (a matrix and a tensor) by adding
    /// their corresponding values together. A tensor is a mathematical object that can represent multi-dimensional data.
    /// For this operation to work, the tensor must have the same shape as the matrix.</para>
    /// <para><b>Performance:</b> Uses vectorized Add operation for SIMD acceleration (5-15x faster with AVX2).</para>
    /// </remarks>
    public Matrix<T> Add(Tensor<T> tensor)
    {
        if (tensor.Shape.Length != 2 || tensor.Shape[0] != Rows || tensor.Shape[1] != Columns)
        {
            throw new ArgumentException("Tensor dimensions must match matrix dimensions for addition.");
        }

        var result = new Matrix<T>(Rows, Columns);
        _numOps.Add(new ReadOnlySpan<T>(_data), tensor.AsSpan(), result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Sets a portion of this matrix to the values from another matrix.
    /// </summary>
    /// <param name="startRow">The starting row index where the submatrix will be placed.</param>
    /// <param name="startColumn">The starting column index where the submatrix will be placed.</param>
    /// <param name="subMatrix">The matrix containing values to insert.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to insert a smaller matrix into a specific position
    /// within this larger matrix. Think of it like pasting a small image into a specific location of a larger image.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row for SIMD acceleration when possible.</para>
    /// </remarks>
    public void SetSubMatrix(int startRow, int startColumn, Matrix<T> subMatrix)
    {
        if (startColumn == 0 && subMatrix.Columns == Columns)
        {
            for (int i = 0; i < subMatrix.Rows; i++)
            {
                _numOps.Copy(subMatrix.GetRowReadOnlySpan(i), GetRowSpan(startRow + i));
            }
        }
        else
        {
            for (int i = 0; i < subMatrix.Rows; i++)
            {
                for (int j = 0; j < subMatrix.Columns; j++)
                {
                    this[startRow + i, startColumn + j] = subMatrix[i, j];
                }
            }
        }
    }

    /// <summary>
    /// Creates a matrix from a collection of row vectors.
    /// </summary>
    /// <param name="vectors">The row vectors to form the matrix.</param>
    /// <returns>A new matrix with each vector as a row.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix by stacking multiple vectors on top of each other.
    /// Each vector becomes one row in the resulting matrix.</para>
    /// </remarks>
    public static Matrix<T> FromRows(params IEnumerable<T>[] vectors)
    {
        return FromRowVectors(vectors);
    }

    /// <summary>
    /// Creates a matrix from a collection of column vectors.
    /// </summary>
    /// <param name="vectors">The column vectors to form the matrix.</param>
    /// <returns>A new matrix with each vector as a column.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix by placing multiple vectors side by side.
    /// Each vector becomes one column in the resulting matrix.</para>
    /// </remarks>
    public static Matrix<T> FromColumns(params IEnumerable<T>[] vectors)
    {
        return FromColumnVectors(vectors);
    }

    /// <summary>
    /// Creates a matrix from a single vector.
    /// </summary>
    /// <param name="vector">The vector to convert to a matrix.</param>
    /// <returns>A new matrix with a single column containing the vector's values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a vector (a one-dimensional array of numbers) into a matrix
    /// with a single column. Each element of the vector becomes a row in the matrix.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation for SIMD acceleration.</para>
    /// </remarks>
    public static Matrix<T> FromVector(Vector<T> vector)
    {
        var matrix = new Matrix<T>(vector.Length, 1);
        _numOps.Copy(vector.AsSpan(), new Span<T>(matrix._data));
        return matrix;
    }

    /// <summary>
    /// Creates a matrix from a collection of row vectors.
    /// </summary>
    /// <param name="vectors">The collection of vectors to form the rows of the matrix.</param>
    /// <returns>A new matrix with each vector as a row.</returns>
    /// <exception cref="ArgumentNullException">Thrown when vectors is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the vector list is empty or vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix by stacking multiple vectors on top of each other.
    /// Each vector becomes one row in the resulting matrix. All vectors must have the same length to form a valid matrix.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row for SIMD acceleration.</para>
    /// </remarks>
    public static Matrix<T> FromRowVectors(IEnumerable<IEnumerable<T>> vectors)
    {
        if (vectors == null)
            throw new ArgumentNullException(nameof(vectors));

        var vectorList = new List<List<T>>();
        foreach (var vector in vectors)
        {
            if (vector == null)
                throw new ArgumentException("Vector collection contains null elements", nameof(vectors));
            vectorList.Add(vector.ToList());
        }

        if (vectorList.Count == 0)
            throw new ArgumentException("Vector list cannot be empty", nameof(vectors));

        int cols = vectorList[0].Count;
        if (vectorList.Any(v => v.Count != cols))
            throw new ArgumentException("All vectors must have the same length", nameof(vectors));

        var matrix = new Matrix<T>(vectorList.Count, cols);

        for (int i = 0; i < vectorList.Count; i++)
        {
            T[] rowArray = vectorList[i].ToArray();
            _numOps.Copy(new ReadOnlySpan<T>(rowArray), matrix.GetRowSpan(i));
        }

        return matrix;
    }

    /// <summary>
    /// Finds the maximum value in each row of the matrix.
    /// </summary>
    /// <returns>A vector containing the maximum value from each row.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method examines each row of the matrix and finds the largest number in that row.
    /// It then returns a vector (a one-dimensional array) where each element is the maximum value from the corresponding row.</para>
    /// </remarks>
    public Vector<T> RowWiseMax()
    {
        if (Columns == 0)
            throw new InvalidOperationException("Cannot compute row-wise maximum of a matrix with zero columns");

        Vector<T> result = new(Rows);
        for (int i = 0; i < Rows; i++)
        {
            // Use vectorized Max for each row (8-12x speedup with AVX2)
            result[i] = _numOps.Max(GetRowReadOnlySpan(i));
        }

        return result;
    }

    /// <summary>
    /// Applies a transformation function to each element of the matrix.
    /// </summary>
    /// <param name="transformer">A function that takes the current value, row index, and column index and returns a new value.</param>
    /// <returns>A new matrix with transformed values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change every value in the matrix using a custom function.
    /// The function receives the current value and its position (row and column) and returns the new value to use.
    /// This is useful for operations like scaling all values, applying mathematical functions, or conditional transformations.</para>
    /// </remarks>
    public Matrix<T> Transform(Func<T, int, int, T> transformer)
    {
        Matrix<T> result = new(Rows, Columns);

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[i, j] = transformer(this[i, j], i, j);
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the sum of values in each row of the matrix.
    /// </summary>
    /// <returns>A vector containing the sum of each row.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds up all the numbers in each row of the matrix.
    /// It returns a vector (a one-dimensional array) where each element is the sum of the corresponding row in the matrix.</para>
    /// </remarks>
    public Vector<T> RowWiseSum()
    {
        Vector<T> result = new(Rows);

        for (int i = 0; i < Rows; i++)
        {
            // Use vectorized Sum for each row (8-12x speedup with AVX2)
            result[i] = _numOps.Sum(GetRowReadOnlySpan(i));
        }

        return result;
    }

    /// <summary>
    /// Divides each element of this matrix by the corresponding element in another matrix.
    /// </summary>
    /// <param name="other">The matrix to divide by.</param>
    /// <returns>A new matrix containing the result of the division.</returns>
    /// <exception cref="ArgumentException">Thrown when matrices have different dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method performs element-by-element division between two matrices.
    /// Each element in this matrix is divided by the corresponding element in the other matrix.
    /// Both matrices must have the same number of rows and columns.</para>
    /// </remarks>
    public Matrix<T> PointwiseDivide(Matrix<T> other)
    {
        return Divide(other);
    }

    /// <summary>
    /// Divides each element of the matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new matrix with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method divides every value in the matrix by the same number (scalar).
    /// For example, dividing a matrix by 2 would halve all values in the matrix.</para>
    /// <para><b>Performance:</b> Uses SIMD-accelerated operations (5-15x faster with AVX2).</para>
    /// </remarks>
    public Matrix<T> Divide(T scalar)
    {
        Matrix<T> result = new(Rows, Columns);
        _numOps.DivideScalar(new ReadOnlySpan<T>(_data), scalar, result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Divides each element of the matrix by a scalar value in-place.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated division.</para>
    /// </remarks>
    public void DivideInPlace(T scalar)
    {
        _numOps.DivideScalar(new ReadOnlySpan<T>(_data), scalar, new Span<T>(_data));
    }

    /// <summary>
    /// Divides each element of this matrix by the corresponding element in another matrix.
    /// </summary>
    /// <param name="other">The matrix to divide by.</param>
    /// <returns>A new matrix containing the result of the division.</returns>
    /// <exception cref="ArgumentException">Thrown when matrices have different dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method performs element-by-element division between two matrices.
    /// Each element in this matrix is divided by the corresponding element in the other matrix at the same position.
    /// Both matrices must have the same number of rows and columns for this operation to work.</para>
    /// </remarks>
    public Matrix<T> Divide(Matrix<T> other)
    {
        if (this.Rows != other.Rows || this.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for division.");
        }

        Matrix<T> result = new(Rows, Columns);
        // Use vectorized Divide operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Divide(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Calculates the outer product of two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A matrix representing the outer product of the two vectors.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either vector is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> The outer product is a way to multiply two vectors to create a matrix.
    /// If vector a has length m and vector b has length n, the result will be an m x n matrix.
    /// Each element (i,j) in the resulting matrix is calculated by multiplying the i-th element of vector a
    /// by the j-th element of vector b. This operation is useful in many machine learning algorithms.</para>
    /// <para><b>Performance:</b> Uses vectorized MultiplyScalar operation per row for SIMD acceleration (5-15x faster with AVX2).</para>
    /// </remarks>
    public static Matrix<T> OuterProduct(Vector<T> a, Vector<T> b)
    {
        if (a == null || b == null)
        {
            throw new ArgumentNullException(a == null ? nameof(a) : nameof(b), "Vectors cannot be null.");
        }

        int rows = a.Length;
        int cols = b.Length;
        var result = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            _numOps.MultiplyScalar(b.AsSpan(), a[i], result.GetRowSpan(i));
        }

        return result;
    }

    /// <summary>
    /// Converts this matrix to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Serialization is the process of converting an object (in this case, a matrix)
    /// into a format that can be easily stored or transmitted. This method converts the matrix into a sequence of bytes
    /// that can be saved to a file or sent over a network. You can later reconstruct the original matrix using the
    /// Deserialize method.</para>
    /// </remarks>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write dimensions
        writer.Write(Rows);
        writer.Write(Columns);

        // Write each element as bytes (row-major order)
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                double value = _numOps.ToDouble(this[i, j]);
                writer.Write(value);
            }
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a matrix from a previously serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized matrix data.</param>
    /// <returns>A matrix reconstructed from the serialized data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reconstructs a matrix from a byte array that was previously created
    /// using the Serialize method. It's like unpacking a compressed file to get back the original content.</para>
    /// </remarks>
    public static Matrix<T> Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read dimensions
        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();

        // Read each element
        var result = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                double value = reader.ReadDouble();
                result[i, j] = _numOps.FromDouble(value);
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a new matrix containing a subset of consecutive rows from this matrix.
    /// </summary>
    /// <param name="startRow">The index of the first row to include.</param>
    /// <param name="rowCount">The number of rows to include.</param>
    /// <returns>A new matrix containing the specified rows.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when startRow or rowCount are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a portion of the matrix by selecting a specific range of rows.
    /// Think of it like cutting out a horizontal strip from the matrix. The new matrix will have the same number of columns
    /// as the original, but only include the rows you specified.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row for SIMD acceleration.</para>
    /// </remarks>
    public new Matrix<T> Slice(int startRow, int rowCount)
    {
        if (startRow < 0 || startRow >= Rows)
            throw new ArgumentOutOfRangeException(nameof(startRow));
        if (rowCount < 1 || startRow + rowCount > Rows)
            throw new ArgumentOutOfRangeException(nameof(rowCount));

        Matrix<T> result = new Matrix<T>(rowCount, Columns);
        for (int i = 0; i < rowCount; i++)
        {
            _numOps.Copy(GetRowReadOnlySpan(startRow + i), result.GetRowSpan(i));
        }

        return result;
    }

    /// <summary>
    /// Gets all columns of the matrix as a sequence of vectors.
    /// </summary>
    /// <returns>An enumerable collection of vectors, each representing a column of the matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method provides a way to access each column of the matrix as a separate vector.
    /// A vector is essentially a one-dimensional array of numbers. This is useful when you need to process each column
    /// individually, such as in feature extraction or statistical analysis.</para>
    /// </remarks>
    public IEnumerable<Vector<T>> GetColumns()
    {
        for (var i = 0; i < Columns; i++)
        {
            yield return GetColumn(i);
        }
    }

    /// <summary>
    /// Gets all rows of the matrix as a sequence of vectors.
    /// </summary>
    /// <returns>An enumerable collection of vectors, each representing a row of the matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method provides a way to access each row of the matrix as a separate vector.
    /// This is useful when you need to process each row individually, such as when each row represents a different
    /// data sample or observation in your dataset.</para>
    /// </remarks>
    public IEnumerable<Vector<T>> GetRows()
    {
        for (var i = 0; i < Rows; i++)
        {
            yield return GetRow(i);
        }
    }

    /// <summary>
    /// Creates a new matrix with a specified row removed.
    /// </summary>
    /// <param name="rowIndex">The index of the row to remove.</param>
    /// <returns>A new matrix with the specified row removed.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when rowIndex is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a copy of the matrix but leaves out one specific row.
    /// The resulting matrix will have one fewer row than the original. This is useful in data preprocessing
    /// when you need to exclude certain observations from your dataset.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row for SIMD acceleration.</para>
    /// </remarks>
    public Matrix<T> RemoveRow(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));

        var newMatrix = new Matrix<T>(Rows - 1, Columns);
        int newRow = 0;

        for (int i = 0; i < Rows; i++)
        {
            if (i == rowIndex) continue;
            _numOps.Copy(GetRowReadOnlySpan(i), newMatrix.GetRowSpan(newRow));
            newRow++;
        }

        return newMatrix;
    }

    /// <summary>
    /// Creates a new matrix with a specified column removed.
    /// </summary>
    /// <param name="columnIndex">The index of the column to remove.</param>
    /// <returns>A new matrix with the specified column removed.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when columnIndex is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a copy of the matrix but leaves out one specific column.
    /// The resulting matrix will have one fewer column than the original. This is useful in feature selection
    /// when you want to exclude a particular feature (column) from your dataset.</para>
    /// </remarks>
    public Matrix<T> RemoveColumn(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        var newMatrix = new Matrix<T>(Rows, Columns - 1);

        for (int i = 0; i < Rows; i++)
        {
            int newColumn = 0;
            for (int j = 0; j < Columns; j++)
            {
                if (j == columnIndex) continue;
                newMatrix[i, newColumn] = this[i, j];
                newColumn++;
            }
        }

        return newMatrix;
    }

    /// <summary>
    /// Creates a new matrix containing only the specified rows from this matrix.
    /// </summary>
    /// <param name="indices">The indices of the rows to include in the new matrix.</param>
    /// <returns>A new matrix containing only the specified rows.</returns>
    /// <exception cref="ArgumentNullException">Thrown when indices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any index is negative or greater than or equal to the number of rows.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to select specific rows from the matrix and create a new matrix
    /// containing only those rows. This is useful for data sampling or when you need to extract a subset of your data
    /// based on certain criteria. For example, you might use this to select only the data points that belong to a
    /// particular category.</para>
    /// <para><b>Performance:</b> Uses vectorized Copy operation per row for SIMD acceleration.</para>
    /// </remarks>
    public Matrix<T> GetRows(IEnumerable<int> indices)
    {
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        var indexArray = indices.ToArray();

        for (int i = 0; i < indexArray.Length; i++)
        {
            if (indexArray[i] < 0 || indexArray[i] >= Rows)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {indexArray[i]} at position {i} is out of range. Valid range is 0 to {Rows - 1}.");
        }

        var newRows = indexArray.Length;
        var result = new Matrix<T>(newRows, Columns);
        for (int i = 0; i < newRows; i++)
        {
            _numOps.Copy(GetRowReadOnlySpan(indexArray[i]), result.GetRowSpan(i));
        }

        return result;
    }

    /// <summary>
    /// Returns an enumerator that iterates through the matrix elements.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to iterate through all elements of the matrix
    /// in a row-by-row manner. It's useful when you need to process each element of the matrix sequentially,
    /// regardless of its position in rows or columns.</para>
    /// </remarks>
    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                yield return this[i, j];
            }
        }
    }

    /// <summary>
    /// Returns an enumerator that iterates through the matrix elements.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is an implementation of the non-generic IEnumerable interface.
    /// It allows the matrix to be used in foreach loops and other constructs that expect an IEnumerable.
    /// It simply calls the generic GetEnumerator method above.</para>
    /// </remarks>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <summary>
    /// Gets a span over a specific row of the matrix for efficient SIMD operations.
    /// </summary>
    /// <param name="rowIndex">The index of the row.</param>
    /// <returns>A Span representing the row's data.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when rowIndex is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> A Span provides a high-performance, zero-allocation view over a matrix row.
    /// This is efficient because matrix data is stored in row-major order (rows are contiguous in memory).
    /// Use this for SIMD vectorization with TensorPrimitives.</para>
    /// </remarks>
    public Span<T> GetRowSpan(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        int startIndex = rowIndex * Columns;
        return _data.AsSpan(startIndex, Columns);
    }

    /// <summary>
    /// Gets a read-only span over a specific row of the matrix for efficient SIMD operations.
    /// </summary>
    /// <param name="rowIndex">The index of the row.</param>
    /// <returns>A ReadOnlySpan representing the row's data.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when rowIndex is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> A ReadOnlySpan provides a high-performance, zero-allocation view over a matrix row
    /// that prevents modifications. This is efficient for reading row data without copying.</para>
    /// </remarks>
    public ReadOnlySpan<T> GetRowReadOnlySpan(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        int startIndex = rowIndex * Columns;
        return _data.AsSpan(startIndex, Columns);
    }

    /// <summary>
    /// Gets a column from the matrix as an array for use with Span operations.
    /// </summary>
    /// <param name="columnIndex">The index of the column.</param>
    /// <returns>An array containing the column data.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when columnIndex is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Unlike rows, columns are not stored contiguously in memory
    /// (due to row-major storage). This method copies the column data into a new array to enable Span access.
    /// For performance-critical code, prefer GetRowSpan when possible.</para>
    /// </remarks>
    public T[] GetColumnAsArray(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        T[] columnData = new T[Rows];
        for (int i = 0; i < Rows; i++)
        {
            columnData[i] = this[i, columnIndex];
        }
        return columnData;
    }
}
