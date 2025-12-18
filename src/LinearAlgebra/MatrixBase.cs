global using System.Text;

namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Base class for matrix operations in the AiDotNet library.
/// </summary>
/// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A matrix is a rectangular array of numbers arranged in rows and columns.
/// Matrices are fundamental in machine learning for representing data and transformations.</para>
/// </remarks>
public abstract class MatrixBase<T>
{
    /// <summary>
    /// The internal array storing matrix data in a flattened format.
    /// </summary>
    protected readonly T[] _data;
    
    /// <summary>
    /// The number of rows in the matrix.
    /// </summary>
    protected readonly int _rows;
    
    /// <summary>
    /// The number of columns in the matrix.
    /// </summary>
    protected readonly int _cols;
    
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    protected static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a new matrix with the specified dimensions.
    /// </summary>
    /// <param name="rows">Number of rows in the matrix.</param>
    /// <param name="cols">Number of columns in the matrix.</param>
    /// <exception cref="ArgumentException">Thrown when rows or columns are not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates an empty matrix with the given size.
    /// For example, a matrix with 3 rows and 2 columns would look like:
    /// [0, 0]
    /// [0, 0]
    /// [0, 0]
    /// where each value is initially the default value for type T.</para>
    /// </remarks>
    protected MatrixBase(int rows, int cols)
    {
        if (rows <= 0) throw new ArgumentException("Rows must be positive", nameof(rows));
        if (cols <= 0) throw new ArgumentException("Columns must be positive", nameof(cols));

        this._rows = rows;
        this._cols = cols;
        this._data = new T[rows * cols];
    }

    /// <summary>
    /// Creates a matrix from a collection of row values.
    /// </summary>
    /// <param name="values">A collection where each inner collection represents a row of the matrix.</param>
    /// <exception cref="ArgumentException">Thrown when rows have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates a matrix from a list of lists, where each inner list
    /// represents one row of the matrix. All rows must have the same number of elements.</para>
    /// </remarks>
    protected MatrixBase(IEnumerable<IEnumerable<T>> values)
    {
        var valuesList = values.Select(v => v.ToArray()).ToList();
        this._rows = valuesList.Count;
        this._cols = valuesList.First().Length;
        this._data = new T[_rows * _cols];

        for (int i = 0; i < _rows; i++)
        {
            var row = valuesList[i];
            if (row.Length != _cols)
            {
                throw new ArgumentException("All rows must have the same number of columns.", nameof(values));
            }

            for (int j = 0; j < _cols; j++)
            {
                _data[i * _cols + j] = row[j];
            }
        }
    }

    /// <summary>
    /// Creates a matrix from a 2D array.
    /// </summary>
    /// <param name="data">The 2D array containing matrix values.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates a matrix from a 2D array (an array of arrays).
    /// The first dimension represents rows, and the second dimension represents columns.</para>
    /// </remarks>
    protected MatrixBase(T[,] data)
    {
        this._rows = data.GetLength(0);
        this._cols = data.GetLength(1);
        this._data = new T[_rows * _cols];

        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                this._data[i * _cols + j] = data[i, j];
            }
        }
    }

    /// <summary>
    /// Gets the number of rows in the matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of rows is the height of the matrix.</para>
    /// </remarks>
    public int Rows => _rows;

    /// <summary>
    /// Gets the number of columns in the matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of columns is the width of the matrix.</para>
    /// </remarks>
    public int Columns => _cols;

    /// <summary>
    /// Checks if the matrix is empty (has zero rows or columns).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> An empty matrix has either no rows or no columns.</para>
    /// </remarks>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>
    /// Gets or sets the element at the specified position in the matrix.
    /// </summary>
    /// <param name="row">The row index (zero-based).</param>
    /// <param name="col">The column index (zero-based).</param>
    /// <returns>The value at the specified position.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when indices are out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This indexer allows you to access or change individual elements in the matrix.
    /// For example, matrix[2, 3] accesses the element in the 3rd row and 4th column (since indices start at 0).</para>
    /// </remarks>
    public virtual T this[int row, int col]
    {
        get
        {
            ValidateIndices(row, col);
            return _data[row * _cols + col];
        }
        set
        {
            ValidateIndices(row, col);
            _data[row * _cols + col] = value;
        }
    }

    /// <summary>
    /// Creates a matrix filled with ones.
    /// </summary>
    /// <param name="rows">Number of rows in the matrix.</param>
    /// <param name="cols">Number of columns in the matrix.</param>
    /// <returns>A matrix of the specified size filled with ones.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix where every element is 1.
    /// Ones matrices are often used in machine learning for initialization or transformation purposes.</para>
    /// </remarks>
    public virtual MatrixBase<T> Ones(int rows, int cols)
    {
        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = _numOps.One;

        return result;
    }

    /// <summary>
    /// Creates a matrix filled with zeros.
    /// </summary>
    /// <param name="rows">Number of rows in the matrix.</param>
    /// <param name="cols">Number of columns in the matrix.</param>
    /// <returns>A matrix of the specified size filled with zeros.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix where every element is 0.
    /// Zero matrices are commonly used as starting points for many algorithms.</para>
    /// </remarks>
    public virtual MatrixBase<T> Zeros(int rows, int cols)
    {
        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = _numOps.Zero;

        return result;
    }

    /// <summary>
    /// Creates a new matrix containing a subset of rows from this matrix.
    /// </summary>
    /// <param name="startRow">The index of the first row to include.</param>
    /// <param name="rowCount">The number of rows to include.</param>
    /// <returns>A new matrix containing the specified rows.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when row indices are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a portion of the matrix by selecting specific rows.
    /// It's like cutting out a horizontal strip from the matrix.</para>
    /// </remarks>
    public virtual MatrixBase<T> Slice(int startRow, int rowCount)
    {
        if (startRow < 0 || startRow >= _rows)
            throw new ArgumentOutOfRangeException(nameof(startRow));
        if (rowCount < 1 || startRow + rowCount > _rows)
            throw new ArgumentOutOfRangeException(nameof(rowCount));

        MatrixBase<T> result = new Matrix<T>(rowCount, _cols);
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                result[i, j] = this[startRow + i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Sets the values of a column in the matrix.
    /// </summary>
    /// <param name="columnIndex">The index of the column to set.</param>
    /// <param name="vector">The vector containing values to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when column index is out of range.</exception>
    /// <exception cref="ArgumentException">Thrown when vector length doesn't match row count.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method replaces an entire column of the matrix with new values.
    /// The vector must have the same number of elements as the matrix has rows.</para>
    /// </remarks>
    public virtual void SetColumn(int columnIndex, Vector<T> vector)
    {
        if (columnIndex < 0 || columnIndex >= Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        if (vector.Length != Rows)
            throw new ArgumentException("Vector length must match matrix row count");
        for (int i = 0; i < Rows; i++)
        {
            this[i, columnIndex] = vector[i];
        }
    }

    /// <summary>
    /// Sets the values of a row in the matrix.
    /// </summary>
    /// <param name="rowIndex">The index of the row to set.</param>
    /// <param name="vector">The vector containing values to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when row index is out of range.</exception>
    /// <exception cref="ArgumentException">Thrown when vector length doesn't match column count.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method replaces an entire row of the matrix with new values.
    /// The vector must have the same number of elements as the matrix has columns.</para>
    /// </remarks>
    public virtual void SetRow(int rowIndex, Vector<T> vector)
    {
        if (rowIndex < 0 || rowIndex >= Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        if (vector.Length != Columns)
            throw new ArgumentException("Vector length must match matrix column count");
        for (int j = 0; j < Columns; j++)
        {
            this[rowIndex, j] = vector[j];
        }
    }

    /// <summary>
    /// Creates an empty matrix with zero rows and columns.
    /// </summary>
    /// <returns>An empty matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An empty matrix is a matrix with no elements (0 rows and 0 columns).
    /// This is useful as a placeholder or when you need to initialize a matrix variable before determining its actual size.</para>
    /// </remarks>
    public static MatrixBase<T> Empty()
    {
        return new Matrix<T>(0, 0);
    }

    /// <summary>
    /// Gets a specific row from the matrix as a vector.
    /// </summary>
    /// <param name="row">The index of the row to retrieve.</param>
    /// <returns>A vector containing the values from the specified row.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when the row index is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a single row from the matrix and returns it as a vector.
    /// For example, if you have a 3×4 matrix and call GetRow(1), you'll get a vector with 4 elements containing
    /// all values from the second row (remember that indices start at 0).</para>
    /// </remarks>
    public virtual Vector<T> GetRow(int row)
    {
        ValidateIndices(row, 0);
        return new Vector<T>([.. Enumerable.Range(0, _cols).Select(col => this[row, col])]);
    }

    /// <summary>
    /// Gets a specific column from the matrix as a vector.
    /// </summary>
    /// <param name="col">The index of the column to retrieve.</param>
    /// <returns>A vector containing the values from the specified column.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when the column index is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a single column from the matrix and returns it as a vector.
    /// For example, if you have a 3×4 matrix and call GetColumn(2), you'll get a vector with 3 elements containing
    /// all values from the third column (remember that indices start at 0).</para>
    /// </remarks>
    public virtual Vector<T> GetColumn(int col)
    {
        ValidateIndices(0, col);
        return new Vector<T>([.. Enumerable.Range(0, _rows).Select(row => this[row, col])]);
    }

    /// <summary>
    /// Gets the diagonal elements of the matrix as a vector.
    /// </summary>
    /// <returns>A vector containing the diagonal elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The diagonal of a matrix consists of the elements where the row index equals the column index
    /// (e.g., positions [0,0], [1,1], [2,2], etc.). This method extracts these elements into a vector.
    /// The length of the diagonal vector will be the minimum of the matrix's row and column counts.</para>
    /// </remarks>
    public virtual Vector<T> Diagonal()
    {
        int minDimension = Math.Min(Rows, Columns);
        var diagonal = new Vector<T>(minDimension);

        for (int i = 0; i < minDimension; i++)
        {
            diagonal[i] = this[i, i];
        }

        return diagonal;
    }

    /// <summary>
    /// Creates a submatrix by extracting a rectangular portion of this matrix.
    /// </summary>
    /// <param name="startRow">The starting row index (inclusive).</param>
    /// <param name="startCol">The starting column index (inclusive).</param>
    /// <param name="numRows">The number of rows to extract.</param>
    /// <param name="numCols">The number of columns to extract.</param>
    /// <returns>A new matrix containing the specified portion of this matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when the specified region is outside the bounds of the matrix.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a rectangular portion of the matrix.
    /// Think of it like cutting out a rectangular section from the original matrix.
    /// For example, SubMatrix(1, 2, 3, 2) would extract a 3×2 matrix starting from position [1,2]
    /// (the 2nd row and 3rd column, since indices start at 0).</para>
    /// </remarks>
    public Matrix<T> SubMatrix(int startRow, int startCol, int numRows, int numCols)
    {
        if (startRow < 0 || startCol < 0 || startRow + numRows > Rows || startCol + numCols > Columns)
        {
            throw new ArgumentException("Invalid submatrix dimensions");
        }

        var subMatrix = new Matrix<T>(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                subMatrix[i, j] = this[startRow + i, startCol + j];
            }
        }

        return subMatrix;
    }

    /// <summary>
    /// Creates a submatrix by extracting specific rows and columns from this matrix.
    /// </summary>
    /// <param name="startRow">The starting row index (inclusive).</param>
    /// <param name="endRow">The ending row index (exclusive).</param>
    /// <param name="columnIndices">The list of column indices to include.</param>
    /// <returns>A new matrix containing the specified rows and columns.</returns>
    /// <exception cref="ArgumentException">Thrown when the specified indices are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new matrix by selecting specific rows and columns from the original matrix.
    /// It takes all rows from startRow up to (but not including) endRow, and only includes the columns specified in columnIndices.
    /// This is useful when you need to work with a specific subset of your data.</para>
    /// </remarks>
    public Matrix<T> SubMatrix(int startRow, int endRow, List<int> columnIndices)
    {
        if (startRow < 0 || endRow > Rows || startRow >= endRow)
        {
            throw new ArgumentException("Invalid row indices");
        }

        if (columnIndices.Any(i => i < 0 || i >= Columns))
        {
            throw new ArgumentException("Invalid column indices");
        }

        int numRows = endRow - startRow;
        int numCols = columnIndices.Count;

        var subMatrix = new Matrix<T>(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                subMatrix[i, j] = this[startRow + i, columnIndices[j]];
            }
        }

        return subMatrix;
    }

    /// <summary>
    /// Performs element-wise multiplication with another matrix and returns the sum of all products.
    /// </summary>
    /// <param name="other">The matrix to multiply with.</param>
    /// <returns>The sum of all element-wise products.</returns>
    /// <exception cref="ArgumentException">Thrown when matrices have different dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method multiplies each element of this matrix with the corresponding element
    /// in the other matrix, then adds up all these products to produce a single value.
    /// This is also known as the Frobenius inner product of two matrices.
    /// Both matrices must have exactly the same shape (same number of rows and columns).</para>
    /// </remarks>
    public virtual T ElementWiseMultiplyAndSum(MatrixBase<T> other)
    {
        if (Rows != other.Rows || Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for element-wise multiplication.");
        }

        T sum = _numOps.Zero;
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(this[i, j], other[i, j]));
            }
        }

        return sum;
    }

    /// <summary>
    /// Adds another matrix to this matrix.
    /// </summary>
    /// <param name="other">The matrix to add.</param>
    /// <returns>A new matrix containing the sum.</returns>
    /// <exception cref="ArgumentException">Thrown when matrices have different dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds each element of the other matrix to the corresponding element
    /// in this matrix. Both matrices must have exactly the same shape (same number of rows and columns).
    /// The result is a new matrix of the same size where each element is the sum of the corresponding elements
    /// from the two input matrices.</para>
    /// </remarks>
    public virtual MatrixBase<T> Add(MatrixBase<T> other)
    {
        if (_rows != other.Rows || _cols != other.Columns)
            throw new ArgumentException("Matrix dimensions must match for addition.");

        var result = CreateInstance(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result[i, j] = _numOps.Add(this[i, j], other[i, j]);

        return result;
    }

    /// <summary>
    /// Subtracts another matrix from this matrix.
    /// </summary>
    /// <param name="other">The matrix to subtract.</param>
    /// <returns>A new matrix containing the difference.</returns>
    /// <exception cref="ArgumentException">Thrown when matrices have different dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts each element of the other matrix from the corresponding element
    /// in this matrix. Both matrices must have exactly the same shape (same number of rows and columns).
    /// The result is a new matrix of the same size where each element is the difference between the corresponding elements
    /// from the two input matrices.</para>
    /// </remarks>
    public virtual MatrixBase<T> Subtract(MatrixBase<T> other)
    {
        if (_rows != other.Rows || _cols != other.Columns)
            throw new ArgumentException("Matrix dimensions must match for subtraction.");

        var result = CreateInstance(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result[i, j] = _numOps.Subtract(this[i, j], other[i, j]);

        return result;
    }

    /// <summary>
    /// Multiplies this matrix by another matrix.
    /// </summary>
    /// <param name="other">The matrix to multiply with.</param>
    /// <returns>A new matrix containing the product.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of columns in this matrix doesn't equal the number of rows in the other matrix.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Matrix multiplication is different from regular multiplication. 
    /// For two matrices to be multiplied, the first matrix must have the same number of columns as the second matrix has rows.
    /// The result will be a new matrix with the same number of rows as the first matrix and the same number of columns as the second matrix.
    /// Each element in the result is calculated by multiplying corresponding elements in a row of the first matrix with a column of the second matrix and summing them up.</para>
    /// </remarks>
    public virtual MatrixBase<T> Multiply(MatrixBase<T> other)
    {
        if (_cols != other.Rows)
            throw new ArgumentException("Number of columns in the first matrix must equal the number of rows in the second matrix.");

        var result = CreateInstance(_rows, other.Columns);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < other.Columns; j++)
                for (int k = 0; k < _cols; k++)
                    result[i, j] = _numOps.Add(result[i, j], _numOps.Multiply(this[i, k], other[k, j]));

        return result;
    }

    /// <summary>
    /// Multiplies this matrix by a vector.
    /// </summary>
    /// <param name="vector">The vector to multiply with.</param>
    /// <returns>A new vector containing the product.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of columns in the matrix doesn't equal the length of the vector.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> When multiplying a matrix by a vector, the vector is treated as a column vector.
    /// The number of columns in the matrix must equal the length of the vector.
    /// The result will be a new vector with the same number of elements as the matrix has rows.
    /// Each element in the result is calculated by multiplying corresponding elements in a row of the matrix with the vector and summing them up.
    /// This operation is commonly used in machine learning to apply transformations to data points.</para>
    /// </remarks>
    public virtual VectorBase<T> Multiply(Vector<T> vector)
    {
        if (_cols != vector.Length)
            throw new ArgumentException("Number of columns in the matrix must equal the length of the vector.");

        var result = new Vector<T>(_rows);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result[i] = _numOps.Add(result[i], _numOps.Multiply(this[i, j], vector[j]));

        return result;
    }

    /// <summary>
    /// Multiplies this matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply with.</param>
    /// <returns>A new matrix containing the product.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scalar multiplication means multiplying every element in the matrix by the same number (the scalar).
    /// The result is a new matrix of the same size where each element is the product of the corresponding element in the original matrix and the scalar value.
    /// This operation is useful for scaling data or adjusting the magnitude of values in a matrix.</para>
    /// </remarks>
    public virtual MatrixBase<T> Multiply(T scalar)
    {
        var result = CreateInstance(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result[i, j] = _numOps.Multiply(this[i, j], scalar);

        return result;
    }

    /// <summary>
    /// Creates a transposed version of this matrix.
    /// </summary>
    /// <returns>A new matrix that is the transpose of this matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The transpose of a matrix is created by flipping the matrix over its diagonal.
    /// This means that rows become columns and columns become rows.
    /// For example, if you have a 2×3 matrix, its transpose will be a 3×2 matrix.
    /// The element at position [i,j] in the original matrix will be at position [j,i] in the transposed matrix.
    /// Transposing is commonly used in many mathematical operations and algorithms.</para>
    /// </remarks>
    public virtual MatrixBase<T> Transpose()
    {
        var result = CreateInstance(_cols, _rows);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result[j, i] = this[i, j];

        return result;
    }

    /// <summary>
    /// Creates a deep copy of this matrix.
    /// </summary>
    /// <returns>A new matrix with the same values as this matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a completely new matrix with the same values as the original.
    /// Changes made to the copy won't affect the original matrix, and vice versa.
    /// This is useful when you need to preserve the original matrix while performing operations that would modify it.</para>
    /// </remarks>
    public virtual MatrixBase<T> Clone()
    {
        var result = CreateInstance(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result[i, j] = this[i, j];

        return result;
    }

    /// <summary>
    /// Creates a new instance of a matrix with the specified dimensions.
    /// </summary>
    /// <param name="rows">The number of rows for the new matrix.</param>
    /// <param name="cols">The number of columns for the new matrix.</param>
    /// <returns>A new matrix instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an abstract method that must be implemented by derived classes.
    /// It's used internally to create new matrices of the appropriate type during operations.
    /// You typically won't need to call this method directly.</para>
    /// </remarks>
    protected abstract MatrixBase<T> CreateInstance(int rows, int cols);

    /// <summary>
    /// Validates that the provided row and column indices are within the bounds of the matrix.
    /// </summary>
    /// <param name="row">The row index to validate.</param>
    /// <param name="col">The column index to validate.</param>
    /// <exception cref="IndexOutOfRangeException">Thrown when either index is outside the matrix bounds.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helper method checks if the row and column indices are valid for this matrix.
    /// Valid indices must be non-negative and less than the number of rows or columns in the matrix.
    /// This method is used internally to prevent accessing elements outside the matrix boundaries.</para>
    /// </remarks>
    protected void ValidateIndices(int row, int col)
    {
        if (row < 0 || row >= _rows || col < 0 || col >= _cols)
            throw new IndexOutOfRangeException("Invalid matrix indices.");
    }

    /// <summary>
    /// Returns a string representation of the matrix.
    /// </summary>
    /// <returns>A string showing the matrix elements arranged in rows and columns.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a text representation of the matrix,
    /// with each row on a new line and elements within a row separated by spaces.
    /// This is useful for displaying the matrix contents in a readable format,
    /// for example when debugging or logging.</para>
    /// </remarks>
    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                sb.Append(this[i, j]?.ToString()).Append(" ");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }
}