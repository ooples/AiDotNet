﻿global using System.Text;

namespace AiDotNet.LinearAlgebra;

public abstract class MatrixBase<T>
{
    protected readonly T[] data;
    protected readonly int rows;
    protected readonly int cols;
    protected readonly INumericOperations<T> ops;

    protected MatrixBase(int rows, int cols, INumericOperations<T> operations)
    {
        if (rows <= 0) throw new ArgumentException("Rows must be positive", nameof(rows));
        if (cols <= 0) throw new ArgumentException("Columns must be positive", nameof(cols));

        this.rows = rows;
        this.cols = cols;
        this.data = new T[rows * cols];
        this.ops = operations;
    }

    protected MatrixBase(IEnumerable<IEnumerable<T>> values, INumericOperations<T> operations)
    {
        var valuesList = values.Select(v => v.ToArray()).ToList();
        this.rows = valuesList.Count;
        this.cols = valuesList.First().Length;
        this.data = new T[rows * cols];
        this.ops = operations;

        for (int i = 0; i < rows; i++)
        {
            var row = valuesList[i];
            if (row.Length != cols)
            {
                throw new ArgumentException("All rows must have the same number of columns.", nameof(values));
            }

            for (int j = 0; j < cols; j++)
            {
                data[i * cols + j] = row[j];
            }
        }
    }

    public int Rows => rows;
    public int Columns => cols;

    public virtual T this[int row, int col]
    {
        get
        {
            ValidateIndices(row, col);
            return data[row * cols + col];
        }
        set
        {
            ValidateIndices(row, col);
            data[row * cols + col] = value;
        }
    }

    public virtual MatrixBase<T> Ones(int rows, int cols)
    {
        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = ops.One;

        return result;
    }

    public virtual MatrixBase<T> Zeros(int rows, int cols)
    {
        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = ops.Zero;

        return result;
    }

    public virtual void SetColumn(int columnIndex, VectorBase<T> vector)
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

    public virtual void SetRow(int rowIndex, VectorBase<T> vector)
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

    public virtual VectorBase<T> GetRow(int row)
    {
        ValidateIndices(row, 0);
        return new Vector<T>(Enumerable.Range(0, cols).Select(col => this[row, col]).ToArray(), ops);
    }

    public virtual VectorBase<T> GetColumn(int col)
    {
        ValidateIndices(0, col);
        return new Vector<T>(Enumerable.Range(0, rows).Select(row => this[row, col]).ToArray(), ops);
    }

    public virtual MatrixBase<T> Add(MatrixBase<T> other)
    {
        if (rows != other.Rows || cols != other.Columns)
            throw new ArgumentException("Matrix dimensions must match for addition.");

        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = ops.Add(this[i, j], other[i, j]);

        return result;
    }

    public virtual MatrixBase<T> Subtract(MatrixBase<T> other)
    {
        if (rows != other.Rows || cols != other.Columns)
            throw new ArgumentException("Matrix dimensions must match for subtraction.");

        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = ops.Subtract(this[i, j], other[i, j]);

        return result;
    }

    public virtual MatrixBase<T> Multiply(MatrixBase<T> other)
    {
        if (cols != other.Rows)
            throw new ArgumentException("Number of columns in the first matrix must equal the number of rows in the second matrix.");

        var result = CreateInstance(rows, other.Columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < other.Columns; j++)
                for (int k = 0; k < cols; k++)
                    result[i, j] = ops.Add(result[i, j], ops.Multiply(this[i, k], other[k, j]));

        return result;
    }

    public virtual VectorBase<T> Multiply(Vector<T> vector)
    {
        if (cols != vector.Length)
            throw new ArgumentException("Number of columns in the matrix must equal the length of the vector.");

        var result = new Vector<T>(rows, ops);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i] = ops.Add(result[i], ops.Multiply(this[i, j], vector[j]));

        return result;
    }

    public virtual MatrixBase<T> Multiply(T scalar)
    {
        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = ops.Multiply(this[i, j], scalar);

        return result;
    }

    public virtual MatrixBase<T> Transpose()
    {
        var result = CreateInstance(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j, i] = this[i, j];

        return result;
    }

    public virtual MatrixBase<T> Copy()
    {
        var result = CreateInstance(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = this[i, j];

        return result;
    }

    protected abstract MatrixBase<T> CreateInstance(int rows, int cols);

    protected void ValidateIndices(int row, int col)
    {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw new IndexOutOfRangeException("Invalid matrix indices.");
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sb.Append(this[i, j]?.ToString()).Append(" ");
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
}