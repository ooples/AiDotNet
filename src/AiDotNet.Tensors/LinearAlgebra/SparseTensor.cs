using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a 2D sparse tensor with COO/CSR/CSC storage.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class SparseTensor<T>
{
    private readonly INumericOperations<T> _ops;

    public int Rows { get; }
    public int Columns { get; }
    public SparseStorageFormat Format { get; }

    public int[] RowIndices { get; }
    public int[] ColumnIndices { get; }
    public int[] RowPointers { get; }
    public int[] ColumnPointers { get; }
    public T[] Values { get; }

    public int NonZeroCount => Values.Length;

    public SparseTensor(int rows, int columns, int[] rowIndices, int[] columnIndices, T[] values)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Rows must be non-negative.");
        if (columns < 0)
            throw new ArgumentOutOfRangeException(nameof(columns), "Columns must be non-negative.");
        if (rowIndices is null)
            throw new ArgumentNullException(nameof(rowIndices));
        if (columnIndices is null)
            throw new ArgumentNullException(nameof(columnIndices));
        if (values is null)
            throw new ArgumentNullException(nameof(values));
        if (rowIndices.Length != columnIndices.Length || rowIndices.Length != values.Length)
            throw new ArgumentException("COO indices and values must have the same length.");

        Rows = rows;
        Columns = columns;
        RowIndices = rowIndices;
        ColumnIndices = columnIndices;
        Values = values;
        RowPointers = Array.Empty<int>();
        ColumnPointers = Array.Empty<int>();
        Format = SparseStorageFormat.Coo;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    private SparseTensor(int rows, int columns, SparseStorageFormat format, int[] rowIndices, int[] columnIndices, int[] rowPointers, int[] columnPointers, T[] values)
    {
        Rows = rows;
        Columns = columns;
        Format = format;
        RowIndices = rowIndices;
        ColumnIndices = columnIndices;
        RowPointers = rowPointers;
        ColumnPointers = columnPointers;
        Values = values;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public static SparseTensor<T> FromCsr(int rows, int columns, int[] rowPointers, int[] columnIndices, T[] values)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Rows must be non-negative.");
        if (columns < 0)
            throw new ArgumentOutOfRangeException(nameof(columns), "Columns must be non-negative.");
        if (rowPointers is null)
            throw new ArgumentNullException(nameof(rowPointers));
        if (columnIndices is null)
            throw new ArgumentNullException(nameof(columnIndices));
        if (values is null)
            throw new ArgumentNullException(nameof(values));
        if (rowPointers.Length != rows + 1)
            throw new ArgumentException("RowPointers length must be rows + 1.", nameof(rowPointers));
        if (columnIndices.Length != values.Length)
            throw new ArgumentException("ColumnIndices and Values must have the same length.", nameof(columnIndices));

        return new SparseTensor<T>(rows, columns, SparseStorageFormat.Csr,
            Array.Empty<int>(), columnIndices, rowPointers, Array.Empty<int>(), values);
    }

    public static SparseTensor<T> FromCsc(int rows, int columns, int[] columnPointers, int[] rowIndices, T[] values)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Rows must be non-negative.");
        if (columns < 0)
            throw new ArgumentOutOfRangeException(nameof(columns), "Columns must be non-negative.");
        if (columnPointers is null)
            throw new ArgumentNullException(nameof(columnPointers));
        if (rowIndices is null)
            throw new ArgumentNullException(nameof(rowIndices));
        if (values is null)
            throw new ArgumentNullException(nameof(values));
        if (columnPointers.Length != columns + 1)
            throw new ArgumentException("ColumnPointers length must be columns + 1.", nameof(columnPointers));
        if (rowIndices.Length != values.Length)
            throw new ArgumentException("RowIndices and Values must have the same length.", nameof(rowIndices));

        return new SparseTensor<T>(rows, columns, SparseStorageFormat.Csc,
            rowIndices, Array.Empty<int>(), Array.Empty<int>(), columnPointers, values);
    }

    public static SparseTensor<T> FromDense(Tensor<T> dense)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return FromDense(dense, ops.Zero);
    }

    public static SparseTensor<T> FromDense(Tensor<T> dense, T tolerance)
    {
        if (dense is null)
            throw new ArgumentNullException(nameof(dense));
        if (dense.Rank != 2)
            throw new ArgumentException("SparseTensor only supports rank-2 tensors.", nameof(dense));

        var ops = MathHelper.GetNumericOperations<T>();
        var rowIndices = new List<int>();
        var colIndices = new List<int>();
        var values = new List<T>();

        int rows = dense.Shape[0];
        int cols = dense.Shape[1];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                T value = dense[i, j];
                if (ops.LessThanOrEquals(ops.Abs(value), tolerance))
                    continue;

                rowIndices.Add(i);
                colIndices.Add(j);
                values.Add(value);
            }
        }

        return new SparseTensor<T>(rows, cols, rowIndices.ToArray(), colIndices.ToArray(), values.ToArray());
    }

    public SparseTensor<T> ToCoo()
    {
        if (Format == SparseStorageFormat.Coo)
            return this;

        if (Format == SparseStorageFormat.Csr)
        {
            var rowIndices = new int[Values.Length];
            var colIndices = new int[Values.Length];
            int index = 0;
            for (int row = 0; row < Rows; row++)
            {
                for (int i = RowPointers[row]; i < RowPointers[row + 1]; i++)
                {
                    rowIndices[index] = row;
                    colIndices[index] = ColumnIndices[i];
                    index++;
                }
            }
            return new SparseTensor<T>(Rows, Columns, rowIndices, colIndices, (T[])Values.Clone());
        }

        var rowIndicesCsc = new int[Values.Length];
        var colIndicesCsc = new int[Values.Length];
        int idx = 0;
        for (int col = 0; col < Columns; col++)
        {
            for (int i = ColumnPointers[col]; i < ColumnPointers[col + 1]; i++)
            {
                rowIndicesCsc[idx] = RowIndices[i];
                colIndicesCsc[idx] = col;
                idx++;
            }
        }
        return new SparseTensor<T>(Rows, Columns, rowIndicesCsc, colIndicesCsc, (T[])Values.Clone());
    }

    public SparseTensor<T> ToCsr()
    {
        if (Format == SparseStorageFormat.Csr)
            return this;

        var coo = ToCoo().Coalesce();
        int nnz = coo.Values.Length;
        var rowPointers = new int[Rows + 1];
        for (int i = 0; i < nnz; i++)
            rowPointers[coo.RowIndices[i] + 1]++;

        for (int i = 0; i < Rows; i++)
            rowPointers[i + 1] += rowPointers[i];

        var columnIndices = new int[nnz];
        var values = new T[nnz];
        var offsets = (int[])rowPointers.Clone();

        for (int i = 0; i < nnz; i++)
        {
            int row = coo.RowIndices[i];
            int dest = offsets[row]++;
            columnIndices[dest] = coo.ColumnIndices[i];
            values[dest] = coo.Values[i];
        }

        return FromCsr(Rows, Columns, rowPointers, columnIndices, values);
    }

    public SparseTensor<T> ToCsc()
    {
        if (Format == SparseStorageFormat.Csc)
            return this;

        var coo = ToCoo().Coalesce();
        int nnz = coo.Values.Length;
        var columnPointers = new int[Columns + 1];
        for (int i = 0; i < nnz; i++)
            columnPointers[coo.ColumnIndices[i] + 1]++;

        for (int i = 0; i < Columns; i++)
            columnPointers[i + 1] += columnPointers[i];

        var rowIndices = new int[nnz];
        var values = new T[nnz];
        var offsets = (int[])columnPointers.Clone();

        for (int i = 0; i < nnz; i++)
        {
            int col = coo.ColumnIndices[i];
            int dest = offsets[col]++;
            rowIndices[dest] = coo.RowIndices[i];
            values[dest] = coo.Values[i];
        }

        return FromCsc(Rows, Columns, columnPointers, rowIndices, values);
    }

    public SparseTensor<T> Coalesce()
    {
        var coo = ToCoo();
        int nnz = coo.Values.Length;
        if (nnz == 0)
            return coo;

        var entries = new List<(int Row, int Col, T Value)>(nnz);
        for (int i = 0; i < nnz; i++)
            entries.Add((coo.RowIndices[i], coo.ColumnIndices[i], coo.Values[i]));

        entries.Sort((a, b) =>
        {
            int rowCompare = a.Row.CompareTo(b.Row);
            return rowCompare != 0 ? rowCompare : a.Col.CompareTo(b.Col);
        });

        var rowIndices = new List<int>();
        var colIndices = new List<int>();
        var values = new List<T>();

        int currentRow = entries[0].Row;
        int currentCol = entries[0].Col;
        T currentValue = entries[0].Value;

        for (int i = 1; i < entries.Count; i++)
        {
            var entry = entries[i];
            if (entry.Row == currentRow && entry.Col == currentCol)
            {
                currentValue = _ops.Add(currentValue, entry.Value);
            }
            else
            {
                if (!_ops.Equals(currentValue, _ops.Zero))
                {
                    rowIndices.Add(currentRow);
                    colIndices.Add(currentCol);
                    values.Add(currentValue);
                }

                currentRow = entry.Row;
                currentCol = entry.Col;
                currentValue = entry.Value;
            }
        }

        if (!_ops.Equals(currentValue, _ops.Zero))
        {
            rowIndices.Add(currentRow);
            colIndices.Add(currentCol);
            values.Add(currentValue);
        }

        return new SparseTensor<T>(Rows, Columns, rowIndices.ToArray(), colIndices.ToArray(), values.ToArray());
    }

    public SparseTensor<T> Transpose()
    {
        if (Format == SparseStorageFormat.Coo)
        {
            return new SparseTensor<T>(Columns, Rows, (int[])ColumnIndices.Clone(), (int[])RowIndices.Clone(), (T[])Values.Clone());
        }

        if (Format == SparseStorageFormat.Csr)
        {
            return FromCsc(Columns, Rows, (int[])RowPointers.Clone(), (int[])ColumnIndices.Clone(), (T[])Values.Clone());
        }

        return FromCsr(Columns, Rows, (int[])ColumnPointers.Clone(), (int[])RowIndices.Clone(), (T[])Values.Clone());
    }

    public Tensor<T> ToDense()
    {
        var dense = new Tensor<T>(new[] { Rows, Columns });
        if (NonZeroCount == 0)
            return dense;

        var coo = ToCoo();
        for (int i = 0; i < coo.Values.Length; i++)
        {
            dense[coo.RowIndices[i], coo.ColumnIndices[i]] = coo.Values[i];
        }

        return dense;
    }
}
