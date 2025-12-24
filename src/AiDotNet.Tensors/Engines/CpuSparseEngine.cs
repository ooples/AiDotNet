using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// CPU implementation of sparse tensor operations.
/// </summary>
/// <remarks>
/// <para>
/// CpuSparseEngine provides efficient sparse matrix operations using standard algorithms
/// optimized for CPU execution. All operations work with the SparseTensor type which
/// supports COO, CSR, and CSC storage formats.
/// </para>
/// <para><b>For Beginners:</b> This class does the actual work for sparse operations on the CPU.
/// It's used when you don't have a GPU or when working with custom numeric types.
/// </para>
/// </remarks>
public sealed class CpuSparseEngine : ISparseEngine
{
    /// <summary>
    /// Singleton instance for convenience.
    /// </summary>
    public static CpuSparseEngine Instance { get; } = new CpuSparseEngine();

    #region Sparse Matrix-Vector Operations

    /// <inheritdoc/>
    public Vector<T> SpMV<T>(SparseTensor<T> sparse, Vector<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Columns != dense.Length)
        {
            throw new ArgumentException(
                $"Sparse matrix columns ({sparse.Columns}) must match vector length ({dense.Length}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new T[sparse.Rows];

        // Initialize result to zero
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ops.Zero;
        }

        // Convert to CSR format for efficient row-major access
        var csr = sparse.ToCsr();
        var rowPtrs = csr.RowPointers;
        var colIndices = csr.ColumnIndices;
        var values = csr.Values;

        // SpMV using CSR format: y[i] = sum_j A[i,j] * x[j]
        for (int row = 0; row < sparse.Rows; row++)
        {
            T sum = ops.Zero;
            int start = rowPtrs[row];
            int end = rowPtrs[row + 1];

            for (int idx = start; idx < end; idx++)
            {
                int col = colIndices[idx];
                T val = values[idx];
                sum = ops.Add(sum, ops.Multiply(val, dense[col]));
            }

            result[row] = sum;
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Vector<T> SpMVTranspose<T>(SparseTensor<T> sparse, Vector<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Rows != dense.Length)
        {
            throw new ArgumentException(
                $"Sparse matrix rows ({sparse.Rows}) must match vector length ({dense.Length}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new T[sparse.Columns];

        // Initialize result to zero
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ops.Zero;
        }

        // Convert to CSR format
        var csr = sparse.ToCsr();
        var rowPtrs = csr.RowPointers;
        var colIndices = csr.ColumnIndices;
        var values = csr.Values;

        // SpMV transpose: y[j] = sum_i A[i,j] * x[i] = sum_i A^T[j,i] * x[i]
        for (int row = 0; row < sparse.Rows; row++)
        {
            int start = rowPtrs[row];
            int end = rowPtrs[row + 1];
            T xVal = dense[row];

            for (int idx = start; idx < end; idx++)
            {
                int col = colIndices[idx];
                T val = values[idx];
                result[col] = ops.Add(result[col], ops.Multiply(val, xVal));
            }
        }

        return new Vector<T>(result);
    }

    #endregion

    #region Sparse Matrix-Matrix Operations

    /// <inheritdoc/>
    public Matrix<T> SpMM<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Columns != dense.Rows)
        {
            throw new ArgumentException(
                $"Sparse matrix columns ({sparse.Columns}) must match dense matrix rows ({dense.Rows}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(sparse.Rows, dense.Columns);

        // Convert to CSR format
        var csr = sparse.ToCsr();
        var rowPtrs = csr.RowPointers;
        var colIndices = csr.ColumnIndices;
        var values = csr.Values;

        // SpMM: C[i,k] = sum_j A[i,j] * B[j,k]
        for (int i = 0; i < sparse.Rows; i++)
        {
            int start = rowPtrs[i];
            int end = rowPtrs[i + 1];

            for (int k = 0; k < dense.Columns; k++)
            {
                T sum = ops.Zero;

                for (int idx = start; idx < end; idx++)
                {
                    int j = colIndices[idx];
                    T aVal = values[idx];
                    sum = ops.Add(sum, ops.Multiply(aVal, dense[j, k]));
                }

                result[i, k] = sum;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public SparseTensor<T> SpSpMM<T>(SparseTensor<T> a, SparseTensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix A columns ({a.Columns}) must match matrix B rows ({b.Rows}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        // Convert both to CSR format for efficient row access
        var aCsr = a.ToCsr();
        var bCsr = b.ToCsr();

        var aRowPtrs = aCsr.RowPointers;
        var aColIndices = aCsr.ColumnIndices;
        var aValues = aCsr.Values;
        var bRowPtrs = bCsr.RowPointers;
        var bColIndices = bCsr.ColumnIndices;
        var bValues = bCsr.Values;

        // Use hash map to accumulate results
        var resultEntries = new List<(int row, int col, T value)>();

        for (int i = 0; i < a.Rows; i++)
        {
            // Hash map for row i of result
            var rowAccum = new Dictionary<int, T>();

            int aStart = aRowPtrs[i];
            int aEnd = aRowPtrs[i + 1];

            for (int aIdx = aStart; aIdx < aEnd; aIdx++)
            {
                int k = aColIndices[aIdx];
                T aVal = aValues[aIdx];

                // Multiply row k of B
                int bStart = bRowPtrs[k];
                int bEnd = bRowPtrs[k + 1];

                for (int bIdx = bStart; bIdx < bEnd; bIdx++)
                {
                    int j = bColIndices[bIdx];
                    T bVal = bValues[bIdx];
                    T product = ops.Multiply(aVal, bVal);

                    if (rowAccum.TryGetValue(j, out T? existing) && existing is not null)
                    {
                        rowAccum[j] = ops.Add(existing, product);
                    }
                    else
                    {
                        rowAccum[j] = product;
                    }
                }
            }

            // Add non-zero entries to result
            foreach (var kvp in rowAccum)
            {
                if (!ops.Equals(kvp.Value, ops.Zero))
                {
                    resultEntries.Add((i, kvp.Key, kvp.Value));
                }
            }
        }

        // Build result sparse tensor
        var rows = new int[resultEntries.Count];
        var cols = new int[resultEntries.Count];
        var vals = new T[resultEntries.Count];

        for (int i = 0; i < resultEntries.Count; i++)
        {
            rows[i] = resultEntries[i].row;
            cols[i] = resultEntries[i].col;
            vals[i] = resultEntries[i].value;
        }

        return new SparseTensor<T>(a.Rows, b.Columns, rows, cols, vals);
    }

    #endregion

    #region Sparse-Dense Element-wise Operations

    /// <inheritdoc/>
    public Matrix<T> AddSparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Rows != dense.Rows || sparse.Columns != dense.Columns)
        {
            throw new ArgumentException("Sparse and dense matrix dimensions must match.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(dense.Rows, dense.Columns);

        // Copy dense matrix
        for (int i = 0; i < dense.Rows; i++)
        {
            for (int j = 0; j < dense.Columns; j++)
            {
                result[i, j] = dense[i, j];
            }
        }

        // Add sparse entries - use COO format
        var coo = sparse.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;
        var values = coo.Values;

        for (int idx = 0; idx < values.Length; idx++)
        {
            int row = rowIndices[idx];
            int col = colIndices[idx];
            result[row, col] = ops.Add(result[row, col], values[idx]);
        }

        return result;
    }

    /// <inheritdoc/>
    public SparseTensor<T> MultiplySparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Rows != dense.Rows || sparse.Columns != dense.Columns)
        {
            throw new ArgumentException("Sparse and dense matrix dimensions must match.");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        // Use COO format for element-wise access
        var coo = sparse.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;
        var oldValues = coo.Values;

        var newValues = new T[oldValues.Length];

        for (int idx = 0; idx < oldValues.Length; idx++)
        {
            int row = rowIndices[idx];
            int col = colIndices[idx];
            newValues[idx] = ops.Multiply(oldValues[idx], dense[row, col]);
        }

        // Create new sparse tensor with same structure but new values
        var newRowIndices = new int[rowIndices.Length];
        var newColIndices = new int[colIndices.Length];
        Array.Copy(rowIndices, newRowIndices, rowIndices.Length);
        Array.Copy(colIndices, newColIndices, colIndices.Length);

        return new SparseTensor<T>(sparse.Rows, sparse.Columns, newRowIndices, newColIndices, newValues);
    }

    #endregion

    #region Gather and Scatter Operations

    /// <inheritdoc/>
    public Vector<T> SparseGather<T>(Matrix<T> source, SparseTensor<T> indices)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (indices is null) throw new ArgumentNullException(nameof(indices));

        // Use COO format for index access
        var coo = indices.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;

        var result = new T[rowIndices.Length];

        for (int i = 0; i < rowIndices.Length; i++)
        {
            int row = rowIndices[i];
            int col = colIndices[i];

            if (row < 0 || row >= source.Rows || col < 0 || col >= source.Columns)
            {
                throw new ArgumentOutOfRangeException($"Index ({row}, {col}) is out of bounds for matrix of size ({source.Rows}, {source.Columns}).");
            }

            result[i] = source[row, col];
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Matrix<T> SparseScatter<T>(Vector<T> values, SparseTensor<T> indices, int rows, int cols)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (indices is null) throw new ArgumentNullException(nameof(indices));

        // Use COO format for index access
        var coo = indices.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;

        if (values.Length != rowIndices.Length)
        {
            throw new ArgumentException($"Values length ({values.Length}) must match number of indices ({rowIndices.Length}).");
        }

        var result = new Matrix<T>(rows, cols);

        for (int i = 0; i < values.Length; i++)
        {
            int row = rowIndices[i];
            int col = colIndices[i];

            if (row < 0 || row >= rows || col < 0 || col >= cols)
            {
                throw new ArgumentOutOfRangeException($"Index ({row}, {col}) is out of bounds for matrix of size ({rows}, {cols}).");
            }

            result[row, col] = values[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public void SparseScatterAdd<T>(Vector<T> values, (int[] rows, int[] cols) indices, Matrix<T> target)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (indices.rows is null) throw new ArgumentNullException(nameof(indices));
        if (indices.cols is null) throw new ArgumentNullException(nameof(indices));
        if (target is null) throw new ArgumentNullException(nameof(target));

        if (values.Length != indices.rows.Length || values.Length != indices.cols.Length)
        {
            throw new ArgumentException("Values length must match indices length.");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < values.Length; i++)
        {
            int row = indices.rows[i];
            int col = indices.cols[i];

            if (row < 0 || row >= target.Rows || col < 0 || col >= target.Columns)
            {
                throw new ArgumentOutOfRangeException($"Index ({row}, {col}) is out of bounds.");
            }

            target[row, col] = ops.Add(target[row, col], values[i]);
        }
    }

    #endregion

    #region Sparse Tensor Utilities

    /// <inheritdoc/>
    public Matrix<T> SparseToDense<T>(SparseTensor<T> sparse)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));

        var result = new Matrix<T>(sparse.Rows, sparse.Columns);

        // Use COO format for iteration
        var coo = sparse.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;
        var values = coo.Values;

        for (int idx = 0; idx < values.Length; idx++)
        {
            result[rowIndices[idx], colIndices[idx]] = values[idx];
        }

        return result;
    }

    /// <inheritdoc/>
    public SparseTensor<T> DenseToSparse<T>(Matrix<T> dense, T threshold)
    {
        if (dense is null) throw new ArgumentNullException(nameof(dense));

        var ops = MathHelper.GetNumericOperations<T>();
        var entries = new List<(int row, int col, T value)>();

        for (int i = 0; i < dense.Rows; i++)
        {
            for (int j = 0; j < dense.Columns; j++)
            {
                T val = dense[i, j];
                T absVal = ops.Abs(val);

                if (ops.GreaterThan(absVal, threshold))
                {
                    entries.Add((i, j, val));
                }
            }
        }

        var rows = new int[entries.Count];
        var cols = new int[entries.Count];
        var vals = new T[entries.Count];

        for (int i = 0; i < entries.Count; i++)
        {
            rows[i] = entries[i].row;
            cols[i] = entries[i].col;
            vals[i] = entries[i].value;
        }

        return new SparseTensor<T>(dense.Rows, dense.Columns, rows, cols, vals);
    }

    /// <inheritdoc/>
    public SparseTensor<T> Coalesce<T>(SparseTensor<T> sparse)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));

        // SparseTensor already has a Coalesce method - delegate to it
        return sparse.Coalesce();
    }

    /// <inheritdoc/>
    public SparseTensor<T> SparseTranspose<T>(SparseTensor<T> sparse)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));

        // SparseTensor already has a Transpose method - delegate to it
        return sparse.Transpose();
    }

    #endregion
}
