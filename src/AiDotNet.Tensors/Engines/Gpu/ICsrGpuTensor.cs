using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Interface for GPU-resident sparse tensors in CSR (Compressed Sparse Row) format.
/// CSR format is efficient for sparse matrix-dense matrix multiplication (SpMM) operations.
/// </summary>
/// <typeparam name="T">The element type of the tensor values.</typeparam>
/// <remarks>
/// <para><b>CSR Format:</b></para>
/// <para>
/// A sparse matrix of shape [M, K] is stored as three arrays:
/// - Values: Non-zero values in row-major order [nnz elements]
/// - ColumnIndices: Column index for each value [nnz elements]
/// - RowPointers: Start of each row in values array [M+1 elements]
/// </para>
/// <para><b>Example:</b></para>
/// <code>
/// Matrix:        Values:   ColIndices:  RowPointers:
/// [1 0 2]        [1,2,3]   [0,2,1]      [0,2,3]
/// [0 3 0]
///
/// Row 0: values at indices 0..1 (positions 0,2 with values 1,2)
/// Row 1: values at indices 2..2 (position 1 with value 3)
/// </code>
/// <para>
/// This format enables O(nnz) SpMM operations instead of O(M*K) for dense matrices,
/// providing significant speedup for sparse graphs (typically 90%+ sparse).
/// </para>
/// </remarks>
public interface ICsrGpuTensor<T> : IDisposable
{
    /// <summary>
    /// Gets the GPU buffer containing the non-zero values.
    /// </summary>
    IGpuBuffer Values { get; }

    /// <summary>
    /// Gets the GPU buffer containing column indices for each non-zero value (int32).
    /// </summary>
    IGpuBuffer ColumnIndices { get; }

    /// <summary>
    /// Gets the GPU buffer containing row pointers (int32).
    /// RowPointers[i] is the index in Values where row i starts.
    /// RowPointers[Rows] equals Nnz.
    /// </summary>
    IGpuBuffer RowPointers { get; }

    /// <summary>
    /// Gets the number of rows (M dimension).
    /// </summary>
    int Rows { get; }

    /// <summary>
    /// Gets the number of columns (K dimension).
    /// </summary>
    int Cols { get; }

    /// <summary>
    /// Gets the number of non-zero elements.
    /// </summary>
    int Nnz { get; }

    /// <summary>
    /// Gets the shape as an array [Rows, Cols].
    /// </summary>
    int[] Shape { get; }

    /// <summary>
    /// Downloads the sparse tensor to CPU as a dense tensor.
    /// </summary>
    /// <returns>Dense tensor with the sparse values filled in.</returns>
    Tensor<T> ToDenseTensor();

    /// <summary>
    /// Downloads the CSR components to CPU.
    /// </summary>
    /// <returns>Tuple of (values, columnIndices, rowPointers).</returns>
    (T[] Values, int[] ColumnIndices, int[] RowPointers) ToCsr();
}

/// <summary>
/// GPU-resident sparse tensor in CSR format.
/// </summary>
/// <typeparam name="T">The element type.</typeparam>
public sealed class CsrGpuTensor<T> : ICsrGpuTensor<T>
{
    private readonly IDirectGpuBackend _backend;
    private readonly bool _ownsBuffers;
    private bool _disposed;

    /// <inheritdoc/>
    public IGpuBuffer Values { get; }

    /// <inheritdoc/>
    public IGpuBuffer ColumnIndices { get; }

    /// <inheritdoc/>
    public IGpuBuffer RowPointers { get; }

    /// <inheritdoc/>
    public int Rows { get; }

    /// <inheritdoc/>
    public int Cols { get; }

    /// <inheritdoc/>
    public int Nnz { get; }

    /// <inheritdoc/>
    public int[] Shape => [Rows, Cols];

    /// <summary>
    /// Creates a CSR GPU tensor from existing GPU buffers.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="values">GPU buffer containing non-zero values.</param>
    /// <param name="columnIndices">GPU buffer containing column indices.</param>
    /// <param name="rowPointers">GPU buffer containing row pointers.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="nnz">Number of non-zero elements.</param>
    /// <param name="ownsBuffers">Whether this tensor owns the buffers and should dispose them.</param>
    public CsrGpuTensor(
        IDirectGpuBackend backend,
        IGpuBuffer values,
        IGpuBuffer columnIndices,
        IGpuBuffer rowPointers,
        int rows,
        int cols,
        int nnz,
        bool ownsBuffers = true)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        Values = values ?? throw new ArgumentNullException(nameof(values));
        ColumnIndices = columnIndices ?? throw new ArgumentNullException(nameof(columnIndices));
        RowPointers = rowPointers ?? throw new ArgumentNullException(nameof(rowPointers));
        Rows = rows;
        Cols = cols;
        Nnz = nnz;
        _ownsBuffers = ownsBuffers;
    }

    /// <summary>
    /// Creates a CSR GPU tensor by uploading CPU data.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="values">Non-zero values array.</param>
    /// <param name="columnIndices">Column indices array.</param>
    /// <param name="rowPointers">Row pointers array.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    public CsrGpuTensor(
        IDirectGpuBackend backend,
        float[] values,
        int[] columnIndices,
        int[] rowPointers,
        int rows,
        int cols)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));

        if (values == null || columnIndices == null || rowPointers == null)
            throw new ArgumentNullException("CSR arrays cannot be null");

        if (rowPointers.Length != rows + 1)
            throw new ArgumentException($"RowPointers must have {rows + 1} elements, got {rowPointers.Length}");

        int nnz = values.Length;
        if (columnIndices.Length != nnz)
            throw new ArgumentException($"ColumnIndices length ({columnIndices.Length}) must match values length ({nnz})");

        Values = backend.AllocateBuffer(values);
        ColumnIndices = backend.AllocateIntBuffer(columnIndices);
        RowPointers = backend.AllocateIntBuffer(rowPointers);
        Rows = rows;
        Cols = cols;
        Nnz = nnz;
        _ownsBuffers = true;
    }

    /// <inheritdoc/>
    public Tensor<T> ToDenseTensor()
    {
        ThrowIfDisposed();

        var (values, colIndices, rowPtrs) = ToCsr();

        var dense = new T[Rows * Cols];

        for (int row = 0; row < Rows; row++)
        {
            int start = rowPtrs[row];
            int end = rowPtrs[row + 1];

            for (int i = start; i < end; i++)
            {
                int col = colIndices[i];
                dense[row * Cols + col] = values[i]; // Values are already type T from ToCsr()
            }
        }

        return new Tensor<T>(dense, [Rows, Cols]);
    }

    /// <inheritdoc/>
    public (T[] Values, int[] ColumnIndices, int[] RowPointers) ToCsr()
    {
        ThrowIfDisposed();

        float[] floatValues = _backend.DownloadBuffer(Values);

        // Column indices and row pointers are stored as floats in GPU buffers
        // (since AllocateIntBuffer stores them as float representation)
        // Download and convert back to int
        float[] floatColIndices = _backend.DownloadBuffer(ColumnIndices);
        float[] floatRowPtrs = _backend.DownloadBuffer(RowPointers);

        int[] colIndices = new int[floatColIndices.Length];
        for (int i = 0; i < floatColIndices.Length; i++)
        {
            colIndices[i] = (int)floatColIndices[i];
        }

        int[] rowPtrs = new int[floatRowPtrs.Length];
        for (int i = 0; i < floatRowPtrs.Length; i++)
        {
            rowPtrs[i] = (int)floatRowPtrs[i];
        }

        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        T[] typedValues = new T[floatValues.Length];
        for (int i = 0; i < floatValues.Length; i++)
        {
            typedValues[i] = numOps.FromDouble(floatValues[i]);
        }

        return (typedValues, colIndices, rowPtrs);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CsrGpuTensor<T>));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        if (_ownsBuffers)
        {
            Values.Dispose();
            ColumnIndices.Dispose();
            RowPointers.Dispose();
        }
    }
}

/// <summary>
/// Factory methods for creating CSR GPU tensors.
/// </summary>
public static class CsrGpuTensorFactory
{
    /// <summary>
    /// Creates a CSR GPU tensor from a dense tensor by extracting non-zero elements.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="denseTensor">The dense tensor to convert (must be 2D).</param>
    /// <param name="threshold">Values with absolute value below this are treated as zero.</param>
    /// <returns>A CSR GPU tensor.</returns>
    public static CsrGpuTensor<T> FromDenseTensor<T>(
        IDirectGpuBackend backend,
        Tensor<T> denseTensor,
        float threshold = 1e-6f)
    {
        if (denseTensor.Rank != 2)
            throw new ArgumentException("Dense tensor must be 2D for CSR conversion", nameof(denseTensor));

        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        int rows = denseTensor.Shape[0];
        int cols = denseTensor.Shape[1];

        // Count non-zeros and build CSR structure
        var values = new System.Collections.Generic.List<float>();
        var colIndices = new System.Collections.Generic.List<int>();
        var rowPointers = new System.Collections.Generic.List<int> { 0 };

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                float val = (float)numOps.ToDouble(denseTensor[row, col]);
                if (MathF.Abs(val) > threshold)
                {
                    values.Add(val);
                    colIndices.Add(col);
                }
            }
            rowPointers.Add(values.Count);
        }

        return new CsrGpuTensor<T>(
            backend,
            values.ToArray(),
            colIndices.ToArray(),
            rowPointers.ToArray(),
            rows,
            cols);
    }

    /// <summary>
    /// Creates a CSR GPU tensor from edge indices (COO-like input).
    /// Useful for graph neural networks where edges are specified as (source, target) pairs.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="sourceIndices">Source node indices for each edge.</param>
    /// <param name="targetIndices">Target node indices for each edge.</param>
    /// <param name="values">Edge values (weights). If null, uses 1.0 for all edges.</param>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <returns>A CSR GPU tensor representing the adjacency matrix.</returns>
    public static CsrGpuTensor<T> FromEdgeIndices<T>(
        IDirectGpuBackend backend,
        int[] sourceIndices,
        int[] targetIndices,
        float[]? values,
        int numNodes)
    {
        if (sourceIndices.Length != targetIndices.Length)
            throw new ArgumentException("Source and target indices must have the same length");

        int numEdges = sourceIndices.Length;
        float[] edgeValues = values ?? Enumerable.Repeat(1.0f, numEdges).ToArray();

        // Sort edges by source index for CSR format
        var edges = new (int src, int tgt, float val)[numEdges];
        for (int i = 0; i < numEdges; i++)
        {
            edges[i] = (sourceIndices[i], targetIndices[i], edgeValues[i]);
        }
        Array.Sort(edges, (a, b) => a.src.CompareTo(b.src) != 0 ? a.src.CompareTo(b.src) : a.tgt.CompareTo(b.tgt));

        // Build CSR arrays
        var csrValues = new float[numEdges];
        var csrColIndices = new int[numEdges];
        var csrRowPointers = new int[numNodes + 1];

        int currentRow = 0;
        csrRowPointers[0] = 0;

        for (int i = 0; i < numEdges; i++)
        {
            // Fill row pointers for empty rows
            while (currentRow < edges[i].src)
            {
                currentRow++;
                csrRowPointers[currentRow] = i;
            }

            csrValues[i] = edges[i].val;
            csrColIndices[i] = edges[i].tgt;
        }

        // Fill remaining row pointers
        while (currentRow < numNodes)
        {
            currentRow++;
            csrRowPointers[currentRow] = numEdges;
        }

        return new CsrGpuTensor<T>(
            backend,
            csrValues,
            csrColIndices,
            csrRowPointers,
            numNodes,
            numNodes);
    }
}
