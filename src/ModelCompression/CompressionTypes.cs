namespace AiDotNet.ModelCompression;

/// <summary>
/// Result of a compression operation.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public class CompressionResult<T>
{
    /// <summary>
    /// The compressed data as a tensor.
    /// </summary>
    public required Tensor<T> CompressedData { get; init; }

    /// <summary>
    /// Metadata required for decompression.
    /// </summary>
    public required ICompressionMetadata<T> Metadata { get; init; }

    /// <summary>
    /// Original shape of the data before compression.
    /// </summary>
    public required int[] OriginalShape { get; init; }

    /// <summary>
    /// Original size in bytes.
    /// </summary>
    public long OriginalSizeBytes { get; init; }

    /// <summary>
    /// Compressed size in bytes (data + metadata).
    /// </summary>
    public long CompressedSizeBytes { get; init; }

    /// <summary>
    /// Achieved compression ratio.
    /// </summary>
    public double CompressionRatio => OriginalSizeBytes > 0
        ? (double)OriginalSizeBytes / CompressedSizeBytes
        : 1.0;
}

/// <summary>
/// Result of sparse compression operation.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public class SparseCompressionResult<T>
{
    /// <summary>
    /// The sparse format used.
    /// </summary>
    public required SparseFormat Format { get; init; }

    /// <summary>
    /// Non-zero values.
    /// </summary>
    public required T[] Values { get; init; }

    /// <summary>
    /// Row indices (for COO, CSR formats).
    /// </summary>
    public int[]? RowIndices { get; init; }

    /// <summary>
    /// Column indices (for COO, CSC formats).
    /// </summary>
    public int[]? ColumnIndices { get; init; }

    /// <summary>
    /// Row pointers (for CSR format).
    /// </summary>
    public int[]? RowPointers { get; init; }

    /// <summary>
    /// Column pointers (for CSC format).
    /// </summary>
    public int[]? ColumnPointers { get; init; }

    /// <summary>
    /// Block indices (for block-sparse format).
    /// </summary>
    public int[]? BlockIndices { get; init; }

    /// <summary>
    /// Block size for block-sparse format.
    /// </summary>
    public int BlockSize { get; init; }

    /// <summary>
    /// N value for N:M sparsity patterns.
    /// </summary>
    public int SparsityN { get; init; }

    /// <summary>
    /// M value for N:M sparsity patterns.
    /// </summary>
    public int SparsityM { get; init; }

    /// <summary>
    /// Mask for 2:4 or N:M structured sparsity.
    /// </summary>
    public byte[]? SparsityMask { get; init; }

    /// <summary>
    /// Original dense shape.
    /// </summary>
    public required int[] OriginalShape { get; init; }

    /// <summary>
    /// Number of non-zero elements.
    /// </summary>
    public int NonZeroCount => Values.Length;

    /// <summary>
    /// Sparsity ratio (fraction of zeros).
    /// </summary>
    public double Sparsity
    {
        get
        {
            long totalElements = 1;
            foreach (var dim in OriginalShape)
                totalElements *= dim;
            return totalElements > 0
                ? 1.0 - ((double)NonZeroCount / totalElements)
                : 0.0;
        }
    }

    /// <summary>
    /// Gets the compressed size in bytes.
    /// </summary>
    public long GetCompressedSizeBytes(int elementSize)
    {
        long size = Values.Length * elementSize;

        if (RowIndices != null) size += RowIndices.Length * sizeof(int);
        if (ColumnIndices != null) size += ColumnIndices.Length * sizeof(int);
        if (RowPointers != null) size += RowPointers.Length * sizeof(int);
        if (ColumnPointers != null) size += ColumnPointers.Length * sizeof(int);
        if (BlockIndices != null) size += BlockIndices.Length * sizeof(int);
        if (SparsityMask != null) size += SparsityMask.Length;

        // Metadata overhead
        size += sizeof(int) * (OriginalShape.Length + 4); // shape + format + block size + N + M

        return size;
    }
}

/// <summary>
/// Sparse storage formats.
/// </summary>
public enum SparseFormat
{
    /// <summary>
    /// Coordinate format (COO) - stores (row, col, value) triplets.
    /// Best for: Construction and incremental updates.
    /// </summary>
    COO,

    /// <summary>
    /// Compressed Sparse Row (CSR) - row pointers + column indices + values.
    /// Best for: Row-wise operations, matrix-vector multiplication.
    /// </summary>
    CSR,

    /// <summary>
    /// Compressed Sparse Column (CSC) - column pointers + row indices + values.
    /// Best for: Column-wise operations.
    /// </summary>
    CSC,

    /// <summary>
    /// Block Sparse Row (BSR) - like CSR but with dense blocks.
    /// Best for: Structured sparsity patterns.
    /// </summary>
    BSR,

    /// <summary>
    /// 2:4 Structured Sparsity - 2 zeros per 4 elements (NVIDIA Ampere compatible).
    /// Best for: GPU acceleration with Sparse Tensor Cores (2x throughput).
    /// </summary>
    Structured2to4,

    /// <summary>
    /// N:M Fine-grained structured sparsity (generalized).
    /// Best for: Hardware-accelerated inference with customizable sparsity.
    /// </summary>
    StructuredNtoM,

    /// <summary>
    /// Diagonal format - for diagonal or banded matrices.
    /// Best for: Diagonal-dominant weight matrices.
    /// </summary>
    DIA,

    /// <summary>
    /// ELLPACK format - fixed number of non-zeros per row.
    /// Best for: Regular sparsity patterns, GPU acceleration.
    /// </summary>
    ELL
}
