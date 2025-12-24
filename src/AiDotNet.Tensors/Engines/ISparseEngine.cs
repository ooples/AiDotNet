using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Extension interface for sparse tensor operations.
/// </summary>
/// <remarks>
/// <para>
/// ISparseEngine provides operations for sparse tensors that are too memory-intensive
/// to store as dense arrays. Common in graph neural networks, recommendation systems,
/// and NLP (bag-of-words) applications.
/// </para>
/// <para><b>For Beginners:</b> Sparse tensors store only non-zero values.
///
/// Example: A 1000x1000 matrix with only 100 non-zero values:
/// - Dense storage: 1,000,000 values (8MB for doubles)
/// - Sparse storage: ~300 values (indices + values = ~2.4KB)
///
/// SpMM = Sparse Matrix × Dense Matrix
/// SpMV = Sparse Matrix × Dense Vector
/// </para>
/// </remarks>
public interface ISparseEngine
{
    #region Sparse Matrix-Vector Operations

    /// <summary>
    /// Sparse matrix-vector multiplication (SpMV): y = A * x
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse matrix A in CSR format.</param>
    /// <param name="dense">The dense vector x.</param>
    /// <returns>The result vector y.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies a sparse matrix by a dense vector.
    /// Only non-zero elements of the sparse matrix contribute to the computation,
    /// making this much faster than dense matrix-vector multiplication for sparse matrices.
    /// </para>
    /// <para>
    /// Time complexity: O(nnz) where nnz is the number of non-zeros in the sparse matrix.
    /// </para>
    /// </remarks>
    Vector<T> SpMV<T>(SparseTensor<T> sparse, Vector<T> dense);

    /// <summary>
    /// Sparse matrix-vector multiplication with transpose: y = A^T * x
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse matrix A in CSR format.</param>
    /// <param name="dense">The dense vector x.</param>
    /// <returns>The result vector y = A^T * x.</returns>
    Vector<T> SpMVTranspose<T>(SparseTensor<T> sparse, Vector<T> dense);

    #endregion

    #region Sparse Matrix-Matrix Operations

    /// <summary>
    /// Sparse matrix-dense matrix multiplication (SpMM): C = A * B
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse matrix A.</param>
    /// <param name="dense">The dense matrix B.</param>
    /// <returns>The dense result matrix C.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multiplies a sparse matrix by a dense matrix.
    /// Common in graph neural networks where the adjacency matrix is sparse
    /// but feature matrices are dense.
    /// </para>
    /// <para>
    /// Time complexity: O(nnz * K) where K is the number of columns in B.
    /// </para>
    /// </remarks>
    Matrix<T> SpMM<T>(SparseTensor<T> sparse, Matrix<T> dense);

    /// <summary>
    /// Sparse matrix-sparse matrix multiplication: C = A * B (both sparse)
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="a">The first sparse matrix.</param>
    /// <param name="b">The second sparse matrix.</param>
    /// <returns>The sparse result matrix C.</returns>
    /// <remarks>
    /// <para>
    /// Result sparsity depends on the structure of inputs. May produce a denser result.
    /// Uses hash-based accumulation for efficient computation.
    /// </para>
    /// </remarks>
    SparseTensor<T> SpSpMM<T>(SparseTensor<T> a, SparseTensor<T> b);

    #endregion

    #region Sparse-Dense Element-wise Operations

    /// <summary>
    /// Element-wise addition of sparse and dense matrices.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse matrix.</param>
    /// <param name="dense">The dense matrix.</param>
    /// <returns>A dense matrix containing the sum.</returns>
    Matrix<T> AddSparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense);

    /// <summary>
    /// Element-wise multiplication of sparse and dense matrices (Hadamard product).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse matrix.</param>
    /// <param name="dense">The dense matrix.</param>
    /// <returns>A sparse matrix (zeros in sparse remain zero).</returns>
    /// <remarks>
    /// <para>
    /// The result is sparse because any position where the sparse matrix has 0
    /// will remain 0 in the output.
    /// </para>
    /// </remarks>
    SparseTensor<T> MultiplySparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense);

    #endregion

    #region Gather and Scatter Operations

    /// <summary>
    /// Gathers elements from a dense tensor using sparse indices.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="source">The dense source tensor.</param>
    /// <param name="indices">The indices to gather (as sparse tensor with 1s at gather positions).</param>
    /// <returns>A vector of gathered values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> "Gather" selects specific elements from a tensor.
    /// Like picking specific items from a list using their positions.
    /// </para>
    /// </remarks>
    Vector<T> SparseGather<T>(Matrix<T> source, SparseTensor<T> indices);

    /// <summary>
    /// Scatters values from a vector into a dense matrix at sparse positions.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="values">The values to scatter.</param>
    /// <param name="indices">The sparse tensor indicating target positions.</param>
    /// <param name="rows">Number of rows in the output matrix.</param>
    /// <param name="cols">Number of columns in the output matrix.</param>
    /// <returns>A matrix with values scattered at the specified positions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> "Scatter" places values at specific positions.
    /// Like putting items back into a list at their original positions.
    /// </para>
    /// </remarks>
    Matrix<T> SparseScatter<T>(Vector<T> values, SparseTensor<T> indices, int rows, int cols);

    /// <summary>
    /// Scatter-add: Adds values into a matrix at sparse positions (accumulating duplicates).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="values">The values to scatter.</param>
    /// <param name="indices">Row and column indices for each value.</param>
    /// <param name="target">The target matrix (modified in place).</param>
    /// <remarks>
    /// <para>
    /// Unlike regular scatter, scatter-add accumulates when multiple values
    /// target the same position. Essential for gradient computation in sparse operations.
    /// </para>
    /// </remarks>
    void SparseScatterAdd<T>(Vector<T> values, (int[] rows, int[] cols) indices, Matrix<T> target);

    #endregion

    #region Sparse Tensor Utilities

    /// <summary>
    /// Converts a sparse tensor to dense format.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse tensor.</param>
    /// <returns>A dense matrix with the same values.</returns>
    /// <remarks>
    /// <para>
    /// Warning: For very sparse, large tensors, this may consume significant memory.
    /// </para>
    /// </remarks>
    Matrix<T> SparseToDense<T>(SparseTensor<T> sparse);

    /// <summary>
    /// Converts a dense matrix to sparse format using a threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="dense">The dense matrix.</param>
    /// <param name="threshold">Values with absolute value below this are treated as zero.</param>
    /// <returns>A sparse tensor in CSR format.</returns>
    SparseTensor<T> DenseToSparse<T>(Matrix<T> dense, T threshold);

    /// <summary>
    /// Coalesces a sparse tensor by combining duplicate indices.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse tensor (may have duplicate indices).</param>
    /// <returns>A coalesced sparse tensor with unique indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sometimes sparse tensors accumulate duplicate
    /// entries during operations. Coalesce combines them by summing values at the same position.
    /// </para>
    /// </remarks>
    SparseTensor<T> Coalesce<T>(SparseTensor<T> sparse);

    /// <summary>
    /// Transposes a sparse matrix efficiently without densification.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="sparse">The sparse matrix to transpose.</param>
    /// <returns>The transposed sparse matrix.</returns>
    SparseTensor<T> SparseTranspose<T>(SparseTensor<T> sparse);

    #endregion
}
